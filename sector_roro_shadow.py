"""Causal Sector Rotation SHADOW using RRG, 20D alpha and exact-date RORO context."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from roro_shadow import SERIES_VERSION as RORO_SERIES_VERSION
from roro_shadow import read_shadow_history
from sector_rotation import SECTOR_ETFS, build_rrg_signals, latest_snapshot, load_price_matrix
from sector_rotation_forecast import (
    BASE_FEATURE_COLUMNS,
    _block_bootstrap_mean_ci,
    build_sector_feature_frame,
    fit_leading_forecast,
)


MODEL_VERSION = "sector-roro-context-v1-spy-rel-ridge-a10-m156-w20"
OUTPUT_START = pd.Timestamp("2026-01-07")
FROZEN_MIN_TRAIN_WEEKS = 156
FROZEN_ALPHA = 10.0
FROZEN_TOP_K = 3
ABLATION_NAMES = ("alpha-only", "alpha+RRG", "alpha+RORO", "full")
RISK_REGIMES = ("GREEN", "YELLOW", "RED")


def _validate_frozen_spec(
    min_train_weeks: int, alpha: float, top_k: int
) -> None:
    if (
        min_train_weeks != FROZEN_MIN_TRAIN_WEEKS
        or alpha != FROZEN_ALPHA
        or top_k != FROZEN_TOP_K
    ):
        raise ValueError(
            "frozen Sector RORO v1 requires min_train_weeks=156, alpha=10, top_k=3; "
            "use a new model version for different settings"
        )
SUPPORTIVE_HEADINGS = {"NE_STRENGTHENING", "NW_EARLY_RECOVERY"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def roro_context_from_history(
    shadow_database: str | Path,
    *,
    series_version: str = RORO_SERIES_VERSION,
) -> pd.DataFrame:
    """Read only the exact aggregate context needed by the sector SHADOW."""
    rows = []
    for observation in read_shadow_history(shadow_database, series_version=series_version):
        payload = observation["payload"]
        rows.append(
            {
                "date": pd.Timestamp(observation["date"]),
                "market_risk_score": payload.get("market_risk_score"),
                "risk_regime": payload.get("risk_regime"),
                "trajectory_5d": payload.get("trajectory_5d"),
                "data_status": payload.get("data_status"),
            }
        )
    result = pd.DataFrame(
        rows,
        columns=[
            "date",
            "market_risk_score",
            "risk_regime",
            "trajectory_5d",
            "data_status",
        ],
    )
    if result.empty:
        raise ValueError(f"no RORO context for series: {series_version}")
    if result["date"].duplicated().any():
        raise ValueError("RORO context dates must be unique")
    return result.sort_values("date").reset_index(drop=True)


def sector_beta_ranks(
    prices: pd.DataFrame,
    as_of: str | pd.Timestamp,
    *,
    benchmark: str = "SPY",
    window: int = 60,
    minimum_observations: int = 40,
) -> pd.DataFrame:
    """Calculate current 60D SPY beta and its cross-sectional mid-rank."""
    if window < minimum_observations or minimum_observations < 20:
        raise ValueError("beta window must be >= minimum observations >= 20")
    cutoff = pd.Timestamp(as_of)
    clean = prices.loc[prices.index <= cutoff].sort_index().apply(
        pd.to_numeric, errors="coerce"
    )
    if cutoff not in clean.index:
        raise ValueError(f"sector price date is unavailable: {cutoff.date()}")
    if benchmark not in clean:
        raise ValueError(f"benchmark column is missing: {benchmark}")
    returns = clean.pct_change(fill_method=None).tail(window)
    rows = []
    for ticker in SECTOR_ETFS:
        if ticker not in returns:
            continue
        pair = returns[[ticker, benchmark]].dropna()
        beta = np.nan
        if len(pair) >= minimum_observations:
            variance = float(pair[benchmark].var(ddof=1))
            if np.isfinite(variance) and variance > 0:
                beta = float(pair[ticker].cov(pair[benchmark]) / variance)
        rows.append({"ticker": ticker, "sector_beta_60": beta})
    result = pd.DataFrame(rows)
    finite = pd.to_numeric(result["sector_beta_60"], errors="coerce").dropna()
    if len(finite) < 3:
        raise ValueError("at least three sectors need a valid 60D beta")
    ranks = finite.rank(method="average")
    result["beta_rank"] = np.nan
    result.loc[finite.index, "beta_rank"] = (ranks - 0.5) / len(finite)
    return result


def apply_roro_context(
    forecast: pd.DataFrame,
    rrg_snapshot: pd.DataFrame,
    beta_ranks: pd.DataFrame,
    roro_row: pd.Series,
    *,
    top_k: int = 3,
) -> pd.DataFrame:
    """Combine three independently visible signals using the frozen status table."""
    score = pd.to_numeric(pd.Series([roro_row.get("market_risk_score")]), errors="coerce").iloc[0]
    if not np.isfinite(score) or score < 0 or score > 100:
        raise ValueError("exact RORO score must be between 0 and 100")
    columns = [
        "ticker",
        "quadrant",
        "quadrant_transition",
        "tail_heading",
        "delta_rs_ratio",
        "delta_rs_momentum",
    ]
    missing = [column for column in columns if column not in rrg_snapshot]
    if missing:
        raise ValueError("missing RRG snapshot columns: " + ", ".join(missing))
    result = forecast.merge(
        rrg_snapshot[columns], on="ticker", how="left", validate="one_to_one"
    ).merge(beta_ranks, on="ticker", how="left", validate="one_to_one")
    if result[["tail_heading", "beta_rank"]].isna().any().any():
        raise ValueError("RRG heading and beta rank must be complete")
    risk_appetite = 1.0 - float(score) / 100.0
    result["roro_fit"] = (
        risk_appetite * result["beta_rank"]
        + (1.0 - risk_appetite) * (1.0 - result["beta_rank"])
    )
    result["alpha_positive"] = result["predicted_excess_20d"] > 0
    result["alpha_top3"] = result["forecast_rank"] <= top_k
    result["rrg_supportive"] = result["tail_heading"].isin(SUPPORTIVE_HEADINGS)
    result["roro_context"] = np.where(result["roro_fit"] >= 0.5, "FIT", "RORO_MISMATCH")
    result["alignment"] = np.where(
        result["alpha_positive"] == result["rrg_supportive"],
        "ALIGNED",
        "DIVERGENCE",
    )
    select = (
        result["alpha_positive"]
        & result["alpha_top3"]
        & result["rrg_supportive"]
        & (result["roro_fit"] >= 0.5)
    )
    avoid = ~result["alpha_positive"] & ~result["rrg_supportive"]
    result["sector_status"] = np.select(
        [select, avoid], ["SELECT", "AVOID"], default="WATCH"
    )
    result["market_risk_score"] = int(score)
    result["risk_regime"] = roro_row.get("risk_regime")
    result["roro_trajectory_5d"] = roro_row.get("trajectory_5d")
    result["roro_data_status"] = roro_row.get("data_status")
    return result.sort_values(["forecast_rank", "ticker"]).reset_index(drop=True)


def fit_sector_roro_shadow(
    feature_frame: pd.DataFrame,
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    roro_context: pd.DataFrame,
    *,
    as_of: str | pd.Timestamp | None = None,
    min_train_weeks: int = 156,
    alpha: float = 10.0,
    top_k: int = 3,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit causal alpha, then attach same-date RRG and RORO context."""
    _validate_frozen_spec(min_train_weeks, alpha, top_k)
    forecast, metadata = fit_leading_forecast(
        feature_frame,
        as_of=as_of,
        min_train_weeks=min_train_weeks,
        alpha=alpha,
        top_k=top_k,
        feature_columns=BASE_FEATURE_COLUMNS,
    )
    current_date = pd.Timestamp(metadata["as_of"])
    exact = roro_context.loc[roro_context["date"] == current_date]
    if len(exact) != 1:
        raise ValueError(f"no exact RORO context for {current_date.date()}")
    snapshot = latest_snapshot(signals, current_date, tail_days=5)
    beta = sector_beta_ranks(prices, current_date)
    result = apply_roro_context(
        forecast,
        snapshot,
        beta,
        exact.iloc[0],
        top_k=top_k,
    )
    metadata = {
        **metadata,
        "model_version": MODEL_VERSION,
        "roro_series_version": RORO_SERIES_VERSION,
        "context_type": "RORO_EXACT_DATE_BETA_FIT",
        "roro_score": int(exact.iloc[0]["market_risk_score"]),
        "roro_regime": exact.iloc[0]["risk_regime"],
        "roro_trajectory_5d": exact.iloc[0]["trajectory_5d"],
        "select_count": int((result["sector_status"] == "SELECT").sum()),
        "status_contract": "SELECT/WATCH/AVOID; no forced Top3 fill",
    }
    return result, metadata


def _equal_weight_turnover(previous: set[str], current: set[str]) -> float:
    names = previous | current
    previous_weight = 1.0 / len(previous) if previous else 0.0
    current_weight = 1.0 / len(current) if current else 0.0
    asset_turnover = sum(
        abs((current_weight if name in current else 0.0) - (previous_weight if name in previous else 0.0))
        for name in names
    )
    previous_cash = 0.0 if previous else 1.0
    current_cash = 0.0 if current else 1.0
    return 0.5 * (asset_turnover + abs(current_cash - previous_cash))


def _is_expected_data_insufficiency(error: ValueError) -> bool:
    message = str(error)
    prefixes = (
        "no forecast features are available on or before as_of",
        "no completed weekly signal is available",
        "not enough sectors have complete current features",
        "sector price date is unavailable:",
        "at least three sectors need a valid 60D beta",
        "no exact RORO context for ",
        "no signal is available on or before the requested date",
        "RRG heading and beta rank must be complete",
    )
    return message.startswith(prefixes) or (
        message.startswith("need ") and " known training weeks; found " in message
    )


def _cutoff_inputs(
    feature_frame: pd.DataFrame,
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    roro_context: pd.DataFrame,
    as_of: str | pd.Timestamp | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Timestamp | None]:
    if as_of is None:
        return feature_frame, prices, signals, roro_context, None
    cutoff = pd.Timestamp(as_of)
    features = feature_frame.loc[feature_frame["date"] <= cutoff].copy()
    price_dates = pd.to_datetime(prices.index)
    prices = prices.loc[price_dates <= cutoff].copy()
    signals = (
        signals.loc[signals["date"] <= cutoff].copy() if not signals.empty else signals.copy()
    )
    roro_context = (
        roro_context.loc[roro_context["date"] <= cutoff].copy()
        if not roro_context.empty
        else roro_context.copy()
    )
    return features, prices, signals, roro_context, cutoff


def validate_sector_roro_shadow(
    feature_frame: pd.DataFrame,
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    roro_context: pd.DataFrame,
    *,
    min_train_weeks: int = 156,
    alpha: float = 10.0,
    top_k: int = 3,
    cost_bps: float = 10.0,
    bootstrap_samples: int = 1000,
    as_of: str | pd.Timestamp | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate weekly 2026 context without pretending it is a long history."""
    _validate_frozen_spec(min_train_weeks, alpha, top_k)
    feature_frame, prices, signals, roro_context, cutoff = _cutoff_inputs(
        feature_frame, prices, signals, roro_context, as_of
    )
    weekly_dates = (
        feature_frame.groupby(feature_frame["date"].dt.to_period("W-FRI"))["date"]
        .max()
        .sort_values()
    )
    if cutoff is not None:
        weekly_dates = weekly_dates.loc[
            (weekly_dates == OUTPUT_START)
            | (
                weekly_dates.dt.to_period("W-FRI").dt.end_time.dt.normalize()
                <= cutoff.normalize()
            )
        ]
    completed_mask = feature_frame["target_excess_20d"].notna() & (
        feature_frame["date"] >= OUTPUT_START
    )
    if cutoff is not None and "label_end" in feature_frame:
        completed_mask &= pd.to_datetime(feature_frame["label_end"], errors="coerce") <= cutoff
    completed = feature_frame.loc[completed_mask, "date"]
    dates = [day for day in weekly_dates if day in set(completed)]
    rows: list[pd.DataFrame] = []
    ablation_previous = {name: set() for name in ABLATION_NAMES}
    ablation_results: dict[str, dict[str, list[Any]]] = {
        name: {"gross": [], "net": [], "turnover": [], "returns": []}
        for name in ABLATION_NAMES
    }
    weekly_top_bottom: list[float] = []
    skipped: list[dict[str, str]] = []
    for current_date in dates:
        try:
            ranked, _ = fit_sector_roro_shadow(
                feature_frame,
                prices,
                signals,
                roro_context,
                as_of=current_date,
                min_train_weeks=min_train_weeks,
                alpha=alpha,
                top_k=top_k,
            )
        except ValueError as error:
            if not _is_expected_data_insufficiency(error):
                raise
            skipped.append({
                "date": pd.Timestamp(current_date).date().isoformat(),
                "reason": str(error),
            })
            continue
        actual = feature_frame.loc[
            feature_frame["date"] == current_date,
            ["ticker", "target_excess_20d"],
        ].dropna()
        scored = ranked.merge(actual, on="ticker", how="inner", validate="one_to_one")
        if len(scored) < top_k:
            skipped.append({
                "date": pd.Timestamp(current_date).date().isoformat(),
                "reason": f"only {len(scored)} labelled sectors; requires {top_k}",
            })
            continue
        regimes = scored["risk_regime"].dropna().unique()
        if len(regimes) != 1 or regimes[0] not in RISK_REGIMES:
            raise ValueError(f"invalid exact-date RORO regime for {current_date.date()}")
        alpha_only = (scored["predicted_excess_20d"] > 0) & (
            scored["forecast_rank"] <= top_k
        )
        ablation_masks = {
            "alpha-only": alpha_only,
            "alpha+RRG": alpha_only & scored["rrg_supportive"].astype(bool),
            "alpha+RORO": alpha_only & (scored["roro_fit"] >= 0.5),
            "full": scored["sector_status"] == "SELECT",
        }
        for name, mask in ablation_masks.items():
            selected_rows = scored.loc[mask]
            current = set(selected_rows["ticker"])
            turnover = _equal_weight_turnover(ablation_previous[name], current)
            gross = (
                float(selected_rows["target_excess_20d"].mean())
                if len(selected_rows)
                else 0.0
            )
            ablation_results[name]["gross"].append(gross)
            ablation_results[name]["net"].append(
                gross - turnover * cost_bps / 10_000.0
            )
            ablation_results[name]["turnover"].append(turnover)
            ablation_results[name]["returns"].extend(
                selected_rows["target_excess_20d"].astype(float).tolist()
            )
            ablation_previous[name] = current
            scored[f"ablation_{name.replace('+', '_').replace('-', '_')}"] = mask
        selected_return = ablation_results["full"]["gross"][-1]
        turnover = ablation_results["full"]["turnover"][-1]
        top = scored.nsmallest(top_k, "forecast_rank")["target_excess_20d"].mean()
        bottom = scored.nlargest(top_k, "forecast_rank")["target_excess_20d"].mean()
        weekly_top_bottom.append(float(top - bottom))
        scored["validation_date"] = current_date
        scored["select_turnover"] = turnover
        scored["selected_net_excess"] = selected_return - turnover * cost_bps / 10_000.0
        rows.append(scored)
    observations = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    weeks = int(observations["validation_date"].nunique()) if not observations.empty else 0
    selected = observations.loc[observations.get("sector_status") == "SELECT"] if not observations.empty else pd.DataFrame()
    select_returns = selected["target_excess_20d"] if not selected.empty else pd.Series(dtype=float)
    weekly_gross = ablation_results["full"]["gross"]
    weekly_net = ablation_results["full"]["net"]
    weekly_turnover = ablation_results["full"]["turnover"]
    select_ci = _block_bootstrap_mean_ci(
        pd.Series(weekly_gross, dtype=float),
        block_weeks=4,
        samples=bootstrap_samples,
    )
    rank_ics = []
    if not observations.empty:
        for _, group in observations.groupby("validation_date"):
            rank_ics.append(
                float(group["predicted_excess_20d"].rank().corr(group["target_excess_20d"].rank()))
            )
    select_hit_rate = float((select_returns > 0).mean()) if len(select_returns) else None
    finite_rank_ics = [value for value in rank_ics if np.isfinite(value)]
    mean_rank_ic = float(np.mean(finite_rank_ics)) if finite_rank_ics else None
    gates = {
        "minimum_weeks": weeks >= 52,
        "minimum_select_observations": len(selected) >= 30,
        "mean_rank_ic_positive": mean_rank_ic is not None and mean_rank_ic > 0,
        "select_hit_rate_above_half": select_hit_rate is not None and select_hit_rate > 0.5,
        "select_excess_ci_above_zero": select_ci[0] is not None and select_ci[0] > 0,
    }
    evidence = gates["minimum_weeks"] and gates["minimum_select_observations"]
    status = "INSUFFICIENT_EVIDENCE" if not evidence else (
        "HISTORICAL_PASS" if all(gates.values()) else "HISTORICAL_FAIL"
    )
    ablations = {}
    for name, values in ablation_results.items():
        returns = pd.Series(values["returns"], dtype=float)
        ablations[name] = {
            "weeks": int(len(values["gross"])),
            "observations": int(len(returns)),
            "hit_rate": float((returns > 0).mean()) if len(returns) else None,
            "mean_weekly_gross_excess": (
                float(np.mean(values["gross"])) if values["gross"] else None
            ),
            "mean_weekly_net_excess_10bp": (
                float(np.mean(values["net"])) if values["net"] else None
            ),
            "mean_weekly_turnover": (
                float(np.mean(values["turnover"])) if values["turnover"] else None
            ),
        }
    regime_breakdown = {}
    for regime in RISK_REGIMES:
        regime_rows = (
            observations.loc[observations["risk_regime"] == regime]
            if not observations.empty
            else pd.DataFrame()
        )
        regime_select = (
            regime_rows.loc[regime_rows["sector_status"] == "SELECT"]
            if not regime_rows.empty
            else pd.DataFrame()
        )
        regime_returns = (
            regime_select["target_excess_20d"]
            if not regime_select.empty
            else pd.Series(dtype=float)
        )
        regime_breakdown[regime] = {
            "weeks": int(regime_rows["validation_date"].nunique()) if not regime_rows.empty else 0,
            "observations": int(len(regime_select)),
            "hit_rate": float((regime_returns > 0).mean()) if len(regime_returns) else None,
            "mean_excess_20d": float(regime_returns.mean()) if len(regime_returns) else None,
        }
    summary: dict[str, Any] = {
        "model_version": MODEL_VERSION,
        "status": status,
        "overall_pass": status == "HISTORICAL_PASS",
        "definition": {
            "market_context": "NQ=F RORO exact date",
            "relative_benchmark": "SPY",
            "target": "t+1..t+20 sector return minus equal-sector return",
            "features": BASE_FEATURE_COLUMNS,
            "minimum_training_weeks": min_train_weeks,
            "cost_bps": cost_bps,
            "as_of": cutoff.date().isoformat() if cutoff is not None else None,
        },
        "coverage": {
            "candidate_weeks": int(len(dates)),
            "weeks": weeks,
            "skipped_weeks": int(len(skipped)),
            "skipped": skipped,
            "start": observations["validation_date"].min().date().isoformat() if weeks else None,
            "end": observations["validation_date"].max().date().isoformat() if weeks else None,
            "sector_rows": int(len(observations)),
            "select_observations": int(len(selected)),
        },
        "metrics": {
            "mean_rank_ic": mean_rank_ic,
            "select_hit_rate": select_hit_rate,
            "mean_weekly_select_gross_excess": float(np.mean(weekly_gross)) if weekly_gross else None,
            "select_excess_95ci": list(select_ci),
            "mean_weekly_net_excess_10bp": float(np.mean(weekly_net)) if weekly_net else None,
            "mean_weekly_turnover": float(np.mean(weekly_turnover)) if weekly_turnover else None,
            "mean_top3_minus_bottom3_excess": float(np.mean(weekly_top_bottom)) if weekly_top_bottom else None,
            "rrg_alpha_alignment_rate": float((observations["alignment"] == "ALIGNED").mean()) if weeks else None,
        },
        "ablations": ablations,
        "roro_regime_breakdown": regime_breakdown,
        "gates": gates,
    }
    return summary, observations


def load_inputs(
    publish_database: str | Path,
    shadow_database: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prices = load_price_matrix(Path(publish_database))
    signals = build_rrg_signals(prices)
    features = build_sector_feature_frame(prices, signals, horizon=20)
    roro = roro_context_from_history(shadow_database)
    return prices, signals, features, roro


def build_research_history(
    feature_frame: pd.DataFrame,
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    roro_context: pd.DataFrame,
    *,
    start: str | pd.Timestamp = "2025-01-01",
    as_of: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Replay weekly alpha/RRG; mark pre-portfolio RORO context as N/A."""
    start_date = pd.Timestamp(start).normalize()
    cutoff = pd.Timestamp(as_of).normalize() if as_of is not None else feature_frame["date"].max()
    eligible_dates = feature_frame.loc[
        feature_frame["date"].between(start_date, cutoff), "date"
    ]
    dates = eligible_dates.groupby(eligible_dates.dt.to_period("W-FRI")).max().sort_values()
    dates = dates.loc[dates.dt.to_period("W-FRI").dt.end_time.dt.normalize() <= cutoff]
    exact_roro_dates = set(
        pd.to_datetime(roro_context.get("date", pd.Series(dtype="datetime64[ns]"))).dt.normalize()
    )
    rows: list[pd.DataFrame] = []
    for decision_date in dates:
        if decision_date in exact_roro_dates:
            ranked, _ = fit_sector_roro_shadow(
                feature_frame, prices, signals, roro_context, as_of=decision_date
            )
        else:
            ranked, _ = fit_leading_forecast(
                feature_frame,
                as_of=decision_date,
                min_train_weeks=FROZEN_MIN_TRAIN_WEEKS,
                alpha=FROZEN_ALPHA,
                top_k=FROZEN_TOP_K,
                feature_columns=BASE_FEATURE_COLUMNS,
            )
            ranked = ranked.merge(
                latest_snapshot(signals, decision_date, tail_days=5)[
                    ["ticker", "quadrant", "quadrant_transition", "tail_heading"]
                ],
                on="ticker",
                how="left",
                validate="one_to_one",
            )
            ranked["roro_fit"] = np.nan
            ranked["roro_context"] = "N/A"
            ranked["sector_status"] = "N/A"
        actual = feature_frame.loc[
            feature_frame["date"] == decision_date,
            ["ticker", "target_excess_20d"],
        ].rename(columns={"target_excess_20d": "actual_excess_20d"})
        ranked = ranked.merge(actual, on="ticker", how="left", validate="one_to_one")
        ranked.insert(0, "decision_date", decision_date)
        ranked["alpha_hit"] = np.where(
            ranked["actual_excess_20d"].notna(),
            (ranked["predicted_excess_20d"] > 0)
            == (ranked["actual_excess_20d"] > 0),
            pd.NA,
        )
        rows.append(ranked)
    if not rows:
        raise ValueError("no weekly Sector research dates are available")
    history = pd.concat(rows, ignore_index=True).sort_values(["ticker", "decision_date"])
    previous = history.groupby("ticker")["roro_context"].shift()
    valid = history["roro_context"].isin(["FIT", "RORO_MISMATCH"])
    previous_valid = previous.isin(["FIT", "RORO_MISMATCH"])
    history["fit_change"] = np.where(history["roro_context"].eq("N/A"), "N/A", "—")
    changed = valid & previous_valid & history["roro_context"].ne(previous)
    history.loc[changed, "fit_change"] = (
        previous[changed].str.replace("RORO_", "", regex=False)
        + " → "
        + history.loc[changed, "roro_context"].str.replace("RORO_", "", regex=False)
    )
    return history.sort_values(["decision_date", "forecast_rank", "ticker"]).reset_index(drop=True)


def _format_metric(value: Any, spec: str) -> str:
    if value is None or pd.isna(value):
        return "null"
    return format(float(value), spec)


def markdown_report(summary: dict[str, Any]) -> str:
    coverage = summary["coverage"]
    metrics = summary["metrics"]
    missing = []
    if coverage["weeks"] < 52:
        missing.append(f"weeks {coverage['weeks']}/52")
    if coverage["select_observations"] < 30:
        missing.append(f"SELECT observations {coverage['select_observations']}/30")
    evidence_line = (
        "Frozen minimum evidence not yet met: " + ", ".join(missing) + "."
        if missing
        else "Frozen minimum evidence met."
    )
    lines = [
        "# Sector RORO SHADOW Validation",
        "",
        f"- Status: **{summary['status']}**",
        f"- Weeks: {coverage['weeks']}",
        f"- Skipped candidate weeks: {coverage['skipped_weeks']}",
        f"- SELECT observations: {coverage['select_observations']}",
        f"- Mean rank IC: {_format_metric(metrics['mean_rank_ic'], '.3f')}",
        f"- SELECT hit rate: {_format_metric(metrics['select_hit_rate'], '.1%')}",
        "- Weekly SELECT gross excess: "
        f"{_format_metric(metrics['mean_weekly_select_gross_excess'], '.2%')}",
        "",
        evidence_line,
        "",
        "## Fixed ablations",
        "",
        "| Variant | Weeks | Observations | Hit rate | Gross | Net (10bp) | Turnover |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ABLATION_NAMES:
        row = summary["ablations"][name]
        lines.append(
            f"| {name} | {row['weeks']} | {row['observations']} | "
            f"{_format_metric(row['hit_rate'], '.1%')} | "
            f"{_format_metric(row['mean_weekly_gross_excess'], '.2%')} | "
            f"{_format_metric(row['mean_weekly_net_excess_10bp'], '.2%')} | "
            f"{_format_metric(row['mean_weekly_turnover'], '.2%')} |"
        )
    lines.extend(
        [
            "",
            "## Full SELECT by RORO regime",
            "",
            "| Regime | Weeks | Observations | Hit rate | Mean 20D excess |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for regime in RISK_REGIMES:
        row = summary["roro_regime_breakdown"][regime]
        lines.append(
            f"| {regime} | {row['weeks']} | {row['observations']} | "
            f"{_format_metric(row['hit_rate'], '.1%')} | "
            f"{_format_metric(row['mean_excess_20d'], '.2%')} |"
        )
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    status = commands.add_parser("status")
    validate = commands.add_parser("validate")
    history = commands.add_parser("history")
    for command in (status, validate, history):
        command.add_argument("publish_database")
        command.add_argument("shadow_database")
        command.add_argument("--min-train-weeks", type=int, default=156)
        command.add_argument("--alpha", type=float, default=10.0)
        command.add_argument("--top-k", type=int, default=3)
    status.add_argument("--date")
    validate.add_argument("--output", default="outputs/sector_roro_shadow/validation.json")
    validate.add_argument("--bootstrap-samples", type=int, default=1000)
    validate.add_argument("--as-of")
    history.add_argument("--start", default="2025-01-01")
    history.add_argument("--as-of")
    history.add_argument("--output", default="outputs/sector_roro_shadow/history.csv")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    prices, signals, features, roro = load_inputs(
        args.publish_database, args.shadow_database
    )
    if args.command == "status":
        result, metadata = fit_sector_roro_shadow(
            features,
            prices,
            signals,
            roro,
            as_of=args.date,
            min_train_weeks=args.min_train_weeks,
            alpha=args.alpha,
            top_k=args.top_k,
        )
        print(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False))
        print(result.to_string(index=False))
        return
    if args.command == "history":
        result = build_research_history(
            features, prices, signals, roro, start=args.start, as_of=args.as_of
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output, index=False)
        summary = {
            "start": result["decision_date"].min().date().isoformat(),
            "end": result["decision_date"].max().date().isoformat(),
            "weeks": int(result["decision_date"].nunique()),
            "sector_rows": int(len(result)),
            "roro_na_rows": int(result["roro_context"].eq("N/A").sum()),
            "fit_changes": int(result["fit_change"].str.contains("→", na=False).sum()),
            "output": str(output.resolve()),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
        return
    summary, observations = validate_sector_roro_shadow(
        features,
        prices,
        signals,
        roro,
        min_train_weeks=args.min_train_weeks,
        alpha=args.alpha,
        top_k=args.top_k,
        bootstrap_samples=args.bootstrap_samples,
        as_of=args.as_of,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    publish = Path(args.publish_database)
    shadow = Path(args.shadow_database)
    summary["roro_series_version"] = RORO_SERIES_VERSION
    summary["provenance"] = {
        "publish_database_sha256": _sha256(publish),
        "roro_shadow_database_sha256": _sha256(shadow),
        "model_spec_sha256": _sha256(Path(__file__).with_name("MODEL_SPEC_SECTOR_RORO_v1.md")),
        "source_sha256": _sha256(Path(__file__)),
    }
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    observations.to_csv(output.with_name(output.stem + "_observations.csv"), index=False)
    output.with_suffix(".md").write_text(markdown_report(summary), encoding="utf-8")
    output.with_suffix(".log").write_text(
        f"status={summary['status']}\nweeks={summary['coverage']['weeks']}\n"
        f"skipped_weeks={summary['coverage']['skipped_weeks']}\n"
        f"select_observations={summary['coverage']['select_observations']}\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
