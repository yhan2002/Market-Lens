"""Research-only PIT-conscious sector flow comparison.

This CLI reads Bloomberg weekly fund flow/AUM history and the publish database in
read-only mode.  It never writes operational databases or web artifacts.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sector_rotation import SECTOR_ETFS, build_rrg_signals
from sector_rotation_forecast import (
    BASE_FEATURE_COLUMNS,
    _block_bootstrap_mean_ci,
    build_sector_feature_frame,
)
from sector_shadow_store import _load_price_matrix_read_only


FLOW_UNIVERSE = ("SPY", *SECTOR_ETFS)
FLOW_FEATURES = ("flow_rank_1w", "flow_rank_4w", "flow_rank_accel")
EARLY_EVENT_RULE = {
    "flow_rank_accel_min": 0.20,
    "flow_rank_4w_min": 0.0,
    "price_rank_20_max": 8 / 11,
    "rrg_momentum_delta_5_min": 0.0,
}
MODEL_FEATURES = {
    "B2_PRICE_RIDGE": tuple(BASE_FEATURE_COLUMNS),
    "C1_FLOW_RIDGE": FLOW_FEATURES,
    "C2_PRICE_FLOW_RIDGE": (*BASE_FEATURE_COLUMNS, *FLOW_FEATURES),
}
MODEL_ORDER = ("B1_RRG", *MODEL_FEATURES)
REQUIRED_FLOW_COLUMNS = {
    "local_ticker",
    "date",
    "FUND_FLOW",
    "FUND_TOTAL_ASSETS",
}
PIT_STATUS = "PIT_PARTIAL"
TARGET_DESCRIPTION = (
    "P[t+20]/P[t]-1 sector return minus the same-date equal-weight return "
    "of all 11 sector ETFs"
)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def load_flow_history(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    raw = pd.read_csv(source)
    missing = sorted(REQUIRED_FLOW_COLUMNS - set(raw.columns))
    if missing:
        raise ValueError("missing Bloomberg flow columns: " + ", ".join(missing))
    frame = raw[["local_ticker", "date", "FUND_FLOW", "FUND_TOTAL_ASSETS"]].rename(
        columns={
            "local_ticker": "ticker",
            "FUND_FLOW": "fund_flow_1w",
            "FUND_TOTAL_ASSETS": "aum",
        }
    )
    frame["ticker"] = frame["ticker"].astype(str).str.strip().str.upper()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["fund_flow_1w"] = pd.to_numeric(frame["fund_flow_1w"], errors="coerce")
    frame["aum"] = pd.to_numeric(frame["aum"], errors="coerce")
    if frame[["ticker", "date", "fund_flow_1w", "aum"]].isna().any().any():
        raise ValueError("Bloomberg flow history contains missing or invalid required values")
    if frame.duplicated(["ticker", "date"]).any():
        raise ValueError("Bloomberg flow history contains duplicate ticker/date rows")
    if (frame["aum"] <= 0).any():
        raise ValueError("FUND_TOTAL_ASSETS must be positive")
    expected = set(FLOW_UNIVERSE)
    actual = set(frame["ticker"])
    if actual != expected:
        raise ValueError(
            "flow universe mismatch; missing="
            + ",".join(sorted(expected - actual))
            + " unexpected="
            + ",".join(sorted(actual - expected))
        )
    if not frame["date"].dt.dayofweek.eq(4).all():
        raise ValueError("Bloomberg flow dates must be W-FRI source weeks")
    coverage = frame.groupby("date")["ticker"].agg(["size", "nunique"])
    if not coverage.eq(len(FLOW_UNIVERSE)).all().all():
        raise ValueError("each Bloomberg source week must contain all 12 ETFs exactly once")
    return frame.sort_values(["ticker", "date"]).reset_index(drop=True)


def _centered_rank(frame: pd.DataFrame, value: str) -> pd.Series:
    ranks = frame.groupby("source_week")[value].rank(method="average", pct=True)
    return ranks - ranks.groupby(frame["source_week"]).transform("mean")


def build_flow_features(
    flow: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    *,
    universe: Sequence[str] = FLOW_UNIVERSE,
) -> pd.DataFrame:
    """Create three ranks and attach source week only to exact W+1 trading week."""
    expected = set(universe)
    if set(flow["ticker"]) != expected:
        raise ValueError("flow feature universe does not match the requested universe")
    frame = flow.sort_values(["ticker", "date"]).copy()
    grouped = frame.groupby("ticker", group_keys=False)
    frame["prior_aum"] = grouped["aum"].shift(1)
    frame["flow_aum_1w"] = frame["fund_flow_1w"] / frame["prior_aum"]
    frame["flow_aum_4w"] = grouped["flow_aum_1w"].transform(
        lambda values: values.rolling(4, min_periods=4).sum()
    )
    prior_four_week_mean = grouped["flow_aum_1w"].transform(
        lambda values: values.shift(1).rolling(4, min_periods=4).mean()
    )
    frame["flow_aum_accel"] = frame["flow_aum_1w"] - prior_four_week_mean
    frame["source_week"] = frame["date"].dt.to_period("W-FRI")
    frame["usable_week"] = frame["source_week"] + 1
    rankable = frame.loc[frame["ticker"] != "SPY"] if "SPY" in expected else frame
    frame["flow_rank_1w"] = _centered_rank(rankable, "flow_aum_1w")
    frame["flow_rank_4w"] = _centered_rank(rankable, "flow_aum_4w")
    frame["flow_rank_accel"] = _centered_rank(rankable, "flow_aum_accel")

    calendar = pd.DataFrame({"date": pd.DatetimeIndex(trading_dates).normalize()})
    calendar["usable_week"] = calendar["date"].dt.to_period("W-FRI")
    week_ends = calendar.groupby("usable_week", as_index=False)["date"].max()
    frame = frame.merge(week_ends, on="usable_week", how="left", validate="many_to_one")
    frame = frame.dropna(subset=["date_y", *FLOW_FEATURES]).rename(
        columns={"date_x": "flow_source_date", "date_y": "date"}
    )
    return frame[
        ["date", "ticker", "flow_source_date", "source_week", "usable_week", *FLOW_FEATURES]
    ].sort_values(["date", "ticker"]).reset_index(drop=True)


def build_common_panel(
    prices: pd.DataFrame,
    flow_features: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    signals = build_rrg_signals(prices)
    daily = build_sector_feature_frame(prices, signals, horizon=20)
    daily["sector"] = daily["ticker"].map(SECTOR_ETFS)
    panel = daily.merge(flow_features, on=["date", "ticker"], how="inner", validate="one_to_one")
    required = [
        "shadow_score",
        "target_excess_20d",
        "label_end",
        *BASE_FEATURE_COLUMNS,
        *FLOW_FEATURES,
    ]
    expected = set(SECTOR_ETFS)
    valid_dates: list[pd.Timestamp] = []
    for decision_date, rows in panel.groupby("date", sort=True):
        complete = (
            set(rows["ticker"]) == expected
            and len(rows) == len(expected)
            and rows[required].notna().all().all()
        )
        if complete:
            valid_dates.append(pd.Timestamp(decision_date))
    complete = panel.loc[panel["date"].isin(valid_dates)].copy()
    if complete.empty:
        raise ValueError("no complete 11-sector price/flow weeks are available")
    target_means = complete.groupby("date")["target_excess_20d"].mean().abs()
    if (target_means > 1e-10).any():
        raise ValueError("equal-sector target is not centered across all 11 sectors")
    metadata = {
        "price_flow_join_weeks": int(panel["date"].nunique()),
        "common_complete_weeks": int(complete["date"].nunique()),
        "common_complete_rows": int(len(complete)),
    }
    return complete.sort_values(["date", "ticker"]).reset_index(drop=True), metadata


def build_early_rotation_events(panel: pd.DataFrame) -> pd.DataFrame:
    """Flag an early candidate before a sector has entered the top-three price leaders.

    This is a fixed research screen, not a fitted forecast.  Flow is available only
    with the one-week lag already enforced by build_flow_features.
    """
    required = {
        "date",
        "ticker",
        "target_excess_20d",
        "flow_rank_accel",
        "flow_rank_4w",
        "relative_return_20",
        "rrg_momentum_delta_5",
    }
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError("early-event panel missing: " + ", ".join(missing))
    events = panel.copy()
    events["price_rank_20"] = events.groupby("date")["relative_return_20"].rank(pct=True)
    events["early_rotation_in"] = (
        (events["flow_rank_accel"] >= EARLY_EVENT_RULE["flow_rank_accel_min"])
        & (events["flow_rank_4w"] >= EARLY_EVENT_RULE["flow_rank_4w_min"])
        & (events["price_rank_20"] <= EARLY_EVENT_RULE["price_rank_20_max"])
        & (events["rrg_momentum_delta_5"] > EARLY_EVENT_RULE["rrg_momentum_delta_5_min"])
    )
    columns = [
        "date",
        "ticker",
        "target_excess_20d",
        "flow_rank_accel",
        "flow_rank_4w",
        "price_rank_20",
        "rrg_momentum_delta_5",
        "early_rotation_in",
    ]
    return events[columns].sort_values(["date", "ticker"]).reset_index(drop=True)


def summarize_early_rotation_events(
    events: pd.DataFrame, *, evaluation_dates: Sequence[pd.Timestamp], bootstrap_samples: int
) -> dict[str, Any]:
    """Summarize OOS candidate events by decision week, avoiding sector-cluster weighting."""
    evaluated = events.loc[events["date"].isin(pd.to_datetime(list(evaluation_dates)))].copy()
    selected = evaluated.loc[evaluated["early_rotation_in"]].copy()
    by_week = selected.groupby("date", as_index=False)["target_excess_20d"].mean()
    non_selected = evaluated.loc[~evaluated["early_rotation_in"]].groupby("date")[
        "target_excess_20d"
    ].mean()
    by_week["non_event_excess_20d"] = by_week["date"].map(non_selected)
    by_week["event_minus_non_event"] = (
        by_week["target_excess_20d"] - by_week["non_event_excess_20d"]
    )
    return {
        "label": "EXPLORATORY_FIXED_SCREEN_OOS",
        "rule": EARLY_EVENT_RULE,
        "evaluation_weeks": int(evaluated["date"].nunique()),
        "event_rows": int(len(selected)),
        "event_weeks": int(len(by_week)),
        "event_hit_rate": _finite((selected["target_excess_20d"] > 0).mean()),
        "mean_event_excess_20d": _finite(by_week["target_excess_20d"].mean()),
        "mean_event_minus_non_event_20d": _finite(by_week["event_minus_non_event"].mean()),
        "block_5w_95ci": {
            "event_excess_20d": list(
                _block_bootstrap_mean_ci(
                    by_week["target_excess_20d"], block_weeks=5, samples=bootstrap_samples
                )
            ),
            "event_minus_non_event_20d": list(
                _block_bootstrap_mean_ci(
                    by_week["event_minus_non_event"], block_weeks=5, samples=bootstrap_samples
                )
            ),
        },
        "note": (
            "The thresholds are a research hypothesis declared in code; the short, "
            "PIT_PARTIAL sample cannot establish a leading signal."
        ),
    }


def _common_folds(panel: pd.DataFrame, min_train_weeks: int) -> list[pd.Timestamp]:
    if min_train_weeks < 2:
        raise ValueError("min_train_weeks must be at least 2")
    folds: list[pd.Timestamp] = []
    for decision_date in sorted(pd.to_datetime(panel["date"].unique())):
        known = panel.loc[pd.to_datetime(panel["label_end"]) < decision_date]
        if known["date"].nunique() >= min_train_weeks:
            folds.append(pd.Timestamp(decision_date))
    if not folds:
        raise ValueError("no expanding-window OOS folds satisfy the minimum training history")
    return folds


def _fit_ridge(
    panel: pd.DataFrame,
    decision_date: pd.Timestamp,
    features: Sequence[str],
    *,
    alpha: float,
) -> tuple[pd.DataFrame, int]:
    training = panel.loc[pd.to_datetime(panel["label_end"]) < decision_date].dropna(
        subset=[*features, "target_excess_20d"]
    )
    current = panel.loc[panel["date"] == decision_date].dropna(subset=list(features))
    if current.empty or training.empty:
        raise ValueError("incomplete expanding-window fold")
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(training[list(features)], training["target_excess_20d"])
    scored = current[["date", "ticker", "sector", "target_excess_20d", "flow_source_date"]].copy()
    scored["prediction"] = model.predict(current[list(features)])
    return scored, int(training["date"].nunique())


def expanding_predictions(
    panel: pd.DataFrame,
    folds: Sequence[pd.Timestamp],
    *,
    alpha: float,
    top_k: int,
) -> tuple[pd.DataFrame, list[str]]:
    rows: list[pd.DataFrame] = []
    skipped: list[str] = []
    for decision_date in folds:
        current = panel.loc[panel["date"] == decision_date]
        training_weeks = int(
            panel.loc[pd.to_datetime(panel["label_end"]) < decision_date, "date"].nunique()
        )
        fold_scores: dict[str, pd.DataFrame] = {
            "B1_RRG": current[
                ["date", "ticker", "sector", "target_excess_20d", "flow_source_date", "shadow_score"]
            ].rename(columns={"shadow_score": "prediction"})
        }
        try:
            for model_name, features in MODEL_FEATURES.items():
                fold_scores[model_name], fitted_weeks = _fit_ridge(
                    panel, decision_date, features, alpha=alpha
                )
                if fitted_weeks != training_weeks:
                    raise ValueError("models do not share an identical training window")
        except (ValueError, FloatingPointError):
            skipped.append(decision_date.date().isoformat())
            continue
        if any(
            len(scored) != len(SECTOR_ETFS)
            or not np.isfinite(scored["prediction"]).all()
            or scored["prediction"].nunique() < 2
            for scored in fold_scores.values()
        ):
            skipped.append(decision_date.date().isoformat())
            continue
        for model_name in MODEL_ORDER:
            scored = fold_scores[model_name].copy()
            scored["model"] = model_name
            scored["rank"] = scored["prediction"].rank(
                ascending=False, method="first"
            ).astype(int)
            scored["is_top3"] = scored["rank"] <= top_k
            scored["is_bottom3"] = scored["rank"] > len(scored) - top_k
            scored["training_weeks"] = training_weeks
            rows.append(scored)
    if not rows:
        raise ValueError("all common OOS folds were incomplete")
    predictions = pd.concat(rows, ignore_index=True)
    common_dates = predictions.groupby("date")["model"].nunique()
    common_dates = common_dates.loc[common_dates == len(MODEL_ORDER)].index
    predictions = predictions.loc[predictions["date"].isin(common_dates)].copy()
    return predictions.sort_values(["date", "model", "rank"]).reset_index(drop=True), skipped


def _turnover(previous: set[str], current: set[str], top_k: int) -> float:
    return 1.0 if not previous else 1.0 - len(previous & current) / top_k


def weekly_metrics(predictions: pd.DataFrame, *, top_k: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_name in MODEL_ORDER:
        previous_top: set[str] = set()
        previous_bottom: set[str] = set()
        model_rows = predictions.loc[predictions["model"] == model_name]
        for decision_date, current in model_rows.groupby("date", sort=True):
            current = current.sort_values("rank")
            top = current.head(top_k)
            bottom = current.tail(top_k)
            top_set, bottom_set = set(top["ticker"]), set(bottom["ticker"])
            top_turnover = _turnover(previous_top, top_set, top_k)
            bottom_turnover = _turnover(previous_bottom, bottom_set, top_k)
            top_excess = float(top["target_excess_20d"].mean())
            spread = top_excess - float(bottom["target_excess_20d"].mean())
            rank_ic = float(
                current["prediction"].rank().corr(current["target_excess_20d"].rank())
            )
            row: dict[str, Any] = {
                "date": decision_date,
                "model": model_name,
                "rank_ic": rank_ic,
                "top3_excess": top_excess,
                "top_bottom_spread": spread,
                "top_turnover": top_turnover,
                "bottom_turnover": bottom_turnover,
            }
            for bps in (10, 20):
                row[f"top3_net_{bps}bps"] = top_excess - top_turnover * bps / 10_000
                row[f"spread_net_{bps}bps"] = spread - (
                    top_turnover + bottom_turnover
                ) * bps / 10_000
            rows.append(row)
            previous_top, previous_bottom = top_set, bottom_set
    result = pd.DataFrame(rows).sort_values(["date", "model"]).reset_index(drop=True)
    date_order = {day: index % 4 for index, day in enumerate(sorted(result["date"].unique()))}
    result["cohort"] = result["date"].map(date_order).astype(int)
    return result


METRIC_COLUMNS = (
    "rank_ic",
    "top3_excess",
    "top_bottom_spread",
    "top_turnover",
    "bottom_turnover",
    "top3_net_10bps",
    "spread_net_10bps",
    "top3_net_20bps",
    "spread_net_20bps",
)


def _means(frame: pd.DataFrame, columns: Sequence[str] = METRIC_COLUMNS) -> dict[str, float | None]:
    return {column: _finite(frame[column].mean()) for column in columns}


def _gross_sensitivity(
    predictions: pd.DataFrame,
    *,
    top_k: int,
) -> dict[str, float | None]:
    observations: list[dict[str, float]] = []
    for _, rows in predictions.groupby("date", sort=True):
        if len(rows) < 2 * top_k:
            continue
        rows = rows.sort_values("prediction", ascending=False)
        top, bottom = rows.head(top_k), rows.tail(top_k)
        observations.append(
            {
                "rank_ic": float(
                    rows["prediction"].rank().corr(rows["target_excess_20d"].rank())
                ),
                "top3_excess": float(top["target_excess_20d"].mean()),
                "top_bottom_spread": float(
                    top["target_excess_20d"].mean()
                    - bottom["target_excess_20d"].mean()
                ),
            }
        )
    return _means(pd.DataFrame(observations), ("rank_ic", "top3_excess", "top_bottom_spread"))


def _model_summary(
    model_metrics: pd.DataFrame,
    model_predictions: pd.DataFrame,
    *,
    top_k: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    means = _means(model_metrics)
    intervals = {
        column: list(
            _block_bootstrap_mean_ci(
                model_metrics[column],
                block_weeks=5,
                samples=bootstrap_samples,
            )
        )
        for column in (
            "rank_ic",
            "top3_excess",
            "top_bottom_spread",
            "top3_net_10bps",
            "spread_net_10bps",
            "top3_net_20bps",
            "spread_net_20bps",
        )
    }
    cohorts = {
        str(int(cohort)): {
            "weeks": int(len(rows)),
            **_means(
                rows,
                ("rank_ic", "top3_excess", "top_bottom_spread", "spread_net_20bps"),
            ),
        }
        for cohort, rows in model_metrics.groupby("cohort", sort=True)
    }
    best_week_row = model_metrics.loc[model_metrics["spread_net_10bps"].idxmax()]
    without_best_week = model_metrics.loc[model_metrics["date"] != best_week_row["date"]]
    original_spread = means["top_bottom_spread"]
    sector_scenarios: list[tuple[str, dict[str, float | None]]] = []
    for ticker in sorted(model_predictions["ticker"].unique()):
        scenario = _gross_sensitivity(
            model_predictions.loc[model_predictions["ticker"] != ticker], top_k=top_k
        )
        sector_scenarios.append((ticker, scenario))
    best_sector, without_best_sector = min(
        sector_scenarios,
        key=lambda item: (
            float("inf")
            if item[1]["top_bottom_spread"] is None
            else item[1]["top_bottom_spread"]
        ),
    )
    return {
        "weeks": int(model_metrics["date"].nunique()),
        "start": model_metrics["date"].min().date().isoformat(),
        "end": model_metrics["date"].max().date().isoformat(),
        "means": means,
        "rank_ic_positive_rate": _finite((model_metrics["rank_ic"] > 0).mean()),
        "block_5w_95ci": intervals,
        "phase_cohorts_4_approx": cohorts,
        "remove_best_week": {
            "date": best_week_row["date"].date().isoformat(),
            "selection_metric": "spread_net_10bps",
            "means_after_removal": _means(without_best_week),
        },
        "remove_best_sector": {
            "ticker": best_sector,
            "selection_metric": "largest reduction in gross top-bottom spread",
            "means_after_removal": without_best_sector,
            "spread_delta": (
                None
                if original_spread is None or without_best_sector["top_bottom_spread"] is None
                else without_best_sector["top_bottom_spread"] - original_spread
            ),
        },
    }


def _paired_comparisons(
    metrics: pd.DataFrame,
    *,
    bootstrap_samples: int,
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for candidate, baseline in (
        ("B2_PRICE_RIDGE", "B1_RRG"),
        ("C1_FLOW_RIDGE", "B1_RRG"),
        ("C2_PRICE_FLOW_RIDGE", "B2_PRICE_RIDGE"),
    ):
        left = metrics.loc[metrics["model"] == candidate].set_index("date")
        right = metrics.loc[metrics["model"] == baseline].set_index("date")
        common = left.index.intersection(right.index).sort_values()
        deltas: dict[str, Any] = {}
        for column in (
            "rank_ic",
            "top3_excess",
            "top_bottom_spread",
            "spread_net_10bps",
            "spread_net_20bps",
        ):
            values = left.loc[common, column] - right.loc[common, column]
            deltas[column] = {
                "mean": _finite(values.mean()),
                "block_5w_95ci": list(
                    _block_bootstrap_mean_ci(
                        values, block_weeks=5, samples=bootstrap_samples
                    )
                ),
            }
        comparisons[f"{candidate}_minus_{baseline}"] = deltas
    return comparisons


def summarize(
    panel: pd.DataFrame,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    early_event_summary: dict[str, Any],
    *,
    flow_source: Path,
    database: Path,
    panel_metadata: dict[str, int],
    min_train_weeks: int,
    alpha: float,
    top_k: int,
    bootstrap_samples: int,
    skipped_folds: Sequence[str],
) -> dict[str, Any]:
    models = {
        model: _model_summary(
            metrics.loc[metrics["model"] == model],
            predictions.loc[predictions["model"] == model],
            top_k=top_k,
            bootstrap_samples=bootstrap_samples,
        )
        for model in MODEL_ORDER
    }
    oos_weeks = int(metrics["date"].nunique())
    summary: dict[str, Any] = {
        "experiment": "sector-flow-expanding-window-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "pit_status": PIT_STATUS,
        "pit_note": (
            "Historical Bloomberg publication vintages are unavailable. A fixed one-week "
            "lag is conservative but cannot prove original point-in-time availability."
        ),
        "inputs": {
            "flow_csv": str(flow_source.resolve()),
            "price_database_read_only": str(database.resolve()),
        },
        "target": TARGET_DESCRIPTION,
        "target_horizon_trading_days": 20,
        "flow_rules": {
            "source_field": "FUND_FLOW treated as weekly FUND_FLOW_1_WK",
            "denominator": "prior-source-week FUND_TOTAL_ASSETS",
            "availability": "source W-FRI is usable only in exact W+1 W-FRI trading week; no carry",
            "features": list(FLOW_FEATURES),
            "acceleration": "1w flow/AUM minus the mean of the prior four 1w ratios",
        },
        "models": {
            "B1_RRG": "existing RRG shadow_score",
            "B2_PRICE_RIDGE": f"StandardScaler + Ridge(alpha={alpha:g}) on price/RRG features",
            "C1_FLOW_RIDGE": f"StandardScaler + Ridge(alpha={alpha:g}) on three flow ranks",
            "C2_PRICE_FLOW_RIDGE": f"StandardScaler + Ridge(alpha={alpha:g}) on price + flow",
        },
        "fold_contract": {
            "frequency": "weekly W-FRI last trading day",
            "minimum_training_weeks": min_train_weeks,
            "training_filter": "label_end < decision_date",
            "common_complete_panel_first": True,
            "all_models_same_folds": True,
            "top_k": top_k,
        },
        "sample": {
            **panel_metadata,
            "panel_start": panel["date"].min().date().isoformat(),
            "panel_end": panel["date"].max().date().isoformat(),
            "common_oos_weeks": oos_weeks,
            "common_oos_rows_per_model": int(len(predictions) / len(MODEL_ORDER)),
            "approx_independent_20d_blocks": oos_weeks / 4,
            "skipped_common_folds": list(skipped_folds),
        },
        "bootstrap": {"method": "moving block", "block_weeks": 5, "samples": bootstrap_samples},
        "results": models,
        "early_rotation_event": early_event_summary,
        "paired_comparisons": _paired_comparisons(
            metrics, bootstrap_samples=bootstrap_samples
        ),
        "verdict": "RESEARCH_SHADOW_ONLY" if oos_weeks < 104 else "REVIEW_STATISTICAL_GATES",
    }
    return _json_ready(summary)


def _format_percent(value: Any) -> str:
    number = _finite(value)
    return "—" if number is None else f"{number * 100:+.2f}%"


def _format_number(value: Any) -> str:
    number = _finite(value)
    return "—" if number is None else f"{number:+.3f}"


def markdown_report(summary: dict[str, Any]) -> str:
    sample = summary["sample"]
    lines = [
        "# Sector Flow Research Comparison",
        "",
        f"- 판정: **{summary['verdict']}**",
        f"- PIT 상태: **{summary['pit_status']}** — 실제 발표 vintage가 없어 1주 lag만 적용",
        f"- 공통 OOS: {sample['common_oos_weeks']}주 "
        f"(독립 20일 구간 약 {sample['approx_independent_20d_blocks']:.1f}개)",
        f"- 목표: {summary['target']}",
        "- 공통 fold: 모든 모델·11개 섹터·모든 특징·목표가 완전한 주만 사용",
        "- 신뢰구간: 5주 moving-block bootstrap",
        "",
        "## 모델 비교",
        "",
        "| 모델 | IC | IC>0 | Top3 초과 | Top-Bottom | Spread net 10bp | Spread net 20bp |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODEL_ORDER:
        result = summary["results"][model]
        means = result["means"]
        lines.append(
            f"| {model} | {_format_number(means['rank_ic'])} | "
            f"{_format_percent(result['rank_ic_positive_rate'])} | "
            f"{_format_percent(means['top3_excess'])} | "
            f"{_format_percent(means['top_bottom_spread'])} | "
            f"{_format_percent(means['spread_net_10bps'])} | "
            f"{_format_percent(means['spread_net_20bps'])} |"
        )
    event = summary["early_rotation_event"]
    lines.extend(
        [
            "",
            "## 초기 로테이션 후보 이벤트",
            "",
            "자금흐름 가속 상위 3개·4주 누적 흐름 양호·20일 가격 선도 Top3 진입 전·"
            "5일 RRG 모멘텀 개선을 모두 만족한 섹터만 후보로 표시합니다.",
            f"- 이벤트: {event['event_rows']}건 / {event['event_weeks']}주",
            f"- 이벤트 평균 20일 초과수익: {_format_percent(event['mean_event_excess_20d'])} "
            f"(95% CI {_format_percent(event['block_5w_95ci']['event_excess_20d'][0])} ~ "
            f"{_format_percent(event['block_5w_95ci']['event_excess_20d'][1])})",
            f"- 비이벤트 대비: {_format_percent(event['mean_event_minus_non_event_20d'])}",
            "- **탐색 연구용**: 표본이 짧고 PIT가 완전하지 않아 선행 신호로 판정하지 않습니다.",
        ]
    )
    lines.extend(["", "## 핵심 증분 비교", ""])
    for comparison, metrics in summary["paired_comparisons"].items():
        lines.append(
            f"- **{comparison}**: ΔIC {_format_number(metrics['rank_ic']['mean'])}, "
            f"Δ net spread(20bp) {_format_percent(metrics['spread_net_20bps']['mean'])}"
        )
    lines.extend(["", "## 4주 위상별 결과", "", "공휴일 때문에 완전한 비중첩을 보장하지 않는 참고치입니다.", ""])
    for model in MODEL_ORDER:
        cohorts = summary["results"][model]["phase_cohorts_4_approx"]
        rendered = ", ".join(
            f"{offset}: {values['weeks']}주/{_format_percent(values['top_bottom_spread'])}"
            for offset, values in cohorts.items()
        )
        lines.append(f"- **{model}** — {rendered}")
    lines.extend(["", "## 민감도", ""])
    for model in MODEL_ORDER:
        result = summary["results"][model]
        week = result["remove_best_week"]
        sector = result["remove_best_sector"]
        lines.append(
            f"- **{model}**: 최고 주 {week['date']} 제거 spread "
            f"{_format_percent(week['means_after_removal']['top_bottom_spread'])}; "
            f"기여 최대 섹터 {sector['ticker']} 제거 spread "
            f"{_format_percent(sector['means_after_removal']['top_bottom_spread'])}"
        )
    lines.extend(
        [
            "",
            "## 해석 제한",
            "",
            "- 20거래일 목표를 매주 관측하므로 인접 결과가 크게 겹칩니다.",
            "- Bloomberg 원자료의 실제 공개시각·수정 vintage가 없어 완전한 PIT 검증이 아닙니다.",
            "- 104개 OOS 주 이전에는 실운용 적합 판정으로 승격하지 않습니다.",
            "- 최고 섹터 제거 민감도는 모델을 재학습하지 않고 해당 섹터만 제외해 재순위한 진단입니다.",
            "",
        ]
    )
    return "\n".join(lines)


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    flow_source, database, output_dir = args.flow_csv, args.database, args.output_dir
    flow = load_flow_history(flow_source)
    prices = _load_price_matrix_read_only(database)
    flow_features = build_flow_features(flow, prices.index)
    panel, panel_metadata = build_common_panel(prices, flow_features)
    folds = _common_folds(panel, args.min_train_weeks)
    early_events = build_early_rotation_events(panel)
    early_event_summary = summarize_early_rotation_events(
        early_events, evaluation_dates=folds, bootstrap_samples=args.bootstrap_samples
    )
    predictions, skipped = expanding_predictions(
        panel, folds, alpha=args.alpha, top_k=args.top_k
    )
    metrics = weekly_metrics(predictions, top_k=args.top_k)
    summary = summarize(
        panel,
        predictions,
        metrics,
        early_event_summary,
        flow_source=flow_source,
        database=database,
        panel_metadata=panel_metadata,
        min_train_weeks=args.min_train_weeks,
        alpha=args.alpha,
        top_k=args.top_k,
        bootstrap_samples=args.bootstrap_samples,
        skipped_folds=skipped,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    metrics.to_csv(output_dir / "weekly_metrics.csv", index=False)
    early_events.to_csv(output_dir / "early_rotation_events.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (output_dir / "REPORT.md").write_text(markdown_report(summary), encoding="utf-8")
    return summary


def self_test() -> None:
    flow_tickers = ("SPY", "A", "B", "C", "D")
    source_dates = pd.date_range("2025-01-03", periods=8, freq="W-FRI")
    flow = pd.DataFrame(
        [
            {
                "ticker": ticker,
                "date": day,
                "fund_flow_1w": (index + 1) * (week + 1),
                "aum": 1_000 + index * 100 + week * 10,
            }
            for week, day in enumerate(source_dates)
            for index, ticker in enumerate(flow_tickers)
        ]
    )
    features = build_flow_features(
        flow, pd.bdate_range("2025-01-01", "2025-03-31"), universe=flow_tickers
    )
    assert not features.empty
    assert "SPY" not in set(features["ticker"])
    assert (
        features["usable_week"].astype(int) - features["source_week"].astype(int)
    ).eq(1).all()
    centered = features.groupby("date")[list(FLOW_FEATURES)].mean().abs()
    assert (centered < 1e-12).all().all()

    early_panel = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2025-02-07"),
                "ticker": ticker,
                "target_excess_20d": value / 100,
                "flow_rank_accel": accel,
                "flow_rank_4w": flow_4w,
                "relative_return_20": price,
                "rrg_momentum_delta_5": momentum,
            }
            for ticker, value, accel, flow_4w, price, momentum in (
                ("A", 2, 0.30, 0.10, 0.01, 0.20),
                ("B", -1, 0.10, 0.10, 0.02, 0.20),
                ("C", 0, 0.00, -0.10, 0.03, -0.20),
            )
        ]
    )
    early_events = build_early_rotation_events(early_panel)
    assert early_events.loc[early_events["ticker"] == "A", "early_rotation_in"].item()
    assert not early_events.loc[early_events["ticker"] == "B", "early_rotation_in"].item()

    tickers = ("A", "B", "C", "D")
    dates = pd.date_range("2024-01-05", periods=32, freq="W-FRI")
    panel = pd.DataFrame(
        [
            {
                "date": day,
                "label_end": day + pd.Timedelta(days=28),
                "ticker": ticker,
                "sector": ticker,
                "x": index + week / 10,
                "target_excess_20d": (index - 1.5) / 100 + week / 100_000,
                "flow_source_date": day - pd.Timedelta(days=7),
            }
            for week, day in enumerate(dates)
            for index, ticker in enumerate(tickers)
        ]
    )
    scored, training_weeks = _fit_ridge(
        panel, dates[-1], ("x",), alpha=10.0
    )
    assert training_weeks >= 26
    assert len(scored) == len(tickers) and np.isfinite(scored["prediction"]).all()
    print("self-test: OK")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Research-only Bloomberg sector flow expanding-window comparison"
    )
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--flow-csv", type=Path)
    parser.add_argument("--database", type=Path, default=Path("regime_router_publish.db"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--min-train-weeks", type=int, default=26)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.self_test:
        self_test()
        return
    if args.flow_csv is None or args.output_dir is None:
        raise SystemExit("--flow-csv and --output-dir are required unless --self-test is used")
    if args.min_train_weeks < 26:
        raise SystemExit("--min-train-weeks must be at least 26")
    if args.alpha <= 0 or not 1 <= args.top_k <= len(SECTOR_ETFS) // 2:
        raise SystemExit("--alpha must be positive and --top-k must be between 1 and 5")
    if args.bootstrap_samples < 100:
        raise SystemExit("--bootstrap-samples must be at least 100")
    summary = run_experiment(args)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "pit_status": summary["pit_status"],
                "oos_weeks": summary["sample"]["common_oos_weeks"],
                "verdict": summary["verdict"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
