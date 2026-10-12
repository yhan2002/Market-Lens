"""Research-only sector-conditioned macro SHALLOW_BOOST comparison."""

from __future__ import annotations

import argparse
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.preprocessing import StandardScaler

from research_drawdown_onset import causal_trailing_percentile
from sector_turning_pooled_shadow import (
    DIRECTIONS,
    PRICE_FEATURES,
    _early_event_metrics,
    _direction_frame,
    _wilson_interval,
    build_pooled_daily_frame,
    effect_code,
    paired_date_block_pre_pivot_recall_delta_ci,
    purged_training_mask,
)
from sector_turning_recall_model_compare import estimator, model_metrics
from sector_turning_tradable_label_shadow import qualify_labels
from turning_macro_shadow_experiment import (
    ensure_output_boundary,
    load_macro_history,
    merge_macro_next_session,
)


BASELINE = "PRICE_SHALLOW"
DRIVERS = (
    "curve_2s10s",
    "real_rate_10y",
    "breakeven_10y",
    "wti_spot",
    "brent_wti_spread",
    "yen_strength",
    "vix",
    "move",
)
RAW_MACRO_FEATURES = tuple(
    column
    for driver in DRIVERS
    for column in (f"{driver}_shock_5", f"{driver}_acceleration_5")
)
SECTOR_MACRO_FEATURES = tuple(
    column
    for driver in DRIVERS
    for column in (f"{driver}_beta_126", f"{driver}_impact_5")
)
NETWORK_FEATURES = (
    "peer_relative_return_5",
    "peer_relative_return_20",
    "cross_section_rank_5",
    "cross_section_rank_20",
    "cross_section_dispersion_20",
    "peer_corr_change_63_252",
)
RATES_FEATURES = tuple(
    column
    for driver in ("curve_2s10s", "real_rate_10y", "breakeven_10y")
    for column in (f"{driver}_shock_5", f"{driver}_acceleration_5", f"{driver}_beta_126", f"{driver}_impact_5")
)
OIL_FEATURES = tuple(
    column
    for driver in ("wti_spot", "brent_wti_spread")
    for column in (f"{driver}_shock_5", f"{driver}_acceleration_5", f"{driver}_beta_126", f"{driver}_impact_5")
)
RISK_CARRY_FEATURES = tuple(
    column
    for driver in ("yen_strength", "vix", "move")
    for column in (f"{driver}_shock_5", f"{driver}_acceleration_5", f"{driver}_beta_126", f"{driver}_impact_5")
)
FEATURE_SETS = {
    BASELINE: PRICE_FEATURES,
    "RAW_MACRO_SHALLOW": (*PRICE_FEATURES, *RAW_MACRO_FEATURES),
    "SECTOR_MACRO_SHALLOW": (
        *PRICE_FEATURES,
        *RAW_MACRO_FEATURES,
        *SECTOR_MACRO_FEATURES,
    ),
    "NETWORK_SHALLOW": (*PRICE_FEATURES, *NETWORK_FEATURES),
    "PEER_RETURN_SHALLOW": (*PRICE_FEATURES, *NETWORK_FEATURES[:2]),
    "RANK_DISPERSION_SHALLOW": (*PRICE_FEATURES, *NETWORK_FEATURES[2:5]),
    "CORRELATION_SHIFT_SHALLOW": (*PRICE_FEATURES, NETWORK_FEATURES[-1]),
    "RATES_SECTOR_SHALLOW": (*PRICE_FEATURES, *RATES_FEATURES),
    "OIL_SECTOR_SHALLOW": (*PRICE_FEATURES, *OIL_FEATURES),
    "RISK_CARRY_SECTOR_SHALLOW": (*PRICE_FEATURES, *RISK_CARRY_FEATURES),
    "SECTOR_MACRO_NETWORK_SHALLOW": (
        *PRICE_FEATURES,
        *RAW_MACRO_FEATURES,
        *SECTOR_MACRO_FEATURES,
        *NETWORK_FEATURES,
    ),
}
MODELS = tuple(FEATURE_SETS)
ALL_FEATURES = FEATURE_SETS["SECTOR_MACRO_NETWORK_SHALLOW"]
FACTOR_COLUMNS = tuple(
    column
    for driver in DRIVERS
    for column in (
        f"{driver}_shock_1",
        f"{driver}_shock_5",
        f"{driver}_acceleration_5",
    )
)


def load_market_factor_levels(database: Path) -> pd.DataFrame:
    uri = database.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        connection.execute("PRAGMA query_only=ON")
        rows = pd.read_sql_query(
            """
            SELECT date, ticker, COALESCE(adj_close, close) AS value
            FROM daily_prices
            WHERE ticker IN ('JPY=X', '^VIX', '^MOVE')
            ORDER BY date, ticker
            """,
            connection,
            parse_dates=["date"],
        )
    if rows.empty or rows["value"].isna().any() or rows.duplicated(["date", "ticker"]).any():
        raise ValueError("JPY/VIX/MOVE price history is incomplete or duplicated")
    wide = rows.pivot(index="date", columns="ticker", values="value")
    missing = sorted({"JPY=X", "^VIX", "^MOVE"} - set(wide.columns))
    if missing:
        raise ValueError("missing market factor series: " + ", ".join(missing))
    return wide.rename(columns={"JPY=X": "usdjpy", "^VIX": "vix", "^MOVE": "move"})


def engineer_factor_features(levels: pd.DataFrame) -> pd.DataFrame:
    required = set(DRIVERS)
    if missing := sorted(required - set(levels.columns)):
        raise ValueError("missing driver levels: " + ", ".join(missing))
    clean = levels.loc[:, DRIVERS].replace([np.inf, -np.inf], np.nan).dropna().sort_index()
    output = pd.DataFrame(index=clean.index)
    for driver in DRIVERS:
        value = clean[driver].astype(float)
        output[f"{driver}_shock_1"] = value.diff()
        output[f"{driver}_shock_5"] = value.diff(5)
        output[f"{driver}_acceleration_5"] = value.diff(5).diff(5)
    return output


def build_factor_features(macro_csv: Path, database: Path) -> pd.DataFrame:
    history = load_macro_history(macro_csv)
    macro = history.pivot(index="date", columns="series", values="value").sort_index()
    market = load_market_factor_levels(database)
    levels = pd.DataFrame(
        {
            "curve_2s10s": macro["UST_10Y"] - macro["UST_2Y"],
            "real_rate_10y": macro["UST_REAL_10Y"],
            "breakeven_10y": macro["BREAKEVEN_10Y"],
            "wti_spot": macro["WTI_SPOT"],
            "brent_wti_spread": macro["BRENT_SPOT"] - macro["WTI_SPOT"],
        }
    ).join(market, how="inner")
    levels["yen_strength"] = -np.log(levels.pop("usdjpy"))
    levels["vix"] = np.log(levels["vix"])
    levels["move"] = np.log(levels["move"])
    return engineer_factor_features(levels)


def attach_sector_features(pooled: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    dates = pd.DatetimeIndex(pooled["date"].drop_duplicates().sort_values())
    aligned = merge_macro_next_session(
        dates, factors, FACTOR_COLUMNS, asof_column="factor_asof_date"
    ).reset_index()
    frame = pooled.merge(aligned, on="date", how="left", validate="many_to_one")
    frame = frame.sort_values(["sector", "_position"]).copy()
    frame["relative_return_1"] = frame.groupby("sector", sort=False)["relative_price"].pct_change(
        fill_method=None
    )

    for source, target in (
        ("relative_return_1", "peer_relative_return_1"),
        ("relative_return_5", "peer_relative_return_5"),
        ("relative_return_20", "peer_relative_return_20"),
    ):
        totals = frame.groupby("date")[source].transform("sum")
        counts = frame.groupby("date")[source].transform("count")
        frame[target] = (totals - frame[source]) / (counts - 1)
    for horizon in (5, 20):
        source = f"relative_return_{horizon}"
        frame[f"cross_section_rank_{horizon}"] = frame.groupby("date")[source].rank(pct=True)
    frame["cross_section_dispersion_20"] = frame.groupby("date")["relative_return_20"].transform("std")

    parts = []
    for _, sector in frame.groupby("sector", sort=False):
        sector = sector.sort_values("_position").copy()
        corr_63 = sector["relative_return_1"].rolling(63, min_periods=42).corr(
            sector["peer_relative_return_1"]
        ).shift(1)
        corr_252 = sector["relative_return_1"].rolling(252, min_periods=126).corr(
            sector["peer_relative_return_1"]
        ).shift(1)
        sector["peer_corr_change_63_252"] = corr_63 - corr_252
        for driver in DRIVERS:
            shock_1 = sector[f"{driver}_shock_1"]
            variance = shock_1.rolling(126, min_periods=84).var().shift(1)
            covariance = sector["relative_return_1"].rolling(126, min_periods=84).cov(
                shock_1
            ).shift(1)
            beta = covariance.div(variance.where(variance.abs() > 1e-12))
            sector[f"{driver}_beta_126"] = beta
            sector[f"{driver}_impact_5"] = beta * sector[f"{driver}_shock_5"]
        parts.append(sector)
    return pd.concat(parts, ignore_index=True).sort_values(["date", "sector"])


def design_matrix(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    scaler: StandardScaler,
    levels: tuple[str, ...],
) -> pd.DataFrame:
    continuous = pd.DataFrame(
        scaler.transform(frame.loc[:, columns]), index=frame.index, columns=columns
    )
    effects, _ = effect_code(frame["sector"], levels)
    return continuous.join(effects)


def walk_forward(
    pooled: pd.DataFrame,
    direction: str,
    *,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    min_train_dates: int,
    refit_every: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = _direction_frame(pooled, direction, target_mode="PRE_PIVOT")
    frame = frame.loc[frame["date"].between(start_date, end_date)].copy()
    levels = tuple(sorted(frame["sector"].astype(str).unique()))
    dates = pd.DatetimeIndex(frame["date"].drop_duplicates().sort_values())
    complete = frame.loc[:, ALL_FEATURES].notna().all(axis=1)
    rows: list[pd.DataFrame] = []
    fitted = {name: 0 for name in MODELS}
    for block_start in range(0, len(dates), refit_every):
        block_dates = dates[block_start : block_start + refit_every]
        prediction_date = pd.Timestamp(block_dates[0])
        train = frame.loc[purged_training_mask(frame, prediction_date) & complete]
        test = frame.loc[
            frame["at_risk"].astype(bool) & complete & frame["date"].isin(block_dates)
        ]
        if train["date"].nunique() < min_train_dates or train["target"].nunique() < 2 or test.empty:
            continue
        for model_name in MODELS:
            columns = FEATURE_SETS[model_name]
            scaler = StandardScaler().fit(train.loc[:, columns])
            train_design = design_matrix(train, columns, scaler, levels)
            test_design = design_matrix(test, columns, scaler, levels)
            model = estimator("SHALLOW_BOOST").fit(train_design, train["target"].astype(int))
            scored = test[
                [
                    "date", "_position", "sector", "sector_name", "target",
                    "target_event_pivot_position", "target_event_confirmation_position",
                    "target_event_confirmation_date",
                ]
            ].copy()
            scored.insert(0, "direction", direction)
            scored.insert(1, "model", model_name)
            scored["refit_date"] = prediction_date
            scored["training_start_date"] = train["date"].min()
            scored["training_end_date"] = train["date"].max()
            scored["latest_training_label_end_date"] = train["label_end_date"].max()
            scored["raw_score"] = model.predict_proba(test_design)[:, 1]
            rows.append(scored)
            fitted[model_name] += 1
    predictions = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not predictions.empty:
        predictions = predictions.sort_values(["model", "sector", "date"])
        predictions["score"] = predictions.groupby(["model", "sector"], sort=False)["raw_score"].transform(
            lambda values: causal_trailing_percentile(values, window=504, min_history=60)
        )
        predictions = predictions.dropna(subset=["score"]).reset_index(drop=True)
    return predictions, {
        "window_dates": int(len(dates)),
        "complete_rows": int(complete.sum()),
        "fitted_blocks": fitted,
    }


def recall_uplifts(predictions: pd.DataFrame, cutoff: float = 0.60) -> dict[str, Any]:
    baseline = predictions.loc[predictions["model"].eq(BASELINE)]
    calendar = pd.DatetimeIndex(predictions["date"].drop_duplicates().sort_values())
    return {
        candidate: paired_date_block_pre_pivot_recall_delta_ci(
            predictions.loc[predictions["model"].eq(candidate)],
            baseline,
            samples=500,
            block_dates=40,
            calendar_dates=calendar,
            horizon=10,
            cutoff=cutoff,
        )
        for candidate in MODELS[1:]
    }


def nearest_alert_rate_cutoff(scores: pd.Series, target_rate: float) -> float:
    values = np.sort(pd.to_numeric(scores, errors="coerce").dropna().unique())
    if not len(values):
        raise ValueError("cannot match alert rate without scores")
    return float(min(values, key=lambda value: (abs(float((scores >= value).mean()) - target_rate), -value)))


def matched_alert_rate_metrics(
    predictions: pd.DataFrame,
    direction: str,
    baseline_cutoff: float = 0.60,
) -> dict[str, Any]:
    baseline = predictions.loc[predictions["model"].eq(BASELINE)].copy()
    target_rate = float((baseline["score"] >= baseline_cutoff).mean())
    calendar = pd.DatetimeIndex(predictions["date"].drop_duplicates().sort_values())
    result: dict[str, Any] = {}
    for model_name in MODELS:
        rows = predictions.loc[predictions["model"].eq(model_name)].copy()
        cutoff = (
            baseline_cutoff
            if model_name == BASELINE
            else nearest_alert_rate_cutoff(rows["score"], target_rate)
        )
        event = _early_event_metrics(rows, pd.DataFrame(), direction, horizon=10, cutoff=cutoff)
        low, high = _wilson_interval(event["pre_pivot_hits"], event["eligible_confirmations"])
        model_result: dict[str, Any] = {
            **event,
            "matched_cutoff": cutoff,
            "target_alert_day_rate": target_rate,
            "pre_pivot_recall_ci_low": low,
            "pre_pivot_recall_ci_high": high,
        }
        if model_name != BASELINE:
            shifted = rows.copy()
            shifted["score"] += baseline_cutoff - cutoff
            model_result["recall_uplift_vs_baseline"] = paired_date_block_pre_pivot_recall_delta_ci(
                shifted,
                baseline,
                samples=500,
                block_dates=40,
                calendar_dates=calendar,
                horizon=10,
                cutoff=baseline_cutoff,
            )
        result[model_name] = model_result
    return result


def markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Sector-Conditioned Macro SHALLOW_BOOST",
        "",
        f"- 기간: {summary['spec']['start']} ~ {summary['spec']['end']}",
        "- 목표: 확인된 고점·저점 종료를 확정 전에 포착",
        "- 모든 모델은 동일 행·동일 Shallow Boost·동일 purged walk-forward 사용",
        "- PIT: 연구용 CURRENT_REVISED_HISTORY_ONLY",
        "",
        "| Direction | Model | AUC | Recall@40% | 95% low | Lead | Alert days |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for direction in DIRECTIONS:
        for model_name in MODELS:
            metric = summary["directions"][direction]["models"][model_name]
            event = metric["budgets"]["40pct"]
            lines.append(
                f"| {direction} | {model_name} | {metric['roc_auc']:.3f} | "
                f"{event['pre_pivot_recall']:.1%} | {event['pre_pivot_recall_ci_low']:.1%} | "
                f"{event['median_lead_to_pivot_sessions'] or 0:.1f} | {event['alert_day_rate']:.1%} |"
            )
    for direction in DIRECTIONS:
        lines.append("")
        for candidate, uplift in summary["directions"][direction]["recall_uplift_vs_baseline"].items():
            interval = (
                "표본 부족"
                if not np.isfinite(uplift["early_capture_ci_low"])
                else f"[{uplift['early_capture_ci_low']:+.1%}, {uplift['early_capture_ci_high']:+.1%}]"
            )
            lines.append(
                f"- {direction} · {candidate} vs baseline: "
                f"{uplift['delta_pre_pivot_recall']:+.1%} {interval}"
            )
        lines.extend([
            "",
            f"### {direction} · 가격모델과 동일 경보일 비율",
            "",
            "| Model | Cutoff | Recall | 95% low | Lead | Alert days | Uplift |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for model_name in MODELS:
            event = summary["directions"][direction]["matched_alert_rate"][model_name]
            uplift = event.get("recall_uplift_vs_baseline", {}).get("delta_pre_pivot_recall", 0.0)
            lines.append(
                f"| {model_name} | {event['matched_cutoff']:.3f} | "
                f"{event['pre_pivot_recall']:.1%} | {event['pre_pivot_recall_ci_low']:.1%} | "
                f"{event['median_lead_to_pivot_sessions'] or 0:.1f} | "
                f"{event['alert_day_rate']:.1%} | {uplift:+.1%} |"
            )
    return "\n".join(lines)


def self_test() -> None:
    dates = pd.bdate_range("2024-01-02", periods=170)
    t = np.arange(len(dates), dtype=float)
    levels = pd.DataFrame(
        {driver: np.sin(t / (7 + index)) + t * (index + 1) / 500
         for index, driver in enumerate(DRIVERS)},
        index=dates,
    )
    factors = engineer_factor_features(levels)
    parts = []
    for index, sector in enumerate(("A", "B")):
        relative = pd.Series(np.exp(np.cumsum(0.001 * np.sin(t / (5 + index)))), index=dates)
        part = pd.DataFrame({
            "date": dates,
            "sector": sector,
            "sector_name": sector,
            "_position": np.arange(len(dates)),
            "relative_price": relative.to_numpy(),
            "relative_return_5": relative.pct_change(5).to_numpy(),
            "relative_return_20": relative.pct_change(20).to_numpy(),
        })
        parts.append(part)
    pooled = pd.concat(parts, ignore_index=True)
    original = attach_sector_features(pooled, factors)
    changed = levels.copy()
    changed.loc[dates[-10]:, "real_rate_10y"] += 100
    revised = attach_sector_features(pooled, engineer_factor_features(changed))
    cutoff = dates[-11]
    columns = ["date", "sector", *RAW_MACRO_FEATURES, *SECTOR_MACRO_FEATURES, *NETWORK_FEATURES]
    assert_frame_equal(
        original.loc[original["date"].le(cutoff), columns].reset_index(drop=True),
        revised.loc[revised["date"].le(cutoff), columns].reset_index(drop=True),
    )
    assert factors.loc[dates[20], "yen_strength_shock_5"] == levels["yen_strength"].diff(5).loc[dates[20]]
    scores = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
    matched = nearest_alert_rate_cutoff(scores, 0.4)
    assert matched == 0.4 and float((scores >= matched).mean()) == 0.4


def run(args: argparse.Namespace) -> dict[str, Any]:
    package = args.gics_package_dir.resolve()
    macro_csv = args.macro_csv.resolve()
    output = args.output_dir.resolve()
    ensure_output_boundary(output, package, macro_csv.parent)
    pooled, input_audit, _ = build_pooled_daily_frame(package, None, horizon=10)
    factors = build_factor_features(macro_csv, args.database)
    pooled = attach_sector_features(pooled, factors)
    start_date, end_date = pd.Timestamp(args.start), pd.Timestamp(args.end)
    directions: dict[str, Any] = {}
    prediction_parts = []
    for direction in DIRECTIONS:
        qualified, label_audit = qualify_labels(pooled, direction)
        predictions, walk = walk_forward(
            qualified,
            direction,
            start_date=start_date,
            end_date=end_date,
            min_train_dates=args.min_train_dates,
            refit_every=args.refit_every,
        )
        if predictions.empty:
            raise RuntimeError(f"no predictions for {direction}")
        prediction_parts.append(predictions)
        directions[direction] = {
            "label_audit": label_audit,
            "walk_forward": walk,
            "models": {
                model: model_metrics(predictions.loc[predictions["model"].eq(model)], direction)
                for model in MODELS
            },
            "recall_uplift_vs_baseline": recall_uplifts(predictions),
            "matched_alert_rate": matched_alert_rate_metrics(predictions, direction),
        }
    combined = pd.concat(prediction_parts, ignore_index=True)
    summary = {
        "verdict": "RESEARCH_SHADOW_ONLY",
        "pit_status": "PIT_PARTIAL/CURRENT_REVISED_HISTORY_ONLY",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "spec": {
            "start": str(start_date.date()),
            "end": str(end_date.date()),
            "models": list(MODELS),
            "drivers": list(DRIVERS),
            "feature_sets": {name: list(columns) for name, columns in FEATURE_SETS.items()},
            "min_train_dates": args.min_train_dates,
            "refit_every": args.refit_every,
            "factor_timing": "source date strictly before market date; 7 calendar-day tolerance",
            "sector_sensitivity": "126-session rolling beta shifted one session",
        },
        "input_audit": input_audit,
        "factor_coverage": {
            "first_date": str(factors.dropna().index.min().date()),
            "last_date": str(factors.dropna().index.max().date()),
            "complete_dates": int(factors.dropna().shape[0]),
        },
        "directions": directions,
    }
    output.mkdir(parents=True, exist_ok=False)
    combined.to_csv(output / "predictions.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "REPORT.md").write_text(markdown_report(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gics-package-dir", type=Path)
    parser.add_argument("--macro-csv", type=Path)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--start", default="2023-09-01")
    parser.add_argument("--end", default="2026-08-31")
    parser.add_argument("--min-train-dates", type=int, default=252)
    parser.add_argument("--refit-every", type=int, default=21)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("self-test passed")
        return
    if not all((args.gics_package_dir, args.macro_csv, args.database, args.output_dir)):
        parser.error("--gics-package-dir, --macro-csv, --database, and --output-dir are required")
    summary = run(args)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "verdict": summary["verdict"]}))


if __name__ == "__main__":
    main()
