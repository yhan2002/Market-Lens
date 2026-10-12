"""Research-only forecast of future Bloomberg sector fund-flow rotation."""

from __future__ import annotations

import argparse
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from regime_router_app import HMM_MODEL_VERSION
from sector_flow_experiment import (
    FLOW_UNIVERSE,
    _block_bootstrap_mean_ci,
    _finite,
    _json_ready,
    load_flow_history,
)
from sector_rotation import SECTOR_ETFS, build_rrg_signals, load_regime_probabilities
from sector_rotation_forecast import build_sector_feature_frame
from sector_shadow_store import _load_price_matrix_read_only


SECTORS = tuple(SECTOR_ETFS)
PRICE_FEATURES = (
    "rrg_ratio",
    "rrg_momentum",
    "rrg_ratio_delta_5",
    "rrg_momentum_delta_5",
    "relative_return_20",
)
MACRO_PRICE_TICKERS = ("NQ=F", "HYG", "IEF", "UUP", "^VIX", "^VIX3M", "^MOVE")
MACRO_FEATURES = (
    "nq_return_20",
    "ief_return_20",
    "hyg_minus_ief_20",
    "uup_return_20",
    "vix_level",
    "vix_change_5",
    "vix_curve",
    "move_level",
    "move_change_5",
    "hy_oas_level",
    "hy_oas_change_4w",
    "cash_3m_level",
    "cash_3m_change_4w",
    "p_sideways",
    "p_bear",
)
MODEL_ORDER = (
    "FLOW_PERSISTENCE",
    "PRICE_RRG_RIDGE",
    "MACRO_RIDGE",
    "PRICE_MACRO_RIDGE",
)


def build_future_flow_panel(
    flow: pd.DataFrame, trading_dates: pd.DatetimeIndex
) -> pd.DataFrame:
    """Use flow known at decision time to predict the following four source weeks."""
    if set(flow["ticker"]) != set(FLOW_UNIVERSE):
        raise ValueError("flow universe mismatch")
    frame = flow.sort_values(["ticker", "date"]).copy()
    grouped = frame.groupby("ticker", group_keys=False)
    frame["flow_to_lagged_aum"] = frame["fund_flow_1w"] / grouped["aum"].shift(1)
    future = [
        grouped["flow_to_lagged_aum"].shift(-lag).rename(f"future_{lag}")
        for lag in range(1, 5)
    ]
    frame = pd.concat([frame, *future], axis=1)
    frame["future_flow_4w"] = frame[[f"future_{lag}" for lag in range(1, 5)]].sum(
        axis=1, min_count=4
    )
    frame["source_week"] = frame["date"].dt.to_period("W-FRI")
    frame["decision_week"] = frame["source_week"] + 1
    frame["label_available_week"] = frame["source_week"] + 5

    calendar = pd.DataFrame({"date": pd.DatetimeIndex(trading_dates).normalize()})
    calendar["week"] = calendar["date"].dt.to_period("W-FRI")
    week_end = calendar.groupby("week")["date"].max()
    frame["decision_date"] = frame["decision_week"].map(week_end)
    frame["label_available_date"] = frame["label_available_week"].map(week_end)

    sectors = frame.loc[frame["ticker"].isin(SECTORS)].copy()
    sectors["current_flow_rank"] = sectors.groupby("date")["flow_to_lagged_aum"].rank(
        pct=True, method="average"
    )
    sectors["future_flow_rank"] = sectors.groupby("date")["future_flow_4w"].rank(
        pct=True, method="average"
    )
    sectors["target_future_flow_rank"] = sectors["future_flow_rank"] - sectors.groupby(
        "date"
    )["future_flow_rank"].transform("mean")
    # pct ranks are in (0, 1]; top three begin at 9/11.
    sectors["current_top3"] = sectors["current_flow_rank"] >= 9 / len(SECTORS)
    sectors["future_top3"] = sectors["future_flow_rank"] >= 9 / len(SECTORS)
    required = [
        "decision_date",
        "label_available_date",
        "current_flow_rank",
        "target_future_flow_rank",
    ]
    return sectors.dropna(subset=required)[
        [
            "decision_date",
            "label_available_date",
            "ticker",
            "current_flow_rank",
            "target_future_flow_rank",
            "future_top3",
            "current_top3",
        ]
    ].rename(columns={"decision_date": "date"})


def _read_macro_prices(database: Path) -> pd.DataFrame:
    placeholders = ",".join("?" for _ in MACRO_PRICE_TICKERS)
    with closing(sqlite3.connect(f"file:{database.resolve().as_posix()}?mode=ro", uri=True)) as con:
        rows = pd.read_sql_query(
            f"""
            SELECT date,ticker,COALESCE(adj_close,close) AS value
            FROM daily_prices WHERE ticker IN ({placeholders})
            ORDER BY date,ticker
            """,
            con,
            params=MACRO_PRICE_TICKERS,
            parse_dates=["date"],
        )
    matrix = rows.pivot(index="date", columns="ticker", values="value").sort_index()
    missing = sorted(set(MACRO_PRICE_TICKERS) - set(matrix))
    if missing:
        raise ValueError("missing macro price history: " + ", ".join(missing))
    clean = matrix[list(MACRO_PRICE_TICKERS)].apply(pd.to_numeric, errors="coerce")
    logs = np.log(clean.where(clean > 0))
    result = pd.DataFrame(index=clean.index)
    result["nq_return_20"] = logs["NQ=F"].diff(20)
    result["ief_return_20"] = logs["IEF"].diff(20)
    result["hyg_minus_ief_20"] = logs["HYG"].diff(20) - logs["IEF"].diff(20)
    result["uup_return_20"] = logs["UUP"].diff(20)
    result["vix_level"] = logs["^VIX"]
    result["vix_change_5"] = logs["^VIX"].diff(5)
    result["vix_curve"] = logs["^VIX"] - logs["^VIX3M"]
    result["move_level"] = logs["^MOVE"]
    result["move_change_5"] = logs["^MOVE"].diff(5)
    return result


def _pit_macro_series(
    database: Path, decision_dates: pd.DatetimeIndex, series: str
) -> pd.Series:
    with closing(sqlite3.connect(f"file:{database.resolve().as_posix()}?mode=ro", uri=True)) as con:
        rows = pd.read_sql_query(
            """
            SELECT date,value,available_at FROM macro_observations
            WHERE series=? ORDER BY available_at,date
            """,
            con,
            params=(series,),
            parse_dates=["date"],
        )
    if rows.empty:
        raise ValueError(f"missing PIT macro series: {series}")
    rows["available_day"] = pd.to_datetime(
        rows["available_at"], errors="coerce", utc=True
    ).dt.tz_convert(None).dt.normalize()
    rows["date"] = pd.to_datetime(rows["date"]).dt.normalize()
    values = []
    for decision in decision_dates:
        eligible = rows.loc[rows["available_day"] < decision]
        if eligible.empty:
            values.append(np.nan)
            continue
        latest_observation = eligible["date"].max()
        vintage = eligible.loc[eligible["date"] == latest_observation].iloc[-1]
        values.append(float(vintage["value"]))
    return pd.Series(values, index=decision_dates, dtype=float)


def build_macro_features(database: Path, decision_dates: pd.DatetimeIndex) -> pd.DataFrame:
    features = _read_macro_prices(database).reindex(decision_dates).copy()
    features["hy_oas_level"] = _pit_macro_series(database, decision_dates, "HY_OAS")
    features["cash_3m_level"] = _pit_macro_series(database, decision_dates, "CASH_3M")
    features["hy_oas_change_4w"] = features["hy_oas_level"].diff(4)
    features["cash_3m_change_4w"] = features["cash_3m_level"].diff(4)
    regime = load_regime_probabilities(database, HMM_MODEL_VERSION).set_index("date")
    features = features.join(regime[["p_sideways", "p_bear"]], how="left")
    return features.loc[:, list(MACRO_FEATURES)]


def build_weekly_dataset(
    flow: pd.DataFrame, database: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    prices = _load_price_matrix_read_only(database)
    flow_panel = build_future_flow_panel(flow, prices.index)
    decision_dates = pd.DatetimeIndex(sorted(flow_panel["date"].unique()))
    price = build_sector_feature_frame(prices, build_rrg_signals(prices), horizon=20)
    price = price.loc[
        price["ticker"].isin(SECTORS), ["date", "ticker", *PRICE_FEATURES]
    ]
    panel = flow_panel.merge(price, on=["date", "ticker"], how="inner", validate="one_to_one")

    price_wide = panel.pivot(index="date", columns="ticker", values=list(PRICE_FEATURES))
    price_wide.columns = [f"{feature}__{ticker}" for feature, ticker in price_wide.columns]
    macro = build_macro_features(database, decision_dates)
    weekly = price_wide.join(macro, how="inner")
    label_available = panel.groupby("date")["label_available_date"].first()
    weekly["label_available_date"] = label_available
    complete_dates = weekly.dropna().index
    weekly = weekly.loc[complete_dates].sort_index()
    panel = panel.loc[panel["date"].isin(complete_dates)].copy()

    feature_sets = {
        "PRICE_RRG_RIDGE": list(price_wide.columns),
        "MACRO_RIDGE": list(MACRO_FEATURES),
        "PRICE_MACRO_RIDGE": [*price_wide.columns, *MACRO_FEATURES],
    }
    return weekly, panel, feature_sets


def expanding_predictions(
    weekly: pd.DataFrame,
    panel: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    *,
    min_train_weeks: int,
    alpha: float,
) -> pd.DataFrame:
    target = panel.pivot(index="date", columns="ticker", values="target_future_flow_rank")
    current = panel.pivot(index="date", columns="ticker", values="current_flow_rank")
    outputs = []
    for decision in weekly.index:
        known = weekly.index[weekly["label_available_date"] < decision]
        known = known.intersection(target.dropna().index)
        if len(known) < min_train_weeks or decision not in target.index:
            continue
        base = pd.DataFrame(
            {
                "date": decision,
                "ticker": SECTORS,
                "actual": target.loc[decision, list(SECTORS)].to_numpy(),
                "current_flow_rank": current.loc[decision, list(SECTORS)].to_numpy(),
            }
        )
        persistence = base.copy()
        persistence["model"] = "FLOW_PERSISTENCE"
        persistence["prediction"] = persistence["current_flow_rank"]
        outputs.append(persistence)
        for model, columns in feature_sets.items():
            pipeline = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            pipeline.fit(weekly.loc[known, columns], target.loc[known, list(SECTORS)])
            scored = base.copy()
            scored["model"] = model
            scored["prediction"] = pipeline.predict(weekly.loc[[decision], columns])[0]
            outputs.append(scored)
    if not outputs:
        raise ValueError("no expanding-window folds")
    return pd.concat(outputs, ignore_index=True)


def weekly_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (date, model), current in predictions.groupby(["date", "model"], sort=True):
        predicted_top = set(current.nlargest(3, "prediction")["ticker"])
        actual_top = set(current.nlargest(3, "actual")["ticker"])
        candidates = current.loc[
            current["ticker"].isin(predicted_top) & (current["current_flow_rank"] < 9 / len(SECTORS))
        ]
        rows.append(
            {
                "date": date,
                "model": model,
                "rank_ic": current["prediction"].rank().corr(current["actual"].rank()),
                "top3_overlap": len(predicted_top & actual_top) / 3,
                "rotation_in_precision": (
                    np.nan
                    if candidates.empty
                    else candidates["ticker"].isin(actual_top).mean()
                ),
                "rotation_candidates": len(candidates),
            }
        )
    return pd.DataFrame(rows)


def summarize(
    weekly: pd.DataFrame,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    *,
    flow_csv: Path,
    database: Path,
    min_train_weeks: int,
    alpha: float,
    bootstrap_samples: int,
) -> dict[str, Any]:
    results = {}
    for model in MODEL_ORDER:
        rows = metrics.loc[metrics["model"] == model]
        results[model] = {
            "weeks": int(len(rows)),
            "rank_ic": _finite(rows["rank_ic"].mean()),
            "top3_overlap": _finite(rows["top3_overlap"].mean()),
            "rotation_in_precision": _finite(
                np.average(
                    rows["rotation_in_precision"].dropna(),
                    weights=rows.loc[rows["rotation_in_precision"].notna(), "rotation_candidates"],
                )
            ) if rows["rotation_in_precision"].notna().any() else None,
            "rotation_candidates": int(rows["rotation_candidates"].sum()),
            "block_5w_95ci": {
                metric: list(
                    _block_bootstrap_mean_ci(
                        rows[metric], block_weeks=5, samples=bootstrap_samples
                    )
                )
                for metric in ("rank_ic", "top3_overlap")
            },
        }
    left = metrics.loc[metrics["model"] == "PRICE_MACRO_RIDGE"].set_index("date")
    right = metrics.loc[metrics["model"] == "PRICE_RRG_RIDGE"].set_index("date")
    common = left.index.intersection(right.index)
    uplift = {}
    for metric in ("rank_ic", "top3_overlap"):
        delta = left.loc[common, metric] - right.loc[common, metric]
        uplift[metric] = {
            "mean": _finite(delta.mean()),
            "block_5w_95ci": list(
                _block_bootstrap_mean_ci(
                    delta, block_weeks=5, samples=bootstrap_samples
                )
            ),
        }
    return _json_ready(
        {
            "experiment": "future-sector-flow-forecast-v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "research_only": True,
            "pit_status": "PIT_PARTIAL",
            "target": "cross-sectional rank of next four reported weekly flow/lagged-AUM observations",
            "information_timing": (
                "Flow source week is usable in W+1; target covers the following four source "
                "weeks and enters training only after the last target week is available."
            ),
            "inputs": {
                "flow_csv": str(flow_csv.resolve()),
                "database_read_only": str(database.resolve()),
                "macro_features": list(MACRO_FEATURES),
                "price_features_per_sector": list(PRICE_FEATURES),
            },
            "sample": {
                "complete_weeks": int(len(weekly)),
                "oos_weeks": int(metrics["date"].nunique()),
                "oos_start": metrics["date"].min().date().isoformat(),
                "oos_end": metrics["date"].max().date().isoformat(),
            },
            "settings": {
                "minimum_training_weeks": min_train_weeks,
                "ridge_alpha": alpha,
                "bootstrap_samples": bootstrap_samples,
            },
            "results": results,
            "price_macro_minus_price": uplift,
            "verdict": "RESEARCH_SHADOW_ONLY",
        }
    )


def markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Future Bloomberg Sector Flow Forecast",
        "",
        f"- 판정: **{summary['verdict']}**",
        f"- PIT: **{summary['pit_status']}**",
        f"- OOS: {summary['sample']['oos_weeks']}주",
        f"- 목표: {summary['target']}",
        "",
        "| 모델 | Rank IC | Top3 일치율 | Rotation IN 정밀도 | 후보 수 |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in MODEL_ORDER:
        result = summary["results"][model]
        lines.append(
            f"| {model} | {result['rank_ic']:+.3f} | "
            f"{result['top3_overlap']:.1%} | "
            f"{'—' if result['rotation_in_precision'] is None else f'{result["rotation_in_precision"]:.1%}'} | "
            f"{result['rotation_candidates']} |"
        )
    uplift = summary["price_macro_minus_price"]
    lines.extend(
        [
            "",
            "## 매크로 증분 효과",
            "",
            f"- Δ Rank IC: {uplift['rank_ic']['mean']:+.3f} "
            f"(95% CI {uplift['rank_ic']['block_5w_95ci'][0]:+.3f} ~ "
            f"{uplift['rank_ic']['block_5w_95ci'][1]:+.3f})",
            f"- Δ Top3 일치율: {uplift['top3_overlap']['mean']:+.1%} "
            f"(95% CI {uplift['top3_overlap']['block_5w_95ci'][0]:+.1%} ~ "
            f"{uplift['top3_overlap']['block_5w_95ci'][1]:+.1%})",
            "",
            "실질금리·금리곡선·원유·정제마진은 현재 DB에 없어 이번 사양에서 제외했습니다.",
            "Bloomberg flow vintage가 없어 결과는 연구용이며 운영 모델에 반영하지 않습니다.",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    flow = load_flow_history(args.flow_csv)
    weekly, panel, feature_sets = build_weekly_dataset(flow, args.database)
    predictions = expanding_predictions(
        weekly,
        panel,
        feature_sets,
        min_train_weeks=args.min_train_weeks,
        alpha=args.alpha,
    )
    metrics = weekly_metrics(predictions)
    summary = summarize(
        weekly,
        predictions,
        metrics,
        flow_csv=args.flow_csv,
        database=args.database,
        min_train_weeks=args.min_train_weeks,
        alpha=args.alpha,
        bootstrap_samples=args.bootstrap_samples,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    metrics.to_csv(args.output_dir / "weekly_metrics.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "REPORT.md").write_text(markdown_report(summary), encoding="utf-8")
    return summary


def self_test() -> None:
    tickers = ("SPY", *SECTORS)
    dates = pd.date_range("2023-01-06", periods=20, freq="W-FRI")
    flow = pd.DataFrame(
        [
            {
                "ticker": ticker,
                "date": day,
                "fund_flow_1w": (week + 1) * (index - 4),
                "aum": 1_000 + index * 100 + week,
            }
            for index, ticker in enumerate(tickers)
            for week, day in enumerate(dates)
        ]
    )
    panel = build_future_flow_panel(
        flow, pd.bdate_range("2023-01-01", "2023-07-31")
    )
    assert panel["ticker"].nunique() == len(SECTORS)
    assert panel.groupby("date").size().eq(len(SECTORS)).all()
    assert (panel["label_available_date"] > panel["date"]).all()
    assert panel.groupby("date")["target_future_flow_rank"].mean().abs().lt(1e-12).all()
    sample = panel.loc[panel["date"] == panel["date"].min()].copy()
    sample = sample.rename(columns={"target_future_flow_rank": "actual"})
    sample["model"] = "FLOW_PERSISTENCE"
    sample["prediction"] = sample["current_flow_rank"]
    metric = weekly_metrics(sample).iloc[0]
    assert metric["rotation_candidates"] == 0
    print("self-test: OK")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Research-only forecast of future Bloomberg sector fund flows"
    )
    result.add_argument("--self-test", action="store_true")
    result.add_argument("--flow-csv", type=Path)
    result.add_argument("--database", type=Path, default=Path("regime_router_publish.db"))
    result.add_argument("--output-dir", type=Path)
    result.add_argument("--min-train-weeks", type=int, default=52)
    result.add_argument("--alpha", type=float, default=20.0)
    result.add_argument("--bootstrap-samples", type=int, default=2000)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.self_test:
        self_test()
        return
    if args.flow_csv is None or args.output_dir is None:
        raise SystemExit("--flow-csv and --output-dir are required")
    if args.min_train_weeks < 52 or args.alpha <= 0 or args.bootstrap_samples < 100:
        raise SystemExit("require min_train_weeks>=52, alpha>0, bootstrap_samples>=100")
    summary = run(args)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), **summary["sample"]}))


if __name__ == "__main__":
    main()
