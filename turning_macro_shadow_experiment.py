"""Research-only test of Bloomberg macro features for trend-turn events."""

from __future__ import annotations

import argparse
import json
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gics_l2_shadow_experiment import (
    build_cohort_prices,
    cohort_codes,
    load_package,
)
from research_drawdown_onset import (
    HORIZON,
    Candidate,
    causal_trailing_percentile,
    drawdown_onset_labels,
    evaluate_score,
    paired_auc_delta_ci,
    walk_forward_scores,
)
EXPECTED_MACRO_SERIES = {
    "UST_2Y",
    "UST_10Y",
    "UST_REAL_10Y",
    "BREAKEVEN_10Y",
    "WTI_SPOT",
    "BRENT_SPOT",
}


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
REQUIRED_MACRO_COLUMNS = {
    "series",
    "fred_series_id",
    "security",
    "bloomberg_field",
    "date",
    "value",
    "unit",
    "currency",
    "exact_or_proxy",
    "pit_status",
    "retrieved_at_utc",
    "source",
}
PIT_STATUS = "CURRENT_REVISED_HISTORY_ONLY"
SOURCE = "Bloomberg Desktop API"
PRICE_FEATURES = (
    "relative_return_5",
    "relative_return_20",
    "relative_volatility_20",
    "relative_ma200_gap",
    "relative_drawdown_60",
    "relative_acceleration_5",
)
COMPARISONS = {
    "PRICE_ONLY_vs_PRICE_BBG5": ("PRICE_ONLY", "PRICE_BBG5"),
    "PRICE_BBG5_vs_PRICE_BBG6_WTI": ("PRICE_BBG5", "PRICE_BBG6_WTI"),
}
ALERT_CUTOFF = 0.80


def ensure_output_boundary(output_dir: Path, *input_package_dirs: Path) -> None:
    output = output_dir.resolve()
    for package in input_package_dirs:
        protected = package.resolve()
        if output == protected or output.is_relative_to(protected):
            raise ValueError(f"output_dir must be outside input package: {protected}")


def load_macro_history(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    missing = sorted(REQUIRED_MACRO_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError("macro_history.csv: missing columns: " + ", ".join(missing))
    frame = frame.copy()
    frame["series"] = frame["series"].astype("string").str.strip()
    found = set(frame["series"].dropna())
    if found != EXPECTED_MACRO_SERIES:
        raise ValueError(
            "macro_history.csv: expected exactly six series; found "
            + ", ".join(sorted(found))
        )
    frame["date"] = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    if frame["date"].isna().any():
        raise ValueError("macro_history.csv: invalid date")
    if frame["value"].isna().any() or not np.isfinite(frame["value"]).all():
        raise ValueError("macro_history.csv: value must be finite numeric")
    if frame.duplicated(["series", "date"]).any():
        raise ValueError("macro_history.csv: duplicate series/date")
    if set(frame["source"].astype("string").str.strip()) != {SOURCE}:
        raise ValueError(f"macro_history.csv: source must be {SOURCE!r}")
    if set(frame["pit_status"].astype("string").str.strip()) != {PIT_STATUS}:
        raise ValueError(f"macro_history.csv: pit_status must be {PIT_STATUS!r}")
    if set(frame["bloomberg_field"].astype("string").str.strip()) != {"PX_LAST"}:
        raise ValueError("macro_history.csv: bloomberg_field must be 'PX_LAST'")
    if set(frame["currency"].astype("string").str.strip()) != {"USD"}:
        raise ValueError("macro_history.csv: currency must be 'USD'")
    return frame.sort_values(["date", "series"]).reset_index(drop=True)


def _factor_features(factors: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(index=factors.index)
    for name in factors:
        level = factors[name].astype(float)
        output[f"{name}_level"] = level
        for lag in (1, 5, 20):
            output[f"{name}_change_{lag}"] = level.diff(lag)
        output[f"{name}_acceleration_5"] = level.diff(5).diff(5)
    return output


def build_macro_feature_sets(
    history: pd.DataFrame,
) -> tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...]]:
    """Return nonredundant base features and the post-2017 WTI extension.

    The five non-WTI raw series are audited, but rates are represented by the
    nonredundant 2s10s curve, real 10Y, and breakeven 10Y basis.
    WTI uses levels and dollar changes only, so its negative 2020 print survives.
    """
    wide = history.pivot(index="date", columns="series", values="value").sort_index()
    base_factors = pd.DataFrame(
        {
            "curve_2s10s": wide["UST_10Y"] - wide["UST_2Y"],
            "real_rate_10y": wide["UST_REAL_10Y"],
            "breakeven_10y": wide["BREAKEVEN_10Y"],
            "brent_spot": wide["BRENT_SPOT"],
        },
        index=wide.index,
    )
    wti_factors = pd.DataFrame(
        {
            "wti_spot": wide["WTI_SPOT"],
            "brent_wti_spread": wide["BRENT_SPOT"] - wide["WTI_SPOT"],
        },
        index=wide.index,
    )
    base = _factor_features(base_factors)
    extension = _factor_features(wti_factors)
    return base.join(extension), tuple(base.columns), tuple(extension.columns)


def merge_macro_next_session(
    market_dates: pd.DatetimeIndex,
    macro_features: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    asof_column: str,
) -> pd.DataFrame:
    """Backward as-of join that forbids same-day Bloomberg observations."""
    columns = list(feature_columns)
    if macro_features.index.duplicated().any():
        raise ValueError("macro feature dates must be unique")
    dates = pd.DatetimeIndex(market_dates).tz_localize(None).normalize()
    if dates.duplicated().any():
        raise ValueError("market dates must be unique")
    left = pd.DataFrame({"date": dates, "_order": np.arange(len(dates))})
    right = macro_features.loc[:, columns].dropna().copy()
    right.index = pd.DatetimeIndex(right.index).tz_localize(None).normalize()
    right = right.reset_index(names=asof_column).sort_values(asof_column)
    if right.empty:
        result = left.copy()
        result[asof_column] = pd.NaT
        for column in columns:
            result[column] = np.nan
    else:
        result = pd.merge_asof(
            left.sort_values("date"),
            right,
            left_on="date",
            right_on=asof_column,
            direction="backward",
            allow_exact_matches=False,
            tolerance=pd.Timedelta(days=7),
        )
    return result.sort_values("_order").drop(columns="_order").set_index("date")


def validate_database_read_only(database: Path) -> None:
    if not database.is_file():
        raise FileNotFoundError(database)
    uri = f"file:{database.resolve().as_posix()}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        connection.execute("PRAGMA query_only=ON")
        found = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='daily_prices'"
        ).fetchone()
    if found is None:
        raise ValueError("database: missing daily_prices table")


def load_nq_read_only(database: Path) -> pd.Series:
    uri = f"file:{database.resolve().as_posix()}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        connection.execute("PRAGMA query_only=ON")
        rows = pd.read_sql_query(
            """
            SELECT date,COALESCE(adj_close,close) AS value
            FROM daily_prices WHERE ticker='NQ=F' ORDER BY date
            """,
            connection,
            parse_dates=["date"],
        )
    rows["value"] = pd.to_numeric(rows["value"], errors="coerce")
    if rows.empty or rows["date"].duplicated().any():
        raise ValueError("database: invalid or missing NQ=F history")
    clean = rows.dropna().set_index("date")["value"].astype(float)
    if not np.isfinite(clean).all() or (clean <= 0).any():
        raise ValueError("database: NQ=F prices must be finite and positive")
    return clean


def build_event_frame(price: pd.Series, direction: str) -> pd.DataFrame:
    series = pd.to_numeric(price, errors="coerce").dropna().astype(float)
    series.index = pd.DatetimeIndex(series.index).tz_localize(None).normalize()
    series = series.loc[~series.index.duplicated(keep=False)].sort_index()
    if series.empty or not np.isfinite(series).all() or (series <= 0).any():
        raise ValueError("target price must be unique, finite, and positive")
    one_day = series.pct_change(fill_method=None)
    return_5 = series.pct_change(5, fill_method=None)
    frame = pd.DataFrame(
        {
            "relative_return_5": return_5,
            "relative_return_20": series.pct_change(20, fill_method=None),
            "relative_volatility_20": one_day.rolling(20).std() * np.sqrt(252),
            "relative_ma200_gap": series / series.rolling(200).mean() - 1,
            "relative_drawdown_60": series / series.rolling(60).max() - 1,
            "relative_acceleration_5": return_5 - return_5.shift(5),
        },
        index=series.index,
    )
    if direction == "ROTATION_OUT":
        labelled_price = series
    elif direction == "ROTATION_IN":
        labelled_price = 1.0 / series
    else:
        raise ValueError(f"unknown direction: {direction}")
    frame = frame.join(drawdown_onset_labels(labelled_price))
    frame.insert(0, "date", frame.index)
    frame["_position"] = np.arange(len(frame))
    return frame.reset_index(drop=True)


def attach_macro_features(
    frame: pd.DataFrame,
    macro_features: pd.DataFrame,
    base_features: Sequence[str],
    wti_features: Sequence[str],
) -> pd.DataFrame:
    dates = pd.DatetimeIndex(frame["date"])
    base = merge_macro_next_session(
        dates, macro_features, base_features, asof_column="bbg5_asof_date"
    ).reset_index()
    wti = merge_macro_next_session(
        dates, macro_features, wti_features, asof_column="wti_asof_date"
    ).reset_index()
    additions = base.merge(wti, on="date", how="left", validate="one_to_one")
    return frame.merge(additions, on="date", how="left", validate="one_to_one")


def _candidate(name: str, features: Sequence[str]) -> Candidate:
    def factory() -> object:
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=2_000,
                random_state=42,
            ),
        )

    return Candidate(name, tuple(features), factory)


def _paired_frame(
    frame: pd.DataFrame,
    common_features: Sequence[str],
    *,
    start_date: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    paired = frame.copy()
    complete = paired[list(common_features)].notna().all(axis=1)
    if start_date is not None:
        complete &= paired["date"] >= start_date
    paired["at_risk"] = paired["at_risk"].fillna(False).astype(bool) & complete
    paired.loc[~complete, "target"] = np.nan
    prediction_mask = paired["at_risk"] & paired["target"].notna()
    return paired, prediction_mask


def run_comparison(
    frame: pd.DataFrame,
    *,
    ticker: str,
    target_name: str,
    direction: str,
    comparison: str,
    baseline_name: str,
    baseline_features: Sequence[str],
    enhanced_name: str,
    enhanced_features: Sequence[str],
    min_train: int,
    refit_every: int,
    bootstrap_samples: int,
    start_date: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    universe = "NQ" if ticker == "NQ=F" else "GICS_CORE24"
    common_features = tuple(dict.fromkeys([*baseline_features, *enhanced_features]))
    paired, prediction_mask = _paired_frame(
        frame, common_features, start_date=start_date
    )
    scores: dict[str, pd.Series] = {}
    for name, features in (
        (baseline_name, baseline_features),
        (enhanced_name, enhanced_features),
    ):
        raw = walk_forward_scores(
            paired,
            _candidate(name, features),
            prediction_mask,
            horizon=HORIZON,
            refit_every=refit_every,
            min_train=min_train,
        )
        scores[name] = causal_trailing_percentile(raw, min_history=60)
    common = prediction_mask & scores[baseline_name].notna() & scores[enhanced_name].notna()
    onset_positions = paired.loc[paired["onset"], "_position"].astype(int).tolist()
    metric_rows: list[dict[str, Any]] = []
    for model, score in scores.items():
        result = evaluate_score(
            paired,
            common,
            score,
            onset_positions,
            bootstrap_samples=bootstrap_samples,
            fixed_cutoff=ALERT_CUTOFF,
        )
        events = result.pop("event_metrics")
        sample = paired.loc[common]
        years = (
            max((sample["date"].max() - sample["date"].min()).days / 365.25, 1 / 252)
            if len(sample) > 1
            else 0.0
        )
        metric_rows.append(
            {
                "universe": universe,
                "comparison": comparison,
                "ticker": ticker,
                "target_name": target_name,
                "direction": direction,
                "model": model,
                "evaluation_years": years,
                **result,
                **events,
            }
        )
    sample = paired.loc[common]
    paired_auc = paired_auc_delta_ci(
        sample["target"].astype(int).to_numpy(),
        scores[enhanced_name].loc[common].to_numpy(),
        scores[baseline_name].loc[common].to_numpy(),
        samples=bootstrap_samples,
    )
    predictions = []
    columns = [
        "date",
        "_position",
        "target",
        "onset",
        "at_risk",
        "label_end_position",
        "bbg5_asof_date",
        "wti_asof_date",
    ]
    for model, score in scores.items():
        rows = paired.loc[common, columns].copy()
        rows.insert(0, "universe", universe)
        rows.insert(1, "comparison", comparison)
        rows.insert(2, "ticker", ticker)
        rows.insert(3, "target_name", target_name)
        rows.insert(4, "direction", direction)
        rows.insert(5, "model", model)
        rows["score"] = score.loc[common]
        rows["alert"] = rows["score"] >= ALERT_CUTOFF
        predictions.append(rows)
    return (
        pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(),
        metric_rows,
        {
            "universe": universe,
            "comparison": comparison,
            "ticker": ticker,
            "direction": direction,
            "baseline": baseline_name,
            "enhanced": enhanced_name,
            **paired_auc,
            "n": int(common.sum()),
        },
    )


def aggregate_results(
    predictions: pd.DataFrame, metrics: pd.DataFrame, *, bootstrap_samples: int
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for (universe, comparison, direction), group in predictions.groupby(
        ["universe", "comparison", "direction"], sort=True
    ):
        baseline, enhanced = COMPARISONS[comparison]
        score = group.pivot(
            index=["date", "ticker"], columns="model", values="score"
        ).dropna(subset=[baseline, enhanced])
        target = (
            group.drop_duplicates(["date", "ticker"])
            .set_index(["date", "ticker"])["target"]
            .reindex(score.index)
            .astype(int)
        )
        block_size = min(max(group["ticker"].nunique() * 40, 40), len(score))
        paired = paired_auc_delta_ci(
            target.to_numpy(),
            score[enhanced].to_numpy(),
            score[baseline].to_numpy(),
            block_size=block_size,
            samples=bootstrap_samples,
        )
        model_metrics = {}
        for model in (baseline, enhanced):
            values = score[model].to_numpy()
            model_rows = metrics.loc[
                (metrics["universe"] == universe)
                & (metrics["comparison"] == comparison)
                & (metrics["direction"] == direction)
                & (metrics["model"] == model)
            ]
            eligible = int(model_rows["eligible_onsets"].sum())
            clusters = int(model_rows["alert_clusters"].sum())
            years = float(model_rows["evaluation_years"].sum())
            leads = model_rows["median_lead_days"].dropna()
            model_metrics[model] = {
                "n": len(target),
                "events": int(target.sum()),
                "roc_auc": (
                    float(roc_auc_score(target, values))
                    if target.nunique() == 2
                    else None
                ),
                "pr_auc": (
                    float(average_precision_score(target, values))
                    if target.sum()
                    else None
                ),
                "eligible_onsets": eligible,
                "onset_recall": (
                    float(model_rows["matched_onsets"].sum()) / eligible
                    if eligible
                    else 0.0
                ),
                "cluster_precision": (
                    float(model_rows["matched_clusters"].sum()) / clusters
                    if clusters
                    else 0.0
                ),
                "false_alert_clusters_per_year": (
                    float(model_rows["false_alert_clusters"].sum()) / years
                    if years
                    else 0.0
                ),
                "median_of_sector_median_lead_days": (
                    float(leads.median()) if not leads.empty else None
                ),
            }
        output[f"{universe}:{comparison}:{direction}"] = {
            "baseline": baseline,
            "enhanced": enhanced,
            "models": model_metrics,
            "paired_auc_uplift": paired,
            "moving_block_rows": block_size,
        }
    return output


def markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Bloomberg Macro Trend-Turn SHADOW Experiment",
        "",
        f"- 판정: **{summary['verdict']}**",
        f"- PIT: **{summary['pit_status']}**",
        "- 목표: GICS L2 상대가격과 NQ의 상승·하락 추세 종료를 다음 20거래일 안에 포착",
        "- 경고: Bloomberg 및 GICS 이력은 당시 공개 vintage가 검증되지 않았습니다.",
        "",
        "| 유니버스 | 비교 | 이벤트 | 모델 | 표본 | ROC AUC | PR AUC | Onset recall | Cluster precision | False clusters/yr | Median lead |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, result in summary["aggregate"].items():
        universe, comparison, direction = key.split(":", 2)
        for model, metric in result["models"].items():
            def number(value: Any, digits: int = 3) -> str:
                return "—" if value is None else f"{value:.{digits}f}"

            lines.append(
                f"| {universe} | {comparison} | {direction} | {model} | {metric['n']} | "
                f"{number(metric['roc_auc'])} | {number(metric['pr_auc'])} | "
                f"{metric['onset_recall']:.1%} | {metric['cluster_precision']:.1%} | "
                f"{metric['false_alert_clusters_per_year']:.2f} | "
                f"{number(metric['median_of_sector_median_lead_days'], 1)} |"
            )
        uplift = result["paired_auc_uplift"]
        lines.append(
            f"| {universe} | Δ AUC | {direction} | {result['enhanced']}−{result['baseline']} | — | "
            f"{number(uplift['delta_auc'])} [{number(uplift['ci_low'])}, "
            f"{number(uplift['ci_high'])}] | — | — | — | — | — |"
        )
    lines.extend(
        [
            "",
            "## 방법론 경계",
            "",
            "- 동일 비교 안에서는 학습·예측 표본을 완전히 일치시켰습니다.",
            "- 라벨 종료 위치가 예측 위치 이하인 관측치만 학습했습니다.",
            "- Bloomberg d일 값은 d일에 사용하지 않고 다음 시장 세션부터 최대 7일까지만 사용했습니다.",
            "- 원자료는 보간·forward-fill하지 않았고 WTI에는 로그를 사용하지 않았습니다.",
            "- 10Y 명목금리는 실질금리+BEI와 중복되어 회귀 입력에서 제외했습니다.",
            "- 이 결과는 연구용 SHADOW이며 운영 승격을 금지합니다.",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    database = args.database.resolve()
    macro_csv = args.macro_csv.resolve()
    package_dir = args.gics_package_dir.resolve()
    output_dir = args.output_dir.resolve()
    ensure_output_boundary(output_dir, package_dir, macro_csv.parent)
    validate_database_read_only(database)
    history = load_macro_history(macro_csv)
    macro_features, base_features, wti_features = build_macro_feature_sets(history)
    master, groups, benchmark = load_package(package_dir)
    codes = cohort_codes(master, "core24")
    prices = build_cohort_prices(groups, benchmark, codes)
    names = master.set_index("industry_group_code")["industry_group_name"].to_dict()
    targets: list[tuple[str, str, pd.Series]] = [
        (code, str(names[code]), prices[code].div(prices["SPX"]).dropna())
        for code in codes
    ]
    if args.include_nq:
        targets.append(("NQ=F", "NASDAQ 100 futures", load_nq_read_only(database)))

    prediction_parts: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for ticker, target_name, price in targets:
        for direction in ("ROTATION_OUT", "ROTATION_IN"):
            frame = attach_macro_features(
                build_event_frame(price, direction),
                macro_features,
                base_features,
                wti_features,
            )
            configurations = (
                (
                    "PRICE_ONLY_vs_PRICE_BBG5",
                    "PRICE_ONLY",
                    PRICE_FEATURES,
                    "PRICE_BBG5",
                    (*PRICE_FEATURES, *base_features),
                    None,
                ),
                (
                    "PRICE_BBG5_vs_PRICE_BBG6_WTI",
                    "PRICE_BBG5",
                    (*PRICE_FEATURES, *base_features),
                    "PRICE_BBG6_WTI",
                    (*PRICE_FEATURES, *base_features, *wti_features),
                    pd.Timestamp("2017-01-23"),
                ),
            )
            for comparison, baseline, baseline_columns, enhanced, enhanced_columns, start in configurations:
                predictions, metrics, pair = run_comparison(
                    frame,
                    ticker=ticker,
                    target_name=target_name,
                    direction=direction,
                    comparison=comparison,
                    baseline_name=baseline,
                    baseline_features=baseline_columns,
                    enhanced_name=enhanced,
                    enhanced_features=enhanced_columns,
                    min_train=args.min_train,
                    refit_every=args.refit_every,
                    bootstrap_samples=args.bootstrap_samples,
                    start_date=start,
                )
                if not predictions.empty:
                    prediction_parts.append(predictions)
                metric_rows.extend(metrics)
                pair_rows.append(pair)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    summary = {
        "verdict": "RESEARCH_SHADOW_ONLY",
        "research_only": True,
        "production_promotion": "PROHIBITED",
        "pit_status": "PIT_PARTIAL / CURRENT_REVISED_HISTORY_ONLY",
        "target": {
            "ROTATION_OUT": "drawdown onset of GICS group/SPX relative total-return price",
            "ROTATION_IN": "drawdown onset of reciprocal relative total-return price",
            "horizon_sessions": HORIZON,
        },
        "features": {
            "price": list(PRICE_FEATURES),
            "bbg5_raw_audited": sorted(EXPECTED_MACRO_SERIES - {"WTI_SPOT"}),
            "bbg5_nonredundant": list(base_features),
            "bbg6_wti_extension": list(wti_features),
            "rate_basis": "2s10s curve + real 10Y + breakeven 10Y; raw 2Y/10Y levels are not duplicated",
        },
        "sample": {
            "gics_groups": len(codes),
            "include_nq": bool(args.include_nq),
            "macro_start": str(history["date"].min().date()),
            "macro_end": str(history["date"].max().date()),
            "macro_rows": len(history),
            "predictions": len(predictions),
        },
        "coverage": {
            series: {
                "rows": int(len(rows)),
                "first_date": str(rows["date"].min().date()),
                "last_date": str(rows["date"].max().date()),
            }
            for series, rows in history.groupby("series", sort=True)
        }
        | {
            "bbg5_complete_source_dates": int(
                macro_features[list(base_features)].notna().all(axis=1).sum()
            ),
            "wti_extension_complete_source_dates": int(
                macro_features[list(wti_features)].notna().all(axis=1).sum()
            ),
        },
        "walk_forward": {
            "min_train": args.min_train,
            "refit_every": args.refit_every,
            "purge": "label_end_position <= prediction position",
            "random_split": False,
            "score": "causal trailing percentile; fixed alert cutoff 0.80",
        },
        "macro_alignment": {
            "allow_exact_matches": False,
            "direction": "backward",
            "tolerance_calendar_days": 7,
            "raw_fill_or_interpolation": False,
            "wti_transform": "raw USD level and dollar differences; no log",
        },
        "aggregate": aggregate_results(
            predictions, metrics, bootstrap_samples=args.bootstrap_samples
        ),
        "per_target_paired_auc": pair_rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    clean_summary = _json_ready(summary)
    (output_dir / "summary.json").write_text(
        json.dumps(clean_summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (output_dir / "REPORT.md").write_text(
        markdown_report(clean_summary), encoding="utf-8"
    )
    return clean_summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Research-only Bloomberg macro trend-turn SHADOW experiment"
    )
    result.add_argument("--database", type=Path, default=Path("regime_router_publish.db"))
    result.add_argument("--macro-csv", type=Path, required=True)
    result.add_argument("--gics-package-dir", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--bootstrap-samples", type=int, default=500)
    result.add_argument("--min-train", type=int, default=252)
    result.add_argument("--refit-every", type=int, default=63)
    result.add_argument("--include-nq", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.bootstrap_samples < 100 or args.min_train < 252 or args.refit_every < 1:
        raise SystemExit("require bootstrap_samples>=100, min_train>=252, refit_every>=1")
    summary = run(args)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), **summary["sample"]}))


if __name__ == "__main__":
    main()
