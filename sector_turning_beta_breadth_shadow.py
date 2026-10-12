"""Research-only PRE_PIVOT comparison with causal sector beta adjustment."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from gics_l2_shadow_experiment import build_cohort_prices, load_package
from sector_breadth_features import MODEL_FEATURES as BREADTH_FEATURES
from sector_turning_breadth_shadow import economic_rows, summarize_economics
from sector_turning_pooled_shadow import (
    EVALUATION_START,
    PRICE_FEATURES,
    _direction_frame,
    _json_ready,
    _model_metrics,
    _paired_uplifts,
    _price_features,
    build_pooled_daily_frame,
    causal_trailing_percentile,
    effect_code,
    purged_training_mask,
)


MODELS = (
    "POOLED_PRICE",
    "POOLED_BETA_PRICE",
    "POOLED_BETA_BREADTH",
    "POOLED_PRICE_BETA",
    "POOLED_PRICE_BETA_BREADTH",
)
BETA_FEATURES = tuple(f"beta_{name}" for name in PRICE_FEATURES)
FEATURE_SETS = {
    "POOLED_PRICE": PRICE_FEATURES,
    "POOLED_BETA_PRICE": BETA_FEATURES,
    "POOLED_BETA_BREADTH": (*BETA_FEATURES, *BREADTH_FEATURES),
    "POOLED_PRICE_BETA": (*PRICE_FEATURES, *BETA_FEATURES),
    "POOLED_PRICE_BETA_BREADTH": (*PRICE_FEATURES, *BETA_FEATURES, *BREADTH_FEATURES),
}
ALL_FEATURES = tuple(dict.fromkeys(feature for values in FEATURE_SETS.values() for feature in values))
CUTOFF = 0.90


def causal_beta_features(
    prices: pd.DataFrame,
    codes: list[str],
    *,
    short_window: int = 126,
    structural_window: int = 756,
) -> pd.DataFrame:
    """Estimate beta through t-1, then isolate the sector-specific return at t."""
    market_return = prices["SPX"].pct_change(fill_method=None)
    market_short_var = market_return.rolling(short_window, min_periods=short_window).var()
    market_long_var = market_return.rolling(
        structural_window, min_periods=structural_window
    ).var()
    parts: list[pd.DataFrame] = []
    for code in codes:
        sector_return = prices[code].pct_change(fill_method=None)
        beta_short = (
            sector_return.rolling(short_window, min_periods=short_window).cov(market_return)
            / market_short_var
        )
        beta_structural = (
            sector_return.rolling(structural_window, min_periods=structural_window).cov(
                market_return
            )
            / market_long_var
        )
        reliability = (
            sector_return.rolling(short_window, min_periods=short_window)
            .corr(market_return)
            .pow(2)
            .clip(0, 1)
        )
        beta_used = (
            reliability * beta_short + (1 - reliability) * beta_structural
        ).shift(1)
        beta_expected_return = beta_used * market_return
        residual_return = sector_return - beta_expected_return
        residual_index = (1 + residual_return).cumprod()
        features = _price_features(residual_index).rename(
            columns={name: f"beta_{name}" for name in PRICE_FEATURES}
        )
        part = pd.DataFrame(
            {
                "date": prices.index,
                "sector": code,
                "sector_return_1d": sector_return,
                "market_return_1d": market_return,
                "beta_126": beta_short,
                "beta_756": beta_structural,
                "beta_r2_126": reliability,
                "beta_used_lagged": beta_used,
                "beta_expected_return_1d": beta_expected_return,
                "beta_adjusted_return_1d": residual_return,
                "beta_adjusted_index": residual_index,
            }
        ).join(features)
        for horizon in (5, 20):
            part[f"sector_return_{horizon}d"] = (
                (1 + sector_return).rolling(horizon).apply(np.prod, raw=True) - 1
            )
            part[f"beta_expected_return_{horizon}d"] = (
                (1 + beta_expected_return).rolling(horizon).apply(np.prod, raw=True) - 1
            )
            part[f"beta_adjusted_return_{horizon}d"] = (
                (1 + residual_return).rolling(horizon).apply(np.prod, raw=True) - 1
            )
        parts.append(part.reset_index(drop=True))
    return pd.concat(parts, ignore_index=True).sort_values(["date", "sector"])


def merge_inputs(
    package_dir: Path, breadth_csv: Path, horizon: int
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    pooled, price_audit, _ = build_pooled_daily_frame(package_dir, None, horizon=horizon)
    breadth = pd.read_csv(breadth_csv, dtype={"sector": "string"}, parse_dates=["date"])
    if breadth.duplicated(["date", "sector"]).any():
        raise ValueError("duplicate breadth date/sector rows")
    breadth_codes = set(breadth["sector"].dropna().astype(str))
    price_codes = set(pooled["sector"].astype(str))
    codes = sorted(breadth_codes & price_codes)
    if len(codes) < 2:
        raise ValueError("fewer than two paired price/breadth groups")

    _, groups, benchmark = load_package(package_dir)
    prices = build_cohort_prices(groups, benchmark, codes)
    beta = causal_beta_features(prices, codes)
    names = pooled[["sector", "sector_name"]].drop_duplicates().set_index("sector")["sector_name"]
    beta["sector_name"] = beta["sector"].map(names)

    pooled["sector"] = pooled["sector"].astype(str)
    keep_breadth = ["date", "sector", *BREADTH_FEATURES]
    pooled = pooled.loc[pooled["sector"].isin(codes)].merge(
        breadth[keep_breadth], on=["date", "sector"], how="inner", validate="one_to_one"
    )
    keep_beta = ["date", "sector", *BETA_FEATURES]
    pooled = pooled.merge(beta[keep_beta], on=["date", "sector"], how="inner", validate="one_to_one")
    pooled = pooled.replace([np.inf, -np.inf], np.nan).dropna(subset=ALL_FEATURES)
    pooled = pooled.sort_values(["date", "sector"]).reset_index(drop=True)
    if pooled.empty:
        raise ValueError("no complete paired price/beta/breadth rows")
    audit = {
        "price_input": price_audit,
        "groups": codes,
        "excluded_price_groups_without_breadth": sorted(price_codes - breadth_codes),
        "paired_rows": int(len(pooled)),
        "paired_dates": int(pooled["date"].nunique()),
        "first_paired_date": str(pooled["date"].min().date()),
        "last_paired_date": str(pooled["date"].max().date()),
        "beta_timing": "return_t uses beta estimated through t-1",
        "beta_formula": "R2_126*beta_126 + (1-R2_126)*beta_756",
        "raw_fill_or_interpolation": False,
    }
    return pooled, beta, audit


def _design(
    frame: pd.DataFrame, scaled: np.ndarray, model: str, levels: tuple[str, ...]
) -> pd.DataFrame:
    continuous = pd.DataFrame(scaled, index=frame.index, columns=ALL_FEATURES)
    effects, _ = effect_code(frame["sector"], levels)
    return continuous.loc[:, FEATURE_SETS[model]].join(effects)


def walk_forward(
    pooled: pd.DataFrame,
    direction: str,
    *,
    min_train_dates: int,
    refit_every: int,
    c_value: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = _direction_frame(pooled, direction, target_mode="PRE_PIVOT")
    complete = frame.loc[:, ALL_FEATURES].notna().all(axis=1)
    scorable = frame["at_risk"].astype(bool) & complete
    levels = tuple(sorted(frame["sector"].unique()))
    dates = pd.DatetimeIndex(frame.loc[scorable, "date"].drop_duplicates().sort_values())
    predictions: list[pd.DataFrame] = []
    fitted = skipped_history = skipped_class = 0
    output_columns = [
        "date", "_position", "sector", "sector_name", "state", "pivot_position",
        "pivot_date", "confirmation_position", "confirmation_date", "barrier",
        "cooldown_remaining", "event", "at_risk", "target", "label_end_position",
        "label_end_date", "target_event_pivot_position", "target_event_pivot_date",
        "target_event_confirmation_position", "target_event_confirmation_date",
        "macro_asof_date", "relative_price",
    ]
    for start in range(0, len(dates), refit_every):
        block_dates = dates[start : start + refit_every]
        prediction_date = pd.Timestamp(block_dates[0])
        train = frame.loc[purged_training_mask(frame, prediction_date) & complete]
        if train["date"].nunique() < min_train_dates:
            skipped_history += 1
            continue
        if train["target"].nunique() < 2:
            skipped_class += 1
            continue
        test = frame.loc[scorable & frame["date"].isin(block_dates)]
        if test.empty:
            continue
        scaler = StandardScaler().fit(train.loc[:, ALL_FEATURES])
        train_scaled = scaler.transform(train.loc[:, ALL_FEATURES])
        test_scaled = scaler.transform(test.loc[:, ALL_FEATURES])
        for model_name in MODELS:
            model = LogisticRegression(
                C=c_value, class_weight="balanced", max_iter=2_000, random_state=42
            ).fit(_design(train, train_scaled, model_name, levels), train["target"].astype(int))
            rows = test[output_columns].copy()
            rows.insert(0, "direction", direction)
            rows.insert(1, "model", model_name)
            rows.insert(2, "target_mode", "PRE_PIVOT")
            rows["refit_date"] = prediction_date
            rows["training_start_date"] = train["date"].min()
            rows["training_end_date"] = train["date"].max()
            rows["training_date_count"] = train["date"].nunique()
            rows["score"] = model.predict_proba(
                _design(test, test_scaled, model_name, levels)
            )[:, 1]
            predictions.append(rows)
        fitted += 1
    output = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    if not output.empty:
        output = output.sort_values(["model", "sector", "date"]).reset_index(drop=True)
        output["raw_score"] = output["score"]
        output["score"] = output.groupby(["model", "sector"], sort=False)["raw_score"].transform(
            lambda values: causal_trailing_percentile(values, window=756, min_history=60)
        )
        output = output.dropna(subset=["score"]).reset_index(drop=True)
        output["evaluation_sample"] = output["date"].ge(EVALUATION_START)
    return output, {
        "status": "OK" if not output.empty else "INSUFFICIENT_HISTORY",
        "fitted_blocks": fitted,
        "skipped_history_blocks": skipped_history,
        "skipped_one_class_blocks": skipped_class,
        "scorable_dates": int(len(dates)),
    }


def _format_metric(value: Any, spec: str) -> str:
    return "—" if value is None or not np.isfinite(float(value)) else format(float(value), spec)


def report(summary: dict[str, Any]) -> str:
    lines = [
        "# Sector Beta + Breadth PRE_PIVOT Comparison",
        "",
        f"- 판정: **{summary['verdict']}**",
        f"- 표본: {summary['audit']['paired_dates']:,}일 × {len(summary['audit']['groups'])}개 그룹",
        "- 베타: 전일까지의 126일 베타와 756일 구조적 베타를 126일 R²로 혼합",
        "",
        "| 방향 | 모델 | ROC AUC | 양성/표본 | Pivot 전 재현 | False/sector-yr |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for direction, result in summary["directions"].items():
        for model, metric in result["models"].items():
            event = metric["early_event_metrics"]
            lines.append(
                f"| {direction} | {model} | {metric['roc_auc']:.3f} | "
                f"{metric['positives']}/{metric['n']} | {event['pre_pivot_recall']:.1%} | "
                f"{event['false_alert_clusters_per_sector_year']:.2f} |"
            )
        lines.extend(["", "| 후보 | Δ ROC AUC (95% CI) | Δ Pivot 전 재현 (95% CI) |", "|---|---:|---:|"])
        for model, uplift in result["uplifts"].items():
            lines.append(
                f"| {model} | {_format_metric(uplift['delta_auc'], '+.3f')} "
                f"[{_format_metric(uplift['ci_low'], '+.3f')}, {_format_metric(uplift['ci_high'], '+.3f')}] | "
                f"{_format_metric(uplift['delta_pre_pivot_recall'], '+.1%')} "
                f"[{_format_metric(uplift['early_capture_ci_low'], '+.1%')}, "
                f"{_format_metric(uplift['early_capture_ci_high'], '+.1%')}] |"
            )
        lines.append("")
    lines.append("PIT가 부분 검증 상태이므로 운영 승격 없이 연구용으로만 해석합니다.")
    return "\n".join(lines)


def self_test() -> None:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2020-01-01", periods=900)
    market_return = pd.Series(rng.normal(0, 0.01, len(dates)), index=dates)
    prices = pd.DataFrame(
        {
            "SPX": (1 + market_return).cumprod(),
            "2010": (1 + 2 * market_return).cumprod(),
        },
        index=dates,
    )
    original = causal_beta_features(prices, ["2010"])
    assert abs(float(original["beta_used_lagged"].dropna().iloc[-1]) - 2) < 0.02
    assert abs(float(original["beta_adjusted_return_20d"].dropna().iloc[-1])) < 0.002
    changed = prices.copy()
    changed.iloc[-10:, :] *= np.linspace(1, 1.2, 10)[:, None]
    revised = causal_beta_features(changed, ["2010"])
    cutoff = dates[-11]
    left = original.loc[original["date"].le(cutoff), "beta_used_lagged"]
    right = revised.loc[revised["date"].le(cutoff), "beta_used_lagged"]
    assert np.allclose(left, right, equal_nan=True)
    assert _format_metric(None, "+.3f") == "—"
    print("self-test: PASS")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Research-only causal beta/breadth PRE_PIVOT comparison")
    result.add_argument("--gics-package-dir", type=Path)
    result.add_argument("--breadth-features", type=Path)
    result.add_argument("--output-dir", type=Path)
    result.add_argument("--horizon", type=int, default=10)
    result.add_argument("--min-train-dates", type=int, default=756)
    result.add_argument("--refit-every", type=int, default=63)
    result.add_argument("--c-value", type=float, default=0.1)
    result.add_argument("--bootstrap-samples", type=int, default=1_000)
    result.add_argument("--self-test", action="store_true")
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    pooled, beta, audit = merge_inputs(
        args.gics_package_dir.resolve(), args.breadth_features.resolve(), args.horizon
    )
    prediction_parts: list[pd.DataFrame] = []
    directions: dict[str, Any] = {}
    calendars: list[pd.DatetimeIndex] = []
    for direction in ("ROTATION_OUT", "ROTATION_IN"):
        predictions, walk = walk_forward(
            pooled, direction, min_train_dates=args.min_train_dates,
            refit_every=args.refit_every, c_value=args.c_value,
        )
        prediction_parts.append(predictions)
        evaluated = predictions.loc[predictions["evaluation_sample"] & predictions["target"].notna()].copy()
        calendar = pd.DatetimeIndex(evaluated["date"].drop_duplicates().sort_values())
        calendars.append(calendar)
        models = {
            model: _model_metrics(
                evaluated.loc[evaluated["model"].eq(model)], pooled, direction,
                horizon=args.horizon, cutoff=CUTOFF,
            )
            for model in MODELS
        }
        uplifts = _paired_uplifts(
            evaluated, bootstrap_samples=args.bootstrap_samples, calendar_dates=calendar,
            horizon=args.horizon, cutoff=CUTOFF, models=MODELS,
        )
        directions[direction] = {"walk_forward": walk, "models": models, "uplifts": uplifts}
    predictions = pd.concat(prediction_parts, ignore_index=True)
    evaluated_predictions = predictions.loc[predictions["evaluation_sample"]].copy()
    economics = economic_rows(evaluated_predictions, pooled, args.horizon)
    calendar = pd.DatetimeIndex(sorted(set().union(*(item.tolist() for item in calendars))))
    economic_summary = summarize_economics(economics, calendar, args.bootstrap_samples)
    significant = any(
        uplift.get("statistical_pass") or uplift.get("auc_statistical_pass")
        for result in directions.values() for uplift in result["uplifts"].values()
    )
    summary = {
        "verdict": "STATISTICAL_UPLIFT_FOUND_RESEARCH_ONLY" if significant else "NO_RELIABLE_UPLIFT",
        "pit_status": "PIT_PARTIAL/CURRENT_REVISED_HISTORY_ONLY",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "spec": {
            "models": list(MODELS), "feature_sets": FEATURE_SETS,
            "evaluation_start": str(EVALUATION_START.date()), "cutoff": CUTOFF,
            "horizon": args.horizon, "min_train_dates": args.min_train_dates,
            "refit_every": args.refit_every, "bootstrap_samples": args.bootstrap_samples,
            "economic_entry": "next-session close",
        },
        "audit": audit, "directions": directions, "economics": economic_summary,
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    beta.to_csv(output_dir / "sector_beta_history.csv", index=False)
    latest = beta.sort_values("date").groupby("sector", as_index=False).tail(1)
    latest.to_csv(output_dir / "latest_sector_beta_moves.csv", index=False)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    economics.to_csv(output_dir / "economic_clusters.csv", index=False)
    clean = _json_ready(summary)
    (output_dir / "summary.json").write_text(
        json.dumps(clean, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    (output_dir / "REPORT.md").write_text(report(clean), encoding="utf-8")
    return clean


def main() -> None:
    args = parser().parse_args()
    if args.self_test:
        self_test()
        return
    if not all((args.gics_package_dir, args.breadth_features, args.output_dir)):
        raise ValueError("--gics-package-dir, --breadth-features, and --output-dir are required")
    result = run(args)
    print(json.dumps({"verdict": result["verdict"], "groups": len(result["audit"]["groups"])}))


if __name__ == "__main__":
    main()
