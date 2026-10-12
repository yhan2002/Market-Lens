"""Compare core price features with causal multiscale and curvature features."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from research_drawdown_onset import causal_trailing_percentile
from sector_macro_trend_hint_experiment import PACKAGE
from sector_turning_breadth_shadow import block_mean_ci
from sector_turning_layer_weight_experiment import _json_ready
from sector_turning_model_policy_scan import forward_returns
from sector_turning_pooled_shadow import (
    DIRECTIONS,
    PRICE_FEATURES,
    _direction_frame,
    _early_event_metrics,
    _wilson_interval,
    build_pooled_daily_frame,
    effect_code,
    paired_date_block_auc_delta_ci,
    paired_date_block_pre_pivot_recall_delta_ci,
    purged_training_mask,
)
from sector_turning_recall_model_compare import estimator
from sector_turning_tradable_label_shadow import qualify_labels
from sector_turning_two_stage_confirmation_experiment import cluster_rows


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs" / "sector_turning_price_multiscale_20260904"
BASELINE = "PRICE_CORE"
HORIZON_FEATURES = (
    "relative_return_1", "relative_return_3", "relative_return_10",
    "relative_return_40", "relative_return_60", "relative_return_120",
    "relative_drawdown_20", "relative_drawdown_120",
    "relative_rebound_20", "relative_rebound_60", "relative_vol_ratio_5_20",
)
CURVATURE_FEATURES = (
    "relative_velocity_spread_3_20", "relative_velocity_spread_10_40",
    "relative_acceleration_3", "relative_acceleration_10",
    "relative_range_position_20", "relative_range_position_60",
)
FEATURE_SETS = {
    BASELINE: PRICE_FEATURES,
    "PRICE_HORIZONS": (*PRICE_FEATURES, *HORIZON_FEATURES),
    "PRICE_CURVATURE": (*PRICE_FEATURES, *CURVATURE_FEATURES),
    "PRICE_MULTISCALE": (*PRICE_FEATURES, *HORIZON_FEATURES, *CURVATURE_FEATURES),
}
ALL_FEATURES = tuple(dict.fromkeys(column for columns in FEATURE_SETS.values() for column in columns))
BUDGETS = (0.20, 0.30, 0.40)
WINDOWS = {
    "selection_2021": (pd.Timestamp("2021-01-04"), pd.Timestamp("2021-12-31")),
    "selection_2022": (pd.Timestamp("2022-01-03"), pd.Timestamp("2022-12-30")),
    "evaluation_2023_2026": (pd.Timestamp("2023-01-03"), pd.Timestamp.max),
}


def add_price_path_features(pooled: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, rows in pooled.groupby("sector", sort=False):
        rows = rows.sort_values("_position").copy()
        price = rows["relative_price"].astype(float)
        log_price = np.log(price)
        one_day = price.pct_change(fill_method=None)
        for horizon in (1, 3, 10, 40, 60, 120):
            rows[f"relative_return_{horizon}"] = price.pct_change(horizon, fill_method=None)
        for horizon in (20, 120):
            rows[f"relative_drawdown_{horizon}"] = price / price.rolling(horizon).max() - 1
        for horizon in (20, 60):
            rows[f"relative_rebound_{horizon}"] = price / price.rolling(horizon).min() - 1
            low, high = price.rolling(horizon).min(), price.rolling(horizon).max()
            rows[f"relative_range_position_{horizon}"] = (price - low).div(
                (high - low).where((high - low).abs() > 1e-12)
            )
        velocity = {horizon: log_price.diff(horizon) / horizon for horizon in (3, 10, 20, 40)}
        rows["relative_velocity_spread_3_20"] = velocity[3] - velocity[20]
        rows["relative_velocity_spread_10_40"] = velocity[10] - velocity[40]
        rows["relative_acceleration_3"] = velocity[3] - velocity[3].shift(3)
        rows["relative_acceleration_10"] = velocity[10] - velocity[10].shift(10)
        vol_5 = one_day.rolling(5).std()
        vol_20 = one_day.rolling(20).std()
        rows["relative_vol_ratio_5_20"] = vol_5.div(vol_20.where(vol_20 > 1e-12))
        parts.append(rows)
    return pd.concat(parts, ignore_index=True).sort_values(["date", "sector"])


def design(frame: pd.DataFrame, columns: tuple[str, ...], scaler: StandardScaler, levels: tuple[str, ...]) -> pd.DataFrame:
    continuous = pd.DataFrame(
        scaler.transform(frame.loc[:, columns]), index=frame.index, columns=columns
    )
    effects, _ = effect_code(frame["sector"], levels)
    return continuous.join(effects)


def walk_forward(
    pooled: pd.DataFrame,
    direction: str,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    min_train_dates: int,
    refit_every: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = _direction_frame(pooled, direction, target_mode="PRE_PIVOT")
    complete = frame.loc[:, ALL_FEATURES].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    levels = tuple(sorted(frame["sector"].astype(str).unique()))
    dates = pd.DatetimeIndex(
        frame.loc[frame["at_risk"].astype(bool) & complete & frame["date"].between(start, end), "date"]
        .drop_duplicates().sort_values()
    )
    output: list[pd.DataFrame] = []
    fitted = {name: 0 for name in FEATURE_SETS}
    for block_start in range(0, len(dates), refit_every):
        block_dates = dates[block_start : block_start + refit_every]
        prediction_date = pd.Timestamp(block_dates[0])
        train = frame.loc[purged_training_mask(frame, prediction_date) & complete]
        test = frame.loc[
            frame["at_risk"].astype(bool) & complete & frame["date"].isin(block_dates)
        ]
        if train["date"].nunique() < min_train_dates or train["target"].nunique() < 2 or test.empty:
            continue
        for model_name, columns in FEATURE_SETS.items():
            scaler = StandardScaler().fit(train.loc[:, columns])
            model = estimator("SHALLOW_BOOST").fit(
                design(train, columns, scaler, levels), train["target"].astype(int)
            )
            scored = test[[
                "date", "_position", "sector", "sector_name", "target",
                "target_event_pivot_position", "target_event_confirmation_position",
                "target_event_confirmation_date",
            ]].copy()
            scored.insert(0, "direction", direction)
            scored.insert(1, "model", model_name)
            scored["refit_date"] = prediction_date
            scored["training_start_date"] = train["date"].min()
            scored["training_end_date"] = train["date"].max()
            scored["latest_training_label_end_date"] = train["label_end_date"].max()
            scored["raw_score"] = model.predict_proba(design(test, columns, scaler, levels))[:, 1]
            output.append(scored)
            fitted[model_name] += 1
    predictions = pd.concat(output, ignore_index=True).sort_values(["model", "sector", "date"])
    predictions["score"] = predictions.groupby(["model", "sector"], sort=False)["raw_score"].transform(
        lambda values: causal_trailing_percentile(values, window=504, min_history=60)
    )
    return predictions.dropna(subset=["score"]).reset_index(drop=True), {
        "direction": direction, "dates": int(len(dates)), "fitted_blocks": fitted,
    }


def window(rows: pd.DataFrame, name: str) -> pd.DataFrame:
    start, end = WINDOWS[name]
    return rows.loc[rows["date"].between(start, end) & rows["target"].notna()].copy()


def economic_metrics(rows: pd.DataFrame, cutoff: float, samples: int) -> dict[str, Any]:
    alerts = rows["score"].ge(cutoff)
    clusters = cluster_rows(rows, alerts).dropna(subset=["signed_forward_return_20"])
    calendar = pd.DatetimeIndex(rows["date"].drop_duplicates().sort_values())
    low, high = block_mean_ci(
        clusters.rename(columns={"date": "signal_date"}), calendar,
        "signed_forward_return_20", samples=samples,
    )
    return {
        "clusters": int(len(clusters)),
        "mean_signed_relative_return_20d": (
            float(clusters["signed_forward_return_20"].mean()) if len(clusters) else None
        ),
        "ci_low": float(low) if np.isfinite(low) else None,
        "ci_high": float(high) if np.isfinite(high) else None,
    }


def metrics(rows: pd.DataFrame, samples: int) -> dict[str, Any]:
    target = rows["target"].astype(int)
    result: dict[str, Any] = {
        "rows": int(len(rows)), "positives": int(target.sum()),
        "roc_auc": float(roc_auc_score(target, rows["score"])),
        "pr_auc": float(average_precision_score(target, rows["score"])),
        "budgets": {},
    }
    for budget in BUDGETS:
        cutoff = 1 - budget
        event = _early_event_metrics(rows, pd.DataFrame(), "", horizon=10, cutoff=cutoff)
        low, high = _wilson_interval(event["pre_pivot_hits"], event["eligible_confirmations"])
        result["budgets"][f"{int(budget * 100)}pct"] = {
            **event, "pre_pivot_recall_ci_low": low, "pre_pivot_recall_ci_high": high,
            "economics": economic_metrics(rows, cutoff, samples),
        }
    return result


def choose_model(predictions: pd.DataFrame, samples: int) -> tuple[str, dict[str, Any]]:
    records = []
    selection: dict[str, Any] = {}
    for model in FEATURE_SETS:
        selection[model] = {
            name: metrics(window(predictions.loc[predictions["model"].eq(model)], name), samples)
            for name in ("selection_2021", "selection_2022")
        }
        records.append({
            "model": model,
            "worst_recall": min(
                selection[model][name]["budgets"]["30pct"]["pre_pivot_recall"]
                for name in ("selection_2021", "selection_2022")
            ),
            "mean_recall": np.mean([
                selection[model][name]["budgets"]["30pct"]["pre_pivot_recall"]
                for name in ("selection_2021", "selection_2022")
            ]),
            "mean_auc": np.mean([
                selection[model][name]["roc_auc"] for name in ("selection_2021", "selection_2022")
            ]),
        })
    chosen = pd.DataFrame(records).sort_values(
        ["worst_recall", "mean_recall", "mean_auc"], ascending=False
    ).iloc[0]["model"]
    return str(chosen), selection


def compare(
    predictions: pd.DataFrame, candidate: str, samples: int
) -> dict[str, Any]:
    evaluated = window(predictions, "evaluation_2023_2026")
    candidate_rows = evaluated.loc[evaluated["model"].eq(candidate)]
    baseline_rows = evaluated.loc[evaluated["model"].eq(BASELINE)]
    comparisons: dict[str, Any] = {
        "auc_delta": paired_date_block_auc_delta_ci(
            candidate_rows.set_index(["date", "sector"])["target"],
            candidate_rows.set_index(["date", "sector"])["score"],
            baseline_rows.set_index(["date", "sector"])["score"],
            samples=samples, block_dates=40,
            calendar_dates=candidate_rows["date"].drop_duplicates(),
        )
    }
    for budget in (0.30, 0.40):
        comparisons[f"recall_delta_{int(budget * 100)}pct"] = (
            paired_date_block_pre_pivot_recall_delta_ci(
                candidate_rows, baseline_rows, samples=samples, block_dates=40,
                calendar_dates=candidate_rows["date"].drop_duplicates(),
                horizon=10, cutoff=1 - budget,
            )
        )
    return comparisons


def report(summary: dict[str, Any]) -> str:
    def percent(value: float | None) -> str:
        return "N/A" if value is None else f"{value:+.1%}"

    lines = [
        "# Price multiscale turning experiment", "",
        f"Verdict: **{summary['verdict']}**", "",
        "기존 가격 6개 변수와 다기간 가격 경로·곡률·감속 변수를 동일한 모델과 동일한 표본에서 비교했습니다.",
        "모델 선택은 2021년과 2022년의 30% 경보예산 사전 포착률로 고정했고, 2023~2026년은 반복 검토된 연구 평가입니다.", "",
        "| 방향 | 선택 모델 | 30% 포착 | 기존 | 차이 95% CI | 40% 포착 | 기존 | AUC | 기존 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for direction, item in summary["directions"].items():
        selected, baseline = item["evaluation"]["selected"], item["evaluation"]["baseline"]
        delta = item["comparison"]["recall_delta_30pct"]
        lines.append(
            f"| {direction} | {item['selected_model']} | "
            f"{selected['budgets']['30pct']['pre_pivot_recall']:.1%} | "
            f"{baseline['budgets']['30pct']['pre_pivot_recall']:.1%} | "
            f"[{percent(delta['early_capture_ci_low'])}, {percent(delta['early_capture_ci_high'])}] | "
            f"{selected['budgets']['40pct']['pre_pivot_recall']:.1%} | "
            f"{baseline['budgets']['40pct']['pre_pivot_recall']:.1%} | "
            f"{selected['roc_auc']:.3f} | {baseline['roc_auc']:.3f} |"
        )
    combined = summary.get("combined_comparison")
    if combined:
        delta_30 = combined["recall_delta_30pct"]
        delta_40 = combined["recall_delta_40pct"]
        auc = combined["auc_delta"]
        lines.extend([
            "",
            "## 상승·하락 전환 결합 검정",
            "",
            f"- 30% 경보예산 포착률 개선: {delta_30['delta_pre_pivot_recall']:+.2%} "
            f"[95% CI {delta_30['early_capture_ci_low']:+.2%}, {delta_30['early_capture_ci_high']:+.2%}]",
            f"- 40% 경보예산 포착률 개선: {delta_40['delta_pre_pivot_recall']:+.2%} "
            f"[95% CI {delta_40['early_capture_ci_low']:+.2%}, {delta_40['early_capture_ci_high']:+.2%}]",
            f"- ROC AUC 개선: {auc['delta_auc']:+.3f} "
            f"[95% CI {auc['ci_low']:+.3f}, {auc['ci_high']:+.3f}]",
        ])
    lines.extend(["", "운영 반영 없음. PIT와 진정한 신규 OOS 검증 전에는 연구 신호로만 해석합니다."])
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    pooled, audit, _ = build_pooled_daily_frame(args.gics_package_dir.resolve(), None, horizon=10)
    pooled = add_price_path_features(pooled)
    predictions_parts, directions = [], {}
    for direction in DIRECTIONS:
        qualified, label_audit = qualify_labels(pooled, direction)
        predictions, walk = walk_forward(
            qualified, direction, start=pd.Timestamp(args.start), end=pd.Timestamp(args.end),
            min_train_dates=args.min_train_dates, refit_every=args.refit_every,
        )
        predictions_parts.append(predictions)
        directions[direction] = {"label_audit": label_audit, "walk_forward": walk}
    predictions = pd.concat(predictions_parts, ignore_index=True)
    returns = forward_returns()
    predictions = predictions.merge(
        returns, on=["date", "sector", "_position"], how="left", validate="many_to_one"
    )
    predictions["signed_forward_return_20"] = predictions["forward_relative_return_20"] * np.where(
        predictions["direction"].eq("ROTATION_IN"), 1, -1
    )
    verdicts = []
    for direction in DIRECTIONS:
        rows = predictions.loc[predictions["direction"].eq(direction)].copy()
        selected_model, selection = choose_model(rows, args.bootstrap_samples)
        evaluation = {
            model: metrics(
                window(rows.loc[rows["model"].eq(model)], "evaluation_2023_2026"),
                args.bootstrap_samples,
            )
            for model in FEATURE_SETS
        }
        comparison = compare(rows, selected_model, args.bootstrap_samples)
        selected_delta = comparison["recall_delta_30pct"]
        selected_event = evaluation[selected_model]["budgets"]["30pct"]
        statistical = (
            selected_model != BASELINE
            and (selected_delta["early_capture_ci_low"] or -np.inf) > 0
        )
        early = (
            selected_model != BASELINE
            and selected_event["pre_pivot_recall_ci_low"] > 0.05
            and (selected_event["median_lead_to_pivot_sessions"] or 0) >= 2
            and selected_delta["delta_pre_pivot_recall"] > 0
        )
        verdict = (
            "STATISTICAL_CAPTURE_UPLIFT_RESEARCH_ONLY" if statistical
            else "EARLY_SIGNAL_CANDIDATE_RESEARCH_ONLY" if early
            else "NO_RELIABLE_UPLIFT"
        )
        verdicts.append(verdict)
        directions[direction].update({
            "selected_model": selected_model, "selection": selection,
            "evaluation": {"selected": evaluation[selected_model], "baseline": evaluation[BASELINE]},
            "all_evaluation_models": evaluation, "comparison": comparison, "verdict": verdict,
        })
    verdict = (
        "STATISTICAL_CAPTURE_UPLIFT_RESEARCH_ONLY" if any(v.startswith("STATISTICAL") for v in verdicts)
        else "EARLY_SIGNAL_CANDIDATE_RESEARCH_ONLY" if any(v.startswith("EARLY") for v in verdicts)
        else "NO_RELIABLE_UPLIFT"
    )
    selected_models = {item["selected_model"] for item in directions.values()}
    combined_comparison = None
    if len(selected_models) == 1:
        combined = predictions.copy()
        combined["sector"] = combined["direction"] + "|" + combined["sector"].astype(str)
        combined_comparison = compare(combined, selected_models.pop(), args.bootstrap_samples)
        if (combined_comparison["recall_delta_30pct"]["early_capture_ci_low"] or -np.inf) > 0:
            verdict = "STATISTICAL_CAPTURE_UPLIFT_RESEARCH_ONLY"
    summary = _json_ready({
        "verdict": verdict, "pit_status": "PIT_PARTIAL/CURRENT_REVISED_HISTORY_ONLY",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective": "predict sector trend exhaustion using price path only",
        "models": {name: list(columns) for name, columns in FEATURE_SETS.items()},
        "selection": "2021 and 2022 separately; reviewed evaluation 2023-2026",
        "input_audit": audit, "directions": directions,
        "combined_comparison": combined_comparison, "operational_changes": False,
    })
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    predictions.to_csv(output / "predictions.csv", index=False)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    (output / "REPORT.md").write_text(report(summary), encoding="utf-8")
    return summary


def self_test() -> None:
    dates = pd.bdate_range("2025-01-01", periods=140)
    base = np.exp(np.linspace(0, 0.2, len(dates)) + 0.01 * np.sin(np.arange(len(dates)) / 4))
    frame = pd.DataFrame({
        "date": dates, "_position": range(len(dates)), "sector": "A", "relative_price": base,
    })
    result = add_price_path_features(frame)
    assert set((*HORIZON_FEATURES, *CURVATURE_FEATURES)).issubset(result.columns)
    changed = frame.copy(); changed.loc[len(changed) - 1, "relative_price"] *= 2
    rerun = add_price_path_features(changed)
    new_features = [*HORIZON_FEATURES, *CURVATURE_FEATURES]
    assert result.loc[:-2, new_features].equals(rerun.loc[:-2, new_features])
    assert result["relative_range_position_20"].dropna().between(0, 1).all()
    print("self-test: PASS")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--gics-package-dir", type=Path, default=PACKAGE)
    result.add_argument("--output-dir", type=Path, default=OUTPUT)
    result.add_argument("--start", default="2017-01-03")
    result.add_argument("--end", default="2026-09-03")
    result.add_argument("--min-train-dates", type=int, default=756)
    result.add_argument("--refit-every", type=int, default=63)
    result.add_argument("--bootstrap-samples", type=int, default=2_000)
    result.add_argument("--self-test", action="store_true")
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    if arguments.self_test:
        self_test()
    else:
        answer = run(arguments)
        print(json.dumps({"output": str(arguments.output_dir.resolve()), "verdict": answer["verdict"]}))
