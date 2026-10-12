"""Recall-first, recent-window comparison for sector PRE_PIVOT alerts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from research_drawdown_onset import causal_trailing_percentile
from sector_turning_beta_breadth_shadow import BETA_FEATURES, BREADTH_FEATURES, merge_inputs
from sector_turning_pooled_shadow import (
    DIRECTIONS,
    PRICE_FEATURES,
    _direction_frame,
    _early_event_metrics,
    _wilson_interval,
    effect_code,
    paired_date_block_pre_pivot_recall_delta_ci,
    purged_training_mask,
)
from sector_turning_tradable_label_shadow import qualify_labels


BASELINE = "L2_BASELINE"
CANDIDATES = (
    "ELASTIC_NET",
    "SHALLOW_BOOST",
    "BREADTH_SHALLOW_BOOST",
    "BETA_BREADTH_SHALLOW_BOOST",
    "DURATION_LOGIT",
)
MODELS = (BASELINE, *CANDIDATES)
DURATION_FEATURES = ("log_state_age", "log_sessions_since_extreme", "barrier")
ALERT_BUDGETS = (0.20, 0.30, 0.40)


def add_duration_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.sort_values(["sector", "_position"]).copy()
    new_run = ~(
        result["sector"].eq(result["sector"].shift())
        & result["state"].eq(result["state"].shift())
    )
    state_age = result.groupby(new_run.cumsum(), sort=False).cumcount() + 1
    since_extreme = (result["_position"] - result["pivot_position"]).clip(lower=0)
    result["log_state_age"] = np.log1p(state_age.astype(float))
    result["log_sessions_since_extreme"] = np.log1p(since_extreme)
    return result.sort_index()


def feature_columns(model_name: str) -> tuple[str, ...]:
    if model_name == "DURATION_LOGIT":
        return (*PRICE_FEATURES, *DURATION_FEATURES)
    if model_name == "BREADTH_SHALLOW_BOOST":
        return (*PRICE_FEATURES, *BREADTH_FEATURES)
    if model_name == "BETA_BREADTH_SHALLOW_BOOST":
        return (*PRICE_FEATURES, *BETA_FEATURES, *BREADTH_FEATURES)
    return PRICE_FEATURES


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


def estimator(model_name: str):
    if model_name == "ELASTIC_NET":
        return LogisticRegression(
            solver="saga", l1_ratio=0.5, C=0.1,
            class_weight="balanced", max_iter=4_000, tol=1e-3, random_state=42,
        )
    if model_name in {"SHALLOW_BOOST", "BREADTH_SHALLOW_BOOST", "BETA_BREADTH_SHALLOW_BOOST"}:
        return HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=120, max_depth=3, min_samples_leaf=80,
            l2_regularization=1.0, class_weight="balanced", early_stopping=False,
            random_state=42,
        )
    if model_name in {BASELINE, "DURATION_LOGIT"}:
        return LogisticRegression(
            C=0.1, class_weight="balanced", max_iter=2_000, random_state=42
        )
    raise ValueError(f"unknown model: {model_name}")


def walk_forward(
    pooled: pd.DataFrame,
    direction: str,
    *,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    min_train_dates: int,
    refit_every: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = add_duration_features(_direction_frame(pooled, direction, target_mode="PRE_PIVOT"))
    frame = frame.loc[frame["date"].between(start_date, end_date)].copy()
    levels = tuple(sorted(frame["sector"].astype(str).unique()))
    dates = pd.DatetimeIndex(frame["date"].drop_duplicates().sort_values())
    rows: list[pd.DataFrame] = []
    fitted = {name: 0 for name in MODELS}

    for block_start in range(0, len(dates), refit_every):
        block_dates = dates[block_start : block_start + refit_every]
        prediction_date = pd.Timestamp(block_dates[0])
        for model_name in MODELS:
            columns = feature_columns(model_name)
            complete = frame.loc[:, columns].notna().all(axis=1)
            train = frame.loc[purged_training_mask(frame, prediction_date) & complete]
            if train["date"].nunique() < min_train_dates or train["target"].nunique() < 2:
                continue
            test = frame.loc[
                frame["at_risk"].astype(bool) & complete & frame["date"].isin(block_dates)
            ]
            if test.empty:
                continue
            scaler = StandardScaler().fit(train.loc[:, columns])
            train_design = design_matrix(train, columns, scaler, levels)
            test_design = design_matrix(test, columns, scaler, levels)
            model = estimator(model_name).fit(train_design, train["target"].astype(int))
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
        predictions["score"] = predictions.groupby(["model", "sector"], sort=False)[
            "raw_score"
        ].transform(
            lambda values: causal_trailing_percentile(values, window=504, min_history=60)
        )
        predictions = predictions.dropna(subset=["score"]).reset_index(drop=True)
    return predictions, {
        "direction": direction,
        "window_dates": int(len(dates)),
        "first_date": str(dates.min().date()),
        "last_date": str(dates.max().date()),
        "fitted_blocks": fitted,
    }


def model_metrics(rows: pd.DataFrame, direction: str) -> dict[str, Any]:
    evaluated = rows.loc[rows["target"].notna()].copy()
    target = evaluated["target"].astype(int)
    result: dict[str, Any] = {
        "n": int(len(evaluated)),
        "positives": int(target.sum()),
        "roc_auc": float(roc_auc_score(target, evaluated["score"])) if target.nunique() == 2 else None,
        "pr_auc": float(average_precision_score(target, evaluated["score"])) if target.nunique() == 2 else None,
        "budgets": {},
    }
    for budget in ALERT_BUDGETS:
        cutoff = 1.0 - budget
        event = _early_event_metrics(
            evaluated, pd.DataFrame(), direction, horizon=10, cutoff=cutoff
        )
        low, high = _wilson_interval(event["pre_pivot_hits"], event["eligible_confirmations"])
        result["budgets"][f"{int(budget * 100)}pct"] = {
            **event,
            "pre_pivot_recall_ci_low": low,
            "pre_pivot_recall_ci_high": high,
        }
    return result


def winner_table(directions: dict[str, Any], budget: str = "30pct") -> list[dict[str, Any]]:
    table = []
    for model_name in MODELS:
        by_direction = [
            directions[direction]["models"][model_name]["budgets"][budget]
            for direction in DIRECTIONS
        ]
        table.append(
            {
                "model": model_name,
                "mean_pre_pivot_recall": float(np.mean([row["pre_pivot_recall"] for row in by_direction])),
                "mean_recall_ci_low": float(np.mean([row["pre_pivot_recall_ci_low"] for row in by_direction])),
                "mean_alert_day_rate": float(np.mean([row["alert_day_rate"] for row in by_direction])),
            }
        )
    return sorted(table, key=lambda row: (row["mean_recall_ci_low"], row["mean_pre_pivot_recall"]), reverse=True)


def recall_uplifts(predictions: pd.DataFrame) -> dict[str, Any]:
    baseline = predictions.loc[predictions["model"].eq(BASELINE)]
    calendar = pd.DatetimeIndex(predictions["date"].drop_duplicates().sort_values())
    result: dict[str, Any] = {}
    for model_name in CANDIDATES:
        candidate = predictions.loc[predictions["model"].eq(model_name)]
        result[model_name] = {}
        for budget in ALERT_BUDGETS:
            result[model_name][f"{int(budget * 100)}pct"] = (
                paired_date_block_pre_pivot_recall_delta_ci(
                    candidate,
                    baseline,
                    samples=500,
                    block_dates=40,
                    calendar_dates=calendar,
                    horizon=10,
                    cutoff=1.0 - budget,
                )
            )
    return result


def markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Recent 3-Year Recall-First PRE_PIVOT Model Comparison",
        "",
        f"- 기간: {summary['spec']['start']} ~ {summary['spec']['end']}",
        "- 동일 표본: 가격 기준모델에 Duration·Breadth·베타/Breadth 후보를 각각 추가",
        "- 평가: purged monthly walk-forward, causal per-sector score percentile",
        "- PIT: 연구용 CURRENT_REVISED_HISTORY_ONLY",
        "",
        "| Direction | Model | AUC | PR AUC | Budget | Recall | 95% low | Lead | Alert days | False clusters/y |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for direction in DIRECTIONS:
        for model_name, metric in summary["directions"][direction]["models"].items():
            for budget, event in metric["budgets"].items():
                lines.append(
                    f"| {direction} | {model_name} | {metric['roc_auc']:.3f} | {metric['pr_auc']:.3f} | "
                    f"{budget} | {event['pre_pivot_recall']:.1%} | {event['pre_pivot_recall_ci_low']:.1%} | "
                    f"{event['median_lead_to_pivot_sessions'] or 0:.1f} | {event['alert_day_rate']:.1%} | "
                    f"{event['false_alert_clusters_per_sector_year']:.2f} |"
                )
    lines.extend(["", "## 30% 경보예산 순위", ""])
    for index, row in enumerate(summary["ranking_30pct"], 1):
        lines.append(
            f"{index}. {row['model']}: 평균 사전포착률 {row['mean_pre_pivot_recall']:.1%}, "
            f"평균 95% 하한 {row['mean_recall_ci_low']:.1%}"
        )
    lines.extend(["", "## L2 기준모델 대비 포착률 개선폭", ""])
    for direction in DIRECTIONS:
        for model_name, budgets in summary["directions"][direction]["recall_uplift_vs_baseline"].items():
            for budget in ("30pct", "40pct"):
                uplift = budgets[budget]
                interval = (
                    "95% CI 표본 부족"
                    if not np.isfinite(uplift["early_capture_ci_low"])
                    else f"[{uplift['early_capture_ci_low']:+.1%}, {uplift['early_capture_ci_high']:+.1%}]"
                )
                lines.append(
                    f"- {direction} · {model_name} · {budget}: "
                    f"{uplift['delta_pre_pivot_recall']:+.1%} {interval}"
                )
    lines.extend(["", "## 상승·하락 전환 합산 개선폭", ""])
    for model_name, budgets in summary["combined_recall_uplift_vs_baseline"].items():
        for budget in ("30pct", "40pct"):
            uplift = budgets[budget]
            interval = (
                "95% CI 표본 부족"
                if not np.isfinite(uplift["early_capture_ci_low"])
                else f"[{uplift['early_capture_ci_low']:+.1%}, {uplift['early_capture_ci_high']:+.1%}]"
            )
            lines.append(
                f"- {model_name} · {budget}: "
                f"{uplift['delta_pre_pivot_recall']:+.1%} {interval}"
            )
    lines.extend([
        "",
        "오경보는 탈락 기준이 아니라 보고값입니다. 단, 매일 경보하는 무의미한 해법을 막기 위해 경보예산별로 비교했습니다.",
    ])
    return "\n".join(lines)


def self_test() -> None:
    frame = pd.DataFrame(
        {
            "sector": ["A"] * 5,
            "_position": range(5),
            "state": ["UP_ACTIVE", "UP_ACTIVE", "DOWN_ACTIVE", "DOWN_ACTIVE", "DOWN_ACTIVE"],
            "pivot_position": [0, 1, 2, 2, 3],
            "barrier": 0.05,
        }
    )
    result = add_duration_features(frame)
    assert np.allclose(np.expm1(result["log_state_age"]), [1, 2, 1, 2, 3])
    assert np.allclose(np.expm1(result["log_sessions_since_extreme"]), [0, 0, 0, 1, 1])
    changed = frame.copy()
    changed.loc[4, "state"] = "UP_ACTIVE"
    assert result.loc[:3, DURATION_FEATURES].equals(add_duration_features(changed).loc[:3, DURATION_FEATURES])
    assert feature_columns("BREADTH_SHALLOW_BOOST") == (*PRICE_FEATURES, *BREADTH_FEATURES)
    assert feature_columns("BETA_BREADTH_SHALLOW_BOOST") == (*PRICE_FEATURES, *BETA_FEATURES, *BREADTH_FEATURES)


def run(args: argparse.Namespace) -> dict[str, Any]:
    pooled, _, input_audit = merge_inputs(
        args.gics_package_dir.resolve(), args.breadth_features.resolve(), horizon=10
    )
    start_date, end_date = pd.Timestamp(args.start), pd.Timestamp(args.end)
    if start_date >= end_date:
        raise ValueError("start must be earlier than end")
    directions: dict[str, Any] = {}
    all_predictions = []
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
        all_predictions.append(predictions)
        directions[direction] = {
            "label_audit": label_audit,
            "walk_forward": walk,
            "models": {
                name: model_metrics(predictions.loc[predictions["model"].eq(name)], direction)
                for name in MODELS
            },
            "recall_uplift_vs_baseline": recall_uplifts(predictions),
        }
    combined_predictions = pd.concat(all_predictions, ignore_index=True)
    combined_events = combined_predictions.copy()
    combined_events["sector"] = combined_events["direction"] + "|" + combined_events["sector"]
    summary = {
        "verdict": "RESEARCH_SHADOW_ONLY",
        "pit_status": "PIT_PARTIAL/CURRENT_REVISED_HISTORY_ONLY",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "spec": {
            "start": str(start_date.date()),
            "end": str(end_date.date()),
            "min_train_dates": args.min_train_dates,
            "refit_every": args.refit_every,
            "score_percentile_history": 504,
            "score_percentile_min_history": 60,
            "alert_budgets": list(ALERT_BUDGETS),
            "baseline": BASELINE,
            "candidate_models": list(CANDIDATES),
            "price_features": list(PRICE_FEATURES),
            "breadth_features": list(BREADTH_FEATURES),
            "beta_features": list(BETA_FEATURES),
            "duration_features": list(DURATION_FEATURES),
        },
        "input_audit": input_audit,
        "directions": directions,
        "combined_recall_uplift_vs_baseline": recall_uplifts(combined_events),
    }
    summary["ranking_30pct"] = winner_table(directions)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    combined_predictions.to_csv(output / "predictions.csv", index=False)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "REPORT.md").write_text(markdown_report(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gics-package-dir", type=Path)
    parser.add_argument("--breadth-features", type=Path)
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
    if not all((args.gics_package_dir, args.breadth_features, args.output_dir)):
        parser.error("--gics-package-dir, --breadth-features, and --output-dir are required")
    summary = run(args)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "ranking_30pct": summary["ranking_30pct"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
