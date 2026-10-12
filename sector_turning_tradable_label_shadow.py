"""Research-only test of economically qualified PRE_PIVOT labels."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sector_turning_beta_breadth_shadow import merge_inputs
from sector_turning_breadth_shadow import block_mean_ci, economic_rows, summarize_economics
from sector_turning_pooled_shadow import (
    EVALUATION_START,
    _direction_frame,
    _model_metrics,
    _wilson_interval,
    walk_forward_direction,
)


CUTOFF = 0.90
FOLLOWTHROUGH_HORIZON = 20
MINIMUM_FOLLOWTHROUGH = 0.03


def signed_followthrough(
    price: pd.Series, pivot_position: int, direction: str, horizon: int
) -> tuple[float, int] | None:
    exit_position = pivot_position + horizon
    if pivot_position not in price.index or exit_position not in price.index:
        return None
    raw = float(price.loc[exit_position] / price.loc[pivot_position] - 1)
    return (-raw if direction == "ROTATION_OUT" else raw), exit_position


def qualify_labels(
    pooled: pd.DataFrame,
    direction: str,
    *,
    horizon: int = FOLLOWTHROUGH_HORIZON,
    minimum_move: float = MINIMUM_FOLLOWTHROUGH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    prefix = "top" if direction == "ROTATION_OUT" else "bottom"
    target_column = f"{prefix}_target"
    end_position_column = f"{prefix}_label_end_position"
    end_date_column = f"{prefix}_label_end_date"
    mapped = _direction_frame(pooled, direction, target_mode="CONFIRMATION")
    result = pooled.copy()
    positive = mapped["target"].eq(1.0)
    prices = {
        str(sector): group.set_index("_position")["relative_price"].sort_index()
        for sector, group in pooled.groupby("sector")
    }
    position_dates = {
        str(sector): group.set_index("_position")["date"].sort_index()
        for sector, group in pooled.groupby("sector")
    }
    qualified_events: set[tuple[str, int]] = set()
    rejected_events: set[tuple[str, int]] = set()
    immature_events: set[tuple[str, int]] = set()
    for index, row in mapped.loc[positive].iterrows():
        sector = str(row["sector"])
        pivot_value = row["target_event_pivot_position"]
        if pd.isna(pivot_value):
            result.loc[index, [target_column, end_position_column, end_date_column]] = [
                np.nan, np.nan, pd.NaT
            ]
            continue
        pivot = int(pivot_value)
        outcome = signed_followthrough(prices[sector], pivot, direction, horizon)
        key = (sector, pivot)
        if outcome is None:
            immature_events.add(key)
            result.loc[index, [target_column, end_position_column, end_date_column]] = [
                np.nan, np.nan, pd.NaT
            ]
            continue
        signed_move, exit_position = outcome
        result.loc[index, target_column] = float(signed_move >= minimum_move)
        result.loc[index, end_position_column] = exit_position
        result.loc[index, end_date_column] = position_dates[sector].loc[exit_position]
        (qualified_events if signed_move >= minimum_move else rejected_events).add(key)
    return result, {
        "direction": direction,
        "minimum_followthrough": minimum_move,
        "followthrough_sessions_from_pivot": horizon,
        "qualified_events": len(qualified_events),
        "rejected_events": len(rejected_events),
        "immature_events": len(immature_events),
        "label_availability": "pivot plus 20 sessions; purged before model refit",
    }


def practical_gates(metric: dict[str, Any], economic: dict[str, Any]) -> dict[str, bool]:
    event = metric["early_event_metrics"]
    wilson_low, _ = _wilson_interval(event["pre_pivot_hits"], event["eligible_confirmations"])
    economic_pass = economic["ci_low"] is not None and economic["ci_low"] > 0
    early_pass = all(
        (
            event["eligible_confirmations"] >= 20,
            wilson_low is not None and wilson_low > 0.05,
            event["median_lead_to_pivot_sessions"] is not None
            and event["median_lead_to_pivot_sessions"] >= 3,
            event["false_alert_clusters_per_sector_year"] is not None
            and event["false_alert_clusters_per_sector_year"] <= 2,
            event["alert_day_rate"] <= 0.20,
        )
    )
    return {
        "economic_ci_above_zero": economic_pass,
        "early_signal_gate": early_pass,
        "pre_pivot_recall_ci_low": wilson_low,
    }


def self_test() -> None:
    dates = pd.bdate_range("2026-01-01", periods=35)
    price = pd.Series(np.r_[np.linspace(1, 1.1, 11), np.linspace(1.09, 0.9, 24)], index=range(35))
    pooled = pd.DataFrame(
        {
            "date": dates,
            "_position": range(35),
            "sector": "2010",
            "relative_price": price.to_numpy(),
            "event": ["TOP_END" if index == 15 else "" for index in range(35)],
            "pivot_position": [10 if index >= 15 else np.nan for index in range(35)],
            "pivot_date": [dates[10] if index >= 15 else pd.NaT for index in range(35)],
            "confirmation_position": [15 if index >= 15 else np.nan for index in range(35)],
            "confirmation_date": [dates[15] if index >= 15 else pd.NaT for index in range(35)],
            "top_target": [1.0 if 5 <= index < 15 else 0.0 for index in range(35)],
            "top_at_risk": True,
            "top_label_end_position": 15,
            "top_label_end_date": dates[15],
        }
    )
    gated, audit = qualify_labels(pooled, "ROTATION_OUT")
    assert audit["qualified_events"] == 1
    assert gated.loc[5:14, "top_target"].eq(1).all()
    assert gated.loc[5:14, "top_label_end_position"].eq(30).all()
    print("self-test: PASS")


def report(summary: dict[str, Any]) -> str:
    lines = [
        "# Economically Qualified PRE_PIVOT Label",
        "",
        f"- 판정: **{summary['verdict']}**",
        "- 의미 있는 전환: pivot 이후 20거래일 상대방향 3% 이상",
        "- 평가: 2020-01-01 이후 causal walk-forward, 경보 0.90",
        "",
        "| 방향 | 라벨 | ROC AUC | 20D 평균 (95% CI) | 사전 포착 | Pivot 선행 | False/sector-yr |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    economics = {
        (row["direction"], row["model"]): row
        for row in summary["economics"] if row["period"] == "ALL_OOS"
    }
    for direction, result in summary["directions"].items():
        for model, metric in result["models"].items():
            economic = economics[(direction, model)]
            event = metric["early_event_metrics"]
            lines.append(
                f"| {direction} | {model} | {metric['roc_auc']:.3f} | "
                f"{economic['mean_signed_relative_return_20d']:+.2%} "
                f"[{economic['ci_low']:+.2%}, {economic['ci_high']:+.2%}] | "
                f"{event['pre_pivot_recall']:.1%} | "
                f"{event['median_lead_to_pivot_sessions']:.1f} | "
                f"{event['false_alert_clusters_per_sector_year']:.2f} |"
            )
    lines.extend(
        [
            "",
            "## 2024-2026 고정 홀드아웃",
            "",
            "| 방향 | 20D 평균 (95% CI) | 사전 포착 | 95% CI 하한 | Pivot 선행 | 실용 게이트 |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for result in summary["holdout_2024"]:
        event = result["metrics"]["early_event_metrics"]
        gates = result["gates"]
        lines.append(
            f"| {result['direction']} | {result['mean_signed_relative_return_20d']:+.2%} "
            f"[{result['ci_low']:+.2%}, {result['ci_high']:+.2%}] | "
            f"{event['pre_pivot_recall']:.1%} | {gates['pre_pivot_recall_ci_low']:.1%} | "
            f"{event['median_lead_to_pivot_sessions']:.1f} | "
            f"{'PASS' if gates['economic_ci_above_zero'] or gates['early_signal_gate'] else 'FAIL'} |"
        )
    lines.extend(["", "PIT가 부분 검증 상태이므로 운영 승격 없이 연구용으로만 해석합니다."])
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    pooled, _, input_audit = merge_inputs(
        args.gics_package_dir.resolve(), args.breadth_features.resolve(), horizon=10
    )
    all_predictions: list[pd.DataFrame] = []
    directions: dict[str, Any] = {}
    calendars: list[pd.DatetimeIndex] = []
    qualified_pools: dict[str, pd.DataFrame] = {}
    for direction in ("ROTATION_OUT", "ROTATION_IN"):
        gated, label_audit = qualify_labels(pooled, direction)
        qualified_pools[direction] = gated
        direction_models: dict[str, Any] = {}
        direction_predictions: list[pd.DataFrame] = []
        for name, source in (("STANDARD", pooled), ("TRADABLE", gated)):
            predictions, _, walk = walk_forward_direction(
                source, direction, min_train_dates=756, refit_every=63, c_value=0.1,
                target_mode="PRE_PIVOT", models=("POOLED_PRICE",),
            )
            predictions["model"] = name
            predictions["evaluation_sample"] = predictions["date"].ge(EVALUATION_START)
            evaluated = predictions.loc[
                predictions["evaluation_sample"] & predictions["target"].notna()
            ]
            direction_predictions.append(predictions)
            direction_models[name] = {
                "walk_forward": walk,
                "metrics": _model_metrics(
                    evaluated, source, direction, horizon=10, cutoff=CUTOFF
                ),
            }
        combined = pd.concat(direction_predictions, ignore_index=True)
        all_predictions.append(combined)
        calendar = pd.DatetimeIndex(
            combined.loc[combined["evaluation_sample"], "date"].drop_duplicates().sort_values()
        )
        calendars.append(calendar)
        directions[direction] = {
            "label_audit": label_audit,
            "models": {name: item["metrics"] for name, item in direction_models.items()},
        }
    predictions = pd.concat(all_predictions, ignore_index=True)
    evaluation = predictions.loc[predictions["evaluation_sample"]].copy()
    economics = economic_rows(evaluation, pooled, horizon=10, cutoff=CUTOFF)
    calendar = pd.DatetimeIndex(sorted(set().union(*(item.tolist() for item in calendars))))
    economic_summary = summarize_economics(economics, calendar, args.bootstrap_samples)
    economic_map = {
        (row["direction"], row["model"]): row
        for row in economic_summary if row["period"] == "ALL_OOS"
    }
    all_oos_gates = {
        direction: practical_gates(
            result["models"]["TRADABLE"], economic_map[(direction, "TRADABLE")]
        )
        for direction, result in directions.items()
    }
    holdout: list[dict[str, Any]] = []
    for direction in ("ROTATION_OUT", "ROTATION_IN"):
        selected = evaluation.loc[
            evaluation["direction"].eq(direction)
            & evaluation["model"].eq("TRADABLE")
            & evaluation["date"].ge(pd.Timestamp("2024-01-01"))
            & evaluation["target"].notna()
        ]
        holdout_economics = economic_rows(selected, pooled, horizon=10, cutoff=CUTOFF)
        holdout_calendar = pd.DatetimeIndex(selected["date"].drop_duplicates().sort_values())
        low, high = block_mean_ci(
            holdout_economics,
            holdout_calendar,
            "signed_relative_return_20d",
            samples=args.bootstrap_samples,
        )
        metric = _model_metrics(
            selected, qualified_pools[direction], direction, horizon=10, cutoff=CUTOFF
        )
        economic = {
            "ci_low": float(low) if np.isfinite(low) else None,
            "ci_high": float(high) if np.isfinite(high) else None,
        }
        holdout.append(
            {
                "direction": direction,
                "start": "2024-01-01",
                "clusters": int(len(holdout_economics)),
                "mean_signed_relative_return_20d": float(
                    holdout_economics["signed_relative_return_20d"].mean()
                ),
                **economic,
                "metrics": metric,
                "gates": practical_gates(metric, economic),
            }
        )
    passed = any(
        row["gates"]["economic_ci_above_zero"] or row["gates"]["early_signal_gate"]
        for row in holdout
    )
    summary = {
        "verdict": "PRACTICAL_GATE_PASS_RESEARCH_ONLY" if passed else "NO_PRACTICAL_GATE_PASS",
        "pit_status": "PIT_PARTIAL/CURRENT_REVISED_HISTORY_ONLY",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "spec": {
            "minimum_followthrough": MINIMUM_FOLLOWTHROUGH,
            "followthrough_sessions_from_pivot": FOLLOWTHROUGH_HORIZON,
            "evaluation_start": str(EVALUATION_START.date()),
            "alert_cutoff": CUTOFF,
            "bootstrap_samples": args.bootstrap_samples,
        },
        "input_audit": input_audit,
        "directions": directions,
        "economics": economic_summary,
        "all_oos_practical_gates": all_oos_gates,
        "holdout_2024": holdout,
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    predictions.to_csv(output / "predictions.csv", index=False)
    economics.to_csv(output / "economic_clusters.csv", index=False)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str, allow_nan=False),
        encoding="utf-8",
    )
    (output / "REPORT.md").write_text(report(summary), encoding="utf-8")
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Economically qualified PRE_PIVOT label test")
    result.add_argument("--gics-package-dir", type=Path)
    result.add_argument("--breadth-features", type=Path)
    result.add_argument("--output-dir", type=Path)
    result.add_argument("--bootstrap-samples", type=int, default=1_000)
    result.add_argument("--self-test", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.self_test:
        self_test()
        return
    if not all((args.gics_package_dir, args.breadth_features, args.output_dir)):
        raise ValueError("all input and output paths are required")
    print(json.dumps({"verdict": run(args)["verdict"]}))


if __name__ == "__main__":
    main()
