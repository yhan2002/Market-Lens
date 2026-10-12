"""Chronological threshold selection for the frozen pooled price PRE_PIVOT model."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sector_turning_beta_breadth_shadow import merge_inputs
from sector_turning_breadth_shadow import block_mean_ci, economic_rows
from sector_turning_pooled_shadow import _model_metrics, _wilson_interval


THRESHOLDS = (0.80, 0.85, 0.90, 0.95, 0.975, 0.99)
VALIDATION_START = pd.Timestamp("2020-01-01")
VALIDATION_END = pd.Timestamp("2023-12-31")
HOLDOUT_START = pd.Timestamp("2024-01-01")


def select_threshold(candidates: pd.DataFrame, *, minimum_clusters: int = 50) -> float:
    eligible = candidates.loc[
        candidates["clusters"].ge(minimum_clusters)
        & candidates["alert_day_rate"].le(0.20)
    ]
    if eligible.empty:
        raise ValueError("no validation threshold meets support and alert-rate gates")
    return float(
        eligible.sort_values(
            ["mean_signed_relative_return_20d", "threshold"], ascending=[False, False]
        ).iloc[0]["threshold"]
    )


def period_result(
    predictions: pd.DataFrame,
    pooled: pd.DataFrame,
    direction: str,
    threshold: float,
    start: pd.Timestamp,
    end: pd.Timestamp | None,
    *,
    bootstrap_samples: int,
) -> dict[str, Any]:
    dates = pd.to_datetime(predictions["date"])
    selected = predictions.loc[
        predictions["direction"].eq(direction)
        & predictions["model"].eq("POOLED_PRICE")
        & dates.ge(start)
        & (True if end is None else dates.le(end))
        & predictions["target"].notna()
    ].copy()
    calendar = pd.DatetimeIndex(selected["date"].drop_duplicates().sort_values())
    economics = economic_rows(selected, pooled, horizon=10, cutoff=threshold)
    low, high = (
        block_mean_ci(
            economics, calendar, "signed_relative_return_20d", samples=bootstrap_samples
        )
        if bootstrap_samples
        else (np.nan, np.nan)
    )
    metrics = _model_metrics(selected, pooled, direction, horizon=10, cutoff=threshold)
    event = metrics["early_event_metrics"]
    wilson_low, wilson_high = _wilson_interval(
        event["pre_pivot_hits"], event["eligible_confirmations"]
    )
    return {
        "direction": direction,
        "threshold": threshold,
        "start": str(start.date()),
        "end": None if end is None else str(end.date()),
        "rows": int(len(selected)),
        "clusters": int(len(economics)),
        "mean_signed_relative_return_20d": (
            float(economics["signed_relative_return_20d"].mean()) if len(economics) else None
        ),
        "economic_ci_low": float(low) if np.isfinite(low) else None,
        "economic_ci_high": float(high) if np.isfinite(high) else None,
        "favorable_3pct_rate": (
            float(economics["favorable_3pct"].mean()) if len(economics) else None
        ),
        **event,
        "pre_pivot_recall_ci_low": wilson_low,
        "pre_pivot_recall_ci_high": wilson_high,
    }


def practical_gates(result: dict[str, Any]) -> dict[str, bool]:
    economic_pass = (
        result["economic_ci_low"] is not None and result["economic_ci_low"] > 0
    )
    early_pass = all(
        (
            result["eligible_confirmations"] >= 20,
            result["pre_pivot_recall_ci_low"] is not None
            and result["pre_pivot_recall_ci_low"] > 0.05,
            result["median_lead_to_pivot_sessions"] is not None
            and result["median_lead_to_pivot_sessions"] >= 3,
            result["false_alert_clusters_per_sector_year"] is not None
            and result["false_alert_clusters_per_sector_year"] <= 2,
            result["alert_day_rate"] <= 0.20,
        )
    )
    return {"economic_ci_above_zero": economic_pass, "early_signal_gate": early_pass}


def self_test() -> None:
    candidates = pd.DataFrame(
        {
            "threshold": [0.8, 0.9, 0.95],
            "clusters": [80, 60, 20],
            "alert_day_rate": [0.15, 0.10, 0.02],
            "mean_signed_relative_return_20d": [0.01, 0.02, 0.10],
        }
    )
    assert select_threshold(candidates) == 0.9
    print("self-test: PASS")


def report(summary: dict[str, Any]) -> str:
    lines = [
        "# PRE_PIVOT Threshold Holdout",
        "",
        f"- 판정: **{summary['verdict']}**",
        "- 선택: 2020-2023 / 최종 검증: 2024-2026",
        "",
        "| 방향 | 선택 임계값 | 20D 평균 | 95% CI | 사전 포착 | Pivot 선행 | False/sector-yr |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in summary["holdout"]:
        mean = result["mean_signed_relative_return_20d"]
        low, high = result["economic_ci_low"], result["economic_ci_high"]
        lines.append(
            f"| {result['direction']} | {result['threshold']:.3f} | {mean:+.2%} | "
            f"[{low:+.2%}, {high:+.2%}] | {result['pre_pivot_recall']:.1%} | "
            f"{result['median_lead_to_pivot_sessions']:.1f} | "
            f"{result['false_alert_clusters_per_sector_year']:.2f} |"
        )
    lines.extend(["", "운영 승격 없음. 임계값은 검증구간 결과를 본 뒤 재선택하지 않습니다."])
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    predictions = pd.read_csv(args.predictions.resolve(), parse_dates=["date"])
    predictions = predictions.loc[predictions["model"].eq("POOLED_PRICE")].copy()
    pooled, _, audit = merge_inputs(
        args.gics_package_dir.resolve(), args.breadth_features.resolve(), horizon=10
    )
    candidates: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    for direction in ("ROTATION_OUT", "ROTATION_IN"):
        for threshold in THRESHOLDS:
            result = period_result(
                predictions, pooled, direction, threshold,
                VALIDATION_START, VALIDATION_END, bootstrap_samples=0,
            )
            candidates.append(result)
        chosen = select_threshold(
            pd.DataFrame([row for row in candidates if row["direction"] == direction])
        )
        result = period_result(
            predictions, pooled, direction, chosen,
            HOLDOUT_START, None, bootstrap_samples=args.bootstrap_samples,
        )
        result["gates"] = practical_gates(result)
        holdout.append(result)
    passed = any(any(result["gates"].values()) for result in holdout)
    summary = {
        "verdict": "PRACTICAL_GATE_PASS_RESEARCH_ONLY" if passed else "NO_PRACTICAL_GATE_PASS",
        "pit_status": "PIT_PARTIAL/CURRENT_REVISED_HISTORY_ONLY",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": "highest validation mean 20D signed relative return; >=50 clusters; alert rate <=20%",
        "audit": audit,
        "candidates": candidates,
        "holdout": holdout,
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(candidates).to_csv(output / "validation_candidates.csv", index=False)
    pd.DataFrame(holdout).to_json(output / "holdout_results.json", orient="records", indent=2)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str, allow_nan=False),
        encoding="utf-8",
    )
    (output / "REPORT.md").write_text(report(summary), encoding="utf-8")
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Chronological PRE_PIVOT threshold holdout")
    result.add_argument("--predictions", type=Path)
    result.add_argument("--gics-package-dir", type=Path)
    result.add_argument("--breadth-features", type=Path)
    result.add_argument("--output-dir", type=Path)
    result.add_argument("--bootstrap-samples", type=int, default=2_000)
    result.add_argument("--self-test", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.self_test:
        self_test()
        return
    if not all((args.predictions, args.gics_package_dir, args.breadth_features, args.output_dir)):
        raise ValueError("all input and output paths are required")
    print(json.dumps({"verdict": run(args)["verdict"]}))


if __name__ == "__main__":
    main()
