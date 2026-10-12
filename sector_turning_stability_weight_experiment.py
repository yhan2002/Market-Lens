"""Select one stable low-noise layer weight before a frozen 2026 holdout."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sector_turning_layer_weight_experiment import (
    COMPONENTS,
    OUTPUT as LAYER_OUTPUT,
    _json_ready,
    score_stack,
    weight_grid,
)
from sector_turning_reference_policy_experiment import confirmed_alerts, policy_metrics


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs" / "sector_turning_stability_weights_20260903"
CUTOFFS = tuple(np.arange(0.60, 0.951, 0.025).round(3))
WINDOWS = {
    "selection_2023_2024": (pd.Timestamp("2023-01-03"), pd.Timestamp("2024-12-31")),
    "selection_2025": (pd.Timestamp("2025-01-02"), pd.Timestamp("2025-12-31")),
    "frozen_2026": (pd.Timestamp("2026-01-02"), pd.Timestamp.max),
}
MIN_SELECTION_RECALL = 0.18


def window(rows: pd.DataFrame, name: str) -> pd.DataFrame:
    start, end = WINDOWS[name]
    return rows.loc[rows["date"].between(start, end)]


def run_direction(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    records, scores = [], {}
    for weights in weight_grid():
        score = score_stack(rows, weights)
        scores[weights] = score
        for cutoff in CUTOFFS:
            alerts = confirmed_alerts(rows, score, cutoff, 1).astype(bool)
            metrics = {
                name: policy_metrics(window(rows, name), alerts.loc[window(rows, name).index])
                for name in ("selection_2023_2024", "selection_2025")
            }
            records.append({
                **dict(zip(COMPONENTS, weights, strict=True)), "cutoff": cutoff,
                **{
                    f"{name}_{metric}": values[source]
                    for name, values in metrics.items()
                    for metric, source in (
                        ("recall", "pre_pivot_recall"),
                        ("false_per_sector_year", "false_alert_clusters_per_sector_year"),
                        ("precision", "alert_cluster_precision"),
                        ("lead", "median_lead_to_pivot_sessions"),
                    )
                },
            })
    sweep = pd.DataFrame(records)
    sweep["worst_selection_recall"] = sweep[[
        "selection_2023_2024_recall", "selection_2025_recall"
    ]].min(axis=1)
    sweep["worst_selection_false"] = sweep[[
        "selection_2023_2024_false_per_sector_year", "selection_2025_false_per_sector_year"
    ]].max(axis=1)
    sweep["mean_selection_false"] = sweep[[
        "selection_2023_2024_false_per_sector_year", "selection_2025_false_per_sector_year"
    ]].mean(axis=1)
    eligible = sweep.loc[sweep["worst_selection_recall"].ge(MIN_SELECTION_RECALL)]
    if eligible.empty:
        raise RuntimeError("no stable policy meets the minimum recall floor")
    chosen = eligible.sort_values(
        ["worst_selection_false", "mean_selection_false", "worst_selection_recall"],
        ascending=[True, True, False],
    ).iloc[0]
    weights = tuple(float(chosen[name]) for name in COMPONENTS)
    cutoff = float(chosen["cutoff"])
    score = scores[weights]
    alerts = confirmed_alerts(rows, score, cutoff, 1).astype(bool)
    results = {
        name: policy_metrics(window(rows, name), alerts.loc[window(rows, name).index])
        for name in WINDOWS
    }
    frozen = results["frozen_2026"]
    verdict = (
        "REFERENCE_CANDIDATE" if frozen["false_alert_clusters_per_sector_year"] <= 1.25
        and frozen["pre_pivot_recall"] >= 0.20
        and frozen["alert_cluster_precision"] >= 0.50
        and (frozen["median_lead_to_pivot_sessions"] or 0) >= 2
        else "NOT_READY"
    )
    output = rows.copy()
    output["stable_score"] = score
    output["stable_alert"] = alerts
    return sweep, output, {
        "verdict": verdict,
        "policy": {"weights": dict(zip(COMPONENTS, weights, strict=True)), "cutoff": cutoff},
        "selection": "minimize worst false-cluster rate across 2023-24 and 2025 with recall >=18% in both",
        "frozen_holdout_first_inspection": "2026 only",
        "results": results,
    }


def self_test() -> None:
    assert len(weight_grid()) == 56 and len(CUTOFFS) == 15
    dates = pd.to_datetime(["2024-01-01", "2025-06-01", "2026-02-01"])
    rows = pd.DataFrame({"date": dates})
    assert [len(window(rows, name)) for name in WINDOWS] == [1, 1, 1]
    print("self-test: PASS")


def main() -> None:
    scores = pd.read_csv(
        LAYER_OUTPUT / "scores.csv", parse_dates=["date", "target_event_confirmation_date"],
        dtype={"sector": "string"},
    )
    scores["sector"] = scores["sector"].str.zfill(4)
    sweeps, signals, directions = [], [], {}
    for direction, rows in scores.groupby("direction", sort=True):
        sweep, output, result = run_direction(rows.copy())
        sweep.insert(0, "direction", direction)
        sweeps.append(sweep)
        signals.append(output)
        directions[direction] = result
    summary = {
        "verdict": "REFERENCE_CANDIDATE" if all(
            item["verdict"] == "REFERENCE_CANDIDATE" for item in directions.values()
        ) else "RESEARCH_ONLY",
        "pit_status": "PIT_PARTIAL/CURRENT_REVISED_HISTORY_ONLY",
        "objective": "stable low-noise weighting across the Market Lens sector-layer lineage",
        "directions": directions,
    }
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    pd.concat(sweeps, ignore_index=True).to_csv(OUTPUT / "stability_sweep.csv", index=False)
    pd.concat(signals, ignore_index=True).to_csv(OUTPUT / "stable_signals.csv", index=False)
    clean = _json_ready(summary)
    (OUTPUT / "summary.json").write_text(
        json.dumps(clean, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({"output": str(OUTPUT), "verdict": clean["verdict"]}, ensure_ascii=False))


if __name__ == "__main__":
    import sys

    self_test() if "--self-test" in sys.argv else main()
