"""Apply a causal per-sector cooldown to existing alert clusters."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sector_turning_breadth_shadow import block_mean_ci
from sector_turning_layer_weight_experiment import _json_ready
from sector_turning_model_policy_scan import (
    CUTOFFS,
    SOURCES,
    economics,
    forward_returns,
    load_source,
)
from sector_turning_reference_policy_experiment import policy_metrics
from sector_turning_stability_weight_experiment import WINDOWS, window
from sector_turning_two_stage_confirmation_experiment import cluster_rows


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs" / "sector_turning_alert_cooldown_scan_20260903"
COOLDOWNS = (0, 21, 42, 63, 84, 126)


def cooldown_alerts(rows: pd.DataFrame, base_alerts: pd.Series, cooldown: int) -> pd.Series:
    ordered = rows.assign(_alert=base_alerts).sort_values(["sector", "_position"])
    output = pd.Series(False, index=ordered.index)
    for _, sector_rows in ordered.loc[ordered["_alert"]].groupby("sector", sort=False):
        cluster_id = sector_rows["_position"].diff().gt(10).cumsum()
        last_kept = -10**9
        for _, cluster in sector_rows.groupby(cluster_id, sort=False):
            start = int(cluster["_position"].iloc[0])
            if start - last_kept > cooldown:
                output.loc[cluster.index] = True
                last_kept = start
    return output.reindex(rows.index)


def scan_candidate(rows: pd.DataFrame) -> list[dict]:
    direction = str(rows["direction"].iloc[0])
    rows = rows.sort_values(["sector", "_position"]).copy()
    rows["signed_forward_return_20"] = rows["forward_relative_return_20"] * (
        1 if direction == "ROTATION_IN" else -1
    )
    records = []
    for cutoff in CUTOFFS:
        base = rows["score"].ge(cutoff)
        for cooldown in COOLDOWNS:
            alerts = cooldown_alerts(rows, base, cooldown)
            metrics = {
                name: policy_metrics(window(rows, name), alerts.loc[window(rows, name).index])
                for name in ("selection_2023_2024", "selection_2025")
            }
            economic = {
                name: economics(rows, alerts, name)
                for name in ("selection_2023_2024", "selection_2025")
            }
            records.append({
                "source": rows["source"].iloc[0], "direction": direction,
                "model": rows["model"].iloc[0], "target_mode": rows["target_mode"].iloc[0],
                "cutoff": cutoff, "cooldown_sessions": cooldown,
                **{
                    f"{name}_{metric}": values[source]
                    for name, values in metrics.items()
                    for metric, source in (
                        ("pre_pivot_recall", "pre_pivot_recall"),
                        ("event_recall", "event_recall"),
                        ("false_per_sector_year", "false_alert_clusters_per_sector_year"),
                        ("precision", "alert_cluster_precision"),
                    )
                },
                **{
                    f"{name}_{metric}": values[source]
                    for name, values in economic.items()
                    for metric, source in (
                        ("economic_clusters", "clusters"),
                        ("mean_signed_return_20d", "mean_signed_relative_return_20d"),
                    )
                },
            })
    return records


def select_policy(sweep: pd.DataFrame, direction: str) -> tuple[pd.Series, bool]:
    candidates = sweep.loc[sweep["direction"].eq(direction)].copy()
    candidates["worst_recall"] = candidates[[
        "selection_2023_2024_pre_pivot_recall", "selection_2025_pre_pivot_recall"
    ]].min(axis=1)
    candidates["worst_false"] = candidates[[
        "selection_2023_2024_false_per_sector_year", "selection_2025_false_per_sector_year"
    ]].max(axis=1)
    candidates["worst_return"] = candidates[[
        "selection_2023_2024_mean_signed_return_20d",
        "selection_2025_mean_signed_return_20d",
    ]].min(axis=1)
    candidates["minimum_clusters"] = candidates[[
        "selection_2023_2024_economic_clusters", "selection_2025_economic_clusters"
    ]].min(axis=1)
    eligible = candidates.loc[
        candidates["worst_recall"].ge(0.15)
        & candidates["worst_return"].ge(0)
        & candidates["minimum_clusters"].ge(15)
    ]
    passed = not eligible.empty
    if not passed:
        eligible = candidates.loc[
            candidates["worst_recall"].ge(0.15) & candidates["minimum_clusters"].ge(15)
        ]
    if eligible.empty:
        eligible = candidates
    return eligible.sort_values(
        ["worst_false", "worst_return", "worst_recall"], ascending=[True, False, False]
    ).iloc[0], passed


def final_evaluation(chosen: pd.Series, returns: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = load_source(str(chosen["source"]), returns)
    rows = rows.loc[
        rows["direction"].eq(chosen["direction"])
        & rows["model"].eq(chosen["model"])
        & rows["target_mode"].eq(chosen["target_mode"])
    ].sort_values(["sector", "_position"]).copy()
    rows["signed_forward_return_20"] = rows["forward_relative_return_20"] * (
        1 if chosen["direction"] == "ROTATION_IN" else -1
    )
    alerts = cooldown_alerts(
        rows, rows["score"].ge(float(chosen["cutoff"])), int(chosen["cooldown_sessions"])
    )
    frozen_rows = window(rows, "frozen_2026")
    metrics = policy_metrics(frozen_rows, alerts.loc[frozen_rows.index])
    economic = economics(rows, alerts, "frozen_2026")
    clusters = window(cluster_rows(rows, alerts), "frozen_2026").dropna(
        subset=["signed_forward_return_20"]
    )
    calendar = pd.DatetimeIndex(frozen_rows["date"].drop_duplicates().sort_values())
    low, high = block_mean_ci(
        clusters.rename(columns={"date": "signal_date"}), calendar,
        "signed_forward_return_20", samples=1000,
    )
    economic["ci_low"] = low if np.isfinite(low) else None
    economic["ci_high"] = high if np.isfinite(high) else None
    verdict = (
        "REFERENCE_CANDIDATE" if metrics["false_alert_clusters_per_sector_year"] <= 1.25
        and metrics["pre_pivot_recall"] >= 0.20
        and metrics["alert_cluster_precision"] >= 0.50
        and (economic["ci_low"] or -np.inf) > 0
        and (metrics["median_lead_to_pivot_sessions"] or 0) >= 2
        else "NOT_READY"
    )
    output = rows.copy()
    output["selected_alert"] = alerts
    return output, {"verdict": verdict, "metrics": metrics, "economics": economic}


def self_test() -> None:
    rows = pd.DataFrame({"sector": ["A"] * 6, "_position": [1, 2, 20, 21, 50, 51]})
    alerts = pd.Series(True, index=rows.index)
    assert cooldown_alerts(rows, alerts, 0).tolist() == [True] * 6
    assert cooldown_alerts(rows, alerts, 42).tolist() == [True, True, False, False, True, True]
    print("self-test: PASS")


def main() -> None:
    returns = forward_returns()
    records = []
    for source in SOURCES:
        rows = load_source(source, returns)
        for _, candidate in rows.groupby(["direction", "model", "target_mode"], sort=True):
            if all(len(window(candidate, name)) for name in WINDOWS):
                records.extend(scan_candidate(candidate))
    sweep = pd.DataFrame(records)
    outputs, directions = [], {}
    for direction in ("ROTATION_IN", "ROTATION_OUT"):
        chosen, passed = select_policy(sweep, direction)
        output, final = final_evaluation(chosen, returns)
        outputs.append(output)
        directions[direction] = {
            "selection_gate_passed": passed,
            "policy": {
                "source": chosen["source"], "model": chosen["model"],
                "target_mode": chosen["target_mode"], "cutoff": float(chosen["cutoff"]),
                "cooldown_sessions": int(chosen["cooldown_sessions"]),
            },
            "selection": {
                key: _json_ready(chosen[key]) for key in chosen.index
                if key.startswith("selection_") or key in {"worst_recall", "worst_false", "worst_return"}
            },
            "frozen_2026": final,
        }
    summary = {
        "verdict": "REFERENCE_CANDIDATE" if all(
            item["frozen_2026"]["verdict"] == "REFERENCE_CANDIDATE"
            for item in directions.values()
        ) else "RESEARCH_ONLY",
        "pit_status": "PIT_PARTIAL/CURRENT_REVISED_HISTORY_ONLY",
        "objective": "reduce repeated false clusters without changing model scores",
        "sources": list(SOURCES), "cooldowns": list(COOLDOWNS),
        "directions": directions,
    }
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    sweep.to_csv(OUTPUT / "cooldown_sweep.csv", index=False)
    pd.concat(outputs, ignore_index=True).to_csv(OUTPUT / "selected_signals.csv", index=False)
    clean = _json_ready(summary)
    (OUTPUT / "summary.json").write_text(
        json.dumps(clean, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({"output": str(OUTPUT), "verdict": clean["verdict"]}, ensure_ascii=False))


if __name__ == "__main__":
    import sys

    self_test() if "--self-test" in sys.argv else main()
