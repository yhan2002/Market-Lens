"""Evaluate cooldowns only on the selected four-layer Market Lens stack."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sector_turning_alert_cooldown_scan import COOLDOWNS, cooldown_alerts
from sector_turning_breadth_shadow import block_mean_ci
from sector_turning_layer_weight_experiment import _json_ready
from sector_turning_model_policy_scan import economics, forward_returns
from sector_turning_reference_policy_experiment import policy_metrics
from sector_turning_stability_weight_experiment import OUTPUT as STABILITY_OUTPUT, WINDOWS, window
from sector_turning_two_stage_confirmation_experiment import cluster_rows


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs" / "sector_turning_integrated_cooldown_20260903"


def evaluate(rows: pd.DataFrame, alerts: pd.Series, name: str) -> dict:
    sample = window(rows, name)
    result = policy_metrics(sample, alerts.loc[sample.index])
    result["economics"] = economics(rows, alerts, name)
    return result


def choose_cooldown(sweep: pd.DataFrame) -> pd.Series:
    eligible = sweep.loc[
        sweep["worst_recall"].ge(0.15)
        & sweep["worst_return"].ge(0)
        & sweep["minimum_clusters"].ge(15)
    ]
    if eligible.empty:
        eligible = sweep.loc[sweep["worst_recall"].ge(0.15)]
    return eligible.sort_values(
        ["worst_false", "worst_return", "worst_recall"], ascending=[True, False, False]
    ).iloc[0]


def select_cooldown(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    records, alerts_by_cooldown = [], {}
    base = rows["stable_alert"].astype(bool)
    for cooldown in COOLDOWNS:
        alerts = cooldown_alerts(rows, base, cooldown)
        alerts_by_cooldown[cooldown] = alerts
        results = {
            name: evaluate(rows, alerts, name)
            for name in ("selection_2023_2024", "selection_2025")
        }
        records.append({
            "cooldown_sessions": cooldown,
            "worst_recall": min(value["pre_pivot_recall"] for value in results.values()),
            "worst_false": max(value["false_alert_clusters_per_sector_year"] for value in results.values()),
            "worst_return": min(
                value["economics"]["mean_signed_relative_return_20d"]
                for value in results.values()
            ),
            "minimum_clusters": min(value["economics"]["clusters"] for value in results.values()),
            "selection": results,
        })
    sweep = pd.DataFrame(records)
    chosen = choose_cooldown(sweep)
    return sweep, chosen, alerts_by_cooldown[int(chosen["cooldown_sessions"])]


def frozen_result(rows: pd.DataFrame, alerts: pd.Series) -> dict:
    result = evaluate(rows, alerts, "frozen_2026")
    clusters = window(cluster_rows(rows, alerts), "frozen_2026").dropna(
        subset=["signed_forward_return_20"]
    )
    calendar = pd.DatetimeIndex(window(rows, "frozen_2026")["date"].drop_duplicates().sort_values())
    low, high = block_mean_ci(
        clusters.rename(columns={"date": "signal_date"}), calendar,
        "signed_forward_return_20", samples=1_000,
    )
    result["economics"].update({
        "ci_low": low if np.isfinite(low) else None,
        "ci_high": high if np.isfinite(high) else None,
    })
    result["verdict"] = (
        "REFERENCE_CANDIDATE"
        if result["false_alert_clusters_per_sector_year"] <= 1.25
        and result["pre_pivot_recall"] >= 0.30
        and result["alert_cluster_precision"] >= 0.50
        and (result["median_lead_to_pivot_sessions"] or 0) >= 2
        and (result["economics"]["ci_low"] or -np.inf) > 0
        else "NOT_READY"
    )
    return result


def main() -> None:
    summary = json.loads((STABILITY_OUTPUT / "summary.json").read_text(encoding="utf-8"))
    rows = pd.read_csv(
        STABILITY_OUTPUT / "stable_signals.csv",
        parse_dates=["date", "target_event_confirmation_date"], dtype={"sector": "string"},
    )
    rows["sector"] = rows["sector"].str.zfill(4)
    rows = rows.loc[rows["date"].ge("2023-01-03")].merge(
        forward_returns(), on=["date", "sector", "_position"], how="left", validate="many_to_one"
    )
    outputs, sweeps, directions = [], [], {}
    for direction, part in rows.groupby("direction", sort=True):
        part = part.sort_values(["sector", "_position"]).copy()
        part["signed_forward_return_20"] = part["forward_relative_return_20"] * (
            1 if direction == "ROTATION_IN" else -1
        )
        policy = summary["directions"][direction]["policy"]
        sweep, chosen, alerts = select_cooldown(part)
        part["integrated_alert"] = alerts
        outputs.append(part)
        sweep.insert(0, "direction", direction)
        sweeps.append(sweep.drop(columns="selection"))
        directions[direction] = {
            "weights": policy["weights"], "cutoff": policy["cutoff"],
            "cooldown_sessions": int(chosen["cooldown_sessions"]),
            "selection": chosen["selection"],
            "frozen_2026": frozen_result(part, alerts),
        }
    clean = _json_ready({
        "verdict": "REFERENCE_CANDIDATE" if all(
            value["frozen_2026"]["verdict"] == "REFERENCE_CANDIDATE"
            for value in directions.values()
        ) else "RESEARCH_ONLY",
        "pit_status": "PIT_PARTIAL/CURRENT_REVISED_HISTORY_ONLY",
        "objective": "cooldown evaluation constrained to the selected integrated layer stack",
        "directions": directions,
    })
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    pd.concat(sweeps, ignore_index=True).to_csv(OUTPUT / "cooldown_sweep.csv", index=False)
    pd.concat(outputs, ignore_index=True).to_csv(OUTPUT / "integrated_signals.csv", index=False)
    (OUTPUT / "summary.json").write_text(
        json.dumps(clean, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({"output": str(OUTPUT), "verdict": clean["verdict"]}, ensure_ascii=False))


def self_test() -> None:
    sweep = pd.DataFrame({
        "cooldown_sessions": [0, 21], "worst_recall": [0.20, 0.18],
        "worst_false": [1.5, 1.0], "worst_return": [0.01, 0.01],
        "minimum_clusters": [20, 20],
    })
    assert choose_cooldown(sweep)["cooldown_sessions"] == 21
    print("self-test: PASS")


if __name__ == "__main__":
    import sys

    self_test() if "--self-test" in sys.argv else main()
