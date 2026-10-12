"""Separate early WATCH from sparse price-confirmed sector turning alerts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sector_macro_trend_hint_experiment import PACKAGE
from sector_turning_breadth_shadow import block_mean_ci
from sector_turning_layer_weight_experiment import COMPONENTS, _json_ready
from sector_turning_pooled_shadow import build_pooled_daily_frame
from sector_turning_raw_false_positive_filter_experiment import load_scores_with_raw_features
from sector_turning_reference_policy_experiment import OUTPUT as REFERENCE_OUTPUT, policy_metrics
from sector_turning_stability_weight_experiment import WINDOWS, window


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs" / "sector_turning_two_stage_confirmation_20260903"
RETURN_THRESHOLDS = (-0.02, -0.01, 0.00, 0.01, 0.02)
ACCEL_THRESHOLDS = (-0.02, -0.01, 0.00, 0.01, 0.02)
CONTEXT_THRESHOLDS = {
    "ROTATION_IN": (-0.02, -0.05, -0.08, -0.12),
    "ROTATION_OUT": (-0.02, 0.00, 0.03, 0.06),
}
MODES = ("EITHER", "BOTH")


def load_panel() -> pd.DataFrame:
    scores = load_scores_with_raw_features()
    reference = pd.read_csv(
        REFERENCE_OUTPUT / "reference_signals.csv", parse_dates=["date"], dtype={"sector": "string"}
    )
    reference["sector"] = reference["sector"].str.zfill(4)
    scores = scores.merge(
        reference[["direction", "date", "sector", "reference_alert"]],
        on=["direction", "date", "sector"], how="left", validate="one_to_one",
    )
    pooled, _, _ = build_pooled_daily_frame(PACKAGE, None, horizon=10)
    pooled = pooled.sort_values(["sector", "_position"]).copy()
    grouped = pooled.groupby("sector", sort=False)
    entry = grouped["relative_price"].shift(-1)
    exit_value = grouped["relative_price"].shift(-21)
    valid = grouped["_position"].shift(-1).eq(pooled["_position"] + 1) & grouped[
        "_position"
    ].shift(-21).eq(pooled["_position"] + 21)
    pooled["forward_relative_return_20"] = (exit_value / entry - 1).where(valid)
    return scores.merge(
        pooled[["date", "sector", "relative_price", "forward_relative_return_20"]],
        on=["date", "sector"], how="left", validate="many_to_one",
    )


def confirmation_alerts(
    rows: pd.DataFrame,
    direction: str,
    return_threshold: float,
    accel_threshold: float,
    context_threshold: float,
    mode: str,
) -> pd.Series:
    sign = 1 if direction == "ROTATION_IN" else -1
    returning = (sign * rows["relative_return_5"]).ge(return_threshold)
    accelerating = (sign * rows["relative_acceleration_5"]).ge(accel_threshold)
    reversal = returning | accelerating if mode == "EITHER" else returning & accelerating
    context = (
        rows["relative_drawdown_60"].le(context_threshold)
        if direction == "ROTATION_IN"
        else rows["relative_ma200_gap"].ge(context_threshold)
    )
    return rows["reference_alert"].astype(bool) & reversal & context


def cluster_rows(rows: pd.DataFrame, alerts: pd.Series) -> pd.DataFrame:
    selected = rows.loc[alerts].sort_values(["sector", "_position"]).copy()
    starts = selected.groupby("sector")["_position"].diff().gt(10) | ~selected[
        "sector"
    ].duplicated()
    return selected.loc[starts]


def economic_metrics(rows: pd.DataFrame, alerts: pd.Series, name: str) -> dict:
    clusters = cluster_rows(rows, alerts)
    sample = window(clusters, name).dropna(subset=["signed_forward_return_20"])
    return {
        "clusters": int(len(sample)),
        "mean_signed_relative_return_20d": float(sample["signed_forward_return_20"].mean())
        if len(sample) else None,
        "positive_return_rate": float(sample["signed_forward_return_20"].gt(0).mean())
        if len(sample) else None,
    }


def run_direction(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    direction = str(rows["direction"].iloc[0])
    rows = rows.copy()
    rows["signed_forward_return_20"] = rows["forward_relative_return_20"] * (
        1 if direction == "ROTATION_IN" else -1
    )
    records, cache = [], {}
    for return_threshold in RETURN_THRESHOLDS:
        for accel_threshold in ACCEL_THRESHOLDS:
            for context_threshold in CONTEXT_THRESHOLDS[direction]:
                for mode in MODES:
                    key = (return_threshold, accel_threshold, context_threshold, mode)
                    alerts = confirmation_alerts(rows, direction, *key)
                    cache[key] = alerts
                    metrics = {
                        name: policy_metrics(window(rows, name), alerts.loc[window(rows, name).index])
                        for name in ("selection_2023_2024", "selection_2025")
                    }
                    economics = {
                        name: economic_metrics(rows, alerts, name)
                        for name in ("selection_2023_2024", "selection_2025")
                    }
                    records.append({
                        "return_threshold": return_threshold,
                        "accel_threshold": accel_threshold,
                        "context_threshold": context_threshold,
                        "mode": mode,
                        **{
                            f"{name}_{metric}": values[source]
                            for name, values in metrics.items()
                            for metric, source in (
                                ("event_recall", "event_recall"),
                                ("pre_pivot_recall", "pre_pivot_recall"),
                                ("false_per_sector_year", "false_alert_clusters_per_sector_year"),
                                ("precision", "alert_cluster_precision"),
                            )
                        },
                        **{
                            f"{name}_mean_signed_return_20d": values[
                                "mean_signed_relative_return_20d"
                            ] for name, values in economics.items()
                        },
                    })
    sweep = pd.DataFrame(records)
    sweep["worst_event_recall"] = sweep[[
        "selection_2023_2024_event_recall", "selection_2025_event_recall"
    ]].min(axis=1)
    sweep["worst_false"] = sweep[[
        "selection_2023_2024_false_per_sector_year", "selection_2025_false_per_sector_year"
    ]].max(axis=1)
    sweep["worst_signed_return_20d"] = sweep[[
        "selection_2023_2024_mean_signed_return_20d",
        "selection_2025_mean_signed_return_20d",
    ]].min(axis=1)
    eligible = sweep.loc[
        sweep["worst_event_recall"].ge(0.25) & sweep["worst_signed_return_20d"].ge(0)
    ]
    if eligible.empty:
        eligible = sweep.loc[sweep["worst_event_recall"].ge(0.20)]
    if eligible.empty:
        eligible = sweep
    chosen = eligible.sort_values(
        ["worst_false", "worst_signed_return_20d", "worst_event_recall"],
        ascending=[True, False, False],
    ).iloc[0]
    key = (
        float(chosen["return_threshold"]), float(chosen["accel_threshold"]),
        float(chosen["context_threshold"]), str(chosen["mode"]),
    )
    alerts = cache[key]
    results = {
        name: policy_metrics(window(rows, name), alerts.loc[window(rows, name).index])
        for name in WINDOWS
    }
    economics = {name: economic_metrics(rows, alerts, name) for name in WINDOWS}
    frozen_clusters = cluster_rows(rows, alerts)
    frozen_clusters = window(frozen_clusters, "frozen_2026").dropna(
        subset=["signed_forward_return_20"]
    )
    calendar = pd.DatetimeIndex(window(rows, "frozen_2026")["date"].drop_duplicates().sort_values())
    low, high = block_mean_ci(
        frozen_clusters.rename(columns={"date": "signal_date"}), calendar,
        "signed_forward_return_20", samples=1000,
    )
    economics["frozen_2026"]["ci_low"] = low if np.isfinite(low) else None
    economics["frozen_2026"]["ci_high"] = high if np.isfinite(high) else None
    frozen = results["frozen_2026"]
    frozen_economic = economics["frozen_2026"]
    verdict = (
        "REFERENCE_CANDIDATE" if frozen["false_alert_clusters_per_sector_year"] <= 1.25
        and frozen["event_recall"] >= 0.30
        and frozen["alert_cluster_precision"] >= 0.50
        and (frozen_economic["mean_signed_relative_return_20d"] or 0) > 0
        and (frozen["median_lead_to_confirmation_sessions"] or 0) >= 2
        else "NOT_READY"
    )
    output = rows.copy()
    output["confirmation_alert"] = alerts
    return sweep, output, {
        "verdict": verdict,
        "watch_policy": "existing direction-specific reference_alert",
        "confirmation_policy": {
            "mode": key[3], "signed_return_5_min": key[0],
            "signed_acceleration_5_min": key[1],
            "relative_drawdown_60_max" if direction == "ROTATION_IN"
            else "relative_ma200_gap_min": key[2],
        },
        "selection": "2023-24 and 2025: event recall first, then false rate and realized 20d alpha",
        "results": results,
        "economics": economics,
    }


def self_test() -> None:
    rows = pd.DataFrame({
        "reference_alert": [True, True], "relative_return_5": [0.02, -0.02],
        "relative_acceleration_5": [0.01, -0.01], "relative_drawdown_60": [-0.1, -0.1],
        "relative_ma200_gap": [0.1, 0.1],
    })
    assert confirmation_alerts(rows, "ROTATION_IN", 0, 0, -0.05, "BOTH").tolist() == [True, False]
    assert confirmation_alerts(rows, "ROTATION_OUT", 0, 0, 0.05, "BOTH").tolist() == [False, True]
    print("self-test: PASS")


def main() -> None:
    panel = load_panel()
    sweeps, signals, directions = [], [], {}
    for direction, rows in panel.groupby("direction", sort=True):
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
        "objective": "early WATCH plus sparse price-confirmed turning signal",
        "directions": directions,
    }
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    pd.concat(sweeps, ignore_index=True).to_csv(OUTPUT / "confirmation_sweep.csv", index=False)
    pd.concat(signals, ignore_index=True).to_csv(OUTPUT / "confirmation_signals.csv", index=False)
    clean = _json_ready(summary)
    (OUTPUT / "summary.json").write_text(
        json.dumps(clean, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({"output": str(OUTPUT), "verdict": clean["verdict"]}, ensure_ascii=False))


if __name__ == "__main__":
    import sys

    self_test() if "--self-test" in sys.argv else main()
