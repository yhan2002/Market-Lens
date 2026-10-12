"""Turn continuous sector scores into sparse alerts with hysteresis."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from sector_turning_layer_weight_experiment import (
    OUTPUT as LAYER_OUTPUT,
    _json_ready,
    period,
    score_stack,
)
from sector_turning_reference_policy_experiment import TEMPLATES, confirmed_alerts, policy_metrics


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs" / "sector_turning_hysteresis_policy_20260903"
TRIGGERS = (0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
RESETS = (0.30, 0.40, 0.50, 0.60, 0.70)
HOLDS = (1, 3, 5)


def hysteresis_alerts(
    rows: pd.DataFrame, score: pd.Series, trigger: float, reset: float, hold: int
) -> pd.Series:
    ordered = rows.assign(_score=score).sort_values(["sector", "_position"])
    output = pd.Series(False, index=ordered.index)
    for _, group in ordered.groupby("sector", sort=False):
        armed, remaining, previous = True, 0, None
        for index, position, value in group[["_position", "_score"]].itertuples():
            if previous is not None and position != previous + 1:
                armed, remaining = True, 0
            if not armed and value <= reset:
                armed = True
            if armed and value >= trigger:
                armed, remaining = False, hold
            if remaining:
                output.loc[index] = True
                remaining -= 1
            previous = position
    return output.reindex(rows.index)


def evaluate(rows: pd.DataFrame, alerts: pd.Series) -> dict[str, dict]:
    return {
        split: policy_metrics(period(rows, split), alerts.loc[period(rows, split).index])
        for split in ("train", "validation", "test")
    }


def run_direction(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    template_scores = {name: score_stack(rows, weights) for name, weights in TEMPLATES.items()}
    records, cache = [], {}
    for name, score in template_scores.items():
        for trigger in TRIGGERS:
            for reset in RESETS:
                if reset >= trigger:
                    continue
                for hold in HOLDS:
                    key = (name, trigger, reset, hold)
                    alerts = hysteresis_alerts(rows, score, trigger, reset, hold)
                    cache[key] = alerts
                    results = evaluate(rows, alerts)
                    records.append({
                        "template": name, "trigger": trigger, "reset": reset, "hold": hold,
                        **{
                            f"{split}_{metric}": results[split][source]
                            for split in ("train", "validation")
                            for metric, source in (
                                ("recall", "pre_pivot_recall"),
                                ("false_per_sector_year", "false_alert_clusters_per_sector_year"),
                                ("precision", "alert_cluster_precision"),
                                ("alert_rate", "alert_day_rate"),
                            )
                        },
                    })
    sweep = pd.DataFrame(records)
    eligible = sweep.loc[
        sweep["train_recall"].ge(0.15) & sweep["validation_recall"].ge(0.18)
    ].copy()
    if eligible.empty:
        eligible = sweep.loc[
            sweep["train_recall"].ge(0.12) & sweep["validation_recall"].ge(0.15)
        ].copy()
    recall_floor = max(0.18, 0.80 * eligible["validation_recall"].max())
    chosen = eligible.loc[eligible["validation_recall"].ge(recall_floor)].sort_values(
        ["validation_false_per_sector_year", "validation_recall", "train_false_per_sector_year"],
        ascending=[True, False, True],
    ).iloc[0]
    key = (
        str(chosen["template"]), float(chosen["trigger"]),
        float(chosen["reset"]), int(chosen["hold"]),
    )
    alerts = cache[key]
    results = evaluate(rows, alerts)
    baseline_score = template_scores[key[0]]
    baseline_alerts = confirmed_alerts(rows, baseline_score, key[1], 1).astype(bool)
    baseline = evaluate(rows, baseline_alerts)
    test = results["test"]
    verdict = (
        "REFERENCE_CANDIDATE" if test["false_alert_clusters_per_sector_year"] <= 1
        and test["pre_pivot_recall"] >= 0.20
        and (test["median_lead_to_pivot_sessions"] or 0) >= 2
        else "NOT_READY"
    )
    output = rows.copy()
    output["hysteresis_alert"] = alerts
    return sweep, output, {
        "verdict": verdict,
        "policy": {
            "template": key[0], "weights": dict(zip(
                ("rrg_score", "price_score", "breadth_score", "macro_score"),
                TEMPLATES[key[0]], strict=True,
            )),
            "trigger": key[1], "reset": key[2], "active_sessions": key[3],
        },
        "selection": "retain >=80% of best validation recall, then minimize validation false clusters",
        "results": results,
        "same_template_continuous_baseline": baseline,
    }


def self_test() -> None:
    rows = pd.DataFrame({"sector": ["A"] * 8, "_position": range(8)})
    scores = pd.Series([0.2, 0.8, 0.9, 0.7, 0.4, 0.8, 0.4, 0.8])
    alerts = hysteresis_alerts(rows, scores, 0.75, 0.4, 1)
    assert alerts.tolist() == [False, True, False, False, False, True, False, True]
    held = hysteresis_alerts(rows, scores, 0.75, 0.4, 3)
    assert held.tolist() == [False, True, True, True, False, True, True, True]
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
        "objective": "reduce repeated false alerts while preserving early turning capture",
        "directions": directions,
    }
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    pd.concat(sweeps, ignore_index=True).to_csv(OUTPUT / "hysteresis_sweep.csv", index=False)
    pd.concat(signals, ignore_index=True).to_csv(OUTPUT / "hysteresis_signals.csv", index=False)
    clean = _json_ready(summary)
    (OUTPUT / "summary.json").write_text(
        json.dumps(clean, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({"output": str(OUTPUT), "verdict": clean["verdict"]}, ensure_ascii=False))


if __name__ == "__main__":
    import sys

    self_test() if "--self-test" in sys.argv else main()
