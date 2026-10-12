"""Test simple cross-layer agreement as a low-noise turning alert."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from sector_turning_layer_weight_experiment import (
    OUTPUT as LAYER_OUTPUT,
    _json_ready,
    period,
)
from sector_turning_reference_policy_experiment import confirmed_alerts, policy_metrics


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs" / "sector_turning_consensus_policy_20260903"
THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
PERSISTENCE = (1, 2)
RULES = ("PRICE_AND_SUPPORT", "ANY_TWO", "ALL_THREE")


def consensus(rows: pd.DataFrame, rule: str, primary: float, support: float) -> pd.Series:
    price = rows["price_score"].ge(primary)
    breadth = rows["breadth_score"].ge(support)
    macro = rows["macro_score"].ge(support)
    if rule == "PRICE_AND_SUPPORT":
        return price & (breadth | macro)
    if rule == "ANY_TWO":
        return (price.astype(int) + breadth.astype(int) + macro.astype(int)).ge(2)
    if rule == "ALL_THREE":
        return price & breadth & macro
    raise ValueError(rule)


def evaluate(rows: pd.DataFrame, alerts: pd.Series) -> dict[str, dict]:
    return {
        split: policy_metrics(period(rows, split), alerts.loc[period(rows, split).index])
        for split in ("train", "validation", "test")
    }


def run_direction(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    records, alert_cache = [], {}
    for rule in RULES:
        for primary in THRESHOLDS:
            for support in THRESHOLDS:
                if rule in {"ANY_TWO", "ALL_THREE"} and primary != support:
                    continue
                raw = consensus(rows, rule, primary, support).astype(float)
                for required in PERSISTENCE:
                    alerts = confirmed_alerts(rows, raw, 0.5, required).astype(bool)
                    key = (rule, primary, support, required)
                    alert_cache[key] = alerts
                    results = evaluate(rows, alerts)
                    records.append({
                        "rule": rule, "price_threshold": primary,
                        "support_threshold": support, "required_of_3": required,
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
        sweep["train_recall"].ge(0.15)
        & sweep["validation_recall"].ge(0.18)
        & sweep["validation_alert_rate"].le(0.30)
    ]
    if eligible.empty:
        eligible = sweep.loc[
            sweep["train_recall"].ge(0.12) & sweep["validation_recall"].ge(0.15)
        ]
    chosen = eligible.sort_values(
        ["validation_false_per_sector_year", "validation_recall", "train_false_per_sector_year"],
        ascending=[True, False, True],
    ).iloc[0]
    key = (
        str(chosen["rule"]), float(chosen["price_threshold"]),
        float(chosen["support_threshold"]), int(chosen["required_of_3"]),
    )
    alerts = alert_cache[key]
    results = evaluate(rows, alerts)
    test = results["test"]
    verdict = (
        "REFERENCE_CANDIDATE" if test["false_alert_clusters_per_sector_year"] <= 1
        and test["pre_pivot_recall"] >= 0.20
        and (test["median_lead_to_pivot_sessions"] or 0) >= 2
        else "NOT_READY"
    )
    output = rows.copy()
    output["consensus_alert"] = alerts
    return sweep, output, {
        "verdict": verdict,
        "policy": {
            "rule": key[0], "price_threshold": key[1], "support_threshold": key[2],
            "required_of_trailing_3": key[3],
        },
        "selection": "train recall >=15%, validation recall >=18%, then lowest validation false clusters",
        "results": results,
    }


def self_test() -> None:
    rows = pd.DataFrame({
        "price_score": [0.8, 0.8, 0.2],
        "breadth_score": [0.7, 0.2, 0.9],
        "macro_score": [0.1, 0.7, 0.9],
    })
    assert consensus(rows, "PRICE_AND_SUPPORT", 0.6, 0.6).tolist() == [True, True, False]
    assert consensus(rows, "ANY_TWO", 0.6, 0.6).tolist() == [True, True, True]
    assert consensus(rows, "ALL_THREE", 0.6, 0.6).sum() == 0
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
        "objective": "reduce false alarms by requiring agreement across Market Lens layers",
        "directions": directions,
    }
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    pd.concat(sweeps, ignore_index=True).to_csv(OUTPUT / "consensus_sweep.csv", index=False)
    pd.concat(signals, ignore_index=True).to_csv(OUTPUT / "consensus_signals.csv", index=False)
    clean = _json_ready(summary)
    (OUTPUT / "summary.json").write_text(
        json.dumps(clean, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({"output": str(OUTPUT), "verdict": clean["verdict"]}, ensure_ascii=False))


if __name__ == "__main__":
    import sys

    self_test() if "--self-test" in sys.argv else main()
