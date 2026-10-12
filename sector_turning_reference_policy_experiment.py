"""Select an explainable low-noise alert policy over existing sector layers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from sector_turning_layer_weight_experiment import (
    CUTOFF as BASELINE_CUTOFF,
    COMPONENTS,
    OUTPUT as LAYER_OUTPUT,
    _json_ready,
    metrics,
    period,
    score_stack,
)
from sector_turning_pooled_shadow import _early_event_metrics, _wilson_interval


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs" / "sector_turning_reference_policy_20260903"
TEMPLATES = {
    "PRICE": (0.0, 1.0, 0.0, 0.0),
    "BREADTH": (0.0, 0.0, 1.0, 0.0),
    "MACRO": (0.0, 0.0, 0.0, 1.0),
    "PRICE_BREADTH": (0.0, 0.5, 0.5, 0.0),
    "PRICE_MACRO": (0.0, 0.5, 0.0, 0.5),
    "BREADTH_MACRO": (0.0, 0.0, 0.5, 0.5),
    "PRICE_BREADTH_MACRO": (0.0, 1 / 3, 1 / 3, 1 / 3),
}
CUTOFFS = (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
PERSISTENCE = (1, 2, 3)


def confirmed_alerts(rows: pd.DataFrame, score: pd.Series, cutoff: float, required: int) -> pd.Series:
    ordered = rows.assign(_high=score.ge(cutoff)).sort_values(["sector", "_position"])
    runs = ordered.groupby("sector")["_position"].diff().ne(1).groupby(ordered["sector"]).cumsum()
    counts = ordered.groupby([ordered["sector"], runs], sort=False)["_high"].transform(
        lambda values: values.rolling(3, min_periods=1).sum()
    )
    return counts.ge(required).astype(float).reindex(rows.index)


def policy_metrics(rows: pd.DataFrame, alerts: pd.Series) -> dict:
    evaluated = rows.assign(score=alerts).dropna(subset=["score"])
    result = _early_event_metrics(evaluated, pd.DataFrame(), "", horizon=10, cutoff=0.5)
    low, high = _wilson_interval(result["pre_pivot_hits"], result["eligible_confirmations"])
    result["pre_pivot_recall_ci_low"] = low
    result["pre_pivot_recall_ci_high"] = high
    result["alert_cluster_precision"] = (
        result["matched_alert_clusters"] / result["alert_clusters"]
        if result["alert_clusters"] else 0.0
    )
    return result


def run_direction(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    template_scores = {name: score_stack(rows, weights) for name, weights in TEMPLATES.items()}
    records = []
    for name, score in template_scores.items():
        for cutoff in CUTOFFS:
            for required in PERSISTENCE:
                alerts = confirmed_alerts(rows, score, cutoff, required)
                record = {"template": name, "cutoff": cutoff, "required_of_3": required}
                for split in ("train", "validation"):
                    subset = period(rows, split)
                    result = policy_metrics(subset, alerts.loc[subset.index])
                    record.update({
                        f"{split}_recall": result["pre_pivot_recall"],
                        f"{split}_false_per_sector_year": result["false_alert_clusters_per_sector_year"],
                        f"{split}_precision": result["alert_cluster_precision"],
                        f"{split}_alert_rate": result["alert_day_rate"],
                    })
                records.append(record)
    sweep = pd.DataFrame(records)
    eligible = sweep.loc[
        sweep["train_recall"].ge(0.15)
        & sweep["validation_recall"].ge(0.15)
        & sweep["validation_alert_rate"].le(0.25)
    ].copy()
    if eligible.empty:
        eligible = sweep.copy()
    recall_floor = 0.75 * eligible["validation_recall"].max()
    chosen = eligible.loc[eligible["validation_recall"].ge(recall_floor)].sort_values(
        ["validation_false_per_sector_year", "validation_recall", "validation_precision"],
        ascending=[True, False, False],
    ).iloc[0]
    name, cutoff, required = chosen["template"], float(chosen["cutoff"]), int(chosen["required_of_3"])
    score = template_scores[name]
    alerts = confirmed_alerts(rows, score, cutoff, required)
    final = rows.copy()
    final["reference_score"] = score
    final["reference_alert"] = alerts.astype(bool)
    results = {
        split: policy_metrics(period(rows, split), alerts.loc[period(rows, split).index])
        for split in ("train", "validation", "test")
    }

    baseline_name = max(
        ("PRICE", "BREADTH", "MACRO"),
        key=lambda item: metrics(
            period(rows, "validation"),
            template_scores[item].loc[period(rows, "validation").index],
        )["pre_pivot_recall"],
    )
    baseline_alerts = confirmed_alerts(rows, template_scores[baseline_name], BASELINE_CUTOFF, 1)
    baseline = {
        split: policy_metrics(
            period(rows, split), baseline_alerts.loc[period(rows, split).index]
        ) for split in ("validation", "test")
    }
    test = results["test"]
    verdict = (
        "REFERENCE_CANDIDATE"
        if test["false_alert_clusters_per_sector_year"] <= 1
        and test["pre_pivot_recall"] >= 0.20
        and (test["median_lead_to_pivot_sessions"] or 0) >= 2
        else "NOT_READY"
    )
    return sweep, final, {
        "verdict": verdict,
        "policy": {"template": name, "weights": dict(zip(COMPONENTS, TEMPLATES[name], strict=True)),
                   "cutoff": cutoff, "required_of_trailing_3": required},
        "selection_rule": "retain >=75% of best validation recall, then minimize validation false clusters/year",
        "results": results,
        "baseline": {"template": baseline_name, "cutoff": BASELINE_CUTOFF, "results": baseline},
    }


def self_test() -> None:
    rows = pd.DataFrame({"sector": ["A"] * 5, "_position": [1, 2, 3, 5, 6]})
    scores = pd.Series([0.8, 0.7, 0.9, 0.9, 0.9])
    assert confirmed_alerts(rows, scores, 0.75, 2).tolist() == [False, False, True, False, True]
    print("self-test: PASS")


def main() -> None:
    scores = pd.read_csv(
        LAYER_OUTPUT / "scores.csv", parse_dates=["date", "target_event_confirmation_date"],
        dtype={"sector": "string"},
    )
    scores["sector"] = scores["sector"].str.zfill(4)
    sweeps, outputs, directions = [], [], {}
    for direction, rows in scores.groupby("direction", sort=True):
        sweep, output, result = run_direction(rows.copy())
        sweep.insert(0, "direction", direction)
        sweeps.append(sweep)
        outputs.append(output)
        directions[direction] = result
    summary = {
        "verdict": (
            "REFERENCE_CANDIDATE" if all(v["verdict"] == "REFERENCE_CANDIDATE" for v in directions.values())
            else "RESEARCH_ONLY"
        ),
        "pit_status": "PIT_PARTIAL/CURRENT_REVISED_HISTORY_ONLY",
        "objective": "reduce false alert clusters while retaining useful pre-pivot recall",
        "directions": directions,
    }
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    pd.concat(sweeps, ignore_index=True).to_csv(OUTPUT / "policy_sweep.csv", index=False)
    pd.concat(outputs, ignore_index=True).to_csv(OUTPUT / "reference_signals.csv", index=False)
    clean = _json_ready(summary)
    (OUTPUT / "summary.json").write_text(
        json.dumps(clean, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({"output": str(OUTPUT), "verdict": clean["verdict"]}, ensure_ascii=False))


if __name__ == "__main__":
    import sys

    self_test() if "--self-test" in sys.argv else main()
