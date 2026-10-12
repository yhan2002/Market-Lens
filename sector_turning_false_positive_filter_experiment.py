"""Filter existing sector alerts with one explainable logistic model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sector_turning_layer_weight_experiment import (
    COMPONENTS,
    OUTPUT as LAYER_OUTPUT,
    _json_ready,
    period,
    score_stack,
)
from sector_turning_pooled_shadow import _event_capture_outcomes
from sector_turning_reference_policy_experiment import confirmed_alerts, policy_metrics


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs" / "sector_turning_false_positive_filter_20260903"
BASE_POLICY = {
    "ROTATION_IN": ((0.0, 0.5, 0.5, 0.0), 0.90),
    "ROTATION_OUT": ((0.0, 0.5, 0.0, 0.5), 0.70),
}
PROBABILITY_CUTOFFS = tuple(np.arange(0.20, 0.81, 0.05).round(2))
REGULARIZATION = (0.1, 1.0)


def feature_frame(rows: pd.DataFrame) -> pd.DataFrame:
    ordered = rows.sort_values(["sector", "_position"])
    output = ordered.loc[:, COMPONENTS].copy()
    grouped = ordered.groupby("sector", sort=False)
    for column in COMPONENTS:
        for lag in (1, 3, 5):
            output[f"{column}_delta_{lag}"] = grouped[column].diff(lag)
    output["support_count_60"] = ordered.loc[:, COMPONENTS[1:]].ge(0.60).sum(axis=1)
    return output.reindex(rows.index).fillna(0.0)


def future_pre_pivot_label(rows: pd.DataFrame) -> pd.Series:
    events = _event_capture_outcomes(rows.assign(score=0.0), horizon=10, cutoff=0.5)
    label = pd.Series(0, index=rows.index, dtype=int)
    for sector, sector_rows in rows.groupby("sector", sort=False):
        sector_events = events.loc[events["sector"].astype(str).eq(str(sector))]
        for event in sector_events.itertuples():
            hit = sector_rows["_position"].lt(event.pivot_position) & sector_rows[
                "_position"
            ].between(event.confirmation_position - 10, event.confirmation_position - 1)
            label.loc[sector_rows.index[hit]] = 1
    return label


def evaluate(rows: pd.DataFrame, alerts: pd.Series) -> dict[str, dict]:
    return {
        split: policy_metrics(period(rows, split), alerts.loc[period(rows, split).index])
        for split in ("train", "validation", "test")
    }


def run_direction(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    direction = str(rows["direction"].iloc[0])
    weights, base_cutoff = BASE_POLICY[direction]
    base_score = score_stack(rows, weights)
    base_alert = confirmed_alerts(rows, base_score, base_cutoff, 1).astype(bool)
    baseline = evaluate(rows, base_alert)
    features = feature_frame(rows)
    target = future_pre_pivot_label(rows)
    train = period(rows, "train")
    validation = period(rows, "validation")
    train_candidates = train.index.intersection(rows.index[base_alert & rows.index.isin(train.index)])
    if target.loc[train_candidates].nunique() != 2:
        raise RuntimeError("training candidates do not contain both outcomes")

    records, models, probabilities = [], {}, {}
    for c_value in REGULARIZATION:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c_value, class_weight="balanced", max_iter=2000),
        )
        model.fit(features.loc[train_candidates], target.loc[train_candidates])
        probability = pd.Series(model.predict_proba(features)[:, 1], index=rows.index)
        models[c_value], probabilities[c_value] = model, probability
        for cutoff in PROBABILITY_CUTOFFS:
            alerts = base_alert & probability.ge(cutoff)
            results = evaluate(rows, alerts)
            records.append({
                "regularization_c": c_value, "probability_cutoff": cutoff,
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
    recall_floor = max(0.15, 0.75 * baseline["validation"]["pre_pivot_recall"])
    eligible = sweep.loc[
        sweep["train_recall"].ge(0.12)
        & sweep["validation_recall"].ge(recall_floor)
        & sweep["validation_alert_rate"].le(baseline["validation"]["alert_day_rate"])
    ]
    if eligible.empty:
        eligible = sweep
    chosen = eligible.sort_values(
        ["validation_false_per_sector_year", "validation_recall", "train_false_per_sector_year"],
        ascending=[True, False, True],
    ).iloc[0]
    c_value, cutoff = float(chosen["regularization_c"]), float(chosen["probability_cutoff"])
    probability = probabilities[c_value]
    alerts = base_alert & probability.ge(cutoff)
    results = evaluate(rows, alerts)
    logistic = models[c_value].named_steps["logisticregression"]
    coefficients = dict(zip(features.columns, logistic.coef_[0], strict=True))
    test = results["test"]
    verdict = (
        "REFERENCE_CANDIDATE" if test["false_alert_clusters_per_sector_year"] <= 1
        and test["pre_pivot_recall"] >= 0.20
        and (test["median_lead_to_pivot_sessions"] or 0) >= 2
        else "NOT_READY"
    )
    output = rows.copy()
    output["base_alert"] = base_alert
    output["filter_probability"] = probability
    output["filtered_alert"] = alerts
    output["future_pre_pivot_label"] = target
    return sweep, output, {
        "verdict": verdict,
        "base_policy": {"weights": dict(zip(COMPONENTS, weights, strict=True)), "cutoff": base_cutoff},
        "filter": {"model": "standardized balanced logistic regression", "regularization_c": c_value,
                   "probability_cutoff": cutoff, "features": list(features.columns)},
        "selection": "train-only fit; validation retains >=75% baseline recall then minimizes false clusters",
        "coefficients_standardized": coefficients,
        "candidate_rows": {
            "train": int(base_alert.loc[train.index].sum()),
            "validation": int(base_alert.loc[validation.index].sum()),
            "train_positive_days": int(target.loc[train_candidates].sum()),
        },
        "results": results,
        "baseline": baseline,
    }


def self_test() -> None:
    rows = pd.DataFrame({
        "sector": ["A"] * 5,
        "_position": range(5),
        "target_event_confirmation_position": [4] * 5,
        "target_event_confirmation_date": pd.to_datetime(["2026-01-05"] * 5),
        "target_event_pivot_position": [3] * 5,
    })
    label = future_pre_pivot_label(rows)
    assert label.tolist() == [1, 1, 1, 0, 0]
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
        "objective": "remove false positives from the existing Market Lens sector alert",
        "directions": directions,
    }
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    pd.concat(sweeps, ignore_index=True).to_csv(OUTPUT / "filter_sweep.csv", index=False)
    pd.concat(signals, ignore_index=True).to_csv(OUTPUT / "filtered_signals.csv", index=False)
    clean = _json_ready(summary)
    (OUTPUT / "summary.json").write_text(
        json.dumps(clean, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({"output": str(OUTPUT), "verdict": clean["verdict"]}, ensure_ascii=False))


if __name__ == "__main__":
    import sys

    self_test() if "--self-test" in sys.argv else main()
