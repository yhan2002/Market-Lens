"""Compare small causal context sets for filtering existing sector alerts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sector_breadth_features import MODEL_FEATURES as BREADTH_FEATURES
from sector_turning_false_positive_filter_experiment import (
    BASE_POLICY,
    future_pre_pivot_label,
)
from sector_turning_layer_weight_experiment import COMPONENTS, _json_ready, score_stack
from sector_turning_pooled_shadow import PRICE_FEATURES
from sector_turning_raw_false_positive_filter_experiment import (
    RAW_FEATURES,
    load_scores_with_raw_features,
)
from sector_turning_reference_policy_experiment import confirmed_alerts, policy_metrics
from sector_turning_stability_weight_experiment import WINDOWS, window


ROOT = Path(__file__).resolve().parent
BREADTH_CSV = ROOT / "outputs" / "sector_breadth_features_full_20260903_v2" / "sector_breadth_features.csv"
OUTPUT = ROOT / "outputs" / "sector_turning_context_filter_20260903"
FEATURE_SETS = {
    "PRICE": PRICE_FEATURES,
    "PRICE_BREADTH": (*PRICE_FEATURES, *BREADTH_FEATURES),
    "PRICE_MACRO_NETWORK": RAW_FEATURES,
    "ALL_CONTEXT": (*RAW_FEATURES, *BREADTH_FEATURES),
}
REGULARIZATION = (0.01, 0.1, 1.0)
CUTOFFS = tuple(np.arange(0.20, 0.81, 0.05).round(2))


def load_panel() -> pd.DataFrame:
    scores = load_scores_with_raw_features()
    breadth = pd.read_csv(BREADTH_CSV, parse_dates=["date"], dtype={"sector": "string"})
    breadth["sector"] = breadth["sector"].str.zfill(4)
    return scores.merge(
        breadth[["date", "sector", *BREADTH_FEATURES]],
        on=["date", "sector"], how="left", validate="many_to_one",
    )


def evaluate_windows(rows: pd.DataFrame, alerts: pd.Series) -> dict[str, dict]:
    return {
        name: policy_metrics(window(rows, name), alerts.loc[window(rows, name).index])
        for name in WINDOWS
    }


def run_direction(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    direction = str(rows["direction"].iloc[0])
    weights, base_cutoff = BASE_POLICY[direction]
    base_score = score_stack(rows, weights)
    base_alert = confirmed_alerts(rows, base_score, base_cutoff, 1).astype(bool)
    baseline = evaluate_windows(rows, base_alert)
    target = future_pre_pivot_label(rows)
    train = rows["date"].le("2022-12-30")
    records, cache, models = [], {}, {}
    for feature_name, columns in FEATURE_SETS.items():
        complete = rows.loc[:, columns].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
        train_candidates = base_alert & complete & train
        if target.loc[train_candidates].nunique() != 2:
            continue
        for c_value in REGULARIZATION:
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=c_value, class_weight="balanced", max_iter=2000),
            ).fit(rows.loc[train_candidates, columns], target.loc[train_candidates])
            probability = pd.Series(np.nan, index=rows.index)
            probability.loc[complete] = model.predict_proba(rows.loc[complete, columns])[:, 1]
            models[(feature_name, c_value)] = model
            for cutoff in CUTOFFS:
                alerts = base_alert & probability.ge(cutoff)
                key = (feature_name, c_value, cutoff)
                cache[key] = (alerts, probability)
                metrics = evaluate_windows(rows, alerts)
                records.append({
                    "feature_set": feature_name, "regularization_c": c_value,
                    "probability_cutoff": cutoff,
                    **{
                        f"{name}_{metric}": values[source]
                        for name, values in metrics.items()
                        if name != "frozen_2026"
                        for metric, source in (
                            ("recall", "pre_pivot_recall"),
                            ("false_per_sector_year", "false_alert_clusters_per_sector_year"),
                            ("precision", "alert_cluster_precision"),
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
    eligible = sweep.loc[sweep["worst_selection_recall"].ge(0.15)]
    if eligible.empty:
        eligible = sweep
    chosen = eligible.sort_values(
        ["worst_selection_false", "worst_selection_recall"], ascending=[True, False]
    ).iloc[0]
    key = (
        str(chosen["feature_set"]), float(chosen["regularization_c"]),
        float(chosen["probability_cutoff"]),
    )
    alerts, probability = cache[key]
    results = evaluate_windows(rows, alerts)
    columns = FEATURE_SETS[key[0]]
    coefficients = dict(zip(
        columns, models[(key[0], key[1])].named_steps["logisticregression"].coef_[0], strict=True
    ))
    frozen = results["frozen_2026"]
    verdict = (
        "REFERENCE_CANDIDATE" if frozen["false_alert_clusters_per_sector_year"] <= 1.25
        and frozen["pre_pivot_recall"] >= 0.20
        and frozen["alert_cluster_precision"] >= 0.50
        and (frozen["median_lead_to_pivot_sessions"] or 0) >= 2
        else "NOT_READY"
    )
    output = rows.copy()
    output["base_alert"] = base_alert
    output["context_probability"] = probability
    output["context_alert"] = alerts
    return sweep, output, {
        "verdict": verdict,
        "base_policy": {"weights": dict(zip(COMPONENTS, weights, strict=True)), "cutoff": base_cutoff},
        "filter": {"feature_set": key[0], "regularization_c": key[1],
                   "probability_cutoff": key[2], "features": list(columns)},
        "selection": "minimize worst 2023-24/2025 false rate with recall >=15% in both",
        "coefficients_standardized": coefficients,
        "results": results,
        "baseline": baseline,
    }


def self_test() -> None:
    assert len(FEATURE_SETS["ALL_CONTEXT"]) == len(set(FEATURE_SETS["ALL_CONTEXT"])) == 22
    assert set(BREADTH_FEATURES).issubset(FEATURE_SETS["ALL_CONTEXT"])
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
        "objective": "low-noise filter using raw price, direct breadth, and sector macro context",
        "directions": directions,
    }
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    pd.concat(sweeps, ignore_index=True).to_csv(OUTPUT / "context_sweep.csv", index=False)
    pd.concat(signals, ignore_index=True).to_csv(OUTPUT / "context_signals.csv", index=False)
    clean = _json_ready(summary)
    (OUTPUT / "summary.json").write_text(
        json.dumps(clean, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({"output": str(OUTPUT), "verdict": clean["verdict"]}, ensure_ascii=False))


if __name__ == "__main__":
    import sys

    self_test() if "--self-test" in sys.argv else main()
