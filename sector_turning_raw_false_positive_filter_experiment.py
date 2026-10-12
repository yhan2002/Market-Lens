"""Filter Market Lens alerts with causal price and sector-macro features."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sector_turning_false_positive_filter_experiment import (
    BASE_POLICY,
    evaluate,
    future_pre_pivot_label,
)
from sector_turning_layer_weight_experiment import (
    COMPONENTS,
    OUTPUT as LAYER_OUTPUT,
    _json_ready,
    period,
    score_stack,
)
from sector_turning_pooled_shadow import PRICE_FEATURES, build_pooled_daily_frame
from sector_turning_reference_policy_experiment import confirmed_alerts
from sector_turning_sector_macro_boost import (
    DRIVERS,
    NETWORK_FEATURES,
    attach_sector_features,
    build_factor_features,
)


ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT / "outputs" / "gics_l2_staging_20260902_145743" / "bloomberg_gics_l2_20260902_145743"
MACRO_CSV = ROOT / "outputs" / "bloomberg_turning_macro_staging_20260902_212424" / "bloomberg_turning_macro_20260902_212424" / "macro_history.csv"
DATABASE = ROOT / "regime_router_publish.db"
OUTPUT = ROOT / "outputs" / "sector_turning_raw_false_positive_filter_20260903"
RAW_FEATURES = (
    *PRICE_FEATURES,
    *(f"{driver}_impact_5" for driver in DRIVERS),
    "cross_section_rank_5", "cross_section_rank_20",
    "cross_section_dispersion_20", "peer_corr_change_63_252",
)
PROBABILITY_CUTOFFS = tuple(np.arange(0.20, 0.81, 0.05).round(2))
REGULARIZATION = (0.01, 0.1, 1.0)


def load_scores_with_raw_features() -> pd.DataFrame:
    scores = pd.read_csv(
        LAYER_OUTPUT / "scores.csv", parse_dates=["date", "target_event_confirmation_date"],
        dtype={"sector": "string"},
    )
    scores["sector"] = scores["sector"].str.zfill(4)
    pooled, _, _ = build_pooled_daily_frame(PACKAGE, None, horizon=10)
    raw = attach_sector_features(pooled, build_factor_features(MACRO_CSV, DATABASE))
    return scores.merge(
        raw[["date", "sector", *RAW_FEATURES]], on=["date", "sector"],
        how="left", validate="many_to_one",
    )


def run_direction(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    direction = str(rows["direction"].iloc[0])
    weights, base_cutoff = BASE_POLICY[direction]
    base_score = score_stack(rows, weights)
    base_alert = confirmed_alerts(rows, base_score, base_cutoff, 1).astype(bool)
    baseline = evaluate(rows, base_alert)
    target = future_pre_pivot_label(rows)
    complete = rows.loc[:, RAW_FEATURES].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    train = period(rows, "train")
    validation = period(rows, "validation")
    train_candidates = rows.index[base_alert & complete & rows.index.isin(train.index)]
    if target.loc[train_candidates].nunique() != 2:
        raise RuntimeError("training candidates do not contain both outcomes")

    records, models, probabilities = [], {}, {}
    for c_value in REGULARIZATION:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c_value, class_weight="balanced", max_iter=2000),
        )
        model.fit(rows.loc[train_candidates, RAW_FEATURES], target.loc[train_candidates])
        probability = pd.Series(np.nan, index=rows.index)
        probability.loc[complete] = model.predict_proba(rows.loc[complete, RAW_FEATURES])[:, 1]
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
    coefficients = dict(zip(RAW_FEATURES, logistic.coef_[0], strict=True))
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
                   "probability_cutoff": cutoff, "features": list(RAW_FEATURES)},
        "selection": "train-only fit; validation retains >=75% baseline recall then minimizes false clusters",
        "coefficients_standardized": coefficients,
        "complete_feature_rows": int(complete.sum()),
        "candidate_rows": {"train": int(len(train_candidates)),
                           "validation": int((base_alert & complete & rows.index.isin(validation.index)).sum()),
                           "train_positive_days": int(target.loc[train_candidates].sum())},
        "results": results,
        "baseline": baseline,
    }


def self_test() -> None:
    assert len(RAW_FEATURES) == len(set(RAW_FEATURES)) == 18
    assert set(PRICE_FEATURES).issubset(RAW_FEATURES)
    print("self-test: PASS")


def main() -> None:
    scores = load_scores_with_raw_features()
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
        "objective": "filter existing alerts using causal price and sector-macro context",
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
