"""Shrink sector turning policies by economic family without touching live paths."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from sector_turning_layer_weight_experiment import (
    COMPONENTS,
    OUTPUT as LAYER_OUTPUT,
    _json_ready,
    period,
    score_stack,
    weight_grid,
)
from sector_turning_reference_policy_experiment import confirmed_alerts, policy_metrics


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs" / "sector_turning_group_policy_20260903"
GROUP_FOR_TICKER = {
    "XLE": "CYCLICAL", "XLB": "CYCLICAL", "XLI": "CYCLICAL", "XLY": "CYCLICAL",
    "XLP": "DEFENSIVE", "XLV": "DEFENSIVE", "XLU": "DEFENSIVE",
    "XLF": "RATE", "XLRE": "RATE", "XLK": "GROWTH", "XLC": "GROWTH",
}
GLOBAL_POLICY = {
    "ROTATION_IN": ((0.0, 0.5, 0.5, 0.0), 0.90, 1),
    "ROTATION_OUT": ((0.0, 0.5, 0.0, 0.5), 0.70, 1),
}
CUTOFFS = (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
PERSISTENCE = (1, 2, 3)
MIN_TRAIN_EVENTS = 25
MIN_VALIDATION_EVENTS = 15


def evaluate(rows: pd.DataFrame, alerts: pd.Series) -> dict[str, dict]:
    return {
        split: policy_metrics(period(rows, split), alerts.loc[period(rows, split).index])
        for split in ("train", "validation", "test")
    }


def tune_group(
    rows: pd.DataFrame,
    score_by_weight: dict[tuple[float, ...], pd.Series],
    baseline: tuple[tuple[float, ...], float, int],
) -> tuple[pd.DataFrame, pd.Series, dict]:
    weights0, cutoff0, required0 = baseline
    base_score = score_by_weight.get(weights0)
    if base_score is None:
        base_score = score_stack(rows, weights0)
    base_alerts = confirmed_alerts(rows, base_score.loc[rows.index], cutoff0, required0)
    base_results = evaluate(rows, base_alerts)
    support = {
        split: base_results[split]["eligible_confirmations"]
        for split in ("train", "validation", "test")
    }
    if support["train"] < MIN_TRAIN_EVENTS or support["validation"] < MIN_VALIDATION_EVENTS:
        return pd.DataFrame(), base_alerts, {
            "adopted": False,
            "reason": "insufficient train/validation event support; global fallback",
            "support": support,
            "policy": policy_dict(weights0, cutoff0, required0),
            "baseline": base_results,
            "results": base_results,
        }

    records = []
    for weights, full_score in score_by_weight.items():
        score = full_score.loc[rows.index]
        for cutoff in CUTOFFS:
            for required in PERSISTENCE:
                alerts = confirmed_alerts(rows, score, cutoff, required)
                train, validation = evaluate_train_validation(rows, alerts)
                records.append({
                    **dict(zip(COMPONENTS, weights, strict=True)),
                    "cutoff": cutoff,
                    "required_of_3": required,
                    "train_recall": train["pre_pivot_recall"],
                    "train_false_per_sector_year": train["false_alert_clusters_per_sector_year"],
                    "validation_recall": validation["pre_pivot_recall"],
                    "validation_false_per_sector_year": validation["false_alert_clusters_per_sector_year"],
                    "validation_precision": validation["alert_cluster_precision"],
                    "validation_alert_rate": validation["alert_day_rate"],
                })
    sweep = pd.DataFrame(records)
    base_train, base_valid = base_results["train"], base_results["validation"]
    eligible = sweep.loc[
        sweep["train_recall"].ge(max(0.15, base_train["pre_pivot_recall"] - 0.05))
        & sweep["validation_recall"].ge(max(0.15, base_valid["pre_pivot_recall"] - 0.05))
        & sweep["validation_alert_rate"].le(base_valid["alert_day_rate"])
    ]
    if eligible.empty:
        return sweep, base_alerts, {
            "adopted": False, "reason": "no policy preserved the recall floor",
            "support": support, "policy": policy_dict(weights0, cutoff0, required0),
            "baseline": base_results, "results": base_results,
        }
    chosen = eligible.sort_values(
        ["validation_false_per_sector_year", "validation_recall", "train_false_per_sector_year"],
        ascending=[True, False, True],
    ).iloc[0]
    weights = tuple(float(chosen[name]) for name in COMPONENTS)
    cutoff, required = float(chosen["cutoff"]), int(chosen["required_of_3"])
    candidate_alerts = confirmed_alerts(
        rows, score_by_weight[weights].loc[rows.index], cutoff, required
    )
    candidate_results = evaluate(rows, candidate_alerts)
    candidate_train, candidate_valid = candidate_results["train"], candidate_results["validation"]
    adopted = (
        candidate_valid["false_alert_clusters_per_sector_year"]
        <= 0.85 * base_valid["false_alert_clusters_per_sector_year"]
        and candidate_train["false_alert_clusters_per_sector_year"]
        <= base_train["false_alert_clusters_per_sector_year"]
    )
    return sweep, candidate_alerts if adopted else base_alerts, {
        "adopted": adopted,
        "reason": "validation false clusters improved >=15% without train deterioration"
        if adopted else "candidate improvement did not survive the train/validation guard",
        "support": support,
        "policy": policy_dict(weights, cutoff, required) if adopted
        else policy_dict(weights0, cutoff0, required0),
        "candidate": {"policy": policy_dict(weights, cutoff, required), "results": candidate_results},
        "baseline": base_results,
        "results": candidate_results if adopted else base_results,
    }


def evaluate_train_validation(
    rows: pd.DataFrame, alerts: pd.Series
) -> tuple[dict, dict]:
    train = period(rows, "train")
    validation = period(rows, "validation")
    return (
        policy_metrics(train, alerts.loc[train.index]),
        policy_metrics(validation, alerts.loc[validation.index]),
    )


def policy_dict(weights: tuple[float, ...], cutoff: float, required: int) -> dict:
    return {
        "weights": dict(zip(COMPONENTS, weights, strict=True)),
        "cutoff": cutoff,
        "required_of_trailing_3": required,
    }


def run_direction(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rows = rows.copy()
    rows["economic_group"] = rows["parent_ticker"].map(GROUP_FOR_TICKER)
    if rows["economic_group"].isna().any():
        raise ValueError("unmapped parent sector ticker")
    score_by_weight = {weights: score_stack(rows, weights) for weights in weight_grid()}
    baseline = GLOBAL_POLICY[str(rows["direction"].iloc[0])]
    if baseline[0] not in score_by_weight:
        score_by_weight[baseline[0]] = score_stack(rows, baseline[0])

    combined = pd.Series(False, index=rows.index)
    sweeps, groups = [], {}
    for name, group_rows in rows.groupby("economic_group", sort=True):
        sweep, alerts, result = tune_group(group_rows, score_by_weight, baseline)
        if not sweep.empty:
            sweep.insert(0, "economic_group", name)
            sweeps.append(sweep)
        combined.loc[group_rows.index] = alerts.astype(bool)
        groups[name] = result
    results = {
        split: policy_metrics(period(rows, split), combined.loc[period(rows, split).index])
        for split in ("train", "validation", "test")
    }
    global_alerts = confirmed_alerts(
        rows, score_by_weight[baseline[0]], baseline[1], baseline[2]
    )
    global_results = evaluate(rows, global_alerts)
    test = results["test"]
    verdict = (
        "REFERENCE_CANDIDATE" if test["false_alert_clusters_per_sector_year"] <= 1
        and test["pre_pivot_recall"] >= 0.20
        and (test["median_lead_to_pivot_sessions"] or 0) >= 2
        else "NOT_READY"
    )
    output = rows.copy()
    output["group_policy_alert"] = combined
    output["global_policy_alert"] = global_alerts.astype(bool)
    return pd.concat(sweeps, ignore_index=True) if sweeps else pd.DataFrame(), output, {
        "verdict": verdict,
        "selection": "economic-family policy adopted only after >=15% validation false-cluster reduction and no train deterioration",
        "global_policy": policy_dict(*baseline),
        "groups": groups,
        "results": results,
        "global_results": global_results,
    }


def self_test() -> None:
    assert set(GROUP_FOR_TICKER) == {"XLE", "XLB", "XLI", "XLY", "XLP", "XLV", "XLU", "XLF", "XLRE", "XLK", "XLC"}
    assert all(abs(sum(policy[0]) - 1) < 1e-12 for policy in GLOBAL_POLICY.values())
    print("self-test: PASS")


def main() -> None:
    scores = pd.read_csv(
        LAYER_OUTPUT / "scores.csv", parse_dates=["date", "target_event_confirmation_date"],
        dtype={"sector": "string"},
    )
    scores["sector"] = scores["sector"].str.zfill(4)
    sweeps, signals, directions = [], [], {}
    for direction, rows in scores.groupby("direction", sort=True):
        sweep, output, result = run_direction(rows)
        if not sweep.empty:
            sweep.insert(0, "direction", direction)
            sweeps.append(sweep)
        signals.append(output)
        directions[direction] = result
    summary = {
        "verdict": "REFERENCE_CANDIDATE" if all(
            item["verdict"] == "REFERENCE_CANDIDATE" for item in directions.values()
        ) else "RESEARCH_ONLY",
        "pit_status": "PIT_PARTIAL/CURRENT_REVISED_HISTORY_ONLY",
        "objective": "reduce false alarms with shrinkage across the Market Lens layer lineage",
        "directions": directions,
    }
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    if sweeps:
        pd.concat(sweeps, ignore_index=True).to_csv(OUTPUT / "group_policy_sweep.csv", index=False)
    pd.concat(signals, ignore_index=True).to_csv(OUTPUT / "group_signals.csv", index=False)
    clean = _json_ready(summary)
    (OUTPUT / "summary.json").write_text(
        json.dumps(clean, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({"output": str(OUTPUT), "verdict": clean["verdict"]}, ensure_ascii=False))


if __name__ == "__main__":
    import sys

    self_test() if "--self-test" in sys.argv else main()
