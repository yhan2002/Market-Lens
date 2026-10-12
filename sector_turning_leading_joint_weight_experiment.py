"""Freeze a five-layer sector policy in 2021-22, then evaluate 2023-26."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from research_drawdown_onset import causal_trailing_percentile
from sector_turning_alert_cooldown_scan import cooldown_alerts
from sector_turning_breadth_shadow import block_mean_ci
from sector_turning_layer_weight_experiment import COMPONENTS, _json_ready
from sector_turning_leading_confirmation_experiment import (
    COVERED_SECTORS,
    economic_ci,
    evaluate,
    false_delta_ci,
)
from sector_turning_model_policy_scan import forward_returns
from sector_turning_pooled_shadow import paired_date_block_pre_pivot_recall_delta_ci
from sector_turning_reference_policy_experiment import confirmed_alerts
from sector_turning_two_stage_confirmation_experiment import cluster_rows


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "outputs" / "sector_turning_layer_weights_20260903_v2" / "scores.csv"
OPTION = ROOT / "outputs" / "sector_turning_leading_confirmation_20260904_v2" / "predictions.csv"
OUTPUT = ROOT / "outputs" / "sector_turning_leading_joint_weight_20260904"
ALL_COMPONENTS = (*COMPONENTS, "option_score")
CUTOFFS = (0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
COOLDOWNS = (21, 42, 63)
WINDOWS = {
    "selection_2021": (pd.Timestamp("2021-01-04"), pd.Timestamp("2021-12-31")),
    "selection_2022": (pd.Timestamp("2022-01-03"), pd.Timestamp("2022-12-30")),
    "evaluation_2023_2026": (pd.Timestamp("2023-01-03"), pd.Timestamp.max),
}


def weight_grid(size: int, units: int = 5) -> list[tuple[float, ...]]:
    return [
        tuple(value / units for value in values)
        for values in product(range(units + 1), repeat=size)
        if sum(values) == units
    ]


def window(rows: pd.DataFrame, name: str) -> pd.DataFrame:
    start, end = WINDOWS[name]
    return rows.loc[rows["date"].between(start, end)]


def load_panel() -> pd.DataFrame:
    dates = ["date", "target_event_confirmation_date"]
    base = pd.read_csv(BASE, parse_dates=dates, dtype={"sector": "string"})
    base["sector"] = base["sector"].str.zfill(4)
    base = base.loc[base["sector"].isin(COVERED_SECTORS)].copy()
    option = pd.read_csv(OPTION, parse_dates=["date"], dtype={"sector": "string"})
    option["sector"] = option["sector"].str.zfill(4)
    option = option.loc[option["model"].eq("OPTION_LOGIT"), [
        "direction", "date", "sector", "score"
    ]].rename(columns={"score": "option_score"})
    keys = ["direction", "date", "sector"]
    panel = base.merge(option, on=keys, how="inner", validate="one_to_one")
    panel = panel.merge(
        forward_returns(), on=["date", "sector", "_position"], how="left", validate="many_to_one"
    ).sort_values(["direction", "sector", "_position"]).reset_index(drop=True)
    panel["signed_forward_return_20"] = panel["forward_relative_return_20"] * np.where(
        panel["direction"].eq("ROTATION_IN"), 1, -1
    )
    if panel.duplicated(keys).any() or panel[list(ALL_COMPONENTS)].isna().any().any():
        raise ValueError("joint panel is incomplete or duplicated")
    return panel


def stack(rows: pd.DataFrame, columns: Sequence[str], weights: Sequence[float]) -> pd.Series:
    raw = sum(rows[column] * weight for column, weight in zip(columns, weights, strict=True))
    ordered = rows.assign(_raw=raw).sort_values(["sector", "_position"])
    score = ordered.groupby("sector", group_keys=False)["_raw"].transform(
        lambda values: causal_trailing_percentile(values, window=504, min_history=60)
    )
    return score.reindex(rows.index)


def period_metrics(rows: pd.DataFrame, alerts: pd.Series, name: str) -> dict[str, Any]:
    sample = window(rows, name)
    renamed = {
        "selection_2021": "selection_2023_2024",
        "selection_2022": "selection_2025",
        "evaluation_2023_2026": "frozen_2026",
    }[name]
    shifted = sample.copy()
    if renamed != name:
        start, end = WINDOWS[name]
        # evaluate() only dispatches fixed windows; preserve its metric definitions locally.
        from sector_turning_reference_policy_experiment import policy_metrics

        result = policy_metrics(shifted, alerts.loc[shifted.index])
        starts = cluster_rows(rows, alerts)
        clusters = starts.loc[starts["date"].between(start, end)].dropna(
            subset=["signed_forward_return_20"]
        )
        result["economic_clusters"] = int(len(clusters))
        result["mean_signed_relative_return_20d"] = (
            float(clusters["signed_forward_return_20"].mean()) if len(clusters) else None
        )
        return result
    return evaluate(rows, alerts, renamed)


def scan(rows: pd.DataFrame, columns: Sequence[str]) -> tuple[pd.DataFrame, dict[tuple[Any, ...], pd.Series]]:
    records, cache = [], {}
    for weights in weight_grid(len(columns)):
        score = stack(rows, columns, weights)
        for cutoff in CUTOFFS:
            high = confirmed_alerts(rows, score, cutoff, 1).astype(bool)
            for cooldown in COOLDOWNS:
                alerts = cooldown_alerts(rows, high, cooldown)
                key = (*weights, cutoff, cooldown)
                cache[key] = alerts
                results = {
                    name: period_metrics(rows, alerts, name)
                    for name in ("selection_2021", "selection_2022")
                }
                records.append({
                    **dict(zip(columns, weights, strict=True)),
                    "cutoff": cutoff,
                    "cooldown_sessions": cooldown,
                    **{
                        f"{name}_{metric}": values[source]
                        for name, values in results.items()
                        for metric, source in (
                            ("recall", "pre_pivot_recall"),
                            ("false", "false_alert_clusters_per_sector_year"),
                            ("precision", "alert_cluster_precision"),
                            ("return20", "mean_signed_relative_return_20d"),
                            ("clusters", "economic_clusters"),
                        )
                    },
                    "_key": key,
                })
    return pd.DataFrame(records), cache


def choose(sweep: pd.DataFrame) -> tuple[pd.Series, bool]:
    sweep = sweep.copy()
    sweep["worst_recall"] = sweep[["selection_2021_recall", "selection_2022_recall"]].min(axis=1)
    sweep["worst_false"] = sweep[["selection_2021_false", "selection_2022_false"]].max(axis=1)
    sweep["worst_return"] = sweep[["selection_2021_return20", "selection_2022_return20"]].min(axis=1)
    sweep["minimum_clusters"] = sweep[["selection_2021_clusters", "selection_2022_clusters"]].min(axis=1)
    eligible = sweep.loc[
        sweep["worst_recall"].ge(0.15)
        & sweep["worst_return"].ge(0)
        & sweep["minimum_clusters"].ge(5)
    ]
    passed = not eligible.empty
    if eligible.empty:
        eligible = sweep.loc[sweep["worst_recall"].ge(0.15) & sweep["minimum_clusters"].ge(5)]
    if eligible.empty:
        eligible = sweep
    return eligible.sort_values(
        ["worst_false", "worst_return", "worst_recall"], ascending=[True, False, False]
    ).iloc[0], passed


def frozen(rows: pd.DataFrame, alerts: pd.Series) -> dict[str, Any]:
    result = period_metrics(rows, alerts, "evaluation_2023_2026")
    clusters = cluster_rows(rows, alerts)
    clusters = window(clusters, "evaluation_2023_2026").dropna(subset=["signed_forward_return_20"])
    calendar = pd.DatetimeIndex(window(rows, "evaluation_2023_2026")["date"].drop_duplicates())
    low, high = block_mean_ci(
        clusters.rename(columns={"date": "signal_date"}), calendar,
        "signed_forward_return_20", samples=2_000,
    )
    result["economics"] = {
        "clusters": int(len(clusters)),
        "mean_signed_relative_return_20d": (
            float(clusters["signed_forward_return_20"].mean()) if len(clusters) else None
        ),
        "ci_low": float(low) if np.isfinite(low) else None,
        "ci_high": float(high) if np.isfinite(high) else None,
    }
    return result


def run() -> dict[str, Any]:
    panel = load_panel()
    outputs, sweeps, directions = [], [], {}
    for direction, rows in panel.groupby("direction", sort=True):
        rows = rows.copy()
        base_sweep, base_cache = scan(rows, COMPONENTS)
        full_sweep, full_cache = scan(rows, ALL_COMPONENTS)
        base, base_passed = choose(base_sweep)
        full, full_passed = choose(full_sweep)
        base_alerts, full_alerts = base_cache[base["_key"]], full_cache[full["_key"]]
        evaluation = window(rows, "evaluation_2023_2026")
        baseline_result, selected_result = frozen(rows, base_alerts), frozen(rows, full_alerts)
        false_ci = false_delta_ci(evaluation, full_alerts, base_alerts)
        recall_ci = paired_date_block_pre_pivot_recall_delta_ci(
            evaluation.assign(score=full_alerts.astype(float)),
            evaluation.assign(score=base_alerts.astype(float)), samples=2_000,
            block_dates=40, calendar_dates=evaluation["date"].drop_duplicates(),
            horizon=10, cutoff=0.5,
        )
        option_weight = float(full["option_score"])
        early = (
            selected_result["eligible_confirmations"] >= 30
            and (selected_result["pre_pivot_recall_ci_low"] or 0) > 0.05
            and (selected_result["median_lead_to_pivot_sessions"] or 0) >= 2
        )
        economic = (selected_result["economics"]["ci_low"] or -np.inf) > 0
        false_reduction = false_ci["ci_high"] < 0
        verdict = (
            "RESEARCH_REFERENCE_CANDIDATE"
            if option_weight > 0 and false_reduction and (early or economic)
            else "NOT_READY"
        )
        rows["baseline_alert"] = base_alerts
        rows["selected_alert"] = full_alerts
        outputs.append(rows)
        base_sweep.insert(0, "model", "FOUR_LAYER_CONTROL")
        full_sweep.insert(0, "model", "FIVE_LAYER_OPTION")
        sweeps.extend([base_sweep.drop(columns="_key"), full_sweep.drop(columns="_key")])
        directions[direction] = {
            "selection_constraints_passed": {"control": base_passed, "option": full_passed},
            "control_policy": {key: _json_ready(base[key]) for key in [*COMPONENTS, "cutoff", "cooldown_sessions"]},
            "option_policy": {key: _json_ready(full[key]) for key in [*ALL_COMPONENTS, "cutoff", "cooldown_sessions"]},
            "evaluation_2023_2026": {
                "selected": selected_result,
                "control": baseline_result,
                "false_alarm_delta": false_ci,
                "recall_delta": recall_ci,
                "gates": {"false_reduction_ci_below_zero": false_reduction, "early_signal": early, "economic_ci_above_zero": economic},
                "verdict": verdict,
            },
        }
    summary = _json_ready({
        "verdict": "RESEARCH_REFERENCE_CANDIDATE" if all(
            value["evaluation_2023_2026"]["verdict"] == "RESEARCH_REFERENCE_CANDIDATE"
            for value in directions.values()
        ) else "RESEARCH_ONLY",
        "pit_status": "PIT_NOT_VERIFIED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective": "jointly weight existing Market Lens layers and lagged options, prioritizing false-alert reduction",
        "selection": "2021 and 2022 separately; frozen evaluation 2023-2026",
        "panel": {"rows": len(panel), "groups": panel["sector"].nunique(), "first_date": str(panel["date"].min().date()), "last_date": str(panel["date"].max().date())},
        "directions": directions,
        "operational_changes": False,
    })
    OUTPUT.mkdir(parents=True, exist_ok=False)
    pd.concat(sweeps, ignore_index=True).to_csv(OUTPUT / "policy_sweep.csv", index=False)
    pd.concat(outputs, ignore_index=True).to_csv(OUTPUT / "signals.csv", index=False)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (OUTPUT / "REPORT.md").write_text(
        "# Joint-weight shadow result\n\n"
        f"Verdict: **{summary['verdict']}**\n\n"
        "2021·2022에서 정책을 고정하고 2023–2026을 평가했습니다. Bloomberg 입력은 PIT 미검증이며 운영 반영은 없습니다.\n",
        encoding="utf-8",
    )
    return summary


def self_test() -> None:
    assert len(weight_grid(4)) == 56
    assert len(weight_grid(5)) == 126
    assert all(abs(sum(weights) - 1) < 1e-12 for weights in weight_grid(5))
    print("self-test: PASS")


if __name__ == "__main__":
    import sys

    if "--self-test" in sys.argv:
        self_test()
    else:
        result = run()
        print(json.dumps({"output": str(OUTPUT), "verdict": result["verdict"]}, ensure_ascii=False))
