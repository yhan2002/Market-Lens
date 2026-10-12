"""Keep price as the watch signal; use macro and breadth only as causal confirmation."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research_drawdown_onset import causal_trailing_percentile
from sector_breadth_features import MODEL_FEATURES as BREADTH_FEATURES
from sector_turning_alert_cooldown_scan import cooldown_alerts
from sector_turning_breadth_shadow import block_mean_ci
from sector_turning_layer_weight_experiment import _json_ready
from sector_turning_leading_confirmation_experiment import false_delta_ci
from sector_turning_leading_joint_weight_experiment import WINDOWS, window
from sector_turning_model_policy_scan import forward_returns
from sector_turning_pooled_shadow import paired_date_block_pre_pivot_recall_delta_ci
from sector_turning_raw_false_positive_filter_experiment import load_scores_with_raw_features
from sector_turning_reference_policy_experiment import confirmed_alerts, policy_metrics
from sector_turning_sector_macro_boost import DRIVERS
from sector_turning_two_stage_confirmation_experiment import cluster_rows


ROOT = Path(__file__).resolve().parent
BREADTH_CSV = (
    ROOT / "outputs" / "sector_breadth_features_full_20260903_v2" / "sector_breadth_features.csv"
)
OUTPUT = ROOT / "outputs" / "sector_turning_price_context_confirmation_20260904"
MACRO_IMPACTS = tuple(f"{driver}_impact_5" for driver in DRIVERS)
PRICE_FLOORS = (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
CONTEXT_CUTOFFS = (0.40, 0.50, 0.60, 0.70)
MODES = ("MACRO_OR_BREADTH", "MACRO_AND_BREADTH", "MACRO_ONLY", "BREADTH_ONLY")
BASELINE_CUTOFF = 0.90
COOLDOWN = 63


def causal_rank(rows: pd.DataFrame, values: pd.Series, min_history: int = 60) -> pd.Series:
    ordered = rows.assign(_value=values).sort_values(["sector", "_position"])
    ranked = ordered.groupby("sector", group_keys=False)["_value"].transform(
        lambda series: causal_trailing_percentile(series, window=504, min_history=min_history)
    )
    return ranked.reindex(rows.index)


def context_scores(
    rows: pd.DataFrame, direction: str, min_history: int = 60
) -> tuple[pd.Series, pd.Series]:
    sign = 1 if direction == "ROTATION_IN" else -1
    macro = causal_rank(rows, sign * rows.loc[:, MACRO_IMPACTS].sum(axis=1), min_history)
    breadth_parts = [
        causal_rank(rows, sign * rows[column], min_history) for column in BREADTH_FEATURES
    ]
    breadth = pd.concat(breadth_parts, axis=1).mean(axis=1, skipna=False)
    return macro, breadth


def supported_alerts(
    rows: pd.DataFrame,
    price_floor: float,
    macro_cutoff: float,
    breadth_cutoff: float,
    mode: str,
) -> pd.Series:
    price_watch = confirmed_alerts(rows, rows["price_score"], price_floor, 1).astype(bool)
    macro = rows["macro_context_score"].ge(macro_cutoff)
    breadth = rows["breadth_context_score"].ge(breadth_cutoff)
    support = {
        "MACRO_OR_BREADTH": macro | breadth,
        "MACRO_AND_BREADTH": macro & breadth,
        "MACRO_ONLY": macro,
        "BREADTH_ONLY": breadth,
    }[mode]
    return cooldown_alerts(rows, price_watch & support, COOLDOWN)


def load_panel(breadth_csv: Path) -> pd.DataFrame:
    rows = load_scores_with_raw_features()
    breadth = pd.read_csv(breadth_csv, parse_dates=["date"], dtype={"sector": "string"})
    breadth["sector"] = breadth["sector"].str.zfill(4)
    if breadth.duplicated(["date", "sector"]).any():
        raise ValueError("duplicate breadth date/sector rows")
    rows = rows.merge(
        breadth[["date", "sector", *BREADTH_FEATURES]],
        on=["date", "sector"], how="inner", validate="many_to_one",
    ).merge(
        forward_returns(), on=["date", "sector", "_position"], how="left", validate="many_to_one",
    )
    required = ["price_score", *MACRO_IMPACTS, *BREADTH_FEATURES, "target"]
    rows = rows.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
    rows["signed_forward_return_20"] = rows["forward_relative_return_20"] * np.where(
        rows["direction"].eq("ROTATION_IN"), 1, -1
    )
    if rows.duplicated(["direction", "date", "sector"]).any():
        raise ValueError("duplicate direction/date/sector rows")
    return rows.sort_values(["direction", "sector", "_position"]).reset_index(drop=True)


def period_metrics(rows: pd.DataFrame, alerts: pd.Series, name: str) -> dict[str, Any]:
    sample = window(rows, name)
    return policy_metrics(sample, alerts.loc[sample.index])


def scan(rows: pd.DataFrame, baseline: pd.Series) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    baseline_metrics = {
        name: period_metrics(rows, baseline, name) for name in ("selection_2021", "selection_2022")
    }
    records: list[dict[str, Any]] = []
    cache: dict[tuple[Any, ...], pd.Series] = {}
    for mode in MODES:
        macro_values = CONTEXT_CUTOFFS if "MACRO" in mode else (0.0,)
        breadth_values = CONTEXT_CUTOFFS if "BREADTH" in mode else (0.0,)
        for price_floor in PRICE_FLOORS:
            for macro_cutoff in macro_values:
                for breadth_cutoff in breadth_values:
                    key = (price_floor, macro_cutoff, breadth_cutoff, mode)
                    alerts = supported_alerts(rows, *key)
                    cache[key] = alerts
                    metrics = {
                        name: period_metrics(rows, alerts, name)
                        for name in ("selection_2021", "selection_2022")
                    }
                    records.append({
                        "price_floor": price_floor,
                        "macro_cutoff": macro_cutoff if "MACRO" in mode else np.nan,
                        "breadth_cutoff": breadth_cutoff if "BREADTH" in mode else np.nan,
                        "mode": mode,
                        **{
                            f"{name}_{metric}": values[source]
                            for name, values in metrics.items()
                            for metric, source in (
                                ("recall", "pre_pivot_recall"),
                                ("clusters", "alert_clusters"),
                                ("false", "false_alert_clusters_per_sector_year"),
                                ("precision", "alert_cluster_precision"),
                            )
                        },
                        "_key": key,
                    })
    sweep = pd.DataFrame(records)
    eligible = sweep.copy()
    for name in ("selection_2021", "selection_2022"):
        eligible = eligible.loc[
            eligible[f"{name}_clusters"].le(baseline_metrics[name]["alert_clusters"])
        ]
    if eligible.empty:
        raise RuntimeError("no context policy fits the price-only alert budget")
    eligible = eligible.assign(
        worst_recall=eligible[["selection_2021_recall", "selection_2022_recall"]].min(axis=1),
        mean_recall=eligible[["selection_2021_recall", "selection_2022_recall"]].mean(axis=1),
        worst_precision=eligible[["selection_2021_precision", "selection_2022_precision"]].min(axis=1),
        worst_false=eligible[["selection_2021_false", "selection_2022_false"]].max(axis=1),
    )
    chosen = eligible.sort_values(
        ["worst_recall", "mean_recall", "worst_precision", "worst_false"],
        ascending=[False, False, False, True],
    ).iloc[0]
    return sweep, cache[chosen["_key"]], chosen


def economic_ci(rows: pd.DataFrame, alerts: pd.Series, samples: int) -> dict[str, Any]:
    sample = window(cluster_rows(rows, alerts), "evaluation_2023_2026").dropna(
        subset=["signed_forward_return_20"]
    ).rename(columns={"date": "signal_date"})
    calendar = pd.DatetimeIndex(
        window(rows, "evaluation_2023_2026")["date"].drop_duplicates().sort_values()
    )
    low, high = block_mean_ci(sample, calendar, "signed_forward_return_20", samples=samples)
    return {
        "clusters": int(len(sample)),
        "mean_signed_relative_return_20d": (
            float(sample["signed_forward_return_20"].mean()) if len(sample) else None
        ),
        "ci_low": float(low) if np.isfinite(low) else None,
        "ci_high": float(high) if np.isfinite(high) else None,
    }


def evaluate(
    rows: pd.DataFrame, candidate: pd.Series, baseline: pd.Series, samples: int
) -> dict[str, Any]:
    evaluation = window(rows, "evaluation_2023_2026")
    selected = period_metrics(rows, candidate, "evaluation_2023_2026")
    control = period_metrics(rows, baseline, "evaluation_2023_2026")
    recall_delta = paired_date_block_pre_pivot_recall_delta_ci(
        evaluation.assign(score=candidate.loc[evaluation.index].astype(float)),
        evaluation.assign(score=baseline.loc[evaluation.index].astype(float)),
        samples=samples, block_dates=40, calendar_dates=evaluation["date"].drop_duplicates(),
        horizon=10, cutoff=0.5,
    )
    false_delta = false_delta_ci(
        evaluation, candidate.loc[evaluation.index], baseline.loc[evaluation.index], samples=samples
    )
    within_budget = selected["alert_clusters"] <= 1.05 * control["alert_clusters"]
    statistical = (recall_delta["early_capture_ci_low"] or -np.inf) > 0 and within_budget
    early = (
        selected["eligible_confirmations"] >= 30
        and (selected["pre_pivot_recall_ci_low"] or 0) > 0.05
        and (selected["median_lead_to_pivot_sessions"] or 0) >= 2
        and selected["pre_pivot_recall"] >= control["pre_pivot_recall"]
        and within_budget
    )
    return {
        "selected": selected,
        "price_only": control,
        "recall_delta": recall_delta,
        "false_alarm_delta": false_delta,
        "economics": {
            "selected": economic_ci(rows, candidate, samples),
            "price_only": economic_ci(rows, baseline, samples),
        },
        "gates": {
            "evaluation_alert_budget_within_5pct": within_budget,
            "recall_delta_ci_above_zero": statistical,
            "early_signal_without_significant_uplift": early,
        },
        "verdict": (
            "STATISTICAL_CAPTURE_UPLIFT_RESEARCH_ONLY" if statistical
            else "EARLY_SIGNAL_CANDIDATE_RESEARCH_ONLY" if early
            else "NO_RELIABLE_UPLIFT"
        ),
    }


def report(summary: dict[str, Any]) -> str:
    def percent(value: float | None) -> str:
        return "N/A" if value is None else f"{value:.1%}"

    lines = [
        "# Price watch + context confirmation",
        "",
        f"Verdict: **{summary['verdict']}**",
        "",
        "가격은 조기 경보의 중심으로 유지하고, 섹터별 매크로 영향과 내부 breadth는 낮은 가격 문턱을 허용하는 확인 조건으로만 사용했습니다.",
        "후보는 2021년과 2022년 각각 가격 단독보다 경보가 많지 않도록 선택했습니다.",
        "",
        "| 방향 | 정책 | 사전 포착 | 가격 단독 | 차이 95% CI | 선행일 | 경보 수 | 가격 단독 | 판정 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for direction, item in summary["directions"].items():
        selected, control, delta = item["evaluation"]["selected"], item["evaluation"]["price_only"], item["evaluation"]["recall_delta"]
        policy = item["policy"]
        lines.append(
            f"| {direction} | {policy['mode']} · price≥{policy['price_floor']:.2f} | "
            f"{selected['pre_pivot_recall']:.1%} | {control['pre_pivot_recall']:.1%} | "
            f"[{percent(delta['early_capture_ci_low'])}, {percent(delta['early_capture_ci_high'])}] | "
            f"{selected['median_lead_to_pivot_sessions'] or 0:.1f} | {selected['alert_clusters']} | "
            f"{control['alert_clusters']} | {item['evaluation']['verdict']} |"
        )
    lines.extend([
        "",
        "주의: 2023~2026 구간은 반복 검토된 연구 백테스트이며 진정한 미관측 OOS가 아닙니다. Bloomberg breadth와 매크로 이력도 PIT가 완전히 검증되지 않았습니다.",
    ])
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    panel = load_panel(args.breadth_csv.resolve())
    directions: dict[str, Any] = {}
    signals, sweeps = [], []
    for direction, rows in panel.groupby("direction", sort=True):
        rows = rows.copy()
        rows["macro_context_score"], rows["breadth_context_score"] = context_scores(rows, direction)
        rows = rows.dropna(subset=["macro_context_score", "breadth_context_score"])
        baseline = cooldown_alerts(
            rows, confirmed_alerts(rows, rows["price_score"], BASELINE_CUTOFF, 1).astype(bool), COOLDOWN
        )
        sweep, candidate, chosen = scan(rows, baseline)
        rows["price_watch_alert"] = baseline
        rows["context_confirmed_alert"] = candidate
        signals.append(rows)
        clean_sweep = sweep.drop(columns="_key")
        clean_sweep.insert(0, "direction", direction)
        sweeps.append(clean_sweep)
        directions[direction] = {
            "policy": {
                "price_floor": float(chosen["price_floor"]),
                "macro_cutoff": None if pd.isna(chosen["macro_cutoff"]) else float(chosen["macro_cutoff"]),
                "breadth_cutoff": None if pd.isna(chosen["breadth_cutoff"]) else float(chosen["breadth_cutoff"]),
                "mode": str(chosen["mode"]),
                "cooldown_sessions": COOLDOWN,
            },
            "selection_rule": "maximize worst 2021/2022 pre-pivot recall without exceeding price-only alert clusters in either year",
            "evaluation": evaluate(rows, candidate, baseline, args.bootstrap_samples),
        }
    verdicts = [item["evaluation"]["verdict"] for item in directions.values()]
    verdict = (
        "STATISTICAL_CAPTURE_UPLIFT_RESEARCH_ONLY" if any(value.startswith("STATISTICAL") for value in verdicts)
        else "EARLY_SIGNAL_CANDIDATE_RESEARCH_ONLY" if any(value.startswith("EARLY") for value in verdicts)
        else "NO_RELIABLE_UPLIFT"
    )
    summary = _json_ready({
        "verdict": verdict,
        "pit_status": "PIT_PARTIAL/CURRENT_REVISED_HISTORY_ONLY",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective": "price-led early turning watch with sector macro and breadth confirmation",
        "selection": "2021 and 2022 separately; reviewed evaluation 2023-2026",
        "panel": {
            "rows": len(panel), "groups": panel["sector"].nunique(),
            "first_date": str(panel["date"].min().date()), "last_date": str(panel["date"].max().date()),
        },
        "directions": directions,
        "operational_changes": False,
    })
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    pd.concat(sweeps, ignore_index=True).to_csv(output_dir / "policy_sweep.csv", index=False)
    pd.concat(signals, ignore_index=True).to_csv(output_dir / "signals.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    (output_dir / "REPORT.md").write_text(report(summary), encoding="utf-8")
    return summary


def self_test() -> None:
    dates = pd.bdate_range("2026-01-01", periods=5)
    rows = pd.DataFrame({"sector": ["A"] * 5, "_position": range(5)})
    for index, column in enumerate(MACRO_IMPACTS):
        rows[column] = np.arange(5, dtype=float) + index
    for column in BREADTH_FEATURES:
        rows[column] = np.arange(5, dtype=float)
    before = context_scores(rows, "ROTATION_IN", min_history=2)
    rows.loc[4, MACRO_IMPACTS[0]] = -999
    after = context_scores(rows, "ROTATION_IN", min_history=2)
    assert before[0].iloc[:4].equals(after[0].iloc[:4])
    assert before[0].notna().sum() == before[1].notna().sum() == 3
    assert len(dates) == len(rows)
    print("self-test: PASS")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--breadth-csv", type=Path, default=BREADTH_CSV)
    result.add_argument("--output-dir", type=Path, default=OUTPUT)
    result.add_argument("--bootstrap-samples", type=int, default=2_000)
    result.add_argument("--self-test", action="store_true")
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    if arguments.self_test:
        self_test()
    else:
        result = run(arguments)
        print(json.dumps({"output": str(arguments.output_dir.resolve()), "verdict": result["verdict"]}))
