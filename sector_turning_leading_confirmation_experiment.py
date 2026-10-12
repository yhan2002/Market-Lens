"""Test Bloomberg option confirmation against the frozen Market Lens sector stack."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from research_drawdown_onset import causal_trailing_percentile
from sector_macro_trend_hint_experiment import PACKAGE
from sector_turning_alert_cooldown_scan import cooldown_alerts
from sector_turning_breadth_shadow import block_mean_ci
from sector_turning_layer_weight_experiment import _json_ready
from sector_turning_reference_policy_experiment import confirmed_alerts, policy_metrics
from sector_turning_stability_weight_experiment import WINDOWS, window
from sector_turning_tradable_label_shadow import qualify_labels
from sector_turning_two_stage_confirmation_experiment import cluster_rows
from sector_turning_pooled_shadow import (
    _direction_frame,
    build_pooled_daily_frame,
    effect_code,
    paired_date_block_auc_delta_ci,
    paired_date_block_pre_pivot_recall_delta_ci,
    purged_training_mask,
)


ROOT = Path(__file__).resolve().parent
HISTORY = ROOT / "outputs" / "bloomberg_sector_option_full_staging_20260904_184703" / "merged_history.csv"
BASELINE = ROOT / "outputs" / "sector_turning_integrated_cooldown_20260903" / "integrated_signals.csv"
OUTPUT = ROOT / "outputs" / "sector_turning_option_full_20260904"
ETF_TO_SECTORS = {
    "XLE": ("1010",),
    "XLB": ("1510",),
    "XLI": ("2010", "2020", "2030"),
    "XLY": ("2510", "2520", "2530", "2550"),
    "XLP": ("3010", "3020", "3030"),
    "XLV": ("3510", "3520"),
    "XLF": ("4010", "4020", "4030"),
    "IGV": ("4510",),
    "XLK": ("4520",),
    "SOXX": ("4530",),
    "XLC": ("5010", "5020"),
    "XLU": ("5510",),
    "XLRE": ("6010",),
}
COVERED_SECTORS = frozenset(sector for sectors in ETF_TO_SECTORS.values() for sector in sectors)
TRIN_READY = frozenset({"1010", "2010", "4510", "5510"})
OPTION_FIELDS = ("RK324", "RK327", "RK293", "RK298", "PQ950")
OPTION_FEATURES = (
    "option_iv_rel_z252",
    "option_iv_rel_change_5",
    "option_iv_rel_accel_5",
    "option_term_rel_z252",
    "option_term_rel_change_5",
    "option_oi_rel_z252",
    "option_oi_rel_change_5",
)
TRIN_FEATURES = (
    "trin_log_z252",
    "trin_log_change_1",
    "trin_log_change_5",
    "trin_log_accel_5",
)
GATE_CUTOFFS = tuple(np.arange(0.20, 0.801, 0.10).round(2))
BLEND_WEIGHTS = tuple(np.arange(0.10, 0.501, 0.10).round(2))
BLEND_CUTOFFS = tuple(np.arange(0.60, 0.951, 0.05).round(2))
COOLDOWNS = (21, 42, 63)


def rolling_z(values: pd.Series, window_size: int = 252, minimum: int = 126) -> pd.Series:
    mean = values.rolling(window_size, min_periods=minimum).mean()
    scale = values.rolling(window_size, min_periods=minimum).std()
    return (values - mean).div(scale.where(scale.gt(1e-12)))


def load_leading_features(history_csv: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    history = pd.read_csv(history_csv, parse_dates=["date"], dtype={"code": "string"})
    required = {"data_group", "code", "security", "field_id", "date", "value"}
    if missing := sorted(required - set(history.columns)):
        raise ValueError("missing Bloomberg columns: " + ", ".join(missing))
    history["value"] = pd.to_numeric(history["value"], errors="raise")
    if history.duplicated(["security", "field_id", "date"]).any():
        raise ValueError("duplicate Bloomberg security/field/date rows")
    if not np.isfinite(history["value"]).all():
        raise ValueError("non-finite Bloomberg values")

    option = history.loc[history["field_id"].isin(OPTION_FIELDS)].copy()
    option["code"] = option["code"].astype(str)
    pivot = option.pivot(index=["date", "code"], columns="field_id", values="value")
    if set(pivot.columns) != set(OPTION_FIELDS):
        raise ValueError("option fields are incomplete")
    incomplete_option_dates = int(pivot.isna().any(axis=1).sum())
    pivot = pivot.dropna(subset=list(OPTION_FIELDS))
    expected_codes = {*ETF_TO_SECTORS, "SPY"}
    if set(pivot.index.get_level_values("code")) != expected_codes:
        raise ValueError("option ETF universe does not match the registered cohort")
    if pivot.isna().any().any():
        raise ValueError("option fields are incomplete")
    if pivot.loc[:, OPTION_FIELDS].le(0).any().any():
        raise ValueError("option fields must be positive before log transforms")

    equal_rates = {
        "iv30_call_put": float(np.isclose(pivot["RK324"], pivot["RK327"]).mean()),
        "iv3m_call_put": float(np.isclose(pivot["RK293"], pivot["RK298"]).mean()),
    }
    transformed = pd.DataFrame(index=pivot.index)
    iv30 = pivot[["RK324", "RK327"]].mean(axis=1)
    iv3m = pivot[["RK293", "RK298"]].mean(axis=1)
    transformed["iv30"] = np.log(iv30)
    transformed["term"] = np.log(iv3m / iv30)
    transformed["oi"] = np.log(pivot["PQ950"])
    transformed = transformed.groupby(level="code").transform(
        lambda values: values.rolling(5, min_periods=3).median()
    )
    spy = transformed.xs("SPY", level="code").add_suffix("_spy")
    sector = transformed.loc[
        transformed.index.get_level_values("code").isin(ETF_TO_SECTORS)
    ].reset_index().join(spy, on="date")
    sector["sector"] = sector.pop("code").map(ETF_TO_SECTORS)
    sector = sector.explode("sector", ignore_index=True)
    sector["iv_rel"] = sector["iv30"] - sector["iv30_spy"]
    sector["term_rel"] = sector["term"] - sector["term_spy"]
    sector["oi_rel"] = sector["oi"] - sector["oi_spy"]

    option_parts = []
    for _, rows in sector.groupby("sector", sort=False):
        rows = rows.sort_values("date").copy()
        rows["option_iv_rel_z252"] = rolling_z(rows["iv_rel"])
        rows["option_iv_rel_change_5"] = rows["iv_rel"].diff(5)
        rows["option_iv_rel_accel_5"] = rows["iv_rel"].diff(5).diff(5)
        rows["option_term_rel_z252"] = rolling_z(rows["term_rel"])
        rows["option_term_rel_change_5"] = rows["term_rel"].diff(5)
        rows["option_oi_rel_z252"] = rolling_z(rows["oi_rel"])
        rows["option_oi_rel_change_5"] = rows["oi_rel"].diff(5)
        option_parts.append(rows[["date", "sector", *OPTION_FEATURES]])
    features = pd.concat(option_parts, ignore_index=True)

    trin = history.loc[history["field_id"].eq("IX116"), ["date", "code", "value"]].copy()
    trin["sector"] = trin.pop("code").astype(str).str.zfill(4)
    if trin["value"].le(0).any():
        raise ValueError("TRIN must be positive before log transform")
    trin_parts = []
    for _, rows in trin.groupby("sector", sort=False):
        rows = rows.sort_values("date").copy()
        log_value = np.log(rows["value"])
        rows["trin_log_z252"] = rolling_z(log_value)
        rows["trin_log_change_1"] = log_value.diff()
        rows["trin_log_change_5"] = log_value.diff(5)
        rows["trin_log_accel_5"] = log_value.diff(5).diff(5)
        trin_parts.append(rows[["date", "sector", *TRIN_FEATURES]])
    features = features.merge(
        pd.concat(trin_parts, ignore_index=True),
        on=["date", "sector"], how="left", validate="one_to_one",
    ).sort_values(["sector", "date"]).reset_index(drop=True)
    if features.duplicated(["date", "sector"]).any():
        raise ValueError("duplicate engineered date/sector rows")
    return features, {
        "source_rows": int(len(history)),
        "option_series": int(len(expected_codes) * len(OPTION_FIELDS)),
        "incomplete_option_dates_excluded": incomplete_option_dates,
        "option_complete_rows": int(features.loc[:, OPTION_FEATURES].notna().all(axis=1).sum()),
        "put_call_exact_equal_rate": equal_rates,
        "causal_smoothing": "trailing 5-observation median before strict next-session alignment",
        "trin_ready_groups": sorted(TRIN_READY),
        "timing": "Bloomberg EOD date t is usable only on the first market date strictly after t",
        "fill_or_interpolation": False,
    }


def attach_features(pooled: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for sector, left in pooled.loc[pooled["sector"].astype(str).isin(COVERED_SECTORS)].groupby(
        "sector", sort=False
    ):
        right = features.loc[features["sector"].eq(str(sector))].drop(columns="sector").rename(
            columns={"date": "leading_asof_date"}
        )
        merged = pd.merge_asof(
            left.sort_values("date"), right.sort_values("leading_asof_date"),
            left_on="date", right_on="leading_asof_date", direction="backward",
            allow_exact_matches=False, tolerance=pd.Timedelta(days=7),
        )
        parts.append(merged)
    result = pd.concat(parts, ignore_index=True).sort_values(["date", "sector"])
    known = result["leading_asof_date"].notna()
    if not (result.loc[known, "leading_asof_date"] < result.loc[known, "date"]).all():
        raise AssertionError("Bloomberg feature timing is not strictly lagged")
    return result.reset_index(drop=True)


def design(
    frame: pd.DataFrame,
    columns: Sequence[str],
    scaler: StandardScaler,
    levels: Sequence[str],
) -> pd.DataFrame:
    continuous = pd.DataFrame(
        scaler.transform(frame.loc[:, columns]), index=frame.index, columns=columns
    )
    effects, _ = effect_code(frame["sector"], levels)
    return continuous.join(effects)


def walk_forward(
    pooled: pd.DataFrame,
    direction: str,
    columns: Sequence[str],
    model_name: str,
    *,
    min_train_dates: int,
    refit_every: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    qualified, label_audit = qualify_labels(pooled, direction)
    frame = _direction_frame(qualified, direction, target_mode="PRE_PIVOT")
    complete = frame.loc[:, columns].notna().all(axis=1)
    scorable = frame["at_risk"].astype(bool) & complete
    dates = pd.DatetimeIndex(frame.loc[scorable, "date"].drop_duplicates().sort_values())
    levels = tuple(sorted(frame["sector"].astype(str).unique()))
    output_parts: list[pd.DataFrame] = []
    coefficient_parts: list[pd.DataFrame] = []
    fitted = 0
    for start in range(0, len(dates), refit_every):
        block_dates = dates[start : start + refit_every]
        prediction_date = pd.Timestamp(block_dates[0])
        train = frame.loc[purged_training_mask(frame, prediction_date) & complete]
        test = frame.loc[scorable & frame["date"].isin(block_dates)]
        if train["date"].nunique() < min_train_dates or train["target"].nunique() < 2 or test.empty:
            continue
        scaler = StandardScaler().fit(train.loc[:, columns])
        train_design = design(train, columns, scaler, levels)
        test_design = design(test, columns, scaler, levels)
        model = LogisticRegression(
            C=0.1, class_weight="balanced", max_iter=2_000, random_state=42
        ).fit(train_design, train["target"].astype(int))
        scored = test[[
            "date", "_position", "sector", "sector_name", "target",
            "target_event_pivot_position", "target_event_confirmation_position",
            "target_event_confirmation_date", "relative_price", "leading_asof_date",
        ]].copy()
        scored.insert(0, "direction", direction)
        scored.insert(1, "model", model_name)
        scored["refit_date"] = prediction_date
        scored["training_end_date"] = train["date"].max()
        scored["raw_score"] = model.predict_proba(test_design)[:, 1]
        output_parts.append(scored)
        coefficient_parts.append(pd.DataFrame({
            "direction": direction,
            "model": model_name,
            "refit_date": prediction_date,
            "factor": train_design.columns,
            "coefficient": model.coef_[0],
        }))
        fitted += 1
    if not output_parts:
        raise RuntimeError(f"no predictions for {direction}/{model_name}")
    output = pd.concat(output_parts, ignore_index=True).sort_values(["sector", "date"])
    output["score"] = output.groupby("sector", sort=False)["raw_score"].transform(
        lambda values: causal_trailing_percentile(values, window=504, min_history=60)
    )
    output = output.dropna(subset=["score"]).reset_index(drop=True)
    return output, pd.concat(coefficient_parts, ignore_index=True), {
        "model": model_name,
        "features": list(columns),
        "groups": list(levels),
        "fitted_blocks": fitted,
        "first_score_date": str(output["date"].min().date()),
        "last_score_date": str(output["date"].max().date()),
        "label_audit": label_audit,
    }


def as_bool(values: pd.Series) -> pd.Series:
    return values if pd.api.types.is_bool_dtype(values) else values.astype(str).str.lower().eq("true")


def load_comparison_panel(option_scores: pd.DataFrame, baseline_csv: Path) -> pd.DataFrame:
    baseline = pd.read_csv(
        baseline_csv, parse_dates=["date", "target_event_confirmation_date"],
        dtype={"sector": "string"},
    )
    baseline["sector"] = baseline["sector"].str.zfill(4)
    baseline = baseline.loc[baseline["sector"].isin(COVERED_SECTORS)].copy()
    baseline["baseline_alert"] = as_bool(baseline.pop("integrated_alert"))
    scores = option_scores.rename(columns={"score": "option_score", "target": "option_target"})
    panel = baseline.merge(
        scores[["direction", "date", "sector", "option_score", "option_target"]],
        on=["direction", "date", "sector"], how="inner", validate="one_to_one",
    )
    if not np.allclose(panel["target"], panel["option_target"]):
        raise AssertionError("baseline and option labels do not match")
    return panel.drop(columns="option_target").sort_values(["direction", "sector", "_position"])


def evaluate(rows: pd.DataFrame, alerts: pd.Series, period_name: str) -> dict[str, Any]:
    sample = window(rows, period_name)
    metrics = policy_metrics(sample, alerts.loc[sample.index])
    clusters = window(cluster_rows(rows, alerts), period_name).dropna(
        subset=["signed_forward_return_20"]
    )
    return {
        **metrics,
        "economic_clusters": int(len(clusters)),
        "mean_signed_relative_return_20d": (
            float(clusters["signed_forward_return_20"].mean()) if len(clusters) else None
        ),
    }


def candidate_grid(rows: pd.DataFrame) -> tuple[pd.DataFrame, dict[tuple[Any, ...], pd.Series]]:
    cache: dict[tuple[Any, ...], pd.Series] = {("BASELINE",): rows["baseline_alert"].astype(bool)}
    for cutoff in GATE_CUTOFFS:
        cache[("OPTION_GATE", cutoff)] = rows["baseline_alert"].astype(bool) & rows[
            "option_score"
        ].ge(cutoff)
    for weight, cutoff, cooldown in product(BLEND_WEIGHTS, BLEND_CUTOFFS, COOLDOWNS):
        score = (1 - weight) * rows["stable_score"] + weight * rows["option_score"]
        base = confirmed_alerts(rows, score, cutoff, 1).astype(bool)
        cache[("BLEND", weight, cutoff, cooldown)] = cooldown_alerts(rows, base, cooldown)

    records = []
    for key, alerts in cache.items():
        results = {
            name: evaluate(rows, alerts, name)
            for name in ("selection_2023_2024", "selection_2025")
        }
        record: dict[str, Any] = {
            "mode": key[0],
            "option_gate_cutoff": key[1] if key[0] == "OPTION_GATE" else np.nan,
            "option_weight": key[1] if key[0] == "BLEND" else np.nan,
            "score_cutoff": key[2] if key[0] == "BLEND" else np.nan,
            "cooldown_sessions": key[3] if key[0] == "BLEND" else np.nan,
        }
        for name, values in results.items():
            for output, source in (
                ("recall", "pre_pivot_recall"),
                ("false", "false_alert_clusters_per_sector_year"),
                ("precision", "alert_cluster_precision"),
                ("lead", "median_lead_to_pivot_sessions"),
                ("return20", "mean_signed_relative_return_20d"),
                ("clusters", "economic_clusters"),
            ):
                record[f"{name}_{output}"] = values[source]
        record["_key"] = key
        records.append(record)
    return pd.DataFrame(records), cache


def choose_candidate(sweep: pd.DataFrame) -> tuple[pd.Series, dict[str, float]]:
    baseline = sweep.loc[sweep["mode"].eq("BASELINE")].iloc[0]
    floors = {
        name: round(min(
            float(baseline[f"{name}_recall"]),
            max(0.15, 0.80 * float(baseline[f"{name}_recall"])),
        ), 12)
        for name in ("selection_2023_2024", "selection_2025")
    }
    eligible = sweep.loc[
        sweep["selection_2023_2024_recall"].ge(floors["selection_2023_2024"])
        & sweep["selection_2025_recall"].ge(floors["selection_2025"])
        & sweep["selection_2023_2024_false"].le(baseline["selection_2023_2024_false"])
        & sweep["selection_2025_false"].le(baseline["selection_2025_false"])
    ].copy()
    eligible["worst_false"] = eligible[[
        "selection_2023_2024_false", "selection_2025_false"
    ]].max(axis=1)
    eligible["mean_false"] = eligible[[
        "selection_2023_2024_false", "selection_2025_false"
    ]].mean(axis=1)
    eligible["worst_recall"] = eligible[[
        "selection_2023_2024_recall", "selection_2025_recall"
    ]].min(axis=1)
    chosen = eligible.sort_values(
        ["worst_false", "mean_false", "worst_recall"], ascending=[True, True, False]
    ).iloc[0]
    return chosen, floors


def false_delta_ci(
    rows: pd.DataFrame,
    candidate: pd.Series,
    baseline: pd.Series,
    *,
    samples: int = 2_000,
) -> dict[str, Any]:
    records = []
    years = rows["date"].dt.year
    for (sector, year), sample in rows.groupby(["sector", years], sort=True):
        candidate_metrics = policy_metrics(sample, candidate.loc[sample.index])
        baseline_metrics = policy_metrics(sample, baseline.loc[sample.index])
        records.append({
            "sector": sector,
            "year": int(year),
            "delta": candidate_metrics["false_alert_clusters"] - baseline_metrics["false_alert_clusters"],
        })
    values = pd.DataFrame(records)["delta"].to_numpy(dtype=float)
    rng = np.random.default_rng(42)
    estimates = np.array([
        rng.choice(values, size=len(values), replace=True).mean() for _ in range(samples)
    ])
    return {
        "delta_false_clusters_per_sector_year": float(values.mean()),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
        "sector_year_units": int(len(values)),
        "bootstrap_samples": samples,
    }


def economic_ci(rows: pd.DataFrame, alerts: pd.Series) -> dict[str, Any]:
    sample = window(cluster_rows(rows, alerts), "frozen_2026").dropna(
        subset=["signed_forward_return_20"]
    ).rename(columns={"date": "signal_date"})
    calendar = pd.DatetimeIndex(window(rows, "frozen_2026")["date"].drop_duplicates().sort_values())
    low, high = block_mean_ci(sample, calendar, "signed_forward_return_20", samples=1_000)
    return {
        "clusters": int(len(sample)),
        "mean_signed_relative_return_20d": (
            float(sample["signed_forward_return_20"].mean()) if len(sample) else None
        ),
        "ci_low": float(low) if np.isfinite(low) else None,
        "ci_high": float(high) if np.isfinite(high) else None,
    }


def evaluate_selected(
    rows: pd.DataFrame,
    selected: pd.Series,
    baseline: pd.Series,
) -> dict[str, Any]:
    selected_result = evaluate(rows, selected, "frozen_2026")
    baseline_result = evaluate(rows, baseline, "frozen_2026")
    selected_frame = window(rows.assign(score=selected.astype(float)), "frozen_2026")
    baseline_frame = window(rows.assign(score=baseline.astype(float)), "frozen_2026")
    recall_delta = paired_date_block_pre_pivot_recall_delta_ci(
        selected_frame, baseline_frame, samples=1_000, block_dates=40,
        calendar_dates=selected_frame["date"].drop_duplicates(), horizon=10, cutoff=0.5,
    )
    false_ci = false_delta_ci(selected_frame, selected, baseline)
    economics = economic_ci(rows, selected)
    early_gate = (
        selected_result["eligible_confirmations"] >= 10
        and (selected_result["pre_pivot_recall_ci_low"] or 0) > 0.05
        and (selected_result["median_lead_to_pivot_sessions"] or 0) >= 2
        and selected_result["pre_pivot_recall"] >= max(
            0.15, 0.80 * baseline_result["pre_pivot_recall"]
        )
    )
    significant_false_reduction = false_ci["ci_high"] < 0
    economic_gate = economics["ci_low"] is not None and economics["ci_low"] > 0
    return {
        "selected": selected_result,
        "baseline": baseline_result,
        "recall_delta": recall_delta,
        "false_alarm_delta": false_ci,
        "economics": economics,
        "gates": {
            "significant_false_reduction": significant_false_reduction,
            "early_signal": early_gate,
            "economic_ci_above_zero": economic_gate,
        },
        "verdict": (
            "RESEARCH_REFERENCE_CANDIDATE"
            if significant_false_reduction and (early_gate or economic_gate)
            else "NOT_READY"
        ),
    }


def trin_diagnostic(option_scores: pd.DataFrame, trin_scores: pd.DataFrame) -> dict[str, Any]:
    keys = ["direction", "date", "sector"]
    paired = option_scores.loc[option_scores["sector"].isin(TRIN_READY)].merge(
        trin_scores[[*keys, "score"]].rename(columns={"score": "trin_score"}),
        on=keys, how="inner", validate="one_to_one",
    ).rename(columns={"score": "option_score"}).dropna(
        subset=["target", "option_score", "trin_score"]
    )
    output: dict[str, Any] = {}
    for direction, rows in paired.groupby("direction", sort=True):
        if rows["target"].nunique() < 2:
            output[direction] = {"status": "INSUFFICIENT_MATURE_LABELS", "rows": int(len(rows))}
            continue
        index = pd.MultiIndex.from_arrays([rows["date"], rows["sector"]], names=["date", "sector"])
        auc_input = pd.DataFrame({
            "target": rows["target"].astype(int).to_numpy(),
            "candidate": rows["trin_score"].to_numpy(),
            "baseline": rows["option_score"].to_numpy(),
        }, index=index)
        output[direction] = {
            "rows": int(len(rows)),
            "groups": sorted(rows["sector"].unique()),
            "option_auc": float(roc_auc_score(rows["target"].astype(int), rows["option_score"])),
            "option_plus_trin_auc": float(roc_auc_score(rows["target"].astype(int), rows["trin_score"])),
            "auc_delta_ci": paired_date_block_auc_delta_ci(
                auc_input["target"], auc_input["candidate"], auc_input["baseline"],
                samples=1_000, block_dates=40, calendar_dates=rows["date"].drop_duplicates(),
            ),
        }
    return output


def report(summary: dict[str, Any]) -> str:
    lines = [
        "# Bloomberg option confirmation SHADOW",
        "",
        f"- 판정: **{summary['verdict']}**",
        "- 목표: 기존 Market Lens 통합 경보의 오경보를 줄이되 사전 포착률의 80% 이상을 유지",
        "- 입력: 30일 IV, 3개월-30일 IV 기간구조, Put/Call OI의 섹터-SPY 상대값",
        "- 시점: Bloomberg EOD t는 다음 시장일부터만 사용",
        "",
        "| 방향 | 선택 정책 | 2026 오경보/섹터년 | 기존 | 사전 포착 | 기존 | 평가 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for direction, result in summary["directions"].items():
        chosen = result["policy"]
        final = result["frozen_2026"]
        lines.append(
            f"| {direction} | {chosen['mode']} | "
            f"{final['selected']['false_alert_clusters_per_sector_year']:.2f} | "
            f"{final['baseline']['false_alert_clusters_per_sector_year']:.2f} | "
            f"{final['selected']['pre_pivot_recall']:.1%} | "
            f"{final['baseline']['pre_pivot_recall']:.1%} | {final['verdict']} |"
        )
    lines.extend([
        "",
        "TRIN은 Banks·Semiconductors coverage가 80% 미만이므로 4개 산업군 진단에만 사용했습니다.",
        "모든 Bloomberg 필드는 PIT_NOT_VERIFIED이며 운영·배포에 반영하지 않았습니다.",
        "",
    ])
    return "\n".join(lines)


def self_test() -> None:
    dates = pd.bdate_range("2025-01-01", periods=140)
    records = []
    for code_no, code in enumerate([*ETF_TO_SECTORS, "SPY"]):
        for date_no, date in enumerate(dates):
            for field, value in (
                ("RK324", 20 + code_no + 0.01 * date_no),
                ("RK327", 20 + code_no + 0.01 * date_no),
                ("RK293", 22 + code_no + 0.01 * date_no),
                ("RK298", 22 + code_no + 0.01 * date_no),
                ("PQ950", 1.2 + 0.01 * code_no + 0.001 * date_no),
            ):
                records.append({
                    "data_group": "ETF_OPTIONS", "code": code,
                    "security": f"{code} US Equity", "field_id": field,
                    "date": date, "value": value,
                })
    for code in TRIN_READY:
        for date_no, date in enumerate(dates):
            records.append({
                "data_group": "INDUSTRY_TRIN", "code": code,
                "security": f"{code} Index", "field_id": "IX116",
                "date": date, "value": 1 + 0.001 * date_no,
            })
    temp = ROOT / "outputs" / "_leading_confirmation_self_test.csv"
    pd.DataFrame(records).to_csv(temp, index=False)
    try:
        features, audit = load_leading_features(temp)
    finally:
        temp.unlink(missing_ok=True)
    assert audit["option_series"] == len([*ETF_TO_SECTORS, "SPY"]) * len(OPTION_FIELDS)
    assert audit["put_call_exact_equal_rate"] == {"iv30_call_put": 1.0, "iv3m_call_put": 1.0}
    complete = features.assign(_complete=features.loc[:, OPTION_FEATURES].notna().all(axis=1))
    counts = complete.groupby("sector")["_complete"].sum()
    assert counts.nunique() == 1 and counts.iloc[0] > 0
    pooled = pd.DataFrame({
        "date": dates[-3:], "sector": "1010", "_position": range(3),
    })
    aligned = attach_features(pooled, features)
    assert (aligned["leading_asof_date"] < aligned["date"]).all()
    selection = pd.DataFrame({
        "mode": ["BASELINE", "OPTION_GATE"],
        "selection_2023_2024_recall": [0.10, 0.10],
        "selection_2025_recall": [0.20, 0.16],
        "selection_2023_2024_false": [2.0, 1.0],
        "selection_2025_false": [2.0, 1.0],
    })
    chosen, floors = choose_candidate(selection)
    assert np.isclose(floors["selection_2023_2024"], 0.10)
    assert np.isclose(floors["selection_2025"], 0.16)
    assert chosen["mode"] == "OPTION_GATE"
    print("self-test: PASS")


def run(args: argparse.Namespace) -> dict[str, Any]:
    features, feature_audit = load_leading_features(args.history_csv.resolve())
    pooled, input_audit, _ = build_pooled_daily_frame(PACKAGE, None, horizon=10)
    pooled = attach_features(pooled, features)

    option_parts, coefficient_parts, model_audit = [], [], {}
    trin_parts = []
    for direction in ("ROTATION_OUT", "ROTATION_IN"):
        option, coefficients, audit = walk_forward(
            pooled, direction, OPTION_FEATURES, "OPTION_LOGIT",
            min_train_dates=args.min_train_dates, refit_every=args.refit_every,
        )
        option_parts.append(option)
        coefficient_parts.append(coefficients)
        model_audit[f"{direction}/OPTION_LOGIT"] = audit

        trin_pool = pooled.loc[pooled["sector"].astype(str).isin(TRIN_READY)].copy()
        trin, coefficients, audit = walk_forward(
            trin_pool, direction, (*OPTION_FEATURES, *TRIN_FEATURES), "OPTION_TRIN_LOGIT",
            min_train_dates=args.min_train_dates, refit_every=args.refit_every,
        )
        trin_parts.append(trin)
        coefficient_parts.append(coefficients)
        model_audit[f"{direction}/OPTION_TRIN_LOGIT"] = audit
    option_scores = pd.concat(option_parts, ignore_index=True)
    trin_scores = pd.concat(trin_parts, ignore_index=True)
    panel = load_comparison_panel(option_scores, args.baseline_csv.resolve())

    sweep_parts, selected_parts, directions = [], [], {}
    for direction, rows in panel.groupby("direction", sort=True):
        rows = rows.copy()
        sweep, cache = candidate_grid(rows)
        chosen, floors = choose_candidate(sweep)
        key = chosen["_key"]
        selected = cache[key]
        baseline = cache[("BASELINE",)]
        final = evaluate_selected(rows, selected, baseline)
        rows["selected_alert"] = selected
        rows["baseline_alert"] = baseline
        selected_parts.append(rows)
        clean_sweep = sweep.drop(columns="_key")
        clean_sweep.insert(0, "direction", direction)
        sweep_parts.append(clean_sweep)
        directions[direction] = {
            "policy": {
                key: _json_ready(chosen[key])
                for key in ("mode", "option_gate_cutoff", "option_weight", "score_cutoff", "cooldown_sessions")
            },
            "selection_recall_floors": floors,
            "selection": {
                key: _json_ready(chosen[key])
                for key in chosen.index
                if key.startswith("selection_") or key in {"worst_false", "mean_false", "worst_recall"}
            },
            "frozen_2026": final,
        }

    overall = all(
        item["frozen_2026"]["verdict"] == "RESEARCH_REFERENCE_CANDIDATE"
        for item in directions.values()
    )
    summary = _json_ready({
        "verdict": "RESEARCH_REFERENCE_CANDIDATE" if overall else "RESEARCH_ONLY",
        "pit_status": "PIT_NOT_VERIFIED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective": "reduce false sector-turning alert clusters with lagged Bloomberg option confirmation",
        "spec": {
            "option_features": list(OPTION_FEATURES),
            "trin_features": list(TRIN_FEATURES),
            "selection": "retain at least 80% of baseline pre-pivot recall; when baseline is below 15%, permit no further recall loss; then minimize worst false clusters across 2023-24 and 2025",
            "frozen_evaluation": "2026; first inspection for these Bloomberg option features, but baseline research history was previously inspected",
            "execution": "Bloomberg EOD t used no earlier than next market session",
            "model": "regularized pooled logistic with sector intercepts; causal walk-forward",
        },
        "input_audit": input_audit,
        "feature_audit": feature_audit,
        "model_audit": model_audit,
        "panel": {
            "rows": int(len(panel)), "groups": int(panel["sector"].nunique()),
            "first_date": str(panel["date"].min().date()),
            "last_date": str(panel["date"].max().date()),
        },
        "directions": directions,
        "trin_diagnostic": trin_diagnostic(option_scores, trin_scores),
        "operational_changes": False,
    })
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    pd.concat(sweep_parts, ignore_index=True).to_csv(output / "policy_sweep.csv", index=False)
    pd.concat(selected_parts, ignore_index=True).to_csv(output / "selected_signals.csv", index=False)
    pd.concat([option_scores, trin_scores], ignore_index=True).to_csv(output / "predictions.csv", index=False)
    pd.concat(coefficient_parts, ignore_index=True).to_csv(output / "coefficients.csv", index=False)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    (output / "REPORT.md").write_text(report(summary), encoding="utf-8")
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--history-csv", type=Path, default=HISTORY)
    result.add_argument("--baseline-csv", type=Path, default=BASELINE)
    result.add_argument("--output-dir", type=Path, default=OUTPUT)
    result.add_argument("--min-train-dates", type=int, default=504)
    result.add_argument("--refit-every", type=int, default=63)
    result.add_argument("--self-test", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.self_test:
        self_test()
        return
    summary = run(args)
    print(json.dumps({"output": str(args.output_dir.resolve()), "verdict": summary["verdict"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
