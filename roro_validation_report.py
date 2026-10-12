"""Build an auditable local validation report for the immutable RORO SHADOW."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from regime_router import build_features_v2, load_benchmark_from_database
from roro_axis import long_exposure_from_risk_score
from roro_data import load_roro_panel
from roro_engine import build_roro_scores, causal_midrank_percentile
from roro_q20 import (
    FEATURE_COLUMNS as Q20_FEATURE_COLUMNS,
    assess_q20_readiness,
    build_q20_research_frame,
)
from roro_policy import (
    FAST_CONDITIONS,
    FAST_TRIGGER_REQUIRED,
    FAST_TRIGGER_TOTAL,
    MODE_COMBINED,
    MODE_OVERLAY_ONLY,
    MODE_SCORE_ONLY,
    replay_roro_policy,
)
from roro_shadow import SERIES_VERSION, read_shadow_history
from router_score_v2 import ROUTER_SCORE_VERSION, build_router_scores_from_database
from validate_roro import (
    build_causal_validation_frame,
    coverage_status_summary,
    evaluate_exposure_policy,
    markdown_report,
    red_alert_onset_diagnostics,
    reentry_relapse_diagnostics,
    validate_roro,
)


MODEL_COLUMN = "roro_3axis_v1"
COMPARATOR_COLUMN = "router_score_v2"
HMM_CORE_COLUMN = "hmm_core_v1"
A2D_COLUMN = "hmm_a2d_active"
VOLATILITY_COLUMN = "nq_volatility_20"
VOLATILITY_POLICY_COLUMN = "_nq_volatility_20_policy_score"
TREND_VOL_COLUMN = "nq_trend_volatility"
CAUSAL_RORO_EW_COLUMN = "roro_ew_v1_causal"
ORIGINAL_RORO_COLUMN = "roro_txt_original"
RORO_DATA_STATUS_COLUMN = "roro_data_status"
RORO_POLICY_COLUMN = "roro_policy"
ORIGINAL_RORO_SHA256 = "7140ce128f1b9707b6304470ef9a19b0b63cbe106a67e3122ada415c9e03e7ce"
LIVE_HMM_MODEL_VERSION = "hmm3-sticky-nq-core-v1-k25-m252-w1260-r20-s42-n3"
A2D_MODEL_VERSION = (
    "hmm3-sticky-nq-independent-a2d-logvol-v1-diag-mc25-"
    "k25-m252-w1260-r20-s42-n3"
)
RANKED_COMPARATOR_COLUMNS = (
    COMPARATOR_COLUMN,
    HMM_CORE_COLUMN,
    A2D_COLUMN,
    VOLATILITY_COLUMN,
    TREND_VOL_COLUMN,
    CAUSAL_RORO_EW_COLUMN,
)
OUTPUT_START = "2026-01-07"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _probability_scores(database: Path, model_version: str, column: str) -> pd.DataFrame:
    """Read one immutable HMM series and expose its causal Risk-Off posterior."""
    uri = database.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        connection.execute("PRAGMA query_only=ON")
        frame = pd.read_sql_query(
            """
            SELECT date,p_bull,p_sideways,p_bear
            FROM regime_model_probabilities
            WHERE model_version=?
            ORDER BY date
            """,
            connection,
            params=(model_version,),
            parse_dates=["date"],
        )
    if frame.empty:
        raise ValueError(f"missing HMM comparator: {model_version}")
    probabilities = frame[["p_bull", "p_sideways", "p_bear"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if (
        not np.isfinite(probabilities.to_numpy()).all()
        or not probabilities.ge(0).all().all()
        or not probabilities.le(1).all().all()
        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    ):
        raise ValueError(f"invalid HMM probabilities: {model_version}")
    frame[column] = 100.0 * probabilities["p_bear"]
    return frame[["date", column]]


def _simple_nq_scores(prices: pd.DataFrame) -> pd.DataFrame:
    """Build the raw-vol ranking baseline and a causal trend/vol score."""
    features = build_features_v2(
        prices.rename(columns={"nq_close": "close"})[["date", "close"]]
    )
    volatility = causal_midrank_percentile(features["volatility_20"]) * 100
    trend = causal_midrank_percentile(-features["momentum_20"]) * 100
    result = pd.DataFrame(
        {
            VOLATILITY_COLUMN: features["volatility_20"],
            VOLATILITY_POLICY_COLUMN: volatility,
            TREND_VOL_COLUMN: pd.concat([trend, volatility], axis=1).mean(
                axis=1, skipna=False
            ),
        },
        index=features.index,
    )
    return result.rename_axis("date").reset_index()


def build_additional_comparators(
    publish_database: str | Path,
    prices: pd.DataFrame,
    *,
    as_of: str | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    """Return ranked comparators plus explicitly blocked/reference-only baselines."""
    publish = Path(publish_database)
    frames = [
        _probability_scores(publish, LIVE_HMM_MODEL_VERSION, HMM_CORE_COLUMN),
        _probability_scores(publish, A2D_MODEL_VERSION, A2D_COLUMN),
        _simple_nq_scores(prices),
    ]
    combined = frames[0]
    for frame in frames[1:]:
        combined = combined.merge(frame, on="date", how="outer", validate="one_to_one")

    panel = load_roro_panel(publish, as_of=as_of or prices["date"].max())
    causal_ew = build_roro_scores(panel).loc[:, ["risk_score"]].rename(
        columns={"risk_score": CAUSAL_RORO_EW_COLUMN}
    )
    combined = combined.merge(
        causal_ew.rename_axis("date").reset_index(),
        on="date",
        how="outer",
        validate="one_to_one",
    )
    combined[ORIGINAL_RORO_COLUMN] = np.nan
    metadata = {
        HMM_CORE_COLUMN: {
            "model_version": LIVE_HMM_MODEL_VERSION,
            "status": "FORMER_LIVE_ROLLBACK_COMPARATOR",
            "score_definition": "100 * p_bear",
        },
        A2D_COLUMN: {
            "model_version": A2D_MODEL_VERSION,
            "status": "MANUAL_OVERRIDE_ACTIVE_HISTORICAL_FAIL_COMPARATOR",
            "score_definition": "100 * p_bear",
        },
        VOLATILITY_COLUMN: {
            "model_version": "nq-volatility20-raw-causal",
            "status": "COMPARATOR",
            "policy_mapping": "causal percentile w756/m252 before frozen Long anchors",
        },
        TREND_VOL_COLUMN: {
            "model_version": "nq-trend20-volatility20-ew-causal-percentile-w756-m252",
            "status": "DESCRIPTIVE_COMPARATOR_NOT_PREREGISTERED",
            "score_definition": "mean(causal percentile(-NQ 20D return), causal percentile(NQ volatility20))",
        },
        CAUSAL_RORO_EW_COLUMN: {
            "model_version": "roro-ew-v1-w756-m252-u20",
            "status": "COMPARATOR" if causal_ew[CAUSAL_RORO_EW_COLUMN].notna().any() else "DATA_BLOCKED",
            "reason": None
            if causal_ew[CAUSAL_RORO_EW_COLUMN].notna().any()
            else "fewer than five complete independent blocks; missing inputs remain unavailable",
        },
        ORIGINAL_RORO_COLUMN: {
            "model_version": "RORO.txt-original",
            "status": "REFERENCE_ONLY",
            "reason": "full-sample scaling, PCA/HMM and smoothed probabilities are non-causal",
            "source_sha256": ORIGINAL_RORO_SHA256,
        },
    }
    return combined.sort_values("date").reset_index(drop=True), metadata


def build_validation_daily(
    publish_database: str | Path,
    shadow_database: str | Path,
    *,
    series_version: str = SERIES_VERSION,
    as_of: str | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    """Exact-date join NQ prices, immutable RORO rows and every frozen comparator."""
    publish = Path(publish_database)
    shadow = Path(shadow_database)
    prices = load_benchmark_from_database(publish, "NQ=F").rename(
        columns={"close": "nq_close"}
    )
    prices["date"] = pd.to_datetime(prices["date"])

    history = read_shadow_history(shadow, series_version=series_version)
    roro_rows = []
    for observation in history:
        payload = observation["payload"]
        roro_rows.append(
            {
                "date": pd.Timestamp(observation["date"]),
                MODEL_COLUMN: payload.get("market_risk_score"),
                "recommended_long_pct": payload.get("recommended_long"),
                RORO_DATA_STATUS_COLUMN: payload.get("data_status"),
                RORO_POLICY_COLUMN: payload.get("policy"),
                **{column: payload.get(column) for column in Q20_FEATURE_COLUMNS},
            }
        )
    roro = pd.DataFrame(
        roro_rows,
        columns=[
            "date",
            MODEL_COLUMN,
            "recommended_long_pct",
            RORO_DATA_STATUS_COLUMN,
            RORO_POLICY_COLUMN,
            *Q20_FEATURE_COLUMNS,
        ],
    )
    if roro.empty:
        raise ValueError(f"no RORO rows for series: {series_version}")
    if roro["date"].duplicated().any():
        raise ValueError("RORO history dates must be unique")

    router = build_router_scores_from_database(publish, benchmark="NQ=F")
    router = (
        router.loc[:, ["risk_score"]]
        .rename(columns={"risk_score": COMPARATOR_COLUMN})
        .rename_axis("date")
        .reset_index()
    )
    router["date"] = pd.to_datetime(router["date"])
    additional, metadata = build_additional_comparators(
        publish,
        prices,
        as_of=as_of,
    )
    daily = prices.merge(roro, on="date", how="left", validate="one_to_one")
    daily = daily.merge(router, on="date", how="left", validate="one_to_one")
    daily = daily.merge(additional, on="date", how="left", validate="one_to_one")
    return daily.sort_values("date").reset_index(drop=True), metadata


def _policy_comparison(
    daily: pd.DataFrame,
    score_columns: list[str],
    *,
    as_of: str,
) -> dict[str, object]:
    """Apply the same frozen RORO Long anchors and 10bp cost to every score."""
    cutoff = pd.Timestamp(as_of)
    source = daily.loc[daily["date"] <= cutoff].copy()
    numeric = source[score_columns].apply(pd.to_numeric, errors="coerce")
    common = numeric.notna().all(axis=1) & np.isfinite(numeric).all(axis=1)
    common_dates = source.loc[common, "date"]
    summaries: dict[str, object] = {}
    for column in score_columns:
        long_column = f"_{column}_long"
        policy_score = (
            source[VOLATILITY_POLICY_COLUMN]
            if column == VOLATILITY_COLUMN
            else source[column]
        )
        source[long_column] = long_exposure_from_risk_score(
            pd.to_numeric(policy_score, errors="coerce")
        ).where(common)
        _, summaries[column] = evaluate_exposure_policy(
            source,
            long_column=long_column,
            output_start=OUTPUT_START,
        )
    return {
        "mapping": "same frozen RORO anchors (0,100),(33,90),(66,60),(100,25)",
        "execution": "signal t close, apply t+1",
        "transaction_cost_bps": 10,
        "common_signal_rows": int(common.sum()),
        "common_signal_start": common_dates.min().date().isoformat()
        if not common_dates.empty
        else None,
        "common_signal_end": common_dates.max().date().isoformat()
        if not common_dates.empty
        else None,
        "models": summaries,
    }


def evaluate_policy_ablation(
    daily: pd.DataFrame, *, as_of: str
) -> dict[str, object]:
    """Replay exact score/Fast ablations solely from immutable policy audits."""
    required = [
        "date",
        "nq_close",
        MODEL_COLUMN,
        "recommended_long_pct",
        RORO_POLICY_COLUMN,
    ]
    missing = [column for column in required if column not in daily]
    if missing:
        return {
            "status": "DATA_BLOCKED",
            "reason": "missing ablation columns: " + ", ".join(missing),
        }
    source = daily.loc[
        pd.to_datetime(daily["date"], errors="raise").between(
            pd.Timestamp(OUTPUT_START), pd.Timestamp(as_of)
        )
    ].copy()
    if source.empty or source[RORO_POLICY_COLUMN].isna().any():
        return {
            "status": "DATA_BLOCKED",
            "reason": "complete nested policy audits are unavailable",
        }
    if source[["nq_close", MODEL_COLUMN, "recommended_long_pct"]].isna().any().any():
        return {
            "status": "DATA_BLOCKED",
            "reason": "complete NQ, score, and stored Long rows are unavailable",
        }

    decisions: list[dict[str, object]] = []
    stored_longs: list[int] = []
    for _, row in source.iterrows():
        day = pd.Timestamp(row["date"]).date().isoformat()
        score = row[MODEL_COLUMN]
        stored_long = row["recommended_long_pct"]
        policy = row[RORO_POLICY_COLUMN]
        if (
            isinstance(score, (bool, np.bool_))
            or not float(score).is_integer()
            or not 0 <= int(score) <= 100
            or isinstance(stored_long, (bool, np.bool_))
            or not float(stored_long).is_integer()
            or not 0 <= int(stored_long) <= 100
            or not isinstance(policy, dict)
        ):
            raise ValueError("ablation score, stored Long, or policy audit is invalid")
        active = policy.get("active_fast_conditions")
        unavailable = policy.get("unavailable_fast_conditions")
        if not isinstance(active, list) or not isinstance(unavailable, list):
            raise ValueError("policy Fast condition audits must be lists")
        if (
            any(not isinstance(key, str) for key in [*active, *unavailable])
            or len(active) != len(set(active))
            or len(unavailable) != len(set(unavailable))
            or not set(active).isdisjoint(unavailable)
            or not set(active).issubset(FAST_CONDITIONS)
            or not set(unavailable).issubset(FAST_CONDITIONS)
        ):
            raise ValueError("policy Fast condition audits violate the frozen keys")
        status_by_key = {
            key: "TRIGGERED"
            if key in active
            else "UNAVAILABLE"
            if key in unavailable
            else "CLEAR"
            for key in FAST_CONDITIONS
        }
        complete = not unavailable
        derived_status = (
            "TRIGGERED"
            if len(active) >= FAST_TRIGGER_REQUIRED
            else "CLEAR"
            if complete
            else "INCOMPLETE"
        )
        if (
            policy.get("fast_trigger_count") != len(active)
            or policy.get("fast_available_count")
            != FAST_TRIGGER_TOTAL - len(unavailable)
            or policy.get("fast_status") != derived_status
            or policy.get("recommended_long") != int(stored_long)
            or not isinstance(policy.get("primary_driver"), str)
            or not policy["primary_driver"].strip()
        ):
            raise ValueError("policy Fast scalars or stored Long disagree with its audit")
        snapshot = {
            "as_of": day,
            "status": derived_status,
            "count": len(active),
            "trigger_count": len(active),
            "required": FAST_TRIGGER_REQUIRED,
            "total": FAST_TRIGGER_TOTAL,
            "complete": complete,
            "primary_driver": policy["primary_driver"],
            "conditions": [
                {
                    "key": key,
                    "label": key,
                    "status": status_by_key[key],
                    "data_quality": "UNAVAILABLE"
                    if status_by_key[key] == "UNAVAILABLE"
                    else "EXACT",
                    "as_of": day,
                }
                for key in FAST_CONDITIONS
            ],
        }
        decisions.append(
            {"as_of": day, "risk_score": int(score), "fast_snapshot": snapshot}
        )
        stored_longs.append(int(stored_long))

    replayed = {
        mode: replay_roro_policy(decisions, mode=mode)
        for mode in (MODE_SCORE_ONLY, MODE_OVERLAY_ONLY, MODE_COMBINED)
    }
    combined_longs = [item["recommended_long"] for item in replayed[MODE_COMBINED]]
    if combined_longs != stored_longs:
        raise ValueError("combined ablation replay disagrees with every stored Long row")

    summaries: dict[str, object] = {}
    result_daily = source.loc[:, ["date", "nq_close"]].copy()
    for mode, results in replayed.items():
        column = f"_{mode}_long"
        result_daily[column] = [item["recommended_long"] for item in results]
        _, summaries[mode] = evaluate_exposure_policy(
            result_daily, long_column=column, output_start=OUTPUT_START
        )
    if (
        len({summary["rows"] for summary in summaries.values()}) != 1
        or len({summary["start"] for summary in summaries.values()}) != 1
        or len({summary["end"] for summary in summaries.values()}) != 1
    ):
        raise ValueError("policy ablations must use exact same signal dates")
    return {
        "status": "AVAILABLE",
        "modes": [MODE_SCORE_ONLY, MODE_OVERLAY_ONLY, MODE_COMBINED],
        "common_signal_rows": len(decisions),
        "execution": "signal t close, apply t+1",
        "transaction_cost_bps": 10,
        "combined_replay_matches_stored": True,
        "models": summaries,
    }


def generate_report(
    publish_database: str | Path,
    shadow_database: str | Path,
    *,
    as_of: str,
    series_version: str = SERIES_VERSION,
    bootstrap_samples: int = 500,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run the frozen 20-day NQ validation without modifying either database."""
    publish = Path(publish_database)
    shadow = Path(shadow_database)
    daily, comparator_metadata = build_validation_daily(
        publish,
        shadow,
        series_version=series_version,
        as_of=as_of,
    )
    ranked_comparators = [
        column
        for column in RANKED_COMPARATOR_COLUMNS
        if column in daily and daily[column].notna().any()
    ]
    report = validate_roro(
        daily,
        candidate_column=MODEL_COLUMN,
        benchmark_columns=ranked_comparators,
        reference_only_columns=[ORIGINAL_RORO_COLUMN],
        long_column="recommended_long_pct",
        output_start=OUTPUT_START,
        as_of=as_of,
        horizon=20,
        drawdown_threshold=-0.05,
        bootstrap_samples=bootstrap_samples,
        block_days=40,
        random_state=42,
    )
    blocked = comparator_metadata[CAUSAL_RORO_EW_COLUMN]
    if blocked["status"] == "DATA_BLOCKED":
        report["blocked_comparators"] = {
            CAUSAL_RORO_EW_COLUMN: {
                "role": "COMPARATOR",
                "status": "DATA_BLOCKED",
                "reason": blocked["reason"],
            }
        }
    report["comparator_catalog"] = comparator_metadata
    report["policy_comparison_10bp"] = _policy_comparison(
        daily,
        [MODEL_COLUMN, *ranked_comparators],
        as_of=as_of,
    )
    report["policy_ablation_10bp"] = evaluate_policy_ablation(daily, as_of=as_of)
    cutoff = pd.Timestamp(as_of)
    output_start = pd.Timestamp(OUTPUT_START)
    diagnostic_daily = daily.loc[daily["date"] <= cutoff].copy()
    output_daily = diagnostic_daily.loc[diagnostic_daily["date"] >= output_start]
    report["coverage_status"] = coverage_status_summary(
        output_daily[MODEL_COLUMN], output_daily[RORO_DATA_STATUS_COLUMN]
    )
    report["red_alert_onset"] = red_alert_onset_diagnostics(
        diagnostic_daily,
        score_column=MODEL_COLUMN,
        output_start=OUTPUT_START,
    )
    report["reentry_relapse_10d"] = reentry_relapse_diagnostics(
        diagnostic_daily,
        policy_column=RORO_POLICY_COLUMN,
        output_start=OUTPUT_START,
    )
    report["series_version"] = series_version
    report["comparator_versions"] = {
        COMPARATOR_COLUMN: ROUTER_SCORE_VERSION,
        **{
            name: item["model_version"]
            for name, item in comparator_metadata.items()
            if name in ranked_comparators
        },
    }
    q20_frame = build_q20_research_frame(
        daily.loc[:, ["date", "nq_close", *Q20_FEATURE_COLUMNS]],
        as_of=as_of,
    )
    q20 = assess_q20_readiness(q20_frame)
    report["q20_readiness"] = {
        **q20,
        "complete_labels": q20["complete_feature_labels"],
        "independent_episodes": q20["independent_events"],
        "minimum_independent_episodes": q20["minimum_independent_events"],
    }
    report["provenance"] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "as_of": pd.Timestamp(as_of).date().isoformat(),
        "publish_database_sha256": _sha256(publish),
        "shadow_database_sha256": _sha256(shadow),
        "validate_roro_sha256": _sha256(Path(__file__).with_name("validate_roro.py")),
        "runner_sha256": _sha256(Path(__file__)),
        "roro_txt_sha256": ORIGINAL_RORO_SHA256,
        "causal_contract": "signal at t close; label t+1..t+20; Long applied t+1",
    }
    labelled = build_causal_validation_frame(
        daily,
        score_columns=[MODEL_COLUMN, *ranked_comparators],
        output_start=OUTPUT_START,
        as_of=as_of,
        horizon=20,
        drawdown_threshold=-0.05,
    )
    return report, labelled


def write_report(
    report: dict[str, Any],
    labelled: pd.DataFrame,
    output_directory: str | Path,
) -> dict[str, str]:
    """Write JSON, Markdown and a compact run log; no database output is created."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "roro_validation.json"
    markdown_path = output / "roro_validation.md"
    log_path = output / "run.log"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    markdown = [markdown_report(report).rstrip(), "", "## Same-sample 10bp Long comparison", ""]
    markdown.extend(
        [
            "All ranked scores use the same frozen RORO Long anchors and are applied from the next NQ bar.",
            "",
            "| Score | Rows | Net total | Net CAGR | Net Sharpe | Net MDD | NQ CAGR | NQ Sharpe | NQ MDD | Return drag | Relative MDD improvement | Turnover |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    def percent(value: object) -> str:
        return "—" if value is None else f"{100 * float(value):.2f}%"

    def number(value: object) -> str:
        return "—" if value is None else f"{float(value):.3f}"

    for name, summary in report["policy_comparison_10bp"]["models"].items():
        markdown.append(
            f"| {name} | {summary['rows']} | {percent(summary['net_10bp_total_return'])} | "
            f"{percent(summary['net_10bp_cagr'])} | {number(summary['net_10bp_sharpe'])} | "
            f"{percent(summary['net_10bp_max_drawdown'])} | "
            f"{percent(summary['nq_benchmark_cagr'])} | "
            f"{number(summary['nq_benchmark_sharpe'])} | "
            f"{percent(summary['nq_benchmark_max_drawdown'])} | "
            f"{percent(summary['annual_return_drag'])} | "
            f"{percent(summary['relative_mdd_improvement'])} | "
            f"{summary['total_turnover']:.2f} |"
        )
    ablation = report.get("policy_ablation_10bp", {})
    markdown.extend(["", "## Score and Fast policy ablation", ""])
    if ablation.get("status") == "DATA_BLOCKED":
        markdown.append(f"- **DATA_BLOCKED** — {ablation['reason']}")
    else:
        markdown.extend(
            [
                "The three modes replay the same stored decision dates with the unchanged reducer and 10bp cost.",
                "",
                "| Mode | Rows | Net total | Net CAGR | Net Sharpe | Net MDD | Turnover |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name, summary in ablation["models"].items():
            markdown.append(
                f"| {name} | {summary['rows']} | "
                f"{percent(summary['net_10bp_total_return'])} | "
                f"{percent(summary['net_10bp_cagr'])} | "
                f"{number(summary['net_10bp_sharpe'])} | "
                f"{percent(summary['net_10bp_max_drawdown'])} | "
                f"{summary['total_turnover']:.2f} |"
            )
    a2d_status = report.get("comparator_catalog", {}).get(A2D_COLUMN, {}).get("status")
    if a2d_status:
        markdown.extend(
            [
                "",
                f"> `{A2D_COLUMN}` governance: **{a2d_status}**. "
                "The short 2026 comparison is descriptive and does not reverse its historical validation result.",
            ]
        )
    blocked = report.get("blocked_comparators", {})
    if blocked:
        markdown.extend(["", "## Blocked comparators", ""])
        for name, item in blocked.items():
            markdown.append(f"- `{name}`: **{item['status']}** — {item['reason']}")
    q20 = report["q20_readiness"]
    markdown.extend(
        [
            "",
            "## q20 readiness",
            "",
            f"- Status: **{q20['status']}**",
            f"- Complete labels: {q20['complete_labels']} / {q20['minimum_complete_labels']}",
            f"- Independent episodes: {q20['independent_episodes']} / {q20['minimum_independent_episodes']}",
            "- Probability: `null` until both frozen minimums are met",
            "",
        ]
    )
    markdown_path.write_text("\n".join(markdown), encoding="utf-8")
    coverage = report["coverage"]
    log_path.write_text(
        "\n".join(
            [
                f"status={report['status']}",
                f"series_version={report['series_version']}",
                f"output_rows={coverage['output_rows']}",
                f"complete_label_rows={coverage['complete_label_rows']}",
                f"common_labelled_rows={coverage['common_labelled_rows']}",
                f"independent_episodes={coverage['independent_episodes']}",
                f"labelled_frame_rows={len(labelled)}",
                f"as_of={report['provenance']['as_of']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "json": str(json_path.resolve()),
        "markdown": str(markdown_path.resolve()),
        "log": str(log_path.resolve()),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("publish_database")
    parser.add_argument("shadow_database")
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--output-directory", default="outputs/roro_shadow/validation")
    parser.add_argument("--series-version", default=SERIES_VERSION)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    report, labelled = generate_report(
        args.publish_database,
        args.shadow_database,
        as_of=args.as_of,
        series_version=args.series_version,
        bootstrap_samples=args.bootstrap_samples,
    )
    paths = write_report(report, labelled, args.output_directory)
    print(
        json.dumps(
            {
                "status": report["status"],
                "coverage": report["coverage"],
                "outputs": paths,
            },
            ensure_ascii=False,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
