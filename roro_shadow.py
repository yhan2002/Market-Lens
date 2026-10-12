"""Append-only persistence for the isolated RORO SHADOW series."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
import struct
from contextlib import closing
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from roro_axis import COMPONENT_SCORE_COLUMNS
from roro_axis import MODEL_VERSION as AXIS_MODEL_VERSION
from roro_axis import build_roro_axis_scores
from roro_data import load_roro_panel
from roro_engine import MODEL_VERSION as RORO_MODEL_VERSION
from roro_engine import build_roro_scores
from roro_market_breadth import CONTRACT_VERSION as BREADTH_OBSERVATION_MODEL_VERSION
from roro_market_breadth import SOURCE as BREADTH_OBSERVATION_SOURCE
from roro_market_breadth import TABLE as BREADTH_OBSERVATION_TABLE
from roro_market_breadth import breadth_readiness
from roro_phase import MODEL_VERSION as PHASE_MODEL_VERSION
from roro_phase import build_trend_phase
from roro_portfolio import load_portfolio_internal_series
from roro_price_observations import CONTRACT_VERSION as PRICE_OBSERVATION_MODEL_VERSION
from roro_price_observations import SOURCE as PRICE_OBSERVATION_SOURCE
from roro_price_observations import TABLE as PRICE_OBSERVATION_TABLE
from roro_policy import FAST_CONDITIONS
from roro_policy import POLICY_VERSION as POLICY_MODEL_VERSION
from roro_policy import RECOVERY_CONFIRM_DAYS
from roro_policy import long_anchor
from roro_policy import replay_roro_policy
from roro_q20 import FEATURE_COLUMNS as Q20_FEATURE_COLUMNS
from roro_q20 import MODEL_VERSION as Q20_MODEL_VERSION
from roro_q20 import build_q20_readiness_history
from roro_q20 import validate_q20_readiness_metadata
from risk_signals import build_fast_signal_snapshot


SHADOW_START = pd.Timestamp("2025-10-08")
PORTFOLIO_MODEL_VERSION = "portfolio-internals-nq-v2-c85-incomplete-snapshot-qc"
LEGACY_SERIES_VERSION = (
    f"{AXIS_MODEL_VERSION}__{PHASE_MODEL_VERSION}__p-c85-isqc__shadow-r3"
)
Q20_SERIES_VERSION = (
    f"{AXIS_MODEL_VERSION}__{PHASE_MODEL_VERSION}__p-c85-isqc__shadow-r4-q20"
)
R5_SERIES_VERSION = (
    f"{AXIS_MODEL_VERSION}__{PHASE_MODEL_VERSION}__p-c85-isqc__shadow-r5-q20-policy"
)
R6_SERIES_VERSION = (
    f"{AXIS_MODEL_VERSION}__{PHASE_MODEL_VERSION}__p-c85-isqc__"
    "shadow-r6-q20-policy-fxgold"
)
R7_SERIES_VERSION = (
    f"{AXIS_MODEL_VERSION}__{PHASE_MODEL_VERSION}__p-c85-isqc__"
    "shadow-r7-credit-move"
)
R8_SERIES_VERSION = (
    f"{AXIS_MODEL_VERSION}__{PHASE_MODEL_VERSION}__p-c85-isqc__"
    "shadow-r8-narrative"
)
R9_SERIES_VERSION = (
    f"{AXIS_MODEL_VERSION}__{PHASE_MODEL_VERSION}__p-c85-isqc__"
    "shadow-r9-market-narrative"
)
R10_SERIES_VERSION = (
    f"{AXIS_MODEL_VERSION}__{PHASE_MODEL_VERSION}__p-c85-isqc__"
    "shadow-r10-ndth-breadth"
)
R11_SERIES_VERSION = (
    f"{AXIS_MODEL_VERSION}__{PHASE_MODEL_VERSION}__p-c85-isqc__"
    "shadow-r11-ndth-move-reviewed"
)
R12_SERIES_VERSION = (
    f"{AXIS_MODEL_VERSION}__{PHASE_MODEL_VERSION}__p-c85-isqc__"
    "shadow-r12-reviewed-move-priority"
)
R13_SERIES_VERSION = (
    f"{AXIS_MODEL_VERSION}__{PHASE_MODEL_VERSION}__p-c85-isqc__"
    "shadow-r13-revised-hy-history"
)
R14_SERIES_VERSION = (
    f"{AXIS_MODEL_VERSION}__{PHASE_MODEL_VERSION}__p-c85-isqc__"
    "shadow-r14-fred-usdjpy-fast"
)
R15_SERIES_VERSION = (
    f"{AXIS_MODEL_VERSION}__{PHASE_MODEL_VERSION}__p-c85-isqc__"
    "shadow-r15-yahoo-usdjpy-fast"
)
R16_SERIES_VERSION = (
    f"{AXIS_MODEL_VERSION}__{PHASE_MODEL_VERSION}__p-c85-isqc__"
    "shadow-r16-nyclose-plus1h"
)
R17_SERIES_VERSION = (
    f"{AXIS_MODEL_VERSION}__{PHASE_MODEL_VERSION}__p-c85-isqc__"
    "shadow-r17-exact-market-inputs"
)
R18_SERIES_VERSION = (
    f"{AXIS_MODEL_VERSION}__{PHASE_MODEL_VERSION}__p-c85-isqc__"
    "shadow-r18-history-from-20251008"
)
R19_SERIES_VERSION = (
    f"{AXIS_MODEL_VERSION}__{PHASE_MODEL_VERSION}__p-c85-isqc__"
    "shadow-r19-holdings-history-recalc"
)
SERIES_VERSION = (
    f"{AXIS_MODEL_VERSION}__{PHASE_MODEL_VERSION}__p-c85-isqc__"
    "shadow-r21-breadth-correction"
)
# A same-date correction is a new immutable row, never a rewrite of the live series.
BREADTH_REPAIR_SERIES_VERSION = (
    f"{AXIS_MODEL_VERSION}__{PHASE_MODEL_VERSION}__p-c85-isqc__"
    "shadow-r22-late-breadth-repair"
)
Q20_NO_PHASE_SERIES_VERSION = f"{Q20_SERIES_VERSION}__diagnostic-no-phase"
R5_NO_PHASE_SERIES_VERSION = f"{R5_SERIES_VERSION}__diagnostic-no-phase"
NO_PHASE_SERIES_VERSION = f"{SERIES_VERSION}__diagnostic-no-phase"
TABLE = "roro_shadow_observations"

PORTFOLIO_RAW_COLUMNS = (
    "portfolio_correlation_20",
    "portfolio_correlation_trajectory_5d",
    "nq_beta_60",
    "effective_positions",
    "breadth_above_ma20",
    "negative_participation_20",
    "portfolio_drawdown_60",
    "price_weight_coverage",
    "history_weight_coverage_20d",
    "history_weight_coverage_60d",
    "data_quality",
)
PHASE_PAYLOAD_COLUMNS = (
    "vix_curve_available",
    "hy_oas_available",
    "portfolio_correlation_available",
    "market_breadth_available",
    "portfolio_breadth_ma20_available",
    "nq_ma5_available",
    "bottom_vix_curve",
    "bottom_hy_oas",
    "bottom_portfolio_correlation",
    "bottom_market_breadth",
    "bottom_portfolio_breadth_ma20",
    "bottom_nq_ma5",
    "top_vix_curve",
    "top_hy_oas",
    "top_portfolio_correlation",
    "top_market_breadth",
    "top_portfolio_breadth_ma20",
    "top_nq_ma5",
    "checklist_available_count",
    "checklist_required_count",
    "checklist_eligible",
    "bottom_confirmation_count",
    "top_confirmation_count",
    "bottom_checklist_pass",
    "top_checklist_pass",
    "bottom_confirmed_evidence",
    "top_confirmed_evidence",
    "bottom_available_evidence",
    "top_available_evidence",
    "missing_evidence",
    "risk_level",
    "risk_direction_5d",
    "nq_ma5",
    "nq_ma20",
    "score_20d_max",
    "score_20d_min",
    "score_drop_from_20d_max",
    "score_rise_from_20d_min",
    "bottoming_candidate_signal",
    "prior_bottoming_candidate_20d",
    "recovery_confirmation_signal",
    "topping_candidate_signal",
    "trend_phase",
    "turn_evidence",
)
COMPONENT_DIAGNOSTIC_COLUMNS = (
    *(
        f"{component}_contribution"
        for component in COMPONENT_SCORE_COLUMNS
    ),
    *(
        f"{component}_score_trajectory_{lag}d"
        for component in COMPONENT_SCORE_COLUMNS
        for lag in (1, 5, 20)
    ),
    "available_components",
    "component_risk_drivers",
    "component_relief_drivers",
)
POLICY_AUDIT_FIELDS = (
    "model_version",
    "score_based_long",
    "raw_target_long",
    "recommended_long",
    "fast_status",
    "fast_trigger_count",
    "fast_available_count",
    "active_fast_conditions",
    "unavailable_fast_conditions",
    "phase",
    "episode_status",
    "clear_streak",
    "primary_driver",
)

# Explicit allow-list: upstream frames may contain private holding detail later;
# only portfolio aggregates and model diagnostics can cross this boundary.
PAYLOAD_COLUMNS = (
    "credit_score",
    "rates_volatility_score",
    "equity_volatility_score",
    "fx_carry_score",
    "safe_haven_score",
    "equity_trend_score",
    "breadth_risk_percentile",
    "portfolio_correlation_score",
    "portfolio_beta_score",
    "portfolio_concentration_score",
    "portfolio_participation_score",
    "portfolio_drawdown_score",
    "portfolio_breadth_ma20_score",
    "portfolio_breadth_ma50_score",
    "portfolio_breadth_ma200_score",
    "portfolio_breadth_score",
    *PORTFOLIO_RAW_COLUMNS,
    "portfolio_available_component_count",
    "portfolio_internal_score",
    "portfolio_history_blocked",
    "portfolio_data_status",
    "macro_available_component_count",
    "macro_risk_score",
    "macro_data_status",
    "market_available_component_count",
    "market_trend_score",
    "market_data_status",
    "macro_completeness",
    "market_completeness",
    "portfolio_completeness",
    "available_axis_count",
    "market_risk_score_raw",
    "market_risk_score",
    "data_completeness",
    "axis_disagreement",
    "axis_agreement",
    "confidence",
    "data_status",
    "risk_regime",
    "recommended_long",
    "macro_risk_contribution",
    "market_trend_contribution",
    "portfolio_internal_contribution",
    "risk_drivers",
    "relief_drivers",
    "available_axes",
    "axis_completeness",
    *COMPONENT_DIAGNOSTIC_COLUMNS,
    "trajectory_1d",
    "trajectory_5d",
    "trajectory_20d",
    "q20",
    "policy",
    *PHASE_PAYLOAD_COLUMNS,
)
REQUIRED_PAYLOAD_COLUMNS = (
    "market_risk_score",
    "data_status",
    "risk_regime",
    "recommended_long",
)
LEGACY_MODEL_VERSIONS = {
    "roro": RORO_MODEL_VERSION,
    "axis": AXIS_MODEL_VERSION,
    "portfolio": PORTFOLIO_MODEL_VERSION,
    "phase": PHASE_MODEL_VERSION,
}
Q20_MODEL_VERSIONS = {**LEGACY_MODEL_VERSIONS, "q20": Q20_MODEL_VERSION}
R5_MODEL_VERSIONS = {**Q20_MODEL_VERSIONS, "policy": POLICY_MODEL_VERSION}
MODEL_VERSIONS = {
    **R5_MODEL_VERSIONS,
    "price_observations": PRICE_OBSERVATION_MODEL_VERSION,
}
R10_MODEL_VERSIONS = {**MODEL_VERSIONS, "market_breadth": BREADTH_OBSERVATION_MODEL_VERSION}
SCHEMA_COLUMNS = (
    "series_version",
    "date",
    "input_cutoff",
    "roro_model_version",
    "axis_model_version",
    "portfolio_model_version",
    "provenance_hash",
    "payload_json",
    "created_at",
)


class ImmutableSeriesConflict(RuntimeError):
    """Raised when an existing series-version/date payload would be rewritten."""


_UTC_Z_TIMESTAMP = re.compile(
    r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d{1,6}))?Z$"
)


def _validate_utc_z_timestamp(value: object, name: str) -> str:
    raw = value if isinstance(value, str) else ""
    match = _UTC_Z_TIMESTAMP.fullmatch(raw)
    if match is None:
        raise ValueError(f"{name} must be a valid UTC ISO timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(raw[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError(
            f"{name} must be a valid UTC ISO timestamp ending in Z"
        ) from error
    fraction = match.group(2)
    canonical = parsed.strftime("%Y-%m-%dT%H:%M:%S")
    if fraction is not None:
        canonical += "." + f"{parsed.microsecond:06d}"[: len(fraction)]
    if raw != canonical + "Z":
        raise ValueError(f"{name} must be a valid UTC ISO timestamp ending in Z")
    return raw


def _calendar_date(value: str | date | pd.Timestamp, name: str) -> str:
    try:
        timestamp = pd.Timestamp(value).normalize()
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a calendar date") from error
    if pd.isna(timestamp):
        raise ValueError(f"{name} must be a calendar date")
    return timestamp.date().isoformat()


def _same_file(left: Path, right: Path) -> bool:
    if os.path.normcase(str(left.resolve())) == os.path.normcase(str(right.resolve())):
        return True
    return left.exists() and right.exists() and os.path.samefile(left, right)


def _validate_paths(
    publish_source: str | Path,
    private_source: str | Path,
    shadow_target: str | Path,
) -> tuple[Path, Path, Path]:
    publish = Path(publish_source)
    private = Path(private_source)
    target = Path(shadow_target)
    for source in (publish, private):
        if not source.is_file():
            raise FileNotFoundError(source)
        if _same_file(source, target):
            raise ValueError("shadow_target must be separate from every source database")
    if target.exists() and not target.is_file():
        raise ValueError("shadow_target must be a SQLite file path")
    return publish, private, target


def _json_value(value: Any) -> Any:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime, date, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if math.isnan(numeric):
            return None
        if not math.isfinite(numeric):
            raise ValueError("non-finite SHADOW payload value")
        return numeric
    if isinstance(value, str):
        return value
    raise TypeError(f"unsupported SHADOW payload value: {type(value).__name__}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_provenance_encoding(value: Any) -> str:
    """Encode JSON-compatible values identically in Python and JavaScript."""

    def encode(item: Any) -> str:
        if item is None:
            return "n"
        if isinstance(item, bool):
            return "b1" if item else "b0"
        if isinstance(item, (int, float)):
            numeric = float(item)
            if not math.isfinite(numeric):
                raise ValueError("non-finite provenance number")
            return "d" + struct.pack(">d", numeric).hex()
        if isinstance(item, str):
            encoded = item.encode("utf-8")
            return f"s{len(encoded)}:{encoded.hex()}"
        if isinstance(item, list):
            return f"a{len(item)}[" + "".join(encode(child) for child in item) + "]"
        if isinstance(item, dict):
            keys = sorted(item)
            return (
                f"o{len(keys)}{{"
                + "".join(encode(key) + encode(item[key]) for key in keys)
                + "}"
            )
        raise TypeError(f"unsupported provenance value: {type(item).__name__}")

    return encode(_json_value(value))


def validate_policy_audit(
    value: object,
    *,
    recommended_long: object | None = None,
    risk_score: object | None = None,
    allow_session_jpy: bool = False,
) -> None:
    """Validate the exact aggregate-only r5 policy payload contract."""

    if not isinstance(value, dict) or set(value) != set(POLICY_AUDIT_FIELDS):
        raise ValueError("policy payload must match the exact frozen contract")
    if value.get("model_version") != POLICY_MODEL_VERSION:
        raise ValueError("policy payload model version mismatch")

    def integer(name: str, low: int, high: int) -> int:
        item = value.get(name)
        if isinstance(item, bool) or not isinstance(item, int) or not low <= item <= high:
            raise ValueError(f"policy.{name} must be an integer from {low} to {high}")
        return item

    score_based = integer("score_based_long", 25, 100)
    raw_target = integer("raw_target_long", 25, 100)
    final_long = integer("recommended_long", 25, 100)
    trigger_count = integer("fast_trigger_count", 0, len(FAST_CONDITIONS))
    available_count = integer("fast_available_count", 0, len(FAST_CONDITIONS))
    clear_streak = integer("clear_streak", 0, RECOVERY_CONFIRM_DAYS)

    active = value.get("active_fast_conditions")
    unavailable = value.get("unavailable_fast_conditions")
    if not isinstance(active, list) or not isinstance(unavailable, list):
        raise ValueError("policy Fast condition fields must be lists")
    if not all(isinstance(item, str) for item in (*active, *unavailable)):
        raise ValueError("policy Fast condition keys must be strings")
    if (
        len(active) != len(set(active))
        or len(unavailable) != len(set(unavailable))
        or not set(active).issubset(FAST_CONDITIONS)
        or not set(unavailable).issubset(FAST_CONDITIONS)
        or set(active) & set(unavailable)
    ):
        raise ValueError("policy Fast condition keys are invalid")
    frozen_unavailable = {"portfolio_correlation"}
    if not allow_session_jpy:
        frozen_unavailable.add("jpy_3d_drop")
    if not frozen_unavailable.issubset(unavailable) or frozen_unavailable & set(active):
        raise ValueError("policy non-auditable Fast conditions must be unavailable")
    if trigger_count != len(active) or available_count != len(FAST_CONDITIONS) - len(unavailable):
        raise ValueError("policy Fast counts disagree with condition keys")

    expected_status = (
        "TRIGGERED"
        if trigger_count >= 3
        else "CLEAR"
        if available_count == len(FAST_CONDITIONS)
        else "INCOMPLETE"
    )
    if value.get("fast_status") != expected_status:
        raise ValueError("policy Fast status disagrees with condition keys")
    expected_raw = max(25, score_based - 20) if trigger_count >= 3 else score_based
    if raw_target != expected_raw or final_long > raw_target:
        raise ValueError("policy Long fields disagree with the frozen policy")
    if value.get("phase") not in {
        "COLD_START",
        "RISK_REDUCTION",
        "RECOVERY_HOLD",
        "REENTRY_DIRECT",
        "RECOVERY_CONFIRM",
        "AT_TARGET",
    }:
        raise ValueError("policy phase is invalid")
    episode = value.get("episode_status")
    if episode not in {"ACTIVE", "INACTIVE"} or (episode == "INACTIVE" and clear_streak):
        raise ValueError("policy episode state is invalid")
    driver = value.get("primary_driver")
    if not isinstance(driver, str) or not driver.strip():
        raise ValueError("policy primary_driver must be non-empty")
    if recommended_long is not None:
        if isinstance(recommended_long, bool) or not isinstance(
            recommended_long, (int, np.integer)
        ):
            raise ValueError("top-level recommended_long must be an integer")
        if int(recommended_long) != final_long:
            raise ValueError("top-level recommended_long must equal policy.recommended_long")
    if risk_score is not None:
        if (
            isinstance(risk_score, (bool, np.bool_))
            or not isinstance(risk_score, (int, float, np.integer, np.floating))
            or not math.isfinite(float(risk_score))
            or not float(risk_score).is_integer()
            or not 0 <= int(risk_score) <= 100
        ):
            raise ValueError("top-level market_risk_score must be an integer from 0 to 100")
        if score_based != long_anchor(int(risk_score)):
            raise ValueError("policy score_based_long disagrees with market_risk_score")


def model_versions_for_series(series_version: str) -> dict[str, str]:
    """Keep the immutable r3-r5 provenance contracts stable."""
    if series_version == BREADTH_REPAIR_SERIES_VERSION:
        return R10_MODEL_VERSIONS.copy()
    if series_version == LEGACY_SERIES_VERSION:
        return LEGACY_MODEL_VERSIONS.copy()
    if series_version in {Q20_SERIES_VERSION, Q20_NO_PHASE_SERIES_VERSION}:
        return Q20_MODEL_VERSIONS.copy()
    if series_version in {R5_SERIES_VERSION, R5_NO_PHASE_SERIES_VERSION}:
        return R5_MODEL_VERSIONS.copy()
    if series_version in {R6_SERIES_VERSION, R7_SERIES_VERSION, R8_SERIES_VERSION, R9_SERIES_VERSION}:
        return MODEL_VERSIONS.copy()
    if series_version in {R10_SERIES_VERSION, R11_SERIES_VERSION, R12_SERIES_VERSION, R13_SERIES_VERSION, R15_SERIES_VERSION, R16_SERIES_VERSION, R17_SERIES_VERSION, R18_SERIES_VERSION, R19_SERIES_VERSION, SERIES_VERSION, NO_PHASE_SERIES_VERSION}:
        return R10_MODEL_VERSIONS.copy()
    raise ValueError("unknown RORO SHADOW series version")


def _requires_q20(series_version: str) -> bool:
    model_versions_for_series(series_version)
    if series_version == BREADTH_REPAIR_SERIES_VERSION:
        return True
    return series_version in {
        Q20_SERIES_VERSION,
        Q20_NO_PHASE_SERIES_VERSION,
        R5_SERIES_VERSION,
        R5_NO_PHASE_SERIES_VERSION,
        R6_SERIES_VERSION, R7_SERIES_VERSION, R8_SERIES_VERSION, R9_SERIES_VERSION, R10_SERIES_VERSION, R11_SERIES_VERSION, R12_SERIES_VERSION, R13_SERIES_VERSION, R15_SERIES_VERSION, R16_SERIES_VERSION, R17_SERIES_VERSION, R18_SERIES_VERSION, R19_SERIES_VERSION, SERIES_VERSION,
        NO_PHASE_SERIES_VERSION,
    }


def _requires_policy(series_version: str) -> bool:
    model_versions_for_series(series_version)
    if series_version == BREADTH_REPAIR_SERIES_VERSION:
        return True
    return series_version in {
        R5_SERIES_VERSION,
        R5_NO_PHASE_SERIES_VERSION,
        R6_SERIES_VERSION, R7_SERIES_VERSION, R8_SERIES_VERSION, R9_SERIES_VERSION, R10_SERIES_VERSION, R11_SERIES_VERSION, R12_SERIES_VERSION, R13_SERIES_VERSION, R15_SERIES_VERSION, R16_SERIES_VERSION, R17_SERIES_VERSION, R18_SERIES_VERSION, R19_SERIES_VERSION, SERIES_VERSION,
        NO_PHASE_SERIES_VERSION,
    }


def _allows_session_jpy(series_version: str) -> bool:
    if series_version == BREADTH_REPAIR_SERIES_VERSION:
        return True
    return series_version in {R15_SERIES_VERSION, R16_SERIES_VERSION, R17_SERIES_VERSION, R18_SERIES_VERSION, R19_SERIES_VERSION, SERIES_VERSION, NO_PHASE_SERIES_VERSION}


def _payload_columns_for_series(series_version: str) -> tuple[str, ...]:
    return tuple(
        column
        for column in PAYLOAD_COLUMNS
        if (column != "q20" or _requires_q20(series_version))
        and (column != "policy" or _requires_policy(series_version))
    )


def _provenance(date_value: str, payload_json: str, series_version: str) -> str:
    material = {
        "series_version": series_version,
        "date": date_value,
        # Each row is causal through its own decision date. This keeps an old row
        # idempotent when a later as-of build appends new source observations.
        "input_cutoff": date_value,
        "model_versions": model_versions_for_series(series_version),
        "payload": json.loads(payload_json),
    }
    canonical = (
        _canonical_provenance_encoding(material)
        if series_version in {R6_SERIES_VERSION, R9_SERIES_VERSION, R11_SERIES_VERSION, R12_SERIES_VERSION, R13_SERIES_VERSION, R15_SERIES_VERSION, R16_SERIES_VERSION, R17_SERIES_VERSION, R18_SERIES_VERSION, R19_SERIES_VERSION, SERIES_VERSION, NO_PHASE_SERIES_VERSION, BREADTH_REPAIR_SERIES_VERSION}
        else _canonical_json(material)
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _records(
    frame: pd.DataFrame,
    as_of: str,
    *,
    series_version: str = SERIES_VERSION,
    latest_only: bool = False,
) -> list[tuple[str, ...]]:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("SHADOW frame must use a DatetimeIndex")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("SHADOW frame index must be unique and increasing")
    required = REQUIRED_PAYLOAD_COLUMNS
    if _requires_q20(series_version):
        required = (*required, "q20")
    if _requires_policy(series_version):
        required = (*required, "policy")
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"missing SHADOW payload columns: {', '.join(missing)}")

    cutoff = pd.Timestamp(as_of)
    selected = frame.loc[
        (frame.index >= SHADOW_START) & (frame.index <= cutoff),
        [column for column in _payload_columns_for_series(series_version) if column in frame],
    ]
    if latest_only:
        selected = selected.tail(1)
    if _requires_q20(series_version):
        for value in selected["q20"]:
            validate_q20_readiness_metadata(value)
    if _requires_policy(series_version):
        for _, row in selected.iterrows():
            validate_policy_audit(
                row["policy"],
                recommended_long=row["recommended_long"],
                risk_score=row["market_risk_score"],
                allow_session_jpy=_allows_session_jpy(series_version),
            )
    records: list[tuple[str, ...]] = []
    for timestamp, row in selected.iterrows():
        day = _calendar_date(timestamp, "date")
        payload_json = _canonical_json(row.to_dict())
        records.append(
            (
                series_version,
                day,
                day,
                RORO_MODEL_VERSION,
                AXIS_MODEL_VERSION,
                PORTFOLIO_MODEL_VERSION,
                _provenance(day, payload_json, series_version),
                payload_json,
            )
        )
    return records


def _initialize(connection: sqlite3.Connection) -> None:
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE}(
            series_version TEXT NOT NULL,
            date TEXT NOT NULL,
            input_cutoff TEXT NOT NULL,
            roro_model_version TEXT NOT NULL,
            axis_model_version TEXT NOT NULL,
            portfolio_model_version TEXT NOT NULL,
            provenance_hash TEXT NOT NULL CHECK(length(provenance_hash)=64),
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY(series_version, date)
        ) WITHOUT ROWID
        """
    )
    actual = tuple(row[1] for row in connection.execute(f"PRAGMA table_info({TABLE})"))
    if actual != SCHEMA_COLUMNS:
        raise ValueError(f"incompatible {TABLE} schema")
    connection.execute(
        f"CREATE TRIGGER IF NOT EXISTS {TABLE}_no_update "
        f"BEFORE UPDATE ON {TABLE} BEGIN SELECT RAISE(ABORT,'immutable RORO SHADOW series'); END"
    )
    connection.execute(
        f"CREATE TRIGGER IF NOT EXISTS {TABLE}_no_delete "
        f"BEFORE DELETE ON {TABLE} BEGIN SELECT RAISE(ABORT,'immutable RORO SHADOW series'); END"
    )


def persist_shadow_frame(
    shadow_target: str | Path,
    frame: pd.DataFrame,
    *,
    as_of: str | date | pd.Timestamp,
    series_version: str = SERIES_VERSION,
    latest_only: bool = False,
) -> dict[str, Any]:
    """Insert immutable aggregate rows; identical rows are idempotent."""
    cutoff = _calendar_date(as_of, "as_of")
    target = Path(shadow_target)
    target.parent.mkdir(parents=True, exist_ok=True)
    records = _records(
        frame,
        cutoff,
        series_version=series_version,
        latest_only=latest_only,
    )
    inserted = 0
    idempotent = 0
    with closing(sqlite3.connect(target)) as connection:
        _initialize(connection)
        try:
            connection.execute("BEGIN IMMEDIATE")
            for record in records:
                key = record[:2]
                existing = connection.execute(
                    f"""
                    SELECT series_version,date,input_cutoff,roro_model_version,
                           axis_model_version,portfolio_model_version,
                           provenance_hash,payload_json
                    FROM {TABLE} WHERE series_version=? AND date=?
                    """,
                    key,
                ).fetchone()
                if existing is not None:
                    if tuple(existing) != record:
                        raise ImmutableSeriesConflict(
                            f"immutable SHADOW conflict: {key[0]} {key[1]}"
                        )
                    idempotent += 1
                    continue
                created_at = datetime.now(timezone.utc).isoformat(
                    timespec="microseconds"
                ).replace("+00:00", "Z")
                connection.execute(
                    f"INSERT INTO {TABLE} VALUES (?,?,?,?,?,?,?,?,?)",
                    (*record, created_at),
                )
                inserted += 1
            connection.commit()
        except Exception:
            connection.rollback()
            raise
    return {
        "series_version": series_version,
        "as_of": cutoff,
        "rows_seen": len(records),
        "inserted": inserted,
        "idempotent": idempotent,
    }


def _build_policy_history(
    publish_source: Path,
    panel: pd.DataFrame,
    payload: pd.DataFrame,
    *,
    as_of: str,
    allow_session_jpy: bool = False,
    price_observation_database: Path | None = None,
) -> pd.DataFrame:
    """Replay policy on every ordered NQ decision date, with no calendar gaps."""

    for frame, name in ((panel, "panel"), (payload, "payload")):
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise ValueError(f"{name} must use a DatetimeIndex")
        if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
            raise ValueError(f"{name} index must be unique and increasing")
        if any(timestamp != timestamp.normalize() for timestamp in frame.index):
            raise ValueError(f"{name} index must contain exact calendar dates")
    if "nq" not in panel or "market_risk_score" not in payload:
        raise ValueError("policy replay requires NQ prices and market_risk_score")

    cutoff = pd.Timestamp(as_of)
    nq_calendar = panel.index[
        (panel.index >= SHADOW_START) & (panel.index <= cutoff)
    ]
    decision_calendar = payload.index[
        (payload.index >= SHADOW_START) & (payload.index <= cutoff)
    ]
    if nq_calendar.empty or not decision_calendar.equals(nq_calendar):
        raise ValueError("policy payload must cover every NQ decision date exactly")
    nq = pd.to_numeric(panel.loc[nq_calendar, "nq"], errors="coerce")
    if nq.isna().any() or nq.le(0).any():
        raise ValueError("policy NQ calendar requires a positive exact-date close")

    decisions: list[dict[str, object]] = []
    for timestamp in nq_calendar:
        raw_score = payload.at[timestamp, "market_risk_score"]
        if (
            isinstance(raw_score, (bool, np.bool_))
            or pd.isna(raw_score)
            or not float(raw_score).is_integer()
            or not 0 <= int(raw_score) <= 100
        ):
            raise ValueError("policy market_risk_score must be an integer from 0 to 100")
        day = timestamp.date().isoformat()
        decisions.append(
            {
                "as_of": day,
                "risk_score": int(raw_score),
                "fast_snapshot": _build_roro_fast_snapshot(
                    publish_source, day, allow_session_jpy=allow_session_jpy,
                    price_observation_database=price_observation_database,
                ),
            }
        )

    results = replay_roro_policy(decisions)
    audits: list[dict[str, object]] = []
    final_longs: list[int] = []
    for timestamp, result in zip(nq_calendar, results, strict=True):
        score_based_long = int(result["base_long"])
        upstream_long = payload.at[timestamp, "recommended_long"]
        if pd.isna(upstream_long) or int(upstream_long) != score_based_long:
            raise ValueError("score-based Long disagrees with the frozen RORO anchor")
        audit = {
            "model_version": result["model_version"],
            "score_based_long": score_based_long,
            "raw_target_long": int(result["raw_target_long"]),
            "recommended_long": int(result["recommended_long"]),
            "fast_status": result["fast_status"],
            "fast_trigger_count": int(result["fast_trigger_count"]),
            "fast_available_count": int(result["fast_available_count"]),
            "active_fast_conditions": list(result["active_fast_conditions"]),
            "unavailable_fast_conditions": list(
                result["unavailable_fast_conditions"]
            ),
            "phase": result["phase"],
            "episode_status": result["episode_status"],
            "clear_streak": int(result["clear_streak"]),
            "primary_driver": result["primary_driver"],
        }
        validate_policy_audit(
            audit,
            recommended_long=audit["recommended_long"],
            risk_score=int(payload.at[timestamp, "market_risk_score"]),
            allow_session_jpy=allow_session_jpy,
        )
        audits.append(audit)
        final_longs.append(audit["recommended_long"])

    result_frame = payload.copy()
    result_frame.loc[nq_calendar, "recommended_long"] = pd.Series(
        final_longs, index=nq_calendar, dtype="Int64"
    )
    policy_values = pd.Series([None] * len(result_frame), index=result_frame.index, dtype=object)
    for timestamp, audit in zip(nq_calendar, audits, strict=True):
        policy_values.at[timestamp] = audit
    result_frame["policy"] = policy_values
    return result_frame


def _build_roro_fast_snapshot(
    publish_source: Path, as_of: str, *, allow_session_jpy: bool = False,
    price_observation_database: Path | None = None,
) -> dict[str, Any]:
    """Fail closed on Fast inputs that lack an auditable historical timestamp."""

    snapshot = (
        build_fast_signal_snapshot(
            publish_source, as_of, jpy_observation_database=price_observation_database
        )
        if price_observation_database is not None
        else build_fast_signal_snapshot(publish_source, as_of)
    )
    conditions = [dict(item) for item in snapshot.get("conditions", [])]
    blocked = {"portfolio_correlation": "Historical holdings availability timestamp is not available."}
    if not allow_session_jpy:
        blocked["jpy_3d_drop"] = "USDJPY session-normalized observation is not available."
    by_key = {item.get("key"): item for item in conditions}
    missing = blocked.keys() - by_key.keys()
    if missing:
        raise ValueError(
            "Fast snapshot is missing frozen conditions: " + ", ".join(sorted(missing))
        )
    for key, detail in blocked.items():
        by_key[key].update(
            {
                "status": "UNAVAILABLE",
                "data_quality": "UNAVAILABLE",
                "as_of": as_of,
                "detail": detail,
            }
        )
    triggered = [item for item in conditions if item.get("status") == "TRIGGERED"]
    complete = all(item.get("status") != "UNAVAILABLE" for item in conditions)
    status = "TRIGGERED" if len(triggered) >= 3 else "CLEAR" if complete else "INCOMPLETE"
    primary_driver = (
        ", ".join(str(item["label"]) for item in triggered[:3])
        if triggered
        else "Fast 조건 3개 미만"
        if complete
        else "관측 불충분"
    )
    return {
        **snapshot,
        "conditions": conditions,
        "count": len(triggered),
        "trigger_count": len(triggered),
        "complete": complete,
        "status": status,
        "primary_driver": primary_driver,
    }


def _validate_price_observation_database(
    database: str | Path, shadow_target: Path
) -> Path:
    """Require a separate, non-empty frozen-contract observation database."""

    path = Path(database)
    if not path.is_file():
        raise FileNotFoundError(path)
    if _same_file(path, shadow_target):
        raise ValueError("shadow_target must be separate from the price observation database")
    try:
        with closing(_read_only(path)) as connection:
            row = connection.execute(
                f"SELECT COUNT(*) FROM {PRICE_OBSERVATION_TABLE} "
                "WHERE contract_version=? AND source=? AND ticker IN ('JPY=X','GLD')",
                (PRICE_OBSERVATION_MODEL_VERSION, PRICE_OBSERVATION_SOURCE),
            ).fetchone()
    except sqlite3.DatabaseError as error:
        raise ValueError("invalid price observation database") from error
    if row is None or int(row[0]) <= 0:
        raise ValueError("price observation database has no frozen-contract observations")
    return path


def _validate_breadth_observation_database(database: str | Path, shadow_target: Path) -> Path:
    path = Path(database)
    if not path.is_file():
        raise FileNotFoundError(path)
    if _same_file(path, shadow_target):
        raise ValueError("shadow_target must be separate from the breadth observation database")
    try:
        with closing(_read_only(path)) as connection:
            row = connection.execute(
                f"SELECT COUNT(*) FROM {BREADTH_OBSERVATION_TABLE} WHERE contract_version=? AND source=? AND symbol='NDTH'",
                (BREADTH_OBSERVATION_MODEL_VERSION, BREADTH_OBSERVATION_SOURCE),
            ).fetchone()
    except sqlite3.DatabaseError as error:
        raise ValueError("invalid breadth observation database") from error
    if row is None or int(row[0]) <= 0:
        raise ValueError("breadth observation database has no approved NDTH observations")
    return path


def _latest_ready_cutoff(axis_scores: pd.DataFrame, requested_as_of: str) -> str:
    """Return the last date with the two required axes and a publishable score."""
    if not isinstance(axis_scores.index, pd.DatetimeIndex):
        raise ValueError("axis scores must use a DatetimeIndex")
    if axis_scores.index.has_duplicates or not axis_scores.index.is_monotonic_increasing:
        raise ValueError("axis scores index must be unique and increasing")
    if any(timestamp != timestamp.normalize() for timestamp in axis_scores.index):
        raise ValueError("axis scores index must contain exact calendar dates")
    required = ["market_risk_score", "market_trend_score", "portfolio_internal_score"]
    missing = [column for column in required if column not in axis_scores]
    if missing:
        raise ValueError("axis scores are missing: " + ", ".join(missing))
    eligible = axis_scores.loc[
        (axis_scores.index >= SHADOW_START)
        & (axis_scores.index <= pd.Timestamp(requested_as_of)),
        required,
    ].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    score = eligible["market_risk_score"]
    ready = (
        eligible.notna().all(axis=1)
        & eligible.ge(0).all(axis=1)
        & eligible.le(100).all(axis=1)
        & score.eq(score.round())
    )
    if eligible.empty or eligible.index[0] != SHADOW_START or not ready.iloc[0]:
        raise ValueError("no model-ready Market Lens date at or before requested as_of")
    first_gap = np.flatnonzero(~ready.to_numpy())
    last_ready = len(eligible) - 1 if not len(first_gap) else int(first_gap[0]) - 1
    return _calendar_date(eligible.index[last_ready], "effective as_of")


def _apply_stored_score_trajectories(
    frame: pd.DataFrame, history: list[dict[str, Any]]
) -> pd.DataFrame:
    """Use immutable published scores for the latest row's displayed changes."""
    result = frame.copy()
    if result.empty:
        return result
    latest = result.index.max()
    current = result.at[latest, "market_risk_score"]
    previous = [
        row["payload"]["market_risk_score"]
        for row in history
        if pd.Timestamp(row["date"]) < latest
    ]
    for lag in (1, 5, 20):
        result.at[latest, f"trajectory_{lag}d"] = (
            float(current - previous[-lag]) if len(previous) >= lag else pd.NA
        )
    return result


def build_shadow_series(
    publish_source: str | Path,
    private_source: str | Path,
    shadow_target: str | Path,
    *,
    price_observation_database: str | Path,
    breadth_observation_database: str | Path | None = None,
    as_of: str | date | pd.Timestamp,
    no_phase: bool = False,
    latest_only: bool = False,
    narrative_api_key: str | None = None,
    require_complete_breadth: bool = False,
    repair_missing_breadth: bool = False,
) -> dict[str, Any]:
    """Build aggregate RORO rows from read-only sources into a separate DB."""
    cutoff = _calendar_date(as_of, "as_of")
    if repair_missing_breadth and (not latest_only or no_phase or not require_complete_breadth):
        raise ValueError("breadth repair requires latest_only, complete breadth and production phase")
    publish, private, target = _validate_paths(
        publish_source, private_source, shadow_target
    )
    price_observations = _validate_price_observation_database(
        price_observation_database, target
    )
    breadth_observations = (
        _validate_breadth_observation_database(breadth_observation_database, target)
        if breadth_observation_database is not None
        else None
    )
    panel = load_roro_panel(
        publish,
        as_of=cutoff,
        price_observation_database=price_observations,
        **({"breadth_observation_database": breadth_observations}
           if breadth_observations is not None else {}),
    )
    # Gate before portfolio work, narrative API calls, and especially immutable persistence.
    if require_complete_breadth:
        if breadth_observations is None:
            raise RuntimeError("시장 참여 폭 입력 미완료: NDTH 관측 DB가 필요합니다.")
        readiness = breadth_readiness(breadth_observations, panel.index, cutoff)
        if readiness["status"] != "READY":
            raise RuntimeError(str(readiness["message"]))
    roro_scores = build_roro_scores(panel)
    portfolio = load_portfolio_internal_series(
        private,
        publish,
        as_of=cutoff,
        include_warmup=True,
    )
    axis_scores = build_roro_axis_scores(roro_scores, portfolio)
    if require_complete_breadth and (
        pd.Timestamp(cutoff) not in axis_scores.index
        or pd.isna(axis_scores.loc[pd.Timestamp(cutoff)].get("breadth_risk_percentile"))
    ):
        raise RuntimeError(f"시장 참여 폭 점수 미완료: {cutoff} 원자료 또는 과거 비교 표본을 확인하세요. 저장·게시하지 않았습니다.")
    effective_cutoff = _latest_ready_cutoff(axis_scores, cutoff) if latest_only else cutoff
    pending = effective_cutoff != cutoff
    if require_complete_breadth and pending:
        raise RuntimeError(f"Market Lens 입력 미완료: 요청일 {cutoff}, 생성 가능일 {effective_cutoff}; 저장·게시하지 않았습니다.")
    if latest_only:
        effective_timestamp = pd.Timestamp(effective_cutoff)
        panel = panel.loc[:effective_timestamp]
        portfolio = portfolio.loc[:effective_timestamp]
        axis_scores = axis_scores.loc[:effective_timestamp]
    series_version = NO_PHASE_SERIES_VERSION if no_phase else SERIES_VERSION
    existing = []
    stored_history: list[dict[str, Any]] = []
    if latest_only and target.is_file():
        with closing(_read_only(target)) as connection:
            initialized = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (TABLE,)
            ).fetchone()
        if initialized:
            stored_history = read_shadow_history(
                target,
                series_version=series_version,
                end=effective_cutoff,
            )
            existing = [row for row in stored_history if row["date"] == effective_cutoff]
            if not no_phase:
                repairs = read_shadow_history(target, series_version=BREADTH_REPAIR_SERIES_VERSION, end=effective_cutoff)
                by_date = {row["date"]: row for row in stored_history}
                by_date.update({row["date"]: row for row in repairs})
                stored_history = [by_date[day] for day in sorted(by_date)]
                existing = [row for row in stored_history if row["date"] == effective_cutoff]
    warnings = (
        [f"requested {cutoff} is not model-ready; using {effective_cutoff}"]
        if pending else []
    )
    if existing and repair_missing_breadth and existing[0]["payload"].get("breadth_risk_percentile") is None:
        if existing[0]["series_version"] != SERIES_VERSION:
            raise RuntimeError("기존 보정 버전도 불완전합니다. 추가 검토 없이 덮어쓰지 않습니다.")
        series_version = BREADTH_REPAIR_SERIES_VERSION
        warnings.append("late breadth correction: original observation preserved; not an original point-in-time signal")
        existing = []
    if existing:
        if require_complete_breadth and existing[0]["payload"].get("breadth_risk_percentile") is None:
            raise RuntimeError("저장된 시장 참여 폭 점수가 결측입니다. --repair-missing-breadth로 별도 보정 버전을 생성하세요.")
        series_version = existing[0]["series_version"]
        return {
            "series_version": series_version,
            "as_of": effective_cutoff,
            "effective_as_of": effective_cutoff,
            "requested_as_of": cutoff,
            "pending": pending,
            "warnings": warnings,
            "rows_seen": 1,
            "inserted": 0,
            "idempotent": 1,
            "reused_existing": True,
            "phase_included": not no_phase,
            "latest_only": True,
            "target": str(target.resolve()),
        }
    if latest_only:
        if series_version in {SERIES_VERSION, BREADTH_REPAIR_SERIES_VERSION}:
            prior_history = read_shadow_history(target, series_version=R19_SERIES_VERSION, end=effective_cutoff) if target.is_file() else []
            by_date = {row["date"]: row for row in prior_history if row["date"] < effective_cutoff}
            by_date.update({row["date"]: row for row in stored_history})
            stored_history = [by_date[day] for day in sorted(by_date)]
        axis_scores = _apply_stored_score_trajectories(axis_scores, stored_history)
    payload = axis_scores.join(
        portfolio.loc[:, PORTFOLIO_RAW_COLUMNS], how="left", validate="one_to_one"
    )
    if not no_phase:
        phase = build_trend_phase(axis_scores, panel, portfolio)
        if not phase.index.equals(axis_scores.index):
            raise ValueError("phase output must exactly match the axis calendar")
        payload = payload.join(phase, how="left", validate="one_to_one")
    q20_daily = pd.DataFrame(
        {"date": panel.index, "nq_close": panel["nq"]}, index=panel.index
    )
    for column in Q20_FEATURE_COLUMNS:
        q20_daily[column] = (
            payload[column].reindex(panel.index) if column in payload else np.nan
        )
    q20_daily.loc[q20_daily.index < SHADOW_START, list(Q20_FEATURE_COLUMNS)] = np.nan
    q20_history = build_q20_readiness_history(
        q20_daily.reset_index(drop=True),
        decision_dates=payload.index[
            (payload.index >= SHADOW_START)
            & (payload.index <= pd.Timestamp(effective_cutoff))
        ],
    )
    q20_values = pd.Series([None] * len(payload), index=payload.index, dtype=object)
    for timestamp, value in q20_history["q20"].items():
        q20_values.at[timestamp] = value
    payload = payload.assign(q20=q20_values)
    payload = _build_policy_history(
        publish, panel, payload, as_of=effective_cutoff,
        allow_session_jpy=_allows_session_jpy(series_version),
        price_observation_database=price_observations,
    )
    if not payload.empty:
        from crash_guard import get_crash_guard_narrative_context
        from roro_narrative import generate_summary, fallback_summary
        latest = payload.index.max()
        try:
            crash_guard = get_crash_guard_narrative_context(
                publish, effective_cutoff
            ) if narrative_api_key else None
        except Exception:
            crash_guard = None
        summary = generate_summary(
            payload.loc[latest].to_dict(), narrative_api_key or "",
            crash_guard=crash_guard,
        )
        policy = dict(payload.at[latest, "policy"])
        policy["primary_driver"] = summary or fallback_summary(payload.loc[latest].to_dict())
        payload.at[latest, "policy"] = policy
    result = persist_shadow_frame(
        target,
        payload,
        as_of=effective_cutoff,
        series_version=series_version,
        latest_only=latest_only,
    )
    result["phase_included"] = not no_phase
    result["latest_only"] = latest_only
    result["effective_as_of"] = effective_cutoff
    result["requested_as_of"] = cutoff
    result["pending"] = pending
    result["warnings"] = warnings
    result["reused_existing"] = False
    result["target"] = str(target.resolve())
    return result


def _read_only(database: str | Path) -> sqlite3.Connection:
    path = Path(database)
    if not path.is_file():
        raise FileNotFoundError(path)
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    return connection


def _validated_observation(row: tuple[Any, ...], expected_series: str) -> dict[str, Any]:
    observation = dict(zip(SCHEMA_COLUMNS, row))
    day = _calendar_date(observation["date"], "stored date")
    _validate_utc_z_timestamp(observation["created_at"], "stored created_at")
    if observation["series_version"] != expected_series or observation["input_cutoff"] != day:
        raise ValueError("stored SHADOW series/date metadata mismatch")
    if (
        observation["roro_model_version"] != RORO_MODEL_VERSION
        or observation["axis_model_version"] != AXIS_MODEL_VERSION
        or observation["portfolio_model_version"] != PORTFOLIO_MODEL_VERSION
    ):
        raise ValueError("stored SHADOW model version mismatch")
    try:
        payload = json.loads(observation["payload_json"])
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("stored SHADOW payload is invalid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("stored SHADOW payload must be an object")
    canonical_payload = _canonical_json(payload)
    if observation["payload_json"] != canonical_payload:
        raise ValueError("stored SHADOW payload is not canonical")
    if observation["provenance_hash"] != _provenance(day, canonical_payload, expected_series):
        raise ValueError("stored SHADOW provenance hash mismatch")
    required = set(REQUIRED_PAYLOAD_COLUMNS)
    if _requires_q20(expected_series):
        required.add("q20")
        validate_q20_readiness_metadata(payload.get("q20"))
    if _requires_policy(expected_series):
        required.add("policy")
        validate_policy_audit(
            payload.get("policy"),
            recommended_long=payload.get("recommended_long"),
            risk_score=payload.get("market_risk_score"),
            allow_session_jpy=_allows_session_jpy(expected_series),
        )
    if not required.issubset(payload):
        raise ValueError("stored SHADOW payload is missing required fields")
    observation["payload"] = payload
    observation.pop("payload_json")
    return observation


def read_shadow_history(
    shadow_target: str | Path,
    *,
    series_version: str = SERIES_VERSION,
    start: str | date | pd.Timestamp | None = None,
    end: str | date | pd.Timestamp | None = None,
) -> list[dict[str, Any]]:
    model_versions_for_series(series_version)
    clauses = ["series_version=?"]
    parameters: list[str] = [series_version]
    if start is not None:
        clauses.append("date>=?")
        parameters.append(_calendar_date(start, "start"))
    if end is not None:
        clauses.append("date<=?")
        parameters.append(_calendar_date(end, "end"))
    with closing(_read_only(shadow_target)) as connection:
        rows = connection.execute(
            f"SELECT {','.join(SCHEMA_COLUMNS)} FROM {TABLE} "
            f"WHERE {' AND '.join(clauses)} ORDER BY date",
            parameters,
        ).fetchall()
    history = []
    for row in rows:
        history.append(_validated_observation(row, series_version))
    return history


def latest_shadow(
    shadow_target: str | Path, *, series_version: str = SERIES_VERSION
) -> dict[str, Any] | None:
    history = read_shadow_history(shadow_target, series_version=series_version)
    return history[-1] if history else None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="build the isolated SHADOW series")
    build.add_argument("publish_source")
    build.add_argument("private_source")
    build.add_argument("shadow_target")
    build.add_argument("--price-observation-database", required=True)
    build.add_argument("--breadth-observation-database")
    build.add_argument("--as-of", required=True)
    build.add_argument(
        "--no-phase", action="store_true", help="persist a separate diagnostic series"
    )
    build.add_argument(
        "--latest-only",
        action="store_true",
        help="calculate full history but persist only the last row at or before cutoff",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "build":
        breadth_kwargs = (
            {"breadth_observation_database": args.breadth_observation_database}
            if args.breadth_observation_database is not None
            else {}
        )
        result = build_shadow_series(
            args.publish_source,
            args.private_source,
            args.shadow_target,
            price_observation_database=args.price_observation_database,
            **breadth_kwargs,
            as_of=args.as_of,
            no_phase=args.no_phase,
            latest_only=args.latest_only,
        )
        print(_canonical_json(result))


if __name__ == "__main__":
    main()
