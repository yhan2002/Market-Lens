"""Research-only pooled GICS trend-turn experiment.

This module reads the audited Bloomberg handoff files directly and writes only
to an explicitly supplied output directory.  It never mutates a database or a
published site.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from gics_l2_shadow_experiment import build_cohort_prices, cohort_codes, load_package
from turning_macro_shadow_experiment import (
    build_macro_feature_sets,
    ensure_output_boundary,
    load_macro_history,
    merge_macro_next_session,
)
from research_drawdown_onset import causal_trailing_percentile


VERDICT = "RESEARCH_SHADOW_ONLY"
PIT_STATUS = (
    "PIT_PARTIAL/CURRENT_REVISED_HISTORY_ONLY/"
    "GICS_CURRENT_HIERARCHY_BACKCAST_NOT_VERIFIED"
)
PROMOTION_STATUS = "PRODUCTION_PROMOTION_PROHIBITED"
FROZEN_MODEL_VERSION = "gics-core24-pre-pivot-price-native-v1"
PROSPECTIVE_EARLIEST_INPUT_DATE = pd.Timestamp("2026-09-01")
EXPECTED_GICS_UNIVERSE_SHA256 = "AF03A5E0DBDDC186ED4BA2771E19BB4A7CCBFD79AE4AE00B38A795D493EBA9F9"
EXPECTED_GICS_TRAINING_HISTORY_SHA256 = (
    "7B8F8C58ADB76E94EE500578DEBC7DB519567199C9A327A99E1176465FBE443D"
)
FROZEN_RUNTIME_FILES = (
    "sector_turning_pooled_shadow.py",
    "gics_l2_shadow_experiment.py",
    "turning_macro_shadow_experiment.py",
    "research_drawdown_onset.py",
    "sector_rotation_forecast.py",
    "sector_rotation.py",
)
ALERT_CUTOFF = 0.80
PRE_PIVOT_ALERT_CUTOFF = 0.90
PRE_PIVOT_PRIORITY_CUTOFF = 0.95
EVALUATION_START = pd.Timestamp("2020-01-01")
STATES = ("UNKNOWN", "UP_ACTIVE", "DOWN_ACTIVE")
DIRECTIONS = ("ROTATION_OUT", "ROTATION_IN")
TARGET_MODES = ("CONFIRMATION", "PRE_PIVOT")
MODELS = ("POOLED_PRICE", "POOLED_COMMON_MACRO", "POOLED_PARTIAL_MACRO")
PRICE_FEATURES = (
    "relative_return_5",
    "relative_return_20",
    "relative_volatility_20",
    "relative_ma200_gap",
    "relative_drawdown_60",
    "relative_acceleration_5",
)
MACRO_FEATURES = (
    "curve_2s10s_change_20",
    "real_rate_10y_change_20",
    "breakeven_10y_change_20",
    "brent_spot_change_20",
)
CONTINUOUS_FEATURES = (*PRICE_FEATURES, *MACRO_FEATURES)
PREDICTION_OUTPUT_COLUMNS = (
    "direction",
    "model",
    "target_mode",
    "date",
    "_position",
    "sector",
    "sector_name",
    "state",
    "pivot_position",
    "pivot_date",
    "confirmation_position",
    "confirmation_date",
    "barrier",
    "cooldown_remaining",
    "event",
    "at_risk",
    "target",
    "label_end_position",
    "label_end_date",
    "target_event_pivot_position",
    "target_event_pivot_date",
    "target_event_confirmation_position",
    "target_event_confirmation_date",
    "macro_asof_date",
    "refit_date",
    "training_start_date",
    "training_end_date",
    "latest_training_label_end_date",
    "training_date_count",
    "score",
    "raw_score",
    "evaluation_sample",
)
COEFFICIENT_OUTPUT_COLUMNS = (
    "direction",
    "model",
    "refit_date",
    "factor",
    "sector",
    "common",
    "deviation",
    "total",
    "train_mean",
    "train_scale",
)


def code_bundle_sha256(directory: Path | None = None) -> str:
    root = Path(__file__).resolve().parent if directory is None else Path(directory).resolve()
    digest = hashlib.sha256()
    for name in FROZEN_RUNTIME_FILES:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"frozen runtime file missing: {path}")
        payload = path.read_bytes()
        digest.update(name.encode("utf-8"))
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest().upper()


def gics_input_fingerprints(
    master: pd.DataFrame,
    groups: pd.DataFrame,
    benchmark: pd.DataFrame,
    codes: Sequence[str],
) -> tuple[str, str]:
    resolved_codes = tuple(sorted(str(code) for code in codes))
    selected_master = master.loc[
        master["industry_group_code"].isin(resolved_codes),
        [
            "industry_group_code",
            "industry_group_name",
            "parent_sector_code",
            "group_security",
            "group_bbgid",
            "parent_security",
            "parent_bbgid",
        ],
    ].sort_values("industry_group_code")
    if selected_master["industry_group_code"].tolist() != list(resolved_codes):
        raise ValueError("Core24 universe fingerprint is incomplete")
    universe_digest = hashlib.sha256()
    for row in selected_master.itertuples(index=False, name=None):
        universe_digest.update(("|".join(str(value) for value in row) + "\n").encode("utf-8"))

    cutoff = PROSPECTIVE_EARLIEST_INPUT_DATE
    selected_groups = groups.loc[
        groups["industry_group_code"].isin(resolved_codes) & groups["date"].lt(cutoff),
        ["industry_group_code", "date", "total_return_level"],
    ].sort_values(["industry_group_code", "date"])
    selected_benchmark = benchmark.loc[
        benchmark["date"].lt(cutoff), ["date", "total_return_level"]
    ].sort_values("date")
    if selected_groups.empty or selected_benchmark.empty:
        raise ValueError("frozen training history is empty")
    history_digest = hashlib.sha256()
    for code, day, value in selected_groups.itertuples(index=False, name=None):
        payload = f"G|{code}|{pd.Timestamp(day).date()}|{float(value):.17g}\n"
        history_digest.update(payload.encode("ascii"))
    for day, value in selected_benchmark.itertuples(index=False, name=None):
        payload = f"B|SPX|{pd.Timestamp(day).date()}|{float(value):.17g}\n"
        history_digest.update(payload.encode("ascii"))
    return universe_digest.hexdigest().upper(), history_digest.hexdigest().upper()


def prospective_session_fingerprints(
    prices: pd.DataFrame, codes: Sequence[str]
) -> pd.DataFrame:
    resolved_codes = tuple(sorted(str(code) for code in codes))
    missing = sorted({"SPX", *resolved_codes} - set(prices.columns))
    if missing:
        raise ValueError("prospective price columns missing: " + ", ".join(missing))
    rows: list[dict[str, Any]] = []
    for day, values in prices.loc[prices.index >= PROSPECTIVE_EARLIEST_INPUT_DATE].iterrows():
        digest = hashlib.sha256()
        benchmark_value = values["SPX"]
        digest.update(
            f"B|SPX|{pd.Timestamp(day).date()}|{float(benchmark_value):.17g}\n".encode(
                "ascii"
            )
        )
        present = 0
        for code in resolved_codes:
            value = values[code]
            encoded = "MISSING" if pd.isna(value) else f"{float(value):.17g}"
            present += int(pd.notna(value))
            digest.update(
                f"G|{code}|{pd.Timestamp(day).date()}|{encoded}\n".encode("ascii")
            )
        rows.append(
            {
                "date": pd.Timestamp(day).normalize(),
                "input_session_sha256": digest.hexdigest().upper(),
                "present_group_count": present,
                "expected_group_count": len(resolved_codes),
            }
        )
    return pd.DataFrame.from_records(
        rows,
        columns=[
            "date",
            "input_session_sha256",
            "present_group_count",
            "expected_group_count",
        ],
    )


def ensure_prospective_input_sessions(path: Path, current: pd.DataFrame) -> int:
    required = {
        "date",
        "input_session_sha256",
        "present_group_count",
        "expected_group_count",
    }
    missing = sorted(required - set(current.columns))
    if missing:
        raise ValueError("prospective input ledger missing columns: " + ", ".join(missing))
    incoming = current.loc[:, sorted(required)].copy()
    incoming["date"] = pd.to_datetime(incoming["date"], errors="raise").dt.strftime(
        "%Y-%m-%d"
    )
    incoming["input_session_sha256"] = (
        incoming["input_session_sha256"].astype(str).str.upper()
    )
    if incoming.duplicated("date").any():
        raise ValueError("prospective input ledger has duplicate incoming dates")
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=incoming.columns)
    if not existing.empty:
        missing_existing = sorted(required - set(existing.columns))
        if missing_existing:
            raise RuntimeError(
                "stored prospective input ledger missing columns: "
                + ", ".join(missing_existing)
            )
        existing["date"] = pd.to_datetime(existing["date"], errors="raise").dt.strftime(
            "%Y-%m-%d"
        )
        if existing.duplicated("date").any():
            raise RuntimeError("stored prospective input ledger has duplicate dates")
        overlap = existing.merge(incoming, on="date", suffixes=("_old", "_new"))
        compared = ("input_session_sha256", "present_group_count", "expected_group_count")
        if any(
            overlap[f"{column}_old"].astype(str).ne(overlap[f"{column}_new"].astype(str)).any()
            for column in compared
        ):
            raise RuntimeError("previously observed prospective GICS input was revised")
        incoming = incoming.loc[~incoming["date"].isin(existing["date"])]
    combined = pd.concat([existing, incoming], ignore_index=True).sort_values("date")
    temporary = path.with_suffix(path.suffix + ".tmp")
    combined.to_csv(temporary, index=False)
    temporary.replace(path)
    return int(len(incoming))


def ensure_frozen_manifest(path: Path, manifest: dict[str, Any]) -> str:
    clean = _json_ready(manifest)
    version = str(clean["model_version"])
    registry = {"schema_version": 1, "models": {}}
    if path.exists():
        registry = json.loads(path.read_text(encoding="utf-8"))
        if registry.get("schema_version") != 1 or not isinstance(registry.get("models"), dict):
            raise RuntimeError("invalid frozen manifest registry")
    existing = registry["models"].get(version)
    if existing is not None and existing != clean:
        raise RuntimeError("frozen manifest changed without a model version bump")
    if existing is None:
        registry["models"][version] = clean
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(registry, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        temporary.replace(path)
    payload = json.dumps(clean, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


def validate_expected_latest_date(actual: Any, expected: str | None) -> pd.Timestamp:
    if not expected:
        raise ValueError("--expected-latest-date is required for prospective logging")
    try:
        resolved = pd.Timestamp(expected).normalize()
    except (TypeError, ValueError) as exc:
        raise ValueError("--expected-latest-date must be YYYY-MM-DD") from exc
    observed = pd.Timestamp(actual).normalize()
    if observed != resolved:
        raise RuntimeError(
            f"latest GICS date {observed.date()} != expected completed market date "
            f"{resolved.date()}"
        )
    return resolved


def _validate_price(price: pd.Series) -> pd.Series:
    if not isinstance(price.index, pd.DatetimeIndex):
        raise ValueError("price index must be a DatetimeIndex")
    if price.empty or price.index.has_duplicates:
        raise ValueError("price must be non-empty with unique dates")
    if not price.index.is_monotonic_increasing:
        raise ValueError("price dates must be strictly increasing")
    values = pd.to_numeric(price, errors="coerce").astype(float)
    if values.isna().any() or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("price values must be finite and positive")
    values.index = values.index.tz_localize(None).normalize()
    if values.index.has_duplicates:
        raise ValueError("normalized price dates must be unique")
    return values


def directional_change_labels(
    price: pd.Series,
    *,
    horizon: int = 10,
    cooldown_sessions: int = 5,
    volatility_window: int = 60,
    volatility_multiplier: float = 1.5,
    barrier_horizon: int = 10,
    minimum_barrier: float = 0.03,
    maximum_barrier: float = 0.10,
) -> pd.DataFrame:
    """Label symmetric top/bottom confirmations without using future inputs.

    The barrier is a clipped log-price move.  It is computed from shifted
    trailing log-return volatility and frozen whenever the active extreme moves.
    """

    if horizon < 1 or cooldown_sessions < 0 or volatility_window < 2:
        raise ValueError("horizon>=1, cooldown_sessions>=0, volatility_window>=2 required")
    if barrier_horizon < 1 or not 0 < minimum_barrier <= maximum_barrier:
        raise ValueError("invalid directional-change barrier settings")
    series = _validate_price(price)
    log_price = np.log(series)
    candidate = (
        log_price.diff()
        .rolling(volatility_window, min_periods=volatility_window)
        .std()
        .shift(1)
        .mul(volatility_multiplier * np.sqrt(barrier_horizon))
        .clip(minimum_barrier, maximum_barrier)
    )
    size = len(series)
    state_values = np.full(size, "UNKNOWN", dtype=object)
    pivot_values = np.full(size, np.nan)
    confirmation_values = np.full(size, np.nan)
    pivot_dates = np.full(size, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    confirmation_dates = np.full(size, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    barrier_values = np.full(size, np.nan)
    cooldown_values = np.zeros(size, dtype=int)
    event_values = np.full(size, "", dtype=object)
    top_end = np.zeros(size, dtype=bool)
    bottom_end = np.zeros(size, dtype=bool)
    top_at_risk = np.zeros(size, dtype=bool)
    bottom_at_risk = np.zeros(size, dtype=bool)

    state = "UNKNOWN"
    pivot_position: int | None = None
    frozen_barrier = float("nan")
    cooldown = 0
    unknown_high_position: int | None = None
    unknown_low_position: int | None = None
    logs = log_price.to_numpy(dtype=float)
    barriers = candidate.to_numpy(dtype=float)
    dates = series.index.to_numpy(dtype="datetime64[ns]")

    for position in range(size):
        available_barrier = barriers[position]
        if state == "UNKNOWN":
            if np.isfinite(available_barrier):
                if unknown_high_position is None:
                    unknown_high_position = unknown_low_position = position
                assert unknown_low_position is not None
                if logs[position] > logs[unknown_high_position]:
                    unknown_high_position = position
                if logs[position] < logs[unknown_low_position]:
                    unknown_low_position = position
                up_score = (
                    logs[position] - logs[unknown_low_position]
                ) / available_barrier
                down_score = (
                    logs[unknown_high_position] - logs[position]
                ) / available_barrier
                if max(up_score, down_score) >= 1 and up_score != down_score:
                    state = "UP_ACTIVE" if up_score > down_score else "DOWN_ACTIVE"
                    pivot_position = (
                        unknown_high_position if state == "UP_ACTIVE" else unknown_low_position
                    )
                    pivot_barrier = barriers[pivot_position]
                    frozen_barrier = float(
                        pivot_barrier if np.isfinite(pivot_barrier) else available_barrier
                    )
            state_values[position] = state
            if pivot_position is not None:
                pivot_values[position] = pivot_position
                pivot_dates[position] = dates[pivot_position]
                barrier_values[position] = frozen_barrier
                top_at_risk[position] = state == "UP_ACTIVE"
                bottom_at_risk[position] = state == "DOWN_ACTIVE"
            continue

        row_state = state
        assert pivot_position is not None
        if (state == "UP_ACTIVE" and logs[position] > logs[pivot_position]) or (
            state == "DOWN_ACTIVE" and logs[position] < logs[pivot_position]
        ):
            pivot_position = position
            if np.isfinite(available_barrier):
                frozen_barrier = float(available_barrier)

        event = ""
        if cooldown == 0 and np.isfinite(frozen_barrier):
            move_from_extreme = logs[position] - logs[pivot_position]
            if state == "UP_ACTIVE" and move_from_extreme <= -frozen_barrier:
                event = "TOP_END"
                top_end[position] = True
            elif state == "DOWN_ACTIVE" and move_from_extreme >= frozen_barrier:
                event = "BOTTOM_END"
                bottom_end[position] = True

        state_values[position] = row_state
        pivot_values[position] = pivot_position
        pivot_dates[position] = dates[pivot_position]
        barrier_values[position] = frozen_barrier
        cooldown_values[position] = cooldown_sessions if event else cooldown
        event_values[position] = event
        if event:
            confirmation_values[position] = position
            confirmation_dates[position] = dates[position]
        else:
            top_at_risk[position] = row_state == "UP_ACTIVE" and cooldown == 0
            bottom_at_risk[position] = row_state == "DOWN_ACTIVE" and cooldown == 0

        if event:
            state = "DOWN_ACTIVE" if event == "TOP_END" else "UP_ACTIVE"
            pivot_position = position
            frozen_barrier = float(available_barrier)
            cooldown = cooldown_sessions
        elif cooldown:
            cooldown -= 1

    top_target = np.full(size, np.nan)
    bottom_target = np.full(size, np.nan)
    top_label_end = np.full(size, np.nan)
    bottom_label_end = np.full(size, np.nan)
    top_label_date = np.full(size, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    bottom_label_date = np.full(size, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    for position in range(size - horizon):
        label_end = position + horizon
        if top_at_risk[position]:
            top_target[position] = float(top_end[position + 1 : label_end + 1].any())
            top_label_end[position] = label_end
            top_label_date[position] = dates[label_end]
        if bottom_at_risk[position]:
            bottom_target[position] = float(bottom_end[position + 1 : label_end + 1].any())
            bottom_label_end[position] = label_end
            bottom_label_date[position] = dates[label_end]

    return pd.DataFrame(
        {
            "state": state_values,
            "pivot_position": pivot_values,
            "pivot_date": pd.to_datetime(pivot_dates),
            "confirmation_position": confirmation_values,
            "confirmation_date": pd.to_datetime(confirmation_dates),
            "barrier": barrier_values,
            "cooldown_remaining": cooldown_values,
            "event": event_values,
            "top_end": top_end,
            "bottom_end": bottom_end,
            "top_at_risk": top_at_risk,
            "bottom_at_risk": bottom_at_risk,
            "top_target": top_target,
            "bottom_target": bottom_target,
            "top_label_end_position": top_label_end,
            "bottom_label_end_position": bottom_label_end,
            "top_label_end_date": pd.to_datetime(top_label_date),
            "bottom_label_end_date": pd.to_datetime(bottom_label_date),
        },
        index=series.index,
    )


def _price_features(price: pd.Series) -> pd.DataFrame:
    one_day = price.pct_change(fill_method=None)
    return_5 = price.pct_change(5, fill_method=None)
    return pd.DataFrame(
        {
            "relative_return_5": return_5,
            "relative_return_20": price.pct_change(20, fill_method=None),
            "relative_volatility_20": one_day.rolling(20).std() * np.sqrt(252),
            "relative_ma200_gap": price / price.rolling(200).mean() - 1,
            "relative_drawdown_60": price / price.rolling(60).max() - 1,
            "relative_acceleration_5": return_5 - return_5.shift(5),
        },
        index=price.index,
    )


def _labels_with_calendar_gaps(relative: pd.Series, *, horizon: int) -> pd.DataFrame:
    label_parts: list[pd.DataFrame] = []
    available = relative.notna()
    for _, segment in relative.loc[available].groupby((~available).cumsum().loc[available]):
        labels = directional_change_labels(segment, horizon=horizon)
        offset = int(relative.index.get_indexer([segment.index[0]])[0])
        for column in (
            "pivot_position",
            "confirmation_position",
            "top_label_end_position",
            "bottom_label_end_position",
        ):
            labels[column] = labels[column] + offset
        label_parts.append(labels)
    labels = pd.concat(label_parts).reindex(relative.index)
    labels["state"] = labels["state"].fillna("UNKNOWN")
    labels["event"] = labels["event"].fillna("")
    for column in ("top_end", "bottom_end", "top_at_risk", "bottom_at_risk"):
        labels[column] = labels[column].fillna(False).astype(bool)
    return labels


def build_pooled_daily_frame(
    package_dir: Path, macro_csv: Path | None, *, horizon: int = 10
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    master, groups, benchmark = load_package(package_dir)
    codes = cohort_codes(master, "core24")
    universe_sha256, training_history_sha256 = gics_input_fingerprints(
        master, groups, benchmark, codes
    )
    prices = build_cohort_prices(groups, benchmark, codes)
    session_fingerprints = prospective_session_fingerprints(prices, codes)
    names = master.set_index("industry_group_code")["industry_group_name"].to_dict()
    calendar = prices["SPX"].dropna().index
    parts: list[pd.DataFrame] = []
    missing_sessions: dict[str, int] = {}
    for code in codes:
        relative = prices[code].div(prices["SPX"]).reindex(calendar)
        missing_sessions[code] = int(relative.isna().sum())
        labels = _labels_with_calendar_gaps(relative, horizon=horizon)
        part = _price_features(relative).join(labels)
        part.insert(0, "date", part.index)
        part.insert(1, "sector", code)
        part.insert(2, "sector_name", str(names[code]))
        part.insert(3, "relative_price", relative)
        part.insert(4, "_position", np.arange(len(part)))
        parts.append(part.reset_index(drop=True))
    pooled = pd.concat(parts, ignore_index=True).sort_values(["date", "sector"])

    audit = {
        "gics_groups": len(codes),
        "gics_codes": codes,
        "gics_first_date": str(pooled["date"].min().date()),
        "gics_last_date": str(pooled["date"].max().date()),
        "gics_universe_sha256": universe_sha256,
        "gics_training_history_sha256": training_history_sha256,
        "gics_training_history_cutoff_exclusive": str(
            PROSPECTIVE_EARLIEST_INPUT_DATE.date()
        ),
        "missing_relative_sessions_by_group": missing_sessions,
        "raw_fill_or_interpolation": False,
        "macro_same_day_use": False,
    }
    if macro_csv is None:
        pooled["macro_asof_date"] = pd.NaT
        audit.update(
            {
                "macro_first_date": None,
                "macro_last_date": None,
                "macro_rows": 0,
                "macro_status": "NOT_LOADED_PRICE_ONLY",
            }
        )
    else:
        history = load_macro_history(macro_csv)
        macro, _, _ = build_macro_feature_sets(history)
        missing = sorted(set(MACRO_FEATURES) - set(macro.columns))
        if missing:
            raise ValueError("macro features missing: " + ", ".join(missing))
        aligned = merge_macro_next_session(
            pd.DatetimeIndex(pooled["date"].drop_duplicates().sort_values()),
            macro,
            MACRO_FEATURES,
            asof_column="macro_asof_date",
        ).reset_index()
        pooled = pooled.merge(aligned, on="date", how="left", validate="many_to_one")
        audit.update(
            {
                "macro_first_date": str(history["date"].min().date()),
                "macro_last_date": str(history["date"].max().date()),
                "macro_rows": int(len(history)),
                "macro_status": "LOADED",
            }
        )
    return pooled.reset_index(drop=True), audit, session_fingerprints


def effect_code(
    sectors: pd.Series, levels: Sequence[str] | None = None
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    values = sectors.astype("string")
    resolved = tuple(sorted(values.dropna().unique())) if levels is None else tuple(levels)
    if (
        len(resolved) < 2
        or len(set(resolved)) != len(resolved)
        or values.isna().any()
        or not set(values).issubset(resolved)
    ):
        raise ValueError("at least two known sector levels are required")
    matrix = pd.DataFrame(index=sectors.index)
    center = 1.0 / len(resolved)
    for sector in resolved:
        matrix[f"sector_effect__{sector}"] = values.eq(sector).astype(float) - center
    return matrix, resolved


def effect_deviations(
    coefficients: Sequence[float], levels: Sequence[str]
) -> dict[str, float]:
    resolved = tuple(levels)
    values = np.asarray(coefficients, dtype=float)
    if len(resolved) < 2 or len(values) != len(resolved):
        raise ValueError("effect coefficients must contain one value per level")
    values = values - values.mean()
    return dict(zip(resolved, values.tolist()))


def _direction_frame(
    frame: pd.DataFrame, direction: str, *, target_mode: str = "CONFIRMATION"
) -> pd.DataFrame:
    if direction not in DIRECTIONS:
        raise ValueError(f"unknown direction: {direction}")
    if target_mode not in TARGET_MODES:
        raise ValueError(f"unknown target mode: {target_mode}")
    prefix = "top" if direction == "ROTATION_OUT" else "bottom"
    result = frame.copy()
    result["at_risk"] = result[f"{prefix}_at_risk"].astype(bool)
    result["target"] = result[f"{prefix}_target"]
    result["label_end_position"] = result[f"{prefix}_label_end_position"]
    result["label_end_date"] = result[f"{prefix}_label_end_date"]
    result["target_event_pivot_position"] = np.nan
    result["target_event_pivot_date"] = pd.NaT
    result["target_event_confirmation_position"] = np.nan
    result["target_event_confirmation_date"] = pd.NaT
    event_name = "TOP_END" if direction == "ROTATION_OUT" else "BOTTOM_END"
    for sector, group in result.groupby("sector", sort=False):
        events = group.loc[
            group["event"].eq(event_name),
            ["pivot_position", "pivot_date", "confirmation_position", "confirmation_date"],
        ].sort_values("confirmation_position")
        if events.empty:
            continue
        confirmations = events["confirmation_position"].to_numpy(dtype=float)
        positives = group.loc[group["target"].eq(1.0)]
        for row_index, row in positives.iterrows():
            match = int(np.searchsorted(confirmations, row["_position"], side="right"))
            if match >= len(events) or confirmations[match] > row["label_end_position"]:
                continue
            event = events.iloc[match]
            result.loc[row_index, "target_event_pivot_position"] = event["pivot_position"]
            result.loc[row_index, "target_event_pivot_date"] = event["pivot_date"]
            result.loc[row_index, "target_event_confirmation_position"] = event[
                "confirmation_position"
            ]
            result.loc[row_index, "target_event_confirmation_date"] = event[
                "confirmation_date"
            ]
    if target_mode == "PRE_PIVOT":
        known = result["target"].notna()
        result.loc[known, "target"] = (
            result.loc[known, "target"].eq(1.0)
            & result.loc[known, "target_event_pivot_position"].notna()
            & (
                result.loc[known, "_position"]
                < result.loc[known, "target_event_pivot_position"]
            )
        ).astype(float)
    return result


def purged_training_mask(frame: pd.DataFrame, prediction_date: pd.Timestamp) -> pd.Series:
    prediction_date = pd.Timestamp(prediction_date)
    return (
        frame["at_risk"].astype(bool)
        & frame["target"].notna()
        & frame["label_end_date"].notna()
        & (frame["date"] < prediction_date)
        & (frame["label_end_date"] < prediction_date)
    )


def _design(
    frame: pd.DataFrame,
    standardized: np.ndarray,
    levels: Sequence[str],
    model_name: str,
    continuous_features: Sequence[str] = CONTINUOUS_FEATURES,
) -> pd.DataFrame:
    continuous = pd.DataFrame(
        standardized, index=frame.index, columns=continuous_features
    )
    effects, _ = effect_code(frame["sector"], levels)
    result = continuous.loc[:, PRICE_FEATURES].join(effects)
    if model_name != "POOLED_PRICE":
        result = result.join(continuous.loc[:, MACRO_FEATURES])
    if model_name == "POOLED_PARTIAL_MACRO":
        interactions = {
            f"{macro}__x__{effect}": continuous[macro] * effects[effect]
            for macro in MACRO_FEATURES
            for effect in effects
        }
        result = result.join(pd.DataFrame(interactions, index=frame.index))
    if model_name not in MODELS:
        raise ValueError(f"unknown model: {model_name}")
    return result


def _coefficient_rows(
    model: LogisticRegression,
    columns: Sequence[str],
    scaler: StandardScaler,
    *,
    direction: str,
    model_name: str,
    refit_date: pd.Timestamp,
    levels: Sequence[str],
    continuous_features: Sequence[str] = CONTINUOUS_FEATURES,
) -> list[dict[str, Any]]:
    coefficients = dict(zip(columns, model.coef_[0]))
    rows: list[dict[str, Any]] = []
    means = dict(zip(continuous_features, scaler.mean_))
    scales = dict(zip(continuous_features, scaler.scale_))
    intercept_effects = [coefficients[f"sector_effect__{level}"] for level in levels]
    for sector, deviation in effect_deviations(intercept_effects, levels).items():
        rows.append(
            {
                "direction": direction,
                "model": model_name,
                "refit_date": refit_date,
                "factor": "INTERCEPT",
                "sector": sector,
                "common": float(model.intercept_[0]),
                "deviation": deviation,
                "total": float(model.intercept_[0]) + deviation,
                "train_mean": np.nan,
                "train_scale": np.nan,
            }
        )
    for factor in PRICE_FEATURES:
        common = float(coefficients[factor])
        for sector in levels:
            rows.append(
                {
                    "direction": direction,
                    "model": model_name,
                    "refit_date": refit_date,
                    "factor": factor,
                    "sector": sector,
                    "common": common,
                    "deviation": 0.0,
                    "total": common,
                    "train_mean": float(means[factor]),
                    "train_scale": float(scales[factor]),
                }
            )
    if model_name == "POOLED_PRICE":
        return rows
    for macro in MACRO_FEATURES:
        common = float(coefficients[macro])
        if model_name == "POOLED_PARTIAL_MACRO":
            interaction = [
                coefficients[f"{macro}__x__sector_effect__{level}"]
                for level in levels
            ]
            deviations = effect_deviations(interaction, levels)
        else:
            deviations = {level: 0.0 for level in levels}
        for sector, deviation in deviations.items():
            rows.append(
                {
                    "direction": direction,
                    "model": model_name,
                    "refit_date": refit_date,
                    "factor": macro,
                    "sector": sector,
                    "common": common,
                    "deviation": deviation,
                    "total": common + deviation,
                    "train_mean": float(means[macro]),
                    "train_scale": float(scales[macro]),
                }
            )
    return rows


def walk_forward_direction(
    pooled: pd.DataFrame,
    direction: str,
    *,
    min_train_dates: int = 756,
    refit_every: int = 63,
    c_value: float = 0.1,
    target_mode: str = "CONFIRMATION",
    models: Sequence[str] = MODELS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if min_train_dates < 1 or refit_every < 1 or c_value <= 0:
        raise ValueError("positive walk-forward settings are required")
    frame = _direction_frame(pooled, direction, target_mode=target_mode)
    resolved_models = tuple(models)
    if not resolved_models or not set(resolved_models).issubset(MODELS):
        raise ValueError("models must contain known model names")
    continuous_features = (
        PRICE_FEATURES if resolved_models == ("POOLED_PRICE",) else CONTINUOUS_FEATURES
    )
    complete = frame.loc[:, continuous_features].notna().all(axis=1)
    scorable = frame["at_risk"] & complete
    eligible = scorable & frame["target"].notna()
    levels = tuple(sorted(frame["sector"].astype(str).unique()))
    dates = pd.DatetimeIndex(frame.loc[scorable, "date"].drop_duplicates().sort_values())
    predictions: list[pd.DataFrame] = []
    coefficient_rows: list[dict[str, Any]] = []
    skipped_history = 0
    skipped_one_class = 0
    fitted_blocks = 0

    for start in range(0, len(dates), refit_every):
        block_dates = dates[start : start + refit_every]
        prediction_date = pd.Timestamp(block_dates[0])
        train_mask = purged_training_mask(frame, prediction_date) & complete
        train = frame.loc[train_mask]
        if train["date"].nunique() < min_train_dates:
            skipped_history += 1
            continue
        if train["target"].nunique() < 2:
            skipped_one_class += 1
            continue
        test_mask = scorable & frame["date"].isin(block_dates)
        test = frame.loc[test_mask]
        if test.empty:
            continue
        scaler = StandardScaler().fit(train.loc[:, continuous_features])
        train_standard = scaler.transform(train.loc[:, continuous_features])
        test_standard = scaler.transform(test.loc[:, continuous_features])
        target = train["target"].astype(int)
        for model_name in resolved_models:
            train_design = _design(
                train, train_standard, levels, model_name, continuous_features
            )
            test_design = _design(
                test, test_standard, levels, model_name, continuous_features
            )
            model = LogisticRegression(
                C=c_value,
                class_weight="balanced",
                max_iter=2_000,
                random_state=42,
            ).fit(train_design, target)
            rows = test[
                [
                    "date",
                    "_position",
                    "sector",
                    "sector_name",
                    "state",
                    "pivot_position",
                    "pivot_date",
                    "confirmation_position",
                    "confirmation_date",
                    "barrier",
                    "cooldown_remaining",
                    "event",
                    "at_risk",
                    "target",
                    "label_end_position",
                    "label_end_date",
                    "target_event_pivot_position",
                    "target_event_pivot_date",
                    "target_event_confirmation_position",
                    "target_event_confirmation_date",
                    "macro_asof_date",
                ]
            ].copy()
            rows.insert(0, "direction", direction)
            rows.insert(1, "model", model_name)
            rows.insert(2, "target_mode", target_mode)
            rows["refit_date"] = prediction_date
            rows["training_start_date"] = train["date"].min()
            rows["training_end_date"] = train["date"].max()
            rows["latest_training_label_end_date"] = train["label_end_date"].max()
            rows["training_date_count"] = train["date"].nunique()
            rows["score"] = model.predict_proba(test_design)[:, 1]
            predictions.append(rows)
            coefficient_rows.extend(
                _coefficient_rows(
                    model,
                    train_design.columns,
                    scaler,
                    direction=direction,
                    model_name=model_name,
                    refit_date=prediction_date,
                    levels=levels,
                    continuous_features=continuous_features,
                )
            )
        fitted_blocks += 1

    prediction_frame = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    if not prediction_frame.empty:
        prediction_frame = prediction_frame.sort_values(
            ["model", "sector", "date"]
        ).reset_index(drop=True)
        prediction_frame["raw_score"] = prediction_frame["score"]
        prediction_frame["score"] = prediction_frame.groupby(
            ["model", "sector"], sort=False
        )["raw_score"].transform(
            lambda values: causal_trailing_percentile(
                values, window=756, min_history=60
            )
        )
        prediction_frame = prediction_frame.dropna(subset=["score"]).reset_index(drop=True)
    coefficient_frame = pd.DataFrame(coefficient_rows)
    status = "OK"
    if prediction_frame.empty:
        status = "INVALID_ONE_CLASS" if skipped_one_class else "INSUFFICIENT_HISTORY"
    walk = {
        "status": status,
        "eligible_rows": int(eligible.sum()),
        "eligible_dates": int(frame.loc[eligible, "date"].nunique()),
        "scorable_rows": int(scorable.sum()),
        "scorable_dates": int(frame.loc[scorable, "date"].nunique()),
        "eligible_positives": int(frame.loc[eligible, "target"].sum()),
        "eligible_negatives": int(eligible.sum() - frame.loc[eligible, "target"].sum()),
        "fitted_blocks": fitted_blocks,
        "skipped_insufficient_history_blocks": skipped_history,
        "skipped_one_class_blocks": skipped_one_class,
        "target_mode": target_mode,
    }
    return prediction_frame, coefficient_frame, walk


def _event_capture_analysis(
    rows: pd.DataFrame, *, horizon: int, cutoff: float
) -> tuple[pd.DataFrame, int, int]:
    keys = ["sector", "target_event_confirmation_position"]
    positive = rows.loc[rows["target_event_confirmation_position"].notna()]
    records: list[dict[str, Any]] = []
    for key, event_rows in positive.groupby(keys, sort=True):
        event_rows = event_rows.sort_values("_position")
        pivot = int(event_rows["target_event_pivot_position"].iloc[0])
        confirmation = int(key[1])
        records.append(
            {
                "sector": str(key[0]),
                "confirmation_position": confirmation,
                "confirmation_date": event_rows["target_event_confirmation_date"].iloc[0],
                "pivot_position": pivot,
                "event_hit": 0,
                "pre_pivot_hit": 0,
                "repeated_pre_pivot_hit": 0,
                "lead_to_confirmation_sessions": np.nan,
                "lead_to_pivot_sessions": np.nan,
            }
        )
    outcomes = pd.DataFrame.from_records(
        records,
        columns=[
            "sector",
            "confirmation_position",
            "confirmation_date",
            "pivot_position",
            "event_hit",
            "pre_pivot_hit",
            "repeated_pre_pivot_hit",
            "lead_to_confirmation_sessions",
            "lead_to_pivot_sessions",
        ],
    )
    alerts = rows.loc[rows["score"] >= cutoff].sort_values(["sector", "_position"])
    clusters = matched_clusters = 0
    for sector, sector_rows in alerts.groupby("sector"):
        sector_clusters: list[list[int]] = []
        for position in sector_rows["_position"].astype(int).unique():
            if not sector_clusters or position - sector_clusters[-1][-1] > horizon:
                sector_clusters.append([position])
            else:
                sector_clusters[-1].append(position)
        unmatched = outcomes.index[outcomes["sector"].eq(str(sector))].tolist()
        unmatched.sort(key=lambda index: outcomes.loc[index, "confirmation_position"])
        clusters += len(sector_clusters)
        for cluster in sector_clusters:
            match = next(
                (
                    index
                    for index in unmatched
                    if any(
                        1
                        <= int(outcomes.loc[index, "confirmation_position"]) - position
                        <= horizon
                        for position in cluster
                    )
                ),
                None,
            )
            if match is None:
                continue
            unmatched.remove(match)
            confirmation = int(outcomes.loc[match, "confirmation_position"])
            pivot = int(outcomes.loc[match, "pivot_position"])
            qualifying = [
                position for position in cluster if 1 <= confirmation - position <= horizon
            ]
            pre_pivot = [position for position in qualifying if position < pivot]
            outcomes.loc[match, "event_hit"] = 1
            outcomes.loc[match, "lead_to_confirmation_sessions"] = confirmation - min(qualifying)
            if pre_pivot:
                outcomes.loc[match, "pre_pivot_hit"] = 1
                outcomes.loc[match, "repeated_pre_pivot_hit"] = int(len(pre_pivot) >= 2)
                outcomes.loc[match, "lead_to_pivot_sessions"] = pivot - min(pre_pivot)
            matched_clusters += 1
    return outcomes, clusters, matched_clusters


def _event_capture_outcomes(
    rows: pd.DataFrame, *, horizon: int = 10, cutoff: float = ALERT_CUTOFF
) -> pd.DataFrame:
    outcomes, _, _ = _event_capture_analysis(rows, horizon=horizon, cutoff=cutoff)
    return outcomes


def _early_event_metrics(
    rows: pd.DataFrame,
    pooled: pd.DataFrame,
    direction: str,
    *,
    horizon: int,
    cutoff: float = ALERT_CUTOFF,
) -> dict[str, Any]:
    del pooled, direction
    outcomes, clusters, matched_clusters = _event_capture_analysis(
        rows, horizon=horizon, cutoff=cutoff
    )
    sector_years = 0.0
    for _, sector_rows in rows.groupby("sector"):
        if len(sector_rows) > 1:
            sector_years += max(
                (sector_rows["date"].max() - sector_rows["date"].min()).days / 365.25,
                1 / 252,
            )
    eligible = len(outcomes)
    hits = int(outcomes["event_hit"].sum()) if eligible else 0
    pre_pivot_hits = int(outcomes["pre_pivot_hit"].sum()) if eligible else 0
    repeated = int(outcomes["repeated_pre_pivot_hit"].sum()) if eligible else 0
    false_clusters = clusters - matched_clusters
    return {
        "alert_cutoff": cutoff,
        "eligible_confirmations": eligible,
        "event_hits": hits,
        "event_recall": hits / eligible if eligible else 0.0,
        "pre_pivot_hits": pre_pivot_hits,
        "pre_pivot_recall": pre_pivot_hits / eligible if eligible else 0.0,
        "repeated_pre_pivot_hits": repeated,
        "repeated_pre_pivot_recall": repeated / eligible if eligible else 0.0,
        "median_lead_to_pivot_sessions": (
            float(outcomes["lead_to_pivot_sessions"].median())
            if eligible and outcomes["lead_to_pivot_sessions"].notna().any()
            else None
        ),
        "median_lead_to_confirmation_sessions": (
            float(outcomes["lead_to_confirmation_sessions"].median())
            if eligible and outcomes["lead_to_confirmation_sessions"].notna().any()
            else None
        ),
        "alert_clusters": clusters,
        "matched_alert_clusters": matched_clusters,
        "false_alert_clusters": false_clusters,
        "false_alert_clusters_per_sector_year": (
            false_clusters / sector_years if sector_years else None
        ),
        "alert_day_rate": float((rows["score"] >= cutoff).mean()) if len(rows) else 0.0,
    }


def _model_metrics(
    rows: pd.DataFrame,
    pooled: pd.DataFrame,
    direction: str,
    *,
    horizon: int,
    cutoff: float = ALERT_CUTOFF,
) -> dict[str, Any]:
    target = rows["target"].astype(int)
    events = _early_event_metrics(rows, pooled, direction, horizon=horizon, cutoff=cutoff)
    if target.nunique() < 2:
        return {
            "status": "INVALID_ONE_CLASS",
            "n": int(len(rows)),
            "positives": int(target.sum()),
            "negatives": int(len(target) - target.sum()),
            "roc_auc": None,
            "pr_auc": None,
            "per_sector_auc": [],
            "early_event_metrics": events,
        }
    per_sector = []
    for (sector, name), group in rows.groupby(["sector", "sector_name"], sort=True):
        sector_target = group["target"].astype(int)
        if sector_target.nunique() != 2:
            continue
        per_sector.append(
            {
                "sector": str(sector),
                "sector_name": str(name),
                "n": int(len(group)),
                "positives": int(sector_target.sum()),
                "roc_auc": float(roc_auc_score(sector_target, group["score"])),
                "pr_auc": float(average_precision_score(sector_target, group["score"])),
            }
        )
    return {
        "status": "OK",
        "n": int(len(rows)),
        "positives": int(target.sum()),
        "negatives": int(len(target) - target.sum()),
        "roc_auc": float(roc_auc_score(target, rows["score"])),
        "pr_auc": float(average_precision_score(target, rows["score"])),
        "per_sector_auc": per_sector,
        "early_event_metrics": events,
    }


def _empty_model_metrics(
    status: str, *, cutoff: float = ALERT_CUTOFF
) -> dict[str, Any]:
    return {
        "status": status,
        "n": 0,
        "positives": 0,
        "negatives": 0,
        "roc_auc": None,
        "pr_auc": None,
        "per_sector_auc": [],
        "early_event_metrics": {
            "alert_cutoff": cutoff,
            "eligible_confirmations": 0,
            "event_hits": 0,
            "event_recall": 0.0,
            "pre_pivot_hits": 0,
            "pre_pivot_recall": 0.0,
            "repeated_pre_pivot_hits": 0,
            "repeated_pre_pivot_recall": 0.0,
            "median_lead_to_pivot_sessions": None,
            "median_lead_to_confirmation_sessions": None,
            "alert_clusters": 0,
            "matched_alert_clusters": 0,
            "false_alert_clusters": 0,
            "false_alert_clusters_per_sector_year": None,
            "alert_day_rate": 0.0,
        },
    }


def paired_date_block_auc_delta_ci(
    target: pd.Series,
    candidate: pd.Series,
    baseline: pd.Series,
    *,
    samples: int = 500,
    block_dates: int = 40,
    random_state: int = 42,
    calendar_dates: Sequence[pd.Timestamp] | None = None,
) -> dict[str, float | int]:
    """Paired bootstrap that keeps every sector from a sampled date together."""

    aligned = pd.concat(
        {"target": target, "candidate": candidate, "baseline": baseline}, axis=1
    ).dropna()
    if not isinstance(aligned.index, pd.MultiIndex) or "date" not in aligned.index.names:
        raise ValueError("paired scores require a MultiIndex containing date")
    observed_dates = pd.DatetimeIndex(
        aligned.index.get_level_values("date").unique()
    ).sort_values()
    dates = (
        observed_dates
        if calendar_dates is None
        else pd.DatetimeIndex(calendar_dates).drop_duplicates().sort_values()
    )
    by_date = aligned.groupby(level="date")["target"]
    positive_dates = int(by_date.max().sum())
    negative_dates = int((by_date.min() == 0).sum())
    support = {
        "calendar_dates": int(len(dates)),
        "positive_dates": positive_dates,
        "negative_dates": negative_dates,
        "requested_bootstrap_samples": int(samples),
        "valid_bootstrap_samples": 0,
    }
    if len(aligned) < 2 or aligned["target"].nunique() < 2:
        return {
            "delta_auc": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            **support,
        }
    observed = float(
        roc_auc_score(aligned["target"], aligned["candidate"])
        - roc_auc_score(aligned["target"], aligned["baseline"])
    )
    block = max(int(block_dates), 20)
    if len(dates) < 2 * block or min(positive_dates, negative_dates) < 10:
        return {
            "delta_auc": observed,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            **support,
        }
    rng = np.random.default_rng(random_state)
    estimates: list[float] = []
    grouped = {date: group for date, group in aligned.groupby(level="date", sort=False)}
    for _ in range(samples):
        sampled_dates: list[pd.Timestamp] = []
        while len(sampled_dates) < len(dates):
            start = int(rng.integers(0, len(dates) - block + 1))
            sampled_dates.extend(dates[start : start + block].tolist())
        parts = [grouped[date] for date in sampled_dates[: len(dates)] if date in grouped]
        if not parts:
            continue
        sample = pd.concat(parts)
        if sample["target"].nunique() == 2:
            estimates.append(
                float(
                    roc_auc_score(sample["target"], sample["candidate"])
                    - roc_auc_score(sample["target"], sample["baseline"])
                )
            )
    support["valid_bootstrap_samples"] = len(estimates)
    enough = len(estimates) >= max(1, int(np.ceil(samples * 0.8)))
    low, high = np.quantile(estimates, [0.025, 0.975]) if enough else (np.nan, np.nan)
    return {
        "delta_auc": observed,
        "ci_low": float(low),
        "ci_high": float(high),
        **support,
    }


def paired_date_block_pre_pivot_recall_delta_ci(
    candidate_rows: pd.DataFrame,
    baseline_rows: pd.DataFrame,
    *,
    samples: int = 500,
    block_dates: int = 40,
    random_state: int = 42,
    calendar_dates: Sequence[pd.Timestamp] | None = None,
    horizon: int = 10,
    cutoff: float = ALERT_CUTOFF,
) -> dict[str, float | int]:
    candidate = _event_capture_outcomes(
        candidate_rows, horizon=horizon, cutoff=cutoff
    ).rename(
        columns={"pre_pivot_hit": "candidate_hit"}
    )
    baseline = _event_capture_outcomes(
        baseline_rows, horizon=horizon, cutoff=cutoff
    ).rename(
        columns={"pre_pivot_hit": "baseline_hit"}
    )
    keys = ["sector", "confirmation_position", "confirmation_date"]
    paired = candidate.loc[:, [*keys, "candidate_hit"]].merge(
        baseline.loc[:, [*keys, "baseline_hit"]],
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not paired["_merge"].eq("both").all():
        raise RuntimeError("candidate and baseline event samples are not paired")
    paired["delta"] = paired["candidate_hit"] - paired["baseline_hit"]
    event_dates = pd.DatetimeIndex(paired["confirmation_date"].drop_duplicates().sort_values())
    if calendar_dates is None:
        calendar = pd.DatetimeIndex(
            pd.concat(
                [candidate_rows["date"], baseline_rows["date"], paired["confirmation_date"]]
            )
            .drop_duplicates()
            .sort_values()
        )
    else:
        calendar = pd.DatetimeIndex(calendar_dates).union(event_dates).sort_values()
    discordant_dates = int(paired.loc[paired["delta"].ne(0), "confirmation_date"].nunique())
    support = {
        "paired_events": int(len(paired)),
        "event_dates": int(len(event_dates)),
        "discordant_event_dates": discordant_dates,
        "requested_early_capture_bootstrap_samples": int(samples),
        "valid_early_capture_bootstrap_samples": 0,
    }
    observed = float(paired["delta"].mean()) if len(paired) else float("nan")
    block = max(int(block_dates), 20)
    if (
        len(paired) < 30
        or len(calendar) < 2 * block
        or len(event_dates) < 20
        or discordant_dates < 10
    ):
        return {
            "delta_pre_pivot_recall": observed,
            "early_capture_ci_low": float("nan"),
            "early_capture_ci_high": float("nan"),
            **support,
        }
    grouped = {
        date: group for date, group in paired.groupby("confirmation_date", sort=False)
    }
    rng = np.random.default_rng(random_state)
    estimates: list[float] = []
    for _ in range(samples):
        sampled_dates: list[pd.Timestamp] = []
        while len(sampled_dates) < len(calendar):
            start = int(rng.integers(0, len(calendar) - block + 1))
            sampled_dates.extend(calendar[start : start + block].tolist())
        parts = [grouped[date] for date in sampled_dates[: len(calendar)] if date in grouped]
        if parts:
            estimates.append(float(pd.concat(parts, ignore_index=True)["delta"].mean()))
    support["valid_early_capture_bootstrap_samples"] = len(estimates)
    enough = len(estimates) >= max(1, int(np.ceil(samples * 0.8)))
    low, high = np.quantile(estimates, [0.025, 0.975]) if enough else (np.nan, np.nan)
    return {
        "delta_pre_pivot_recall": observed,
        "early_capture_ci_low": float(low),
        "early_capture_ci_high": float(high),
        **support,
    }


def _paired_uplifts(
    predictions: pd.DataFrame,
    *,
    bootstrap_samples: int,
    calendar_dates: Sequence[pd.Timestamp],
    horizon: int,
    cutoff: float = ALERT_CUTOFF,
    models: Sequence[str] = MODELS,
) -> dict[str, Any]:
    if predictions.empty:
        return {}
    ordered = predictions.sort_values(["date", "sector", "model"])
    scores = ordered.pivot(index=["date", "sector"], columns="model", values="score")
    targets = (
        ordered.drop_duplicates(["date", "sector"])
        .set_index(["date", "sector"])["target"]
        .reindex(scores.index)
        .astype(int)
    )
    output: dict[str, Any] = {}
    baseline_rows = ordered.loc[ordered["model"].eq("POOLED_PRICE")]
    for candidate in tuple(models)[1:]:
        paired = scores[["POOLED_PRICE", candidate]].dropna()
        target = targets.reindex(paired.index)
        result = paired_date_block_auc_delta_ci(
            target,
            paired[candidate],
            paired["POOLED_PRICE"],
            samples=bootstrap_samples,
            calendar_dates=calendar_dates,
        )
        early_capture = paired_date_block_pre_pivot_recall_delta_ci(
            ordered.loc[ordered["model"].eq(candidate)],
            baseline_rows,
            samples=bootstrap_samples,
            calendar_dates=calendar_dates,
            horizon=horizon,
            cutoff=cutoff,
        )
        output[candidate] = {
            "comparison": f"{candidate}-POOLED_PRICE",
            "primary_metric": "paired pre-pivot event recall uplift",
            "secondary_metric": "paired early-turn target ROC AUC uplift",
            "bootstrap": "40 consecutive calendar dates; all sectors retained per date",
            **result,
            **early_capture,
            "auc_statistical_pass": bool(
                np.isfinite(result["ci_low"]) and result["ci_low"] > 0
            ),
            "statistical_pass": bool(
                np.isfinite(early_capture["early_capture_ci_low"])
                and early_capture["early_capture_ci_low"] > 0
            ),
        }
    return output


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(value))
    return value


def append_prospective_signals(
    path: Path,
    latest: pd.DataFrame,
    *,
    run_manifest_sha256: str,
    input_session_sha256: str,
) -> int:
    if latest.empty:
        raise ValueError("latest signals are empty")
    if len(run_manifest_sha256) != 64 or any(
        character not in "0123456789ABCDEF" for character in run_manifest_sha256.upper()
    ):
        raise ValueError("run_manifest_sha256 must be a SHA-256 hex digest")
    if len(input_session_sha256) != 64 or any(
        character not in "0123456789ABCDEF" for character in input_session_sha256.upper()
    ):
        raise ValueError("input_session_sha256 must be a SHA-256 hex digest")
    rows = latest.copy()
    input_dates = pd.to_datetime(rows["date"], errors="raise").dt.normalize()
    if input_dates.min() < PROSPECTIVE_EARLIEST_INPUT_DATE:
        raise ValueError(
            "prospective log cannot backfill inputs before "
            f"{PROSPECTIVE_EARLIEST_INPUT_DATE.date()}"
        )
    rows.insert(0, "captured_at_utc", datetime.now(timezone.utc).isoformat())
    rows.insert(0, "model_version", FROZEN_MODEL_VERSION)
    rows.insert(1, "run_manifest_sha256", run_manifest_sha256.upper())
    rows.insert(2, "input_session_sha256", input_session_sha256.upper())
    keys = ["model_version", "date", "direction", "sector"]
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=rows.columns)
    rows["date"] = pd.to_datetime(rows["date"]).dt.strftime("%Y-%m-%d")
    for key in ("model_version", "direction", "sector"):
        rows[key] = rows[key].astype(str)
    if not existing.empty:
        if "run_manifest_sha256" not in existing:
            raise RuntimeError("prospective log is missing run manifest provenance")
        existing["date"] = pd.to_datetime(existing["date"]).dt.strftime("%Y-%m-%d")
        for key in ("model_version", "direction", "sector"):
            existing[key] = existing[key].astype(str)
        if existing.duplicated(keys).any():
            raise RuntimeError("prospective log contains duplicate keys")
        overlap = existing.merge(rows, on=keys, suffixes=("_old", "_new"))
        if not overlap.empty and (
            ~np.isclose(overlap["score_old"], overlap["score_new"], equal_nan=True)
            | ~np.isclose(overlap["raw_score_old"], overlap["raw_score_new"], equal_nan=True)
            | overlap["run_manifest_sha256_old"].ne(overlap["run_manifest_sha256_new"])
            | overlap["input_session_sha256_old"].ne(
                overlap["input_session_sha256_new"]
            )
        ).any():
            raise RuntimeError("immutable prospective signal conflict; bump model version")
        rows = rows.merge(existing.loc[:, keys], on=keys, how="left", indicator=True)
        rows = rows.loc[rows["_merge"].eq("left_only")].drop(columns="_merge")
    combined = pd.concat([existing, rows], ignore_index=True).sort_values(keys)
    combined.to_csv(path, index=False)
    return int(len(rows))


def build_prospective_outcomes(path: Path, pooled: pd.DataFrame) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    signals = pd.read_csv(path)
    keys = ["model_version", "date", "direction", "sector"]
    required = {
        *keys,
        "run_manifest_sha256",
        "input_session_sha256",
        "captured_at_utc",
        "raw_score",
        "score",
    }
    missing = sorted(required - set(signals.columns))
    if missing:
        raise ValueError("prospective log missing columns: " + ", ".join(missing))
    signals["date"] = pd.to_datetime(signals["date"], errors="raise").dt.normalize()
    for column in ("model_version", "direction", "sector"):
        signals[column] = signals[column].astype(str)
    if signals.duplicated(keys).any():
        raise RuntimeError("prospective log contains duplicate keys")
    signals = signals.drop(columns=["target", "label_status"], errors="ignore")
    if "signal_level" not in signals:
        signals["signal_level"] = shadow_signal_level(pd.to_numeric(signals["score"]))
    signals["alert"] = signals["signal_level"].ne("NONE")

    parts: list[pd.DataFrame] = []
    label_columns = [
        "date",
        "sector",
        "_position",
        "target",
        "label_end_date",
        "target_event_pivot_position",
        "target_event_pivot_date",
        "target_event_confirmation_position",
        "target_event_confirmation_date",
    ]
    for direction in DIRECTIONS:
        selected = signals.loc[signals["direction"].eq(direction)]
        if selected.empty:
            continue
        labels = _direction_frame(pooled, direction, target_mode="PRE_PIVOT").loc[
            :, label_columns
        ]
        labels["sector"] = labels["sector"].astype(str)
        if labels.duplicated(["date", "sector"]).any():
            raise RuntimeError(f"{direction}: duplicate outcome date/sector keys")
        parts.append(
            selected.merge(labels, on=["date", "sector"], how="left", validate="many_to_one")
        )
    if not parts:
        return pd.DataFrame()
    outcomes = pd.concat(parts, ignore_index=True)
    outcomes["outcome_status"] = np.select(
        [outcomes["_position"].isna(), outcomes["target"].isna()],
        ["INPUT_DATE_NOT_FOUND", "PENDING"],
        default="MATURED",
    )
    outcomes["pre_pivot_lead_sessions"] = np.where(
        outcomes["target"].eq(1.0),
        outcomes["target_event_pivot_position"] - outcomes["_position"],
        np.nan,
    )
    return outcomes.sort_values(keys).reset_index(drop=True)


def summarize_prospective_outcomes(outcomes: pd.DataFrame) -> dict[str, Any]:
    if outcomes.empty:
        return {"status": "NO_LOG", "rows": 0}
    matured = outcomes["outcome_status"].eq("MATURED")
    positive = matured & outcomes["target"].eq(1.0)
    events = outcomes.loc[
        positive & outcomes["target_event_confirmation_date"].notna(),
        ["direction", "sector", "target_event_confirmation_date"],
    ].drop_duplicates()
    direction_metrics: dict[str, Any] = {}
    enough_events = True
    for direction in DIRECTIONS:
        rows = outcomes.loc[
            outcomes["direction"].eq(direction)
            & outcomes["outcome_status"].eq("MATURED")
        ]
        thresholds: dict[str, Any] = {}
        for name, cutoff in (
            ("WATCH", PRE_PIVOT_ALERT_CUTOFF),
            ("PRIORITY", PRE_PIVOT_PRIORITY_CUTOFF),
        ):
            metrics = _early_event_metrics(rows, pooled=pd.DataFrame(), direction=direction, horizon=10, cutoff=cutoff)
            low, high = _wilson_interval(
                metrics["pre_pivot_hits"], metrics["eligible_confirmations"]
            )
            false_rate = metrics["false_alert_clusters_per_sector_year"]
            gates = {
                "at_least_20_independent_events": metrics["eligible_confirmations"] >= 20,
                "pre_pivot_recall_ci_low_above_5pct": low is not None and low > 0.05,
                "median_lead_at_least_3_sessions": (
                    metrics["median_lead_to_pivot_sessions"] is not None
                    and metrics["median_lead_to_pivot_sessions"] >= 3
                ),
                "false_clusters_at_most_2_per_sector_year": (
                    false_rate is not None and false_rate <= 2.0
                ),
                "alert_day_rate_at_most_20pct": metrics["alert_day_rate"] <= 0.20,
            }
            thresholds[name] = {
                **metrics,
                "pre_pivot_recall_ci_95": [low, high],
                "practical_gate_pass": all(gates.values()),
                "practical_gates": gates,
            }
        direction_metrics[direction] = thresholds
        enough_events &= thresholds["WATCH"]["eligible_confirmations"] >= 20
    return {
        "status": "REVIEW_READY" if enough_events else "COLLECTING_PROSPECTIVE_OUTCOMES",
        "rows": int(len(outcomes)),
        "captured_dates": int(outcomes["date"].nunique()),
        "matured_rows": int(matured.sum()),
        "pending_rows": int(outcomes["outcome_status"].eq("PENDING").sum()),
        "input_date_not_found_rows": int(
            outcomes["outcome_status"].eq("INPUT_DATE_NOT_FOUND").sum()
        ),
        "matured_positive_rows": int(positive.sum()),
        "independent_confirmed_events": int(len(events)),
        "minimum_events_before_review": 20,
        "pre_registered_review_floor": "95% CI lower bound of pre-pivot recall > 5%",
        "directions": direction_metrics,
        "automatic_promotion": False,
    }


def _wilson_interval(successes: int, trials: int) -> tuple[float | None, float | None]:
    if trials <= 0 or successes < 0 or successes > trials:
        return None, None
    z = 1.959963984540054
    rate = successes / trials
    denominator = 1 + z * z / trials
    center = (rate + z * z / (2 * trials)) / denominator
    half = z * np.sqrt(rate * (1 - rate) / trials + z * z / (4 * trials * trials)) / denominator
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def shadow_signal_level(scores: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [scores >= PRE_PIVOT_PRIORITY_CUTOFF, scores >= PRE_PIVOT_ALERT_CUTOFF],
            ["PRIORITY", "WATCH"],
            default="NONE",
        ),
        index=scores.index,
        dtype="string",
    )


def markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Pooled GICS Trend-Turn SHADOW Experiment",
        "",
        f"- 판정: **{summary['verdict']}**",
        f"- PIT: **{summary['pit_status']}**",
        f"- 운영 승격: **{summary['production_promotion']}**",
        f"- 목표: GICS Core24 상대가격의 고점·저점 종료를 다음 {summary['target']['horizon_sessions']}거래일 안에 조기 포착",
        f"- 고정 평가 시작일: **{summary['evaluation']['start_date']}**",
    ]
    prospective = summary.get("prospective_evaluation")
    if prospective:
        lines.extend(
            [
                f"- Prospective: **{prospective['status']}** — "
                f"성숙 {prospective.get('matured_rows', 0)}행 / "
                f"독립 확인 이벤트 {prospective.get('independent_confirmed_events', 0)}건",
            ]
        )
    lines.extend(
        [
        "",
        "| 방향 | 모델 | 상태 | OOS 표본 | 양성 | ROC AUC | PR AUC | 이벤트 재현 | Pivot 전 재현 | 반복 사전경고 | Pivot 선행 | False/sector-yr |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for direction, result in summary["directions"].items():
        for model, metric in result["models"].items():
            auc = "—" if metric["roc_auc"] is None else f"{metric['roc_auc']:.3f}"
            pr_auc = "—" if metric["pr_auc"] is None else f"{metric['pr_auc']:.3f}"
            event = metric["early_event_metrics"]
            pivot_lead = (
                "—"
                if event["median_lead_to_pivot_sessions"] is None
                else f"{event['median_lead_to_pivot_sessions']:.1f}"
            )
            false_rate = (
                "—"
                if event["false_alert_clusters_per_sector_year"] is None
                else f"{event['false_alert_clusters_per_sector_year']:.2f}"
            )
            lines.append(
                f"| {direction} | {model} | {metric['status']} | {metric['n']} | "
                f"{metric['positives']} | {auc} | {pr_auc} | {event['event_recall']:.1%} | "
                f"{event['pre_pivot_recall']:.1%} | {event['repeated_pre_pivot_recall']:.1%} | "
                f"{pivot_lead} | {false_rate} |"
            )
        lines.extend(
            [
                "",
                "| 방향 | 비교 | Δ Pivot 전 재현 (95% CI) | Δ ROC AUC (95% CI) | 통계 판정 | 실용 후보 |",
                "|---|---|---:|---:|---|---|",
            ]
        )
        for model, uplift in result["paired_early_turn_uplift"].items():
            delta = uplift["delta_auc"]
            auc_interval = "—"
            if delta is not None and uplift["ci_low"] is not None and uplift["ci_high"] is not None:
                auc_interval = f"{delta:.3f} [{uplift['ci_low']:.3f}, {uplift['ci_high']:.3f}]"
            capture_delta = uplift["delta_pre_pivot_recall"]
            capture_interval = "—"
            if (
                capture_delta is not None
                and uplift["early_capture_ci_low"] is not None
                and uplift["early_capture_ci_high"] is not None
            ):
                capture_interval = (
                    f"{capture_delta:.1%} "
                    f"[{uplift['early_capture_ci_low']:.1%}, {uplift['early_capture_ci_high']:.1%}]"
                )
            lines.append(
                f"| {direction} | {uplift['comparison']} | {capture_interval} | {auc_interval} | "
                f"{'PASS' if uplift['statistical_pass'] else 'NO'} | "
                f"{'YES' if uplift['practical_candidate_if_ci_includes_zero'] else 'NO'} |"
            )
    lines.extend(["", "## 방법론 경계", ""])
    if summary["models"]["run_mode"] == "PRICE_ONLY":
        lines.append("- 동결된 가격 전용 후보만 계산하며 매크로 데이터는 불러오지 않습니다.")
    else:
        lines.extend(
            [
                "- 세 모델은 동일한 완전관측 행에서 비교합니다.",
                "- Bloomberg d일 값은 d일에 쓰지 않고 다음 시장 세션부터 최대 7일만 사용합니다.",
                f"- 섹터별 매크로 편차는 합이 0인 effect coding이며 L2 C={summary['models']['l2_c']}을 사용합니다.",
            ]
        )
    lines.extend(
        [
            "- 고점·저점 라벨은 shifted 60일 변동성으로 만든 대칭 방향전환 장벽을 사용합니다.",
            "- 학습 라벨 종료일은 예측 블록 시작일보다 반드시 이전이어야 합니다.",
            "- 당시 공개 vintage와 역사적 GICS 분류가 검증되지 않아 연구용 SHADOW 외 사용을 금지합니다.",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    package_dir = args.gics_package_dir.resolve()
    price_only = bool(args.price_only)
    macro_csv = None if price_only else args.macro_csv.resolve()
    output_dir = args.output_dir.resolve()
    active_models = ("POOLED_PRICE",) if price_only else MODELS
    evaluation_start = EVALUATION_START
    alert_cutoff = (
        PRE_PIVOT_ALERT_CUTOFF if args.target_mode == "PRE_PIVOT" else ALERT_CUTOFF
    )
    input_dirs = (package_dir,) if macro_csv is None else (package_dir, macro_csv.parent)
    ensure_output_boundary(output_dir, *input_dirs)
    pooled, audit, session_fingerprints = build_pooled_daily_frame(
        package_dir, macro_csv, horizon=args.horizon
    )
    latest_input_date = pd.Timestamp(pooled["date"].max())
    frozen_manifest: dict[str, Any] | None = None
    if args.append_prospective:
        if args.target_mode != "PRE_PIVOT" or not price_only:
            raise ValueError("prospective logging requires PRE_PIVOT --price-only")
        if audit["gics_universe_sha256"] != EXPECTED_GICS_UNIVERSE_SHA256:
            raise RuntimeError("Core24 Bloomberg universe changed; review and bump model version")
        if (
            audit["gics_training_history_sha256"]
            != EXPECTED_GICS_TRAINING_HISTORY_SHA256
        ):
            raise RuntimeError("frozen GICS training history changed; review and bump model version")
        validate_expected_latest_date(latest_input_date, args.expected_latest_date)
        frozen_manifest = {
            "model_version": FROZEN_MODEL_VERSION,
            "code_bundle_sha256": code_bundle_sha256(),
            "gics_universe_sha256": audit["gics_universe_sha256"],
            "gics_training_history_sha256": audit["gics_training_history_sha256"],
            "training_history_cutoff_exclusive": str(
                PROSPECTIVE_EARLIEST_INPUT_DATE.date()
            ),
            "runtime": {
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": package_version("scikit-learn"),
            },
            "model_spec": {
                "target_mode": args.target_mode,
                "horizon_sessions": args.horizon,
                "minimum_training_dates": args.min_train_dates,
                "refit_every_sessions": args.refit_every,
                "logistic_l2_c": args.c_value,
                "features": list(PRICE_FEATURES),
                "watch_cutoff": PRE_PIVOT_ALERT_CUTOFF,
                "priority_cutoff": PRE_PIVOT_PRIORITY_CUTOFF,
                "universe": "GICS_CORE24_RELATIVE_TO_SPX",
            },
        }
    evaluation_calendar = pd.DatetimeIndex(
        pooled.loc[pooled["date"] >= evaluation_start, "date"].drop_duplicates().sort_values()
    )
    prediction_parts: list[pd.DataFrame] = []
    coefficient_parts: list[pd.DataFrame] = []
    directions: dict[str, Any] = {}
    for direction in DIRECTIONS:
        predictions, coefficients, walk = walk_forward_direction(
            pooled,
            direction,
            min_train_dates=args.min_train_dates,
            refit_every=args.refit_every,
            c_value=args.c_value,
            target_mode=args.target_mode,
            models=active_models,
        )
        if not predictions.empty:
            predictions["evaluation_sample"] = predictions["date"] >= evaluation_start
            prediction_parts.append(predictions)
        if not coefficients.empty:
            coefficient_parts.append(coefficients)
        models: dict[str, Any] = {}
        evaluation_predictions = (
            predictions.loc[
                predictions["evaluation_sample"] & predictions["target"].notna()
            ].copy()
            if not predictions.empty
            else pd.DataFrame()
        )
        for model in active_models:
            rows = (
                evaluation_predictions.loc[evaluation_predictions["model"] == model]
                if not evaluation_predictions.empty
                else pd.DataFrame()
            )
            models[model] = (
                _model_metrics(
                    rows,
                    pooled,
                    direction,
                    horizon=args.horizon,
                    cutoff=alert_cutoff,
                )
                if not rows.empty
                else _empty_model_metrics(
                    walk["status"] if predictions.empty else "NO_EVALUATION_ROWS",
                    cutoff=alert_cutoff,
                )
            )
        counts = {model: models[model]["n"] for model in active_models}
        if len(set(counts.values())) != 1:
            raise RuntimeError(f"{direction}: candidate rows are not paired: {counts}")
        baseline_auc = models["POOLED_PRICE"]["roc_auc"]
        for model in active_models[1:]:
            value = models[model]["roc_auc"]
            models[model]["auc_delta_vs_pooled_price"] = (
                value - baseline_auc if value is not None and baseline_auc is not None else None
            )
        uplifts = _paired_uplifts(
            evaluation_predictions,
            bootstrap_samples=args.bootstrap_samples,
            calendar_dates=evaluation_calendar,
            horizon=args.horizon,
            cutoff=alert_cutoff,
            models=active_models,
        )
        baseline_events = models["POOLED_PRICE"]["early_event_metrics"]
        for model in active_models[1:]:
            candidate_events = models[model]["early_event_metrics"]
            false_rate = candidate_events["false_alert_clusters_per_sector_year"]
            if model in uplifts:
                baseline_false = baseline_events["false_alert_clusters_per_sector_year"]
                gates = {
                    "positive_auc_delta": (uplifts[model]["delta_auc"] or 0) > 0,
                    "pre_pivot_recall_uplift_at_least_10pp": (
                        candidate_events["pre_pivot_recall"]
                        - baseline_events["pre_pivot_recall"]
                        >= 0.10
                    ),
                    "median_lead_to_pivot_at_least_3": (
                        candidate_events["median_lead_to_pivot_sessions"] or 0
                    )
                    >= 3,
                    "confirmation_event_recall_at_least_60pct": (
                        candidate_events["event_recall"] >= 0.60
                    ),
                    "has_repeated_pre_pivot_alerts": (
                        candidate_events["repeated_pre_pivot_hits"] > 0
                    ),
                    "false_clusters_at_most_2_per_sector_year": (
                        false_rate is not None and false_rate <= 2.0
                    ),
                    "false_cluster_increase_at_most_0_25": (
                        false_rate is not None
                        and baseline_false is not None
                        and false_rate - baseline_false <= 0.25
                    ),
                    "alert_day_rate_at_most_20pct": (
                        candidate_events["alert_day_rate"] <= 0.20
                    ),
                }
                uplifts[model]["practical_gates"] = gates
                ci_low = uplifts[model]["early_capture_ci_low"]
                ci_high = uplifts[model]["early_capture_ci_high"]
                ci_includes_zero = bool(
                    np.isfinite(ci_low)
                    and np.isfinite(ci_high)
                    and ci_low <= 0 <= ci_high
                )
                uplifts[model]["ci_includes_zero"] = ci_includes_zero
                uplifts[model]["practical_candidate_if_ci_includes_zero"] = bool(
                    ci_includes_zero and all(gates.values())
                )
        directions[direction] = {
            "walk_forward": walk,
            "models": models,
            "paired_early_turn_uplift": uplifts,
        }

    predictions = pd.concat(prediction_parts, ignore_index=True) if prediction_parts else pd.DataFrame()
    coefficients = pd.concat(coefficient_parts, ignore_index=True) if coefficient_parts else pd.DataFrame()
    latest_signals = (
        predictions.loc[
            predictions["date"].eq(latest_input_date)
            & predictions["model"].eq("POOLED_PRICE"),
            [
                "direction",
                "date",
                "sector",
                "sector_name",
                "state",
                "barrier",
                "raw_score",
                "score",
                "target",
                "refit_date",
                "training_end_date",
                "latest_training_label_end_date",
            ],
        ].copy()
        if not predictions.empty
        else pd.DataFrame()
    )
    if not latest_signals.empty:
        latest_signals["signal_level"] = shadow_signal_level(latest_signals["score"])
        latest_signals["alert"] = latest_signals["signal_level"].ne("NONE")
        latest_signals["label_status"] = np.where(
            latest_signals["target"].notna(), "MATURED", "PENDING"
        )
    events = pooled.loc[
        pooled["event"].ne(""),
        [
            "date",
            "_position",
            "sector",
            "sector_name",
            "event",
            "pivot_position",
            "pivot_date",
            "confirmation_position",
            "confirmation_date",
            "barrier",
        ],
    ].copy()
    summary = _json_ready(
        {
            "verdict": VERDICT,
            "research_only": True,
            "pit_status": PIT_STATUS,
            "production_promotion": PROMOTION_STATUS,
            "universe": "GICS_CORE24_RELATIVE_TO_SPX; NQ_NOT_POOLED",
            "evaluation": {
                "start_date": str(evaluation_start.date()),
                "locked_before_first_pooled_run": True,
            },
            "target": {
                "mode": args.target_mode,
                "ROTATION_OUT": (
                    "UP_ACTIVE row strictly before the pivot of a TOP_END confirmed in t+1..t+H"
                    if args.target_mode == "PRE_PIVOT"
                    else "UP_ACTIVE row followed by TOP_END in t+1..t+H"
                ),
                "ROTATION_IN": (
                    "DOWN_ACTIVE row strictly before the pivot of a BOTTOM_END confirmed in t+1..t+H"
                    if args.target_mode == "PRE_PIVOT"
                    else "DOWN_ACTIVE row followed by BOTTOM_END in t+1..t+H"
                ),
                "horizon_sessions": args.horizon,
                "last_horizon_rows": "CENSORED",
                "cooldown_sessions_after_confirmation": 5,
                "barrier": "clip(1.5 * shifted 60d log-return vol * sqrt(10), 3%, 10%)",
            },
            "models": {
                "candidates": list(active_models),
                "run_mode": "PRICE_ONLY" if price_only else "COMPARISON",
                "l2_c": args.c_value,
                "common_price_features": list(PRICE_FEATURES),
                "sector_interacted_macro_features": [] if price_only else list(MACRO_FEATURES),
                "effect_coding": "centered full one-hot; symmetric sum-to-zero deviations",
                "standardization": "continuous inputs fit on each training block only",
                "score_calibration": (
                    "causal 756-observation sector/model percentile; minimum 60 prior scores"
                ),
                "alert_cutoff": alert_cutoff,
            },
            "prospective_shadow_candidate": {
                "status": (
                    "HISTORICAL_DEVELOPMENT_FROZEN"
                    if args.target_mode == "PRE_PIVOT"
                    else "NOT_SELECTED"
                ),
                "model": "POOLED_PRICE" if args.target_mode == "PRE_PIVOT" else None,
                "model_version": (
                    FROZEN_MODEL_VERSION if args.target_mode == "PRE_PIVOT" else None
                ),
                "latest_input_date": str(latest_input_date.date()),
                "latest_scorable_rows": int(len(latest_signals)),
                "latest_alerts": (
                    int(latest_signals["alert"].sum()) if not latest_signals.empty else 0
                ),
                "latest_priority_alerts": (
                    int(latest_signals["signal_level"].eq("PRIORITY").sum())
                    if not latest_signals.empty
                    else 0
                ),
                "watch_cutoff": PRE_PIVOT_ALERT_CUTOFF,
                "priority_cutoff": PRE_PIVOT_PRIORITY_CUTOFF,
                "operation": "SHADOW_ONLY; NO POSITION CHANGE",
            },
            "walk_forward": {
                "expanding": True,
                "calendar_date_blocks": True,
                "refit_every_sessions": args.refit_every,
                "minimum_training_dates": args.min_train_dates,
                "purge": "label_end_date < prediction_block_start_date",
                "random_split": False,
                "paired_uplift_bootstrap_samples": args.bootstrap_samples,
            },
            "input_audit": audit,
            "directions": directions,
            "artifacts": [
                "summary.json",
                "REPORT.md",
                "predictions.csv",
                "coefficients.csv",
                "events.csv",
                "latest_signals.csv",
            ],
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    prospective_path = output_dir / "prospective_signals.csv"
    if args.append_prospective:
        if frozen_manifest is None:
            raise RuntimeError("prospective frozen manifest was not built")
        run_manifest_sha256 = ensure_frozen_manifest(
            output_dir / "frozen_manifest.json", frozen_manifest
        )
        ensure_prospective_input_sessions(
            output_dir / "prospective_input_sessions.csv", session_fingerprints
        )
        latest_session = session_fingerprints.loc[
            session_fingerprints["date"].eq(latest_input_date), "input_session_sha256"
        ]
        if len(latest_session) != 1:
            raise RuntimeError("latest GICS input session fingerprint is missing")
        append_prospective_signals(
            prospective_path,
            latest_signals,
            run_manifest_sha256=run_manifest_sha256,
            input_session_sha256=str(latest_session.iloc[0]),
        )
        summary["prospective_shadow_candidate"]["run_manifest_sha256"] = (
            run_manifest_sha256
        )
        summary["artifacts"].append("frozen_manifest.json")
        summary["artifacts"].append("prospective_input_sessions.csv")
    prospective_outcomes = build_prospective_outcomes(prospective_path, pooled)
    if not prospective_outcomes.empty:
        prospective_outcomes.to_csv(output_dir / "prospective_outcomes.csv", index=False)
        summary["prospective_evaluation"] = summarize_prospective_outcomes(
            prospective_outcomes
        )
        summary["artifacts"].append("prospective_outcomes.csv")
    if predictions.empty:
        predictions = pd.DataFrame(columns=PREDICTION_OUTPUT_COLUMNS)
    if coefficients.empty:
        coefficients = pd.DataFrame(columns=COEFFICIENT_OUTPUT_COLUMNS)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    coefficients.to_csv(output_dir / "coefficients.csv", index=False)
    events.to_csv(output_dir / "events.csv", index=False)
    latest_signals.to_csv(output_dir / "latest_signals.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    (output_dir / "REPORT.md").write_text(markdown_report(summary), encoding="utf-8")
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Research-only pooled GICS turning SHADOW")
    result.add_argument("--gics-package-dir", type=Path, required=True)
    result.add_argument("--macro-csv", type=Path)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--horizon", type=int, default=10)
    result.add_argument("--min-train-dates", type=int, default=756)
    result.add_argument("--refit-every", type=int, default=63)
    result.add_argument("--c-value", type=float, default=0.1)
    result.add_argument("--bootstrap-samples", type=int, default=500)
    result.add_argument(
        "--target-mode", choices=TARGET_MODES, default="CONFIRMATION"
    )
    result.add_argument("--append-prospective", action="store_true")
    result.add_argument("--price-only", action="store_true")
    result.add_argument("--expected-latest-date")
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if not args.price_only and args.macro_csv is None:
        raise ValueError("--macro-csv is required unless --price-only is used")
    if (
        args.horizon < 1
        or args.min_train_dates < 1
        or args.refit_every < 1
        or args.c_value <= 0
        or args.bootstrap_samples < 100
    ):
        raise ValueError("positive horizon, training, refit, and C settings are required")
    summary = run(args)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "verdict": summary["verdict"],
                "pit_status": summary["pit_status"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
