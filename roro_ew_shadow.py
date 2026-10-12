"""Immutable storage for the frozen equal-weight RORO benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
from contextlib import closing
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from roro_data import load_roro_panel
from roro_engine import (
    BLOCKS,
    INPUT_COLUMNS,
    MODEL_VERSION as RORO_MODEL_VERSION,
    OUTPUT_START,
    build_roro_scores,
)


SERIES_VERSION = f"{RORO_MODEL_VERSION}__immutable-shadow-r1"
TABLE = "roro_ew_shadow_observations"
SCHEMA_COLUMNS = (
    "series_version",
    "date",
    "input_cutoff",
    "roro_model_version",
    "provenance_hash",
    "payload_json",
    "created_at",
)
GLOBAL_PAYLOAD_COLUMNS = (
    "available_block_count",
    "completeness",
    "score_raw",
    "risk_score",
    "disagreement",
    "agreement",
    "data_status",
    "roro_regime",
    "trajectory_1d",
    "trajectory_5d",
    "trajectory_20d",
    "risk_drivers",
    "relief_drivers",
)
BLOCK_PAYLOAD_COLUMNS = tuple(
    column
    for block in BLOCKS
    for column in (
        f"{block}_score",
        f"{block}_regime",
        f"{block}_contribution",
        f"{block}_trajectory_1d",
        f"{block}_trajectory_5d",
        f"{block}_trajectory_20d",
    )
)
REQUIRED_SCORE_COLUMNS = (*GLOBAL_PAYLOAD_COLUMNS, *BLOCK_PAYLOAD_COLUMNS)
DERIVED_PAYLOAD_COLUMNS = (
    "available_blocks",
    "unavailable_blocks",
    "block_availability",
    "input_availability",
)
PAYLOAD_COLUMNS = (*REQUIRED_SCORE_COLUMNS, *DERIVED_PAYLOAD_COLUMNS)


class ImmutableSeriesConflict(RuntimeError):
    """Raised when an existing series/date observation would be rewritten."""


def _calendar_date(value: str | date | pd.Timestamp, name: str) -> str:
    try:
        timestamp = pd.Timestamp(value).normalize()
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a calendar date") from error
    if pd.isna(timestamp):
        raise ValueError(f"{name} must be a calendar date")
    return timestamp.date().isoformat()


def _json_value(value: Any) -> Any:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number):
            return None
        if not math.isfinite(number):
            raise ValueError("non-finite RORO-EW payload value")
        return number
    if isinstance(value, (pd.Timestamp, datetime, date, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, str):
        return value
    raise TypeError(f"unsupported RORO-EW payload value: {type(value).__name__}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _same_file(left: Path, right: Path) -> bool:
    if os.path.normcase(str(left.resolve())) == os.path.normcase(str(right.resolve())):
        return True
    return left.exists() and right.exists() and os.path.samefile(left, right)


def _provenance(day: str, payload_json: str) -> str:
    material = {
        "series_version": SERIES_VERSION,
        "date": day,
        "input_cutoff": day,
        "roro_model_version": RORO_MODEL_VERSION,
        "payload": json.loads(payload_json),
    }
    return hashlib.sha256(_canonical_json(material).encode("utf-8")).hexdigest()


def _payloads(scores: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(scores.index, pd.DatetimeIndex):
        raise ValueError("RORO-EW scores must use a DatetimeIndex")
    if scores.index.has_duplicates or not scores.index.is_monotonic_increasing:
        raise ValueError("RORO-EW score index must be unique and increasing")
    missing = [column for column in REQUIRED_SCORE_COLUMNS if column not in scores]
    if missing:
        raise ValueError(f"missing RORO-EW payload columns: {', '.join(missing)}")
    if (
        not isinstance(panel.index, pd.DatetimeIndex)
        or panel.index.has_duplicates
        or not panel.index.is_monotonic_increasing
    ):
        raise ValueError("RORO-EW source panel index must be unique and increasing")
    missing_inputs = [column for column in INPUT_COLUMNS if column not in panel]
    if missing_inputs:
        raise ValueError(f"missing RORO-EW inputs: {', '.join(missing_inputs)}")
    if not scores.index.isin(panel.index).all():
        raise ValueError("RORO-EW score dates must exist in the source panel")

    records: list[dict[str, Any]] = []
    for timestamp, row in scores.iterrows():
        block_availability = {
            block: bool(pd.notna(row[f"{block}_score"])) for block in BLOCKS
        }
        payload = {column: row[column] for column in REQUIRED_SCORE_COLUMNS}
        payload.update(
            {
                "available_blocks": tuple(
                    block for block, available in block_availability.items() if available
                ),
                "unavailable_blocks": tuple(
                    block
                    for block, available in block_availability.items()
                    if not available
                ),
                "block_availability": block_availability,
                "input_availability": {
                    column: bool(pd.notna(panel.at[timestamp, column]))
                    for column in INPUT_COLUMNS
                },
            }
        )
        records.append(payload)
    return pd.DataFrame(records, index=scores.index)


def _records(
    payloads: pd.DataFrame, as_of: str, *, latest_only: bool = False
) -> list[tuple[str, ...]]:
    missing = [column for column in PAYLOAD_COLUMNS if column not in payloads]
    if missing:
        raise ValueError(f"missing RORO-EW persisted fields: {', '.join(missing)}")
    cutoff = pd.Timestamp(as_of)
    selected = payloads.loc[
        (payloads.index >= OUTPUT_START) & (payloads.index <= cutoff),
        list(PAYLOAD_COLUMNS),
    ]
    if latest_only:
        selected = selected.tail(1)
    records: list[tuple[str, ...]] = []
    for timestamp, row in selected.iterrows():
        day = _calendar_date(timestamp, "date")
        payload_json = _canonical_json(row.to_dict())
        records.append(
            (
                SERIES_VERSION,
                day,
                day,
                RORO_MODEL_VERSION,
                _provenance(day, payload_json),
                payload_json,
            )
        )
    return records


def _initialize(connection: sqlite3.Connection) -> None:
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    }
    if tables - {TABLE}:
        raise ValueError("RORO-EW target must contain only its aggregate table")
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE}(
            series_version TEXT NOT NULL,
            date TEXT NOT NULL,
            input_cutoff TEXT NOT NULL,
            roro_model_version TEXT NOT NULL,
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
        f"BEFORE UPDATE ON {TABLE} BEGIN SELECT RAISE(ABORT,'immutable RORO-EW series'); END"
    )
    connection.execute(
        f"CREATE TRIGGER IF NOT EXISTS {TABLE}_no_delete "
        f"BEFORE DELETE ON {TABLE} BEGIN SELECT RAISE(ABORT,'immutable RORO-EW series'); END"
    )


def persist_ew_shadow(
    target: str | Path,
    payloads: pd.DataFrame,
    *,
    as_of: str | date | pd.Timestamp,
    latest_only: bool = False,
) -> dict[str, Any]:
    """Insert aggregate observations atomically; identical rows are idempotent."""
    cutoff = _calendar_date(as_of, "as_of")
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = _records(payloads, cutoff, latest_only=latest_only)
    inserted = idempotent = 0
    with closing(sqlite3.connect(path)) as connection:
        try:
            connection.execute("BEGIN IMMEDIATE")
            _initialize(connection)
            for record in records:
                existing = connection.execute(
                    f"""
                    SELECT series_version,date,input_cutoff,roro_model_version,
                           provenance_hash,payload_json
                    FROM {TABLE} WHERE series_version=? AND date=?
                    """,
                    record[:2],
                ).fetchone()
                if existing is not None:
                    if tuple(existing) != record:
                        raise ImmutableSeriesConflict(
                            f"immutable RORO-EW conflict: {record[0]} {record[1]}"
                        )
                    idempotent += 1
                    continue
                created_at = datetime.now(timezone.utc).isoformat(
                    timespec="microseconds"
                ).replace("+00:00", "Z")
                connection.execute(
                    f"INSERT INTO {TABLE} VALUES (?,?,?,?,?,?,?)",
                    (*record, created_at),
                )
                inserted += 1
            connection.commit()
        except Exception:
            connection.rollback()
            raise
    return {
        "series_version": SERIES_VERSION,
        "as_of": cutoff,
        "rows_seen": len(records),
        "inserted": inserted,
        "idempotent": idempotent,
    }


def build_ew_shadow_series(
    source: str | Path,
    target: str | Path,
    *,
    as_of: str | date | pd.Timestamp,
    latest_only: bool = False,
) -> dict[str, Any]:
    """Read the operating DB without mutation and build the isolated benchmark."""
    cutoff = _calendar_date(as_of, "as_of")
    source_path, target_path = Path(source), Path(target)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if _same_file(source_path, target_path):
        raise ValueError("target must be separate from the source database")
    if target_path.exists() and not target_path.is_file():
        raise ValueError("target must be a SQLite file path")

    panel = load_roro_panel(source_path, as_of=cutoff)
    scores = build_roro_scores(panel, output_start=OUTPUT_START)
    result = persist_ew_shadow(
        target_path,
        _payloads(scores, panel),
        as_of=cutoff,
        latest_only=latest_only,
    )
    result.update(
        {
            "latest_only": latest_only,
            "source_mode": panel.attrs.get("database_mode"),
            "target": str(target_path.resolve()),
        }
    )
    return result


def _read_only(database: str | Path) -> sqlite3.Connection:
    path = Path(database)
    if not path.is_file():
        raise FileNotFoundError(path)
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    return connection


def read_ew_shadow_history(
    target: str | Path,
    *,
    start: str | date | pd.Timestamp | None = None,
    end: str | date | pd.Timestamp | None = None,
) -> list[dict[str, Any]]:
    clauses = ["series_version=?"]
    parameters = [SERIES_VERSION]
    if start is not None:
        clauses.append("date>=?")
        parameters.append(_calendar_date(start, "start"))
    if end is not None:
        clauses.append("date<=?")
        parameters.append(_calendar_date(end, "end"))
    with closing(_read_only(target)) as connection:
        rows = connection.execute(
            f"SELECT {','.join(SCHEMA_COLUMNS)} FROM {TABLE} "
            f"WHERE {' AND '.join(clauses)} ORDER BY date",
            parameters,
        ).fetchall()
    history = []
    for row in rows:
        observation = dict(zip(SCHEMA_COLUMNS, row))
        payload_json = observation.pop("payload_json")
        payload = json.loads(payload_json)
        if set(payload) != set(PAYLOAD_COLUMNS):
            raise ValueError("stored RORO-EW payload violates the aggregate allowlist")
        if observation["input_cutoff"] != observation["date"]:
            raise ValueError("stored RORO-EW input cutoff does not match its date")
        if observation["roro_model_version"] != RORO_MODEL_VERSION:
            raise ValueError("stored RORO-EW model version is unexpected")
        if observation["provenance_hash"] != _provenance(
            observation["date"], payload_json
        ):
            raise ValueError("stored RORO-EW provenance hash mismatch")
        observation["payload"] = payload
        history.append(observation)
    return history


def latest_ew_shadow(target: str | Path) -> dict[str, Any] | None:
    history = read_ew_shadow_history(target)
    return history[-1] if history else None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="build the isolated RORO-EW series")
    build.add_argument("source")
    build.add_argument("target")
    build.add_argument("--as-of", required=True)
    build.add_argument("--latest-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "build":
        result = build_ew_shadow_series(
            args.source,
            args.target,
            as_of=args.as_of,
            latest_only=args.latest_only,
        )
        print(_canonical_json(result))


if __name__ == "__main__":
    main()
