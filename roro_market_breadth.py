"""Immutable direct-market breadth observations for the Market Lens."""

from __future__ import annotations

import csv
import argparse
import json
import math
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from contextlib import closing
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import pandas as pd


CONTRACT_VERSION = "external-breadth-v1-ndth-200dma"
SOURCE = "HISTORYOFMARKET_NDX_BREADTH_JSON"
CSV_SOURCE = "USER_SUPPLIED_NDTH_CSV"
ALLOWED_SOURCES = frozenset({SOURCE, CSV_SOURCE})
SYMBOL = "NDTH"
TABLE = "roro_market_breadth_observations"
SCHEMA_COLUMNS = (
    "contract_version", "source", "symbol", "observation_date", "available_at", "value_pct", "fetched_at",
)
KEY_COLUMNS = ("contract_version", "source", "symbol", "observation_date", "available_at")


class ImmutableBreadthObservationConflict(ValueError):
    """Raised when an existing breadth observation is changed."""


def _day(value: object, field: str) -> str:
    raw = str(value).strip()
    try:
        parsed = date.fromisoformat(raw)
    except ValueError as error:
        raise ValueError(f"{field} must be YYYY-MM-DD") from error
    if parsed.isoformat() != raw:
        raise ValueError(f"{field} must be YYYY-MM-DD")
    return raw


def _record(record: Mapping[str, object]) -> tuple[Any, ...]:
    missing = [key for key in SCHEMA_COLUMNS if key not in record]
    extra = [key for key in record if key not in SCHEMA_COLUMNS]
    if missing or extra:
        raise ValueError("breadth observation fields must exactly match the contract")
    if str(record["contract_version"]).strip() != CONTRACT_VERSION:
        raise ValueError(f"contract_version must be {CONTRACT_VERSION}")
    source = str(record["source"]).strip()
    if source not in ALLOWED_SOURCES:
        raise ValueError("source must be an approved NDTH source")
    if str(record["symbol"]).strip() != SYMBOL:
        raise ValueError(f"symbol must be {SYMBOL}")
    observation = _day(record["observation_date"], "observation_date")
    available = _day(record["available_at"], "available_at")
    fetched = _day(record["fetched_at"], "fetched_at")
    if available < observation:
        raise ValueError("available_at must be on or after observation_date")
    if fetched < available:
        raise ValueError("fetched_at must be on or after available_at")
    try:
        value = float(record["value_pct"])
    except (TypeError, ValueError) as error:
        raise ValueError("value_pct must be a finite percentage from 0 to 100") from error
    if not math.isfinite(value) or not 0 <= value <= 100:
        raise ValueError("value_pct must be a finite percentage from 0 to 100")
    return (CONTRACT_VERSION, source, SYMBOL, observation, available, value, fetched)


def _schema(connection: sqlite3.Connection) -> None:
    tables = {str(row[0]) for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}
    if tables != {TABLE}:
        raise ValueError(f"breadth observation database must contain only {TABLE}")
    columns = tuple(row[1] for row in connection.execute(f'PRAGMA table_info("{TABLE}")'))
    if columns != SCHEMA_COLUMNS:
        raise ValueError(f"incompatible {TABLE} schema")


def _initialize(connection: sqlite3.Connection) -> None:
    connection.execute(
        f"CREATE TABLE IF NOT EXISTS {TABLE}(contract_version TEXT NOT NULL, source TEXT NOT NULL, symbol TEXT NOT NULL, observation_date TEXT NOT NULL, available_at TEXT NOT NULL, value_pct REAL NOT NULL, fetched_at TEXT NOT NULL, PRIMARY KEY(contract_version,source,symbol,observation_date,available_at)) WITHOUT ROWID"
    )
    _schema(connection)
    for action in ("UPDATE", "DELETE"):
        connection.execute(
            f"CREATE TRIGGER IF NOT EXISTS {TABLE}_no_{action.lower()} BEFORE {action} ON {TABLE} BEGIN SELECT RAISE(ABORT,'immutable breadth observation'); END"
        )


def persist_breadth_observations(target: str | Path, observations: Iterable[Mapping[str, object]]) -> dict[str, int | str]:
    """Append exact NDTH records atomically; exact repeats are idempotent."""
    records = [_record(item) for item in observations]
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    inserted = idempotent = 0
    with closing(sqlite3.connect(path)) as connection:
        try:
            connection.execute("BEGIN IMMEDIATE")
            _initialize(connection)
            for record in records:
                key = record[:5]
                existing = connection.execute(
                    f"SELECT {','.join(SCHEMA_COLUMNS)} FROM {TABLE} WHERE contract_version=? AND source=? AND symbol=? AND observation_date=? AND available_at=?", key
                ).fetchone()
                if existing is not None:
                    if tuple(existing) != record:
                        raise ImmutableBreadthObservationConflict("immutable breadth observation conflict")
                    idempotent += 1
                else:
                    connection.execute(f"INSERT INTO {TABLE} VALUES ({','.join('?' for _ in SCHEMA_COLUMNS)})", record)
                    inserted += 1
            connection.commit()
        except Exception:
            connection.rollback()
            raise
    return {"contract_version": CONTRACT_VERSION, "inserted": inserted, "idempotent": idempotent}


def import_ndth_csv(target: str | Path, source_csv: str | Path, *, fetched_at: str) -> dict[str, int | str]:
    """Import a user-reviewed NDTH CSV with date,value_pct[,available_at] columns."""
    with Path(source_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not rows[0] or not {"date", "value_pct"}.issubset(rows[0]):
        raise ValueError("NDTH CSV requires date,value_pct columns")
    records = [
        {
            "contract_version": CONTRACT_VERSION, "source": CSV_SOURCE, "symbol": SYMBOL,
            "observation_date": row["date"], "available_at": row.get("available_at") or row["date"],
            "value_pct": row["value_pct"], "fetched_at": fetched_at,
        }
        for row in rows
    ]
    return persist_breadth_observations(target, records)


def _nq_calendar(nq_sessions: Sequence[object] | pd.DatetimeIndex) -> pd.DatetimeIndex:
    calendar = pd.DatetimeIndex(pd.to_datetime(nq_sessions), name="date")
    if calendar.tz is not None or calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise ValueError("NQ sessions must be unique, increasing calendar dates")
    return calendar


def _ndth_rows(database: str | Path, source: str) -> list[tuple[str, str, float]]:
    path = Path(database)
    if not path.is_file():
        raise FileNotFoundError(path)
    if source not in ALLOWED_SOURCES:
        raise ValueError("source must be an approved NDTH source")
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        connection.execute("PRAGMA query_only=ON")
        _schema(connection)
        return connection.execute(
            f"SELECT observation_date,available_at,value_pct FROM {TABLE} WHERE contract_version=? AND source=? AND symbol=? ORDER BY available_at,observation_date",
            (CONTRACT_VERSION, source, SYMBOL),
        ).fetchall()


def _session_observations(
    rows: Sequence[tuple[str, str, float]], calendar: pd.DatetimeIndex,
) -> dict[int, tuple[pd.Timestamp, str, float]]:
    selected: dict[int, tuple[pd.Timestamp, str, float]] = {}
    for raw_day, raw_available, raw_value in rows:
        available = pd.Timestamp(str(raw_available))
        position = int(calendar.searchsorted(available, side="right"))
        if position < len(calendar):
            candidate = (available, str(raw_day), float(raw_value) / 100.0)
            if position not in selected or candidate[0] >= selected[position][0]:
                selected[position] = candidate
    return selected


def load_ndth_series(
    database: str | Path,
    nq_sessions: Sequence[object] | pd.DatetimeIndex,
    *,
    source: str = SOURCE,
) -> pd.Series:
    """Use each observation only on the next NQ session; never fill a missing day."""
    calendar = _nq_calendar(nq_sessions)
    selected = _session_observations(_ndth_rows(database, source), calendar)
    result = pd.Series(float("nan"), index=calendar, dtype=float)
    for position, (_, _, value) in selected.items():
        result.iloc[position] = value
    return result


def breadth_readiness(
    database: str | Path,
    nq_sessions: Sequence[object] | pd.DatetimeIndex,
    as_of: str | date,
    *,
    source: str = SOURCE,
) -> dict[str, str | None]:
    """Check the exact prior NQ observation without filling gaps or writing data.

    ``as_of`` must be a supplied NQ session. ``latest_usable_date`` is the last
    NQ session receiving a value under the loader's existing availability rule.
    """
    cutoff = _day(as_of, "as_of")
    calendar = _nq_calendar(nq_sessions)
    calendar = calendar[calendar <= pd.Timestamp(cutoff)]
    position = int(calendar.searchsorted(pd.Timestamp(cutoff)))
    target_exists = position < len(calendar) and calendar[position] == pd.Timestamp(cutoff)
    required = calendar[position - 1].date().isoformat() if target_exists and position > 0 else None
    result: dict[str, str | None] = {
        "status": "MISSING", "as_of": cutoff, "required_observation_date": required,
        "latest_observation_date": None, "latest_usable_date": None,
        "message": "NDTH 관측 데이터가 없습니다.",
    }
    try:
        rows = _ndth_rows(database, source)
    except FileNotFoundError:
        return result
    rows = [row for row in rows if str(row[0]) <= cutoff]
    selected = _session_observations(rows, calendar)
    result["latest_observation_date"] = max((str(row[0]) for row in rows), default=None)
    result["latest_usable_date"] = max(
        (calendar[index].date().isoformat() for index, (_, _, value) in selected.items() if math.isfinite(value)),
        default=None,
    )
    if required is None:
        result["message"] = f"{cutoff} 및 직전 NQ 거래일을 확인할 수 없습니다."
    elif rows:
        observation = selected.get(position)
        ready = observation is not None and observation[1] == required and math.isfinite(observation[2])
        result["status"] = "READY" if ready else "DELAYED"
        result["message"] = (
            f"NDTH {required} 관측으로 {cutoff} 계산이 가능합니다."
            if ready else f"NDTH {required} 관측 대기: {cutoff} 계산을 보류합니다."
        )
    return result


def _historyofmarket_records(*, as_of: str, timeout: int) -> list[dict[str, object]]:
    """Download the published Nasdaq-100 breadth series and store pct200 only.

    The provider describes this as current-constituent breadth; rows are therefore
    explicitly CONDITIONAL_EXTERNAL_BREADTH, never constituent-level PIT evidence.
    """
    fetched = _day(as_of, "as_of")
    request = Request(
        "https://historyofmarket.com/api/ndx/breadth.json",
        headers={"User-Agent": "Market Lens-Market-Lens/1.0", "Accept": "application/json"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
        rows = payload["series"]
        source_note = str(payload["source"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise RuntimeError("History of Market NDTH request returned an invalid payload") from error
    if "current constituents" not in source_note.lower() or not isinstance(rows, list):
        raise RuntimeError("History of Market NDTH provenance is not the expected current-constituent series")
    records: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, Mapping) or "date" not in row or "pct200" not in row:
            raise RuntimeError("History of Market NDTH row is malformed")
        observation = _day(row["date"], "date")
        if observation > fetched:
            raise RuntimeError("History of Market NDTH row exceeds --as-of")
        records.append({
            "contract_version": CONTRACT_VERSION, "source": SOURCE, "symbol": SYMBOL,
            "observation_date": observation, "available_at": observation,
            "value_pct": row["pct200"], "fetched_at": fetched,
        })
    if not records:
        raise RuntimeError("History of Market NDTH returned no rows")
    return records


def download_historyofmarket_ndth(
    target: str | Path, *, as_of: str, timeout: int = 20
) -> dict[str, int | str]:
    """Initial full import of the published Nasdaq-100 breadth series."""
    records = _historyofmarket_records(as_of=as_of, timeout=timeout)
    return persist_breadth_observations(target, records)


def refresh_historyofmarket_ndth(
    target: str | Path, *, as_of: str, timeout: int = 20
) -> dict[str, int | str]:
    """Append missing NDTH observation keys, including late gaps; never revise history."""
    path = Path(target)
    if not path.is_file():
        raise FileNotFoundError(path)
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        connection.execute("PRAGMA query_only=ON")
        _schema(connection)
        rows = connection.execute(
            f"SELECT observation_date,available_at FROM {TABLE} WHERE contract_version=? AND source=? AND symbol=?",
            (CONTRACT_VERSION, SOURCE, SYMBOL),
        ).fetchall()
    existing = {(str(observation), str(available)) for observation, available in rows}
    latest = max((observation for observation, _ in existing), default="")
    records = [record for record in _historyofmarket_records(as_of=as_of, timeout=timeout)
               if (str(record["observation_date"]), str(record["available_at"])) not in existing]
    result = persist_breadth_observations(path, records)
    return {**result, "latest_before": latest or None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("download", "refresh"))
    parser.add_argument("target", type=Path)
    parser.add_argument("--as-of", default=datetime.now(timezone.utc).date().isoformat())
    parser.add_argument("--required-as-of", help="require breadth ready for this NQ session")
    parser.add_argument("--publish-database", type=Path, help="read the actual NQ session calendar")
    args = parser.parse_args()
    if bool(args.required_as_of) != bool(args.publish_database):
        parser.error("--required-as-of and --publish-database must be used together")
    if args.required_as_of:
        required_as_of = _day(args.required_as_of, "required_as_of")
        with closing(sqlite3.connect(args.publish_database.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
            connection.execute("PRAGMA query_only=ON")
            nq_sessions = [row[0] for row in connection.execute(
                "SELECT substr(date,1,10) FROM daily_prices WHERE ticker='NQ=F' "
                "AND substr(date,1,10)<=? ORDER BY substr(date,1,10)", (required_as_of,),
            )]
        _nq_calendar(nq_sessions)
    action = download_historyofmarket_ndth if args.command == "download" else refresh_historyofmarket_ndth
    result = action(args.target, as_of=args.as_of)
    if args.required_as_of:
        readiness = breadth_readiness(args.target, nq_sessions, required_as_of)
        print(json.dumps({**result, "readiness": readiness}, ensure_ascii=False))
        if readiness["status"] != "READY":
            raise SystemExit(3)
    else:
        print(result)


if __name__ == "__main__":
    main()
