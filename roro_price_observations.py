"""Append-only Yahoo session observations for the SHADOW RORO inputs."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from contextlib import closing
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd


CONTRACT_VERSION = "yahoo-session-v1-jpyx-gld-close"
SOURCE = "YAHOO_CHART_V1"
ALLOWED_TICKERS = frozenset({"JPY=X", "GLD"})
PRICE_FIELD = "close"
NQ_DECISION_CUTOFF = time(17)  # New York regular close plus one hour.
NQ_TIMEZONE = ZoneInfo("America/New_York")
TABLE = "roro_price_observations"
SCHEMA_COLUMNS = (
    "contract_version",
    "source",
    "ticker",
    "raw_timestamp_utc",
    "source_timezone",
    "observation_session_date",
    "available_at",
    "close",
    "adj_close",
    "fetched_at",
)
KEY_COLUMNS = (
    "contract_version",
    "source",
    "ticker",
    "raw_timestamp_utc",
    "available_at",
)


class ImmutablePriceObservationConflict(ValueError):
    """Raised when an existing observation key is supplied with changed content."""


class YahooChartError(RuntimeError):
    """Raised when Yahoo's chart response cannot satisfy the frozen contract."""


def _calendar_date(value: object, field: str) -> str:
    raw = str(value)
    try:
        parsed = date.fromisoformat(raw)
    except ValueError as error:
        raise ValueError(f"{field} must be YYYY-MM-DD") from error
    if parsed.isoformat() != raw:
        raise ValueError(f"{field} must be YYYY-MM-DD")
    return raw


def _utc_timestamp(value: object, field: str) -> str:
    raw = str(value).strip()
    try:
        parsed = pd.Timestamp(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be an aware UTC timestamp") from error
    offset = parsed.utcoffset() if parsed.tzinfo is not None else None
    if pd.isna(parsed) or offset is None or offset.total_seconds() != 0:
        raise ValueError(f"{field} must be an aware UTC timestamp")
    return parsed.tz_convert("UTC").isoformat().replace("+00:00", "Z")


def _timezone(value: object) -> str:
    name = str(value).strip()
    if not name:
        raise ValueError("source_timezone is required")
    try:
        ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as error:
        raise ValueError("source_timezone must be an IANA timezone") from error
    return name


def _price(value: object, field: str, *, nullable: bool = False) -> float | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite positive number")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a finite positive number") from error
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{field} must be a finite positive number")
    return number


def _normalize_record(record: Mapping[str, object]) -> tuple[Any, ...]:
    missing = [column for column in SCHEMA_COLUMNS if column not in record]
    extra = [column for column in record if column not in SCHEMA_COLUMNS]
    if missing or extra:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unexpected " + ", ".join(sorted(extra)))
        raise ValueError("invalid price observation fields: " + "; ".join(details))

    contract_version = str(record["contract_version"]).strip()
    source = str(record["source"]).strip()
    ticker = str(record["ticker"]).strip()
    if contract_version != CONTRACT_VERSION:
        raise ValueError(f"contract_version must be {CONTRACT_VERSION}")
    if source != SOURCE:
        raise ValueError(f"source must be {SOURCE}")
    if ticker not in ALLOWED_TICKERS:
        raise ValueError("ticker must be JPY=X or GLD")

    raw_timestamp_utc = _utc_timestamp(record["raw_timestamp_utc"], "raw_timestamp_utc")
    source_timezone = _timezone(record["source_timezone"])
    observation_session_date = _calendar_date(
        record["observation_session_date"], "observation_session_date"
    )
    available_at = _utc_timestamp(record["available_at"], "available_at")
    fetched_at = _utc_timestamp(record["fetched_at"], "fetched_at")
    raw_timestamp = pd.Timestamp(raw_timestamp_utc)
    available_timestamp = pd.Timestamp(available_at)
    fetched_timestamp = pd.Timestamp(fetched_at)
    if raw_timestamp.tz_convert(source_timezone).date().isoformat() != observation_session_date:
        raise ValueError(
            "observation_session_date must equal raw_timestamp_utc in source_timezone"
        )
    if available_timestamp < raw_timestamp:
        raise ValueError("available_at must be on or after raw_timestamp_utc")
    if fetched_timestamp < available_timestamp:
        raise ValueError("fetched_at must be on or after available_at")
    return (
        contract_version,
        source,
        ticker,
        raw_timestamp_utc,
        source_timezone,
        observation_session_date,
        available_at,
        _price(record["close"], "close"),
        _price(record["adj_close"], "adj_close", nullable=True),
        fetched_at,
    )


def _records(observations: Iterable[Mapping[str, object]] | pd.DataFrame) -> list[tuple[Any, ...]]:
    raw_records = (
        observations.to_dict(orient="records")
        if isinstance(observations, pd.DataFrame)
        else list(observations)
    )
    if any(not isinstance(record, Mapping) for record in raw_records):
        raise ValueError("each price observation must be a mapping")
    return [_normalize_record(record) for record in raw_records]


def _validate_schema(connection: sqlite3.Connection) -> None:
    tables = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    }
    if tables != {TABLE}:
        raise ValueError(f"price observation database must contain only {TABLE}")
    actual = tuple(row[1] for row in connection.execute(f'PRAGMA table_info("{TABLE}")'))
    if actual != SCHEMA_COLUMNS:
        raise ValueError(f"incompatible {TABLE} schema")


def _initialize(connection: sqlite3.Connection) -> None:
    existing = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    }
    if existing - {TABLE}:
        raise ValueError(f"price observation database must contain only {TABLE}")
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE}(
            contract_version TEXT NOT NULL,
            source TEXT NOT NULL,
            ticker TEXT NOT NULL,
            raw_timestamp_utc TEXT NOT NULL,
            source_timezone TEXT NOT NULL,
            observation_session_date TEXT NOT NULL,
            available_at TEXT NOT NULL,
            close REAL NOT NULL,
            adj_close REAL,
            fetched_at TEXT NOT NULL,
            PRIMARY KEY(contract_version,source,ticker,raw_timestamp_utc,available_at)
        ) WITHOUT ROWID
        """
    )
    _validate_schema(connection)
    connection.execute(
        f"CREATE TRIGGER IF NOT EXISTS {TABLE}_no_update "
        f"BEFORE UPDATE ON {TABLE} BEGIN SELECT RAISE(ABORT,'immutable price observation'); END"
    )
    connection.execute(
        f"CREATE TRIGGER IF NOT EXISTS {TABLE}_no_delete "
        f"BEFORE DELETE ON {TABLE} BEGIN SELECT RAISE(ABORT,'immutable price observation'); END"
    )


def persist_price_observations(
    target: str | Path,
    observations: Iterable[Mapping[str, object]] | pd.DataFrame,
) -> dict[str, int | str]:
    """Atomically append observations; identical rows are idempotent."""
    records = _records(observations)
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    inserted = idempotent = 0
    with closing(sqlite3.connect(path)) as connection:
        try:
            connection.execute("BEGIN IMMEDIATE")
            _initialize(connection)
            for record in records:
                key = tuple(record[SCHEMA_COLUMNS.index(column)] for column in KEY_COLUMNS)
                existing = connection.execute(
                    f"SELECT {','.join(SCHEMA_COLUMNS)} FROM {TABLE} "
                    "WHERE contract_version=? AND source=? AND ticker=? "
                    "AND raw_timestamp_utc=? AND available_at=?",
                    key,
                ).fetchone()
                if existing is not None:
                    if tuple(existing) != record:
                        raise ImmutablePriceObservationConflict(
                            "immutable price observation conflict: "
                            + " ".join(str(value) for value in key)
                        )
                    idempotent += 1
                    continue
                connection.execute(
                    f"INSERT INTO {TABLE} VALUES ({','.join('?' for _ in SCHEMA_COLUMNS)})",
                    record,
                )
                inserted += 1
            connection.commit()
        except Exception:
            connection.rollback()
            raise
    return {
        "contract_version": CONTRACT_VERSION,
        "inserted": inserted,
        "idempotent": idempotent,
    }


def _calendar(nq_sessions: Sequence[object] | pd.DatetimeIndex) -> pd.DatetimeIndex:
    calendar = pd.DatetimeIndex(pd.to_datetime(nq_sessions), name="date")
    if calendar.tz is not None:
        raise ValueError("NQ sessions must be timezone-naive calendar dates")
    if calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise ValueError("NQ sessions must be unique and increasing")
    if any(timestamp != timestamp.normalize() for timestamp in calendar):
        raise ValueError("NQ sessions must be calendar dates")
    return calendar


def map_observations_to_nq_sessions(
    observations: Iterable[Mapping[str, object]] | pd.DataFrame,
    nq_sessions: Sequence[object] | pd.DatetimeIndex,
) -> pd.Series:
    """Place a close on the first NQ decision cutoff (17:00 New York) after it is available."""
    calendar = _calendar(nq_sessions)
    records = _records(observations)
    tickers = {str(record[2]) for record in records}
    if len(tickers) > 1:
        raise ValueError("map one ticker at a time")

    cutoffs = pd.DatetimeIndex(
        [
            pd.Timestamp(session.date()).tz_localize(NQ_TIMEZONE).replace(
                hour=NQ_DECISION_CUTOFF.hour,
                minute=NQ_DECISION_CUTOFF.minute,
            )
            for session in calendar
        ]
    ).tz_convert("UTC")
    selected: dict[int, tuple[pd.Timestamp, pd.Timestamp, float]] = {}
    for record in records:
        raw_timestamp = pd.Timestamp(record[3])
        available_at = pd.Timestamp(record[6])
        position = int(cutoffs.searchsorted(available_at, side="left"))
        if position >= len(calendar):
            continue
        candidate = (available_at, raw_timestamp, float(record[7]))
        prior = selected.get(position)
        if prior is None or candidate[:2] > prior[:2]:
            selected[position] = candidate

    result = pd.Series(float("nan"), index=calendar, dtype=float)
    for position, (_, _, value) in selected.items():
        result.iloc[position] = value
    return result


def load_observation_series(
    database: str | Path,
    *,
    ticker: str,
    nq_sessions: Sequence[object] | pd.DatetimeIndex,
) -> pd.Series:
    """Read one approved ticker in query-only mode and apply its session contract."""
    if ticker not in ALLOWED_TICKERS:
        raise ValueError("ticker must be JPY=X or GLD")
    path = Path(database)
    if not path.is_file():
        raise FileNotFoundError(path)
    uri = path.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        connection.execute("PRAGMA query_only=ON")
        _validate_schema(connection)
        rows = connection.execute(
            f"SELECT {','.join(SCHEMA_COLUMNS)} FROM {TABLE} "
            "WHERE contract_version=? AND source=? AND ticker=? "
            "ORDER BY available_at,raw_timestamp_utc",
            (CONTRACT_VERSION, SOURCE, ticker),
        ).fetchall()
    observations = [dict(zip(SCHEMA_COLUMNS, row, strict=True)) for row in rows]
    return map_observations_to_nq_sessions(observations, nq_sessions)


def _parse_chart_payload(
    payload: object,
    *,
    ticker: str,
    start: date,
    end: date,
    as_of: pd.Timestamp,
    fetched_at: str,
) -> list[dict[str, object]]:
    try:
        chart = payload["chart"]  # type: ignore[index]
        if chart.get("error") is not None:
            raise YahooChartError(f"Yahoo chart error for {ticker}: {chart['error']}")
        results = chart["result"]
        if not isinstance(results, list) or len(results) != 1:
            raise YahooChartError(f"Yahoo chart result missing for {ticker}")
        result = results[0]
        timezone_name = _timezone(result["meta"]["exchangeTimezoneName"])
        timestamps = result["timestamp"]
        indicators = result["indicators"]
        quotes = indicators["quote"]
        if not isinstance(timestamps, list) or not timestamps:
            raise YahooChartError(f"Yahoo chart returned no rows for {ticker}")
        if not isinstance(quotes, list) or len(quotes) != 1:
            raise YahooChartError(f"Yahoo quote rows malformed for {ticker}")
        closes = quotes[0]["close"]
        adjusted_blocks = indicators.get("adjclose") or []
        if not isinstance(closes, list) or len(closes) != len(timestamps):
            raise YahooChartError(f"Yahoo close rows malformed for {ticker}")
        if adjusted_blocks:
            if not isinstance(adjusted_blocks, list) or len(adjusted_blocks) != 1:
                raise YahooChartError(f"Yahoo adjusted-close rows malformed for {ticker}")
            adjusted = adjusted_blocks[0]["adjclose"]
            if not isinstance(adjusted, list) or len(adjusted) != len(timestamps):
                raise YahooChartError(f"Yahoo adjusted-close rows malformed for {ticker}")
        else:
            adjusted = [None] * len(timestamps)
    except YahooChartError:
        raise
    except (AttributeError, KeyError, TypeError, ZoneInfoNotFoundError, ValueError) as error:
        raise YahooChartError(f"Yahoo chart payload malformed for {ticker}") from error

    records: list[dict[str, object]] = []
    seen_timestamps: set[int] = set()
    source_zone = ZoneInfo(timezone_name)
    for raw, close, adj_close in zip(timestamps, closes, adjusted, strict=True):
        if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
            raise YahooChartError(f"Yahoo timestamp malformed for {ticker}")
        if raw in seen_timestamps:
            raise YahooChartError(f"Yahoo duplicate timestamp for {ticker}")
        seen_timestamps.add(raw)
        try:
            raw_time = pd.Timestamp(datetime.fromtimestamp(raw, tz=timezone.utc))
            session_date = raw_time.tz_convert(source_zone).date()
            if session_date < start or session_date > end:
                continue
            if raw_time > as_of:
                raise YahooChartError(f"Yahoo row exceeds --as-of for {ticker}")
            if close is None:
                continue
            record = {
                "contract_version": CONTRACT_VERSION,
                "source": SOURCE,
                "ticker": ticker,
                "raw_timestamp_utc": raw_time.isoformat().replace("+00:00", "Z"),
                "source_timezone": timezone_name,
                "observation_session_date": session_date.isoformat(),
                # Yahoo history has no vintage timestamp.  The raw chart timestamp plus
                # next-NQ-session mapper is the frozen conservative daily-bar contract.
                "available_at": raw_time.isoformat().replace("+00:00", "Z"),
                "close": _price(close, "close"),
                "adj_close": _price(adj_close, "adj_close", nullable=True),
                "fetched_at": fetched_at,
            }
            _normalize_record(record)
        except YahooChartError:
            raise
        except (OSError, OverflowError, ValueError) as error:
            raise YahooChartError(f"Yahoo price row malformed for {ticker}") from error
        records.append(record)
    if not records:
        raise YahooChartError(f"Yahoo chart returned no in-range rows for {ticker}")
    return records


def download_yahoo_observations(
    *,
    ticker: str,
    start: str | date,
    end: str | date,
    as_of: str,
    fetched_at: str | None = None,
    timeout: int = 20,
) -> list[dict[str, object]]:
    """Download one frozen ticker from Yahoo Chart without persisting it."""
    if ticker not in ALLOWED_TICKERS:
        raise ValueError("ticker must be JPY=X or GLD")
    start_date = date.fromisoformat(_calendar_date(start, "start"))
    end_date = date.fromisoformat(_calendar_date(end, "end"))
    if start_date > end_date:
        raise ValueError("start must be on or before end")
    as_of_time = pd.Timestamp(_utc_timestamp(as_of, "as_of"))
    if end_date > as_of_time.date():
        raise ValueError("end must be on or before the UTC --as-of date")
    fetched = _utc_timestamp(fetched_at or as_of_time.isoformat(), "fetched_at")
    period1 = datetime.combine(start_date, time.min, tzinfo=timezone.utc)
    period2 = min(
        datetime.combine(end_date + timedelta(days=1), time.min, tzinfo=timezone.utc),
        as_of_time.to_pydatetime() + timedelta(seconds=1),
    )
    query = urlencode(
        {
            "period1": int(period1.timestamp()),
            "period2": int(period2.timestamp()),
            "interval": "1d",
            "events": "div,splits",
            "includeAdjustedClose": "true",
        }
    )
    request = Request(
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{quote(ticker, safe='')}?{query}",
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except (OSError, ValueError, TypeError) as error:
        raise YahooChartError(f"Yahoo chart request failed for {ticker}") from error
    return _parse_chart_payload(
        payload,
        ticker=ticker,
        start=start_date,
        end=end_date,
        as_of=as_of_time,
        fetched_at=fetched,
    )


def download_and_persist(
    target: str | Path,
    *,
    start: str | date,
    end: str | date,
    as_of: str,
) -> dict[str, int | str]:
    """Download both frozen tickers, then atomically append all validated rows."""
    fetched_at = _utc_timestamp(as_of, "as_of")
    observations = [
        record
        for ticker in sorted(ALLOWED_TICKERS)
        for record in download_yahoo_observations(
            ticker=ticker,
            start=start,
            end=end,
            as_of=as_of,
            fetched_at=fetched_at,
        )
    ]
    return persist_price_observations(target, observations)


def refresh_from_publish_database(
    target: str | Path,
    publish_database: str | Path,
    *,
    as_of: str | None = None,
) -> dict[str, int | str]:
    """Append only new Yahoo sessions through the current NQ decision date.

    A missing or uneven seed is rejected instead of fetching an overlapping
    range with a new fetch timestamp and risking immutable-history conflicts.
    """
    target_path = Path(target)
    if not target_path.is_file():
        raise FileNotFoundError(target_path)
    publish_path = Path(publish_database)
    if not publish_path.is_file():
        raise FileNotFoundError(publish_path)
    with closing(sqlite3.connect(target_path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        connection.execute("PRAGMA query_only=ON")
        _validate_schema(connection)
        rows = connection.execute(
            f"SELECT ticker, MAX(observation_session_date) FROM {TABLE} "
            "WHERE contract_version=? AND source=? GROUP BY ticker",
            (CONTRACT_VERSION, SOURCE),
        ).fetchall()
    sessions = {str(ticker): str(day) for ticker, day in rows if day is not None}
    if set(sessions) != ALLOWED_TICKERS:
        raise ValueError("price observation seed must contain both JPY=X and GLD")
    with closing(sqlite3.connect(publish_path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        connection.execute("PRAGMA query_only=ON")
        row = connection.execute(
            "SELECT MAX(date) FROM daily_prices WHERE ticker='NQ=F'"
        ).fetchone()
    if row is None or row[0] is None:
        raise ValueError("publish database has no NQ=F date")
    latest_nq = _calendar_date(row[0], "latest NQ=F date")
    last_session = min(sessions.values())
    if all(session >= latest_nq for session in sessions.values()):
        return {
            "contract_version": CONTRACT_VERSION,
            "inserted": 0,
            "idempotent": 0,
            "latest_nq_date": latest_nq,
            "last_session_date": last_session,
        }
    fetched_at = as_of or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    observations = [
        record
        for ticker, session in sorted(sessions.items())
        if session < latest_nq
        for record in download_yahoo_observations(
            ticker=ticker,
            start=(date.fromisoformat(session) + timedelta(days=1)).isoformat(),
            end=latest_nq,
            as_of=fetched_at,
            fetched_at=fetched_at,
        )
    ]
    result = persist_price_observations(target_path, observations)
    return {
        **result,
        "latest_nq_date": latest_nq,
        "last_session_date": last_session,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="download frozen Yahoo observations")
    download.add_argument("database", type=Path)
    download.add_argument("--start", required=True)
    download.add_argument("--end", required=True)
    download.add_argument("--as-of", required=True, help="aware UTC cutoff timestamp")
    refresh = commands.add_parser("refresh", help="append new sessions through the latest NQ date")
    refresh.add_argument("database", type=Path)
    refresh.add_argument("--publish-database", type=Path, required=True)
    refresh.add_argument("--as-of", help="aware UTC fetch timestamp; defaults to now")
    args = parser.parse_args()
    if args.command == "download":
        result = download_and_persist(
            args.database,
            start=args.start,
            end=args.end,
            as_of=args.as_of,
        )
        print(
            f"{result['contract_version']}: inserted={result['inserted']} "
            f"idempotent={result['idempotent']}"
        )
    if args.command == "refresh":
        result = refresh_from_publish_database(
            args.database,
            args.publish_database,
            as_of=args.as_of,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
