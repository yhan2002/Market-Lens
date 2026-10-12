"""Point-in-time S&P 500 membership and market-breadth metrics."""

from __future__ import annotations

import argparse
import csv
import io
import sqlite3
from contextlib import closing
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


SP500_BREADTH_GROUP = "SP500_BREADTH"
EXACT_BREADTH_SOURCE = "EXACT_PIT_SP500_BREADTH_V1"
LIVE_BREADTH_SOURCE = "LIVE_CURRENT_SP500_BREADTH_V1"
EXACT_MEMBERSHIP_SOURCE_PREFIXES = ("LICENSED_PIT:", "BLOOMBERG_PIT:")
PUBLIC_MEMBERSHIP_SOURCE_PREFIX = "PUBLIC_CURRENT:"
DEFAULT_SOURCE = "SP500_MEMBERSHIP_DERIVED"
OPEN_DATA_SP500_URL = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/"
    "main/data/constituents.csv"
)
OPEN_DATA_MEMBERSHIP_SOURCE = "PUBLIC_CURRENT:DATASETS_SP500_PDDL_V1"
MIN_CURRENT_MEMBERS = 490
MAX_CURRENT_MEMBERS = 510
MAX_CURRENT_CHURN = 20
MIN_LIVE_BREADTH_COVERAGE = 0.98
KOREA_TIMEZONE = timezone(timedelta(hours=9))
SCHEMA = f"""
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sp500_membership (
    ticker TEXT NOT NULL,
    effective_from TEXT NOT NULL,
    effective_to TEXT,
    membership_as_of TEXT NOT NULL,
    available_at TEXT NOT NULL,
    source TEXT NOT NULL,
    PRIMARY KEY (ticker, effective_from),
    CHECK (effective_to IS NULL OR effective_to >= effective_from),
    CHECK (
        source GLOB 'LICENSED_PIT:?*'
        OR source GLOB 'BLOOMBERG_PIT:?*'
        OR source GLOB 'PUBLIC_CURRENT:?*'
    )
);
CREATE INDEX IF NOT EXISTS idx_sp500_membership_dates
    ON sp500_membership(effective_from, effective_to);
CREATE TABLE IF NOT EXISTS breadth_metrics (
    date TEXT PRIMARY KEY,
    breadth_200dma REAL NOT NULL,
    breadth_change_1d REAL,
    new_52w_lows INTEGER NOT NULL,
    advancers INTEGER NOT NULL,
    decliners INTEGER NOT NULL,
    coverage REAL NOT NULL,
    source TEXT NOT NULL,
    membership_as_of TEXT NOT NULL,
    available_at TEXT NOT NULL,
    CHECK (
        (source = '{EXACT_BREADTH_SOURCE}' AND coverage = 1.0)
        OR (
            source = '{LIVE_BREADTH_SOURCE}'
            AND coverage >= {MIN_LIVE_BREADTH_COVERAGE}
            AND coverage <= 1.0
        )
    ),
    CHECK (membership_as_of <= date),
    CHECK (available_at <= date)
);
"""


def initialize_database(database: Path) -> None:
    database.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(database)) as connection:
        connection.executescript(SCHEMA)
        for table in ("sp500_membership", "breadth_metrics"):
            columns = {
                str(row[1]) for row in connection.execute(f'PRAGMA table_info("{table}")')
            }
            for column in ("membership_as_of", "available_at"):
                if column not in columns:
                    connection.execute(f'ALTER TABLE "{table}" ADD COLUMN "{column}" TEXT')
        _migrate_source_checks(connection)
        connection.commit()


def _iso_date(value: date | str, field: str, *, allow_blank: bool = False) -> str | None:
    if isinstance(value, datetime):
        value = value.date()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    if not text and allow_blank:
        return None
    try:
        parsed = date.fromisoformat(text)
    except ValueError as error:
        raise ValueError(f"{field} must be YYYY-MM-DD: {value!r}") from error
    if parsed.isoformat() != text:
        raise ValueError(f"{field} must be YYYY-MM-DD: {value!r}")
    return text


def current_korea_date() -> date:
    """Return the capture date used by the unattended Windows publisher."""
    return datetime.now(KOREA_TIMEZONE).date()


def _is_exact_membership_source(source: str) -> bool:
    return any(source.startswith(prefix) and len(source) > len(prefix) for prefix in EXACT_MEMBERSHIP_SOURCE_PREFIXES)


def _is_public_membership_source(source: str) -> bool:
    return source.startswith(PUBLIC_MEMBERSHIP_SOURCE_PREFIX) and len(source) > len(
        PUBLIC_MEMBERSHIP_SOURCE_PREFIX
    )


def _is_allowed_membership_source(source: str) -> bool:
    return _is_exact_membership_source(source) or _is_public_membership_source(source)


def _migrate_source_checks(connection: sqlite3.Connection) -> None:
    """Expand old exact-only CHECK constraints without weakening provenance."""
    membership_sql_row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='sp500_membership'"
    ).fetchone()
    membership_sql = str(membership_sql_row[0] or "") if membership_sql_row else ""
    if "LICENSED_PIT:" in membership_sql and "PUBLIC_CURRENT:" not in membership_sql:
        connection.execute("DROP INDEX IF EXISTS idx_sp500_membership_dates")
        connection.execute(
            "ALTER TABLE sp500_membership RENAME TO sp500_membership_exact_only"
        )
        connection.executescript(SCHEMA.split("CREATE INDEX", 1)[0])
        connection.execute(
            """
            INSERT INTO sp500_membership(
                ticker, effective_from, effective_to,
                membership_as_of, available_at, source
            )
            SELECT ticker, effective_from, effective_to,
                   membership_as_of, available_at, source
            FROM sp500_membership_exact_only
            """
        )
        connection.execute("DROP TABLE sp500_membership_exact_only")
        connection.execute(
            "CREATE INDEX idx_sp500_membership_dates "
            "ON sp500_membership(effective_from, effective_to)"
        )

    breadth_sql_row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='breadth_metrics'"
    ).fetchone()
    breadth_sql = str(breadth_sql_row[0] or "") if breadth_sql_row else ""
    if EXACT_BREADTH_SOURCE in breadth_sql and (
        LIVE_BREADTH_SOURCE not in breadth_sql
        or str(MIN_LIVE_BREADTH_COVERAGE) not in breadth_sql
    ):
        connection.execute("ALTER TABLE breadth_metrics RENAME TO breadth_metrics_exact_only")
        breadth_table_sql = SCHEMA.split("CREATE TABLE IF NOT EXISTS breadth_metrics", 1)[1]
        connection.execute(
            "CREATE TABLE breadth_metrics" + breadth_table_sql.split(";", 1)[0]
        )
        connection.execute(
            """
            INSERT INTO breadth_metrics(
                date, breadth_200dma, breadth_change_1d, new_52w_lows,
                advancers, decliners, coverage, source,
                membership_as_of, available_at
            )
            SELECT date, breadth_200dma, breadth_change_1d, new_52w_lows,
                   advancers, decliners, coverage, source,
                   membership_as_of, available_at
            FROM breadth_metrics_exact_only
            """
        )
        connection.execute("DROP TABLE breadth_metrics_exact_only")


def _exact_members_cover_public_period(
    exact_rows: list[tuple[str, str, str | None, str, str, str]],
    public_rows: list[tuple[str, str, str | None, str, str, str]],
    latest_public_sync: str | None,
) -> bool:
    """Require a complete exact universe before replacing public live history."""
    if not public_rows:
        return True
    first_public = min(row[1] for row in public_rows)
    last_public = latest_public_sync or max(row[2] or row[1] for row in public_rows)
    checkpoints = {first_public, last_public}
    for _ticker, start, end, _membership_as_of, _available_at, _source in exact_rows:
        if first_public <= start <= last_public:
            checkpoints.add(start)
        if end is not None and first_public <= end <= last_public:
            checkpoints.add(end)
    for checkpoint in checkpoints:
        active = {
            ticker
            for ticker, start, end, membership_as_of, available_at, _source in exact_rows
            if start <= checkpoint
            and (end is None or end >= checkpoint)
            and membership_as_of <= checkpoint
            and available_at <= checkpoint
        }
        if len(active) < MIN_CURRENT_MEMBERS:
            return False
    return True


def _validate_intervals(
    rows: list[tuple[str, str, str | None, str | None, str | None, str]]
) -> None:
    by_ticker: dict[str, list[tuple[str, str | None]]] = {}
    for ticker, effective_from, effective_to, *_provenance in rows:
        by_ticker.setdefault(ticker, []).append((effective_from, effective_to))
    for ticker, intervals in by_ticker.items():
        first = True
        previous_end: str | None = None
        for effective_from, effective_to in sorted(intervals, key=lambda item: item[0]):
            if not first and (previous_end is None or effective_from <= previous_end):
                raise ValueError(f"overlapping membership intervals for {ticker}")
            first = False
            previous_end = effective_to


def _parse_current_members(payload: bytes) -> tuple[str, ...]:
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
    if "Symbol" not in (reader.fieldnames or ()):
        raise ValueError("current S&P 500 CSV must contain a Symbol column")
    raw_symbols = [str(row.get("Symbol") or "").strip().upper() for row in reader]
    if any(not symbol for symbol in raw_symbols):
        raise ValueError("current S&P 500 CSV contains an empty Symbol")
    tickers = tuple(sorted({symbol.replace(".", "-") for symbol in raw_symbols}))
    if len(tickers) != len(raw_symbols):
        raise ValueError("current S&P 500 CSV contains duplicate Yahoo symbols")
    if not MIN_CURRENT_MEMBERS <= len(tickers) <= MAX_CURRENT_MEMBERS:
        raise ValueError(
            f"current S&P 500 member count {len(tickers)} is outside "
            f"{MIN_CURRENT_MEMBERS}..{MAX_CURRENT_MEMBERS}"
        )
    return tickers


def sync_current_open_data_membership(
    database: Path,
    as_of: date | str,
    *,
    url: str = OPEN_DATA_SP500_URL,
) -> tuple[int, int, int]:
    """Capture today's public component list without backdating it."""
    as_of_iso = _iso_date(as_of, "as_of")
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept": "text/csv,text/plain,*/*",
        },
    )
    with urlopen(request, timeout=30) as response:
        tickers = _parse_current_members(response.read())

    initialize_database(database)
    with closing(sqlite3.connect(database)) as connection:
        conflict = connection.execute(
            """
            SELECT 1
            FROM sp500_membership
            WHERE (source GLOB 'LICENSED_PIT:?*' OR source GLOB 'BLOOMBERG_PIT:?*')
              AND effective_from<=?
              AND (effective_to IS NULL OR effective_to>=?)
            LIMIT 1
            """,
            (as_of_iso, as_of_iso),
        ).fetchone()
        if conflict:
            raise ValueError("licensed or Bloomberg membership already covers as_of")

        latest_snapshot_row = connection.execute(
            "SELECT value FROM metadata WHERE key='last_public_sp500_membership_sync'"
        ).fetchone()
        latest_snapshot = latest_snapshot_row[0] if latest_snapshot_row else None
        if latest_snapshot and latest_snapshot > as_of_iso:
            raise ValueError("current membership cannot be synced before its latest snapshot")

        open_rows = connection.execute(
            """
            SELECT ticker, effective_from
            FROM sp500_membership
            WHERE source GLOB 'PUBLIC_CURRENT:?*' AND effective_to IS NULL
            """,
        ).fetchall()
        active = {str(ticker): str(start) for ticker, start in open_rows}
        current = set(tickers)
        added = sorted(current - set(active))
        removed = sorted(set(active) - current)
        if active and len(added) + len(removed) > MAX_CURRENT_CHURN:
            raise ValueError(
                f"current S&P 500 list changed by {len(added) + len(removed)} symbols; "
                "refusing abnormal churn"
            )
        prior_date = (date.fromisoformat(as_of_iso) - timedelta(days=1)).isoformat()

        for ticker in removed:
            if active[ticker] == as_of_iso:
                connection.execute(
                    "DELETE FROM sp500_membership WHERE ticker=? AND effective_from=?",
                    (ticker, as_of_iso),
                )
            else:
                connection.execute(
                    """
                    UPDATE sp500_membership
                    SET effective_to=?
                    WHERE ticker=? AND effective_from=? AND source GLOB 'PUBLIC_CURRENT:?*'
                    """,
                    (prior_date, ticker, active[ticker]),
                )
        connection.executemany(
            """
            INSERT INTO sp500_membership(
                ticker, effective_from, effective_to,
                membership_as_of, available_at, source
            ) VALUES (?, ?, NULL, ?, ?, ?)
            """,
            [
                (ticker, as_of_iso, as_of_iso, as_of_iso, OPEN_DATA_MEMBERSHIP_SOURCE)
                for ticker in added
            ],
        )
        all_rows = connection.execute(
            """
            SELECT ticker, effective_from, effective_to,
                   membership_as_of, available_at, source
            FROM sp500_membership
            """
        ).fetchall()
        _validate_intervals([tuple(row) for row in all_rows])
        connection.execute(
            """
            INSERT OR REPLACE INTO metadata(key, value)
            VALUES ('last_public_sp500_membership_sync', ?)
            """,
            (as_of_iso,),
        )
        connection.commit()
    return len(current), len(added), len(removed)


def import_membership(
    database: Path, csv_path: Path, *, source: str | None = None
) -> int:
    """Validate and upsert inclusive membership intervals from a CSV file."""
    expected_source = source.strip() if source is not None else None
    if expected_source is not None and not _is_exact_membership_source(expected_source):
        raise ValueError("source must start with LICENSED_PIT: or BLOOMBERG_PIT:")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "ticker",
            "effective_from",
            "effective_to",
            "membership_as_of",
            "available_at",
            "source",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"membership CSV missing columns: {', '.join(sorted(missing))}")
        records: list[tuple[str, str, str | None, str, str, str]] = []
        for line_number, row in enumerate(reader, start=2):
            if not any(str(value or "").strip() for value in row.values()):
                continue
            ticker = str(row.get("ticker") or "").strip().upper()
            if not ticker:
                raise ValueError(f"ticker must not be empty at line {line_number}")
            effective_from = _iso_date(
                str(row.get("effective_from") or ""), f"effective_from at line {line_number}"
            )
            effective_to = _iso_date(
                str(row.get("effective_to") or ""),
                f"effective_to at line {line_number}",
                allow_blank=True,
            )
            if effective_to is not None and effective_to < effective_from:
                raise ValueError(f"effective_to precedes effective_from at line {line_number}")
            membership_as_of = _iso_date(
                str(row.get("membership_as_of") or ""),
                f"membership_as_of at line {line_number}",
            )
            available_at = _iso_date(
                str(row.get("available_at") or ""),
                f"available_at at line {line_number}",
            )
            row_source = str(row.get("source") or "").strip()
            if not _is_exact_membership_source(row_source):
                raise ValueError(
                    f"source at line {line_number} must start with "
                    "LICENSED_PIT: or BLOOMBERG_PIT:"
                )
            if expected_source is not None and row_source != expected_source:
                raise ValueError(f"source mismatch at line {line_number}")
            records.append(
                (
                    ticker,
                    effective_from,
                    effective_to,
                    membership_as_of,
                    available_at,
                    row_source,
                )
            )
    if not records:
        raise ValueError("membership CSV contains no rows")
    _validate_intervals(records)

    initialize_database(database)
    with closing(sqlite3.connect(database)) as connection:
        existing = connection.execute(
            "SELECT ticker, effective_from, effective_to, membership_as_of, "
            "available_at, source FROM sp500_membership"
        ).fetchall()
        exact_existing = [tuple(row) for row in existing if _is_exact_membership_source(row[5])]
        public_existing = [tuple(row) for row in existing if _is_public_membership_source(row[5])]
        merged = {(row[0], row[1]): row for row in exact_existing}
        for row in records:
            merged[(row[0], row[1])] = row
        exact_merged = list(merged.values())
        if public_existing:
            latest_public_row = connection.execute(
                "SELECT value FROM metadata WHERE key='last_public_sp500_membership_sync'"
            ).fetchone()
            latest_public = str(latest_public_row[0]) if latest_public_row else None
            if not _exact_members_cover_public_period(
                exact_merged, public_existing, latest_public
            ):
                raise ValueError(
                    "exact membership does not fully cover the existing PUBLIC_CURRENT "
                    "period; import a complete PIT file before replacing public live data"
                )
        _validate_intervals(exact_merged)
        if public_existing:
            connection.execute(
                "DELETE FROM sp500_membership WHERE source GLOB 'PUBLIC_CURRENT:?*'"
            )
            connection.execute(
                "DELETE FROM breadth_metrics WHERE source=?", (LIVE_BREADTH_SOURCE,)
            )
            connection.execute(
                "DELETE FROM metadata WHERE key='last_public_sp500_membership_sync'"
            )
        connection.executemany(
            """
            INSERT INTO sp500_membership(
                ticker, effective_from, effective_to, membership_as_of, available_at, source
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker, effective_from) DO UPDATE SET
                effective_to=excluded.effective_to,
                membership_as_of=excluded.membership_as_of,
                available_at=excluded.available_at,
                source=excluded.source
            """,
            records,
        )
        connection.commit()
    return len(records)


def active_members(database: Path, as_of: date | str | None = None) -> tuple[str, ...]:
    """Return members active on ``as_of``; effective_to is inclusive."""
    as_of_iso = _iso_date(as_of or current_korea_date(), "as_of")
    initialize_database(database)
    with closing(sqlite3.connect(database)) as connection:
        return tuple(
            row[0]
            for row in connection.execute(
                """
                SELECT DISTINCT ticker
                FROM sp500_membership
                WHERE effective_from <= ?
                  AND (effective_to IS NULL OR effective_to >= ?)
                  AND membership_as_of <= ?
                  AND available_at <= ?
                  AND date(effective_from)=effective_from
                  AND (effective_to IS NULL OR date(effective_to)=effective_to)
                  AND date(membership_as_of)=membership_as_of
                  AND date(available_at)=available_at
                  AND (
                      source GLOB 'LICENSED_PIT:?*'
                      OR source GLOB 'BLOOMBERG_PIT:?*'
                      OR source GLOB 'PUBLIC_CURRENT:?*'
                  )
                ORDER BY ticker
                """,
                (as_of_iso, as_of_iso, as_of_iso, as_of_iso),
            )
        )


def members_overlapping(
    database: Path, start: date | str, end: date | str
) -> tuple[str, ...]:
    """Return every member whose inclusive interval overlaps ``start``..``end``."""
    start_iso = _iso_date(start, "start")
    end_iso = _iso_date(end, "end")
    if start_iso > end_iso:
        raise ValueError("start must not be after end")
    initialize_database(database)
    with closing(sqlite3.connect(database)) as connection:
        return tuple(
            row[0]
            for row in connection.execute(
                """
                SELECT DISTINCT ticker
                FROM sp500_membership
                WHERE effective_from <= ?
                  AND (effective_to IS NULL OR effective_to >= ?)
                  AND membership_as_of <= ?
                  AND available_at <= ?
                  AND date(effective_from)=effective_from
                  AND (effective_to IS NULL OR date(effective_to)=effective_to)
                  AND date(membership_as_of)=membership_as_of
                  AND date(available_at)=available_at
                  AND (
                      source GLOB 'LICENSED_PIT:?*'
                      OR source GLOB 'BLOOMBERG_PIT:?*'
                      OR source GLOB 'PUBLIC_CURRENT:?*'
                  )
                ORDER BY ticker
                """,
                (end_iso, start_iso, end_iso, end_iso),
            )
        )


def compute_breadth_metrics(database: Path) -> int:
    """Recompute causal daily breadth over point-in-time active membership."""
    initialize_database(database)
    with closing(sqlite3.connect(database)) as connection:
        columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(daily_prices)")
        }
        if not {"ticker", "date"}.issubset(columns) or not ({"adj_close", "close"} & columns):
            raise ValueError("daily_prices must contain ticker, date, and adj_close or close")
        adjusted = "adj_close" if "adj_close" in columns else "NULL"
        close = "close" if "close" in columns else "NULL"
        prices = pd.read_sql_query(
            f"""
            SELECT ticker, substr(date, 1, 10) AS date,
                   {adjusted} AS adj_close, {close} AS close
            FROM daily_prices
            WHERE ticker IN (SELECT DISTINCT ticker FROM sp500_membership)
            ORDER BY ticker, date
            """,
            connection,
        )
        memberships = [
            (
                str(ticker),
                str(start),
                None if end is None else str(end),
                str(membership_as_of),
                str(available_at),
                str(row_source),
            )
            for ticker, start, end, membership_as_of, available_at, row_source
            in connection.execute(
                "SELECT ticker, effective_from, effective_to, membership_as_of, "
                "available_at, source FROM sp500_membership "
                "WHERE membership_as_of IS NOT NULL AND available_at IS NOT NULL "
                "AND date(effective_from)=effective_from "
                "AND (effective_to IS NULL OR date(effective_to)=effective_to) "
                "AND date(membership_as_of)=membership_as_of "
                "AND date(available_at)=available_at "
                "AND (source GLOB 'LICENSED_PIT:?*' "
                "OR source GLOB 'BLOOMBERG_PIT:?*' "
                "OR source GLOB 'PUBLIC_CURRENT:?*')"
            )
        ]

    if prices.empty:
        with closing(sqlite3.connect(database)) as connection:
            connection.execute("DELETE FROM breadth_metrics")
            connection.commit()
        return 0
    parsed_dates = pd.to_datetime(prices["date"], format="%Y-%m-%d", errors="coerce")
    prices = prices.loc[parsed_dates.notna()].copy()
    if prices.empty:
        with closing(sqlite3.connect(database)) as connection:
            connection.execute("DELETE FROM breadth_metrics")
            connection.commit()
        return 0
    prices["date"] = parsed_dates.loc[prices.index].dt.strftime("%Y-%m-%d")

    adjusted = pd.to_numeric(prices["adj_close"], errors="coerce")
    close = pd.to_numeric(prices["close"], errors="coerce")
    valid_adjusted = adjusted.where(np.isfinite(adjusted) & adjusted.gt(0.0))
    valid_close = close.where(np.isfinite(close) & close.gt(0.0))
    prices["price"] = valid_adjusted.fillna(valid_close)
    prices = prices.dropna(subset=["price"]).drop_duplicates(["ticker", "date"], keep="last")
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

    if not prices.empty:
        grouped = prices.groupby("ticker", sort=False)["price"]
        prices["previous"] = grouped.shift(1)
        prices["ma200"] = (
            grouped.rolling(200, min_periods=200).mean().reset_index(level=0, drop=True)
        )
        prices["low252"] = (
            grouped.rolling(252, min_periods=252).min().reset_index(level=0, drop=True)
        )

    _validate_intervals(memberships)
    rows: list[list[object]] = []
    metric_dates = sorted(prices["date"].unique())
    for metric_date, daily in prices.groupby("date", sort=True):
        active_rows = [
            row
            for row in memberships
            if row[1] <= metric_date
            and (row[2] is None or row[2] >= metric_date)
            and row[3] <= metric_date
            and row[4] <= metric_date
        ]
        active = {
            ticker
            for ticker, _effective_from, _effective_to, _membership_as_of, _available_at, _source
            in active_rows
        }
        if not active:
            continue
        source_kinds = {
            "EXACT"
            if _is_exact_membership_source(row[5])
            else "PUBLIC_LIVE"
            if _is_public_membership_source(row[5])
            else "INVALID"
            for row in active_rows
        }
        if "INVALID" in source_kinds or len(source_kinds) != 1:
            raise ValueError(f"mixed or invalid membership provenance on {metric_date}")
        metric_source = (
            EXACT_BREADTH_SOURCE
            if source_kinds == {"EXACT"}
            else LIVE_BREADTH_SOURCE
        )
        eligible = daily[
            daily["ticker"].isin(active)
            & daily[["previous", "ma200", "low252"]].notna().all(axis=1)
        ]
        coverage = len(eligible) / len(active)
        minimum_coverage = (
            1.0
            if metric_source == EXACT_BREADTH_SOURCE
            else MIN_LIVE_BREADTH_COVERAGE
        )
        if coverage < minimum_coverage:
            continue
        breadth = float(eligible["price"].gt(eligible["ma200"]).mean())
        rows.append(
            [
                metric_date,
                breadth,
                None,
                int(eligible["price"].le(eligible["low252"]).sum()),
                int(eligible["price"].gt(eligible["previous"]).sum()),
                int(eligible["price"].lt(eligible["previous"]).sum()),
                coverage,
                metric_source,
                max(row[3] for row in active_rows),
                max(row[4] for row in active_rows),
            ]
        )
    rows_by_date = {str(row[0]): row for row in rows}
    for previous_date, current_date in zip(metric_dates, metric_dates[1:]):
        previous = rows_by_date.get(str(previous_date))
        current = rows_by_date.get(str(current_date))
        if (
            previous is not None
            and current is not None
            and previous[7] == current[7]
        ):
            current[2] = float(current[1]) - float(previous[1])

    with closing(sqlite3.connect(database)) as connection:
        connection.execute("DELETE FROM breadth_metrics")
        connection.executemany(
            """
            INSERT INTO breadth_metrics(
                date, breadth_200dma, breadth_change_1d, new_52w_lows,
                advancers, decliners, coverage, source, membership_as_of, available_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        connection.commit()
    return len(rows)


def register_active_members(
    database: Path,
    as_of: date | str | None = None,
    *,
    group_name: str = SP500_BREADTH_GROUP,
    source: str = DEFAULT_SOURCE,
) -> int:
    tickers = active_members(database, as_of)
    if not tickers:
        raise ValueError("no active S&P 500 members for as_of")
    from market_data import register_universe

    return register_universe(database, tickers, group_name=group_name, source=source)


def register_members_overlapping(
    database: Path,
    start: date | str,
    end: date | str,
    *,
    group_name: str = SP500_BREADTH_GROUP,
    source: str = DEFAULT_SOURCE,
) -> int:
    tickers = members_overlapping(database, start, end)
    if not tickers:
        raise ValueError("no S&P 500 members overlap the requested range")
    from market_data import register_universe

    return register_universe(database, tickers, group_name=group_name, source=source)


def main() -> None:
    parser = argparse.ArgumentParser(description="Point-in-time S&P 500 breadth metrics")
    commands = parser.add_subparsers(dest="command", required=True)

    membership = commands.add_parser("import-membership")
    membership.add_argument("database", type=Path)
    membership.add_argument("csv", type=Path)
    membership.add_argument("--source")

    sync_current = commands.add_parser(
        "sync-current-open-data",
        help="capture the current public S&P 500 list from this date forward",
    )
    sync_current.add_argument("database", type=Path)

    compute = commands.add_parser("compute")
    compute.add_argument("database", type=Path)

    register = commands.add_parser("register-active")
    register.add_argument("database", type=Path)
    register.add_argument("--as-of", default=current_korea_date().isoformat())
    register.add_argument("--group-name", default=SP500_BREADTH_GROUP)
    register.add_argument("--source", default=DEFAULT_SOURCE)

    register_range = commands.add_parser("register-range")
    register_range.add_argument("database", type=Path)
    register_range.add_argument("--start", required=True)
    register_range.add_argument("--end", required=True)
    register_range.add_argument("--group-name", default=SP500_BREADTH_GROUP)
    register_range.add_argument("--source", default=DEFAULT_SOURCE)

    args = parser.parse_args()
    if args.command == "import-membership":
        count = import_membership(args.database, args.csv, source=args.source)
        print(f"imported {count} membership intervals")
    elif args.command == "sync-current-open-data":
        capture_date = current_korea_date().isoformat()
        active, added, removed = sync_current_open_data_membership(
            args.database, capture_date
        )
        print(
            f"captured {active} current members as of {capture_date} "
            f"(added {added}, removed {removed})"
        )
    elif args.command == "compute":
        count = compute_breadth_metrics(args.database)
        print(f"computed {count} breadth dates")
    elif args.command == "register-active":
        count = register_active_members(
            args.database,
            args.as_of,
            group_name=args.group_name,
            source=args.source,
        )
        print(f"registered {count} active tickers in {args.group_name}")
    else:
        count = register_members_overlapping(
            args.database,
            args.start,
            args.end,
            group_name=args.group_name,
            source=args.source,
        )
        print(f"registered {count} interval members in {args.group_name}")


if __name__ == "__main__":
    main()
