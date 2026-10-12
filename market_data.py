from __future__ import annotations

import argparse
import ast
import csv
import io
import json
import math
import os
import re
import sqlite3
import sys
from contextlib import closing
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from time import sleep
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo


CROSS_ASSET_TICKERS = (
    "^VIX",
    "^VIX3M",
    "^MOVE",
    "NQ=F",
    "JPY=X",
    "HYG",
    "IEF",
    "RSP",
    "IWM",
    "UUP",
)
EQUITY_UNIVERSE_GROUP = "CSQ_MP_CP"
EQUITY_UNIVERSE_COMPONENTS = ("CSQ_MP", "CSQ_CP")
RESEARCH_UNIVERSE_GROUP = "CSQ_MP_CP_UNIV"
YAHOO_BATCH_SIZE = 10
YAHOO_BATCH_PAUSE_SECONDS = 1.0
KOREA_TIMEZONE = timezone(timedelta(hours=9))
NEW_YORK_TIMEZONE = ZoneInfo("America/New_York")
MARKET_PROBE_TICKERS = ("SPY", "NQ=F")
CBOE_VIX3M_HISTORY_URL = (
    "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX3M_History.csv"
)
CBOE_VIX_HISTORY_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
UNIVERSE_GROUP_COMPONENTS = {
    EQUITY_UNIVERSE_GROUP: EQUITY_UNIVERSE_COMPONENTS,
    RESEARCH_UNIVERSE_GROUP: ("CSQ_MP", "CSQ_CP", "CSQ_UNIV"),
}


SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS universe (
    ticker TEXT NOT NULL,
    group_name TEXT NOT NULL,
    source TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
    PRIMARY KEY (ticker, group_name)
);
CREATE TABLE IF NOT EXISTS daily_prices (
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    adj_close REAL,
    volume REAL,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (ticker, date)
);
CREATE INDEX IF NOT EXISTS idx_daily_prices_date ON daily_prices(date);
CREATE TABLE IF NOT EXISTS reviewed_move_closes (
    date TEXT PRIMARY KEY,
    close REAL NOT NULL CHECK (close > 0),
    source TEXT NOT NULL,
    imported_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS regime_probabilities (
    date TEXT PRIMARY KEY,
    p_bull REAL NOT NULL,
    p_sideways REAL NOT NULL,
    p_bear REAL NOT NULL,
    model_version TEXT NOT NULL
);
"""


def read_universe_group(config_path: Path, group_name: str) -> list[str]:
    """Read a literal universe group without executing QuantBot config code."""
    tree = ast.parse(config_path.read_text(encoding="utf-8-sig"), filename=str(config_path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "universe_groups" for target in node.targets):
            groups = ast.literal_eval(node.value)
            selected_groups = UNIVERSE_GROUP_COMPONENTS.get(group_name, (group_name,))
            missing = [name for name in selected_groups if name not in groups]
            if missing:
                raise KeyError(f"universe group not found: {', '.join(missing)}")
            return sorted(
                {
                    str(ticker).strip().upper()
                    for name in selected_groups
                    for ticker in groups[name]
                    if str(ticker).strip()
                }
            )
    raise ValueError(f"literal universe_groups assignment not found in {config_path}")


def initialize_database(
    database: Path,
    config_path: Path,
    *,
    group_name: str = EQUITY_UNIVERSE_GROUP,
    benchmarks: tuple[str, ...] = ("SPY", "QQQ"),
) -> tuple[int, int]:
    tickers = read_universe_group(config_path, group_name)
    database.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(database)) as connection:
        connection.executescript(SCHEMA)
        connection.execute("UPDATE universe SET active = 0 WHERE group_name = ?", (group_name,))
        connection.executemany(
            """
            INSERT INTO universe(ticker, group_name, source, active) VALUES (?, ?, ?, 1)
            ON CONFLICT(ticker, group_name) DO UPDATE SET source=excluded.source, active=1
            """,
            [(ticker, group_name, str(config_path)) for ticker in tickers],
        )
        connection.executemany(
            """
            INSERT INTO universe(ticker, group_name, source, active) VALUES (?, 'REGIME_BENCHMARK', 'built-in', 1)
            ON CONFLICT(ticker, group_name) DO UPDATE SET active=1
            """,
            [(ticker,) for ticker in benchmarks],
        )
        metadata = {
            "universe_group": group_name,
            "universe_config": str(config_path),
            "initialized_at": datetime.now(timezone.utc).isoformat(),
        }
        connection.executemany(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)", metadata.items()
        )
        connection.commit()
    return len(tickers), len(benchmarks)


def register_universe(
    database: Path,
    tickers: tuple[str, ...],
    *,
    group_name: str = "REGIME_CROSS_ASSET",
    source: str = "built-in",
) -> int:
    normalized = sorted({ticker.strip().upper() for ticker in tickers if ticker.strip()})
    if not normalized:
        raise ValueError("at least one ticker is required")
    database.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(database)) as connection:
        connection.executescript(SCHEMA)
        connection.execute("UPDATE universe SET active=0 WHERE group_name=?", (group_name,))
        connection.executemany(
            """
            INSERT INTO universe(ticker, group_name, source, active) VALUES (?, ?, ?, 1)
            ON CONFLICT(ticker, group_name) DO UPDATE SET source=excluded.source, active=1
            """,
            [(ticker, group_name, source) for ticker in normalized],
        )
        connection.commit()
    return len(normalized)


def _download_start_dates(
    database: Path,
    *,
    years: int,
    today: date | None = None,
    group_name: str | None = None,
    tickers: tuple[str, ...] | None = None,
    backfill: bool = False,
) -> dict[str, date]:
    if years < 2:
        raise ValueError("years must be at least 2 for regime estimation")
    if group_name and tickers:
        raise ValueError("choose either group_name or tickers, not both")
    where = ["u.active=1"]
    params: list[str] = []
    if group_name:
        where.append("u.group_name=?")
        params.append(group_name)
    elif tickers:
        selected = sorted({ticker.strip().upper() for ticker in tickers if ticker.strip()})
        if not selected:
            raise ValueError("at least one ticker is required")
        where.append(f"u.ticker IN ({','.join('?' for _ in selected)})")
        params.extend(selected)
    query = f"""
        SELECT u.ticker, MAX(p.date)
        FROM universe u
        LEFT JOIN daily_prices p ON p.ticker=u.ticker
        WHERE {' AND '.join(where)}
        GROUP BY u.ticker
        ORDER BY u.ticker
    """
    with closing(sqlite3.connect(database)) as connection:
        rows = connection.execute(query, params).fetchall()
    if not rows:
        raise ValueError("no active tickers match the requested scope")
    current = today or date.today()
    earliest = current - timedelta(days=round(years * 365.25))
    return {
        ticker: earliest
        if backfill or not latest
        else max(earliest, date.fromisoformat(latest) - timedelta(days=7))
        for ticker, latest in rows
    }


def download_prices(
    database: Path,
    *,
    years: int = 10,
    group_name: str | None = None,
    tickers: tuple[str, ...] | None = None,
    backfill: bool = False,
    require_date: date | None = None,
) -> tuple[int, int, str]:
    """Download active tickers from Yahoo Finance and upsert adjusted OHLCV."""
    try:
        import pandas as pd
        _configure_yahoo_ca_bundle()
        import yfinance as yf
        _configure_yahoo_cache(yf)
    except ImportError as error:
        raise RuntimeError("install requirements.txt before downloading") from error

    starts = _download_start_dates(
        database,
        years=years,
        group_name=group_name,
        tickers=tickers,
        backfill=backfill,
    )
    end = _yahoo_end_date_kst()
    fetched_at = datetime.now(timezone.utc).isoformat()
    records: list[tuple[object, ...]] = []
    start_groups: dict[date, list[str]] = {}
    for ticker, start in starts.items():
        start_groups.setdefault(start, []).append(ticker)
    for start, group_tickers in sorted(start_groups.items()):
        for offset in range(0, len(group_tickers), YAHOO_BATCH_SIZE):
            batch = group_tickers[offset : offset + YAHOO_BATCH_SIZE]
            options = {
                "tickers": batch,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "auto_adjust": False,
                "actions": False,
                "group_by": "ticker",
                "progress": False,
            }
            direct_rows = _download_direct_chart_batch(batch, start, end, fetched_at)
            direct_tickers = {str(row[0]) for row in direct_rows}
            direct_required = {
                str(row[0])
                for row in direct_rows
                if require_date is not None
                and str(row[1]) == require_date.isoformat()
                and (_number(row[5]) or 0) > 0
            }
            if direct_tickers == set(batch) and (
                require_date is None or direct_required == set(batch)
            ):
                records.extend(direct_rows)
                if offset + YAHOO_BATCH_SIZE < len(group_tickers):
                    sleep(YAHOO_BATCH_PAUSE_SECONDS)
                continue

            # Keep yfinance only as a fallback for a partially unavailable direct API.
            if require_date is not None:
                options["start"] = require_date.isoformat()
                options["end"] = (require_date + timedelta(days=1)).isoformat()
            raw = yf.download(**options, threads=False)
            if raw.empty:
                missing = sorted(set(batch) - direct_tickers)
                raise RuntimeError(
                    f"Yahoo Finance returned no rows for {len(missing)} requested tickers"
                )
            for ticker in batch:
                if isinstance(raw.columns, pd.MultiIndex):
                    if ticker in raw.columns.get_level_values(0):
                        frame = raw[ticker].copy()
                    elif ticker in raw.columns.get_level_values(1):
                        frame = raw.xs(ticker, axis=1, level=1).copy()
                    else:
                        continue
                else:
                    frame = raw.copy()
                frame.columns = [
                    str(column).strip().lower().replace(" ", "_") for column in frame.columns
                ]
                if "close" not in frame:
                    continue
                for timestamp, row in frame.dropna(subset=["close"]).iterrows():
                    records.append(
                        (
                            ticker,
                            pd.Timestamp(timestamp).date().isoformat(),
                            _number(row.get("open")),
                            _number(row.get("high")),
                            _number(row.get("low")),
                            _number(row.get("close")),
                            _number(row.get("adj_close", row.get("close"))),
                            _number(row.get("volume")),
                            fetched_at,
                        )
                    )
            if offset + YAHOO_BATCH_SIZE < len(group_tickers):
                sleep(YAHOO_BATCH_PAUSE_SECONDS)
            if require_date is not None:
                required_rows = {
                    str(row[0])
                    for row in records
                    if str(row[1]) == require_date.isoformat() and (_number(row[5]) or 0) > 0
                }
                missing = sorted(set(batch) - required_rows)
                if missing:
                    raise RuntimeError(
                        f"Yahoo Finance returned no {require_date.isoformat()} close for {', '.join(missing)}"
                    )
    if not records:
        raise RuntimeError("Yahoo Finance returned no usable price rows")

    with closing(sqlite3.connect(database)) as connection:
        connection.executemany(
            """
            INSERT INTO daily_prices(
                ticker, date, open, high, low, close, adj_close, volume, fetched_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker, date) DO UPDATE SET
                open=excluded.open, high=excluded.high, low=excluded.low,
                close=excluded.close, adj_close=excluded.adj_close,
                volume=excluded.volume, fetched_at=excluded.fetched_at
            """,
            records,
        )
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES ('last_price_download_at', ?)",
            (fetched_at,),
        )
        connection.commit()
    return len(starts), len(records), min(starts.values()).isoformat()


def _configure_yahoo_ca_bundle() -> None:
    """Keep curl-cffi off the non-ASCII workspace certificate path on Windows."""
    try:
        import certifi
    except ImportError:
        return
    candidates = (
        Path(sys.base_prefix) / "Lib" / "site-packages" / "pip" / "_vendor" / "certifi" / "cacert.pem",
        Path(certifi.where()),
    )
    for bundle in candidates:
        if bundle.is_file():
            bundle_path = str(bundle)
            os.environ["CURL_CA_BUNDLE"] = bundle_path
            os.environ["SSL_CERT_FILE"] = bundle_path
            certifi.where = lambda: bundle_path
            return


def _configure_yahoo_cache(yfinance: object) -> None:
    """Use a local ASCII cache path; yfinance's default can be unwritable here."""
    setter = getattr(yfinance, "set_tz_cache_location", None)
    if not callable(setter):
        return
    cache_dir = Path(__file__).resolve().parent / ".yfinance-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    setter(str(cache_dir))


def _yahoo_end_date_kst(now: datetime | None = None) -> date:
    """Return Yahoo's exclusive end after the latest completed US session."""
    instant = now or datetime.now(timezone.utc)
    if instant.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    new_york_now = instant.astimezone(NEW_YORK_TIMEZONE)
    return new_york_now.date() + timedelta(days=new_york_now.time() >= time(17))


def _download_direct_chart_batch(
    tickers: list[str], start: date, end: date, fetched_at: str
) -> list[tuple[object, ...]]:
    rows: list[tuple[object, ...]] = []
    for ticker in tickers:
        rows.extend(_download_direct_chart_rows(ticker, start, end, fetched_at))
        sleep(0.15)
    return rows


def probe_latest_completed_market_date(
    *,
    now: datetime | None = None,
    tickers: tuple[str, ...] = MARKET_PROBE_TICKERS,
    fallback_database: Path | None = None,
) -> date:
    """Return the common latest completed US session without writing to the DB."""
    instant = now or datetime.now(timezone.utc)
    if instant.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    new_york_now = instant.astimezone(NEW_YORK_TIMEZONE)
    today = new_york_now.date()
    exclusive_cutoff = _yahoo_end_date_kst(instant)
    fetched_at = instant.astimezone(timezone.utc).isoformat()
    latest: dict[str, date] = {}
    for ticker in tickers:
        rows = _download_direct_chart_rows(
            ticker,
            today - timedelta(days=14),
            today + timedelta(days=1),
            fetched_at,
        )
        completed = [
            observed
            for row in rows
            if (observed := date.fromisoformat(str(row[1]))) < exclusive_cutoff
        ]
        if not completed:
            if fallback_database is None:
                raise RuntimeError(f"Yahoo latest-session probe failed for {ticker}")
            continue
        latest[ticker] = max(completed)
    # Local rows are useful for offline analysis, but cannot establish today's latest session.
    # Keep fallback_database in the signature for existing callers; do not use it as freshness proof.
    if len(latest) != len(tickers):
        missing = ", ".join(ticker for ticker in tickers if ticker not in latest)
        raise RuntimeError(f"Yahoo latest-session probe failed for {missing}; 최신 거래일 확인 불가, 캐시 최신성 미확인")
    if len(set(latest.values())) != 1:
        summary = ", ".join(f"{ticker}={observed}" for ticker, observed in latest.items())
        raise RuntimeError(f"Yahoo latest-session mismatch: {summary}")
    return next(iter(latest.values()))


def _download_direct_chart_rows(
    ticker: str, start: date, end: date, fetched_at: str
) -> list[tuple[object, ...]]:
    """Read Yahoo daily OHLCV without yfinance's cookie/crumb session."""
    if ticker in {"^VIX", "^VIX3M"}:
        official_rows = _download_cboe_index_rows(ticker, start, end, fetched_at)
        if official_rows:
            return official_rows

    midnight = datetime.min.time()
    query = urlencode(
        {
            "period1": int(
                datetime.combine(start, midnight, tzinfo=timezone.utc).timestamp()
            ),
            "period2": int(
                datetime.combine(end, midnight, tzinfo=timezone.utc).timestamp()
            ),
            "interval": "1d",
            "events": "div,splits",
            "includeAdjustedClose": "true",
        }
    )
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{quote(ticker, safe='')}?{query}"
    )
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=20) as response:
            result = json.load(response)["chart"]["result"][0]
        timestamps = result.get("timestamp") or []
        indicators = result.get("indicators") or {}
        quote_rows = (indicators.get("quote") or [{}])[0]
        adjusted = (indicators.get("adjclose") or [{}])[0].get("adjclose") or []
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return []

    rows = []
    for index, timestamp in enumerate(timestamps):
        close = _at(quote_rows.get("close"), index)
        if close is None:
            continue
        adjusted_close = _at(adjusted, index)
        rows.append(
            (
                ticker,
                datetime.fromtimestamp(timestamp, tz=timezone.utc).date().isoformat(),
                _number(_at(quote_rows.get("open"), index)),
                _number(_at(quote_rows.get("high"), index)),
                _number(_at(quote_rows.get("low"), index)),
                _number(close),
                _number(close if adjusted_close is None else adjusted_close),
                _number(_at(quote_rows.get("volume"), index)),
                fetched_at,
            )
        )
    return rows


def _download_cboe_index_rows(
    ticker: str, start: date, end: date, fetched_at: str
) -> list[tuple[object, ...]]:
    """Read official Cboe VIX-family daily history CSVs."""
    url = {"^VIX": CBOE_VIX_HISTORY_URL, "^VIX3M": CBOE_VIX3M_HISTORY_URL}.get(ticker)
    if url is None:
        return []
    request = Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
    )
    try:
        with urlopen(request, timeout=20) as response:
            payload = response.read().decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(payload))
        required = {"DATE", "OPEN", "HIGH", "LOW", "CLOSE"}
        if not required.issubset(set(reader.fieldnames or [])):
            return []
        rows = []
        for record in reader:
            observed = datetime.strptime(record["DATE"].strip(), "%m/%d/%Y").date()
            if not start <= observed < end:
                continue
            close = float(record["CLOSE"])
            rows.append(
                (
                    ticker,
                    observed.isoformat(),
                    float(record["OPEN"]),
                    float(record["HIGH"]),
                    float(record["LOW"]),
                    close,
                    close,
                    None,
                    fetched_at,
                )
            )
        return rows
    except (OSError, UnicodeError, ValueError, KeyError, TypeError):
        return []


def _download_cboe_vix3m_rows(
    start: date, end: date, fetched_at: str
) -> list[tuple[object, ...]]:
    """Compatibility wrapper for the VIX3M parser."""
    return _download_cboe_index_rows("^VIX3M", start, end, fetched_at)


def _at(values: object, index: int) -> object | None:
    try:
        return values[index]  # type: ignore[index]
    except (IndexError, TypeError):
        return None


def _number(value: object) -> float | None:
    try:
        number = float(value)
        return number if number == number else None
    except (TypeError, ValueError):
        return None


def database_status(database: Path) -> list[tuple[str, object]]:
    with closing(sqlite3.connect(database)) as connection:
        return [
            ("active_tickers", connection.execute("SELECT COUNT(DISTINCT ticker) FROM universe WHERE active=1").fetchone()[0]),
            ("price_rows", connection.execute("SELECT COUNT(*) FROM daily_prices").fetchone()[0]),
            ("first_date", connection.execute("SELECT MIN(date) FROM daily_prices").fetchone()[0]),
            ("last_date", connection.execute("SELECT MAX(date) FROM daily_prices").fetchone()[0]),
        ]


def import_move_close_csv(database: Path, csv_path: Path, *, tolerance: float = 0.02) -> tuple[int, int]:
    """Store reviewed MOVE closes as immutable overrides over Yahoo's daily prices."""
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"날짜", "종가"} <= set(reader.fieldnames):
            raise ValueError("MOVE CSV must contain 날짜 and 종가 columns")
        rows: dict[str, float] = {}
        for line, row in enumerate(reader, start=2):
            raw_date = re.sub(r"\s+", "", str(row.get("날짜") or ""))
            try:
                parsed_date = date.fromisoformat(raw_date).isoformat()
                close = float(str(row.get("종가") or "").replace(",", ""))
            except ValueError as error:
                raise ValueError(f"invalid MOVE CSV row {line}") from error
            if not math.isfinite(close) or close <= 0 or parsed_date in rows:
                raise ValueError(f"invalid or duplicate MOVE CSV row {line}")
            rows[parsed_date] = close
    if not rows:
        raise ValueError("MOVE CSV has no rows")
    fetched_at = datetime.now(timezone.utc).isoformat()
    inserted = verified = 0
    with closing(sqlite3.connect(database)) as connection:
        connection.executescript(SCHEMA)
        for row_date, close in sorted(rows.items()):
            existing = connection.execute(
                "SELECT close FROM reviewed_move_closes WHERE date=?", (row_date,)
            ).fetchone()
            if existing is not None:
                if abs(float(existing[0]) - close) > tolerance:
                    raise ValueError(f"MOVE CSV conflicts with reviewed value on {row_date}")
                verified += 1
                continue
            connection.execute(
                "INSERT INTO reviewed_move_closes(date,close,source,imported_at) VALUES (?,?,?,?)",
                (row_date, close, "INVESTING_REVIEWED_CSV", fetched_at),
            )
            inserted += 1
        connection.commit()
    return inserted, verified


def main() -> None:
    parser = argparse.ArgumentParser(description="Regime Router SQLite market-data store")
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init", help="create DB and snapshot QuantBot universe")
    init.add_argument("database", type=Path)
    init.add_argument("config", type=Path)
    init.add_argument("--group", default=EQUITY_UNIVERSE_GROUP)

    register = commands.add_parser("register", help="register a fixed research universe")
    register.add_argument("database", type=Path)
    register.add_argument("--group", default="REGIME_CROSS_ASSET")
    register.add_argument("--ticker", action="append", dest="tickers")

    download = commands.add_parser("download", help="incrementally download Yahoo prices")
    download.add_argument("database", type=Path)
    download.add_argument("--years", type=int, default=10)
    download.add_argument(
        "--backfill",
        action="store_true",
        help="ignore existing latest dates and request the full --years window",
    )
    scope = download.add_mutually_exclusive_group()
    scope.add_argument("--group", "--group-name", dest="group")
    scope.add_argument("--ticker", action="append", dest="tickers")

    status = commands.add_parser("status", help="show local DB coverage")
    status.add_argument("database", type=Path)

    move_csv = commands.add_parser("import-move-csv", help="store reviewed ^MOVE closes as priority overrides")
    move_csv.add_argument("database", type=Path)
    move_csv.add_argument("csv", type=Path)

    args = parser.parse_args()
    if args.command == "init":
        universe_count, benchmark_count = initialize_database(
            args.database, args.config, group_name=args.group
        )
        print(
            f"initialized {args.database}: {universe_count} active tickers in "
            f"{args.group} + {benchmark_count} benchmark tickers"
        )
    elif args.command == "register":
        count = register_universe(
            args.database,
            tuple(args.tickers or CROSS_ASSET_TICKERS),
            group_name=args.group,
            source="cli",
        )
        print(f"registered {count} active tickers in {args.group}")
    elif args.command == "download":
        ticker_count, row_count, start = download_prices(
            args.database,
            years=args.years,
            group_name=args.group,
            tickers=tuple(args.tickers) if args.tickers else None,
            backfill=args.backfill,
        )
        print(f"downloaded/upserted {row_count} rows for {ticker_count} tickers since {start}")
    elif args.command == "import-move-csv":
        inserted, verified = import_move_close_csv(args.database, args.csv)
        print(f"MOVE CSV: inserted={inserted} verified={verified}")
    else:
        for key, value in database_status(args.database):
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
