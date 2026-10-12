"""Point-in-time macro observations sourced from official FRED/ALFRED downloads."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import sqlite3
import zipfile
from contextlib import closing
from datetime import date, datetime, timedelta, timezone
from html.parser import HTMLParser
from http.cookiejar import CookieJar
from pathlib import Path
from time import sleep
from typing import Callable, Iterable, Sequence
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import HTTPCookieProcessor, Request, build_opener, urlopen


FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"
ALFRED_DOWNLOAD_URL = "https://alfred.stlouisfed.org/series/downloaddata"
ALFRED_RETRYABLE_HTTP_STATUS = {408, 429, 500, 502, 503, 504}
ALFRED_VINTAGE_BATCH_SIZE = 25
FRED_SERIES = {
    "HY_OAS": "BAMLH0A0HYM2",
    "CASH_3M": "DGS3MO",
    "USDJPY": "DEXJPUS",
    "UST_2Y": "DGS2",
    "UST_10Y": "DGS10",
    "UST_REAL_10Y": "DFII10",
    "BREAKEVEN_10Y": "T10YIE",
    "WTI_SPOT": "DCOILWTICO",
    "BRENT_SPOT": "DCOILBRENTEU",
}
TURNING_MACRO_LIVE_SERIES = (
    "HY_OAS",
    "CASH_3M",
    "UST_2Y",
    "UST_10Y",
    "UST_REAL_10Y",
    "BREAKEVEN_10Y",
    "WTI_SPOT",
    "BRENT_SPOT",
)
TURNING_MACRO_BACKFILL_SERIES = TURNING_MACRO_LIVE_SERIES[2:]
TURNING_MACRO_MAX_LAG_DAYS = {series: 7 for series in TURNING_MACRO_LIVE_SERIES}

SCHEMA = """
CREATE TABLE IF NOT EXISTS macro_observations (
    date TEXT NOT NULL,
    series TEXT NOT NULL,
    value REAL NOT NULL,
    available_at TEXT NOT NULL,
    source TEXT NOT NULL,
    PRIMARY KEY (series, date, available_at)
);
CREATE INDEX IF NOT EXISTS idx_macro_observations_asof
ON macro_observations(series, available_at, date);
"""


def _iso_date(value: str, name: str) -> str:
    try:
        parsed = date.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be YYYY-MM-DD") from error
    if parsed.isoformat() != value:
        raise ValueError(f"{name} must be YYYY-MM-DD")
    return value


def _available_at(value: str | None) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")
    if len(value) == 10:
        return _iso_date(value, "available_at")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("available_at must be an ISO date or timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError("available_at timestamps must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat(timespec="seconds")


def initialize_database(database: Path) -> None:
    database.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(database)) as connection:
        connection.executescript(SCHEMA)
        connection.commit()


def parse_fred_csv(payload: bytes | str, series_id: str) -> list[tuple[str, float]]:
    """Parse an official FRED graph CSV, omitting blank or literal ``.`` missings."""
    if isinstance(payload, bytes):
        text = payload.decode("utf-8-sig")
    elif isinstance(payload, str):
        text = payload.lstrip("\ufeff")
    else:
        raise TypeError("payload must be bytes or str")

    reader = csv.DictReader(io.StringIO(text))
    fields = [field.strip() for field in (reader.fieldnames or []) if field]
    date_column = next(
        (field for field in fields if field.lower() in {"date", "observation_date"}),
        None,
    )
    value_column = next((field for field in fields if field.upper() == series_id.upper()), None)
    if date_column is None or value_column is None:
        raise ValueError(f"FRED CSV must contain a date column and {series_id}")

    observations: list[tuple[str, float]] = []
    seen: set[str] = set()
    for line_number, row in enumerate(reader, start=2):
        raw_date = (row.get(date_column) or "").strip()
        raw_value = (row.get(value_column) or "").strip()
        observation_date = _iso_date(raw_date, f"line {line_number} date")
        if observation_date in seen:
            raise ValueError(f"duplicate FRED observation date: {observation_date}")
        seen.add(observation_date)
        if raw_value in {"", "."}:
            continue
        try:
            value = float(raw_value)
        except ValueError as error:
            raise ValueError(f"line {line_number} has an invalid numeric value") from error
        if not math.isfinite(value):
            raise ValueError(f"line {line_number} has a non-finite value")
        observations.append((observation_date, value))
    return observations


def download_fred_csv(
    series_id: str,
    start: str,
    end: str,
    *,
    opener: Callable[..., object] | None = None,
    timeout: float = 30.0,
) -> bytes | str:
    """Download a bounded FRED graph CSV using only the standard library."""
    start = _iso_date(start, "start")
    end = _iso_date(end, "end")
    if start > end:
        raise ValueError("start must not be after end")
    query = urlencode({"id": series_id, "cosd": start, "coed": end})
    request = Request(
        f"{FRED_CSV_URL}?{query}",
        headers={"User-Agent": "RegimeRouter/1.0"},
    )
    response = (opener or urlopen)(request, timeout=timeout)
    try:
        payload = response.read()  # type: ignore[attr-defined]
    finally:
        close = getattr(response, "close", None)
        if close:
            close()
    if not isinstance(payload, (bytes, str)):
        raise TypeError("FRED response body must be bytes or str")
    return payload


def parse_fred_vintage_json(
    payload: bytes | str, series_id: str
) -> list[tuple[str, float, str]]:
    """Parse FRED output-type 3 rows as observation, value, availability."""
    if isinstance(payload, bytes):
        text = payload.decode("utf-8-sig")
    elif isinstance(payload, str):
        text = payload.lstrip("\ufeff")
    else:
        raise TypeError("payload must be bytes or str")
    try:
        document = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("FRED vintage response must be valid JSON") from error
    if not isinstance(document, dict):
        raise ValueError("FRED vintage response must be an object")
    if document.get("error_code") is not None:
        raise ValueError(
            f"FRED API error {document.get('error_code')}: {document.get('error_message', '')}"
        )
    observations = document.get("observations")
    if not isinstance(observations, list):
        raise ValueError("FRED vintage response must contain observations")
    try:
        expected = int(document.get("count", len(observations)))
    except (TypeError, ValueError) as error:
        raise ValueError("FRED vintage response count must be an integer") from error
    if expected > len(observations):
        raise ValueError("FRED vintage response was truncated; narrow the date range")

    result: list[tuple[str, float, str]] = []
    seen: set[tuple[str, str]] = set()
    for index, row in enumerate(observations, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"FRED vintage observation {index} must be an object")
        observation_date = _iso_date(str(row.get("date") or ""), f"observation {index} date")
        availability = _iso_date(
            str(row.get("realtime_start") or ""),
            f"observation {index} realtime_start",
        )
        if availability < observation_date:
            raise ValueError("FRED realtime_start must not precede the observation date")
        key = (observation_date, availability)
        if key in seen:
            raise ValueError(f"duplicate FRED vintage observation: {key}")
        seen.add(key)
        raw_value = str(row.get("value") or "").strip()
        if raw_value in {"", "."}:
            continue
        try:
            value = float(raw_value)
        except ValueError as error:
            raise ValueError(f"observation {index} has an invalid numeric value") from error
        if not math.isfinite(value):
            raise ValueError(f"observation {index} has a non-finite value")
        result.append((observation_date, value, availability))
    return result


def download_fred_vintages(
    series_id: str,
    start: str,
    end: str,
    *,
    api_key: str,
    realtime_end: str | None = None,
    opener: Callable[..., object] | None = None,
    timeout: float = 30.0,
) -> bytes | str:
    """Download bounded new/revised FRED vintages through the official API."""
    start = _iso_date(start, "start")
    end = _iso_date(end, "end")
    if start > end:
        raise ValueError("start must not be after end")
    key = api_key.strip()
    if not key:
        raise ValueError("FRED_API_KEY is required for vintage backfill")
    last_realtime_date = _iso_date(realtime_end or date.today().isoformat(), "realtime_end")
    query = urlencode(
        {
            "series_id": series_id,
            "api_key": key,
            "file_type": "json",
            "output_type": 3,
            "observation_start": start,
            "observation_end": end,
            "realtime_start": start,
            "realtime_end": last_realtime_date,
            "sort_order": "asc",
            "limit": 2000,
        }
    )
    request = Request(
        f"{FRED_API_URL}?{query}",
        headers={"User-Agent": "RegimeRouter/1.0"},
    )
    response = (opener or urlopen)(request, timeout=timeout)
    try:
        payload = response.read()  # type: ignore[attr-defined]
    finally:
        close = getattr(response, "close", None)
        if close:
            close()
    if not isinstance(payload, (bytes, str)):
        raise TypeError("FRED response body must be bytes or str")
    return payload


class _AlfredVintageDateParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._inside_vintage_select = False
        self.dates: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        attributes = dict(attrs)
        if tag == "select" and attributes.get("id") == "form_selected_vintage_dates":
            self._inside_vintage_select = True
        elif tag == "option" and self._inside_vintage_select:
            raw = attributes.get("value") or ""
            try:
                self.dates.append(_iso_date(raw, "ALFRED vintage date"))
            except ValueError:
                return

    def handle_endtag(self, tag: str) -> None:
        if tag == "select" and self._inside_vintage_select:
            self._inside_vintage_select = False


def _read_response(response: object) -> bytes | str:
    try:
        payload = response.read()  # type: ignore[attr-defined]
    finally:
        close = getattr(response, "close", None)
        if close:
            close()
    if not isinstance(payload, (bytes, str)):
        raise TypeError("HTTP response body must be bytes or str")
    return payload


def _read_response_with_retry(
    requester: Callable[..., object], request: Request, *, timeout: float
) -> bytes | str:
    """Retry transient ALFRED network failures without retrying bad payloads."""
    for delay in (1.0, 3.0, None):
        try:
            return _read_response(requester(request, timeout=timeout))
        except HTTPError as error:
            if error.code not in ALFRED_RETRYABLE_HTTP_STATUS or delay is None:
                raise
        except OSError:
            if delay is None:
                raise
        sleep(delay)
    raise AssertionError("unreachable")


def _alfred_csv_texts(payload: bytes) -> list[str]:
    if not isinstance(payload, bytes):
        raise TypeError("ALFRED ZIP response must be bytes")
    if zipfile.is_zipfile(io.BytesIO(payload)):
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            csv_names = [
                name
                for name in archive.namelist()
                if name.lower().endswith(".csv") and "/" not in name.replace("\\", "/")
            ]
            if not csv_names:
                raise ValueError("ALFRED ZIP must contain a root CSV")
            return [archive.read(name).decode("utf-8-sig") for name in csv_names]
    text = payload.decode("utf-8-sig", errors="replace")
    if not text.lstrip().startswith("observation_date,"):
        raise ValueError("ALFRED response must be a ZIP archive or ALFRED CSV")
    return [text]


def _alfred_payload_vintages(payload: bytes, series_id: str) -> set[str]:
    prefix = f"{series_id}_"
    return {
        f"{suffix[:4]}-{suffix[4:6]}-{suffix[6:]}"
        for text in _alfred_csv_texts(payload)
        for field in (csv.DictReader(io.StringIO(text)).fieldnames or [])
        if field.startswith(prefix)
        for suffix in [field[len(prefix) :]]
        if len(suffix) == 8 and suffix.isdigit()
    }


def parse_alfred_vintage_zip(
    payload: bytes, series_id: str
) -> list[tuple[str, float, str]]:
    """Parse ALFRED output-type 3 ZIP rows as observation/value/availability."""
    texts = _alfred_csv_texts(payload)

    prefix = f"{series_id}_"
    result: list[tuple[str, float, str]] = []
    seen: set[tuple[str, str]] = set()
    for text in texts:
        reader = csv.DictReader(io.StringIO(text))
        fields = reader.fieldnames or []
        if "observation_date" not in fields:
            raise ValueError("ALFRED CSV must contain observation_date")
        vintage_columns: dict[str, str] = {}
        for field in fields:
            if not field.startswith(prefix):
                continue
            suffix = field[len(prefix) :]
            if len(suffix) != 8 or not suffix.isdigit():
                raise ValueError(f"invalid ALFRED vintage column: {field}")
            vintage_columns[field] = _iso_date(
                f"{suffix[:4]}-{suffix[4:6]}-{suffix[6:]}",
                "ALFRED vintage column",
            )
        if not vintage_columns:
            raise ValueError(f"ALFRED CSV contains no {series_id} vintage columns")
        for line_number, row in enumerate(reader, start=2):
            observation_date = _iso_date(
                str(row.get("observation_date") or "").strip(),
                f"ALFRED line {line_number} date",
            )
            for field, availability in vintage_columns.items():
                raw_value = str(row.get(field) or "").strip()
                if raw_value in {"", "."}:
                    continue
                if availability < observation_date:
                    raise ValueError("ALFRED vintage date must not precede observation date")
                key = (observation_date, availability)
                if key in seen:
                    raise ValueError(f"duplicate ALFRED vintage observation: {key}")
                seen.add(key)
                try:
                    value = float(raw_value)
                except ValueError as error:
                    raise ValueError(
                        f"ALFRED line {line_number} has an invalid numeric value"
                    ) from error
                if not math.isfinite(value):
                    raise ValueError(f"ALFRED line {line_number} has a non-finite value")
                result.append((observation_date, value, availability))
    if not result:
        raise ValueError("ALFRED CSV contains no observations")
    return result


def download_alfred_vintages(
    series_id: str,
    start: str,
    end: str,
    *,
    opener: Callable[..., object] | None = None,
    timeout: float = 120.0,
    observation_start: str | None = None,
    observation_end: str | None = None,
) -> bytes:
    """Download official ALFRED new/revised vintages without an API key."""
    start = _iso_date(start, "start")
    end = _iso_date(end, "end")
    if start > end:
        raise ValueError("start must not be after end")
    observation_start = _iso_date(observation_start or start, "observation_start")
    observation_end = _iso_date(observation_end or end, "observation_end")
    if observation_start > observation_end:
        raise ValueError("observation_start must not be after observation_end")
    url = f"{ALFRED_DOWNLOAD_URL}?{urlencode({'seid': series_id})}"
    requester = opener or build_opener(HTTPCookieProcessor(CookieJar())).open
    try:
        page = _read_response_with_retry(
            requester,
            Request(url, headers={"User-Agent": "RegimeRouter/1.0"}),
            timeout=timeout,
        )
    except OSError as error:
        raise OSError(
            f"ALFRED vintage-page GET {series_id} {start}..{end} failed: "
            f"{type(error).__name__}: {error}"
        ) from error
    page_text = page.decode("utf-8-sig") if isinstance(page, bytes) else page
    parser = _AlfredVintageDateParser()
    parser.feed(page_text)
    vintage_dates = sorted({day for day in parser.dates if start <= day <= end})
    if not vintage_dates:
        raise ValueError("ALFRED page contains no vintage dates in the requested range")

    payloads: list[bytes] = []
    pending = [
        vintage_dates[offset : offset + ALFRED_VINTAGE_BATCH_SIZE]
        for offset in range(0, len(vintage_dates), ALFRED_VINTAGE_BATCH_SIZE)
    ]
    while pending:
        selected = pending.pop(0)
        fields: list[tuple[str, str]] = [
            ("form[units]", "lin"),
            ("form[obs_start_date]", observation_start),
            ("form[obs_end_date]", observation_end),
        ]
        fields.extend(("form[selected_vintage_dates][]", day) for day in selected)
        fields.extend(
            [
                ("form[entered_vintage_dates]", ""),
                ("form[file_type]", "3"),
                ("form[file_format]", "csv"),
                ("form[download_data]", "Download data"),
            ]
        )
        request = Request(
            url,
            data=urlencode(fields).encode("utf-8"),
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": "RegimeRouter/1.0",
            },
            method="POST",
        )
        try:
            payload = _read_response(requester(request, timeout=timeout))
        except HTTPError as error:
            if error.code not in ALFRED_RETRYABLE_HTTP_STATUS:
                raise
            payload = b""
        except TimeoutError:
            if len(selected) > 1:
                middle = len(selected) // 2
                pending[0:0] = [selected[:middle], selected[middle:]]
                continue
            payload = b""
        except OSError:
            payload = b""
        if not isinstance(payload, bytes) or not zipfile.is_zipfile(io.BytesIO(payload)):
            fallback_url = "https://alfred.stlouisfed.org/graph/alfredgraph.csv?" + urlencode(
                {
                    "id": series_id,
                    "vintage_date": ",".join(selected),
                    "cosd": observation_start,
                    "coed": observation_end,
                }
            )
            try:
                payload = _read_response_with_retry(
                    requester,
                    Request(fallback_url, headers={"User-Agent": "RegimeRouter/1.0"}),
                    timeout=timeout,
                )
            except HTTPError as error:
                if error.code not in ALFRED_RETRYABLE_HTTP_STATUS:
                    raise
                if len(selected) > 1:
                    middle = len(selected) // 2
                    pending[0:0] = [selected[:middle], selected[middle:]]
                    continue
                raise OSError(
                    f"ALFRED CSV fallback {series_id} {selected[0]} failed: "
                    f"HTTP {error.code}"
                ) from error
            except OSError as error:
                if len(selected) > 1:
                    middle = len(selected) // 2
                    pending[0:0] = [selected[:middle], selected[middle:]]
                    continue
                raise OSError(
                    f"ALFRED CSV fallback {series_id} {selected[0]} failed: "
                    f"{type(error).__name__}: {error}"
                ) from error
        if not isinstance(payload, bytes):
            raise TypeError("ALFRED ZIP response must be bytes")
        actual_vintages = _alfred_payload_vintages(payload, series_id)
        if actual_vintages != set(selected):
            raise ValueError(
                "ALFRED response vintage columns differ from the requested batch"
            )
        payloads.append(payload)

    if len(payloads) == 1:
        return payloads[0]
    combined = io.BytesIO()
    with zipfile.ZipFile(combined, "w", zipfile.ZIP_DEFLATED) as archive:
        for batch_number, payload in enumerate(payloads):
            for file_number, text in enumerate(_alfred_csv_texts(payload)):
                archive.writestr(
                    f"batch_{batch_number:03d}_{file_number:03d}.csv", text
                )
    return combined.getvalue()


def _observation_rows(
    series: str,
    observations: Iterable[tuple[str, float]],
    available_at: str,
    source: str,
) -> list[tuple[str, str, float, str, str]]:
    normalized_series = series.strip().upper()
    if not normalized_series:
        raise ValueError("series is required")
    vintage = _available_at(available_at)
    candidates: list[tuple[str, str, float, str, str]] = []
    seen_dates: set[str] = set()
    for observation_date, raw_value in observations:
        observation_date = _iso_date(observation_date, "observation date")
        if observation_date in seen_dates:
            raise ValueError(f"duplicate observation date: {observation_date}")
        seen_dates.add(observation_date)
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError("observation values must be finite")
        candidates.append((observation_date, normalized_series, value, vintage, source))
    return candidates


def upsert_observation_batches(
    database: Path,
    batches: Iterable[tuple[str, Iterable[tuple[str, float]], str, str]],
) -> dict[str, int]:
    """Atomically store multiple FRED series after every payload has validated."""
    candidates = [
        row
        for series, observations, available_at, source in batches
        for row in _observation_rows(series, observations, available_at, source)
    ]

    initialize_database(database)
    with closing(sqlite3.connect(database)) as connection:
        history: dict[tuple[str, str], list[tuple[str, float, str]]] = {}
        requested_series = sorted({row[1] for row in candidates})
        placeholders = ",".join("?" for _ in requested_series)
        query = (
            "SELECT series, date, available_at, value, source "
            "FROM macro_observations "
            f"WHERE upper(series) IN ({placeholders}) "
            "ORDER BY series, date, available_at"
        )
        existing = (
            connection.execute(query, requested_series) if requested_series else ()
        )
        for raw_series, raw_date, raw_available_at, raw_value, raw_source in existing:
            history.setdefault((str(raw_series).upper(), str(raw_date)), []).append(
                (str(raw_available_at), float(raw_value), str(raw_source))
            )
        rows: list[tuple[str, str, float, str, str]] = []
        counts = {series: 0 for series in requested_series}
        for row in candidates:
            prior = [
                item for item in history.get((row[1], row[0]), ()) if item[0] <= row[3]
            ]
            latest_as_of = prior[-1][1:] if prior else None
            # Repeated pulls do not create a new vintage, while an earlier
            # point-in-time backfill is retained even when a later value matches.
            if latest_as_of != (row[2], row[4]):
                rows.append(row)
                counts[row[1]] += 1
        connection.executemany(
            """
            INSERT INTO macro_observations(date, series, value, available_at, source)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(series, date, available_at) DO UPDATE SET
                value=excluded.value,
                source=excluded.source
            """,
            rows,
        )
        connection.commit()
    return counts


def upsert_observations(
    database: Path,
    series: str,
    observations: Iterable[tuple[str, float]],
    *,
    available_at: str,
    source: str,
) -> int:
    normalized_series = series.strip().upper()
    return upsert_observation_batches(
        database, [(normalized_series, observations, available_at, source)]
    ).get(normalized_series, 0)


def fetch_fred_series(
    database: Path,
    series: str,
    start: str,
    end: str,
    *,
    opener: Callable[..., object] | None = None,
    available_at: str | None = None,
) -> int:
    normalized_series = series.strip().upper()
    try:
        series_id = FRED_SERIES[normalized_series]
    except KeyError as error:
        raise ValueError(f"unknown FRED series: {series}") from error
    observations = parse_fred_csv(
        download_fred_csv(series_id, start, end, opener=opener), series_id
    )
    return upsert_observations(
        database,
        normalized_series,
        observations,
        available_at=_available_at(available_at),
        source=f"FRED:{series_id}",
    )


def _latest_stored_date(database: Path, series: str, end: str) -> str | None:
    if not database.exists():
        return None
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        try:
            row = connection.execute(
                "SELECT MAX(substr(date,1,10)) FROM macro_observations "
                "WHERE upper(series)=? AND substr(date,1,10)<=?",
                (series.upper(), end),
            ).fetchone()
        except sqlite3.DatabaseError:
            return None
    return str(row[0]) if row and row[0] else None


def validate_turning_macro_coverage(database: Path, end: str) -> dict[str, str]:
    end_date = date.fromisoformat(_iso_date(end, "end"))
    latest: dict[str, str] = {}
    stale: list[str] = []
    for series in TURNING_MACRO_LIVE_SERIES:
        latest_day = _latest_stored_date(database, series, end)
        if latest_day is None:
            stale.append(f"{series}=missing")
            continue
        latest[series] = latest_day
        lag = (end_date - date.fromisoformat(latest_day)).days
        if lag > TURNING_MACRO_MAX_LAG_DAYS[series]:
            stale.append(f"{series}={latest_day}({lag}d)")
    if stale:
        raise ValueError("turning macro coverage is stale: " + ", ".join(stale))
    return latest


def refresh_turning_macro_inputs(
    database: Path,
    end: str,
    *,
    opener: Callable[..., object] | None = None,
    available_at: str | None = None,
    initial_years: int = 3,
) -> tuple[dict[str, int], bool]:
    """Refresh live turning-point inputs atomically; use recent stored data on outage."""
    if initial_years < 1:
        raise ValueError("initial_years must be positive")
    end_date = date.fromisoformat(_iso_date(end, "end"))
    vintage = _available_at(available_at)
    batches = []
    try:
        for series in TURNING_MACRO_LIVE_SERIES:
            latest = _latest_stored_date(database, series, end)
            if latest:
                start_date = date.fromisoformat(latest) - timedelta(days=14)
            else:
                try:
                    start_date = end_date.replace(year=end_date.year - initial_years)
                except ValueError:
                    start_date = end_date.replace(
                        year=end_date.year - initial_years, day=28
                    )
            series_id = FRED_SERIES[series]
            observations = parse_fred_csv(
                download_fred_csv(
                    series_id, start_date.isoformat(), end, opener=opener
                ),
                series_id,
            )
            if not observations:
                raise ValueError(f"FRED {series_id} returned no observations")
            latest_download = max(date.fromisoformat(day) for day, _ in observations)
            if (end_date - latest_download).days > TURNING_MACRO_MAX_LAG_DAYS[series]:
                raise ValueError(f"FRED {series_id} is stale at {latest_download}")
            batches.append((series, observations, vintage, f"FRED:{series_id}"))
    except (OSError, RuntimeError, ValueError, sqlite3.DatabaseError):
        validate_turning_macro_coverage(database, end)
        return ({series: 0 for series in TURNING_MACRO_LIVE_SERIES}, True)

    counts = upsert_observation_batches(database, batches)
    validate_turning_macro_coverage(database, end)
    return counts, False


def backfill_turning_macro_alfred(
    database: Path,
    start: str,
    end: str,
    *,
    opener: Callable[..., object] | None = None,
) -> dict[str, int]:
    """Backfill the six new raw inputs with ALFRED vintage availability."""
    return {
        series: _backfill_alfred_series(
            database,
            start,
            end,
            series=series,
            series_id=FRED_SERIES[series],
            opener=opener,
        )
        for series in TURNING_MACRO_BACKFILL_SERIES
    }


def fetch_hy_oas(
    database: Path,
    start: str,
    end: str,
    *,
    opener: Callable[..., object] | None = None,
    available_at: str | None = None,
) -> int:
    """Fetch and store FRED's BAMLH0A0HYM2 in its native percentage-point unit."""
    return fetch_fred_series(
        database, "HY_OAS", start, end, opener=opener, available_at=available_at
    )


def fetch_cash_rate(
    database: Path,
    start: str,
    end: str,
    *,
    opener: Callable[..., object] | None = None,
    available_at: str | None = None,
) -> int:
    """Fetch the free daily 3-month Treasury yield for live cash carry."""
    return fetch_fred_series(
        database, "CASH_3M", start, end, opener=opener, available_at=available_at
    )


def backfill_hy_oas_vintages(
    database: Path,
    start: str,
    end: str,
    *,
    api_key: str,
    realtime_end: str | None = None,
    opener: Callable[..., object] | None = None,
) -> int:
    """Store official FRED new/revised vintages with date-level availability."""
    series_id = FRED_SERIES["HY_OAS"]
    payload = download_fred_vintages(
        series_id,
        start,
        end,
        api_key=api_key,
        realtime_end=realtime_end,
        opener=opener,
    )
    by_availability: dict[str, list[tuple[str, float]]] = {}
    for observation_date, value, availability in parse_fred_vintage_json(
        payload, series_id
    ):
        by_availability.setdefault(availability, []).append((observation_date, value))
    source = f"FRED_API:{series_id}:OUTPUT3"
    return sum(
        upsert_observations(
            database,
            "HY_OAS",
            observations,
            available_at=availability,
            source=source,
        )
        for availability, observations in sorted(by_availability.items())
    )


def _backfill_alfred_series(
    database: Path,
    start: str,
    end: str,
    *,
    series: str,
    series_id: str,
    opener: Callable[..., object] | None = None,
) -> int:
    start_date = date.fromisoformat(_iso_date(start, "start"))
    end_date = date.fromisoformat(_iso_date(end, "end"))
    if start_date > end_date:
        raise ValueError("start must not be after end")
    source = f"ALFRED:{series_id}:OUTPUT3"
    inserted = 0
    cursor = start_date
    while cursor <= end_date:
        chunk_end = min(end_date, date(cursor.year, 12, 31))
        try:
            payload = download_alfred_vintages(
                series_id,
                cursor.isoformat(),
                chunk_end.isoformat(),
                opener=opener,
                observation_start=start_date.isoformat(),
                observation_end=chunk_end.isoformat(),
            )
        except OSError as error:
            raise RuntimeError(
                f"ALFRED {series} {cursor.isoformat()}..{chunk_end.isoformat()} "
                f"failed: {type(error).__name__}: {error}"
            ) from error
        by_availability: dict[str, list[tuple[str, float]]] = {}
        for observation_date, value, availability in parse_alfred_vintage_zip(
            payload, series_id
        ):
            by_availability.setdefault(availability, []).append(
                (observation_date, value)
            )
        inserted += upsert_observation_batches(
            database,
            [
                (series, observations, availability, source)
                for availability, observations in sorted(by_availability.items())
            ],
        ).get(series, 0)
        cursor = date(cursor.year + 1, 1, 1)
    return inserted


def backfill_hy_oas_alfred(
    database: Path,
    start: str,
    end: str,
    *,
    opener: Callable[..., object] | None = None,
) -> int:
    """Store free official ALFRED HY OAS vintages."""
    return _backfill_alfred_series(
        database,
        start,
        end,
        series="HY_OAS",
        series_id=FRED_SERIES["HY_OAS"],
        opener=opener,
    )


def backfill_cash_rate_alfred(
    database: Path,
    start: str,
    end: str,
    *,
    opener: Callable[..., object] | None = None,
) -> int:
    """Store free official ALFRED 3-month Treasury vintages."""
    return _backfill_alfred_series(
        database,
        start,
        end,
        series="CASH_3M",
        series_id=FRED_SERIES["CASH_3M"],
        opener=opener,
    )


def import_hy_oas_csv(database: Path, csv_path: Path, *, source: str) -> int:
    """Import licensed PIT history with date,value,available_at columns."""
    source = source.strip()
    if not source:
        raise ValueError("source is required")
    by_availability: dict[str, list[tuple[str, float]]] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"date", "value", "available_at"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"HY OAS CSV missing columns: {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            if not any(str(value or "").strip() for value in row.values()):
                continue
            observation_date = _iso_date(
                str(row.get("date") or "").strip(), f"date at line {line_number}"
            )
            availability = _available_at(str(row.get("available_at") or "").strip())
            try:
                value = float(str(row.get("value") or "").strip())
            except ValueError as error:
                raise ValueError(f"value must be numeric at line {line_number}") from error
            if not math.isfinite(value):
                raise ValueError(f"value must be finite at line {line_number}")
            by_availability.setdefault(availability, []).append((observation_date, value))
    if not by_availability:
        raise ValueError("HY OAS CSV contains no observations")
    return sum(
        upsert_observations(
            database,
            "HY_OAS",
            observations,
            available_at=availability,
            source=source,
        )
        for availability, observations in sorted(by_availability.items())
    )


def import_hy_oas_revised_history_csv(database: Path, csv_path: Path) -> int:
    """Import a current FRED history export for historical score comparison."""
    source = "FRED_REVISED_HISTORY:BAMLH0A0HYM2"
    rows: list[tuple[str, str, float, str, str]] = []
    seen_dates: set[str] = set()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"observation_date", FRED_SERIES["HY_OAS"]}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"HY OAS CSV missing columns: {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            raw_date = str(row.get("observation_date") or "").strip()
            raw_value = str(row.get(FRED_SERIES["HY_OAS"]) or "").strip()
            if not raw_date and not raw_value:
                continue
            if raw_value in {"", "."}:
                continue
            observation_date = _iso_date(raw_date, f"observation_date at line {line_number}")
            if observation_date in seen_dates:
                raise ValueError(f"duplicate observation date at line {line_number}")
            try:
                value = float(raw_value)
            except ValueError as error:
                raise ValueError(f"value must be numeric at line {line_number}") from error
            if not math.isfinite(value):
                raise ValueError(f"value must be finite at line {line_number}")
            seen_dates.add(observation_date)
            rows.append((observation_date, "HY_OAS", value, observation_date, source))
    if not rows:
        raise ValueError("HY OAS CSV contains no observations")
    initialize_database(database)
    with closing(sqlite3.connect(database)) as connection:
        connection.executemany(
            """
            INSERT INTO macro_observations(date, series, value, available_at, source)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(series, date, available_at) DO UPDATE SET
                value=excluded.value, source=excluded.source
            """,
            rows,
        )
        connection.commit()
    return len(rows)


def import_usdjpy_revised_history_csv(database: Path, csv_path: Path) -> int:
    """Import FRED DEXJPUS history for the USD/JPY Fast condition.

    The file is a revised-history reference, not an assertion that each old
    observation was available intraday.  The Fast condition uses it only on
    the next model-series rebuild.
    """
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    observations = parse_fred_csv(csv_path.read_bytes(), FRED_SERIES["USDJPY"])
    if not observations:
        raise ValueError("DEXJPUS CSV contains no observations")
    return _store_usdjpy_revised_history(database, observations)


def _store_usdjpy_revised_history(
    database: Path, observations: Iterable[tuple[str, float]]
) -> int:
    source = "FRED_REVISED_HISTORY:DEXJPUS"
    rows = [(day, "USDJPY_FRED", value, day, source) for day, value in observations]
    initialize_database(database)
    with closing(sqlite3.connect(database)) as connection:
        connection.executemany(
            """
            INSERT INTO macro_observations(date, series, value, available_at, source)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(series, date, available_at) DO UPDATE SET
                value=excluded.value, source=excluded.source
            """,
            rows,
        )
        connection.commit()
    return len(rows)


def fetch_usdjpy_revised_history(
    database: Path, start: str, end: str, *, opener: Callable[..., object] | None = None
) -> int:
    """Fetch current official DEXJPUS history for the Fast USD/JPY input."""
    observations = parse_fred_csv(
        download_fred_csv(FRED_SERIES["USDJPY"], start, end, opener=opener),
        FRED_SERIES["USDJPY"],
    )
    return _store_usdjpy_revised_history(database, observations)


def series_as_of(
    database: Path,
    series: str,
    as_of: str,
    *,
    limit: int | None = None,
) -> list[tuple[str, float, str, str]]:
    """Return each observation's latest vintage available at the cutoff, newest first."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    cutoff = _available_at(as_of)
    cutoff_date = cutoff[:10]
    if not database.exists():
        return []
    availability_sql = (
        "substr(available_at, 1, 10)<=?" if len(as_of) == 10 else "available_at<=?"
    )
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        try:
            rows = connection.execute(
                f"""
                SELECT date, value, available_at, source
                FROM macro_observations
                WHERE upper(series)=?
                  AND substr(date, 1, 10)<=?
                  AND {availability_sql}
                ORDER BY substr(date, 1, 10) DESC, available_at DESC
                """,
                (series.strip().upper(), cutoff_date, cutoff),
            ).fetchall()
        except sqlite3.DatabaseError:
            return []

    result: list[tuple[str, float, str, str]] = []
    seen: set[str] = set()
    for raw_date, raw_value, raw_available, raw_source in rows:
        observation_date = str(raw_date)[:10]
        if observation_date in seen:
            continue
        seen.add(observation_date)
        value = float(raw_value)
        if not math.isfinite(value):
            continue
        result.append((observation_date, value, str(raw_available), str(raw_source)))
        if limit is not None and len(result) == limit:
            break
    return result


def latest_as_of(
    database: Path, series: str, as_of: str
) -> tuple[str, float, str, str] | None:
    rows = series_as_of(database, series, as_of, limit=1)
    return rows[0] if rows else None


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Point-in-time FRED macro data store")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch-hy-oas", help="fetch free FRED HY OAS CSV")
    fetch.add_argument("database", type=Path)
    fetch.add_argument("--start", required=True)
    fetch.add_argument("--end", required=True)
    cash = commands.add_parser(
        "fetch-cash-rate", help="fetch free FRED 3-month Treasury yield"
    )
    cash.add_argument("database", type=Path)
    cash.add_argument("--start", required=True)
    cash.add_argument("--end", required=True)
    vintages = commands.add_parser(
        "backfill-hy-oas-vintages",
        help="backfill official FRED HY OAS vintage availability",
    )
    vintages.add_argument("database", type=Path)
    vintages.add_argument("--start", required=True)
    vintages.add_argument("--end", required=True)
    vintages.add_argument("--realtime-end")
    alfred = commands.add_parser(
        "backfill-hy-oas-alfred",
        help="backfill free official ALFRED HY OAS vintage availability",
    )
    alfred.add_argument("database", type=Path)
    alfred.add_argument("--start", required=True)
    alfred.add_argument("--end", required=True)
    cash_alfred = commands.add_parser(
        "backfill-cash-rate-alfred",
        help="backfill free ALFRED 3-month Treasury vintages",
    )
    cash_alfred.add_argument("database", type=Path)
    cash_alfred.add_argument("--start", required=True)
    cash_alfred.add_argument("--end", required=True)
    turning_refresh = commands.add_parser(
        "refresh-turning-macro", help="refresh live macro inputs used by turning research"
    )
    turning_refresh.add_argument("database", type=Path)
    turning_refresh.add_argument("--end", required=True)
    turning_backfill = commands.add_parser(
        "backfill-turning-macro-alfred",
        help="backfill six turning inputs with free ALFRED vintages",
    )
    turning_backfill.add_argument("database", type=Path)
    turning_backfill.add_argument("--start", required=True)
    turning_backfill.add_argument("--end", required=True)
    imported = commands.add_parser(
        "import-hy-oas", help="import licensed point-in-time HY OAS CSV"
    )
    imported.add_argument("database", type=Path)
    imported.add_argument("csv", type=Path)
    imported.add_argument("--source", required=True)
    revised = commands.add_parser(
        "import-hy-oas-revised-history",
        help="import current FRED HY OAS history for score comparison",
    )
    revised.add_argument("database", type=Path)
    revised.add_argument("csv", type=Path)
    usdjpy = commands.add_parser(
        "import-usdjpy-revised-history",
        help="import FRED DEXJPUS history for the USD/JPY Fast condition",
    )
    usdjpy.add_argument("database", type=Path)
    usdjpy.add_argument("csv", type=Path)
    usdjpy_fetch = commands.add_parser(
        "fetch-usdjpy", help="fetch current official FRED DEXJPUS history"
    )
    usdjpy_fetch.add_argument("database", type=Path)
    usdjpy_fetch.add_argument("--start", required=True)
    usdjpy_fetch.add_argument("--end", required=True)
    args = parser.parse_args(argv)

    if args.command == "fetch-hy-oas":
        count = fetch_hy_oas(args.database, args.start, args.end)
        print(f"upserted {count} HY_OAS observations into {args.database}")
    elif args.command == "fetch-cash-rate":
        count = fetch_cash_rate(args.database, args.start, args.end)
        print(f"upserted {count} CASH_3M observations into {args.database}")
    elif args.command == "backfill-hy-oas-vintages":
        count = backfill_hy_oas_vintages(
            args.database,
            args.start,
            args.end,
            api_key=os.environ.get("FRED_API_KEY", ""),
            realtime_end=args.realtime_end,
        )
        print(f"backfilled {count} HY_OAS vintages into {args.database}")
    elif args.command == "backfill-hy-oas-alfred":
        count = backfill_hy_oas_alfred(args.database, args.start, args.end)
        print(f"backfilled {count} ALFRED HY_OAS vintages into {args.database}")
    elif args.command == "backfill-cash-rate-alfred":
        count = backfill_cash_rate_alfred(args.database, args.start, args.end)
        print(f"backfilled {count} ALFRED CASH_3M vintages into {args.database}")
    elif args.command == "refresh-turning-macro":
        counts, used_stored = refresh_turning_macro_inputs(args.database, args.end)
        print(json.dumps({"counts": counts, "used_stored": used_stored}, sort_keys=True))
    elif args.command == "backfill-turning-macro-alfred":
        counts = backfill_turning_macro_alfred(
            args.database, args.start, args.end
        )
        print(json.dumps({"counts": counts}, sort_keys=True))
    elif args.command == "import-hy-oas":
        count = import_hy_oas_csv(args.database, args.csv, source=args.source)
        print(f"imported {count} HY_OAS vintages into {args.database}")
    elif args.command == "import-hy-oas-revised-history":
        count = import_hy_oas_revised_history_csv(args.database, args.csv)
        print(f"imported {count} revised HY_OAS observations into {args.database}")
    elif args.command == "import-usdjpy-revised-history":
        count = import_usdjpy_revised_history_csv(args.database, args.csv)
        print(f"imported {count} revised USDJPY_FRED observations into {args.database}")
    else:
        count = fetch_usdjpy_revised_history(args.database, args.start, args.end)
        print(f"fetched {count} revised USDJPY_FRED observations into {args.database}")


if __name__ == "__main__":
    main()
