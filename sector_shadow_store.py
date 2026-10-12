"""Immutable SQLite store for causal Sector RORO SHADOW decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from collections.abc import Mapping, Sequence
from contextlib import closing
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from roro_shadow import SERIES_VERSION as RORO_SERIES_VERSION
from roro_shadow import _calendar_date, _canonical_json
from sector_rotation import SECTOR_ETFS, build_rrg_signals
from sector_rotation_forecast import build_sector_feature_frame
from sector_roro_shadow import (
    MODEL_VERSION,
    OUTPUT_START,
    fit_sector_roro_shadow,
    _validate_frozen_spec,
    roro_context_from_history,
)


TABLE = "sector_roro_shadow_observations"
SERIES_VERSION = f"{MODEL_VERSION}__{RORO_SERIES_VERSION}__immutable-r1"
DEFAULT_TARGET = "outputs/sector_shadow/sector_shadow.db"

ROW_PAYLOAD_MAP = (
    ("sector_name", "sector"),
    ("rrg_score", "shadow_score"),
    ("predicted_excess_20d", "predicted_excess_20d"),
    ("forecast_score", "forecast_score"),
    ("forecast_rank", "forecast_rank"),
    ("signal", "signal"),
    ("quadrant", "quadrant"),
    ("quadrant_transition", "quadrant_transition"),
    ("tail_heading", "tail_heading"),
    ("delta_rs_ratio", "delta_rs_ratio"),
    ("delta_rs_momentum", "delta_rs_momentum"),
    ("sector_beta_60", "sector_beta_60"),
    ("beta_rank", "beta_rank"),
    ("roro_fit", "roro_fit"),
    ("alpha_positive", "alpha_positive"),
    ("alpha_top3", "alpha_top3"),
    ("rrg_supportive", "rrg_supportive"),
    ("roro_context", "roro_context"),
    ("alignment", "alignment"),
    ("sector_status", "sector_status"),
    ("market_risk_score", "market_risk_score"),
    ("risk_regime", "risk_regime"),
    ("roro_trajectory_5d", "roro_trajectory_5d"),
    ("roro_data_status", "roro_data_status"),
)
METADATA_PAYLOAD_MAP = (
    ("forecast_target", "target"),
    ("training_rows", "training_rows"),
    ("training_weeks", "training_weeks"),
    ("training_start", "training_start"),
    ("training_end", "training_end"),
    ("latest_training_label_end", "latest_training_label_end"),
    ("alpha", "alpha"),
    ("top_k", "top_k"),
    ("context_type", "context_type"),
    ("status_contract", "status_contract"),
)
PAYLOAD_FIELDS = tuple(key for key, _ in (*ROW_PAYLOAD_MAP, *METADATA_PAYLOAD_MAP))
SCHEMA_COLUMNS = (
    "series_version",
    "date",
    "sector",
    "input_cutoff",
    "model_version",
    "roro_series_version",
    "provenance_hash",
    "payload_json",
    "created_at",
)


class ImmutableSectorSeriesConflict(RuntimeError):
    """Raised when an existing series/date/sector payload would be rewritten."""


def _same_file(left: Path, right: Path) -> bool:
    if os.path.normcase(str(left.resolve())) == os.path.normcase(str(right.resolve())):
        return True
    return left.exists() and right.exists() and os.path.samefile(left, right)


def _validate_paths(
    publish_source: str | Path,
    roro_source: str | Path,
    shadow_target: str | Path,
) -> tuple[Path, Path, Path]:
    publish, roro, target = map(Path, (publish_source, roro_source, shadow_target))
    for source in (publish, roro):
        if not source.is_file():
            raise FileNotFoundError(source)
        if _same_file(source, target):
            raise ValueError("shadow_target must be separate from every source database")
    if target.exists() and not target.is_file():
        raise ValueError("shadow_target must be a SQLite file path")
    return publish, roro, target


def _provenance(
    series_version: str,
    day: str,
    sector: str,
    payload_json: str,
) -> str:
    material = {
        "series_version": series_version,
        "date": day,
        "sector": sector,
        "input_cutoff": day,
        "model_versions": {
            "sector": MODEL_VERSION,
            "roro": RORO_SERIES_VERSION,
        },
        "payload": json.loads(payload_json),
    }
    return hashlib.sha256(_canonical_json(material).encode("utf-8")).hexdigest()


def _snapshot_records(
    scored: pd.DataFrame,
    metadata: Mapping[str, Any],
    *,
    series_version: str,
) -> list[tuple[str, ...]]:
    required = {"date", "ticker", *(source for _, source in ROW_PAYLOAD_MAP)}
    missing = sorted(required - set(scored.columns))
    missing_metadata = sorted(
        {"as_of", "model_version", "roro_series_version", *(source for _, source in METADATA_PAYLOAD_MAP)}
        - set(metadata)
    )
    if missing or missing_metadata:
        details = [*(f"row:{name}" for name in missing), *(f"metadata:{name}" for name in missing_metadata)]
        raise ValueError("missing Sector SHADOW fields: " + ", ".join(details))

    day = _calendar_date(metadata["as_of"], "as_of")
    if pd.Timestamp(day) < OUTPUT_START:
        raise ValueError(f"Sector SHADOW starts on {OUTPUT_START.date().isoformat()}")
    if metadata["model_version"] != MODEL_VERSION:
        raise ValueError("unexpected Sector RORO model version")
    if metadata["roro_series_version"] != RORO_SERIES_VERSION:
        raise ValueError("unexpected RORO context series version")
    row_days = {_calendar_date(value, "row date") for value in scored["date"]}
    if row_days != {day}:
        raise ValueError("every sector row must use the exact decision date")
    sectors = scored["ticker"].astype(str)
    if sectors.duplicated().any() or set(sectors) != set(SECTOR_ETFS):
        raise ValueError("one complete row for each of the 11 sector ETFs is required")
    if _calendar_date(metadata["training_end"], "training_end") >= day:
        raise ValueError("training_end must precede the decision date")
    if _calendar_date(metadata["latest_training_label_end"], "latest_training_label_end") >= day:
        raise ValueError("latest training label must be known before the decision date")

    records = []
    for _, row in scored.assign(ticker=sectors).sort_values("ticker").iterrows():
        sector = str(row["ticker"])
        payload = {target: row[source] for target, source in ROW_PAYLOAD_MAP}
        payload.update({target: metadata[source] for target, source in METADATA_PAYLOAD_MAP})
        payload_json = _canonical_json(payload)
        if tuple(sorted(json.loads(payload_json))) != tuple(sorted(PAYLOAD_FIELDS)):
            raise ValueError("Sector SHADOW payload does not match the aggregate allowlist")
        records.append(
            (
                series_version,
                day,
                sector,
                day,
                MODEL_VERSION,
                RORO_SERIES_VERSION,
                _provenance(series_version, day, sector, payload_json),
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
        raise ValueError("Sector SHADOW target must contain only its aggregate table")
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE}(
            series_version TEXT NOT NULL,
            date TEXT NOT NULL,
            sector TEXT NOT NULL,
            input_cutoff TEXT NOT NULL,
            model_version TEXT NOT NULL,
            roro_series_version TEXT NOT NULL,
            provenance_hash TEXT NOT NULL CHECK(length(provenance_hash)=64),
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY(series_version,date,sector)
        ) WITHOUT ROWID
        """
    )
    actual = tuple(row[1] for row in connection.execute(f"PRAGMA table_info({TABLE})"))
    if actual != SCHEMA_COLUMNS:
        raise ValueError(f"incompatible {TABLE} schema")
    connection.execute(
        f"CREATE TRIGGER IF NOT EXISTS {TABLE}_no_update "
        f"BEFORE UPDATE ON {TABLE} BEGIN SELECT RAISE(ABORT,'immutable Sector SHADOW series'); END"
    )
    connection.execute(
        f"CREATE TRIGGER IF NOT EXISTS {TABLE}_no_delete "
        f"BEFORE DELETE ON {TABLE} BEGIN SELECT RAISE(ABORT,'immutable Sector SHADOW series'); END"
    )


def persist_sector_snapshots(
    shadow_target: str | Path,
    snapshots: Sequence[tuple[pd.DataFrame, Mapping[str, Any]]],
    *,
    series_version: str = SERIES_VERSION,
) -> dict[str, Any]:
    """Atomically insert immutable aggregate rows; identical rows are idempotent."""
    records_by_key: dict[tuple[str, str, str], tuple[str, ...]] = {}
    for scored, metadata in snapshots:
        for record in _snapshot_records(scored, metadata, series_version=series_version):
            key = record[:3]
            if key in records_by_key and records_by_key[key] != record:
                raise ImmutableSectorSeriesConflict(
                    f"duplicate Sector SHADOW conflict: {' '.join(key)}"
                )
            records_by_key[key] = record
    records = sorted(records_by_key.values(), key=lambda row: (row[1], row[2]))
    target = Path(shadow_target)
    target.parent.mkdir(parents=True, exist_ok=True)
    inserted = idempotent = 0
    with closing(sqlite3.connect(target)) as connection:
        try:
            connection.execute("BEGIN IMMEDIATE")
            _initialize(connection)
            for record in records:
                key = record[:3]
                existing = connection.execute(
                    f"SELECT {','.join(SCHEMA_COLUMNS[:-1])} FROM {TABLE} "
                    "WHERE series_version=? AND date=? AND sector=?",
                    key,
                ).fetchone()
                if existing is not None:
                    if tuple(existing) != record:
                        raise ImmutableSectorSeriesConflict(
                            f"immutable Sector SHADOW conflict: {' '.join(key)}"
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
    dates = sorted({record[1] for record in records})
    return {
        "series_version": series_version,
        "start": dates[0] if dates else None,
        "as_of": dates[-1] if dates else None,
        "dates_seen": len(dates),
        "rows_seen": len(records),
        "inserted": inserted,
        "idempotent": idempotent,
    }


def _read_only(database: str | Path) -> sqlite3.Connection:
    path = Path(database)
    if not path.is_file():
        raise FileNotFoundError(path)
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    return connection


def read_sector_history(
    shadow_target: str | Path,
    *,
    series_version: str = SERIES_VERSION,
    start: str | date | pd.Timestamp | None = None,
    end: str | date | pd.Timestamp | None = None,
    sector: str | None = None,
) -> list[dict[str, Any]]:
    clauses, parameters = ["series_version=?"], [series_version]
    if start is not None:
        clauses.append("date>=?")
        parameters.append(_calendar_date(start, "start"))
    if end is not None:
        clauses.append("date<=?")
        parameters.append(_calendar_date(end, "end"))
    if sector is not None:
        if sector not in SECTOR_ETFS:
            raise ValueError(f"unknown sector ETF: {sector}")
        clauses.append("sector=?")
        parameters.append(sector)
    with closing(_read_only(shadow_target)) as connection:
        rows = connection.execute(
            f"SELECT {','.join(SCHEMA_COLUMNS)} FROM {TABLE} "
            f"WHERE {' AND '.join(clauses)} ORDER BY date,sector",
            parameters,
        ).fetchall()
    history = []
    for row in rows:
        observation = dict(zip(SCHEMA_COLUMNS, row))
        payload_json = observation.pop("payload_json")
        payload = json.loads(payload_json)
        if set(payload) != set(PAYLOAD_FIELDS):
            raise ValueError("stored Sector SHADOW payload violates the aggregate allowlist")
        if observation["input_cutoff"] != observation["date"]:
            raise ValueError("stored Sector SHADOW input cutoff does not match its date")
        if observation["model_version"] != MODEL_VERSION:
            raise ValueError("stored Sector SHADOW model version is unexpected")
        if observation["roro_series_version"] != RORO_SERIES_VERSION:
            raise ValueError("stored Sector SHADOW RORO version is unexpected")
        if observation["provenance_hash"] != _provenance(
            observation["series_version"],
            observation["date"],
            observation["sector"],
            payload_json,
        ):
            raise ValueError("stored Sector SHADOW provenance hash mismatch")
        observation["payload"] = payload
        history.append(observation)
    return history


def latest_sector_snapshot(
    shadow_target: str | Path,
    *,
    series_version: str = SERIES_VERSION,
) -> list[dict[str, Any]]:
    history = read_sector_history(shadow_target, series_version=series_version)
    if not history:
        return []
    latest = history[-1]["date"]
    return [row for row in history if row["date"] == latest]


def _load_price_matrix_read_only(
    database: str | Path,
    benchmark: str = "SPY",
) -> pd.DataFrame:
    tickers = [benchmark, *SECTOR_ETFS]
    placeholders = ",".join("?" for _ in tickers)
    with closing(_read_only(database)) as connection:
        prices = pd.read_sql_query(
            f"""
            SELECT ticker,date,COALESCE(adj_close,close) AS price
            FROM daily_prices
            WHERE ticker IN ({placeholders})
            ORDER BY date,ticker
            """,
            connection,
            params=tickers,
            parse_dates=["date"],
        )
    if prices.empty:
        raise ValueError("sector prices are missing")
    matrix = prices.pivot(index="date", columns="ticker", values="price").sort_index()
    missing = [ticker for ticker in tickers if ticker not in matrix]
    if missing:
        raise ValueError("missing price history: " + ", ".join(missing))
    return matrix


def _load_inputs_read_only(
    publish_source: str | Path,
    roro_source: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prices = _load_price_matrix_read_only(publish_source)
    signals = build_rrg_signals(prices)
    features = build_sector_feature_frame(prices, signals, horizon=20)
    roro = roro_context_from_history(roro_source)
    return prices, signals, features, roro


def _decision_dates(
    features: pd.DataFrame,
    roro: pd.DataFrame,
    cutoff: str,
    *,
    latest_only: bool,
) -> list[pd.Timestamp]:
    feature_dates = pd.DatetimeIndex(pd.to_datetime(features["date"]).dropna().unique()).normalize()
    roro_dates = set(pd.to_datetime(roro["date"]).dt.normalize())
    cutoff_date = pd.Timestamp(cutoff).normalize()
    eligible = feature_dates[
        (feature_dates >= OUTPUT_START) & (feature_dates <= cutoff_date)
    ]
    if eligible.empty:
        raise ValueError("no Sector decision date is available")
    if OUTPUT_START not in eligible:
        raise ValueError("Sector decision calendar is missing the 2026-01-07 anchor")
    weekly = [
        pd.Timestamp(day)
        for period, day in pd.Series(eligible).groupby(eligible.to_period("W-FRI")).max().items()
        if period.end_time.normalize() <= cutoff_date
    ]
    first = OUTPUT_START
    decisions = sorted(set([first, *weekly]))
    if latest_only:
        decisions = decisions[-1:]
    missing_roro = [day.date().isoformat() for day in decisions if day not in roro_dates]
    if missing_roro:
        raise ValueError(
            "exact RORO context is unavailable for Sector decision dates: "
            + ", ".join(missing_roro)
        )
    return decisions


def build_sector_shadow_series(
    publish_source: str | Path,
    roro_source: str | Path,
    shadow_target: str | Path,
    *,
    as_of: str | date | pd.Timestamp,
    latest_only: bool = False,
    min_train_weeks: int = 156,
    alpha: float = 10.0,
    top_k: int = 3,
) -> dict[str, Any]:
    """Build weekly or latest decisions from read-only sources into a separate DB."""
    _validate_frozen_spec(min_train_weeks, alpha, top_k)
    cutoff = _calendar_date(as_of, "as_of")
    if pd.Timestamp(cutoff) < OUTPUT_START:
        raise ValueError(f"Sector SHADOW starts on {OUTPUT_START.date().isoformat()}")
    publish, roro_database, target = _validate_paths(
        publish_source, roro_source, shadow_target
    )
    prices, signals, features, roro = _load_inputs_read_only(publish, roro_database)
    snapshots = []
    for decision_date in _decision_dates(
        features, roro, cutoff, latest_only=latest_only
    ):
        snapshots.append(
            fit_sector_roro_shadow(
                features,
                prices,
                signals,
                roro,
                as_of=decision_date,
                min_train_weeks=min_train_weeks,
                alpha=alpha,
                top_k=top_k,
            )
        )
    result = persist_sector_snapshots(target, snapshots)
    result.update(
        {
            "requested_as_of": cutoff,
            "latest_only": latest_only,
            "target": str(target.resolve()),
            "source_database_mode": "mode=ro; query_only=ON",
        }
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="build the isolated Sector SHADOW series")
    build.add_argument("publish_source")
    build.add_argument("roro_source")
    build.add_argument("shadow_target", nargs="?", default=DEFAULT_TARGET)
    build.add_argument("--as-of", required=True)
    build.add_argument("--latest-only", action="store_true")
    build.add_argument("--min-train-weeks", type=int, default=156)
    build.add_argument("--alpha", type=float, default=10.0)
    build.add_argument("--top-k", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    result = build_sector_shadow_series(
        args.publish_source,
        args.roro_source,
        args.shadow_target,
        as_of=args.as_of,
        latest_only=args.latest_only,
        min_train_weeks=args.min_train_weeks,
        alpha=args.alpha,
        top_k=args.top_k,
    )
    print(_canonical_json(result))


if __name__ == "__main__":
    main()
