from __future__ import annotations

import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from roro_shadow import SERIES_VERSION as RORO_SERIES_VERSION
from sector_rotation import SECTOR_ETFS
from sector_roro_shadow import MODEL_VERSION
from sector_shadow_store import (
    DEFAULT_TARGET,
    PAYLOAD_FIELDS,
    SCHEMA_COLUMNS,
    SERIES_VERSION,
    TABLE,
    ImmutableSectorSeriesConflict,
    _load_price_matrix_read_only,
    _decision_dates,
    build_sector_shadow_series,
    latest_sector_snapshot,
    main,
    persist_sector_snapshots,
    read_sector_history,
)


def _snapshot(day: str, *, shift: float = 0.0) -> tuple[pd.DataFrame, dict[str, object]]:
    rows = []
    for rank, (ticker, name) in enumerate(SECTOR_ETFS.items(), 1):
        predicted = 0.02 - rank * 0.002 + shift
        rows.append(
            {
                "date": pd.Timestamp(day),
                "ticker": ticker,
                "sector": name,
                "shadow_score": 0.9 - rank * 0.03,
                "predicted_excess_20d": predicted,
                "forecast_score": (12 - rank) / 11,
                "forecast_rank": rank,
                "signal": "MODEL_TOP3" if rank <= 3 else "NEGATIVE",
                "quadrant": "LEADING" if rank <= 3 else "LAGGING",
                "quadrant_transition": "IMPROVING->LEADING",
                "tail_heading": "NE_STRENGTHENING" if rank <= 3 else "SW_DETERIORATING",
                "delta_rs_ratio": 1.0 - rank / 10,
                "delta_rs_momentum": 0.5 - rank / 20,
                "sector_beta_60": 0.5 + rank / 10,
                "beta_rank": (rank - 0.5) / 11,
                "roro_fit": rank / 11,
                "alpha_positive": predicted > 0,
                "alpha_top3": rank <= 3,
                "rrg_supportive": rank <= 3,
                "roro_context": "FIT" if rank >= 6 else "RORO_MISMATCH",
                "alignment": "ALIGNED",
                "sector_status": "SELECT" if rank <= 3 else "WATCH",
                "market_risk_score": 35,
                "risk_regime": "YELLOW",
                "roro_trajectory_5d": -5,
                "roro_data_status": "CONDITIONAL_EVENT_PIT",
                "holding_name": "SECRET MUST NOT PERSIST",
            }
        )
    metadata = {
        "as_of": day,
        "model_version": MODEL_VERSION,
        "roro_series_version": RORO_SERIES_VERSION,
        "target": "next-20-trading-day sector return minus equal-sector return",
        "training_rows": 2000,
        "training_weeks": 156,
        "training_start": "2017-01-06",
        "training_end": "2025-12-05",
        "latest_training_label_end": "2025-12-31",
        "alpha": 10.0,
        "top_k": 3,
        "context_type": "RORO_EXACT_DATE_BETA_FIT",
        "status_contract": "SELECT/WATCH/AVOID; no forced Top3 fill",
    }
    return pd.DataFrame(rows), metadata


class SectorShadowStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.target = self.root / "outputs" / "sector.db"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_exact_allowlist_canonical_hash_idempotency_and_readers(self) -> None:
        frame, metadata = _snapshot("2026-01-07")
        first = persist_sector_snapshots(self.target, [(frame, metadata)])
        created = latest_sector_snapshot(self.target)[0]["created_at"]
        second = persist_sector_snapshots(
            self.target,
            [(frame.sample(frac=1, random_state=7), metadata)],
        )

        self.assertEqual(first["inserted"], 11)
        self.assertEqual(second["inserted"], 0)
        self.assertEqual(second["idempotent"], 11)
        self.assertEqual(latest_sector_snapshot(self.target)[0]["created_at"], created)
        with closing(sqlite3.connect(self.target)) as connection:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                )
            }
            columns = tuple(row[1] for row in connection.execute(f"PRAGMA table_info({TABLE})"))
            payloads = [row[0] for row in connection.execute(f"SELECT payload_json FROM {TABLE}")]
        self.assertEqual(tables, {TABLE})
        self.assertEqual(DEFAULT_TARGET, "outputs/sector_shadow/sector_shadow.db")
        self.assertEqual(columns, SCHEMA_COLUMNS)
        self.assertFalse(set(columns) & {"ticker", "holding_name", "quantity", "weight"})
        self.assertNotIn("SECRET", "".join(payloads))
        self.assertEqual(set(json.loads(payloads[0])), set(PAYLOAD_FIELDS))

        history = read_sector_history(
            self.target, start="2026-01-07", end="2026-01-07", sector="XLK"
        )
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["sector"], "XLK")
        self.assertEqual(history[0]["input_cutoff"], "2026-01-07")
        self.assertEqual(len(history[0]["provenance_hash"]), 64)
        self.assertEqual(len(latest_sector_snapshot(self.target)), 11)

        other = self.root / "other.db"
        reversed_frame = frame.iloc[::-1, ::-1]
        persist_sector_snapshots(other, [(reversed_frame, metadata)])
        self.assertEqual(
            read_sector_history(self.target, sector="XLK")[0]["provenance_hash"],
            read_sector_history(other, sector="XLK")[0]["provenance_hash"],
        )

    def test_rows_are_write_protected_and_tamper_evident(self) -> None:
        persist_sector_snapshots(self.target, [_snapshot("2026-01-07")])
        with closing(sqlite3.connect(self.target)) as connection:
            triggers = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='trigger'"
                )
            }
            self.assertEqual(
                triggers, {f"{TABLE}_no_update", f"{TABLE}_no_delete"}
            )
            with self.assertRaises(sqlite3.DatabaseError):
                connection.execute(f"DELETE FROM {TABLE} WHERE sector='XLK'")
            connection.rollback()
            connection.execute(f"DROP TRIGGER {TABLE}_no_update")
            payload = json.loads(
                connection.execute(
                    f"SELECT payload_json FROM {TABLE} WHERE sector='XLK'"
                ).fetchone()[0]
            )
            payload["predicted_excess_20d"] = 999.0
            connection.execute(
                f"UPDATE {TABLE} SET payload_json=? WHERE sector='XLK'",
                (json.dumps(payload, sort_keys=True, separators=(",", ":")),),
            )
            connection.commit()
        with self.assertRaisesRegex(ValueError, "provenance hash mismatch"):
            read_sector_history(self.target)

    def test_conflict_rolls_back_rows_inserted_earlier_in_batch(self) -> None:
        day1 = _snapshot("2026-01-07")
        day3 = _snapshot("2026-01-09")
        persist_sector_snapshots(self.target, [day1, day3])

        with self.assertRaisesRegex(ImmutableSectorSeriesConflict, "2026-01-09"):
            persist_sector_snapshots(
                self.target,
                [day1, _snapshot("2026-01-08"), _snapshot("2026-01-09", shift=0.01)],
            )
        history = read_sector_history(self.target)
        self.assertEqual(sorted({row["date"] for row in history}), ["2026-01-07", "2026-01-09"])
        self.assertEqual(len(history), 22)

    def test_cutoff_complete_sector_set_and_causal_metadata_are_enforced(self) -> None:
        frame, metadata = _snapshot("2026-01-06")
        with self.assertRaisesRegex(ValueError, "starts on 2026-01-07"):
            persist_sector_snapshots(self.target, [(frame, metadata)])

        frame, metadata = _snapshot("2026-01-07")
        with self.assertRaisesRegex(ValueError, "11 sector ETFs"):
            persist_sector_snapshots(self.target, [(frame.iloc[:-1], metadata)])
        future = {**metadata, "latest_training_label_end": "2026-01-07"}
        with self.assertRaisesRegex(ValueError, "known before"):
            persist_sector_snapshots(self.target, [(frame, future)])

        missing = self.root / "missing.db"
        with self.assertRaises(FileNotFoundError):
            read_sector_history(missing)
        self.assertFalse(missing.exists())

        publish, roro_source = self.root / "publish.db", self.root / "roro.db"
        publish.touch()
        roro_source.touch()
        with self.assertRaisesRegex(ValueError, "separate"):
            build_sector_shadow_series(
                publish,
                roro_source,
                publish,
                as_of="2026-01-07",
            )

    def test_price_loader_opens_publish_database_in_read_only_uri_mode(self) -> None:
        source = self.root / "publish.db"
        with closing(sqlite3.connect(source)) as connection:
            connection.execute(
                "CREATE TABLE daily_prices(ticker TEXT,date TEXT,close REAL,adj_close REAL)"
            )
            connection.executemany(
                "INSERT INTO daily_prices VALUES (?,?,?,?)",
                [
                    (ticker, "2026-01-07", 100.0, 100.0)
                    for ticker in ("SPY", *SECTOR_ETFS)
                ],
            )
            connection.commit()
        real_connect = sqlite3.connect
        calls: list[tuple[object, dict[str, object]]] = []

        def capture(database: object, *args: object, **kwargs: object) -> sqlite3.Connection:
            calls.append((database, kwargs))
            return real_connect(database, *args, **kwargs)

        with patch("sector_shadow_store.sqlite3.connect", side_effect=capture):
            matrix = _load_price_matrix_read_only(source)
        self.assertEqual(set(matrix.columns), {"SPY", *SECTOR_ETFS})
        self.assertIn("?mode=ro", str(calls[0][0]))
        self.assertTrue(calls[0][1]["uri"])

    def test_build_reuses_causal_fit_for_start_anchor_and_weekly_dates(self) -> None:
        publish, roro_source = self.root / "publish.db", self.root / "roro.db"
        publish.touch()
        roro_source.touch()
        dates = pd.to_datetime(["2026-01-07", "2026-01-09", "2026-01-12"])
        features = pd.DataFrame({"date": dates})
        roro = pd.DataFrame({"date": dates})

        def fake_fit(*args: object, as_of: object, **kwargs: object):
            return _snapshot(pd.Timestamp(as_of).date().isoformat())

        with (
            patch(
                "sector_shadow_store._load_inputs_read_only",
                return_value=(pd.DataFrame(), pd.DataFrame(), features, roro),
            ),
            patch("sector_shadow_store.fit_sector_roro_shadow", side_effect=fake_fit) as fit,
        ):
            result = build_sector_shadow_series(
                publish,
                roro_source,
                self.target,
                as_of="2026-01-12",
            )
        self.assertEqual(fit.call_count, 2)
        self.assertEqual(result["dates_seen"], 2)
        self.assertEqual(result["inserted"], 22)
        self.assertEqual(result["source_database_mode"], "mode=ro; query_only=ON")

    def test_decision_dates_exclude_incomplete_week_and_require_exact_roro(self) -> None:
        dates = pd.to_datetime(["2026-01-07", "2026-01-09", "2026-01-12"])
        features = pd.DataFrame({"date": dates})
        roro = pd.DataFrame({"date": dates})
        expected = [pd.Timestamp("2026-01-07"), pd.Timestamp("2026-01-09")]
        self.assertEqual(
            _decision_dates(features, roro, "2026-01-12", latest_only=False),
            expected,
        )
        self.assertEqual(
            _decision_dates(features, roro, "2026-01-12", latest_only=True),
            [pd.Timestamp("2026-01-09")],
        )
        with self.assertRaisesRegex(ValueError, "exact RORO context.*2026-01-09"):
            _decision_dates(
                features,
                pd.DataFrame({"date": dates.delete(1)}),
                "2026-01-12",
                latest_only=False,
            )
        recent_roro = pd.DataFrame({"date": dates[1:]})
        self.assertEqual(
            _decision_dates(features, recent_roro, "2026-01-12", latest_only=True),
            [pd.Timestamp("2026-01-09")],
        )
        with self.assertRaisesRegex(ValueError, "exact RORO context.*2026-01-07"):
            _decision_dates(features, recent_roro, "2026-01-12", latest_only=False)
        with self.assertRaisesRegex(ValueError, "exact RORO context.*2026-01-09"):
            _decision_dates(features, recent_roro.iloc[1:], "2026-01-12", latest_only=True)

    def test_cli_build_prints_canonical_result(self) -> None:
        expected = {
            "series_version": SERIES_VERSION,
            "inserted": 11,
            "target": str(self.target),
        }
        output = io.StringIO()
        with (
            patch("sector_shadow_store.build_sector_shadow_series", return_value=expected) as build,
            redirect_stdout(output),
        ):
            main(
                [
                    "build",
                    "publish.db",
                    "roro.db",
                    str(self.target),
                    "--as-of",
                    "2026-01-07",
                    "--latest-only",
                ]
            )
        self.assertEqual(json.loads(output.getvalue()), expected)
        self.assertTrue(build.call_args.kwargs["latest_only"])


if __name__ == "__main__":
    unittest.main()
