import hashlib
import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from roro_market_breadth import (
    CONTRACT_VERSION,
    SOURCE,
    TABLE,
    breadth_readiness,
    import_ndth_csv,
    load_ndth_series,
    main,
    persist_breadth_observations,
    refresh_historyofmarket_ndth,
)


class MarketBreadthTest(unittest.TestCase):
    @staticmethod
    def _record(day: str, value: float, available: str | None = None) -> dict[str, object]:
        return {
            "contract_version": CONTRACT_VERSION, "source": SOURCE, "symbol": "NDTH",
            "observation_date": day, "available_at": available or day,
            "value_pct": value, "fetched_at": "2026-08-14",
        }

    def test_next_session_no_fill_and_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "ndth.db"
            persist_breadth_observations(database, [self._record("2026-01-05", 65)])
            before = hashlib.sha256(database.read_bytes()).digest()
            series = load_ndth_series(database, ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"])
            after = hashlib.sha256(database.read_bytes()).digest()
        self.assertTrue(pd.isna(series.loc["2026-01-05"]))
        self.assertEqual(series.loc["2026-01-06"], 0.65)
        self.assertTrue(pd.isna(series.loc["2026-01-07"]))
        self.assertEqual(before, after)

    def test_future_available_vintage_does_not_rewrite_prior_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "ndth.db"
            persist_breadth_observations(database, [
                self._record("2026-01-05", 60),
                self._record("2026-01-05", 80, "2026-01-06"),
            ])
            series = load_ndth_series(database, ["2026-01-05", "2026-01-06", "2026-01-07"])
        self.assertEqual(series.loc["2026-01-06"], 0.60)
        self.assertEqual(series.loc["2026-01-07"], 0.80)

    def test_csv_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "ndth.csv"
            source.write_text("date,value_pct\n2026-01-05,55.5\n", encoding="utf-8")
            database = Path(directory) / "ndth.db"
            result = import_ndth_csv(database, source, fetched_at="2026-08-14")
        self.assertEqual(result["inserted"], 1)

    def test_readiness_uses_actual_sessions_and_preserves_database(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "ndth.db"
            persist_breadth_observations(database, [self._record("2026-01-16", 65)])
            before = database.read_bytes()
            # The supplied NQ calendar has a weekend and holiday before Tuesday.
            status = breadth_readiness(database, ["2026-01-15", "2026-01-16", "2026-01-20"], "2026-01-20")
            self.assertEqual(database.read_bytes(), before)
        self.assertEqual(status["status"], "READY")
        self.assertEqual(status["required_observation_date"], "2026-01-16")
        self.assertEqual(status["latest_observation_date"], "2026-01-16")
        self.assertEqual(status["latest_usable_date"], "2026-01-20")

    def test_readiness_detects_gaps_and_late_availability_without_fill(self) -> None:
        sessions = ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"]
        cases = (
            # A current/future observation does not replace a missing prior day.
            ([self._record("2026-01-02", 60), self._record("2026-01-06", 70)], "2026-01-06"),
            # The loader has a value, but it belongs to an older observation.
            ([self._record("2026-01-02", 60, "2026-01-05")], "2026-01-02"),
            # Availability on the target day is only usable the following session.
            ([self._record("2026-01-05", 60, "2026-01-06")], "2026-01-05"),
        )
        for records, latest in cases:
            with self.subTest(records=records), tempfile.TemporaryDirectory() as directory:
                database = Path(directory) / "ndth.db"
                persist_breadth_observations(database, records)
                status = breadth_readiness(database, sessions, "2026-01-06")
                self.assertEqual(status["status"], "DELAYED")
                self.assertEqual(status["required_observation_date"], "2026-01-05")
                self.assertEqual(status["latest_observation_date"], latest)

    def test_readiness_ignores_future_observations_and_reports_missing(self) -> None:
        sessions = ["2026-01-05", "2026-01-06", "2026-01-07"]
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "ndth.db"
            self.assertEqual(breadth_readiness(database, sessions, "2026-01-06")["status"], "MISSING")
            self.assertFalse(database.exists())
            persist_breadth_observations(database, [self._record("2026-01-07", 60)])
            status = breadth_readiness(database, sessions, "2026-01-06")
            self.assertEqual(status["status"], "MISSING")
            self.assertIsNone(status["latest_observation_date"])
            self.assertIsNone(status["latest_usable_date"])
            persist_breadth_observations(database, [self._record("2026-01-05", 65)])
            status = breadth_readiness(database, sessions, "2026-01-06")
            self.assertEqual(status["status"], "READY")
            self.assertEqual(status["latest_observation_date"], "2026-01-05")
            self.assertEqual(status["latest_usable_date"], "2026-01-06")
            self.assertEqual(breadth_readiness(database, sessions, "2026-01-10")["status"], "MISSING")

    def test_cli_requires_ready_breadth_even_when_refresh_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "ndth.db"
            publish = Path(directory) / "publish.db"
            persist_breadth_observations(database, [self._record("2026-01-02", 60)])
            with closing(sqlite3.connect(publish)) as connection:
                connection.execute("CREATE TABLE daily_prices(ticker TEXT, date TEXT)")
                connection.executemany("INSERT INTO daily_prices VALUES ('NQ=F',?)", [(day,) for day in ["2026-01-02", "2026-01-05", "2026-01-06"]])
                connection.commit()
            publish_before = publish.read_bytes()
            args = ["roro_market_breadth.py", "refresh", str(database), "--as-of", "2026-01-08", "--required-as-of", "2026-01-06", "--publish-database", str(publish)]
            for expected in ("DELAYED", "READY"):
                with self.subTest(expected=expected), patch("sys.argv", args), patch(
                    "roro_market_breadth.refresh_historyofmarket_ndth", return_value={"inserted": 0}
                ) as refresh, redirect_stdout(io.StringIO()) as output:
                    if expected == "DELAYED":
                        with self.assertRaises(SystemExit) as error:
                            main()
                        self.assertEqual(error.exception.code, 3)
                    else:
                        main()
                    self.assertEqual(json.loads(output.getvalue())["readiness"]["status"], expected)
                    refresh.assert_called_once_with(database, as_of="2026-01-08")
                persist_breadth_observations(database, [self._record("2026-01-05", 65)])
            self.assertEqual(publish.read_bytes(), publish_before)

    def test_cli_incomplete_readiness_flags_fail_before_fetch(self) -> None:
        for flags in (["--required-as-of", "2026-01-06"], ["--publish-database", "unused.db"]):
            with self.subTest(flags=flags), patch("sys.argv", ["breadth", "refresh", "unused.db", *flags]), patch(
                "roro_market_breadth.refresh_historyofmarket_ndth"
            ) as refresh, redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main()
            self.assertEqual(error.exception.code, 2)
            refresh.assert_not_called()

    def test_refresh_appends_missing_keys_without_revising_saved_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "ndth.db"
            original = [self._record("2026-01-02", 60), self._record("2026-01-06", 70)]
            persist_breadth_observations(database, original)
            with closing(sqlite3.connect(database)) as connection:
                saved = connection.execute(f"SELECT * FROM {TABLE}").fetchall()
            records = [
                {**record, "value_pct": 99, "fetched_at": "2026-08-15"} for record in original
            ] + [
                {**self._record(day, value, available), "fetched_at": "2026-08-15"}
                for day, value, available in (
                    ("2026-01-02", 62, "2026-01-03"),
                    ("2026-01-05", 65, "2026-01-05"),
                    ("2026-01-07", 75, "2026-01-07"),
                )
            ]
            with patch("roro_market_breadth._historyofmarket_records", return_value=records):
                result = refresh_historyofmarket_ndth(database, as_of="2026-08-15")
                repeat = refresh_historyofmarket_ndth(database, as_of="2026-08-15")
            with closing(sqlite3.connect(database)) as connection:
                after = connection.execute(f"SELECT * FROM {TABLE}").fetchall()
            self.assertEqual(result["inserted"], 3)
            self.assertEqual(result["latest_before"], "2026-01-06")
            self.assertEqual(repeat["inserted"], 0)
            self.assertEqual(len(after), 5)
            self.assertTrue(set(saved).issubset(after))
            self.assertEqual(breadth_readiness(database, ["2026-01-02", "2026-01-05", "2026-01-06"], "2026-01-06")["status"], "READY")


if __name__ == "__main__":
    unittest.main()
