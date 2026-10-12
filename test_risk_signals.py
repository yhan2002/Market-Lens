import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import date, timedelta
from pathlib import Path

from breadth_metrics import EXACT_BREADTH_SOURCE, LIVE_BREADTH_SOURCE
from portfolio_risk import EVENT_HOLDINGS_SOURCE
from risk_signals import (
    build_fast_signal_snapshot,
    load_breadth_history,
    load_hy_oas_history,
)
from roro_price_observations import CONTRACT_VERSION, SOURCE, persist_price_observations


class FastSignalSnapshotTests(unittest.TestCase):
    def test_session_aligned_yahoo_jpy_overrides_fred_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "risk.db"
            observations = Path(directory) / "observations.db"
            self._create_database(database)
            dates = ["2026-08-04", "2026-08-05", "2026-08-06", "2026-08-07"]
            with closing(sqlite3.connect(database)) as connection:
                connection.executemany(
                    "INSERT INTO daily_prices VALUES (?, ?, ?, ?)",
                    [("NQ=F", day, 100.0, 100.0) for day in dates],
                )
                connection.executemany(
                    "INSERT INTO macro_observations VALUES (?, ?, ?, ?, ?)",
                    [(day, "USDJPY_FRED", 100.0, day, "FRED") for day in dates],
                )
                connection.commit()
            persist_price_observations(observations, [{
                "contract_version": CONTRACT_VERSION, "source": SOURCE, "ticker": "JPY=X",
                "raw_timestamp_utc": "2026-08-03T21:00:00Z", "source_timezone": "America/New_York",
                "observation_session_date": "2026-08-03", "available_at": "2026-08-03T21:00:00Z",
                "close": 100.0, "adj_close": None, "fetched_at": "2026-08-03T21:00:00Z",
            }, {
                "contract_version": CONTRACT_VERSION, "source": SOURCE, "ticker": "JPY=X",
                "raw_timestamp_utc": "2026-08-04T21:00:00Z", "source_timezone": "America/New_York",
                "observation_session_date": "2026-08-04", "available_at": "2026-08-04T21:00:00Z",
                "close": 100.0, "adj_close": None, "fetched_at": "2026-08-04T21:00:00Z",
            }, {
                "contract_version": CONTRACT_VERSION, "source": SOURCE, "ticker": "JPY=X",
                "raw_timestamp_utc": "2026-08-05T21:00:00Z", "source_timezone": "America/New_York",
                "observation_session_date": "2026-08-05", "available_at": "2026-08-05T21:00:00Z",
                "close": 100.0, "adj_close": None, "fetched_at": "2026-08-05T21:00:00Z",
            }, {
                "contract_version": CONTRACT_VERSION, "source": SOURCE, "ticker": "JPY=X",
                "raw_timestamp_utc": "2026-08-06T21:00:00Z", "source_timezone": "America/New_York",
                "observation_session_date": "2026-08-06", "available_at": "2026-08-06T21:00:00Z",
                "close": 95.0, "adj_close": None, "fetched_at": "2026-08-06T21:00:00Z",
            }])
            result = build_fast_signal_snapshot(
                database, "2026-08-07", jpy_observation_database=observations
            )
        self.assertEqual(self._condition(result, "jpy_3d_drop")["status"], "TRIGGERED")
    @staticmethod
    def _create_database(path: Path, *, aggregates: bool = True) -> None:
        with closing(sqlite3.connect(path)) as connection:
            connection.execute(
                "CREATE TABLE daily_prices(ticker TEXT, date TEXT, close REAL, adj_close REAL)"
            )
            if aggregates:
                connection.execute(
                    """
                    CREATE TABLE macro_observations(
                        date TEXT, series TEXT, value REAL, available_at TEXT, source TEXT
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE breadth_metrics(
                        date TEXT, breadth_200dma REAL, breadth_change_1d REAL,
                        new_52w_lows REAL, advancers REAL, decliners REAL,
                        coverage REAL, source TEXT,
                        membership_as_of TEXT, available_at TEXT
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE portfolio_risk_metrics(
                        date TEXT, holdings_date TEXT, average_correlation REAL,
                        beta REAL, coverage REAL, source TEXT
                    )
                    """
                )
            connection.commit()

    @staticmethod
    def _condition(snapshot: dict, key: str) -> dict:
        return next(item for item in snapshot["conditions"] if item["key"] == key)

    def test_all_eight_exact_thresholds_trigger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "risk.db"
            self._create_database(database)
            with closing(sqlite3.connect(database)) as connection:
                connection.executemany(
                    "INSERT INTO daily_prices VALUES (?, ?, ?, ?)",
                    [
                        ("^VIX", "2026-08-06", 10.0, 10.0),
                        ("^VIX", "2026-08-07", 12.0, 12.0),
                        ("^VIX3M", "2026-08-07", 11.0, 11.0),
                        ("NQ=F", "2026-08-06", 100.0, 100.0),
                        ("NQ=F", "2026-08-07", 97.5, 97.5),
                        ("^MOVE", "2026-08-06", 100.0, 100.0),
                        ("^MOVE", "2026-08-07", 110.0, 110.0),
                    ],
                )
                connection.executemany(
                    "INSERT INTO macro_observations VALUES (?, ?, ?, ?, ?)",
                    [
                        ("2026-08-06", "HY_OAS", 3.50, "2026-08-06", "FRED"),
                        ("2026-08-07", "HY_OAS", 3.70, "2026-08-07", "FRED"),
                        ("2026-08-04", "USDJPY_FRED", 100.0, "2026-08-04", "FRED"),
                        ("2026-08-05", "USDJPY_FRED", 99.0, "2026-08-05", "FRED"),
                        ("2026-08-06", "USDJPY_FRED", 98.0, "2026-08-06", "FRED"),
                        ("2026-08-07", "USDJPY_FRED", 96.0, "2026-08-07", "FRED"),
                    ],
                )
                connection.execute(
                    "INSERT INTO breadth_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        "2026-08-07", 0.42, -0.10, 120.0, 900.0, 1600.0, 1.0,
                        EXACT_BREADTH_SOURCE, "2026-08-07", "2026-08-07",
                    ),
                )
                connection.execute(
                    "INSERT INTO portfolio_risk_metrics VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        "2026-08-07", "2026-08-06", 0.71, 1.2, 0.95,
                        "DAILY_HOLDINGS_ADJUSTED_CLOSE",
                    ),
                )
                connection.commit()

            result = build_fast_signal_snapshot(database, "2026-08-07")

        self.assertEqual(result["status"], "TRIGGERED")
        self.assertEqual(result["count"], 8)
        self.assertEqual(result["required"], 3)
        self.assertEqual(result["total"], 8)
        self.assertTrue(result["complete"])
        self.assertEqual(len(result["conditions"]), 8)
        self.assertTrue(all(item["status"] == "TRIGGERED" for item in result["conditions"]))
        self.assertTrue(all(item["data_quality"] == "EXACT" for item in result["conditions"]))
        self.assertTrue(all(item["as_of"] == "2026-08-07" for item in result["conditions"]))

    def test_nq_and_hy_price_proxies_are_not_used(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "risk.db"
            self._create_database(database, aggregates=False)
            with closing(sqlite3.connect(database)) as connection:
                connection.executemany(
                    "INSERT INTO daily_prices VALUES (?, ?, ?, ?)",
                    [
                        ("QQQ", "2026-08-06", 100.0, 100.0),
                        ("QQQ", "2026-08-07", 95.0, 95.0),
                        ("HYG", "2026-08-06", 90.0, 90.0),
                        ("HYG", "2026-08-07", 85.0, 85.0),
                        ("IEF", "2026-08-06", 100.0, 100.0),
                        ("IEF", "2026-08-07", 100.0, 100.0),
                    ],
                )
                connection.commit()

            result = build_fast_signal_snapshot(database, "2026-08-07")

        nq = self._condition(result, "nq_1d_drop")
        hy = self._condition(result, "hy_oas_widening")
        self.assertEqual(nq["status"], "UNAVAILABLE")
        self.assertEqual(nq["data_quality"], "UNAVAILABLE")
        self.assertNotIn("QQQ", nq["detail"])
        self.assertEqual(hy["status"], "UNAVAILABLE")
        self.assertEqual(hy["data_quality"], "UNAVAILABLE")
        self.assertEqual(result["status"], "INCOMPLETE")

    def test_three_observed_triggers_are_actionable_even_when_other_inputs_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "risk.db"
            self._create_database(database, aggregates=False)
            with closing(sqlite3.connect(database)) as connection:
                connection.executemany(
                    "INSERT INTO daily_prices VALUES (?, ?, ?, ?)",
                    [
                        ("^VIX", "2026-08-06", 10.0, 10.0),
                        ("^VIX", "2026-08-07", 13.0, 13.0),
                        ("^VIX3M", "2026-08-07", 11.0, 11.0),
                        ("NQ=F", "2026-08-06", 100.0, 100.0),
                        ("NQ=F", "2026-08-07", 96.0, 96.0),
                    ],
                )
                connection.commit()

            result = build_fast_signal_snapshot(database, "2026-08-07")

        self.assertEqual(result["count"], 3)
        self.assertEqual(result["status"], "TRIGGERED")
        self.assertFalse(result["complete"])

    def test_vix_backwardation_uses_strict_one_point_zero_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "risk.db"
            self._create_database(database, aggregates=False)
            with closing(sqlite3.connect(database)) as connection:
                connection.executemany(
                    "INSERT INTO daily_prices VALUES (?, ?, ?, ?)",
                    [
                        ("^VIX", "2026-08-06", 10.0, 10.0),
                        ("^VIX3M", "2026-08-06", 10.0, 10.0),
                        ("^VIX", "2026-08-07", 10.01, 10.01),
                        ("^VIX3M", "2026-08-07", 10.0, 10.0),
                    ],
                )
                connection.commit()

            equal = build_fast_signal_snapshot(database, "2026-08-06")
            inverted = build_fast_signal_snapshot(database, "2026-08-07")

        self.assertEqual(self._condition(equal, "vix_term_inversion")["status"], "CLEAR")
        self.assertEqual(self._condition(inverted, "vix_term_inversion")["status"], "TRIGGERED")

    def test_fully_observed_benign_snapshot_is_clear(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "risk.db"
            self._create_database(database)
            with closing(sqlite3.connect(database)) as connection:
                rows = [
                    ("^VIX", "2026-08-06", 15.0, 15.0),
                    ("^VIX", "2026-08-07", 16.0, 16.0),
                    ("^VIX3M", "2026-08-07", 18.0, 18.0),
                    ("NQ=F", "2026-08-06", 100.0, 100.0),
                    ("NQ=F", "2026-08-07", 99.0, 99.0),
                    ("^MOVE", "2026-08-06", 100.0, 100.0),
                    ("^MOVE", "2026-08-07", 105.0, 105.0),
                ]
                connection.executemany("INSERT INTO daily_prices VALUES (?, ?, ?, ?)", rows)
                connection.executemany(
                    "INSERT INTO macro_observations VALUES (?, ?, ?, ?, ?)",
                    [
                        ("2026-08-06", "HY_OAS", 3.50, "2026-08-06", "FRED"),
                        ("2026-08-07", "HY_OAS", 3.55, "2026-08-07", "FRED"),
                        ("2026-08-04", "USDJPY_FRED", 100.0, "2026-08-04", "FRED"),
                        ("2026-08-05", "USDJPY_FRED", 100.0, "2026-08-05", "FRED"),
                        ("2026-08-06", "USDJPY_FRED", 100.0, "2026-08-06", "FRED"),
                        ("2026-08-07", "USDJPY_FRED", 99.0, "2026-08-07", "FRED"),
                    ],
                )
                connection.execute(
                    "INSERT INTO breadth_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        "2026-08-07", 0.50, -0.02, 20.0, 2400.0, 1100.0, 1.0,
                        EXACT_BREADTH_SOURCE, "2026-08-07", "2026-08-07",
                    ),
                )
                connection.execute(
                    "INSERT INTO portfolio_risk_metrics VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        "2026-08-07", "2026-08-06", 0.50, 1.0, 0.95,
                        "DAILY_HOLDINGS_ADJUSTED_CLOSE",
                    ),
                )
                connection.commit()

            result = build_fast_signal_snapshot(database, "2026-08-07")

        self.assertEqual(result["status"], "CLEAR")
        self.assertEqual(result["count"], 0)
        self.assertTrue(result["complete"])

    def test_future_prices_and_late_macro_releases_are_not_used(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "risk.db"
            self._create_database(database)
            with closing(sqlite3.connect(database)) as connection:
                connection.executemany(
                    "INSERT INTO daily_prices VALUES (?, ?, ?, ?)",
                    [
                        ("QQQ", "2026-08-06", 100.0, 100.0),
                        ("QQQ", "2026-08-07", 99.0, 99.0),
                        ("QQQ", "2026-08-08", 50.0, 50.0),
                    ],
                )
                connection.executemany(
                    "INSERT INTO macro_observations VALUES (?, ?, ?, ?, ?)",
                    [
                        ("2026-08-06", "HY_OAS", 3.00, "2026-08-06", "FRED"),
                        ("2026-08-07", "HY_OAS", 5.00, "2026-08-08", "FRED"),
                    ],
                )
                connection.commit()

            result = build_fast_signal_snapshot(database, "2026-08-07")
            history = load_hy_oas_history(database, "2026-08-07")

        self.assertEqual(self._condition(result, "nq_1d_drop")["status"], "UNAVAILABLE")
        self.assertEqual(self._condition(result, "hy_oas_widening")["status"], "UNAVAILABLE")
        self.assertEqual(history, [("2026-08-06", 300.0)])

    def test_historical_hy_uses_only_vintages_available_by_the_date(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "risk.db"
            self._create_database(database)
            with closing(sqlite3.connect(database)) as connection:
                connection.executemany(
                    "INSERT INTO macro_observations VALUES (?, ?, ?, ?, ?)",
                    [
                        ("2026-06-29", "HY_OAS", 2.80, "2026-06-30", "FRED_API"),
                        ("2026-06-30", "HY_OAS", 2.75, "2026-07-01", "FRED_API"),
                        ("2026-07-01", "HY_OAS", 2.74, "2026-07-02", "FRED_API"),
                    ],
                )
                connection.commit()

            july_first = load_hy_oas_history(database, "2026-07-01")
            july_second = load_hy_oas_history(database, "2026-07-02")

        self.assertEqual(july_first, [("2026-06-30", 275.0), ("2026-06-29", 280.0)])
        self.assertEqual(july_second[0], ("2026-07-01", 274.0))

    def test_schema_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "risk.db"
            with closing(sqlite3.connect(database)) as connection:
                connection.execute("CREATE TABLE macro_observations(date TEXT, value REAL)")
                connection.execute("CREATE TABLE breadth_metrics(date TEXT, breadth_change_1d REAL)")
                connection.execute(
                    "CREATE TABLE portfolio_risk_metrics(date TEXT, average_correlation REAL)"
                )
                connection.commit()

            result = build_fast_signal_snapshot(database, "2026-08-07")

        self.assertEqual(len(result["conditions"]), 8)
        self.assertTrue(all(item["status"] == "UNAVAILABLE" for item in result["conditions"]))
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertFalse(result["complete"])

    def test_legacy_breadth_schema_fails_closed_even_with_exact_source_label(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "risk.db"
            with closing(sqlite3.connect(database)) as connection:
                connection.execute(
                    """
                    CREATE TABLE breadth_metrics(
                        date TEXT, breadth_200dma REAL, breadth_change_1d REAL,
                        new_52w_lows REAL, advancers REAL, decliners REAL,
                        coverage REAL, source TEXT
                    )
                    """
                )
                connection.execute(
                    "INSERT INTO breadth_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        "2026-08-07", 0.4, -0.2, 100, 900, 1600, 1.0,
                        EXACT_BREADTH_SOURCE,
                    ),
                )
                connection.commit()

            self.assertEqual(load_breadth_history(database, "2026-08-07"), [])
            breadth = self._condition(
                build_fast_signal_snapshot(database, "2026-08-07"),
                "breadth_1d_collapse",
            )
            self.assertEqual(breadth["status"], "UNAVAILABLE")

    def test_future_breadth_vintage_is_unavailable(self) -> None:
        for future_column in ("membership_as_of", "available_at"):
            with self.subTest(future_column=future_column), tempfile.TemporaryDirectory() as directory:
                database = Path(directory) / "risk.db"
                self._create_database(database)
                membership_as_of = "2026-08-08" if future_column == "membership_as_of" else "2026-08-07"
                available_at = "2026-08-08" if future_column == "available_at" else "2026-08-07"
                with closing(sqlite3.connect(database)) as connection:
                    connection.execute(
                        "INSERT INTO breadth_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            "2026-08-07", 0.4, -0.2, 100, 900, 1600, 1.0,
                            EXACT_BREADTH_SOURCE, membership_as_of, available_at,
                        ),
                    )
                    connection.commit()

                self.assertEqual(load_breadth_history(database, "2026-08-07"), [])
                breadth = self._condition(
                    build_fast_signal_snapshot(database, "2026-08-07"),
                    "breadth_1d_collapse",
                )
                self.assertEqual(breadth["status"], "UNAVAILABLE")

    def test_low_coverage_breadth_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "risk.db"
            self._create_database(database)
            with closing(sqlite3.connect(database)) as connection:
                connection.execute(
                    "INSERT INTO breadth_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        "2026-08-07", 0.40, -0.20, 100, 900, 1600, 0.70,
                        EXACT_BREADTH_SOURCE, "2026-08-07", "2026-08-07",
                    ),
                )
                connection.commit()
            result = build_fast_signal_snapshot(database, "2026-08-07")

        breadth = self._condition(result, "breadth_1d_collapse")
        self.assertEqual(breadth["status"], "UNAVAILABLE")
        self.assertEqual(breadth["data_quality"], "UNAVAILABLE")

    def test_live_public_breadth_is_labeled_and_used_only_as_public_live(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "risk.db"
            self._create_database(database)
            with closing(sqlite3.connect(database)) as connection:
                connection.execute(
                    "INSERT INTO breadth_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        "2026-08-07", 0.40, -0.20, 100, 900, 1600, 0.98,
                        LIVE_BREADTH_SOURCE, "2026-08-06", "2026-08-06",
                    ),
                )
                connection.commit()

            breadth = self._condition(
                build_fast_signal_snapshot(database, "2026-08-07"),
                "breadth_1d_collapse",
            )
            with closing(sqlite3.connect(database)) as connection:
                connection.execute("UPDATE breadth_metrics SET coverage=0.97")
                connection.commit()
            below_threshold = self._condition(
                build_fast_signal_snapshot(database, "2026-08-07"),
                "breadth_1d_collapse",
            )

        self.assertEqual(breadth["status"], "TRIGGERED")
        self.assertEqual(breadth["data_quality"], "PUBLIC_LIVE")
        self.assertEqual(below_threshold["status"], "UNAVAILABLE")

    def test_proxy_provenance_and_stale_holdings_are_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "risk.db"
            self._create_database(database)
            with closing(sqlite3.connect(database)) as connection:
                connection.executemany(
                    "INSERT INTO macro_observations VALUES (?, ?, ?, ?, ?)",
                    [
                        ("2026-08-06", "HY_OAS", 3.5, "2026-08-06", "PROXY"),
                        ("2026-08-07", "HY_OAS", 3.8, "2026-08-07", "PROXY"),
                    ],
                )
                connection.execute(
                    "INSERT INTO breadth_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        "2026-08-07", 0.4, -0.2, 100, 900, 1600, 1.0,
                        "PROXY", "2026-08-07", "2026-08-07",
                    ),
                )
                connection.execute(
                    "INSERT INTO portfolio_risk_metrics VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        "2026-08-07", "2026-07-01", 0.9, 1.0, 1.0,
                        "DAILY_HOLDINGS_ADJUSTED_CLOSE",
                    ),
                )
                connection.commit()

            result = build_fast_signal_snapshot(database, "2026-08-07")

        for key in ("hy_oas_widening", "breadth_1d_collapse", "portfolio_correlation"):
            self.assertEqual(self._condition(result, key)["status"], "UNAVAILABLE")

    def test_old_event_holdings_aggregate_remains_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "risk.db"
            self._create_database(database)
            with closing(sqlite3.connect(database)) as connection:
                connection.execute(
                    "INSERT INTO portfolio_risk_metrics VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        "2026-08-07",
                        "2026-07-01",
                        0.71,
                        1.0,
                        1.0,
                        EVENT_HOLDINGS_SOURCE,
                    ),
                )
                connection.commit()

            result = build_fast_signal_snapshot(database, "2026-08-07")

        correlation = self._condition(result, "portfolio_correlation")
        self.assertEqual(correlation["status"], "TRIGGERED")
        self.assertEqual(correlation["data_quality"], "EXACT")

    def test_holdings_fallback_emits_only_exact_aggregate_correlation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "risk.db"
            self._create_database(database, aggregates=False)
            with closing(sqlite3.connect(database)) as connection:
                connection.execute(
                    "CREATE TABLE daily_holdings(date TEXT, ticker TEXT, market_value_base REAL)"
                )
                connection.executemany(
                    "INSERT INTO daily_holdings VALUES (?, ?, ?)",
                    [
                        ("2026-08-06", "SECRET_A", 60.0),
                        ("2026-08-06", "SECRET_B", 40.0),
                    ],
                )
                start = date(2026, 7, 18)
                prices = []
                for offset in range(21):
                    price_date = (start + timedelta(days=offset)).isoformat()
                    price = 100.0 * (1.01**offset)
                    prices.extend(
                        [
                            ("SECRET_A", price_date, price, price),
                            ("SECRET_B", price_date, 2.0 * price, 2.0 * price),
                        ]
                    )
                # Make the last complete aggregate correspond exactly to the snapshot date.
                prices.extend(
                    [
                        ("SECRET_A", "2026-08-07", 125.0, 125.0),
                        ("SECRET_B", "2026-08-07", 250.0, 250.0),
                    ]
                )
                connection.executemany("INSERT INTO daily_prices VALUES (?, ?, ?, ?)", prices)
                connection.commit()

            result = build_fast_signal_snapshot(database, "2026-08-07")

        correlation = self._condition(result, "portfolio_correlation")
        self.assertEqual(correlation["status"], "TRIGGERED")
        self.assertEqual(correlation["data_quality"], "EXACT")
        self.assertNotIn("SECRET_A", str(result))
        self.assertNotIn("SECRET_B", str(result))


if __name__ == "__main__":
    unittest.main()
