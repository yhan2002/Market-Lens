import io
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from breadth_metrics import (
    EXACT_BREADTH_SOURCE,
    LIVE_BREADTH_SOURCE,
    MIN_LIVE_BREADTH_COVERAGE,
    SP500_BREADTH_GROUP,
    OPEN_DATA_MEMBERSHIP_SOURCE,
    active_members,
    compute_breadth_metrics,
    import_membership,
    initialize_database,
    members_overlapping,
    register_active_members,
    register_members_overlapping,
    sync_current_open_data_membership,
)


class BreadthMetricsTest(unittest.TestCase):
    @staticmethod
    def _current_members_payload(*, replace: tuple[str, str] | None = None) -> bytes:
        symbols = ["BRK.B", "BF.B", *(f"T{index}" for index in range(498))]
        if replace:
            symbols[symbols.index(replace[0])] = replace[1]
        return ("Symbol,Security\n" + "".join(f"{symbol},Name\n" for symbol in symbols)).encode()

    @staticmethod
    def _create_price_table(database: Path) -> None:
        initialize_database(database)
        with closing(sqlite3.connect(database)) as connection:
            connection.execute(
                "CREATE TABLE daily_prices(ticker TEXT, date TEXT, close REAL, adj_close REAL)"
            )
            connection.commit()

    @staticmethod
    def _insert_members(database: Path, tickers: list[str], effective_from: str) -> None:
        with closing(sqlite3.connect(database)) as connection:
            connection.executemany(
                """
                INSERT INTO sp500_membership(
                    ticker, effective_from, effective_to,
                    membership_as_of, available_at, source
                ) VALUES (?, ?, NULL, ?, ?, 'LICENSED_PIT:TEST')
                """,
                [(ticker, effective_from, effective_from, effective_from) for ticker in tickers],
            )
            connection.commit()

    def test_membership_dates_are_point_in_time_and_end_is_inclusive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "router.db"
            membership = root / "membership.csv"
            membership.write_text(
                "ticker,effective_from,effective_to,membership_as_of,available_at,source\n"
                "old,2020-01-01,2020-06-30,2020-01-01,2020-01-01,LICENSED_PIT:TEST\n"
                "new,2020-07-01,,2020-07-01,2020-07-01,LICENSED_PIT:TEST\n"
                "current,2021-01-01,,2021-01-01,2021-01-01,LICENSED_PIT:TEST\n",
                encoding="utf-8",
            )

            self.assertEqual(
                import_membership(database, membership, source="LICENSED_PIT:TEST"), 3
            )
            self.assertEqual(active_members(database, "2019-12-31"), ())
            self.assertEqual(active_members(database, "2020-06-30"), ("OLD",))
            self.assertEqual(active_members(database, "2020-07-01"), ("NEW",))
            self.assertEqual(active_members(database, "2020-12-31"), ("NEW",))
            self.assertEqual(active_members(database, "2021-01-01"), ("CURRENT", "NEW"))

            overlap = root / "overlap.csv"
            overlap.write_text(
                "ticker,effective_from,effective_to,membership_as_of,available_at,source\n"
                "old,2020-06-30,2020-12-31,2020-06-30,2020-06-30,LICENSED_PIT:TEST\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "overlapping membership intervals"):
                import_membership(database, overlap, source="LICENSED_PIT:TEST")

    def test_current_open_data_sync_starts_on_capture_date_and_maps_yahoo_symbols(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "router.db"
            with patch(
                "breadth_metrics.urlopen",
                return_value=io.BytesIO(self._current_members_payload()),
            ):
                self.assertEqual(
                    sync_current_open_data_membership(database, "2026-08-11"),
                    (500, 500, 0),
                )
            self.assertIn("BRK-B", active_members(database, "2026-08-11"))
            self.assertEqual(active_members(database, "2026-08-10"), ())

            with patch(
                "breadth_metrics.urlopen",
                return_value=io.BytesIO(
                    self._current_members_payload(replace=("T0", "NEW"))
                ),
            ):
                self.assertEqual(
                    sync_current_open_data_membership(database, "2026-08-12"),
                    (500, 1, 1),
                )
            with closing(sqlite3.connect(database)) as connection:
                removed = connection.execute(
                    "SELECT effective_to FROM sp500_membership WHERE ticker='T0'"
                ).fetchone()
                added = connection.execute(
                    "SELECT effective_from, source FROM sp500_membership WHERE ticker='NEW'"
                ).fetchone()
            self.assertEqual(removed, ("2026-08-11",))
            self.assertEqual(added, ("2026-08-12", OPEN_DATA_MEMBERSHIP_SOURCE))

    def test_complete_exact_import_replaces_public_live_period_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "router.db"
            with patch(
                "breadth_metrics.urlopen",
                return_value=io.BytesIO(self._current_members_payload()),
            ):
                sync_current_open_data_membership(database, "2026-08-11")
            with closing(sqlite3.connect(database)) as connection:
                connection.execute(
                    "INSERT INTO breadth_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        "2026-08-11", 0.6, None, 1, 300, 199, 1.0,
                        LIVE_BREADTH_SOURCE, "2026-08-11", "2026-08-11",
                    ),
                )
                connection.commit()

            membership = root / "exact.csv"
            symbols = ["BRK-B", "BF-B", *(f"T{index}" for index in range(498))]
            membership.write_text(
                "ticker,effective_from,effective_to,membership_as_of,available_at,source\n"
                + "".join(
                    f"{ticker},2026-08-11,,2026-08-11,2026-08-11,LICENSED_PIT:TEST\n"
                    for ticker in symbols
                ),
                encoding="utf-8",
            )
            self.assertEqual(import_membership(database, membership), 500)
            with closing(sqlite3.connect(database)) as connection:
                public_count = connection.execute(
                    "SELECT COUNT(*) FROM sp500_membership "
                    "WHERE source GLOB 'PUBLIC_CURRENT:?*'"
                ).fetchone()[0]
                exact_count = connection.execute(
                    "SELECT COUNT(*) FROM sp500_membership "
                    "WHERE source GLOB 'LICENSED_PIT:?*'"
                ).fetchone()[0]
                live_count = connection.execute(
                    "SELECT COUNT(*) FROM breadth_metrics WHERE source=?",
                    (LIVE_BREADTH_SOURCE,),
                ).fetchone()[0]
            self.assertEqual((public_count, exact_count, live_count), (0, 500, 0))

    def test_current_open_data_sync_rejects_incomplete_list(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "router.db"
            with patch(
                "breadth_metrics.urlopen",
                return_value=io.BytesIO(b"Symbol,Security\nA,Name\n"),
            ):
                with self.assertRaisesRegex(ValueError, "outside 490..510"):
                    sync_current_open_data_membership(database, "2026-08-11")
            self.assertFalse(database.exists())

    def test_metrics_are_causal_and_prefer_adjusted_close(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "router.db"
            self._create_price_table(database)
            dates = [timestamp.date().isoformat() for timestamp in pd.bdate_range("2024-01-02", periods=253)]
            self._insert_members(database, ["A", "B"], dates[0])
            rows = [
                (ticker, day, close, adjusted)
                for index, day in enumerate(dates[:252])
                for ticker, close, adjusted in (
                    ("A", 1000.0 - index, 100.0 + index),
                    ("B", 2000.0 - index, 200.0 + index),
                )
            ]
            with closing(sqlite3.connect(database)) as connection:
                connection.executemany("INSERT INTO daily_prices VALUES (?, ?, ?, ?)", rows)
                connection.commit()

            self.assertEqual(compute_breadth_metrics(database), 1)
            with closing(sqlite3.connect(database)) as connection:
                before = connection.execute(
                    "SELECT breadth_200dma, new_52w_lows, advancers, decliners, coverage, "
                    "source, membership_as_of, available_at "
                    "FROM breadth_metrics WHERE date=?",
                    (dates[251],),
                ).fetchone()
                connection.executemany(
                    "INSERT INTO daily_prices VALUES (?, ?, ?, ?)",
                    [("A", dates[252], 5000.0, 1.0), ("B", dates[252], 6000.0, 2.0)],
                )
                connection.commit()

            self.assertEqual(
                before,
                (1.0, 0, 2, 0, 1.0, EXACT_BREADTH_SOURCE, dates[0], dates[0]),
            )
            self.assertEqual(compute_breadth_metrics(database), 2)
            with closing(sqlite3.connect(database)) as connection:
                after = connection.execute(
                    "SELECT breadth_200dma, new_52w_lows, advancers, decliners, coverage, "
                    "source, membership_as_of, available_at "
                    "FROM breadth_metrics WHERE date=?",
                    (dates[251],),
                ).fetchone()
                latest_change = connection.execute(
                    "SELECT breadth_change_1d FROM breadth_metrics WHERE date=?",
                    (dates[252],),
                ).fetchone()[0]
            self.assertEqual(after, before)
            self.assertEqual(latest_change, -1.0)

    def test_public_current_membership_produces_live_only_breadth(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "router.db"
            self._create_price_table(database)
            dates = [
                timestamp.date().isoformat()
                for timestamp in pd.bdate_range("2025-08-12", periods=252)
            ]
            with closing(sqlite3.connect(database)) as connection:
                connection.execute(
                    """
                    INSERT INTO sp500_membership(
                        ticker, effective_from, effective_to,
                        membership_as_of, available_at, source
                    ) VALUES ('A', ?, NULL, ?, ?, ?)
                    """,
                    (dates[0], dates[0], dates[0], OPEN_DATA_MEMBERSHIP_SOURCE),
                )
                connection.executemany(
                    "INSERT INTO daily_prices VALUES (?, ?, ?, ?)",
                    [("A", day, 100.0 + index, 100.0 + index) for index, day in enumerate(dates)],
                )
                connection.commit()

            self.assertEqual(compute_breadth_metrics(database), 1)
            with closing(sqlite3.connect(database)) as connection:
                row = connection.execute(
                    "SELECT date, source FROM breadth_metrics"
                ).fetchone()
            self.assertEqual(row, (dates[-1], LIVE_BREADTH_SOURCE))

    def test_public_live_allows_98_percent_but_exact_stays_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "router.db"
            self._create_price_table(database)
            dates = [
                timestamp.date().isoformat()
                for timestamp in pd.bdate_range("2025-08-12", periods=252)
            ]
            tickers = [f"T{index}" for index in range(50)]
            with closing(sqlite3.connect(database)) as connection:
                connection.executemany(
                    """
                    INSERT INTO sp500_membership(
                        ticker, effective_from, effective_to,
                        membership_as_of, available_at, source
                    ) VALUES (?, ?, NULL, ?, ?, ?)
                    """,
                    [
                        (
                            ticker, dates[0], dates[0], dates[0],
                            OPEN_DATA_MEMBERSHIP_SOURCE,
                        )
                        for ticker in tickers
                    ],
                )
                connection.executemany(
                    "INSERT INTO daily_prices VALUES (?, ?, ?, ?)",
                    [
                        (ticker, day, 100.0 + index, 100.0 + index)
                        for ticker in tickers[:49]
                        for index, day in enumerate(dates)
                    ],
                )
                connection.commit()

            self.assertEqual(compute_breadth_metrics(database), 1)
            with closing(sqlite3.connect(database)) as connection:
                row = connection.execute(
                    "SELECT coverage, source FROM breadth_metrics"
                ).fetchone()
                connection.execute("DELETE FROM daily_prices WHERE ticker='T48'")
                connection.commit()
            self.assertEqual(row, (MIN_LIVE_BREADTH_COVERAGE, LIVE_BREADTH_SOURCE))
            self.assertEqual(compute_breadth_metrics(database), 0)

    def test_coverage_gate_removes_stale_failed_date(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "router.db"
            self._create_price_table(database)
            dates = [timestamp.date().isoformat() for timestamp in pd.bdate_range("2024-01-02", periods=252)]
            tickers = [f"T{index}" for index in range(10)]
            self._insert_members(database, tickers, dates[0])
            rows = [
                (ticker, day, 100.0 + index, 100.0 + index)
                for ticker in tickers[:9]
                for index, day in enumerate(dates)
            ]
            with closing(sqlite3.connect(database)) as connection:
                connection.executemany("INSERT INTO daily_prices VALUES (?, ?, ?, ?)", rows)
                connection.commit()

            self.assertEqual(compute_breadth_metrics(database), 0)
            with closing(sqlite3.connect(database)) as connection:
                self.assertIsNone(
                    connection.execute(
                        "SELECT coverage FROM breadth_metrics WHERE date=?", (dates[-1],)
                    ).fetchone()
                )
                connection.executemany(
                    "INSERT INTO daily_prices VALUES (?, ?, ?, ?)",
                    [("T9", day, 100.0 + index, 100.0 + index) for index, day in enumerate(dates)],
                )
                connection.commit()

            self.assertEqual(compute_breadth_metrics(database), 1)
            with closing(sqlite3.connect(database)) as connection:
                self.assertEqual(
                    connection.execute(
                        "SELECT coverage FROM breadth_metrics WHERE date=?", (dates[-1],)
                    ).fetchone(),
                    (1.0,),
                )
                connection.execute(
                    "DELETE FROM daily_prices WHERE ticker='T8' AND date=?", (dates[-1],)
                )
                connection.commit()

            self.assertEqual(compute_breadth_metrics(database), 0)
            with closing(sqlite3.connect(database)) as connection:
                self.assertIsNone(
                    connection.execute(
                        "SELECT coverage FROM breadth_metrics WHERE date=?", (dates[-1],)
                    ).fetchone()
                )
                connection.execute(
                    """
                    INSERT INTO breadth_metrics VALUES (
                        ?, 0.5, NULL, 0, 1, 0, 1.0, ?, ?, ?
                    )
                    """,
                    (dates[-1], EXACT_BREADTH_SOURCE, dates[0], dates[0]),
                )
                connection.execute("DELETE FROM daily_prices")
                connection.commit()

            self.assertEqual(compute_breadth_metrics(database), 0)
            with closing(sqlite3.connect(database)) as connection:
                self.assertEqual(
                    connection.execute("SELECT COUNT(*) FROM breadth_metrics").fetchone()[0],
                    0,
                )

    def test_change_fails_closed_when_previous_market_date_is_ineligible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "router.db"
            self._create_price_table(database)
            dates = [timestamp.date().isoformat() for timestamp in pd.bdate_range("2024-01-02", periods=254)]
            self._insert_members(database, ["A", "B"], dates[0])
            rows = []
            for index, day in enumerate(dates):
                rows.append(("A", day, 100.0 + index, 1.0 if index == 253 else 100.0 + index))
                if index != 252:
                    rows.append(("B", day, 200.0 + index, 200.0 + index))
            with closing(sqlite3.connect(database)) as connection:
                connection.executemany("INSERT INTO daily_prices VALUES (?, ?, ?, ?)", rows)
                connection.commit()

            self.assertEqual(compute_breadth_metrics(database), 2)
            with closing(sqlite3.connect(database)) as connection:
                metrics = connection.execute(
                    "SELECT date, breadth_200dma, breadth_change_1d FROM breadth_metrics ORDER BY date"
                ).fetchall()
            self.assertEqual(metrics[0], (dates[251], 1.0, None))
            self.assertEqual(metrics[1], (dates[253], 0.5, None))

    def test_register_active_uses_market_data_without_downloading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "router.db"
            initialize_database(database)
            with closing(sqlite3.connect(database)) as connection:
                connection.executemany(
                    """
                    INSERT INTO sp500_membership(
                        ticker, effective_from, effective_to,
                        membership_as_of, available_at, source
                    ) VALUES (?, ?, ?, ?, ?, 'LICENSED_PIT:TEST')
                    """,
                    [
                        ("OLD", "2020-01-01", "2024-12-31", "2020-01-01", "2020-01-01"),
                        ("CURRENT", "2025-01-01", None, "2025-01-01", "2025-01-01"),
                    ],
                )
                connection.commit()

            with patch("market_data.register_universe", return_value=1) as register:
                self.assertEqual(
                    register_active_members(database, "2025-01-01", source="membership-test"),
                    1,
                )
            register.assert_called_once_with(
                database,
                ("CURRENT",),
                group_name=SP500_BREADTH_GROUP,
                source="membership-test",
            )

    def test_register_range_includes_former_members_and_blocks_survivor_bias(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "router.db"
            initialize_database(database)
            with closing(sqlite3.connect(database)) as connection:
                connection.executemany(
                    """
                    INSERT INTO sp500_membership(
                        ticker, effective_from, effective_to,
                        membership_as_of, available_at, source
                    ) VALUES (?, ?, ?, ?, ?, 'LICENSED_PIT:TEST')
                    """,
                    [
                        ("FORMER", "2024-01-01", "2025-02-28", "2024-01-01", "2024-01-01"),
                        ("CURRENT", "2025-01-01", None, "2025-01-01", "2025-01-01"),
                        ("TOO_OLD", "2020-01-01", "2023-12-31", "2020-01-01", "2020-01-01"),
                    ],
                )
                connection.commit()

            self.assertEqual(
                members_overlapping(database, "2024-08-01", "2026-08-01"),
                ("CURRENT", "FORMER"),
            )
            with patch("market_data.register_universe", return_value=2) as register:
                count = register_members_overlapping(
                    database,
                    "2024-08-01",
                    "2026-08-01",
                    source="membership-test",
                )
            self.assertEqual(count, 2)
            register.assert_called_once_with(
                database,
                ("CURRENT", "FORMER"),
                group_name=SP500_BREADTH_GROUP,
                source="membership-test",
            )

    def test_future_vintage_does_not_rewrite_historical_membership(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "router.db"
            membership = root / "membership.csv"
            membership.write_text(
                "ticker,effective_from,effective_to,membership_as_of,available_at,source\n"
                "former,2020-01-01,2020-12-31,2020-01-01,2020-01-01,LICENSED_PIT:TEST\n"
                "survivor,2020-01-01,,2021-01-01,2021-01-01,LICENSED_PIT:TEST\n",
                encoding="utf-8",
            )
            import_membership(database, membership)

            self.assertEqual(active_members(database, "2020-06-30"), ("FORMER",))
            self.assertEqual(
                members_overlapping(database, "2020-01-01", "2020-12-31"),
                ("FORMER",),
            )
            self.assertEqual(active_members(database, "2021-01-01"), ("SURVIVOR",))

    def test_import_requires_vintage_columns_and_exact_producer_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "router.db"
            missing = root / "missing.csv"
            missing.write_text(
                "ticker,effective_from,effective_to\nA,2020-01-01,\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "membership CSV missing columns"):
                import_membership(database, missing)

            proxy = root / "proxy.csv"
            proxy.write_text(
                "ticker,effective_from,effective_to,membership_as_of,available_at,source\n"
                "A,2020-01-01,,2020-01-01,2020-01-01,PROXY\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "LICENSED_PIT"):
                import_membership(database, proxy)

    def test_legacy_schema_migrates_without_trusting_existing_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "router.db"
            with closing(sqlite3.connect(database)) as connection:
                connection.executescript(
                    """
                    CREATE TABLE sp500_membership(
                        ticker TEXT, effective_from TEXT, effective_to TEXT, source TEXT,
                        PRIMARY KEY(ticker, effective_from)
                    );
                    INSERT INTO sp500_membership VALUES ('A', '2020-01-01', NULL, 'legacy');
                    CREATE TABLE breadth_metrics(
                        date TEXT PRIMARY KEY, breadth_200dma REAL, breadth_change_1d REAL,
                        new_52w_lows INTEGER, advancers INTEGER, decliners INTEGER,
                        coverage REAL, source TEXT
                    );
                    INSERT INTO breadth_metrics VALUES (
                        '2026-08-07', 0.5, 0.1, 1, 2, 3, 1.0, 'legacy'
                    );
                    """
                )
                connection.commit()

            initialize_database(database)
            with closing(sqlite3.connect(database)) as connection:
                membership_columns = {
                    row[1] for row in connection.execute("PRAGMA table_info(sp500_membership)")
                }
                breadth_columns = {
                    row[1] for row in connection.execute("PRAGMA table_info(breadth_metrics)")
                }
                migrated = connection.execute(
                    "SELECT membership_as_of, available_at FROM sp500_membership"
                ).fetchone()
            self.assertTrue({"membership_as_of", "available_at"} <= membership_columns)
            self.assertTrue({"membership_as_of", "available_at"} <= breadth_columns)
            self.assertEqual(migrated, (None, None))
            self.assertEqual(active_members(database, "2026-08-07"), ())

    def test_exact_only_source_checks_expand_without_losing_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "router.db"
            with closing(sqlite3.connect(database)) as connection:
                connection.executescript(
                    f"""
                    CREATE TABLE sp500_membership(
                        ticker TEXT NOT NULL, effective_from TEXT NOT NULL,
                        effective_to TEXT, membership_as_of TEXT NOT NULL,
                        available_at TEXT NOT NULL, source TEXT NOT NULL,
                        PRIMARY KEY(ticker, effective_from),
                        CHECK(source GLOB 'LICENSED_PIT:?*' OR source GLOB 'BLOOMBERG_PIT:?*')
                    );
                    INSERT INTO sp500_membership VALUES(
                        'A', '2026-08-06', NULL, '2026-08-06', '2026-08-06',
                        'LICENSED_PIT:TEST'
                    );
                    CREATE TABLE breadth_metrics(
                        date TEXT PRIMARY KEY, breadth_200dma REAL NOT NULL,
                        breadth_change_1d REAL, new_52w_lows INTEGER NOT NULL,
                        advancers INTEGER NOT NULL, decliners INTEGER NOT NULL,
                        coverage REAL NOT NULL, source TEXT NOT NULL,
                        membership_as_of TEXT NOT NULL, available_at TEXT NOT NULL,
                        CHECK(source = '{EXACT_BREADTH_SOURCE}')
                    );
                    INSERT INTO breadth_metrics VALUES(
                        '2026-08-06', 0.5, NULL, 1, 250, 249, 1.0,
                        '{EXACT_BREADTH_SOURCE}', '2026-08-06', '2026-08-06'
                    );
                    """
                )
                connection.commit()

            initialize_database(database)
            with closing(sqlite3.connect(database)) as connection:
                connection.execute(
                    "INSERT INTO sp500_membership VALUES (?, ?, NULL, ?, ?, ?)",
                    (
                        "B", "2026-08-07", "2026-08-07", "2026-08-07",
                        OPEN_DATA_MEMBERSHIP_SOURCE,
                    ),
                )
                connection.execute(
                    "INSERT INTO breadth_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        "2026-08-07", 0.6, 0.1, 0, 300, 200,
                        MIN_LIVE_BREADTH_COVERAGE,
                        LIVE_BREADTH_SOURCE, "2026-08-07", "2026-08-07",
                    ),
                )
                connection.commit()
                self.assertEqual(
                    connection.execute("SELECT COUNT(*) FROM sp500_membership").fetchone()[0],
                    2,
                )
                self.assertEqual(
                    connection.execute("SELECT COUNT(*) FROM breadth_metrics").fetchone()[0],
                    2,
                )


if __name__ == "__main__":
    unittest.main()
