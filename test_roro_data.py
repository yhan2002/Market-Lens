import hashlib
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import pandas as pd

from roro_data import HY_MAX_STALE_CALENDAR_DAYS, load_roro_panel
from roro_engine import INPUT_COLUMNS
from roro_market_breadth import CONTRACT_VERSION as BREADTH_CONTRACT
from roro_market_breadth import SOURCE as BREADTH_SOURCE
from roro_market_breadth import persist_breadth_observations
from roro_price_observations import CONTRACT_VERSION, SOURCE, persist_price_observations


class RoroDataTest(unittest.TestCase):
    def _database(self, directory: str) -> Path:
        database = Path(directory) / "roro.db"
        with closing(sqlite3.connect(database)) as connection:
            connection.executescript(
                """
                CREATE TABLE daily_prices(
                    ticker TEXT, date TEXT, close REAL, adj_close REAL,
                    PRIMARY KEY(ticker, date)
                );
                CREATE TABLE macro_observations(
                    date TEXT, series TEXT, value REAL, available_at TEXT, source TEXT,
                    PRIMARY KEY(series, date, available_at)
                );
                CREATE TABLE portfolio_risk_metrics(
                    date TEXT PRIMARY KEY, holdings_date TEXT,
                    average_correlation REAL, coverage REAL, source TEXT
                );
                """
            )
            connection.commit()
        return database

    @staticmethod
    def _prices(database: Path, dates: list[str]) -> None:
        with closing(sqlite3.connect(database)) as connection:
            connection.executemany(
                "INSERT INTO daily_prices VALUES('NQ=F', ?, ?, ?)",
                [(day, 100 + index, 100 + index) for index, day in enumerate(dates)],
            )
            connection.commit()

    def test_nq_calendar_is_preserved_and_prices_are_never_filled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            dates = ["2026-01-02", "2026-01-05", "2026-01-06"]
            self._prices(database, dates)
            with closing(sqlite3.connect(database)) as connection:
                connection.executemany(
                    "INSERT INTO daily_prices VALUES(?, ?, ?, ?)",
                    [
                        ("^MOVE", dates[0], 90, 90),
                        ("^MOVE", dates[2], 92, 92),
                        ("^VIX", dates[1], 18, 18),
                        ("^VIX3M", dates[1], 20, 20),
                    ],
                )
                connection.commit()
            panel = load_roro_panel(database, as_of=dates[-1])

        self.assertEqual(tuple(panel.columns), INPUT_COLUMNS)
        self.assertEqual([day.date().isoformat() for day in panel.index], dates)
        self.assertTrue(pd.isna(panel.loc[dates[1], "move"]))
        self.assertTrue(pd.isna(panel.loc[dates[0], "vix"]))
        self.assertEqual(panel.loc[dates[2], "move"], 92)
        self.assertEqual(panel.attrs["fill_policy"], "exact-date only; no forward/back fill")

    def test_reviewed_move_close_takes_precedence_over_yahoo_price(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            dates = ["2026-01-02", "2026-01-05"]
            self._prices(database, dates)
            with closing(sqlite3.connect(database)) as connection:
                connection.execute(
                    "INSERT INTO daily_prices VALUES('^MOVE', '2026-01-02', 90, 90)"
                )
                connection.executescript(
                    """
                    CREATE TABLE reviewed_move_closes(
                        date TEXT PRIMARY KEY, close REAL, source TEXT, imported_at TEXT
                    );
                    """
                )
                connection.execute(
                    "INSERT INTO reviewed_move_closes VALUES('2026-01-02', 72.66, 'INVESTING_REVIEWED_CSV', '2026-08-14T00:00:00Z')"
                )
                connection.commit()
            panel = load_roro_panel(database, as_of=dates[-1])

        self.assertEqual(panel.loc["2026-01-02", "move"], 72.66)
        self.assertIn("REVIEWED_OVERRIDE", panel.attrs["lineage"]["move"]["status"])

    def test_hy_vintage_uses_next_session_and_fails_when_stale(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            dates = [
                "2026-01-02",
                "2026-01-05",
                "2026-01-06",
                "2026-01-07",
                "2026-01-08",
                "2026-01-09",
            ]
            self._prices(database, dates)
            with closing(sqlite3.connect(database)) as connection:
                connection.executemany(
                    "INSERT INTO macro_observations VALUES(?, 'HY_OAS', ?, ?, 'ALFRED')",
                    [
                        ("2026-01-01", 3.0, "2026-01-02"),
                        ("2026-01-01", 3.1, "2026-01-05"),
                        ("2026-01-05", 4.0, "2026-01-06"),
                    ],
                )
                connection.commit()
            panel = load_roro_panel(database, as_of=dates[-1])

        self.assertEqual(HY_MAX_STALE_CALENDAR_DAYS, 5)
        self.assertTrue(pd.isna(panel.loc["2026-01-02", "hy_oas"]))
        self.assertEqual(panel.loc["2026-01-05", "hy_oas"], 3.0)
        self.assertEqual(panel.loc["2026-01-06", "hy_oas"], 3.1)
        self.assertEqual(panel.loc["2026-01-07", "hy_oas"], 4.0)

    def test_stale_hy_and_strict_blockers_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            dates = ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"]
            self._prices(database, dates)
            with closing(sqlite3.connect(database)) as connection:
                connection.execute(
                    "INSERT INTO macro_observations VALUES"
                    "('2025-12-31','HY_OAS',3.0,'2026-01-02','ALFRED')"
                )
                connection.execute(
                    "INSERT INTO portfolio_risk_metrics VALUES"
                    "('2026-01-07','2026-01-06',0.55,0.95,'DAILY_HOLDINGS_ADJUSTED_CLOSE')"
                )
                connection.commit()
            strict = load_roro_panel(database, as_of=dates[-1])
            conditional = load_roro_panel(
                database, as_of=dates[-1], allow_conditional_portfolio=True
            )

        self.assertTrue(pd.isna(strict.loc["2026-01-07", "hy_oas"]))
        for column in ("usdjpy", "gold", "breadth_200dma", "portfolio_correlation_20"):
            self.assertTrue(strict[column].isna().all())
        self.assertEqual(
            strict.attrs["lineage"]["portfolio_correlation_20"]["status"],
            "DATA_BLOCKED",
        )
        self.assertEqual(conditional.loc["2026-01-07", "portfolio_correlation_20"], 0.55)
        self.assertEqual(
            conditional.attrs["lineage"]["portfolio_correlation_20"]["status"],
            "CONDITIONAL_SHADOW",
        )

    def test_as_of_excludes_future_rows_and_database_is_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            self._prices(database, ["2026-01-02", "2026-01-05", "2026-01-06"])
            with closing(sqlite3.connect(database)) as connection:
                connection.execute(
                    "INSERT INTO daily_prices VALUES('^MOVE','2026-01-06',99,99)"
                )
                connection.executemany(
                    "INSERT INTO macro_observations VALUES(?, 'HY_OAS', ?, ?, 'ALFRED')",
                    [
                        ("2026-01-01", 3.0, "2026-01-02"),
                        ("2026-01-02", 9.0, "2026-01-06"),
                    ],
                )
                connection.commit()
            before = hashlib.sha256(database.read_bytes()).digest()
            panel = load_roro_panel(database, as_of="2026-01-05")
            after = hashlib.sha256(database.read_bytes()).digest()

        self.assertEqual(before, after)
        self.assertEqual(panel.index.max(), pd.Timestamp("2026-01-05"))
        self.assertNotIn(pd.Timestamp("2026-01-06"), panel.index)
        self.assertTrue(panel["move"].isna().all())
        self.assertEqual(panel.loc["2026-01-05", "hy_oas"], 3.0)
        self.assertEqual(panel.attrs["database_mode"], "mode=ro; query_only=ON")

    def test_optional_session_database_populates_fx_and_gold_without_filling(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            dates = ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"]
            self._prices(database, dates)
            observations = Path(directory) / "price_observations.db"
            common = {
                "contract_version": CONTRACT_VERSION,
                "source": SOURCE,
                "source_timezone": "America/New_York",
                "observation_session_date": "2026-01-05",
                "available_at": "2026-01-05T21:05:00Z",
                "adj_close": None,
                "fetched_at": "2026-01-06T01:00:00Z",
            }
            persist_price_observations(
                observations,
                [
                    {
                        **common,
                        "ticker": "JPY=X",
                        "raw_timestamp_utc": "2026-01-05T21:00:00Z",
                        "close": 156.0,
                    },
                    {
                        **common,
                        "ticker": "GLD",
                        "raw_timestamp_utc": "2026-01-05T21:01:00Z",
                        "close": 410.0,
                    },
                ],
            )
            source_before = hashlib.sha256(database.read_bytes()).digest()
            observation_before = hashlib.sha256(observations.read_bytes()).digest()

            strict = load_roro_panel(database, as_of=dates[-1])
            enriched = load_roro_panel(
                database,
                as_of=dates[-1],
                price_observation_database=observations,
            )

            source_after = hashlib.sha256(database.read_bytes()).digest()
            observation_after = hashlib.sha256(observations.read_bytes()).digest()

        self.assertTrue(strict["usdjpy"].isna().all())
        self.assertTrue(strict["gold"].isna().all())
        self.assertEqual(enriched.loc["2026-01-05", "usdjpy"], 156.0)
        self.assertEqual(enriched.loc["2026-01-05", "gold"], 410.0)
        self.assertTrue(pd.isna(enriched.loc["2026-01-06", "usdjpy"]))
        self.assertTrue(pd.isna(enriched.loc["2026-01-06", "gold"]))
        self.assertEqual(
            enriched.attrs["lineage"]["usdjpy"]["status"],
            "CONDITIONAL_PRICE_SESSION_CAUSAL",
        )
        self.assertEqual(
            enriched.attrs["lineage"]["gold"]["status"],
            "CONDITIONAL_PRICE_SESSION_CAUSAL",
        )
        self.assertEqual(source_before, source_after)
        self.assertEqual(observation_before, observation_after)

    def test_optional_ndth_breadth_is_next_session_and_not_pit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            dates = ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"]
            self._prices(database, dates)
            breadth = Path(directory) / "ndth.db"
            persist_breadth_observations(breadth, [{
                "contract_version": BREADTH_CONTRACT, "source": BREADTH_SOURCE,
                "symbol": "NDTH", "observation_date": "2026-01-05",
                "available_at": "2026-01-05", "value_pct": 62.5, "fetched_at": "2026-01-06",
            }])
            panel = load_roro_panel(
                database, as_of=dates[-1], breadth_observation_database=breadth
            )
        self.assertTrue(pd.isna(panel.loc["2026-01-05", "breadth_200dma"]))
        self.assertEqual(panel.loc["2026-01-06", "breadth_200dma"], 0.625)
        self.assertEqual(
            panel.attrs["lineage"]["breadth_200dma"]["status"],
            "CONDITIONAL_EXTERNAL_BREADTH",
        )


if __name__ == "__main__":
    unittest.main()
