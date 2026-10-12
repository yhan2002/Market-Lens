from __future__ import annotations

import hashlib
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from roro_portfolio import (
    DATA_QUALITY,
    INCOMPLETE_SNAPSHOT_MARKER,
    SHORT_HISTORY_EXCLUSION_MARKER,
    SHADOW_START,
    WEIGHT_SOURCE_ACTION,
    load_portfolio_internal_series,
)


class RoroPortfolioSeriesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.private = root / "private.db"
        self.public = root / "public.db"
        self.dates = pd.bdate_range("2025-01-02", "2026-06-30")
        x = np.arange(len(self.dates), dtype=float)
        self.nq_returns = 0.0005 + 0.004 * np.sin(x / 9) + 0.001 * np.cos(x / 4)
        self.prices = {
            "NQ=F": 100 * np.cumprod(1 + self.nq_returns),
            "A": 80 * np.cumprod(1 + 1.2 * self.nq_returns + 0.001 * np.sin(x / 5)),
            "B": 120 * np.cumprod(1 + 0.7 * self.nq_returns - 0.001 * np.cos(x / 6)),
            "C": 60 * np.cumprod(1 + 1.0 * self.nq_returns + 0.002 * np.sin(x / 7)),
        }
        self._initialize_databases()

    def _initialize_databases(self) -> None:
        with closing(sqlite3.connect(self.private)) as connection:
            connection.executescript(
                """
                CREATE TABLE daily_holdings(
                    date TEXT, ticker TEXT, quantity REAL
                );
                CREATE TABLE daily_prices(
                    ticker TEXT, date TEXT, close REAL, adj_close REAL
                );
                """
            )
            connection.executemany(
                "INSERT INTO daily_holdings VALUES (?, ?, ?)",
                [
                    ("2025-12-31", "A", 1.0),
                    ("2025-12-31", "B", 1.0),
                    ("2026-03-02", "A", 2.0),
                    ("2026-03-02", "B", 1.0),
                    ("2026-03-02", "C", 1.0),
                ],
            )
            connection.executemany(
                "INSERT INTO daily_prices VALUES (?, ?, ?, ?)",
                [
                    (ticker, day.date().isoformat(), float(price), float(price))
                    for ticker in ("A", "B", "C")
                    for day, price in zip(self.dates, self.prices[ticker])
                ],
            )
            connection.commit()
        with closing(sqlite3.connect(self.public)) as connection:
            connection.execute(
                "CREATE TABLE daily_prices(ticker TEXT,date TEXT,close REAL,adj_close REAL)"
            )
            connection.executemany(
                "INSERT INTO daily_prices VALUES ('NQ=F', ?, ?, ?)",
                [
                    (day.date().isoformat(), float(price), float(price))
                    for day, price in zip(self.dates, self.prices["NQ=F"])
                ],
            )
            connection.commit()

    def test_carries_holdings_forward_without_using_future_snapshot(self) -> None:
        frame = load_portfolio_internal_series(
            self.private, self.public, as_of="2026-03-10"
        )

        self.assertEqual(frame.loc["2026-02-27", "holdings_date"], "2025-12-31")
        self.assertEqual(frame.loc["2026-03-02", "holdings_date"], "2026-03-02")
        self.assertEqual(frame.loc["2026-02-27", "data_quality"], DATA_QUALITY)

    def test_future_append_does_not_change_prior_output(self) -> None:
        before = load_portfolio_internal_series(
            self.private, self.public, as_of="2026-04-30"
        )
        with closing(sqlite3.connect(self.private)) as connection:
            connection.execute(
                "INSERT INTO daily_holdings VALUES ('2026-06-01', 'C', 100)"
            )
            connection.execute(
                "INSERT OR REPLACE INTO daily_prices VALUES ('A','2026-06-01',9999,9999)"
            )
            connection.commit()

        after = load_portfolio_internal_series(
            self.private, self.public, as_of="2026-04-30"
        )

        assert_frame_equal(before, after)

    def test_loader_does_not_write_source_databases(self) -> None:
        def digest(path: Path) -> str:
            return hashlib.sha256(path.read_bytes()).hexdigest()

        before = (digest(self.private), digest(self.public))
        load_portfolio_internal_series(
            self.private, self.public, as_of="2026-03-10"
        )

        self.assertEqual((digest(self.private), digest(self.public)), before)

    def test_reads_nq_from_separate_database_and_returns_metrics(self) -> None:
        frame = load_portfolio_internal_series(
            self.private, self.public, as_of="2026-06-30"
        )
        latest = frame.iloc[-1]

        self.assertTrue(np.isfinite(latest["nq_beta_60"]))
        self.assertTrue(np.isfinite(latest["portfolio_correlation_20"]))
        self.assertTrue(0 <= latest["breadth_above_ma20"] <= 1)
        self.assertGreaterEqual(latest["price_weight_coverage"], 0.90)
        self.assertEqual(frame.attrs["benchmark"], "NQ=F")

    def test_beta_requires_minimum_weight_coverage_each_day(self) -> None:
        with closing(sqlite3.connect(self.private)) as connection:
            connection.execute("DELETE FROM daily_holdings")
            connection.executemany(
                "INSERT INTO daily_holdings VALUES ('2025-12-31', ?, ?)",
                [("A", 1.0), ("B", 0.25)],
            )
            latest = self.dates[-1].date().isoformat()
            connection.execute(
                "DELETE FROM daily_prices WHERE ticker='B' AND date<>?", (latest,)
            )
            connection.commit()

        frame = load_portfolio_internal_series(
            self.private, self.public, as_of=self.dates[-1].date().isoformat()
        )

        self.assertEqual(frame.iloc[-1]["price_weight_coverage"], 1.0)
        self.assertTrue(np.isnan(frame.iloc[-1]["nq_beta_60"]))

    def test_missing_price_proxy_fails_closed(self) -> None:
        with closing(sqlite3.connect(self.private)) as connection:
            connection.execute("DELETE FROM daily_holdings")
            connection.executemany(
                "INSERT INTO daily_holdings VALUES ('2025-12-31', ?, ?)",
                [("A", 1.0), ("MISSING", 1.0)],
            )
            connection.commit()

        frame = load_portfolio_internal_series(
            self.private, self.public, as_of="2026-02-02"
        )
        latest = frame.iloc[-1]

        self.assertTrue(np.isnan(latest["price_weight_coverage"]))
        self.assertTrue(np.isnan(latest["portfolio_correlation_20"]))
        self.assertTrue(np.isnan(latest["nq_beta_60"]))

    def test_actual_weights_allow_small_missing_price_sleeve(self) -> None:
        with closing(sqlite3.connect(self.private)) as connection:
            connection.execute(
                "ALTER TABLE daily_holdings ADD COLUMN action_weight_source REAL"
            )
            connection.execute(
                "UPDATE daily_holdings SET action_weight_source="
                "CASE ticker WHEN 'A' THEN 60 WHEN 'B' THEN 35 ELSE 5 END"
            )
            connection.execute("DELETE FROM daily_prices WHERE ticker='C'")
            connection.commit()

        frame = load_portfolio_internal_series(
            self.private, self.public, as_of="2026-06-30"
        )
        latest = frame.iloc[-1]

        self.assertAlmostEqual(latest["price_weight_coverage"], 0.95)
        self.assertIn(WEIGHT_SOURCE_ACTION, latest["data_quality"])
        self.assertIn(SHORT_HISTORY_EXCLUSION_MARKER, latest["data_quality"])
        self.assertTrue(np.isfinite(latest["portfolio_correlation_20"]))

    def _replace_with_short_history_sleeve(self, sleeve_weight: float) -> None:
        with closing(sqlite3.connect(self.private)) as connection:
            connection.execute(
                "ALTER TABLE daily_holdings ADD COLUMN action_weight_source REAL"
            )
            connection.execute("DELETE FROM daily_holdings")
            connection.executemany(
                "INSERT INTO daily_holdings VALUES ('2025-12-31', ?, 1, ?)",
                [
                    ("A", 40.0),
                    ("B", 30.0),
                    ("C", 30.0 - sleeve_weight),
                    ("D", sleeve_weight),
                ],
            )
            connection.executemany(
                "INSERT INTO daily_prices VALUES ('D', ?, ?, ?)",
                [
                    (day.date().isoformat(), float(price), float(price))
                    for day, price in zip(
                        self.dates[-11:],
                        np.linspace(10.0, 12.0, 11),
                    )
                ],
            )
            connection.commit()

    def test_short_history_sleeve_is_excluded_with_exact_coverage(self) -> None:
        self._replace_with_short_history_sleeve(10.5365)

        latest = load_portfolio_internal_series(
            self.private, self.public, as_of="2026-06-30"
        ).iloc[-1]

        expected = 0.894635
        self.assertAlmostEqual(latest["history_weight_coverage_20d"], expected)
        self.assertAlmostEqual(latest["history_weight_coverage_60d"], expected)
        self.assertIn(SHORT_HISTORY_EXCLUSION_MARKER, latest["data_quality"])
        self.assertTrue(np.isfinite(latest["negative_participation_20"]))
        self.assertTrue(np.isfinite(latest["nq_beta_60"]))
        self.assertTrue(np.isfinite(latest["portfolio_drawdown_60"]))

    def test_more_than_fifteen_percent_short_history_fails_closed(self) -> None:
        self._replace_with_short_history_sleeve(15.1)

        latest = load_portfolio_internal_series(
            self.private, self.public, as_of="2026-06-30"
        ).iloc[-1]

        self.assertAlmostEqual(latest["history_weight_coverage_20d"], 0.849)
        self.assertAlmostEqual(latest["history_weight_coverage_60d"], 0.849)
        self.assertNotIn(SHORT_HISTORY_EXCLUSION_MARKER, latest["data_quality"])
        self.assertTrue(np.isnan(latest["negative_participation_20"]))
        self.assertTrue(np.isnan(latest["nq_beta_60"]))
        self.assertTrue(np.isnan(latest["portfolio_drawdown_60"]))

    def test_true_one_name_book_has_no_correlation(self) -> None:
        with closing(sqlite3.connect(self.private)) as connection:
            connection.execute("DELETE FROM daily_holdings")
            connection.execute(
                "INSERT INTO daily_holdings VALUES ('2025-12-31', 'A', 1)"
            )
            connection.commit()

        latest = load_portfolio_internal_series(
            self.private, self.public, as_of="2026-06-30"
        ).iloc[-1]

        self.assertEqual(latest["price_weight_coverage"], 1.0)
        self.assertTrue(np.isnan(latest["portfolio_correlation_20"]))

    def test_incomplete_one_name_snapshot_carries_prior_book_forward(self) -> None:
        with closing(sqlite3.connect(self.private)) as connection:
            connection.execute(
                "ALTER TABLE daily_holdings ADD COLUMN action_weight_source REAL"
            )
            connection.execute("UPDATE daily_holdings SET action_weight_source=30")
            connection.execute(
                "INSERT INTO daily_holdings VALUES ('2026-03-02','D',1,30)"
            )
            connection.execute(
                "INSERT INTO daily_holdings VALUES ('2026-03-10','A',1,5)"
            )
            connection.commit()

        frame = load_portfolio_internal_series(
            self.private, self.public, as_of="2026-03-10"
        )
        latest = frame.iloc[-1]

        self.assertEqual(latest["holdings_date"], "2026-03-02")
        self.assertIn(INCOMPLETE_SNAPSHOT_MARKER, latest["data_quality"])

    def test_output_exposes_no_holding_identity(self) -> None:
        frame = load_portfolio_internal_series(
            self.private, self.public, as_of="2026-03-10"
        )
        forbidden = {"ticker", "isin", "name", "quantity"}

        self.assertFalse(forbidden & {column.lower() for column in frame.columns})
        text_values = set(
            frame[["holdings_date", "data_quality"]].stack().astype(str)
        )
        self.assertFalse({"A", "B", "C"} & text_values)
        self.assertEqual(frame.attrs["identity_exposure"], "aggregate-only")

    def test_output_is_cut_off_at_2026_start(self) -> None:
        frame = load_portfolio_internal_series(
            self.private, self.public, as_of="2026-01-09"
        )

        self.assertGreaterEqual(frame.index.min(), SHADOW_START)
        self.assertEqual(frame.index.max(), pd.Timestamp("2026-01-09"))

    def test_pre_2026_rows_are_available_only_when_requested_as_warmup(self) -> None:
        public = load_portfolio_internal_series(
            self.private, self.public, as_of="2026-01-09"
        )
        warmup = load_portfolio_internal_series(
            self.private,
            self.public,
            as_of="2026-01-09",
            include_warmup=True,
        )

        self.assertGreaterEqual(public.index.min(), SHADOW_START)
        self.assertLess(warmup.index.min(), SHADOW_START)
        self.assertFalse(public.attrs["includes_warmup"])
        self.assertTrue(warmup.attrs["includes_warmup"])

    def test_correlation_trajectories_are_backwards_differences(self) -> None:
        frame = load_portfolio_internal_series(
            self.private, self.public, as_of="2026-06-30"
        )
        correlation = frame["portfolio_correlation_20"]

        for lag in (1, 5, 20):
            pd.testing.assert_series_equal(
                frame[f"portfolio_correlation_trajectory_{lag}d"].iloc[lag:],
                correlation.diff(lag).iloc[lag:],
                check_names=False,
            )


if __name__ == "__main__":
    unittest.main()
