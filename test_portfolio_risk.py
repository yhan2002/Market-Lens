import unittest
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

from portfolio_risk import (
    DAILY_HOLDINGS_SOURCE,
    EVENT_HOLDINGS_SOURCE,
    average_portfolio_correlation,
    backfill_portfolio_risk_metrics,
    calculate_portfolio_risk_from_database,
    portfolio_beta,
    portfolio_holdings_date_is_valid,
    portfolio_internal_snapshot,
    position_quality_score,
    rank_positions_for_reduction,
    score_position_quality_table,
    store_portfolio_risk_metric,
)


class PortfolioRiskTest(unittest.TestCase):
    def test_event_holdings_remain_valid_until_the_next_change(self) -> None:
        self.assertTrue(
            portfolio_holdings_date_is_valid(
                "2026-08-11", "2026-07-01", EVENT_HOLDINGS_SOURCE
            )
        )
        self.assertFalse(
            portfolio_holdings_date_is_valid(
                "2026-08-11", "2026-07-01", DAILY_HOLDINGS_SOURCE
            )
        )
        self.assertFalse(
            portfolio_holdings_date_is_valid(
                "2026-08-11", "2026-08-12", EVENT_HOLDINGS_SOURCE
            )
        )

    def test_average_correlation_uses_current_long_weights(self) -> None:
        returns = pd.DataFrame(
            {
                "A": np.linspace(-0.01, 0.02, 30),
                "B": np.linspace(-0.02, 0.04, 30),
                "C": np.linspace(0.03, -0.01, 30),
            }
        )
        prices = 100 * (1 + returns).cumprod()

        correlation = average_portfolio_correlation(
            prices, {"A": 0.6, "B": 0.3, "C": 0.1}
        )

        self.assertGreater(correlation, 0.0)
        self.assertLessEqual(correlation, 1.0)
        self.assertEqual(
            average_portfolio_correlation(prices[["A"]], {"A": 1.0}),
            1.0,
        )
        newly_listed = prices.copy()
        newly_listed.loc[newly_listed.index[:-11], "C"] = np.nan
        self.assertTrue(
            np.isfinite(
                average_portfolio_correlation(
                    newly_listed,
                    {"A": 0.45, "B": 0.45, "C": 0.10},
                )
            )
        )

    def test_beta_recovers_linear_portfolio_sensitivity(self) -> None:
        benchmark_returns = pd.Series(np.linspace(-0.015, 0.02, 80))
        prices = pd.DataFrame(
            {
                "A": 100 * (1 + benchmark_returns * 2.0).cumprod(),
                "B": 100 * (1 + benchmark_returns).cumprod(),
            }
        )
        benchmark = 100 * (1 + benchmark_returns).cumprod()

        beta = portfolio_beta(prices, {"A": 0.5, "B": 0.5}, benchmark)

        self.assertAlmostEqual(beta, 1.5, places=10)

    def test_internal_snapshot_is_aggregate_causal_and_weighted(self) -> None:
        dates = pd.bdate_range("2025-01-02", periods=320)
        market_returns = pd.Series(
            0.0003 + 0.008 * np.sin(np.arange(len(dates)) / 9), index=dates
        )
        prices = pd.DataFrame(
            {
                "A": 100 * (1 + market_returns * 1.4).cumprod(),
                "B": 100 * (1 + market_returns * 0.8).cumprod(),
            },
            index=dates,
        )
        benchmark = 100 * (1 + market_returns).cumprod()
        cutoff = dates[-21]

        snapshot = portfolio_internal_snapshot(
            prices, {"A": 0.75, "B": 0.25}, benchmark, as_of=cutoff
        )
        self.assertEqual(snapshot["date"], cutoff.date().isoformat())
        self.assertAlmostEqual(snapshot["concentration_hhi"], 0.625)
        self.assertAlmostEqual(snapshot["effective_positions"], 1.6)
        self.assertAlmostEqual(snapshot["beta_60"], 1.25, places=10)
        self.assertGreater(snapshot["average_correlation_20"], 0.99)
        self.assertTrue(0 <= snapshot["breadth_above_ma200"] <= 1)
        self.assertTrue(0 <= snapshot["negative_participation_20"] <= 1)
        self.assertLessEqual(snapshot["portfolio_drawdown_60"], 0)
        self.assertNotIn("A", snapshot)
        self.assertNotIn("B", snapshot)

        changed_future = prices.copy()
        changed_future.loc[changed_future.index > cutoff, "A"] *= 5
        self.assertEqual(
            snapshot,
            portfolio_internal_snapshot(
                changed_future, {"A": 0.75, "B": 0.25}, benchmark, as_of=cutoff
            ),
        )

    def test_internal_snapshot_fails_when_weight_coverage_is_low(self) -> None:
        dates = pd.bdate_range("2025-01-02", periods=260)
        prices = pd.DataFrame({"A": np.linspace(100, 130, len(dates))}, index=dates)
        benchmark = pd.Series(np.linspace(100, 120, len(dates)), index=dates)
        with self.assertRaisesRegex(ValueError, "coverage"):
            portfolio_internal_snapshot(
                prices, {"A": 0.80, "MISSING": 0.20}, benchmark
            )

    def test_position_quality_uses_pdf_weights_and_integer_rounding(self) -> None:
        score = position_quality_score(
            fundamental_momentum=90,
            earnings_revision=80,
            relative_strength=70,
            liquidity=60,
        )

        self.assertEqual(score, 79)

    def test_weakest_position_is_reduced_first_without_order_logic(self) -> None:
        self.assertEqual(
            rank_positions_for_reduction({"A": 95, "C": 55, "B": 70}),
            ["C", "B", "A"],
        )

    def test_manager_supplied_quality_table_is_ranked_without_orders(self) -> None:
        frame = pd.DataFrame(
            [
                {"ticker": "A", "fundamental_momentum": 90, "earnings_revision": 90,
                 "relative_strength": 90, "liquidity": 90},
                {"ticker": "B", "fundamental_momentum": 50, "earnings_revision": 50,
                 "relative_strength": 50, "liquidity": 50},
            ]
        )

        scored = score_position_quality_table(frame)

        self.assertEqual(scored["ticker"].tolist(), ["B", "A"])
        self.assertEqual(scored["quality_score"].tolist(), [50, 90])

    def test_invalid_scores_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            position_quality_score(
                fundamental_momentum=101,
                earnings_revision=80,
                relative_strength=70,
                liquidity=60,
            )

    def test_database_calculation_persists_only_aggregate_metric(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=90)
        benchmark_returns = np.linspace(-0.01, 0.015, len(dates))
        series = {
            "SPY": 100 * (1 + benchmark_returns).cumprod(),
            "A": 100 * (1 + benchmark_returns * 1.2).cumprod(),
            "B": 100 * (1 + benchmark_returns * 0.8).cumprod(),
        }
        with tempfile.TemporaryDirectory() as directory:
            holdings_database = Path(directory) / "private.db"
            prices_database = Path(directory) / "publish.db"
            with closing(sqlite3.connect(holdings_database)) as connection:
                connection.executescript(
                    """
                    CREATE TABLE daily_holdings(
                        date TEXT, ticker TEXT, quantity REAL, market_value_base REAL
                    );
                    """
                )
                holding_date = dates[-1].date().isoformat()
                connection.executemany(
                    "INSERT INTO daily_holdings VALUES (?, ?, ?, ?)",
                    [(holding_date, "A", 0.6, 0.0), (holding_date, "B", 0.2, 0.0)],
                )
                connection.commit()
            with closing(sqlite3.connect(prices_database)) as connection:
                connection.execute(
                    """
                    CREATE TABLE daily_prices(
                        ticker TEXT, date TEXT, close REAL, adj_close REAL
                    )
                    """
                )
                connection.executemany(
                    "INSERT INTO daily_prices VALUES (?, ?, ?, ?)",
                    [
                        (ticker, current.date().isoformat(), float(value), float(value))
                        for ticker, values in series.items()
                        for current, value in zip(dates, values)
                    ],
                )
                connection.commit()

            metric = calculate_portfolio_risk_from_database(
                holdings_database,
                holding_date,
                prices_database=prices_database,
                benchmark="SPY",
            )
            self.assertIn("concentration_hhi", metric)
            self.assertIn("breadth_above_ma20", metric)
            self.assertNotIn("ticker", metric)
            store_portfolio_risk_metric(holdings_database, metric)

            with closing(sqlite3.connect(holdings_database)) as connection:
                columns = [
                    row[1]
                    for row in connection.execute("PRAGMA table_info(portfolio_risk_metrics)")
                ]
                saved = connection.execute(
                    "SELECT average_correlation, coverage FROM portfolio_risk_metrics"
                ).fetchone()
            self.assertNotIn("ticker", columns)
            self.assertGreater(saved[0], 0.99)
            self.assertEqual(saved[1], 1.0)

            target_database = Path(directory) / "public.db"
            metrics = backfill_portfolio_risk_metrics(
                holdings_database,
                holding_date,
                end=holding_date,
                prices_database=prices_database,
                target_database=target_database,
                benchmark="SPY",
            )
            with closing(sqlite3.connect(target_database)) as connection:
                public_count = connection.execute(
                    "SELECT COUNT(*) FROM portfolio_risk_metrics"
                ).fetchone()[0]
            self.assertEqual(len(metrics), 1)
            self.assertEqual(public_count, 1)

    def test_database_calculation_can_read_benchmark_from_separate_store(self) -> None:
        dates = pd.bdate_range("2025-01-02", periods=260)
        returns = pd.Series(0.001 + 0.005 * np.sin(np.arange(len(dates)) / 7))
        with tempfile.TemporaryDirectory() as directory:
            private = Path(directory) / "private.db"
            public = Path(directory) / "public.db"
            holding_date = dates[-1].date().isoformat()
            with closing(sqlite3.connect(private)) as connection:
                connection.executescript(
                    """
                    CREATE TABLE daily_holdings(
                        date TEXT, ticker TEXT, quantity REAL, market_value_base REAL
                    );
                    CREATE TABLE daily_prices(
                        ticker TEXT, date TEXT, close REAL, adj_close REAL
                    );
                    """
                )
                connection.executemany(
                    "INSERT INTO daily_holdings VALUES (?, ?, ?, 0)",
                    [(holding_date, "A", 1.0), (holding_date, "B", 1.0)],
                )
                connection.executemany(
                    "INSERT INTO daily_prices VALUES (?, ?, ?, ?)",
                    [
                        (ticker, day.date().isoformat(), float(price), float(price))
                        for ticker, multiplier in (("A", 1.2), ("B", 0.8))
                        for day, price in zip(dates, 100 * (1 + returns * multiplier).cumprod())
                    ],
                )
                connection.commit()
            with closing(sqlite3.connect(public)) as connection:
                connection.execute(
                    "CREATE TABLE daily_prices(ticker TEXT,date TEXT,close REAL,adj_close REAL)"
                )
                connection.executemany(
                    "INSERT INTO daily_prices VALUES ('NQ=F', ?, ?, ?)",
                    [
                        (day.date().isoformat(), float(price), float(price))
                        for day, price in zip(dates, 100 * (1 + returns).cumprod())
                    ],
                )
                connection.commit()

            metric = calculate_portfolio_risk_from_database(
                private,
                holding_date,
                prices_database=private,
                benchmark_database=public,
                benchmark="NQ=F",
                holdings_source=EVENT_HOLDINGS_SOURCE,
            )
            self.assertEqual(metric["date"], holding_date)
            self.assertAlmostEqual(metric["beta"], 1.0, delta=0.02)
            self.assertEqual(metric["benchmark_ticker"], "NQ=F")
            self.assertEqual(metric["metric_version"], "portfolio-internals-nq-v1")
            self.assertEqual(metric["weight_method"], "WEIGHT_PROXY_ADJ_CLOSE")
            self.assertIn("negative_participation_20", metric)

    def test_aggregate_store_rejects_bad_provenance_and_stale_holdings(self) -> None:
        base = {
            "date": "2026-08-07",
            "holdings_date": "2026-08-06",
            "average_correlation": 0.5,
            "beta": 1.1,
            "coverage": 0.95,
            "source": "DAILY_HOLDINGS_ADJUSTED_CLOSE",
        }
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "risk.db"
            for invalid in (
                {**base, "source": "PROXY"},
                {**base, "holdings_date": "2026-07-01"},
                {**base, "average_correlation": 1.1},
                {**base, "coverage": 0.89},
            ):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    store_portfolio_risk_metric(database, invalid)
            store_portfolio_risk_metric(
                database,
                {
                    **base,
                    "holdings_date": "2026-07-01",
                    "source": EVENT_HOLDINGS_SOURCE,
                },
            )


if __name__ == "__main__":
    unittest.main()
