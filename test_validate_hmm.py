import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

from validate_hmm import (
    _future_drawdown,
    _independent_episode_count,
    load_validation_frame,
    load_probability_history,
    paired_auc_comparison,
    posterior_health,
    score_metrics,
)


class HmmValidationTest(unittest.TestCase):
    def test_saturated_recent_probabilities_fail_health_gate(self) -> None:
        probabilities = pd.DataFrame(
            {
                "p_bull": np.full(252, 0.005),
                "p_sideways": np.full(252, 0.99),
                "p_bear": np.full(252, 0.005),
            }
        )
        health = posterior_health(probabilities)
        self.assertFalse(health["pass"])
        self.assertFalse(health["windows"]["63"]["checks"]["low_entropy_share"])
        self.assertFalse(health["windows"]["63"]["checks"]["score_diversity"])

    def test_moving_probabilities_pass_health_gate(self) -> None:
        phase = np.arange(252) // 2
        bear = 0.10 + 0.25 * (1 + np.sin(phase / 5)) / 2
        sideways = 0.15 + 0.25 * (1 + np.cos(phase / 7)) / 2
        bull = 1 - bear - sideways
        health = posterior_health(
            pd.DataFrame({"p_bull": bull, "p_sideways": sideways, "p_bear": bear})
        )
        self.assertTrue(health["pass"], health)

    def test_future_drawdown_uses_only_forward_window(self) -> None:
        prices = np.array([100.0, 95.0, 90.0, 110.0])
        result = _future_drawdown(prices, 2)
        self.assertAlmostEqual(result[0], -0.10)
        self.assertAlmostEqual(result[1], -0.0526315789)
        self.assertTrue(np.isnan(result[2]))

    def test_independent_episode_count_uses_contiguous_runs(self) -> None:
        self.assertEqual(_independent_episode_count(np.array([0, 1, 1, 0, 1, 0])), 2)
        self.assertEqual(
            _independent_episode_count(
                np.array([0, 1, 1, 0, 1, 0]), separation=4
            ),
            1,
        )

    def test_perfect_risk_score_has_perfect_auc(self) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=120, freq="D"),
                "drawdown_event": [0] * 60 + [1] * 60,
                "risk": np.linspace(0, 1, 120),
            }
        )
        metrics = score_metrics(frame, "risk", bootstrap_samples=50)
        self.assertEqual(metrics["roc_auc"], 1.0)
        self.assertGreater(metrics["top_quintile_lift"], 1.5)

    def test_paired_auc_uses_shared_bootstrap_samples(self) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=180, freq="D"),
                "drawdown_event": [0] * 90 + [1] * 90,
                "candidate": np.linspace(0, 1, 180),
                "champion": np.linspace(1, 0, 180),
            }
        )
        comparison = paired_auc_comparison(
            frame, "candidate", "champion", bootstrap_samples=50
        )
        self.assertEqual(comparison["candidate_auc"], 1.0)
        self.assertEqual(comparison["champion_auc"], 0.0)
        self.assertEqual(comparison["auc_delta"], 1.0)

    def test_candidate_model_and_start_date_are_selected(self) -> None:
        dates = pd.bdate_range("2022-01-03", periods=300)
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "router.db"
            with closing(sqlite3.connect(database)) as connection:
                connection.executescript(
                    """
                    CREATE TABLE daily_prices(
                        ticker TEXT, date TEXT, adj_close REAL
                    );
                    CREATE TABLE regime_model_probabilities(
                        model_version TEXT, date TEXT, p_bull REAL,
                        p_sideways REAL, p_bear REAL
                    );
                    """
                )
                price_rows = [
                    (ticker, day.strftime("%Y-%m-%d"), base + offset)
                    for ticker, base in (("NQ=F", 100.0), ("SPY", 1_000.0))
                    for offset, day in enumerate(dates)
                ]
                probability_rows = [
                    ("candidate", day.strftime("%Y-%m-%d"), 0.2, 0.3, 0.5)
                    for day in dates
                ]
                connection.executemany("INSERT INTO daily_prices VALUES (?, ?, ?)", price_rows)
                connection.executemany(
                    "INSERT INTO regime_model_probabilities VALUES (?, ?, ?, ?, ?)",
                    probability_rows,
                )
                connection.commit()

            frame, invalid = load_validation_frame(
                database,
                model_version="candidate",
                start_date="2023-01-01",
                horizon=5,
            )
            spy_frame, spy_invalid = load_validation_frame(
                database,
                benchmark="SPY",
                model_version="candidate",
                start_date="2023-01-01",
                horizon=5,
            )

            self.assertEqual(invalid, 0)
            self.assertEqual(spy_invalid, 0)
            self.assertGreaterEqual(frame["date"].min(), pd.Timestamp("2023-01-01"))
            self.assertTrue(np.allclose(frame["p_bear"], 0.5))
            self.assertLess(frame["adj_close"].iloc[0], spy_frame["adj_close"].iloc[0])

    def test_probability_history_keeps_latest_unlabelled_dates(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=300)
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "router.db"
            with closing(sqlite3.connect(database)) as connection:
                connection.execute(
                    """CREATE TABLE regime_model_probabilities(
                    model_version TEXT, date TEXT, p_bull REAL,
                    p_sideways REAL, p_bear REAL)"""
                )
                connection.executemany(
                    "INSERT INTO regime_model_probabilities VALUES (?, ?, ?, ?, ?)",
                    [
                        ("candidate", day.date().isoformat(), 0.2, 0.6, 0.2)
                        for day in dates
                    ],
                )
                connection.commit()
            history, invalid = load_probability_history(
                database, model_version="candidate"
            )
            self.assertEqual(invalid, 0)
            self.assertEqual(history["date"].max(), dates[-1])


if __name__ == "__main__":
    unittest.main()
