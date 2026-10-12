import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

from crash_guard import (
    REQUIRED_TICKERS,
    build_crash_guard_scores,
    get_crash_guard_narrative_context,
    get_crash_guard_status,
)


def sample_prices(dates: pd.DatetimeIndex) -> pd.DataFrame:
    step = np.arange(len(dates), dtype=float)
    return pd.DataFrame(
        {
            "NQ=F": 100 * np.exp(0.0007 * step + 0.015 * np.sin(step / 8)),
            "SPY": 100 * np.exp(0.0005 * step + 0.012 * np.sin(step / 8)),
            "^VIX": 18 * np.exp(0.08 * np.sin(step / 5) + 0.02 * np.cos(step / 2)),
            "HYG": 80 * np.exp(0.0002 * step + 0.006 * np.sin(step / 7)),
            "IEF": 95 * np.exp(0.0001 * step + 0.004 * np.cos(step / 9)),
            "RSP": 120 * np.exp(0.0004 * step + 0.010 * np.sin(step / 10)),
            "IWM": 105 * np.exp(0.0003 * step + 0.014 * np.cos(step / 11)),
        },
        index=dates,
    )


class CrashGuardTest(unittest.TestCase):
    def test_scores_are_bounded_and_combined_is_max(self) -> None:
        prices = sample_prices(pd.bdate_range("2023-01-02", periods=150))
        result = build_crash_guard_scores(prices, window=80, min_history=20)
        finite = result[["score_1d", "score_5d", "score"]].stack().dropna()

        self.assertTrue(finite.between(0.0, 1.0).all())
        valid = result.dropna(subset=["score_1d", "score_5d", "score"])
        pd.testing.assert_series_equal(
            valid["score"], valid[["score_1d", "score_5d"]].max(axis=1), check_names=False
        )
        self.assertEqual(result.iloc[0]["risk_band"], "WARMUP")
        self.assertTrue(
            set(result["risk_band"])
            <= {
                "WARMUP",
                "NORMAL_SHADOW",
                "VULNERABLE_SHADOW",
                "CRASH_WATCH_SHADOW",
            }
        )

    def test_future_changes_do_not_rewrite_past_scores(self) -> None:
        prices = sample_prices(pd.bdate_range("2023-01-02", periods=150))
        original = build_crash_guard_scores(prices, window=80, min_history=20)
        changed = prices.copy()
        changed.iloc[111:] *= np.linspace(0.7, 1.4, len(changed) - 111)[:, None]
        revised = build_crash_guard_scores(changed, window=80, min_history=20)

        pd.testing.assert_frame_equal(original.iloc[:111], revised.iloc[:111])

    def test_score_remains_active_beyond_three_percent_drawdown(self) -> None:
        prices = sample_prices(pd.bdate_range("2023-01-02", periods=150))
        peak = float(prices["NQ=F"].iloc[:-5].max())
        prices.loc[prices.index[-5:], "NQ=F"] = np.linspace(peak * 0.96, peak * 0.88, 5)
        result = build_crash_guard_scores(prices, window=80, min_history=20)

        self.assertLess(float(prices["NQ=F"].iloc[-1] / peak - 1), -0.03)
        self.assertTrue(np.isfinite(result.iloc[-1]["score"]))
        self.assertNotEqual(result.iloc[-1]["risk_band"], "WARMUP")

    def test_status_uses_previous_complete_trading_day(self) -> None:
        dates = pd.bdate_range(end="2024-05-06", periods=90)
        prices = sample_prices(dates)
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "router.db"
            with closing(sqlite3.connect(database)) as connection:
                connection.execute(
                    "CREATE TABLE daily_prices(ticker TEXT, date TEXT, close REAL, adj_close REAL)"
                )
                rows = [
                    (ticker, str(day.date()), float(prices.loc[day, ticker]), float(prices.loc[day, ticker]))
                    for day in dates
                    for ticker in REQUIRED_TICKERS
                ]
                connection.executemany("INSERT INTO daily_prices VALUES(?,?,?,?)", rows)
                connection.commit()

            status = get_crash_guard_status(database, "2024-05-05")

            self.assertEqual(status["date"], "2024-05-03")
            self.assertEqual(status["model_version"], "crash-fixed-rank-nq-v2-shadow")
            self.assertEqual(status["score"], max(status["score_1d"], status["score_5d"]))
            self.assertFalse(status["score_is_probability"])
            context = get_crash_guard_narrative_context(database, "2024-05-05")
            self.assertEqual(context["date"], "2024-05-03")
            self.assertEqual(context["vulnerable_cutoff"], 0.60)
            self.assertEqual(context["crash_watch_cutoff"], 0.80)
            self.assertEqual(len(context["components"]), 10)
            self.assertIn("percentile", context["components"]["vix_jump_1d"])
            with self.assertRaisesRegex(ValueError, "YYYY-MM-DD"):
                get_crash_guard_status(database, "2024/05/05")


if __name__ == "__main__":
    unittest.main()
