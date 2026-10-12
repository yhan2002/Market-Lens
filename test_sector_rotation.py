import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from sector_rotation import (
    build_rrg_signals,
    classify_quadrants,
    evaluate_shadow,
    latest_snapshot,
    register_sector_universe,
    SECTOR_ETFS,
)


class SectorRotationTest(unittest.TestCase):
    @patch("market_data.register_universe", return_value=12)
    def test_sector_download_group_keeps_spy_as_its_benchmark(self, register) -> None:
        self.assertEqual(register_sector_universe(Path("prices.db")), 12)
        register.assert_called_once_with(
            Path("prices.db"),
            ("SPY", *SECTOR_ETFS),
            group_name="SP500_SECTOR_ETF",
            source="built-in S&P 500 sector ETF proxies",
        )

    def test_quadrants_follow_relative_strength_and_momentum(self) -> None:
        ratio = pd.Series([101, 101, 99, 99])
        momentum = pd.Series([101, 99, 99, 101])
        self.assertEqual(
            classify_quadrants(ratio, momentum).tolist(),
            ["LEADING", "WEAKENING", "LAGGING", "IMPROVING"],
        )

    def test_rrg_signal_is_causal(self) -> None:
        dates = pd.bdate_range("2020-01-01", periods=220)
        base = np.linspace(100, 150, len(dates))
        prices = pd.DataFrame(
            {
                "SPY": base,
                "XLK": base * (1 + 0.001 * np.arange(len(dates))),
                "XLF": base * (1 - 0.0005 * np.arange(len(dates))),
            },
            index=dates,
        )
        original = build_rrg_signals(prices, ratio_window=30, momentum_lag=5)
        changed = prices.copy()
        changed.loc[dates[-10]:, "XLK"] *= 2
        revised = build_rrg_signals(changed, ratio_window=30, momentum_lag=5)
        cutoff = dates[-11]
        columns = ["date", "ticker", "rs_ratio", "rs_momentum", "shadow_score", "quadrant"]
        pd.testing.assert_frame_equal(
            original.loc[original["date"] <= cutoff, columns].reset_index(drop=True),
            revised.loc[revised["date"] <= cutoff, columns].reset_index(drop=True),
        )

    def test_latest_snapshot_reports_tail_direction_and_transition(self) -> None:
        dates = pd.bdate_range("2026-07-27", periods=6)
        signals = pd.DataFrame(
            {
                "date": dates,
                "ticker": "XLI",
                "sector": "Industrials",
                "rs_ratio": [99, 99.2, 99.5, 100, 100.5, 101],
                "rs_momentum": [99, 99.1, 99.4, 100, 100.5, 101],
                "shadow_score": np.linspace(0.5, 1, len(dates)),
                "quadrant": ["LAGGING", "LAGGING", "LAGGING", "LEADING", "LEADING", "LEADING"],
            }
        )

        current = latest_snapshot(signals, tail_days=5).iloc[0]

        self.assertEqual(current["tail_heading"], "NE_STRENGTHENING")
        self.assertEqual(current["quadrant_transition"], "LAGGING->LEADING")
        self.assertAlmostEqual(current["delta_rs_ratio"], 2.0)
        self.assertAlmostEqual(current["delta_rs_momentum"], 2.0)

    def test_shadow_evaluation_keeps_regime_as_attribution(self) -> None:
        dates = pd.bdate_range("2020-01-01", periods=100)
        prices = pd.DataFrame(
            {
                "SPY": np.linspace(100, 110, len(dates)),
                "XLK": np.linspace(100, 140, len(dates)),
                "XLF": np.linspace(100, 90, len(dates)),
            },
            index=dates,
        )
        signals = pd.DataFrame(
            [
                {"date": day, "ticker": ticker, "shadow_score": score}
                for day in dates
                for ticker, score in (("XLK", 1.0), ("XLF", 0.0))
            ]
        )
        regime = pd.DataFrame(
            {
                "date": dates,
                "p_bull": 0.8,
                "p_sideways": 0.1,
                "p_bear": 0.1,
            }
        )
        summary, observations = evaluate_shadow(
            prices, signals, regime, horizon=5, top_k=1
        )
        self.assertGreater(summary["mean_excess_vs_equal"], 0)
        self.assertEqual(set(observations["regime"]), {"BULL"})
        self.assertEqual(summary["verdict"], "INSUFFICIENT")


if __name__ == "__main__":
    unittest.main()
