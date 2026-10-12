import unittest

import numpy as np
import pandas as pd

from roro_engine import BLOCKS, build_roro_scores, causal_midrank_percentile


def sample_panel(count: int = 1800, *, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2020-01-02", periods=count)
    nq_returns = rng.normal(0.0004, 0.012, count)
    return pd.DataFrame(
        {
            "nq": 100 * np.exp(np.cumsum(nq_returns)),
            "hy_oas": np.clip(3.5 + np.cumsum(rng.normal(0, 0.025, count)), 1, 12),
            "move": np.clip(100 + np.cumsum(rng.normal(0, 0.7, count)), 45, 220),
            "vix": np.clip(19 + np.cumsum(rng.normal(0, 0.25, count)), 9, 70),
            "vix3m": np.clip(21 + np.cumsum(rng.normal(0, 0.2, count)), 10, 65),
            "usdjpy": np.clip(110 * np.exp(np.cumsum(rng.normal(0, 0.003, count))), 70, 180),
            "gold": np.clip(1800 * np.exp(np.cumsum(rng.normal(0, 0.006, count))), 800, 5000),
            "breadth_200dma": np.clip(0.6 + np.cumsum(rng.normal(0, 0.01, count)), 0.05, 0.95),
            "portfolio_correlation_20": np.clip(
                0.3 + np.cumsum(rng.normal(0, 0.008, count)), -0.2, 0.95
            ),
        },
        index=index,
    )


class RoroEngineTest(unittest.TestCase):
    def test_midrank_uses_only_prior_history(self) -> None:
        values = pd.Series([1.0, 2.0, 2.0, 2.0])
        scores = causal_midrank_percentile(
            values, window=3, min_history=3, min_unique=2
        )
        self.assertTrue(scores.iloc[:3].isna().all())
        self.assertAlmostEqual(scores.iloc[3], 0.625)

    def test_score_is_causal_integer_and_starts_at_history_origin(self) -> None:
        panel = sample_panel()
        result = build_roro_scores(panel)
        available = result.dropna(subset=["risk_score"])
        self.assertGreater(len(available), 0)
        self.assertEqual(result.index.min(), pd.Timestamp("2025-09-02"))
        self.assertTrue(available["risk_score"].between(0, 100).all())
        self.assertTrue(all(float(value).is_integer() for value in available["risk_score"]))

        future = sample_panel(30, seed=7)
        future.index = pd.bdate_range(panel.index[-1] + pd.offsets.BDay(), periods=30)
        future = future / future.iloc[0] * panel.iloc[-1]
        extended = build_roro_scores(pd.concat([panel, future]))
        pd.testing.assert_frame_equal(
            result,
            extended.loc[result.index],
        )

    def test_output_cannot_start_before_history_origin(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot precede"):
            build_roro_scores(sample_panel(), output_start="2025-09-01")

    def test_equal_weight_identity_and_drivers(self) -> None:
        row = build_roro_scores(sample_panel()).dropna(subset=["risk_score"]).iloc[-1]
        block_scores = [row[f"{block}_score"] for block in BLOCKS if pd.notna(row[f"{block}_score"])]
        self.assertAlmostEqual(row["score_raw"], float(np.mean(block_scores)))
        contributions = [
            row[f"{block}_contribution"]
            for block in BLOCKS
            if pd.notna(row[f"{block}_contribution"])
        ]
        self.assertAlmostEqual(sum(contributions), row["score_raw"] - 50)
        self.assertLessEqual(len(row["risk_drivers"]), 3)
        self.assertLessEqual(len(row["relief_drivers"]), 3)

    def test_missing_blocks_fail_closed_without_neutral_imputation(self) -> None:
        panel = sample_panel()
        panel["gold"] = np.nan
        partial = build_roro_scores(panel).dropna(subset=["risk_score"])
        self.assertGreater(len(partial), 0)
        self.assertTrue(partial["data_status"].eq("AS_AVAILABLE").all())
        self.assertTrue(partial["available_block_count"].eq(6).all())
        self.assertTrue(
            partial["available_blocks"].map(lambda blocks: "safe_haven" not in blocks).all()
        )

        panel[["hy_oas", "breadth_200dma", "portfolio_correlation_20"]] = np.nan
        failed = build_roro_scores(panel)
        self.assertTrue(failed["risk_score"].isna().all())
        self.assertTrue(failed["data_status"].eq("NO_SCORE").all())

    def test_equity_trend_and_volatility_are_mandatory(self) -> None:
        panel = sample_panel()
        panel["vix"] = np.nan
        result = build_roro_scores(panel)
        self.assertTrue(result["risk_score"].isna().all())

    def test_constant_history_fails_variation_gate(self) -> None:
        values = pd.Series(np.ones(400))
        self.assertTrue(
            causal_midrank_percentile(
                values, window=300, min_history=252, min_unique=20
            ).isna().all()
        )

    def test_trajectory_matches_score_and_block_differences(self) -> None:
        result = build_roro_scores(sample_panel()).dropna(subset=["risk_score"])
        self.assertGreater(len(result), 21)
        self.assertAlmostEqual(
            result["trajectory_5d"].iloc[-1],
            result["risk_score"].iloc[-1] - result["risk_score"].iloc[-6],
        )
        self.assertAlmostEqual(
            result["credit_trajectory_20d"].iloc[-1],
            result["credit_score"].iloc[-1] - result["credit_score"].iloc[-21],
        )


if __name__ == "__main__":
    unittest.main()
