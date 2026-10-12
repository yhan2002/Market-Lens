import unittest

import numpy as np
import pandas as pd

from roro_axis import build_roro_axis_scores, long_exposure_from_risk_score


def inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    portfolio_index = pd.bdate_range("2022-12-01", "2026-06-30")
    x = np.arange(len(portfolio_index), dtype=float)
    portfolio = pd.DataFrame(
        {
            "portfolio_correlation_20": 0.25 + 0.12 * np.sin(x / 17),
            "nq_beta_60": 1.1 + 0.25 * np.cos(x / 23),
            "concentration_hhi": 0.08 + 0.015 * np.sin(x / 29),
            "negative_participation_20": 0.4 + 0.25 * np.cos(x / 13),
            "portfolio_drawdown_60": -0.06 - 0.04 * np.sin(x / 31),
            "breadth_above_ma20": 0.6 + 0.25 * np.sin(x / 11),
            "breadth_above_ma50": 0.58 + 0.2 * np.cos(x / 19),
            "breadth_above_ma200": 0.7 + 0.15 * np.sin(x / 37),
        },
        index=portfolio_index,
    )
    index = portfolio_index[portfolio_index >= "2026-01-02"]
    y = np.arange(len(index), dtype=float)
    roro = pd.DataFrame(
        {
            "credit_score": 45 + 12 * np.sin(y / 9),
            "rates_volatility_score": 52 + 10 * np.cos(y / 12),
            "equity_volatility_score": 48 + 14 * np.sin(y / 15),
            "fx_carry_score": 42 + 8 * np.cos(y / 7),
            "safe_haven_score": 46 + 9 * np.sin(y / 18),
            "equity_trend_score": 40 + 18 * np.cos(y / 16),
            "breadth_risk_percentile": 44 + 16 * np.sin(y / 14),
        },
        index=index,
    )
    return roro, portfolio


class RoroAxisTest(unittest.TestCase):
    def test_exact_axis_identity_integer_output_and_contributions(self) -> None:
        roro, portfolio = inputs()
        result = build_roro_axis_scores(roro, portfolio)
        row = result.dropna(subset=["market_risk_score"]).iloc[-1]
        axes = [
            row["macro_risk_score"],
            row["market_trend_score"],
            row["portfolio_internal_score"],
        ]
        self.assertAlmostEqual(row["market_risk_score_raw"], float(np.mean(axes)))
        self.assertEqual(row["market_risk_score"], np.floor(np.mean(axes) + 0.5))
        self.assertAlmostEqual(
            sum(
                row[column]
                for column in (
                    "macro_risk_contribution",
                    "market_trend_contribution",
                    "portfolio_internal_contribution",
                )
            ),
            row["market_risk_score_raw"] - 50,
        )
        component_contributions = [
            column
            for column in result
            if column.endswith("_contribution")
            and column
            not in {
                "macro_risk_contribution",
                "market_trend_contribution",
                "portfolio_internal_contribution",
            }
        ]
        self.assertAlmostEqual(
            float(row[component_contributions].sum()),
            row["market_risk_score_raw"] - 50,
        )
        self.assertEqual(
            row["credit_score_trajectory_5d"],
            result["credit_score"].diff(5).loc[row.name],
        )
        self.assertEqual(
            set(row["available_components"]),
            {
                "credit",
                "rates_volatility",
                "equity_volatility",
                "fx_carry",
                "safe_haven",
                "equity_trend",
                "market_breadth",
                "portfolio_correlation",
                "portfolio_beta",
                "portfolio_concentration",
                "portfolio_participation",
                "portfolio_drawdown",
            },
        )
        self.assertGreaterEqual(result.index.min(), pd.Timestamp("2025-09-02"))

    def test_future_rows_do_not_change_history(self) -> None:
        roro, portfolio = inputs()
        before = build_roro_axis_scores(roro, portfolio)
        future_index = pd.bdate_range(roro.index[-1] + pd.offsets.BDay(), periods=30)
        last_portfolio = portfolio.iloc[-1]
        future_portfolio = pd.DataFrame(
            [last_portfolio.to_numpy() * (1 + 0.001 * step) for step in range(1, 31)],
            index=future_index,
            columns=portfolio.columns,
        )
        future_roro = pd.DataFrame(
            [roro.iloc[-1].to_numpy() + step / 10 for step in range(1, 31)],
            index=future_index,
            columns=roro.columns,
        )
        after = build_roro_axis_scores(
            pd.concat([roro, future_roro]), pd.concat([portfolio, future_portfolio])
        )
        pd.testing.assert_frame_equal(before, after.loc[before.index])

    def test_missing_inputs_fail_closed_without_neutral_fill(self) -> None:
        roro, portfolio = inputs()
        roro["breadth_risk_percentile"] = np.nan
        nq_only = build_roro_axis_scores(roro, portfolio)
        valid = nq_only.dropna(subset=["market_risk_score"])
        self.assertGreater(len(valid), 0)
        self.assertTrue(valid["market_data_status"].eq("NQ_ONLY").all())
        self.assertTrue(valid["data_status"].eq("CONDITIONAL_EVENT_PIT").all())

        portfolio["breadth_above_ma20"] = np.nan
        failed = build_roro_axis_scores(roro, portfolio)
        self.assertTrue(failed["market_risk_score"].isna().all())
        self.assertTrue(failed["portfolio_internal_score"].isna().all())
        self.assertTrue(failed["data_status"].eq("NO_SCORE").all())

        _, diagnostic_only = inputs()
        diagnostic_only[["breadth_above_ma50", "breadth_above_ma200"]] = np.nan
        still_valid = build_roro_axis_scores(roro, diagnostic_only)
        self.assertGreater(len(still_valid.dropna(subset=["market_risk_score"])), 0)

    def test_portfolio_without_pre_2026_warmup_is_data_blocked(self) -> None:
        roro, portfolio = inputs()
        portfolio = portfolio.loc["2026-01-02":"2026-04-30"]
        result = build_roro_axis_scores(roro.loc[:"2026-04-30"], portfolio)
        self.assertTrue(result["market_risk_score"].isna().all())
        self.assertTrue(result["portfolio_data_status"].eq("DATA_BLOCKED").all())
        self.assertTrue(result["data_status"].eq("DATA_BLOCKED").all())

    def test_frozen_long_anchor_boundaries(self) -> None:
        scores = pd.Series([0, 33, 50, 66, 100, np.nan])
        exposure = long_exposure_from_risk_score(scores)
        self.assertEqual(exposure.iloc[:5].tolist(), [100, 90, 75, 60, 25])
        self.assertTrue(pd.isna(exposure.iloc[-1]))

    def test_output_start_cannot_precede_frozen_date(self) -> None:
        roro, portfolio = inputs()
        with self.assertRaisesRegex(ValueError, "cannot precede"):
            build_roro_axis_scores(roro, portfolio, output_start="2025-09-01")


if __name__ == "__main__":
    unittest.main()
