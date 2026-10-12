import unittest

import numpy as np
import pandas as pd

from roro_phase import build_trend_phase


def frames(periods: int = 50, *, direction: str = "bottom") -> tuple[pd.DataFrame, ...]:
    warmup = pd.bdate_range("2025-11-03", periods=15)
    index = pd.bdate_range("2026-01-07", periods=periods)
    full = warmup.append(index)
    x = np.arange(len(full), dtype=float)
    if direction == "bottom":
        nq = 100 + x
        vix, breadth, corr, portfolio_breadth = 35 - 0.2 * x, 0.2 + 0.005 * x, 0.8 - 0.005 * x, 0.3 + 0.005 * x
        hy = 6 - 0.02 * x
    else:
        nq = 200 - x
        vix, breadth, corr, portfolio_breadth = 15 + 0.2 * x, 0.8 - 0.005 * x, 0.2 + 0.005 * x, 0.8 - 0.005 * x
        hy = 2 + 0.02 * x
    market = pd.DataFrame(
        {
            "nq": nq,
            "vix": vix,
            "vix3m": 25.0,
            "hy_oas": hy,
            "breadth_200dma": breadth,
        },
        index=full,
    )
    portfolio = pd.DataFrame(
        {
            "portfolio_correlation_20": corr,
            "breadth_above_ma20": portfolio_breadth,
        },
        index=full,
    )
    score = pd.Series(80.0 if direction == "bottom" else 20.0, index=index)
    trajectory = pd.Series(0.0, index=index)
    axis = pd.DataFrame(
        {"market_risk_score": score, "trajectory_5d": trajectory}, index=index
    )
    return axis, market, portfolio


class RoroPhaseTest(unittest.TestCase):
    def test_bottom_checklist_and_bottoming_candidate(self) -> None:
        axis, market, portfolio = frames(direction="bottom")
        axis.iloc[-1, axis.columns.get_loc("market_risk_score")] = 70
        axis.iloc[-1, axis.columns.get_loc("trajectory_5d")] = -10
        result = build_trend_phase(axis, market, portfolio)
        row = result.iloc[-1]
        self.assertEqual(row["checklist_available_count"], 6)
        self.assertEqual(row["checklist_required_count"], 4)
        self.assertEqual(row["bottom_confirmation_count"], 6)
        self.assertTrue(row["bottom_checklist_pass"])
        self.assertEqual(row["trend_phase"], "BOTTOMING_CANDIDATE")
        self.assertEqual(len(row["turn_evidence"]), 6)

    def test_topping_candidate_is_the_frozen_mirror(self) -> None:
        axis, market, portfolio = frames(direction="top")
        axis.iloc[-1, axis.columns.get_loc("market_risk_score")] = 30
        axis.iloc[-1, axis.columns.get_loc("trajectory_5d")] = 10
        result = build_trend_phase(axis, market, portfolio)
        row = result.iloc[-1]
        self.assertEqual(row["top_confirmation_count"], 6)
        self.assertTrue(row["top_checklist_pass"])
        self.assertEqual(row["trend_phase"], "TOPPING_CANDIDATE")

    def test_recovery_uses_only_a_prior_bottoming_signal(self) -> None:
        axis, market, portfolio = frames(direction="bottom")
        bottom_date = axis.index[-6]
        axis.loc[bottom_date, ["market_risk_score", "trajectory_5d"]] = [70, -10]
        axis.loc[axis.index[-5]:, "market_risk_score"] = 70
        axis.loc[axis.index[-5]:, "trajectory_5d"] = 0
        result = build_trend_phase(axis, market, portfolio)
        self.assertTrue(result.loc[bottom_date, "bottoming_candidate_signal"])
        self.assertFalse(result.iloc[-1]["bottoming_candidate_signal"])
        self.assertTrue(result.iloc[-1]["prior_bottoming_candidate_20d"])
        self.assertEqual(result.iloc[-1]["trend_phase"], "RECOVERY_CONFIRMING")

    def test_missing_checklist_inputs_are_visible_and_fail_closed(self) -> None:
        axis, market, portfolio = frames(direction="bottom")
        axis.iloc[-1, axis.columns.get_loc("market_risk_score")] = 70
        axis.iloc[-1, axis.columns.get_loc("trajectory_5d")] = -10
        market.loc[axis.index[-1], ["hy_oas", "breadth_200dma"]] = np.nan
        four_available = build_trend_phase(axis, market, portfolio).iloc[-1]
        self.assertEqual(four_available["checklist_available_count"], 4)
        self.assertEqual(four_available["checklist_required_count"], 3)
        self.assertTrue(four_available["checklist_eligible"])
        self.assertTrue(four_available["bottom_checklist_pass"])

        market.loc[axis.index[-1], "vix"] = np.nan
        result = build_trend_phase(axis, market, portfolio)
        row = result.iloc[-1]
        self.assertEqual(row["checklist_available_count"], 3)
        self.assertFalse(row["checklist_eligible"])
        self.assertFalse(row["bottom_checklist_pass"])
        self.assertEqual(row["trend_phase"], "TRANSITION")
        self.assertCountEqual(row["missing_evidence"], ["VIX_CURVE", "HY_OAS", "MARKET_BREADTH"])

    def test_phase_precedence_and_generic_cases(self) -> None:
        axis, market, portfolio = frames(direction="bottom")
        axis.loc[:, "market_risk_score"] = 20
        result = build_trend_phase(axis, market, portfolio)
        self.assertEqual(result.iloc[-1]["trend_phase"], "UPTREND")

        axis.loc[:, "market_risk_score"] = 50
        axis.iloc[-1, axis.columns.get_loc("trajectory_5d")] = 2
        self.assertEqual(build_trend_phase(axis, market, portfolio).iloc[-1]["trend_phase"], "RISK_BUILDING")

        axis.iloc[-1, axis.columns.get_loc("trajectory_5d")] = 0
        self.assertEqual(build_trend_phase(axis, market, portfolio).iloc[-1]["trend_phase"], "TRANSITION")

        falling_axis, falling_market, falling_portfolio = frames(direction="top")
        falling_axis.loc[:, "market_risk_score"] = 80
        self.assertEqual(
            build_trend_phase(falling_axis, falling_market, falling_portfolio).iloc[-1]["trend_phase"],
            "DOWNTREND",
        )

        axis.iloc[-1, axis.columns.get_loc("market_risk_score")] = np.nan
        self.assertEqual(build_trend_phase(axis, market, portfolio).iloc[-1]["trend_phase"], "INSUFFICIENT_EVIDENCE")

    def test_axis_calendar_is_exact_and_market_dates_are_mandatory(self) -> None:
        axis, market, portfolio = frames()
        result = build_trend_phase(axis, market, portfolio)
        pd.testing.assert_index_equal(result.index, axis.index)
        no_fill = build_trend_phase(axis, market, portfolio.drop(axis.index[-1]))
        self.assertEqual(no_fill.iloc[-1]["checklist_available_count"], 4)
        self.assertCountEqual(
            no_fill.iloc[-1]["missing_evidence"],
            ["PORTFOLIO_CORRELATION", "PORTFOLIO_BREADTH_MA20"],
        )
        missing = market.drop(axis.index[4])
        with self.assertRaisesRegex(ValueError, "missing axis dates"):
            build_trend_phase(axis, missing, portfolio)

    def test_future_append_does_not_change_history(self) -> None:
        axis, market, portfolio = frames(periods=45)
        axis.iloc[-1, axis.columns.get_loc("market_risk_score")] = 70
        axis.iloc[-1, axis.columns.get_loc("trajectory_5d")] = -10
        before = build_trend_phase(axis, market, portfolio)
        future = pd.bdate_range(axis.index[-1] + pd.offsets.BDay(), periods=12)
        future_axis = pd.DataFrame(
            {"market_risk_score": np.linspace(60, 90, len(future)), "trajectory_5d": 10.0},
            index=future,
        )
        last_market = market.iloc[-1]
        future_market = pd.DataFrame([last_market.to_dict()] * len(future), index=future)
        last_portfolio = portfolio.iloc[-1]
        future_portfolio = pd.DataFrame([last_portfolio.to_dict()] * len(future), index=future)
        after = build_trend_phase(
            pd.concat([axis, future_axis]),
            pd.concat([market, future_market]),
            pd.concat([portfolio, future_portfolio]),
        )
        pd.testing.assert_frame_equal(before, after.loc[before.index])


if __name__ == "__main__":
    unittest.main()
