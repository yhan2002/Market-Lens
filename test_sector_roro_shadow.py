from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from sector_roro_shadow import (
    _equal_weight_turnover,
    _validate_frozen_spec,
    apply_roro_context,
    build_research_history,
    markdown_report,
    sector_beta_ranks,
    validate_sector_roro_shadow,
)


class SectorRoroShadowTest(unittest.TestCase):
    @patch("sector_roro_shadow.fit_sector_roro_shadow")
    @patch("sector_roro_shadow.latest_snapshot")
    @patch("sector_roro_shadow.fit_leading_forecast")
    def test_research_history_keeps_alpha_and_marks_missing_roro_na(
        self, forecast, snapshot, full_fit
    ) -> None:
        dates = pd.to_datetime(["2025-01-03", "2025-10-10", "2026-01-16"])
        tickers = ["XLK", "XLP"]
        features = pd.DataFrame(
            [
                {"date": day, "ticker": ticker, "target_excess_20d": value}
                for day in dates
                for ticker, value in zip(tickers, [0.02, -0.01], strict=True)
            ]
        )
        forecast.return_value = (
            pd.DataFrame(
                {
                    "date": [dates[0]] * 2,
                    "ticker": tickers,
                    "sector": ["Tech", "Staples"],
                    "shadow_score": [0.8, 0.2],
                    "predicted_excess_20d": [0.03, -0.02],
                    "forecast_score": [1.0, 0.0],
                    "forecast_rank": [1, 2],
                    "signal": ["MODEL_TOP3", "NEGATIVE"],
                }
            ),
            {},
        )
        snapshot.return_value = pd.DataFrame(
            {
                "ticker": tickers,
                "quadrant": ["LEADING", "LAGGING"],
                "quadrant_transition": ["IMPROVING->LEADING", "LAGGING->LAGGING"],
                "tail_heading": ["NE_STRENGTHENING", "SW_DETERIORATING"],
            }
        )

        def full(context: str, day: pd.Timestamp) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "date": [day] * 2,
                    "ticker": tickers,
                    "sector": ["Tech", "Staples"],
                    "shadow_score": [0.8, 0.2],
                    "predicted_excess_20d": [0.03, -0.02],
                    "forecast_score": [1.0, 0.0],
                    "forecast_rank": [1, 2],
                    "signal": ["MODEL_TOP3", "NEGATIVE"],
                    "quadrant": ["LEADING", "LAGGING"],
                    "quadrant_transition": ["LEADING->LEADING", "LAGGING->LAGGING"],
                    "tail_heading": ["NE_STRENGTHENING", "SW_DETERIORATING"],
                    "roro_fit": [0.4 if context == "RORO_MISMATCH" else 0.6] * 2,
                    "roro_context": [context] * 2,
                    "sector_status": ["WATCH", "AVOID"],
                }
            )

        full_fit.side_effect = [
            (full("RORO_MISMATCH", dates[1]), {}),
            (full("FIT", dates[2]), {}),
        ]
        history = build_research_history(
            features,
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame({"date": dates[1:]}),
            start="2025-01-01",
            as_of="2026-01-16",
        )

        tech = history.loc[history["ticker"] == "XLK"].set_index("decision_date")
        self.assertEqual(tech.at[dates[0], "roro_context"], "N/A")
        self.assertEqual(tech.at[dates[0], "fit_change"], "N/A")
        self.assertEqual(tech.at[dates[2], "fit_change"], "MISMATCH → FIT")
        self.assertEqual(tech.at[dates[0], "predicted_excess_20d"], 0.03)

    def test_frozen_version_rejects_parameter_changes(self) -> None:
        _validate_frozen_spec(156, 10.0, 3)
        for arguments in ((155, 10.0, 3), (156, 9.0, 3), (156, 10.0, 4)):
            with self.subTest(arguments=arguments), self.assertRaisesRegex(
                ValueError, "use a new model version"
            ):
                _validate_frozen_spec(*arguments)

    def test_beta_ranks_are_causal_and_ordered(self) -> None:
        dates = pd.bdate_range("2025-01-01", periods=100)
        benchmark = 100 * np.cumprod(1 + np.sin(np.arange(100)) * 0.001)
        prices = pd.DataFrame(
            {
                "SPY": benchmark,
                "XLK": 100 * np.cumprod(1 + np.sin(np.arange(100)) * 0.002),
                "XLP": 100 * np.cumprod(1 + np.sin(np.arange(100)) * 0.0005),
                "XLF": 100 * np.cumprod(1 + np.sin(np.arange(100)) * 0.0012),
            },
            index=dates,
        )
        cutoff = dates[-2]
        original = sector_beta_ranks(prices, cutoff)
        changed = prices.copy()
        changed.loc[dates[-1], "XLK"] *= 100
        repeated = sector_beta_ranks(changed, cutoff)
        pd.testing.assert_frame_equal(original, repeated)
        ranked = original.set_index("ticker")
        self.assertGreater(ranked.at["XLK", "beta_rank"], ranked.at["XLP", "beta_rank"])

    def test_status_table_keeps_alpha_rrg_and_roro_separate(self) -> None:
        forecast = pd.DataFrame(
            {
                "date": pd.Timestamp("2026-08-12"),
                "ticker": ["XLK", "XLP", "XLF"],
                "sector": ["Tech", "Staples", "Financials"],
                "shadow_score": [0.9, 0.2, 0.5],
                "predicted_excess_20d": [0.02, -0.01, 0.01],
                "forecast_score": [1.0, 0.0, 0.5],
                "forecast_rank": [1, 3, 2],
                "signal": ["MODEL_TOP3", "NEGATIVE", "MODEL_TOP3"],
            }
        )
        rrg = pd.DataFrame(
            {
                "ticker": ["XLK", "XLP", "XLF"],
                "quadrant": ["LEADING", "LAGGING", "WEAKENING"],
                "quadrant_transition": ["IMPROVING->LEADING", "LAGGING->LAGGING", "LEADING->WEAKENING"],
                "tail_heading": ["NE_STRENGTHENING", "SW_DETERIORATING", "SW_DETERIORATING"],
                "delta_rs_ratio": [1.0, -1.0, -0.5],
                "delta_rs_momentum": [1.0, -1.0, -0.5],
            }
        )
        beta = pd.DataFrame(
            {
                "ticker": ["XLK", "XLP", "XLF"],
                "sector_beta_60": [1.4, 0.6, 1.0],
                "beta_rank": [0.9, 0.1, 0.5],
            }
        )
        result = apply_roro_context(
            forecast,
            rrg,
            beta,
            pd.Series(
                {
                    "market_risk_score": 20,
                    "risk_regime": "GREEN",
                    "trajectory_5d": -5,
                    "data_status": "CONDITIONAL_EVENT_PIT",
                }
            ),
        ).set_index("ticker")
        self.assertEqual(result.at["XLK", "sector_status"], "SELECT")
        self.assertEqual(result.at["XLP", "sector_status"], "AVOID")
        self.assertEqual(result.at["XLF", "sector_status"], "WATCH")
        self.assertEqual(result.at["XLF", "alignment"], "DIVERGENCE")
        self.assertGreater(result.at["XLK", "roro_fit"], result.at["XLP", "roro_fit"])

    def test_equal_weight_turnover_handles_cash_and_set_changes(self) -> None:
        self.assertEqual(_equal_weight_turnover(set(), set()), 0.0)
        self.assertEqual(_equal_weight_turnover(set(), {"XLK", "XLF"}), 1.0)
        self.assertEqual(_equal_weight_turnover({"XLK", "XLF"}, set()), 1.0)
        self.assertAlmostEqual(
            _equal_weight_turnover({"XLK", "XLF"}, {"XLK", "XLP"}), 0.5
        )

    @patch("sector_roro_shadow.fit_sector_roro_shadow")
    def test_validation_records_skipped_week_reason(self, fit) -> None:
        tickers = ["XLK", "XLP", "XLF"]
        dates = [pd.Timestamp("2026-01-09"), pd.Timestamp("2026-01-16")]
        features = pd.DataFrame([
            {"date": day, "ticker": ticker, "target_excess_20d": value}
            for day in dates
            for ticker, value in zip(tickers, [0.02, -0.01, 0.01])
        ])
        ranked = pd.DataFrame({
            "ticker": tickers,
            "sector_status": ["SELECT", "WATCH", "AVOID"],
            "forecast_rank": [1, 2, 3],
            "predicted_excess_20d": [0.03, 0.01, -0.01],
            "alignment": ["ALIGNED", "DIVERGENCE", "ALIGNED"],
            "rrg_supportive": [True, False, False],
            "roro_fit": [0.8, 0.4, 0.2],
            "risk_regime": ["GREEN"] * 3,
        })
        fit.side_effect = [
            ValueError("need 156 known training weeks; found 12"),
            (ranked, {}),
        ]

        summary, _ = validate_sector_roro_shadow(
            features,
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            bootstrap_samples=20,
        )

        self.assertEqual(summary["coverage"]["candidate_weeks"], 2)
        self.assertEqual(summary["coverage"]["weeks"], 1)
        self.assertEqual(summary["coverage"]["skipped_weeks"], 1)
        self.assertEqual(
            summary["coverage"]["skipped"],
            [{"date": "2026-01-09", "reason": "need 156 known training weeks; found 12"}],
        )
        self.assertIn("null", markdown_report(summary))

    @patch("sector_roro_shadow.fit_sector_roro_shadow")
    def test_validation_propagates_unexpected_value_error(self, fit) -> None:
        features = pd.DataFrame(
            {
                "date": [pd.Timestamp("2026-01-09")] * 3,
                "ticker": ["XLK", "XLP", "XLF"],
                "target_excess_20d": [0.02, -0.01, 0.01],
            }
        )
        fit.side_effect = ValueError("broken forecast schema")
        with self.assertRaisesRegex(ValueError, "broken forecast schema"):
            validate_sector_roro_shadow(
                features,
                pd.DataFrame(),
                pd.DataFrame(),
                pd.DataFrame(),
                bootstrap_samples=20,
            )

    @patch("sector_roro_shadow.fit_sector_roro_shadow")
    def test_fixed_ablations_and_roro_regime_breakdown(self, fit) -> None:
        tickers = ["XLK", "XLP", "XLF", "XLE"]
        dates = [pd.Timestamp("2026-01-09"), pd.Timestamp("2026-01-16")]
        features = pd.DataFrame(
            [
                {
                    "date": day,
                    "ticker": ticker,
                    "target_excess_20d": value,
                }
                for day, values in zip(
                    dates,
                    ([0.02, -0.01, 0.01, -0.02], [-0.01, 0.03, 0.02, -0.02]),
                    strict=True,
                )
                for ticker, value in zip(tickers, values, strict=True)
            ]
        )

        def ranked(regime: str, second_week: bool) -> pd.DataFrame:
            positive = [-0.01, 0.03, 0.02, 0.01] if second_week else [0.03, 0.02, 0.01, 0.01]
            full = ["WATCH", "SELECT", "SELECT", "WATCH"] if second_week else ["SELECT", "WATCH", "WATCH", "WATCH"]
            return pd.DataFrame(
                {
                    "ticker": tickers,
                    "sector_status": full,
                    "forecast_rank": [1, 2, 3, 4],
                    "predicted_excess_20d": positive,
                    "alignment": ["ALIGNED"] * 4,
                    "rrg_supportive": [True, True, False, True]
                    if not second_week
                    else [True] * 4,
                    "roro_fit": [0.8, 0.4, 0.8, 0.8]
                    if not second_week
                    else [0.8] * 4,
                    "risk_regime": [regime] * 4,
                }
            )

        fit.side_effect = [(ranked("GREEN", False), {}), (ranked("RED", True), {})]
        summary, observations = validate_sector_roro_shadow(
            features,
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            bootstrap_samples=20,
        )

        self.assertEqual(list(summary["ablations"]), ["alpha-only", "alpha+RRG", "alpha+RORO", "full"])
        full = summary["ablations"]["full"]
        self.assertEqual(full["weeks"], 2)
        self.assertEqual(full["observations"], 3)
        self.assertEqual(full["hit_rate"], 1.0)
        self.assertAlmostEqual(full["mean_weekly_gross_excess"], 0.0225)
        self.assertAlmostEqual(full["mean_weekly_net_excess_10bp"], 0.0215)
        self.assertEqual(summary["roro_regime_breakdown"]["GREEN"]["weeks"], 1)
        self.assertEqual(summary["roro_regime_breakdown"]["RED"]["observations"], 2)
        self.assertIsNone(summary["roro_regime_breakdown"]["YELLOW"]["hit_rate"])
        self.assertIn("ablation_alpha_RRG", observations)

    @patch("sector_roro_shadow.fit_sector_roro_shadow")
    def test_as_of_cutoff_is_invariant_to_future_appends(self, fit) -> None:
        cutoff = pd.Timestamp("2026-01-09")
        tickers = ["XLK", "XLP", "XLF"]
        ranked = pd.DataFrame(
            {
                "ticker": tickers,
                "sector_status": ["SELECT", "WATCH", "AVOID"],
                "forecast_rank": [1, 2, 3],
                "predicted_excess_20d": [0.03, 0.01, -0.01],
                "alignment": ["ALIGNED", "DIVERGENCE", "ALIGNED"],
                "rrg_supportive": [True, False, False],
                "roro_fit": [0.8, 0.4, 0.2],
                "risk_regime": ["GREEN"] * 3,
            }
        )
        seen_maxima = []

        def fake_fit(features, prices, signals, roro, **kwargs):
            seen_maxima.append(
                (
                    features["date"].max(),
                    prices.index.max(),
                    signals["date"].max(),
                    roro["date"].max(),
                )
            )
            return ranked, {}

        fit.side_effect = fake_fit
        base_features = pd.DataFrame(
            [
                {
                    "date": cutoff,
                    "ticker": ticker,
                    "target_excess_20d": value,
                    "label_end": cutoff,
                }
                for ticker, value in zip(tickers, [0.02, -0.01, 0.01], strict=True)
            ]
        )
        future_features = pd.concat(
            [
                base_features,
                base_features.assign(
                    date=pd.Timestamp("2026-01-16"),
                    label_end=pd.Timestamp("2026-02-13"),
                    target_excess_20d=99.0,
                ),
            ],
            ignore_index=True,
        )
        prices = pd.DataFrame(
            {"SPY": [100.0, 999.0]}, index=[cutoff, pd.Timestamp("2026-01-16")]
        )
        signals = pd.DataFrame({"date": [cutoff, pd.Timestamp("2026-01-16")]})
        roro = pd.DataFrame({"date": [cutoff, pd.Timestamp("2026-01-16")]})

        first = validate_sector_roro_shadow(
            base_features,
            prices.iloc[:1],
            signals.iloc[:1],
            roro.iloc[:1],
            as_of=cutoff,
            bootstrap_samples=20,
        )
        second = validate_sector_roro_shadow(
            future_features,
            prices,
            signals,
            roro,
            as_of=cutoff,
            bootstrap_samples=20,
        )

        self.assertEqual(first[0], second[0])
        pd.testing.assert_frame_equal(first[1], second[1])
        self.assertTrue(all(all(value <= cutoff for value in maxima) for maxima in seen_maxima))


if __name__ == "__main__":
    unittest.main()
