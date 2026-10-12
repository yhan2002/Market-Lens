import unittest
import warnings

import numpy as np
import pandas as pd

from sector_rotation_forecast import (
    BASE_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    build_forecast_frame,
    build_sector_feature_frame,
    fit_leading_forecast,
    walk_forward_validate,
)


TICKERS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP"]


def _training_frame(
    feature_columns: list[str], *, periods: int = 181
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2022-01-07", periods=periods, freq="W-FRI")
    rows = []
    for current_date in dates:
        for ticker_index, ticker in enumerate(TICKERS):
            row = {
                "date": current_date,
                "ticker": ticker,
                "shadow_score": ticker_index / len(TICKERS),
                "target_excess_20d": 0.01 * (ticker_index - 2.5)
                + rng.normal(0, 0.0001),
                "label_end": current_date + pd.offsets.BDay(20),
            }
            row.update(
                {
                    feature: rng.normal(0, 0.05) + ticker_index
                    for feature in feature_columns
                }
            )
            rows.append(row)
    return pd.DataFrame(rows), dates


class SectorRotationForecastTest(unittest.TestCase):
    def test_forecast_uses_only_labels_known_before_as_of(self) -> None:
        frame, dates = _training_frame(FEATURE_COLUMNS)

        forecast, metadata = fit_leading_forecast(
            frame,
            as_of=dates[-1],
            min_train_weeks=156,
            top_k=2,
        )

        self.assertEqual(len(forecast), len(TICKERS))
        self.assertTrue(forecast["sector"].notna().all())
        self.assertEqual(set(forecast["forecast_rank"]), set(range(1, len(TICKERS) + 1)))
        expected_signal = np.select(
            [
                (forecast["predicted_excess_20d"] > 0) & (forecast["forecast_rank"] <= 2),
                forecast["predicted_excess_20d"] > 0,
            ],
            ["MODEL_TOP3", "POSITIVE_WATCH"],
            default="NEGATIVE",
        )
        self.assertListEqual(forecast["signal"].tolist(), expected_signal.tolist())
        self.assertEqual((forecast["signal"] == "MODEL_TOP3").sum(), 2)
        self.assertIn("POSITIVE_WATCH", set(forecast["signal"]))
        self.assertIn("NEGATIVE", set(forecast["signal"]))
        self.assertLess(
            pd.Timestamp(metadata["latest_training_label_end"]),
            pd.Timestamp(metadata["as_of"]),
        )
        self.assertEqual(metadata["context_type"], "HMM")

        explicit_forecast, explicit_metadata = fit_leading_forecast(
            frame,
            as_of=dates[-1],
            min_train_weeks=156,
            top_k=2,
            feature_columns=FEATURE_COLUMNS,
        )
        pd.testing.assert_frame_equal(forecast, explicit_forecast)
        self.assertEqual(metadata, explicit_metadata)

        summary, observations = walk_forward_validate(
            frame,
            min_train_weeks=52,
            top_k=2,
            bootstrap_samples=100,
        )
        self.assertGreaterEqual(len(observations), 104)
        self.assertEqual(summary["bootstrap_block_weeks"], 4)
        self.assertEqual(len(summary["model_net_excess_95ci"]), 2)

    def test_base_feature_frame_is_causal_and_hmm_merge_is_unchanged(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=170)
        step = np.arange(len(dates), dtype=float)
        prices = pd.DataFrame(
            {
                "SPY": 100 * np.exp(0.0005 * step),
                "XLB": 90 * np.exp(0.0007 * step),
                "XLE": 80 * np.exp(0.0003 * step),
            },
            index=dates,
        )
        signals = pd.DataFrame(
            [
                {
                    "date": date,
                    "ticker": ticker,
                    "shadow_score": 0.4 + ticker_index * 0.1,
                    "rs_ratio": 100 + ticker_index + day * 0.01,
                    "rs_momentum": 100 - ticker_index + day * 0.02,
                }
                for day, date in enumerate(dates)
                for ticker_index, ticker in enumerate(("XLB", "XLE"))
            ]
        )
        regime = pd.DataFrame(
            {
                "date": dates[::5],
                "p_sideways": 0.2,
                "p_bear": 0.1,
            }
        )

        base = build_sector_feature_frame(prices, signals)
        expected = pd.merge_asof(
            base.sort_values("date"),
            regime[["date", "p_sideways", "p_bear"]].sort_values("date"),
            on="date",
            direction="backward",
        ).sort_values(["date", "ticker"])
        actual = build_forecast_frame(prices, signals, regime)
        pd.testing.assert_frame_equal(actual, expected)
        self.assertTrue(set(BASE_FEATURE_COLUMNS).issubset(base.columns))
        self.assertTrue({"p_sideways", "p_bear"}.isdisjoint(base.columns))

        cutoff = dates[-25]
        changed_prices = prices.copy()
        changed_prices.loc[changed_prices.index > cutoff, "XLB"] *= 1.5
        changed = build_sector_feature_frame(changed_prices, signals)
        left = base.loc[base["date"] <= cutoff, ["date", "ticker", *BASE_FEATURE_COLUMNS]]
        right = changed.loc[
            changed["date"] <= cutoff, ["date", "ticker", *BASE_FEATURE_COLUMNS]
        ]
        pd.testing.assert_frame_equal(left.reset_index(drop=True), right.reset_index(drop=True))
        original_target = base.loc[
            (base["date"] == cutoff) & (base["ticker"] == "XLB"), "target_excess_20d"
        ].iloc[0]
        changed_target = changed.loc[
            (changed["date"] == cutoff) & (changed["ticker"] == "XLB"),
            "target_excess_20d",
        ].iloc[0]
        self.assertNotEqual(original_target, changed_target)

    def test_base_feature_fit_without_hmm_columns(self) -> None:
        frame, dates = _training_frame(BASE_FEATURE_COLUMNS, periods=70)
        self.assertTrue({"p_sideways", "p_bear"}.isdisjoint(frame.columns))

        forecast, metadata = fit_leading_forecast(
            frame,
            as_of=dates[-1],
            min_train_weeks=52,
            top_k=2,
            feature_columns=BASE_FEATURE_COLUMNS,
        )

        self.assertEqual(len(forecast), len(TICKERS))
        self.assertEqual(metadata["context_type"], "NONE")
        self.assertIsNone(metadata["fixed_regime_context"])
        self.assertEqual(
            list(metadata["standardized_coefficients"]), BASE_FEATURE_COLUMNS
        )

        summary, observations = walk_forward_validate(
            frame,
            min_train_weeks=52,
            top_k=2,
            bootstrap_samples=100,
            feature_columns=BASE_FEATURE_COLUMNS,
        )
        self.assertFalse(observations.empty)
        self.assertEqual(summary["features"], BASE_FEATURE_COLUMNS)

    def test_walk_forward_rrg_ties_are_equal_weighted_without_warning(self) -> None:
        frame, _ = _training_frame(BASE_FEATURE_COLUMNS, periods=70)
        frame["shadow_score"] = 0.5
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            summary, observations = walk_forward_validate(
                frame,
                min_train_weeks=52,
                top_k=1,
                bootstrap_samples=100,
                feature_columns=BASE_FEATURE_COLUMNS,
            )
        self.assertTrue(observations["rrg_rank_ic"].isna().all())
        self.assertTrue(
            observations["rrg_selection_weights"].str.count(":").eq(len(TICKERS)).all()
        )
        self.assertEqual(summary["rrg_rank_ic_observations"], 0)
        self.assertEqual(summary["rrg_boundary_tie_observations"], len(observations))


if __name__ == "__main__":
    unittest.main()
