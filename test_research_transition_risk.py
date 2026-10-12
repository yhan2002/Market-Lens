import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

from research_transition_risk import (
    Q20_FEATURES,
    _eligible_training_frame,
    _future_drawdown,
    _moving_block_indices,
    build_fixed_q20_predictions,
    candidate_gates,
    candidates,
    fixed_q20_candidate,
    fixed_q20_gates,
    independent_episode_count,
    load_frame,
    q20_probability_metrics,
    run_fixed_q20,
    score,
)


class TransitionRiskResearchTest(unittest.TestCase):
    def test_training_labels_are_embargoed_by_horizon(self) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=80, freq="B"),
                "x": np.arange(80, dtype=float),
                "drawdown_event": np.arange(80) % 2,
            }
        )
        train = _eligible_training_frame(
            frame,
            position=50,
            horizon=20,
            feature_columns=("x",),
            train_window=None,
        )
        self.assertEqual(train.index.max(), 30)
        self.assertEqual(len(train), 31)

    def test_metrics_count_independent_episodes(self) -> None:
        target = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 1])
        self.assertEqual(independent_episode_count(target), 3)
        self.assertEqual(independent_episode_count(target, separation=4), 1)
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=len(target), freq="B"),
                "drawdown_event": target,
                "risk": target + np.linspace(0, 0.1, len(target)),
            }
        )
        result = score(
            frame,
            "risk",
            bootstrap_samples=20,
            block_length=4,
            random_state=1,
        )
        self.assertEqual(result["independent_drawdown_episodes"], 3)
        self.assertEqual(result["roc_auc"], 1.0)

    def test_moving_blocks_and_candidate_cap(self) -> None:
        sampled = _moving_block_indices(
            100, block_length=40, rng=np.random.default_rng(1)
        )
        self.assertEqual(len(sampled), 100)
        self.assertLessEqual(len(candidates()), 6)

    def test_candidate_must_clear_every_gate(self) -> None:
        metrics = {
            "n": 1_100,
            "independent_drawdown_episodes": 25,
            "roc_auc": 0.64,
            "auc_ci_low": 0.51,
            "top_quintile_lift": 1.6,
        }
        blocks = {
            "2020-2021_selection": {"roc_auc": 0.61},
            "2022-2023_selection": {"roc_auc": 0.62},
        }
        comparison = {"delta_auc": 0.03, "ci_low": 0.01}

        self.assertTrue(all(candidate_gates(metrics, blocks, comparison).values()))
        comparison["ci_low"] = -0.01
        self.assertFalse(all(candidate_gates(metrics, blocks, comparison).values()))

    def test_fixed_q20_model_and_label_boundary_are_frozen(self) -> None:
        candidate = fixed_q20_candidate()
        model = candidate.factory()
        self.assertEqual(candidate.features, Q20_FEATURES)
        self.assertEqual(model.get_params()["C"], 1.0)
        self.assertIsNone(model.get_params()["class_weight"])
        self.assertEqual(model.get_params()["random_state"], 42)
        target = _future_drawdown(np.linspace(100, 90, 60), 20)
        self.assertTrue(np.isfinite(target[:-20]).all())
        self.assertTrue(np.isnan(target[-20:]).all())

        with self.assertRaisesRegex(ValueError, "frozen NQ core v1 HMM"):
            run_fixed_q20(Path("unused.db"), model_version="another-hmm")
        with self.assertRaisesRegex(ValueError, "validation protocol"):
            run_fixed_q20(
                Path("unused.db"),
                model_version="hmm3-sticky-nq-core-v1-k25-m252-w1260-r20-s42-n3",
                bootstrap_samples=100,
            )
        with self.assertRaisesRegex(ValueError, "benchmark must be NQ=F"):
            run_fixed_q20(
                Path("unused.db"),
                model_version="hmm3-sticky-nq-core-v1-k25-m252-w1260-r20-s42-n3",
                benchmark="SPY",
            )

    def test_fixed_q20_predicts_latest_unlabelled_rows_without_leakage(self) -> None:
        rng = np.random.default_rng(7)
        size = 900
        values = rng.normal(size=(size, len(Q20_FEATURES)))
        frame = pd.DataFrame(values, columns=Q20_FEATURES)
        frame.insert(0, "date", pd.date_range("2018-01-01", periods=size, freq="B"))
        event = (values[:, 0] + 0.6 * values[:, 4] + rng.normal(size=size) > 0).astype(float)
        event[-20:] = np.nan
        frame["drawdown_event"] = event

        result = build_fixed_q20_predictions(
            frame,
            min_train=120,
            train_window=300,
            refit_every=20,
        )

        latest = result.tail(20)
        self.assertTrue(latest["drawdown_event"].isna().all())
        self.assertTrue(latest["q20_raw"].notna().all())
        self.assertTrue(result["q20_raw"].dropna().between(0, 1).all())
        pd.testing.assert_series_equal(
            result["volatility_20_baseline"],
            result["volatility_20"],
            check_names=False,
        )

    def test_fixed_q20_probability_metrics_and_gates(self) -> None:
        target = np.tile([0, 0, 0, 1], 300)
        probability = np.where(target == 1, 0.75, 0.15)
        observed = pd.DataFrame(
            {"drawdown_event": target, "q20_raw": probability}
        )
        probability_metrics = q20_probability_metrics(observed)
        self.assertGreater(probability_metrics["brier_skill"], 0)
        self.assertAlmostEqual(probability_metrics["ece_10"], 0.175)

        metrics = {
            "n": 1_200,
            "independent_drawdown_episodes": 300,
            "auc_ci_low": 0.60,
            "top_quintile_lift": 2.0,
            "brier_skill": 0.20,
            "calibration_slope": 1.0,
            "ece_10": 0.05,
            "yearly_auc": {"2022": 0.60, "2023": 0.55, "2024": 0.70},
            "yearly_auc_total_years": 3,
        }
        gates = fixed_q20_gates(metrics, {"ci_low": 0.0})
        self.assertTrue(all(gates.values()))
        self.assertEqual(metrics["positive_year_ratio"], 1.0)

        metrics["yearly_auc_total_years"] = 6
        gates = fixed_q20_gates(metrics, {"ci_low": 0.0})
        self.assertFalse(gates["positive_year_ratio"])
        self.assertEqual(metrics["positive_year_ratio"], 0.5)

    def test_future_feature_changes_do_not_rewrite_past_q20(self) -> None:
        rng = np.random.default_rng(8)
        size = 240
        values = rng.normal(size=(size, len(Q20_FEATURES)))
        frame = pd.DataFrame(values, columns=Q20_FEATURES)
        frame.insert(0, "date", pd.date_range("2020-01-01", periods=size, freq="B"))
        frame["drawdown_event"] = (values[:, 0] + values[:, 4] > 0).astype(float)
        changed = frame.copy()
        changed.loc[changed.index > 180, "volatility_20"] = 99.0

        first = build_fixed_q20_predictions(
            frame, min_train=60, train_window=120, refit_every=10
        )
        second = build_fixed_q20_predictions(
            changed, min_train=60, train_window=120, refit_every=10
        )

        np.testing.assert_allclose(
            first.loc[:180, "q20_raw"], second.loc[:180, "q20_raw"], equal_nan=True
        )

    def test_loading_q20_inputs_keeps_sqlite_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "q20.db"
            dates = pd.date_range("2020-01-01", periods=240, freq="B")
            with closing(sqlite3.connect(database)) as connection:
                connection.execute(
                    "CREATE TABLE daily_prices(ticker TEXT, date TEXT, adj_close REAL)"
                )
                connection.execute(
                    "CREATE TABLE regime_model_probabilities(model_version TEXT, date TEXT, p_bull REAL, p_sideways REAL, p_bear REAL)"
                )
                connection.execute(
                    "CREATE TABLE regime_probabilities(date TEXT, p_bear REAL)"
                )
                connection.executemany(
                    "INSERT INTO daily_prices VALUES('NQ=F', ?, ?)",
                    [(date.date().isoformat(), 100 + index) for index, date in enumerate(dates)],
                )
                connection.executemany(
                    "INSERT INTO daily_prices VALUES('SPY', ?, ?)",
                    [(date.date().isoformat(), 1_000 + index) for index, date in enumerate(dates)],
                )
                connection.executemany(
                    "INSERT INTO regime_model_probabilities VALUES('nq-test', ?, 0.6, 0.3, 0.1)",
                    [(date.date().isoformat(),) for date in dates],
                )
                connection.executemany(
                    "INSERT INTO regime_probabilities VALUES(?, 0.1)",
                    [(date.date().isoformat(),) for date in dates],
                )
                connection.commit()
                before = connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                ).fetchall()

            frame = load_frame(database, model_version="nq-test")
            spy_frame = load_frame(
                database, benchmark="SPY", model_version="nq-test"
            )

            with closing(sqlite3.connect(database)) as connection:
                after = connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                ).fetchall()
            self.assertFalse(frame.empty)
            self.assertLess(frame["adj_close"].iloc[0], spy_frame["adj_close"].iloc[0])
            self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
