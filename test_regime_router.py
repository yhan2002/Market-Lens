import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from breadth_metrics import EXACT_BREADTH_SOURCE, LIVE_BREADTH_SOURCE
from portfolio_risk import EVENT_HOLDINGS_SOURCE
from regime_router import (
    INDEPENDENT_A2B_HMM_FEATURE_COLUMNS,
    INDEPENDENT_A2B_HMM_MODEL_VERSION,
    INDEPENDENT_A2C_HMM_FEATURE_COLUMNS,
    INDEPENDENT_A2C_HMM_MODEL_VERSION,
    INDEPENDENT_A3A_HMM_FEATURE_COLUMNS,
    INDEPENDENT_A3A_HMM_MODEL_VERSION,
    INDEPENDENT_HMM_FEATURE_COLUMNS,
    INDEPENDENT_HMM_MODEL_VERSION,
    PDF_HMM_BLOCK_COLUMNS,
    PDF_HMM_FEATURE_COLUMNS,
    PDF_HMM_MODEL_VERSION,
    LIVE_HMM_MODEL_VERSION,
    _FlooredDiagGaussianHMM,
    _FlooredFullGaussianHMM,
    _aggregate_public_posterior,
    _composite_state_labels,
    _four_state_diagnostics,
    _initial_transition_matrix,
    _pdf8_state_labels,
    _prepare_pdf8_blocks,
    build_features,
    build_features_v2,
    build_independent_a2b_hmm_features,
    build_independent_a2c_hmm_features,
    build_independent_a3a_hmm_features,
    build_independent_hmm_features,
    build_features_v3,
    build_pdf_market_from_database,
    build_market_from_database,
    diagnose_pdf8_candidate,
    evaluate_actions,
    fit_candidate_from_database,
    fit_pdf_candidate,
    get_regime_status,
    load_vix_curve_market_from_database,
    walk_forward_pdf8_candidate,
    walk_forward_probabilities,
)


class RegimeRouterTest(unittest.TestCase):
    def test_shadow_diag_hmm_enforces_covariance_floor_after_fit(self) -> None:
        rng = np.random.default_rng(7)
        values = np.concatenate(
            [rng.normal(center, 0.001, (100, 2)) for center in (-2.0, 0.0, 2.0)]
        )
        model = _FlooredDiagGaussianHMM(
            n_components=3,
            covariance_type="diag",
            n_iter=20,
            random_state=42,
        )
        model.covariance_floor = 0.10
        model.fit(values)
        self.assertTrue(np.all(model._covars_ >= 0.10))

    def test_shadow_full_hmm_enforces_eigenvalue_floor_after_fit(self) -> None:
        rng = np.random.default_rng(8)
        values = np.concatenate(
            [rng.normal(center, 0.001, (100, 2)) for center in (-2.0, 0.0, 2.0)]
        )
        model = _FlooredFullGaussianHMM(
            n_components=3,
            covariance_type="full",
            n_iter=20,
            random_state=42,
        )
        model.covariance_floor = 0.10
        model.fit(values)
        self.assertTrue(np.all(np.linalg.eigvalsh(model._covars_) >= 0.10 - 1e-12))

    def test_walk_forward_supports_floored_full_covariance(self) -> None:
        rng = np.random.default_rng(9)
        features = pd.DataFrame(
            rng.normal(size=(100, 3)),
            index=pd.bdate_range("2020-01-01", periods=100),
            columns=["momentum_20", "volatility_20", "ma200_slope_20"],
        )
        probabilities = walk_forward_probabilities(
            features,
            min_train=60,
            train_window=100,
            refit_every=100,
            n_init=1,
            sticky_strength=25,
            composite_labels=True,
            covariance_type="full",
            covariance_floor=0.10,
        )
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        self.assertEqual(probabilities.attrs["covariance_type"], "full")
        self.assertEqual(probabilities.attrs["covariance_floor"], 0.10)

    def test_composite_labels_order_states_by_risk(self) -> None:
        columns = pd.Index(
            [
                "momentum_20",
                "momentum_60",
                "volatility_20",
                "drawdown_60",
                "ma200_distance",
                "ma200_slope_20",
            ]
        )
        model = type(
            "Model",
            (),
            {
                "n_components": 3,
                "means_": np.array(
                    [
                        [1.0, 1.0, -1.0, 0.0, 1.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [-1.0, -1.0, 1.0, -1.0, -1.0, -1.0],
                    ]
                ),
            },
        )()

        labels = _composite_state_labels(model, columns)

        self.assertEqual(labels[0], "bull")
        self.assertEqual(labels[1], "sideways")
        self.assertEqual(labels[2], "bear")

    def test_four_state_labels_and_public_aggregation_are_frozen(self) -> None:
        columns = pd.Index(INDEPENDENT_A2C_HMM_FEATURE_COLUMNS)
        model = type(
            "Model",
            (),
            {
                "n_components": 4,
                "means_": np.array(
                    [
                        [1.0, 1.0, -1.0],
                        [0.3, 0.3, -0.3],
                        [-0.3, -0.3, 0.3],
                        [-1.0, -1.0, 1.0],
                    ]
                ),
            },
        )()
        labels = _composite_state_labels(model, columns)
        self.assertEqual(labels, {0: "bull", 1: "sideways", 2: "bear", 3: "bear"})
        public = _aggregate_public_posterior(
            np.array([0.1, 0.2, 0.3, 0.4]), labels
        )
        self.assertEqual(public, {"bull": 0.1, "sideways": 0.2, "bear": 0.7})
        self.assertAlmostEqual(sum(public.values()), 1.0)

    def test_four_state_transition_and_collapse_guards(self) -> None:
        transition = _initial_transition_matrix(4)
        np.testing.assert_allclose(transition.sum(axis=1), 1.0)
        np.testing.assert_allclose(np.diag(transition), 0.96)
        columns = pd.Index(INDEPENDENT_A2C_HMM_FEATURE_COLUMNS)

        class Model:
            n_components = 4
            means_ = np.array(
                [
                    [0.9, 0.9, -0.9],
                    [0.3, 0.3, -0.3],
                    [-0.3, -0.3, 0.3],
                    [-0.9, -0.9, 0.9],
                ]
            )

            def predict(self, values: np.ndarray) -> np.ndarray:
                return np.arange(len(values)) % 4

        healthy = _four_state_diagnostics(Model(), np.ones((100, 3)), columns)
        self.assertTrue(healthy["admissible"])

        class DeadStateModel(Model):
            def predict(self, values: np.ndarray) -> np.ndarray:
                return np.arange(len(values)) % 3

        dead = _four_state_diagnostics(DeadStateModel(), np.ones((100, 3)), columns)
        self.assertFalse(dead["admissible"])

        class CollapsedCentroidModel(Model):
            means_ = Model.means_.copy()
            means_[2] = means_[1]

        collapsed = _four_state_diagnostics(
            CollapsedCentroidModel(), np.ones((100, 3)), columns
        )
        self.assertFalse(collapsed["admissible"])

    def test_independent_a2_features_are_exact_and_causal(self) -> None:
        dates = pd.bdate_range("2020-01-01", periods=360)
        close = pd.Series(100 * np.exp(np.linspace(0, 0.5, len(dates))))
        close += 0.4 * np.sin(np.arange(len(dates)) / 7)
        market = pd.DataFrame({"date": dates, "close": close})

        features = build_independent_hmm_features(market)

        self.assertEqual(list(features.columns), list(INDEPENDENT_HMM_FEATURE_COLUMNS))
        self.assertEqual(features.index.min(), dates[219])
        position = 300
        return_20 = close.iloc[position] / close.iloc[position - 20] - 1
        return_60_as_20 = (close.iloc[position] / close.iloc[position - 60]) ** (1 / 3) - 1
        ma200 = close.rolling(200).mean()
        vol = close.pct_change()
        expected = (
            return_20 - return_60_as_20,
            ma200.iloc[position] / ma200.iloc[position - 20] - 1,
            np.log(vol.rolling(20).std().iloc[position] / vol.rolling(120).std().iloc[position]),
        )
        np.testing.assert_allclose(features.loc[dates[position]].to_numpy(), expected)

        revised = market.copy()
        revised.loc[revised.index[-1], "close"] *= 1.5
        revised_features = build_independent_hmm_features(revised)
        pd.testing.assert_frame_equal(
            features.loc[features.index[:-1]],
            revised_features.loc[features.index[:-1]],
        )

    def test_independent_a2b_changes_only_absolute_volatility(self) -> None:
        dates = pd.bdate_range("2020-01-01", periods=360)
        close = pd.Series(100 * np.exp(np.linspace(0, 0.5, len(dates))))
        close += 0.4 * np.sin(np.arange(len(dates)) / 7)
        market = pd.DataFrame({"date": dates, "close": close})

        a2 = build_independent_hmm_features(market)
        a2b = build_independent_a2b_hmm_features(market)

        self.assertEqual(list(a2b.columns), list(INDEPENDENT_A2B_HMM_FEATURE_COLUMNS))
        pd.testing.assert_index_equal(a2b.index, a2.index)
        pd.testing.assert_frame_equal(a2b.iloc[:, :2], a2.iloc[:, :2])
        dated_close = pd.Series(close.to_numpy(), index=dates)
        expected_volatility = dated_close.pct_change().rolling(20).std() * np.sqrt(252)
        np.testing.assert_allclose(
            a2b["absolute_volatility_20"], expected_volatility.loc[a2b.index]
        )

        revised = market.copy()
        revised.loc[revised.index[-1], "close"] *= 1.5
        revised_a2b = build_independent_a2b_hmm_features(revised)
        pd.testing.assert_frame_equal(a2b.iloc[:-1], revised_a2b.iloc[:-1])

    def test_independent_a2c_changes_only_log_volatility_scale(self) -> None:
        dates = pd.bdate_range("2020-01-01", periods=360)
        close = pd.Series(100 * np.exp(np.linspace(0, 0.5, len(dates))))
        close += 0.4 * np.sin(np.arange(len(dates)) / 7)
        market = pd.DataFrame({"date": dates, "close": close})

        a2b = build_independent_a2b_hmm_features(market)
        a2c = build_independent_a2c_hmm_features(market)

        self.assertEqual(list(a2c.columns), list(INDEPENDENT_A2C_HMM_FEATURE_COLUMNS))
        pd.testing.assert_index_equal(a2c.index, a2b.index)
        pd.testing.assert_frame_equal(a2c.iloc[:, :2], a2b.iloc[:, :2])
        np.testing.assert_allclose(
            a2c["log_absolute_volatility_20"],
            np.log(a2b["absolute_volatility_20"]),
        )

        revised = market.copy()
        revised.loc[revised.index[-1], "close"] *= 1.5
        revised_a2c = build_independent_a2c_hmm_features(revised)
        pd.testing.assert_frame_equal(a2c.iloc[:-1], revised_a2c.iloc[:-1])

    def test_independent_a3a_adds_exact_causal_vix_curve(self) -> None:
        dates = pd.bdate_range("2020-01-01", periods=360)
        close = pd.Series(100 * np.exp(np.linspace(0, 0.5, len(dates))))
        close += 0.4 * np.sin(np.arange(len(dates)) / 7)
        vix = pd.Series(20 + 2 * np.sin(np.arange(len(dates)) / 11))
        vix3m = pd.Series(22 + 1.5 * np.cos(np.arange(len(dates)) / 13))
        market = pd.DataFrame(
            {"date": dates, "close": close, "vix": vix, "vix3m": vix3m}
        )

        a2c = build_independent_a2c_hmm_features(market)
        a3a = build_independent_a3a_hmm_features(market)

        self.assertEqual(list(a3a.columns), list(INDEPENDENT_A3A_HMM_FEATURE_COLUMNS))
        pd.testing.assert_index_equal(a3a.index, a2c.index)
        pd.testing.assert_frame_equal(a3a.iloc[:, :3], a2c)
        expected = np.log(
            pd.Series(vix.to_numpy(), index=dates)
            / pd.Series(vix3m.to_numpy(), index=dates)
        )
        np.testing.assert_allclose(a3a["vix_curve"], expected.loc[a3a.index])

        revised = market.copy()
        revised.loc[revised.index[-1], "vix"] *= 1.5
        revised_a3a = build_independent_a3a_hmm_features(revised)
        pd.testing.assert_frame_equal(a3a.iloc[:-1], revised_a3a.iloc[:-1])
        invalid = market.copy()
        invalid.loc[invalid.index[-1], "vix3m"] = 0.0
        invalid_a3a = build_independent_a3a_hmm_features(invalid)
        self.assertNotIn(dates[-1], invalid_a3a.index)

    def test_vix_curve_loader_uses_exact_positive_intersection_and_cutoff(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "router.db"
            with closing(sqlite3.connect(database)) as connection:
                connection.execute(
                    "CREATE TABLE daily_prices("
                    "ticker TEXT, date TEXT, close REAL, adj_close REAL)"
                )
                connection.executemany(
                    "INSERT INTO daily_prices VALUES (?, ?, ?, ?)",
                    [
                        ("NQ=F", "2026-01-02", 100.0, 100.0),
                        ("NQ=F", "2026-01-05", 101.0, 101.0),
                        ("NQ=F", "2026-01-06", 102.0, 102.0),
                        ("NQ=F", "2026-01-07", 103.0, 103.0),
                        ("^VIX", "2026-01-02", 20.0, 20.0),
                        ("^VIX", "2026-01-06", -1.0, -1.0),
                        ("^VIX", "2026-01-07", 21.0, 21.0),
                        ("^VIX3M", "2026-01-02", 22.0, 22.0),
                        ("^VIX3M", "2026-01-05", 23.0, 23.0),
                        ("^VIX3M", "2026-01-06", 24.0, 24.0),
                        ("^VIX3M", "2026-01-07", 25.0, 25.0),
                    ],
                )
                connection.commit()

            market = load_vix_curve_market_from_database(
                database, benchmark="NQ=F", as_of="2026-01-05"
            )

        self.assertEqual(market["date"].dt.strftime("%Y-%m-%d").tolist(), ["2026-01-02"])
        self.assertEqual(market.loc[0, ["close", "vix", "vix3m"]].tolist(), [100.0, 20.0, 22.0])

    def test_independent_a2_labels_order_states_by_risk(self) -> None:
        model = type(
            "Model",
            (),
            {
                "n_components": 3,
                "means_": np.array(
                    [
                        [1.0, 1.0, -1.0],
                        [0.0, 0.0, 0.0],
                        [-1.0, -1.0, 1.0],
                    ]
                ),
            },
        )()

        labels = _composite_state_labels(
            model, pd.Index(INDEPENDENT_HMM_FEATURE_COLUMNS)
        )

        self.assertEqual(labels, {0: "bull", 1: "sideways", 2: "bear"})
        a2b_labels = _composite_state_labels(
            model, pd.Index(INDEPENDENT_A2B_HMM_FEATURE_COLUMNS)
        )
        self.assertEqual(a2b_labels, labels)
        a2c_labels = _composite_state_labels(
            model, pd.Index(INDEPENDENT_A2C_HMM_FEATURE_COLUMNS)
        )
        self.assertEqual(a2c_labels, labels)
        a3a_model = type(
            "Model",
            (),
            {
                "n_components": 3,
                "means_": np.array(
                    [
                        [1.0, 1.0, -1.0, -1.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [-1.0, -1.0, 1.0, 1.0],
                    ]
                ),
            },
        )()
        a3a_labels = _composite_state_labels(
            a3a_model, pd.Index(INDEPENDENT_A3A_HMM_FEATURE_COLUMNS)
        )
        self.assertEqual(a3a_labels, labels)

    def test_pdf_composite_labels_order_states_by_risk(self) -> None:
        model = type(
            "Model",
            (),
            {
                "means_": np.array(
                    [
                        [-1.0, -1.0, -1.0, -1.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0, 1.0],
                    ]
                ),
            },
        )()

        labels, centroids = _pdf8_state_labels(model)

        self.assertEqual(labels, {0: "bull", 1: "sideways", 2: "bear"})
        self.assertEqual(centroids, [-1.0, 0.0, 1.0])

    def test_pdf_features_are_complete_and_causal(self) -> None:
        rows = 90
        dates = pd.bdate_range("2024-01-01", periods=rows)
        market = pd.DataFrame(
            {
                "date": dates,
                "nq_close": np.linspace(100, 130, rows),
                "vix": np.linspace(14, 24, rows),
                "vix3m": np.linspace(16, 22, rows),
                "move": np.linspace(90, 110, rows),
                "hy_oas": np.linspace(3.0, 4.0, rows),
                "usdjpy": np.linspace(140, 150, rows),
                "breadth_200dma": np.linspace(0.7, 0.4, rows),
                "portfolio_correlation": np.linspace(0.3, 0.8, rows),
            }
        )
        base = build_features_v3(market)
        changed = market.copy()
        changed.loc[changed.index[-1], "nq_close"] *= 2
        changed.loc[changed.index[-1], "vix"] *= 2
        revised = build_features_v3(changed)

        expected = [
            "nq_trend20",
            "nq_rv20",
            "vix_curve",
            "move_level",
            "hy_oas_level",
            "yen_carry5",
            "breadth_risk",
            "correlation20",
        ]
        self.assertEqual(list(PDF_HMM_FEATURE_COLUMNS), expected)
        self.assertEqual(len(PDF_HMM_FEATURE_COLUMNS), 8)
        self.assertEqual(list(base.columns), expected)
        self.assertFalse(base.isna().any().any())
        row = base.iloc[-1]
        self.assertAlmostEqual(
            row["nq_trend20"],
            -np.log(market["nq_close"].iloc[-1] / market["nq_close"].iloc[-21]),
        )
        log_returns = np.log(market["nq_close"] / market["nq_close"].shift(1))
        self.assertAlmostEqual(
            row["nq_rv20"], log_returns.iloc[-20:].std() * np.sqrt(252)
        )
        self.assertAlmostEqual(row["vix_curve"], np.log(24 / 22))
        self.assertAlmostEqual(row["move_level"], np.log(110))
        self.assertAlmostEqual(row["hy_oas_level"], np.log(400))
        self.assertAlmostEqual(
            row["yen_carry5"],
            -np.log(market["usdjpy"].iloc[-1] / market["usdjpy"].iloc[-6]),
        )
        self.assertAlmostEqual(row["breadth_risk"], 0.6)
        self.assertAlmostEqual(row["correlation20"], 0.8)
        common_past = base.index.intersection(revised.index)[:-1]
        pd.testing.assert_frame_equal(base.loc[common_past], revised.loc[common_past])

    def test_pdf_block_preprocessing_is_train_only(self) -> None:
        index = pd.bdate_range("2024-01-01", periods=81)
        raw = pd.DataFrame(
            {
                name: np.linspace(offset, offset + 1.0, len(index))
                for offset, name in enumerate(PDF_HMM_FEATURE_COLUMNS, start=1)
            },
            index=index,
        )
        _, base = _prepare_pdf8_blocks(raw.iloc[:80], raw)
        changed = raw.copy()
        changed.iloc[-1] *= 1_000.0
        train_blocks, revised = _prepare_pdf8_blocks(changed.iloc[:80], changed)

        self.assertEqual(list(base.columns), list(PDF_HMM_BLOCK_COLUMNS))
        self.assertTrue(np.allclose(train_blocks.mean(), 0.0, atol=1e-10))
        pd.testing.assert_frame_equal(base.iloc[:-1], revised.iloc[:-1])

    def test_pdf_block_preprocessing_rejects_constant_training_feature(self) -> None:
        index = pd.bdate_range("2024-01-01", periods=80)
        raw = pd.DataFrame(
            {
                name: np.linspace(offset, offset + 1.0, len(index))
                for offset, name in enumerate(PDF_HMM_FEATURE_COLUMNS, start=1)
            },
            index=index,
        )
        raw["move_level"] = 1.0

        with self.assertRaisesRegex(ValueError, "zero MAD"):
            _prepare_pdf8_blocks(raw, raw)

    def test_pdf_diagnostics_reject_occupancy_saturation_and_centroid_collapse(self) -> None:
        occupied = pd.DataFrame(
            np.tile([0.60, 0.25, 0.15], (300, 1)), columns=["p_bull", "p_sideways", "p_bear"]
        )
        occupancy = diagnose_pdf8_candidate(occupied, risk_centroids=[-1.0, 0.0, 1.0])
        self.assertIn("hard_state_occupancy_outside_5_80pct", occupancy["reasons"])

        saturated = pd.DataFrame(
            np.eye(3)[np.arange(300) % 3] * 0.995
            + (1 - np.eye(3)[np.arange(300) % 3]) * 0.0025,
            columns=["p_bull", "p_sideways", "p_bear"],
        )
        diagnostics = diagnose_pdf8_candidate(
            saturated, risk_centroids=[0.0, 0.1, 1.0]
        )
        self.assertFalse(diagnostics["valid"])
        self.assertIn("recent_posterior_saturation_above_80pct", diagnostics["reasons"])
        self.assertIn("adjacent_risk_centroid_gap_below_0p25z", diagnostics["reasons"])

    def test_pdf_frozen_fit_is_causal_research_only_and_refuses_overrides(self) -> None:
        rows = 800
        dates = pd.bdate_range("2020-01-01", periods=rows)
        market = pd.DataFrame(
            {
                "date": dates,
                "nq_close": 100
                * np.exp(
                    np.linspace(0.0, 0.4, rows)
                    + 0.01 * np.sin(np.arange(rows) / 11)
                ),
                "vix": 20 + 2 * np.sin(np.arange(rows) / 15),
                "vix3m": 21 + np.sin(np.arange(rows) / 18),
                "move": 100 + 5 * np.sin(np.arange(rows) / 20),
                "hy_oas": 3.5 + 0.2 * np.sin(np.arange(rows) / 25),
                "usdjpy": 145 + 2 * np.sin(np.arange(rows) / 30),
                "breadth_200dma": 0.55 + 0.1 * np.sin(np.arange(rows) / 22),
                "portfolio_correlation": 0.45 + 0.1 * np.sin(np.arange(rows) / 17),
            }
        )
        created: list[object] = []

        class FakeHMM:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs
                self.random_state = int(kwargs["random_state"])
                self.monitor_ = type(
                    "Monitor", (), {"history": [0.0, 1.0], "converged": True}
                )()
                self.means_ = np.array(
                    [[-1.0] * 4, [0.0] * 4, [1.0] * 4], dtype=float
                )
                created.append(self)

            def fit(self, values: pd.DataFrame) -> "FakeHMM":
                self.fit_shape = values.shape
                return self

            def score(self, values: pd.DataFrame) -> float:
                return float(-self.random_state)

            def predict_proba(self, values: pd.DataFrame) -> np.ndarray:
                probabilities = np.full((len(values), 3), 0.01)
                probabilities[np.arange(len(values)), np.arange(len(values)) % 3] = 0.98
                return probabilities

        with patch("regime_router.GaussianHMM", FakeHMM):
            base = fit_pdf_candidate(market)
            changed = market.copy()
            changed.loc[changed.index[-1], "vix"] *= 2.0
            revised = fit_pdf_candidate(changed)

        common_past = base.index.intersection(revised.index)[:-1]
        pd.testing.assert_frame_equal(base.loc[common_past], revised.loc[common_past])
        self.assertEqual(base.attrs["model_version"], PDF_HMM_MODEL_VERSION)
        self.assertTrue(base.attrs["research_only"])
        self.assertFalse(base.attrs["promotable"])
        self.assertEqual(
            base.attrs["validation_status"], "VALID_RESEARCH_ONLY"
        )
        self.assertEqual(base.attrs["invalid_reasons"], [])
        self.assertEqual(base.attrs["rejected_refits"], 0)
        self.assertEqual({model.random_state for model in created}, {42, 43, 44})
        for model in created:
            self.assertEqual(model.kwargs["covariance_type"], "diag")
            self.assertEqual(model.kwargs["n_iter"], 500)
            self.assertEqual(model.kwargs["tol"], 1e-4)
            self.assertEqual(model.kwargs["min_covar"], 0.10)
            prior = model.kwargs["transmat_prior"]
            self.assertTrue(np.allclose(np.diag(prior), 25.0))
            self.assertTrue(np.allclose(prior[~np.eye(3, dtype=bool)], 1.0))
            self.assertEqual(model.fit_shape[1], 4)

        with self.assertRaisesRegex(ValueError, "settings are frozen"):
            fit_pdf_candidate(market, min_train=755)
        with self.assertRaisesRegex(ValueError, "settings are frozen"):
            fit_pdf_candidate(market, covariance_type="tied")
        with self.assertRaisesRegex(ValueError, "settings are frozen"):
            fit_pdf_candidate(market, min_covar=0.01)
        with self.assertRaisesRegex(ValueError, "settings are frozen"):
            fit_pdf_candidate(market, model_version="mutable")
        incomplete = market.copy()
        incomplete.loc[incomplete.index[100], "breadth_200dma"] = np.nan
        with self.assertRaisesRegex(ValueError, "complete exact rows"):
            fit_pdf_candidate(incomplete)

    def test_pdf_refit_gate_keeps_last_known_good_model(self) -> None:
        rows = 780
        index = pd.bdate_range("2020-01-01", periods=rows)
        step = np.arange(rows, dtype=float)
        features = pd.DataFrame(
            {
                name: np.linspace(offset, offset + 1.0, rows)
                + 0.1 * np.sin(step / (7 + offset))
                for offset, name in enumerate(PDF_HMM_FEATURE_COLUMNS, start=1)
            },
            index=index,
        )
        created: list[object] = []

        class FakeHMM:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs
                self.random_state = int(kwargs["random_state"])
                self.refit_group = len(created) // 3
                self.monitor_ = type(
                    "Monitor", (), {"history": [0.0, 1.0], "converged": True}
                )()
                levels = [-1.0, 0.0, 1.0] if self.refit_group == 0 else [0.0, 0.1, 0.2]
                self.means_ = np.array([[level] * 4 for level in levels], dtype=float)
                created.append(self)

            def fit(self, values: pd.DataFrame) -> "FakeHMM":
                return self

            def score(self, values: pd.DataFrame) -> float:
                return float(-self.random_state)

            def predict_proba(self, values: pd.DataFrame) -> np.ndarray:
                probabilities = np.full((len(values), 3), 0.01)
                if self.refit_group == 0:
                    probabilities[
                        np.arange(len(values)), np.arange(len(values)) % 3
                    ] = 0.98
                else:
                    probabilities[:, 0] = 0.98
                return probabilities

        with patch("regime_router.GaussianHMM", FakeHMM):
            probabilities = walk_forward_pdf8_candidate(features)

        self.assertEqual(probabilities.attrs["refit_attempts"], 2)
        self.assertEqual(probabilities.attrs["refits"], 1)
        self.assertEqual(probabilities.attrs["rejected_refits"], 1)
        self.assertIn(
            "adjacent_risk_centroid_gap_below_0p25z",
            probabilities.attrs["rejected_refit_reasons"],
        )
        self.assertAlmostEqual(probabilities.loc[index[776], "p_bear"], 0.98)

    def test_pdf_database_market_respects_macro_availability(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=30)
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "pdf.db"
            with closing(sqlite3.connect(database)) as connection:
                connection.executescript(
                    """
                    CREATE TABLE daily_prices(
                        ticker TEXT, date TEXT, close REAL, adj_close REAL
                    );
                    CREATE TABLE macro_observations(
                        date TEXT, series TEXT, value REAL,
                        available_at TEXT, source TEXT
                    );
                    CREATE TABLE breadth_metrics(
                        date TEXT, breadth_200dma REAL, coverage REAL, source TEXT,
                        membership_as_of TEXT, available_at TEXT
                    );
                    CREATE TABLE portfolio_risk_metrics(
                        date TEXT, holdings_date TEXT, average_correlation REAL,
                        coverage REAL, source TEXT
                    );
                    """
                )
                tickers = {
                    "NQ=F": 100.0,
                    "^VIX": 18.0,
                    "^VIX3M": 20.0,
                    "^MOVE": 100.0,
                    "JPY=X": 150.0,
                }
                connection.executemany(
                    "INSERT INTO daily_prices VALUES (?, ?, ?, ?)",
                    [
                        (ticker, current.date().isoformat(), base + offset, base + offset)
                        for offset, current in enumerate(dates)
                        for ticker, base in tickers.items()
                    ],
                )
                connection.executemany(
                    "INSERT INTO macro_observations VALUES (?, 'HY_OAS', ?, ?, 'test')",
                    [
                        (dates[0].date().isoformat(), 3.0, dates[1].date().isoformat()),
                        (dates[15].date().isoformat(), 4.0, dates[16].date().isoformat()),
                        (dates[0].date().isoformat(), 9.0, dates[20].date().isoformat()),
                    ],
                )
                connection.executemany(
                    "INSERT INTO breadth_metrics VALUES (?, ?, 1.0, ?, ?, ?)",
                    [
                        (
                            current.date().isoformat(),
                            0.6,
                            (
                                LIVE_BREADTH_SOURCE
                                if current == dates[23]
                                else EXACT_BREADTH_SOURCE
                            ),
                            (
                                dates[25] if current == dates[24] else current
                            ).date().isoformat(),
                            current.date().isoformat(),
                        )
                        for current in dates
                    ],
                )
                connection.executemany(
                    "INSERT INTO portfolio_risk_metrics VALUES (?, ?, ?, 0.95, ?)",
                    [
                        (
                            current.date().isoformat(),
                            (
                                dates[26] if current == dates[25] else dates[0]
                            ).date().isoformat(),
                            0.4,
                            EVENT_HOLDINGS_SOURCE,
                        )
                        for current in dates
                    ],
                )
                connection.commit()

            market = build_pdf_market_from_database(
                database, dates[-1].date().isoformat()
            ).set_index("date")

            self.assertEqual(market.loc[dates[15], "hy_oas"], 3.0)
            self.assertEqual(market.loc[dates[16], "hy_oas"], 4.0)
            self.assertEqual(market.loc[dates[20], "hy_oas"], 4.0)
            self.assertNotIn(dates[24], market.index)
            self.assertNotIn(dates[25], market.index)
            self.assertNotIn(dates[23], market.index)
            self.assertNotIn(dates[0], market.index)

    def test_status_date_uses_previous_trading_day(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "router.db"
            with closing(sqlite3.connect(database)) as connection:
                connection.execute(
                    "CREATE TABLE regime_probabilities(date TEXT, p_bull REAL, p_sideways REAL, p_bear REAL)"
                )
                connection.execute(
                    "INSERT INTO regime_probabilities VALUES ('2024-01-05', 0.1, 0.2, 0.7)"
                )
                connection.commit()

            status = get_regime_status(database, "2024-01-06")
            self.assertEqual(status["date"], "2024-01-05")
            self.assertEqual(status["state"], "BEAR")

    def test_status_date_can_select_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "router.db"
            with closing(sqlite3.connect(database)) as connection:
                connection.execute(
                    """
                    CREATE TABLE regime_model_probabilities(
                        model_version TEXT, date TEXT, p_bull REAL,
                        p_sideways REAL, p_bear REAL
                    )
                    """
                )
                connection.execute(
                    "INSERT INTO regime_model_probabilities VALUES ('candidate', '2024-01-05', 0.7, 0.2, 0.1)"
                )
                connection.commit()

            status = get_regime_status(
                database, "2024-01-06", model_version="candidate"
            )
            self.assertEqual(status["model_version"], "candidate")
            self.assertEqual(status["state"], "BULL")

    def test_market_database_defaults_to_nq_and_accepts_explicit_spy(self) -> None:
        dates = pd.bdate_range("2023-01-02", periods=205)
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "router.db"
            with closing(sqlite3.connect(database)) as connection:
                connection.executescript(
                    """
                    CREATE TABLE universe(ticker TEXT, group_name TEXT, active INTEGER);
                    CREATE TABLE daily_prices(
                        ticker TEXT, date TEXT, close REAL, adj_close REAL
                    );
                    """
                )
                connection.executemany(
                    "INSERT INTO universe VALUES (?, ?, 1)",
                    [
                        ("UP", "CSQ_MP_CP"),
                        ("DOWN", "CSQ_MP_CP"),
                        ("UP", "CSQ_MP"),
                    ],
                )
                rows = []
                for offset, current_date in enumerate(dates):
                    day = current_date.strftime("%Y-%m-%d")
                    rows.extend(
                        [("NQ=F", day, 100 + offset, 100 + offset),
                         ("SPY", day, 500 + offset, 500 + offset),
                         ("UP", day, 100 + offset, 100 + offset),
                         ("DOWN", day, 300 - offset, 300 - offset)]
                    )
                connection.executemany("INSERT INTO daily_prices VALUES (?, ?, ?, ?)", rows)
                connection.commit()

            market = build_market_from_database(database)
            spy_market = build_market_from_database(database, benchmark="SPY")
            legacy_market = build_market_from_database(
                database, universe_group="CSQ_MP"
            )
            self.assertEqual(len(market), 6)
            self.assertTrue(np.allclose(market["close"], np.arange(299, 305)))
            self.assertTrue(np.allclose(spy_market["close"], np.arange(699, 705)))
            self.assertTrue(np.allclose(market["breadth"], 0.5))
            self.assertTrue(np.allclose(legacy_market["breadth"], 1.0))

    def test_probabilities_are_causal_and_sum_to_one(self) -> None:
        rng = np.random.default_rng(7)
        returns = np.r_[
            rng.normal(0.001, 0.005, 100),
            rng.normal(0.0, 0.003, 100),
            rng.normal(-0.001, 0.008, 100),
        ]
        market = pd.DataFrame(
            {
                "date": pd.bdate_range("2020-01-01", periods=len(returns)),
                "close": 100 * np.cumprod(1 + returns),
            }
        )
        features = build_features(market)
        base = walk_forward_probabilities(features, min_train=80, refit_every=20)

        changed = market.copy()
        changed.loc[changed.index[-1], "close"] *= 1.5
        revised = walk_forward_probabilities(build_features(changed), min_train=80, refit_every=20)

        self.assertTrue(np.allclose(base.sum(axis=1), 1.0))
        common_past = base.index.intersection(revised.index)[:-1]
        pd.testing.assert_frame_equal(base.loc[common_past], revised.loc[common_past])

    def test_sticky_v2_features_and_probabilities(self) -> None:
        rng = np.random.default_rng(11)
        returns = rng.normal(0.0003, 0.008, 420)
        market = pd.DataFrame(
            {
                "date": pd.bdate_range("2020-01-01", periods=len(returns)),
                "close": 100 * np.cumprod(1 + returns),
                "breadth": np.clip(
                    0.5 + pd.Series(returns).rolling(20, min_periods=1).mean() * 10,
                    0,
                    1,
                ),
            }
        )
        features = build_features_v2(market)
        probabilities = walk_forward_probabilities(
            features,
            min_train=80,
            refit_every=100,
            sticky_strength=25.0,
            composite_labels=True,
        )

        self.assertEqual(
            list(features.columns),
            [
                "momentum_20",
                "momentum_60",
                "volatility_20",
                "drawdown_60",
                "ma200_distance",
                "ma200_slope_20",
            ],
        )
        self.assertFalse(features.isna().any().any())
        self.assertTrue(np.allclose(probabilities.sum(axis=1), 1.0))

        selected = (
            "momentum_20",
            "volatility_20",
            "drawdown_60",
            "ma200_slope_20",
        )
        tied = walk_forward_probabilities(
            features.loc[:, list(selected)],
            min_train=80,
            refit_every=100,
            sticky_strength=25.0,
            composite_labels=True,
            covariance_type="tied",
        )
        self.assertTrue(np.allclose(tied.sum(axis=1), 1.0))
        self.assertEqual(tied.attrs["feature_columns"], list(selected))
        self.assertEqual(tied.attrs["covariance_type"], "tied")

    def test_candidate_storage_does_not_replace_v1(self) -> None:
        rng = np.random.default_rng(17)
        returns = rng.normal(0.0003, 0.008, 420)
        dates = pd.bdate_range("2020-01-01", periods=len(returns))
        closes = 100 * np.cumprod(1 + returns)
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "router.db"
            with closing(sqlite3.connect(database)) as connection:
                connection.executescript(
                    """
                    CREATE TABLE daily_prices(
                        ticker TEXT, date TEXT, close REAL, adj_close REAL
                    );
                    CREATE TABLE regime_probabilities(
                        date TEXT, p_bull REAL, p_sideways REAL, p_bear REAL
                    );
                    INSERT INTO regime_probabilities VALUES ('2024-01-05', 0.1, 0.2, 0.7);
                    """
                )
                connection.executemany(
                    "INSERT INTO daily_prices VALUES ('NQ=F', ?, ?, ?)",
                    [
                        (day.strftime("%Y-%m-%d"), float(close), float(close))
                        for day, close in zip(dates, closes)
                    ],
                )
                for ticker, multiplier in (("^VIX", 0.20), ("^VIX3M", 0.22)):
                    connection.executemany(
                        "INSERT INTO daily_prices VALUES (?, ?, ?, ?)",
                        [
                            (
                                ticker,
                                day.strftime("%Y-%m-%d"),
                                float(close * multiplier),
                                float(close * multiplier),
                            )
                            for day, close in zip(dates, closes)
                        ],
                    )
                connection.commit()

            probabilities = fit_candidate_from_database(
                database,
                min_train=80,
                train_window=250,
                refit_every=100,
                n_init=1,
            )
            selected = (
                "momentum_20",
                "volatility_20",
                "drawdown_60",
                "ma200_slope_20",
            )
            tied = fit_candidate_from_database(
                database,
                model_version="hmm3-sticky-nq-core-v2-tied4-test",
                feature_columns=selected,
                covariance_type="tied",
                min_train=80,
                train_window=250,
                refit_every=100,
                n_init=1,
            )
            floored = fit_candidate_from_database(
                database,
                model_version="hmm3-sticky-nq-core-v1-diag-mc10-shadow-a1-test",
                covariance_floor=0.10,
                min_train=80,
                train_window=250,
                refit_every=100,
                n_init=1,
            )
            independent_a2 = fit_candidate_from_database(
                database,
                model_version=INDEPENDENT_HMM_MODEL_VERSION + "-test",
                feature_set="independent-a2",
                covariance_floor=0.10,
                min_train=80,
                train_window=250,
                refit_every=100,
                n_init=1,
            )
            independent_a2b = fit_candidate_from_database(
                database,
                model_version=INDEPENDENT_A2B_HMM_MODEL_VERSION + "-test",
                feature_set="independent-a2b",
                covariance_floor=0.10,
                min_train=80,
                train_window=250,
                refit_every=100,
                n_init=1,
            )
            independent_a2c = fit_candidate_from_database(
                database,
                model_version=INDEPENDENT_A2C_HMM_MODEL_VERSION + "-test",
                feature_set="independent-a2c",
                covariance_floor=0.10,
                min_train=80,
                train_window=250,
                refit_every=100,
                n_init=1,
            )
            independent_a3a = fit_candidate_from_database(
                database,
                model_version=INDEPENDENT_A3A_HMM_MODEL_VERSION + "-test",
                feature_set="independent-a3a",
                covariance_floor=0.10,
                min_train=80,
                train_window=250,
                refit_every=100,
                n_init=1,
            )
            with closing(sqlite3.connect(database)) as connection:
                v1 = connection.execute("SELECT * FROM regime_probabilities").fetchall()
                candidate_count = connection.execute(
                    "SELECT COUNT(*) FROM regime_model_probabilities WHERE model_version=?",
                    (probabilities.attrs["model_version"],),
                ).fetchone()[0]
                tied_count = connection.execute(
                    "SELECT COUNT(*) FROM regime_model_probabilities WHERE model_version=?",
                    (tied.attrs["model_version"],),
                ).fetchone()[0]
                floored_count = connection.execute(
                    "SELECT COUNT(*) FROM regime_model_probabilities WHERE model_version=?",
                    (floored.attrs["model_version"],),
                ).fetchone()[0]
                independent_a2_count = connection.execute(
                    "SELECT COUNT(*) FROM regime_model_probabilities WHERE model_version=?",
                    (independent_a2.attrs["model_version"],),
                ).fetchone()[0]
                independent_a2b_count = connection.execute(
                    "SELECT COUNT(*) FROM regime_model_probabilities WHERE model_version=?",
                    (independent_a2b.attrs["model_version"],),
                ).fetchone()[0]
                independent_a2c_count = connection.execute(
                    "SELECT COUNT(*) FROM regime_model_probabilities WHERE model_version=?",
                    (independent_a2c.attrs["model_version"],),
                ).fetchone()[0]
                independent_a3a_count = connection.execute(
                    "SELECT COUNT(*) FROM regime_model_probabilities WHERE model_version=?",
                    (independent_a3a.attrs["model_version"],),
                ).fetchone()[0]

            self.assertEqual(v1, [("2024-01-05", 0.1, 0.2, 0.7)])
            self.assertEqual(candidate_count, len(probabilities))
            self.assertEqual(tied_count, len(tied))
            self.assertEqual(floored_count, len(floored))
            self.assertEqual(independent_a2_count, len(independent_a2))
            self.assertEqual(independent_a2b_count, len(independent_a2b))
            self.assertEqual(independent_a2c_count, len(independent_a2c))
            self.assertEqual(independent_a3a_count, len(independent_a3a))
            self.assertEqual(tied.attrs["feature_columns"], list(selected))
            self.assertEqual(tied.attrs["covariance_type"], "tied")
            self.assertEqual(floored.attrs["covariance_floor"], 0.10)
            self.assertEqual(
                independent_a2.attrs["feature_columns"],
                list(INDEPENDENT_HMM_FEATURE_COLUMNS),
            )
            self.assertEqual(independent_a2.attrs["feature_set"], "independent-a2")
            self.assertEqual(
                independent_a2b.attrs["feature_columns"],
                list(INDEPENDENT_A2B_HMM_FEATURE_COLUMNS),
            )
            self.assertEqual(independent_a2b.attrs["feature_set"], "independent-a2b")
            self.assertEqual(
                independent_a2c.attrs["feature_columns"],
                list(INDEPENDENT_A2C_HMM_FEATURE_COLUMNS),
            )
            self.assertEqual(independent_a2c.attrs["feature_set"], "independent-a2c")
            self.assertEqual(
                independent_a3a.attrs["feature_columns"],
                list(INDEPENDENT_A3A_HMM_FEATURE_COLUMNS),
            )
            self.assertEqual(independent_a3a.attrs["feature_set"], "independent-a3a")
            self.assertEqual(
                probabilities.attrs["model_version"],
                "hmm3-sticky-v2-nq=f-k25-m80-w250-r100-s42-n1",
            )
            with self.assertRaisesRegex(ValueError, "model_version is required"):
                fit_candidate_from_database(
                    database,
                    feature_columns=selected,
                    covariance_type="tied",
                    min_train=80,
                    train_window=250,
                    refit_every=100,
                    n_init=1,
                )
            with self.assertRaisesRegex(ValueError, "cannot overwrite the live"):
                fit_candidate_from_database(
                    database,
                    model_version=LIVE_HMM_MODEL_VERSION,
                    covariance_floor=0.10,
                    min_train=80,
                    train_window=250,
                    refit_every=100,
                    n_init=1,
                )
            with self.assertRaisesRegex(ValueError, "canonical A3-A"):
                fit_candidate_from_database(
                    database,
                    model_version=INDEPENDENT_A3A_HMM_MODEL_VERSION,
                    feature_set="independent-a3a",
                    benchmark="SPY",
                    covariance_floor=0.10,
                )
            with self.assertRaisesRegex(ValueError, "model_version is required"):
                fit_candidate_from_database(
                    database,
                    covariance_floor=0.10,
                    min_train=80,
                    train_window=250,
                    refit_every=100,
                    n_init=1,
                )
            with self.assertRaisesRegex(ValueError, "model_version is required"):
                fit_candidate_from_database(
                    database,
                    feature_set="independent-a2",
                    min_train=80,
                    train_window=250,
                    refit_every=100,
                    n_init=1,
                )
            with self.assertRaisesRegex(ValueError, "cannot overwrite the live"):
                fit_candidate_from_database(
                    database,
                    model_version=LIVE_HMM_MODEL_VERSION,
                    feature_set="independent-a2",
                    min_train=80,
                    train_window=250,
                    refit_every=100,
                    n_init=1,
                )

    def test_action_accounting(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=8)
        portfolio = pd.DataFrame(
            {"date": dates, "nav": [100, 90, 80, 85, 95, 105, 110, 115]}
        )
        probabilities = pd.DataFrame({"date": dates, "p_bear": 0.8})
        events = pd.DataFrame(
            {
                "event": ["sample"],
                "sell_date": [dates[0]],
                "reentry_date": [dates[4]],
                "end_date": [dates[-1]],
            }
        )
        result = evaluate_actions(
            portfolio, probabilities, events, transaction_cost_bps=0
        ).set_index("action")

        self.assertAlmostEqual(result.loc["HOLD", "ending_wealth"], 1.15)
        self.assertGreater(
            result.loc["ACTUAL_FULL_SELL", "ending_wealth"],
            result.loc["HOLD", "ending_wealth"],
        )
        self.assertEqual(set(result.index), {"ACTUAL_FULL_SELL", "HOLD", "DE_RISK_50", "HMM_POLICY"})


if __name__ == "__main__":
    unittest.main()
