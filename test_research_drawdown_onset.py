import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

from research_drawdown_onset import (
    DIRECT_BENCHMARK,
    MAX_CANDIDATES,
    MAX_RESEARCH_TRIALS,
    REQUIRED_TICKERS,
    SHADOW_MODEL_VERSION,
    build_frame,
    candidate_specs,
    drawdown_onset_labels,
    evaluate_score,
    event_metrics,
    fixed_core_rank_score,
    get_shadow_hazard_status,
    purged_train_mask,
    severity_summary,
)


class DrawdownOnsetResearchTest(unittest.TestCase):
    def test_nq_drives_direct_risk_while_breadth_ratios_keep_spy(self) -> None:
        size = 280
        dates = pd.date_range("2024-01-01", periods=size, freq="B")
        prices = pd.DataFrame(index=dates, columns=REQUIRED_TICKERS, dtype=float)
        prices[DIRECT_BENCHMARK] = [100.0] * 252 + [94.0] * (size - 252)
        prices["SPY"] = np.linspace(100.0, 120.0, size)
        prices["^VIX"] = 15.0
        prices["HYG"] = 75.0
        prices["IEF"] = 95.0
        prices["RSP"] = 100.0
        prices["IWM"] = 200.0
        prices["UUP"] = 30.0

        frame = build_frame(prices).set_index("date")
        onset_date = dates[252]

        self.assertTrue(frame.loc[onset_date, "onset"])
        self.assertAlmostEqual(frame.loc[onset_date, "nq_return_5"], -0.06)
        expected_rsp_spy = (prices["RSP"] / prices["SPY"]).pct_change(20).loc[onset_date]
        self.assertAlmostEqual(frame.loc[onset_date, "rsp_spy_20"], expected_rsp_spy)

    def test_hysteresis_prevents_duplicate_onsets_until_recovery(self) -> None:
        prices = pd.Series([100.0] * 252 + [94.0, 93.0, 98.0, 94.0])
        result = drawdown_onset_labels(prices)

        self.assertEqual(result.index[result["onset"]].tolist(), [252, 255])

    def test_label_uses_future_only_and_includes_horizon_endpoint(self) -> None:
        prices = pd.Series([100.0] * 252 + [100.0] * 21 + [94.0])
        result = drawdown_onset_labels(prices, horizon=20)
        onset = int(result.index[result["onset"]][0])

        self.assertEqual(result.loc[onset - 20, "target"], 1.0)
        self.assertEqual(result.loc[onset - 21, "target"], 0.0)
        self.assertTrue(np.isnan(result.loc[onset, "target"]))

    def test_purge_excludes_labels_ending_after_prediction(self) -> None:
        frame = pd.DataFrame(
            {
                "_position": np.arange(80),
                "at_risk": True,
                "target": 0.0,
                "label_end_position": np.arange(80) + 20,
            }
        )
        selected = frame.loc[purged_train_mask(frame, 50), "_position"]

        self.assertEqual(int(selected.max()), 30)
        self.assertNotIn(31, selected.tolist())

    def test_event_metrics_match_each_episode_once(self) -> None:
        evaluation = pd.DataFrame(
            {
                "_position": np.arange(100),
                "date": pd.date_range("2020-01-01", periods=100),
            }
        )
        scores = pd.Series(0.0, index=evaluation.index)
        scores.loc[[5, 60, 90]] = 1.0

        result = event_metrics(evaluation, [20, 70], scores, 0.8)

        self.assertEqual(result["eligible_onsets"], 2)
        self.assertEqual(result["matched_onsets"], 2)
        self.assertEqual(result["matched_clusters"], 2)
        self.assertEqual(result["false_alert_clusters"], 1)
        self.assertAlmostEqual(result["cluster_precision"], 2 / 3)
        self.assertEqual(result["median_lead_days"], 12.5)

    def test_one_action_cluster_can_warn_two_distinct_onsets(self) -> None:
        evaluation = pd.DataFrame(
            {
                "_position": np.arange(60),
                "date": pd.date_range("2020-01-01", periods=60),
            }
        )
        scores = pd.Series(0.0, index=evaluation.index)
        scores.loc[[10, 29]] = 1.0

        result = event_metrics(evaluation, [20, 40], scores, 0.8)

        self.assertEqual(result["alert_clusters"], 1)
        self.assertEqual(result["matched_clusters"], 1)
        self.assertEqual(result["matched_onsets"], 2)
        self.assertEqual(result["onset_recall"], 1.0)

    def test_candidate_count_is_capped(self) -> None:
        candidates = candidate_specs()

        self.assertLessEqual(len(candidates), MAX_CANDIDATES)
        self.assertEqual(
            [candidate.name for candidate in candidates],
            ["logit_macro", "logit_cross", "shallow_hist_cross"],
        )
        self.assertLessEqual(len(candidates) + 1, MAX_RESEARCH_TRIALS)
        self.assertIn("uup_momentum_20", candidates[1].features)
        self.assertIn("iwm_spy_20", candidates[1].features)

    def test_fixed_rank_score_does_not_use_future_values(self) -> None:
        size = 220
        frame = pd.DataFrame(
            {
                "at_risk": True,
                "vix_z252": np.linspace(-1, 1, size),
                "vix_change_5": np.sin(np.arange(size) / 9),
                "hyg_ief_20": np.cos(np.arange(size) / 11) / 100,
            }
        )
        original = fixed_core_rank_score(frame, window=80, min_history=20)
        changed = frame.copy()
        changed.loc[180:, ["vix_z252", "vix_change_5", "hyg_ief_20"]] = 999.0
        revised = fixed_core_rank_score(changed, window=80, min_history=20)

        pd.testing.assert_series_equal(original.loc[:179], revised.loc[:179])

    def test_severity_summary_counts_only_complete_eligible_episodes(self) -> None:
        details = [
            {"eligible": True, "severity_window_complete": True, "min_drawdown_next_60d": -0.12, "hit": True},
            {"eligible": True, "severity_window_complete": True, "min_drawdown_next_60d": -0.06, "hit": False},
            {"eligible": False, "severity_window_complete": True, "min_drawdown_next_60d": -0.20, "hit": False},
        ]

        result = severity_summary(details)

        self.assertEqual(result["complete_eligible_episodes"], 2)
        self.assertEqual(result["severe_10pct_episodes"], 1)
        self.assertEqual(result["severe_10pct_recall"], 1.0)

    def test_shadow_status_uses_latest_row_on_or_before_date(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "router.db"
            with closing(sqlite3.connect(database)) as connection:
                connection.execute(
                    """
                    CREATE TABLE drawdown_hazard_scores(
                        date TEXT, model_version TEXT, score REAL, at_risk INTEGER,
                        vix_level_rank REAL, vix_acceleration_rank REAL,
                        credit_risk_rank REAL, risk_band TEXT
                    )
                    """
                )
                connection.execute(
                    "INSERT INTO drawdown_hazard_scores VALUES(?,?,?,?,?,?,?,?)",
                    ("2024-01-05", SHADOW_MODEL_VERSION, 0.9, 1, 0.9, 0.8, 1.0, "ALERT_SHADOW"),
                )
                connection.commit()

            result = get_shadow_hazard_status(database, "2024-01-07")

            self.assertEqual(result["date"], "2024-01-05")
            self.assertTrue(result["at_risk"])
            self.assertEqual(result["risk_band"], "ALERT_SHADOW")

    def test_alert_search_is_not_artificially_capped_at_point_80(self) -> None:
        size = 100
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=size),
                "_position": np.arange(size),
                "target": [1 if position in (5, 60) else 0 for position in range(size)],
            }
        )
        scores = pd.Series(0.5, index=frame.index)
        scores.loc[[5, 60]] = 0.72

        result = evaluate_score(
            frame,
            pd.Series(True, index=frame.index),
            scores,
            [20, 70],
            bootstrap_samples=5,
        )

        self.assertAlmostEqual(result["alert_cutoff"], 0.72)
        self.assertEqual(result["event_metrics"]["onset_recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
