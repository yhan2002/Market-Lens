from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from roro_q20 import (
    FEATURE_COLUMNS,
    INSUFFICIENT_EVIDENCE,
    READY_FOR_WALK_FORWARD_FIT,
    InsufficientEvidenceError,
    assess_q20_readiness,
    build_q20_readiness_history,
    build_q20_research_frame,
    exact_feature_matrix,
    require_fit_ready,
    validate_q20_readiness_metadata,
)


def _daily(periods: int = 40) -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-02", periods=periods)
    frame = pd.DataFrame(
        {
            "date": dates,
            "nq_close": np.linspace(100, 120, periods),
            "macro_risk_score": 40.0,
            "market_trend_score": 35.0,
            "portfolio_internal_score": 45.0,
            "trajectory_1d": 1.0,
            "trajectory_5d": -2.0,
            "trajectory_20d": 3.0,
            "data_completeness": 80.0,
            "axis_disagreement": 5.0,
        }
    )
    return frame


class RoroQ20Test(unittest.TestCase):
    def test_frozen_vector_is_compact_and_exact(self) -> None:
        frame = _daily(2)
        frame["market_risk_score"] = 99
        matrix, complete = exact_feature_matrix(frame)

        self.assertEqual(tuple(matrix.columns), FEATURE_COLUMNS)
        self.assertNotIn("market_risk_score", matrix)
        self.assertTrue(complete.all())
        frame.loc[0, "macro_risk_score"] = np.nan
        frame.loc[1, "trajectory_20d"] = 101
        _, complete = exact_feature_matrix(frame)
        self.assertEqual(complete.tolist(), [False, False])

    def test_label_is_t_plus_one_through_t_plus_twenty_and_embargoed(self) -> None:
        daily = _daily(25)
        daily["nq_close"] = 100.0
        daily.loc[0, "nq_close"] = 90.0
        daily.loc[10, "nq_close"] = 94.0
        result = build_q20_research_frame(daily, as_of=daily.loc[20, "date"])

        self.assertEqual(len(result), 21)
        self.assertAlmostEqual(
            result.loc[0, "future_path_drawdown"], 94.0 / 90.0 - 1.0
        )
        self.assertEqual(result.loc[0, "q20_event"], 0)
        self.assertTrue(result.loc[0, "label_complete"])
        self.assertFalse(result.loc[1, "label_complete"])
        self.assertTrue(pd.isna(result.loc[1, "q20_event"]))

    def test_as_of_result_is_invariant_to_later_rows(self) -> None:
        daily = _daily(50)
        cutoff = daily.loc[29, "date"]
        before = build_q20_research_frame(daily, as_of=cutoff)
        changed = daily.copy()
        changed.loc[changed["date"] > cutoff, "nq_close"] = 1.0
        changed.loc[changed["date"] > cutoff, list(FEATURE_COLUMNS)] = 99.0
        after = build_q20_research_frame(changed, as_of=cutoff)

        assert_frame_equal(before, after)

    def test_daily_readiness_history_is_invariant_to_future_append(self) -> None:
        daily = _daily(50)
        targets = daily.loc[:29, "date"]
        before = build_q20_readiness_history(daily, decision_dates=targets)
        extra = _daily(10)
        extra["date"] = pd.bdate_range(daily["date"].iloc[-1] + pd.offsets.BDay(), periods=10)
        extra["nq_close"] = 1.0
        extra.loc[:, list(FEATURE_COLUMNS)] = 99.0
        appended = pd.concat([daily, extra], ignore_index=True)
        after = build_q20_readiness_history(appended, decision_dates=targets)

        assert_frame_equal(before, after)

    def test_daily_readiness_counts_are_causal_and_monotone(self) -> None:
        daily = _daily(45)
        targets = daily.loc[[20, 21, 44], "date"]
        history = build_q20_readiness_history(daily, decision_dates=targets)
        metadata = history["q20"].tolist()

        self.assertEqual([item["feature_rows"] for item in metadata], [21, 22, 45])
        self.assertEqual(
            [item["complete_feature_labels"] for item in metadata], [1, 2, 25]
        )
        self.assertEqual([item["independent_events"] for item in metadata], [0, 0, 0])
        for item in metadata:
            self.assertEqual(validate_q20_readiness_metadata(item), item)
            self.assertEqual(item["status"], INSUFFICIENT_EVIDENCE)
            self.assertIsNone(item["probability"])
            self.assertFalse(item["fit_allowed"])

    def test_daily_readiness_counts_known_independent_event_clusters(self) -> None:
        daily = _daily(66)
        daily["nq_close"] = 100.0
        daily.loc[[20, 62], "nq_close"] = 94.0
        history = build_q20_readiness_history(
            daily, decision_dates=daily.loc[[40, 64], "date"]
        )
        metadata = history["q20"].tolist()

        self.assertEqual(
            [item["complete_feature_labels"] for item in metadata], [21, 45]
        )
        self.assertEqual([item["independent_events"] for item in metadata], [1, 2])

    def test_missing_feature_is_not_imputed_or_counted(self) -> None:
        daily = _daily(45)
        daily.loc[0, "portfolio_internal_score"] = np.nan
        result = build_q20_research_frame(daily)
        readiness = assess_q20_readiness(result)

        self.assertFalse(result.loc[0, "feature_complete"])
        self.assertTrue(result.loc[0, "label_complete"])
        self.assertFalse(result.loc[0, "training_eligible"])
        self.assertEqual(readiness["label_rows"], 25)
        self.assertEqual(readiness["complete_feature_labels"], 24)
        self.assertIsNone(readiness["probability"])

    def test_gate_blocks_then_allows_only_independent_evidence(self) -> None:
        daily = _daily(840)
        daily["nq_close"] = 100.0
        daily.loc[list(range(20, 840, 42)), "nq_close"] = 94.0
        blocked_frame = build_q20_research_frame(
            daily, as_of=daily.loc[251, "date"]
        )
        blocked = assess_q20_readiness(blocked_frame)
        self.assertEqual(blocked["status"], INSUFFICIENT_EVIDENCE)
        self.assertIsNone(blocked["probability"])
        with self.assertRaises(InsufficientEvidenceError):
            require_fit_ready(blocked_frame)

        frame = build_q20_research_frame(daily)
        ready = assess_q20_readiness(frame)
        self.assertEqual(ready["status"], READY_FOR_WALK_FORWARD_FIT)
        self.assertEqual(ready["independent_events"], 20)
        self.assertIsNone(ready["probability"])
        self.assertEqual(require_fit_ready(frame), ready)

    def test_frozen_builder_does_not_accept_label_overrides(self) -> None:
        daily = _daily(25)
        with self.assertRaises(TypeError):
            build_q20_research_frame(daily, horizon=1)
        with self.assertRaises(TypeError):
            build_q20_research_frame(daily, drawdown_threshold=-0.01)

    def test_readiness_rejects_forged_training_eligibility(self) -> None:
        frame = build_q20_research_frame(_daily(40))
        frame.loc[0, "training_eligible"] = not frame.loc[0, "training_eligible"]
        with self.assertRaisesRegex(ValueError, "training_eligible"):
            assess_q20_readiness(frame)

    def test_readiness_rejects_forged_label_path_and_available_date(self) -> None:
        frame = build_q20_research_frame(_daily(45))
        frame.loc[0, "future_path_drawdown"] = -0.99
        frame.loc[0, "q20_event"] = 1
        with self.assertRaisesRegex(ValueError, "frozen NQ path"):
            assess_q20_readiness(frame)

        frame = build_q20_research_frame(_daily(45))
        frame.loc[0, "label_available_date"] = frame.loc[0, "date"]
        with self.assertRaisesRegex(ValueError, "frozen NQ calendar"):
            assess_q20_readiness(frame)

    def test_readiness_rejects_rows_after_as_of(self) -> None:
        daily = _daily(45)
        frame = build_q20_research_frame(daily, as_of=daily.loc[29, "date"])
        future = build_q20_research_frame(daily).iloc[[30]].copy()
        forged = pd.concat([frame, future], ignore_index=True)
        forged.attrs = frame.attrs.copy()

        with self.assertRaisesRegex(ValueError, "after its as_of"):
            assess_q20_readiness(forged)

    def test_same_calendar_day_cannot_count_as_two_nq_sessions(self) -> None:
        daily = _daily(25)
        daily.loc[1, "date"] = daily.loc[0, "date"] + pd.Timedelta(hours=8)
        with self.assertRaisesRegex(ValueError, "dates must be unique"):
            build_q20_research_frame(daily)


if __name__ == "__main__":
    unittest.main()
