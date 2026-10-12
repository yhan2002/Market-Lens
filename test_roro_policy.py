from __future__ import annotations

import copy
import inspect
import unittest

import roro_policy
from roro_axis import LONG_ANCHORS as AXIS_LONG_ANCHORS
from roro_policy import (
    FAST_CONDITIONS,
    MODE_COMBINED,
    MODE_OVERLAY_ONLY,
    MODE_SCORE_ONLY,
    decide_roro_policy,
    long_anchor,
    replay_roro_policy,
)


def fast_snapshot(
    as_of: str,
    *,
    triggered: int = 0,
    unavailable: int = 0,
) -> dict[str, object]:
    conditions: list[dict[str, str]] = []
    for index, key in enumerate(FAST_CONDITIONS):
        status = (
            "TRIGGERED"
            if index < triggered
            else "UNAVAILABLE"
            if index < triggered + unavailable
            else "CLEAR"
        )
        conditions.append(
            {
                "key": key,
                "label": key.replace("_", " ").upper(),
                "status": status,
                "detail": status,
                "data_quality": "UNAVAILABLE" if status == "UNAVAILABLE" else "EXACT",
                "as_of": as_of,
            }
        )
    complete = unavailable == 0
    status = "TRIGGERED" if triggered >= 3 else "CLEAR" if complete else "INCOMPLETE"
    return {
        "as_of": as_of,
        "status": status,
        "count": triggered,
        "trigger_count": triggered,
        "required": 3,
        "total": 8,
        "complete": complete,
        "primary_driver": "FAST TRIGGER" if triggered else "FAST CLEAR",
        "advisory": "test",
        "conditions": conditions,
    }


def row(as_of: str, score: int, *, triggered: int = 0, unavailable: int = 0) -> dict[str, object]:
    return {
        "as_of": as_of,
        "risk_score": score,
        "fast_snapshot": fast_snapshot(
            as_of, triggered=triggered, unavailable=unavailable
        ),
    }


class RoroPolicyTest(unittest.TestCase):
    def test_anchor_matches_frozen_axis_anchor_and_uses_natural_percentages(self) -> None:
        self.assertEqual(tuple(roro_policy.LONG_ANCHORS), tuple(AXIS_LONG_ANCHORS))
        self.assertEqual([long_anchor(score) for score in (0, 33, 66, 100)], [100, 90, 60, 25])
        self.assertTrue(all(isinstance(long_anchor(score), int) for score in range(101)))

    def test_score_and_fast_cuts_are_immediate(self) -> None:
        first = decide_roro_policy("2026-01-07", 0, fast_snapshot("2026-01-07"))
        score_cut = decide_roro_policy(
            "2026-01-08", 66, fast_snapshot("2026-01-08"), first["policy_state"]
        )
        fast_cut = decide_roro_policy(
            "2026-01-09", 66, fast_snapshot("2026-01-09", triggered=3), score_cut["policy_state"]
        )
        self.assertEqual(score_cut["recommended_long"], 60)
        self.assertEqual(score_cut["phase"], "RISK_REDUCTION")
        self.assertEqual(fast_cut["base_long"], 60)
        self.assertEqual(fast_cut["fast_adjustment"], -20)
        self.assertEqual(fast_cut["recommended_long"], 40)

    def test_fast_cut_requires_valid_three_of_eight_even_when_others_are_missing(self) -> None:
        two = decide_roro_policy(
            "2026-01-07", 33, fast_snapshot("2026-01-07", triggered=2)
        )
        three = decide_roro_policy(
            "2026-01-07",
            33,
            fast_snapshot("2026-01-07", triggered=3, unavailable=5),
        )
        self.assertEqual((two["fast_adjustment"], two["raw_target_long"]), (0, 90))
        self.assertEqual((three["fast_adjustment"], three["raw_target_long"]), (-20, 70))

    def test_missing_fast_blocks_hold_increase_but_allow_rerisk_cut(self) -> None:
        first = decide_roro_policy("2026-01-07", 66, fast_snapshot("2026-01-07"))
        held = decide_roro_policy(
            "2026-01-08",
            33,
            fast_snapshot("2026-01-08", unavailable=4),
            first["policy_state"],
        )
        cut = decide_roro_policy(
            "2026-01-09",
            100,
            fast_snapshot("2026-01-09", unavailable=4),
            held["policy_state"],
        )
        self.assertEqual(held["fast_available_count"], 4)
        self.assertEqual(held["recommended_long"], 60)
        self.assertEqual(held["primary_driver"], "FAST DATA BELOW 5/8")
        self.assertEqual(cut["recommended_long"], 25)
        self.assertEqual(cut["phase"], "RISK_REDUCTION")

    def test_three_nonincreasing_days_ignore_changing_targets_and_move_directly(self) -> None:
        decisions = [
            row("2026-01-07", 66),
            row("2026-01-08", 65),
            row("2026-01-09", 60),
            row("2026-01-12", 50),
        ]
        results = replay_roro_policy(decisions)
        self.assertEqual([item["recommended_long"] for item in results], [60, 60, 60, 75])
        self.assertEqual([item["raw_target_long"] for item in results], [60, 61, 65, 75])
        self.assertEqual(results[-1]["phase"], "REENTRY_DIRECT")
        self.assertEqual(results[-1]["episode_status"], "INACTIVE")

    def test_worsening_score_resets_clear_streak(self) -> None:
        results = replay_roro_policy(
            [
                row("2026-01-07", 66),
                row("2026-01-08", 60),
                row("2026-01-09", 62),
                row("2026-01-12", 58),
                row("2026-01-13", 54),
                row("2026-01-14", 50),
            ]
        )
        self.assertEqual([item["clear_streak"] for item in results[1:5]], [1, 0, 1, 2])
        self.assertEqual(results[-1]["recommended_long"], results[-1]["raw_target_long"])
        self.assertEqual(results[-1]["phase"], "REENTRY_DIRECT")

    def test_previous_state_is_exact_versioned_and_never_mutated(self) -> None:
        first = decide_roro_policy("2026-01-07", 66, fast_snapshot("2026-01-07"))
        original = copy.deepcopy(first["policy_state"])
        decide_roro_policy(
            "2026-01-08", 60, fast_snapshot("2026-01-08"), first["policy_state"]
        )
        self.assertEqual(first["policy_state"], original)
        for key, value in (("state_version", 99), ("policy_version", "wrong"), ("mode", MODE_SCORE_ONLY)):
            wrong = dict(first["policy_state"])
            wrong[key] = value
            with self.assertRaises(ValueError):
                decide_roro_policy(
                    "2026-01-08", 60, fast_snapshot("2026-01-08"), wrong
                )
        extra = dict(first["policy_state"], unexpected=True)
        with self.assertRaises(ValueError):
            decide_roro_policy("2026-01-08", 60, fast_snapshot("2026-01-08"), extra)

    def test_future_append_does_not_change_prior_decisions(self) -> None:
        prefix = [
            row("2026-01-07", 66),
            row("2026-01-08", 60),
            row("2026-01-09", 58, unavailable=3),
        ]
        future = [row("2026-01-12", 90, triggered=3), row("2026-01-13", 20)]
        self.assertEqual(
            replay_roro_policy(prefix), replay_roro_policy(prefix + future)[: len(prefix)]
        )

    def test_same_reducer_supports_clean_ablations(self) -> None:
        decision = row("2026-01-07", 33, triggered=3)
        score_only = replay_roro_policy([decision], mode=MODE_SCORE_ONLY)[0]
        overlay_only = replay_roro_policy([decision], mode=MODE_OVERLAY_ONLY)[0]
        combined = replay_roro_policy([decision], mode=MODE_COMBINED)[0]
        self.assertEqual((score_only["base_long"], score_only["raw_target_long"]), (90, 90))
        self.assertEqual((overlay_only["base_long"], overlay_only["raw_target_long"]), (100, 80))
        self.assertEqual((combined["base_long"], combined["raw_target_long"]), (90, 70))

    def test_score_only_ignores_missing_fast_and_overlay_only_ignores_score(self) -> None:
        score_results = replay_roro_policy(
            [row("2026-01-07", 66, unavailable=8), row("2026-01-08", 100, unavailable=8)],
            mode=MODE_SCORE_ONLY,
        )
        self.assertEqual([item["recommended_long"] for item in score_results], [60, 25])
        low_score = replay_roro_policy([row("2026-01-07", 0)], mode=MODE_OVERLAY_ONLY)[0]
        high_score = replay_roro_policy([row("2026-01-07", 100)], mode=MODE_OVERLAY_ONLY)[0]
        self.assertEqual(low_score["recommended_long"], high_score["recommended_long"])

    def test_module_has_no_hidden_regime_model_dependency(self) -> None:
        source = inspect.getsource(roro_policy).lower()
        self.assertNotIn("hmm", source)
        self.assertNotIn("regime_router", source)

    def test_fast_contract_rejects_wrong_counts_and_future_observations(self) -> None:
        wrong = fast_snapshot("2026-01-07")
        wrong["trigger_count"] = 1
        with self.assertRaises(ValueError):
            decide_roro_policy("2026-01-07", 50, wrong)
        future = fast_snapshot("2026-01-07")
        future["conditions"][0]["as_of"] = "2026-01-08"
        with self.assertRaises(ValueError):
            decide_roro_policy("2026-01-07", 50, future)


if __name__ == "__main__":
    unittest.main()
