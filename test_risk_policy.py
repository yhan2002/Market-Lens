import unittest

from risk_policy import (
    POLICY_VERSION,
    RULE_SCORE_POLICY_VERSION,
    RULE_STRESS_SCORE_POLICY_VERSION,
    RULE_STRESS_SCORE_V3_LIVE_POLICY_VERSION,
    RULE_STRESS_SCORE_V3_POLICY_VERSION,
    evaluate_reentry,
    evaluate_risk_policy,
    exposure_from_probabilities,
)


HMM_VERSION = "test-hmm-v1"


def regime(
    bull: int,
    sideways: int,
    bear: int,
    as_of: str = "2026-01-02",
    *,
    model_version: str = HMM_VERSION,
) -> dict[str, object]:
    return {
        "date": as_of,
        "model_version": model_version,
        "state": max(
            {"BULL": bull, "SIDEWAYS": sideways, "BEAR": bear},
            key={"BULL": bull, "SIDEWAYS": sideways, "BEAR": bear}.get,
        ),
        "p_bull": bull / 100,
        "p_sideways": sideways / 100,
        "p_bear": bear / 100,
    }


def fast_overlay(triggered: int = 0, unavailable: int = 0) -> dict[str, object]:
    keys = (
        "vix_1d_spike",
        "vix_term_inversion",
        "nq_1d_drop",
        "move_1d_spike",
        "hy_oas_widening",
        "jpy_3d_drop",
        "breadth_1d_collapse",
        "portfolio_correlation",
    )
    conditions = []
    for index, key in enumerate(keys):
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
                "label": f"Fast {index + 1}",
                "status": status,
                "data_quality": "UNAVAILABLE" if status == "UNAVAILABLE" else "EXACT",
                "detail": "test",
            }
        )
    return {"conditions": conditions}


def reentry_conditions(*statuses: str) -> list[dict[str, str]]:
    keys = (
        "vix_term_structure",
        "hy_spread_stabilized",
        "nq_above_5dma",
        "breadth_divergence",
    )
    return [
        {
            "key": key,
            "label": f"Re-entry {index}",
            "status": status,
            "detail": "test",
            "data_quality": "UNAVAILABLE" if status == "UNAVAILABLE" else "EXACT",
        }
        for index, (key, status) in enumerate(zip(keys, statuses), start=1)
    ]


def eligible_reentry(
    current: float = 0.05,
    previous: float = 0.10,
    peak: float = 0.82,
) -> dict[str, object]:
    return evaluate_reentry(
        current,
        previous,
        reentry_conditions("CONFIRMED", "CONFIRMED", "CONFIRMED", "NOT_CONFIRMED"),
        recent_peak_p_bear=peak,
    )


def decide(
    as_of: str,
    values: tuple[int, int, int],
    *,
    fast: dict[str, object] | None = None,
    reentry: dict[str, object] | None = None,
    previous: dict[str, object] | None = None,
    model_version: str = HMM_VERSION,
) -> dict[str, object]:
    return evaluate_risk_policy(
        regime(*values, as_of, model_version=model_version),
        fast_overlay() if fast is None else fast,
        reentry,
        as_of=as_of,
        previous_state=previous,
    )


class ProbabilityExposureTests(unittest.TestCase):
    def test_approved_examples(self) -> None:
        cases = {
            (85, 10, 5): 0.91,
            (49, 51, 0): 0.83,
            (20, 80, 0): 0.74,
            (18, 47, 35): 0.65,
            (39, 20, 41): 0.70,
            (0, 0, 100): 0.40,
        }
        for probabilities, expected in cases.items():
            with self.subTest(probabilities=probabilities):
                self.assertEqual(
                    exposure_from_probabilities(*(value / 100 for value in probabilities)),
                    expected,
                )

    def test_probability_validation(self) -> None:
        for values in ((0.5, 0.4, 0.2), (-0.1, 0.5, 0.6), (True, 0.0, 0.0)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                exposure_from_probabilities(*values)

    def test_posterior_mix_changes_exposure_within_same_argmax_state(self) -> None:
        self.assertEqual(exposure_from_probabilities(0.9997, 0.0003, 0.0), 0.95)
        self.assertEqual(exposure_from_probabilities(0.51, 0.49, 0.0), 0.83)


class SequentialPolicyTests(unittest.TestCase):
    def test_fast_three_of_eight_cuts_once(self) -> None:
        first = decide("2026-01-02", (100, 0, 0), fast=fast_overlay(3))
        repeated = decide(
            "2026-01-05",
            (100, 0, 0),
            fast=fast_overlay(3),
            previous=first["policy_state"],
        )
        self.assertEqual(first["base_exposure"], 0.95)
        self.assertEqual(first["raw_recommended_exposure"], 0.75)
        self.assertEqual(first["recommended_exposure"], 0.75)
        self.assertEqual(repeated["recommended_exposure"], 0.75)

    def test_normal_recovery_cannot_stick_forever_without_risk_off_peak(self) -> None:
        bull = decide("2026-01-02", (100, 0, 0))
        cut = decide(
            "2026-01-05",
            (0, 0, 100),
            previous=bull["policy_state"],
        )
        self.assertEqual(cut["recommended_exposure"], 0.40)

        state = cut["policy_state"]
        exposures = []
        for as_of in (
            "2026-01-06",
            "2026-01-07",
            "2026-01-08",
            "2026-01-09",
            "2026-01-12",
        ):
            result = decide(as_of, (100, 0, 0), previous=state)
            state = result["policy_state"]
            exposures.append(result["recommended_exposure"])

        self.assertEqual(exposures, [0.40, 0.40, 0.60, 0.80, 0.95])
        self.assertEqual(state["episode_peak_p_bear"], None)

    def test_five_available_fast_channels_are_enough_for_normal_recovery(self) -> None:
        cut = decide("2026-01-02", (0, 0, 100))
        state = cut["policy_state"]
        incomplete = fast_overlay(unavailable=3)
        for as_of in ("2026-01-05", "2026-01-06", "2026-01-07"):
            result = decide(as_of, (100, 0, 0), fast=incomplete, previous=state)
            state = result["policy_state"]
        self.assertFalse(result["fast_overlay"]["complete"])
        self.assertEqual(result["fast_available_count"], 5)
        self.assertEqual(result["recommended_exposure"], 0.60)

    def test_fewer_than_five_available_channels_holds_recovery(self) -> None:
        cut = decide("2026-01-02", (0, 0, 100))
        state = cut["policy_state"]
        for as_of in ("2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"):
            result = decide(
                as_of,
                (100, 0, 0),
                fast=fast_overlay(unavailable=4),
                previous=state,
            )
            state = result["policy_state"]
        self.assertEqual(result["fast_available_count"], 4)
        self.assertEqual(result["recommended_exposure"], 0.40)
        self.assertEqual(result["primary_driver_code"], "RECOVERY_HOLD")

    def test_bottom_fishing_gate_accelerates_genuine_risk_off_episode(self) -> None:
        cut = decide("2026-01-02", (5, 13, 82))
        recovered = decide(
            "2026-01-05",
            (25, 40, 35),
            reentry=eligible_reentry(current=0.35, previous=0.50, peak=0.10),
            previous=cut["policy_state"],
        )
        self.assertEqual(recovered["reentry"]["recent_peak_p_bear"], 0.82)
        self.assertEqual(recovered["reentry"]["status"], "ELIGIBLE_ONE_STEP")
        self.assertEqual(recovered["recommended_exposure"], 0.60)
        self.assertEqual(recovered["primary_driver_code"], "BOTTOM_FISHING")

    def test_same_day_is_idempotent_but_new_risk_cuts_immediately(self) -> None:
        cut = decide("2026-01-02", (0, 0, 100))
        state = cut["policy_state"]
        for as_of in ("2026-01-05", "2026-01-06", "2026-01-07"):
            stepped = decide(as_of, (100, 0, 0), previous=state)
            state = stepped["policy_state"]
        self.assertEqual(stepped["recommended_exposure"], 0.60)

        rerun = decide("2026-01-07", (100, 0, 0), previous=state)
        same_day_cut = decide(
            "2026-01-07",
            (0, 0, 100),
            previous=state,
        )
        self.assertEqual(rerun["recommended_exposure"], 0.60)
        self.assertEqual(same_day_cut["recommended_exposure"], 0.40)

    def test_legacy_or_wrong_hmm_state_cannot_anchor_exposure(self) -> None:
        current = decide("2026-01-05", (100, 0, 0))
        legacy = {**current["policy_state"], "policy_version": "risk-policy-v4"}
        wrong_hmm = {**current["policy_state"], "regime_model_version": "spy-hmm"}
        for previous in (legacy, wrong_hmm):
            result = decide("2026-01-06", (100, 0, 0), previous=previous)
            self.assertEqual(result["decision_status"], "NO_DECISION")
            self.assertIn("PREVIOUS_STATE", result["missing_channels"])

    def test_active_fast_overlay_never_raises_exposure(self) -> None:
        cut = decide("2026-01-02", (0, 0, 100))
        result = decide(
            "2026-01-05",
            (100, 0, 0),
            fast=fast_overlay(3),
            reentry=eligible_reentry(current=0.0),
            previous=cut["policy_state"],
        )
        self.assertEqual(result["raw_recommended_exposure"], 0.75)
        self.assertEqual(result["recommended_exposure"], 0.40)


class ReentryTests(unittest.TestCase):
    def test_peak_turn_and_three_confirm_are_required(self) -> None:
        eligible = eligible_reentry()
        rising = evaluate_reentry(
            0.74,
            0.61,
            reentry_conditions("CONFIRMED", "CONFIRMED", "CONFIRMED", "CONFIRMED"),
            recent_peak_p_bear=0.82,
        )
        shallow = evaluate_reentry(
            0.09,
            0.10,
            reentry_conditions("CONFIRMED", "CONFIRMED", "CONFIRMED", "CONFIRMED"),
            recent_peak_p_bear=0.10,
        )
        self.assertEqual(eligible["status"], "ELIGIBLE_ONE_STEP")
        self.assertEqual(rising["status"], "WAIT_RISK_OFF_TURN")
        self.assertEqual(shallow["status"], "WAIT_RISK_OFF_PEAK")

    def test_proxy_or_wrong_channel_set_is_rejected(self) -> None:
        proxy = fast_overlay(3)
        proxy["conditions"][0]["data_quality"] = "PROXY"
        wrong = fast_overlay(3)
        wrong["conditions"][0]["key"] = "not_a_pdf_channel"
        for overlay in (proxy, wrong):
            result = decide("2026-01-02", (100, 0, 0), fast=overlay)
            self.assertEqual(result["decision_status"], "NO_DECISION")
            self.assertIn("FAST_OVERLAY", result["missing_channels"])

    def test_public_live_breadth_is_accepted_but_not_relabelled_exact(self) -> None:
        overlay = fast_overlay()
        overlay["conditions"][6]["data_quality"] = "PUBLIC_LIVE"
        self.assertEqual(
            decide("2026-01-02", (100, 0, 0), fast=overlay)["decision_status"],
            "OK",
        )

        conditions = reentry_conditions(
            "CONFIRMED", "CONFIRMED", "CONFIRMED", "NOT_CONFIRMED"
        )
        conditions[3]["data_quality"] = "PUBLIC_LIVE"
        result = evaluate_reentry(
            0.05, 0.10, conditions, recent_peak_p_bear=0.82
        )
        self.assertEqual(result["status"], "ELIGIBLE_ONE_STEP")
        self.assertEqual(result["conditions"][3]["data_quality"], "PUBLIC_LIVE")

        policy = decide("2026-01-02", (90, 5, 5), reentry=result)
        self.assertEqual(
            policy["reentry"]["conditions"][3]["data_quality"], "PUBLIC_LIVE"
        )


class ContractTests(unittest.TestCase):
    def test_v3_live_matches_shadow_and_requires_explicit_score(self) -> None:
        inputs = (regime(90, 5, 5), fast_overlay(3), None)
        shadow = evaluate_risk_policy(
            *inputs,
            as_of="2026-01-02",
            slow_risk_score=40,
            policy_version=RULE_STRESS_SCORE_V3_POLICY_VERSION,
        )
        live = evaluate_risk_policy(
            *inputs,
            as_of="2026-01-02",
            slow_risk_score=40,
            policy_version=RULE_STRESS_SCORE_V3_LIVE_POLICY_VERSION,
        )
        self.assertEqual(live["base_exposure"], shadow["base_exposure"])
        self.assertEqual(
            live["recommended_exposure"], shadow["recommended_exposure"]
        )
        self.assertEqual(
            live["model_version"], RULE_STRESS_SCORE_V3_LIVE_POLICY_VERSION
        )
        with self.assertRaisesRegex(ValueError, "explicit risk score"):
            evaluate_risk_policy(
                *inputs,
                as_of="2026-01-02",
                policy_version=RULE_STRESS_SCORE_V3_LIVE_POLICY_VERSION,
            )

    def test_v3_exposure_map_and_fast_cut_are_isolated(self) -> None:
        expected = {0: 1.00, 20: 1.00, 40: 0.90, 60: 0.80, 80: 0.40, 100: 0.25}
        for score, exposure in expected.items():
            with self.subTest(score=score):
                result = evaluate_risk_policy(
                    regime(90, 5, 5),
                    fast_overlay(),
                    None,
                    as_of="2026-01-02",
                    slow_risk_score=score,
                    policy_version=RULE_STRESS_SCORE_V3_POLICY_VERSION,
                )
                self.assertEqual(result["base_exposure"], exposure)
                self.assertEqual(result["recommended_exposure"], exposure)

        v2 = evaluate_risk_policy(
            regime(90, 5, 5),
            fast_overlay(),
            None,
            as_of="2026-01-02",
            slow_risk_score=40,
            policy_version=RULE_STRESS_SCORE_POLICY_VERSION,
        )
        v3_fast = evaluate_risk_policy(
            regime(90, 5, 5),
            fast_overlay(3),
            None,
            as_of="2026-01-02",
            slow_risk_score=40,
            policy_version=RULE_STRESS_SCORE_V3_POLICY_VERSION,
        )
        self.assertEqual(v2["base_exposure"], 0.75)
        self.assertEqual(v3_fast["base_exposure"], 0.90)
        self.assertEqual(v3_fast["raw_recommended_exposure"], 0.70)

    def test_v3_keeps_v2_three_day_stepwise_reentry(self) -> None:
        state = None
        exposures = []
        for offset, score in enumerate((100, 40, 40, 40, 40, 40, 40)):
            as_of = f"2026-02-{offset + 2:02d}"
            result = evaluate_risk_policy(
                regime(90, 5, 5, as_of),
                fast_overlay(),
                None,
                as_of=as_of,
                previous_state=state,
                slow_risk_score=score,
                policy_version=RULE_STRESS_SCORE_V3_POLICY_VERSION,
            )
            state = result["policy_state"]
            exposures.append(result["recommended_exposure"])

        self.assertEqual(exposures, [0.25, 0.25, 0.25, 0.40, 0.60, 0.80, 0.90])
        self.assertIsNone(state["episode_started_as_of"])

        mismatched = evaluate_risk_policy(
            regime(90, 5, 5, "2026-02-10"),
            fast_overlay(),
            None,
            as_of="2026-02-10",
            previous_state={
                **state,
                "policy_version": RULE_STRESS_SCORE_POLICY_VERSION,
            },
            slow_risk_score=40,
            policy_version=RULE_STRESS_SCORE_V3_POLICY_VERSION,
        )
        self.assertEqual(mismatched["decision_status"], "NO_DECISION")
        self.assertIn("PREVIOUS_STATE", mismatched["missing_channels"])

    def test_stress_score_recovery_accepts_improving_targets_and_ends_in_green(self) -> None:
        state = None
        exposures = []
        for offset, score in enumerate((100, 20, 20, 20, 20, 20, 20)):
            as_of = f"2026-01-{offset + 2:02d}"
            result = evaluate_risk_policy(
                regime(90, 5, 5, as_of),
                fast_overlay(),
                None,
                as_of=as_of,
                previous_state=state,
                slow_risk_score=score,
                policy_version=RULE_STRESS_SCORE_POLICY_VERSION,
            )
            state = result["policy_state"]
            exposures.append(result["recommended_exposure"])
        self.assertEqual(exposures, [0.25, 0.25, 0.25, 0.40, 0.60, 0.80, 0.90])
        self.assertIsNone(state["episode_started_as_of"])

        result = evaluate_risk_policy(
            regime(90, 5, 5, "2026-01-09"),
            fast_overlay(),
            None,
            as_of="2026-01-09",
            previous_state=state,
            slow_risk_score=10,
            policy_version=RULE_STRESS_SCORE_POLICY_VERSION,
        )
        self.assertEqual(result["recommended_exposure"], 0.95)
        self.assertEqual(result["recovery_phase"], "AT_TARGET")

    def test_rule_score_shadow_is_version_isolated(self) -> None:
        result = evaluate_risk_policy(
            regime(90, 5, 5),
            fast_overlay(),
            None,
            as_of="2026-01-02",
            slow_risk_score=24,
            policy_version=RULE_SCORE_POLICY_VERSION,
        )
        self.assertEqual(result["model_version"], RULE_SCORE_POLICY_VERSION)
        self.assertEqual(result["risk_score"], 24)
        self.assertEqual(result["recommended_exposure"], 0.87)
        self.assertEqual(
            result["policy_state"]["policy_version"], RULE_SCORE_POLICY_VERSION
        )
        with self.assertRaisesRegex(ValueError, "score override"):
            evaluate_risk_policy(
                regime(90, 5, 5),
                fast_overlay(),
                None,
                as_of="2026-01-02",
                slow_risk_score=24,
            )

    def test_scores_bands_state_and_contract(self) -> None:
        result = decide("2026-01-02", (18, 47, 35))
        self.assertEqual(result["model_version"], POLICY_VERSION)
        self.assertEqual(result["slow_risk_score"], 53)
        self.assertEqual(result["risk_score"], 53)
        self.assertEqual(result["recommended_exposure"], 0.65)
        self.assertEqual(result["risk_band"], "YELLOW")
        self.assertEqual(result["recovery_phase"], "COLD_START")
        self.assertEqual(result["policy_state"]["exposure"], 0.65)
        self.assertTrue(result["advisory_only"])

    def test_invalid_core_input_returns_no_decision(self) -> None:
        invalid_regime = evaluate_risk_policy(
            {
                "date": "2026-01-02",
                "model_version": HMM_VERSION,
                "p_bull": 0.5,
                "p_sideways": 0.5,
                "p_bear": 0.5,
            },
            fast_overlay(),
            None,
            as_of="2026-01-02",
        )
        missing_fast = evaluate_risk_policy(
            regime(85, 10, 5),
            None,
            None,
            as_of="2026-01-02",
        )
        self.assertEqual(invalid_regime["decision_status"], "NO_DECISION")
        self.assertIn("HMM_REGIME", invalid_regime["missing_channels"])
        self.assertEqual(missing_fast["decision_status"], "NO_DECISION")
        self.assertIn("FAST_OVERLAY", missing_fast["missing_channels"])
        self.assertEqual(missing_fast["fast_available_count"], 0)
        self.assertEqual(missing_fast["recovery_phase"], "NO_DECISION")


if __name__ == "__main__":
    unittest.main()
