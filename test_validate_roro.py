from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from validate_roro import (
    RORO_TXT_ORIGINAL,
    build_causal_validation_frame,
    common_labelled_frame,
    coverage_status_summary,
    evaluate_exposure_policy,
    forward_path_drawdown,
    red_alert_onset_diagnostics,
    reentry_relapse_diagnostics,
    score_path_diagnostics,
    validate_roro,
    yearly_auc_summary,
)


class ValidateRoroTest(unittest.TestCase):
    def test_future_label_uses_only_t_plus_one_path_and_embargo(self) -> None:
        prices = np.array([100.0, 110.0, 90.0, 120.0, 80.0])
        drawdown = forward_path_drawdown(prices, horizon=2)
        self.assertAlmostEqual(drawdown[0], -0.10)
        self.assertAlmostEqual(drawdown[1], 90.0 / 110.0 - 1.0)
        self.assertTrue(np.isnan(drawdown[-1]))
        self.assertTrue(np.isnan(drawdown[-2]))

        dates = pd.bdate_range("2026-01-02", periods=5)
        frame = build_causal_validation_frame(
            pd.DataFrame(
                {"date": dates, "nq_close": prices, "roro_score": range(5)}
            ),
            score_columns=["roro_score"],
            horizon=2,
            as_of=dates[2],
        )
        self.assertEqual(frame["label_complete"].tolist(), [True, False, False])
        self.assertEqual(frame.loc[0, "label_available_date"], dates[2])
        self.assertTrue(pd.isna(frame.loc[1, "future_path_drawdown"]))
        self.assertTrue(pd.isna(frame.loc[1, "drawdown_event"]))

    def test_exact_common_date_intersection(self) -> None:
        dates = pd.bdate_range("2026-01-02", periods=30)
        daily = pd.DataFrame(
            {
                "date": dates,
                "nq_close": np.linspace(100, 120, len(dates)),
                "roro_score": np.arange(len(dates), dtype=float),
                "benchmark_a": np.arange(len(dates), dtype=float),
                "benchmark_b": np.arange(len(dates), dtype=float),
            }
        )
        daily.loc[3, "benchmark_a"] = np.nan
        daily.loc[7, "benchmark_b"] = np.nan
        frame = build_causal_validation_frame(
            daily,
            score_columns=["roro_score", "benchmark_a", "benchmark_b"],
            horizon=2,
        )
        common = common_labelled_frame(
            frame, ["roro_score", "benchmark_a", "benchmark_b"]
        )
        expected = set(dates[:28]) - {dates[3], dates[7]}
        self.assertEqual(set(common["date"]), expected)
        self.assertEqual(len(common), 26)

    def test_blocked_and_insufficient_statuses(self) -> None:
        dates = pd.bdate_range("2026-01-02", periods=80)
        base = pd.DataFrame(
            {
                "date": dates,
                "nq_close": 100 + np.sin(np.arange(80)) * 5,
                "roro_score": np.nan,
                "benchmark": np.arange(80, dtype=float),
            }
        )
        blocked = validate_roro(
            base,
            benchmark_columns=["benchmark"],
            horizon=2,
            bootstrap_samples=5,
        )
        self.assertEqual(blocked["status"], "DATA_BLOCKED")
        base["roro_score"] = np.arange(80, dtype=float)
        insufficient = validate_roro(
            base,
            benchmark_columns=["benchmark"],
            horizon=2,
            bootstrap_samples=5,
        )
        self.assertEqual(insufficient["status"], "INSUFFICIENT_EVIDENCE")
        self.assertFalse(insufficient["overall_pass"])

    def test_exposure_is_applied_t_plus_one_with_only_10bp_cost(self) -> None:
        dates = pd.bdate_range("2026-01-02", periods=4)
        result, summary = evaluate_exposure_policy(
            pd.DataFrame(
                {
                    "date": dates,
                    "nq_close": [100.0, 110.0, 99.0, 99.0],
                    "recommended_long_pct": [0.0, 100.0, 50.0, 50.0],
                }
            )
        )
        self.assertEqual(result.iloc[0]["signal_date"], dates[0])
        self.assertEqual(result.iloc[0]["applied_long_pct"], 0.0)
        self.assertAlmostEqual(result.iloc[0]["gross_return"], 0.0)
        self.assertAlmostEqual(result.iloc[0]["turnover"], 1.0)
        self.assertAlmostEqual(result.iloc[0]["transaction_cost_10bp"], 0.001)
        self.assertAlmostEqual(result.iloc[0]["net_10bp_return"], -0.001)
        self.assertAlmostEqual(result.iloc[1]["gross_return"], -0.10)
        self.assertEqual(summary["transaction_cost_bps"], 10)
        self.assertEqual(summary["initial_wealth"], 1.0)
        self.assertTrue(np.isfinite(summary["net_10bp_cagr"]))
        self.assertTrue(np.isfinite(summary["net_10bp_sharpe"]))
        self.assertTrue(np.isfinite(summary["net_10bp_max_drawdown"]))
        self.assertTrue(np.isfinite(summary["nq_benchmark_cagr"]))
        self.assertTrue(np.isfinite(summary["nq_benchmark_sharpe"]))
        self.assertTrue(np.isfinite(summary["nq_benchmark_max_drawdown"]))
        self.assertTrue(np.isfinite(summary["annual_return_drag"]))
        self.assertTrue(np.isfinite(summary["relative_mdd_improvement"]))
        self.assertFalse(any("25" in column for column in result.columns))

    def test_policy_metrics_include_initial_wealth_in_drawdown(self) -> None:
        dates = pd.bdate_range("2026-01-02", periods=4)
        _, summary = evaluate_exposure_policy(
            pd.DataFrame(
                {
                    "date": dates,
                    "nq_close": [100.0, 90.0, 99.0, 89.1],
                    "recommended_long_pct": 100.0,
                }
            )
        )
        self.assertAlmostEqual(summary["net_10bp_max_drawdown"], -0.109)
        self.assertAlmostEqual(summary["nq_benchmark_max_drawdown"], -0.109)
        self.assertAlmostEqual(summary["annual_return_drag"], 0.0)
        self.assertAlmostEqual(summary["relative_mdd_improvement"], 0.0)

    def test_policy_metrics_reject_nonfinite_returns(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            evaluate_exposure_policy(
                pd.DataFrame(
                    {
                        "date": pd.bdate_range("2026-01-02", periods=3),
                        "nq_close": [100.0, 0.0, 100.0],
                        "recommended_long_pct": 100.0,
                    }
                )
            )

    def test_integer_score_and_regime_path_diagnostics(self) -> None:
        diagnostics = score_path_diagnostics([10, 10, 40, 40, 70, 70, 70, 40])
        self.assertEqual(set(diagnostics["windows"]), {"63", "126", "252"})
        recent = diagnostics["windows"]["63"]
        self.assertAlmostEqual(recent["mode_share"], 3 / 8)
        self.assertEqual(recent["unique_scores"], 3)
        self.assertEqual(recent["score_changes"], 3)
        self.assertEqual(recent["longest_run_days"], 3)
        self.assertEqual(recent["terminal_run_days"], 1)
        self.assertEqual(
            recent["regime_occupancy"],
            {"GREEN": 0.25, "YELLOW": 0.375, "RED": 0.375},
        )
        self.assertEqual(recent["regime_runs"]["YELLOW"]["median_run_days"], 1.5)
        self.assertEqual(recent["regime_runs"]["RED"]["max_run_days"], 3)
        self.assertEqual(recent["regime_transitions"], 3)

    def test_red_alert_clusters_lead_false_pending_and_t_exclusion(self) -> None:
        dates = pd.bdate_range("2026-01-02", periods=330)
        prices = np.full(len(dates), 100.0)
        prices[260:270] = 94.0
        prices[270:] = 98.0
        scores = np.full(len(dates), 10.0)
        scores[[215, 235, 255, 260, 285, 320]] = 70.0
        result = red_alert_onset_diagnostics(
            pd.DataFrame(
                {"date": dates, "nq_close": prices, "roro_score": scores}
            )
        )
        self.assertEqual(result["onsets"], 1)
        self.assertEqual(result["matched_onsets"], 1)
        self.assertEqual(result["median_lead_sessions"], 5.0)
        self.assertEqual(result["onset_details"][0]["first_valid_alert_date"], dates[255].date().isoformat())
        self.assertNotEqual(
            result["onset_details"][0]["first_valid_alert_date"],
            dates[260].date().isoformat(),
        )
        self.assertEqual(result["alert_clusters"], 3)
        self.assertEqual(result["matched_clusters"], 1)
        self.assertEqual(result["false_alert_clusters"], 1)
        self.assertEqual(result["pending_clusters"], 1)
        self.assertEqual(result["cluster_details"][0]["alert_count"], 4)

    def test_episode_results_are_invariant_when_future_rows_append(self) -> None:
        dates = pd.bdate_range("2026-01-02", periods=310)
        prices = np.full(len(dates), 100.0)
        prices[270:280] = 94.0
        prices[280:] = 98.0
        scores = np.full(len(dates), 10.0)
        scores[[260, 285]] = 70.0
        prefix = pd.DataFrame(
            {"date": dates, "nq_close": prices, "roro_score": scores}
        )
        appended_dates = pd.bdate_range(dates[-1] + pd.offsets.BDay(), periods=30)
        full = pd.concat(
            [
                prefix,
                pd.DataFrame(
                    {
                        "date": appended_dates,
                        "nq_close": 98.0,
                        "roro_score": 10.0,
                    }
                ),
            ],
            ignore_index=True,
        )
        before = red_alert_onset_diagnostics(prefix)
        after = red_alert_onset_diagnostics(full)
        self.assertEqual(before["onset_details"], after["onset_details"])
        self.assertEqual(before["cluster_details"], after["cluster_details"])

    def test_reentry_relapse_excludes_t_includes_exact_threshold_and_tail(self) -> None:
        dates = pd.bdate_range("2026-01-02", periods=15)
        prices = np.array(
            [100.0, 100.0, 97.0, 97.0, 94.0] + [94.0] * 10
        )
        policies = [{"phase": "AT_TARGET"} for _ in dates]
        policies[0] = {"phase": "REENTRY_DIRECT"}
        policies[4] = {"phase": "REENTRY_DIRECT"}
        policies[10] = {"phase": "REENTRY_DIRECT"}
        daily = pd.DataFrame(
            {"date": dates, "nq_close": prices, "policy": policies}
        )
        result = reentry_relapse_diagnostics(daily)
        self.assertEqual(result["events"], 3)
        self.assertEqual(result["evaluated_events"], 2)
        self.assertEqual(result["failures"], 1)
        self.assertEqual(result["tail_pending_events"], 1)
        self.assertEqual(result["event_details"][0]["status"], "FAIL")
        self.assertAlmostEqual(
            result["event_details"][0]["minimum_forward_return"], -0.06
        )
        self.assertEqual(result["event_details"][1]["status"], "PASS")
        self.assertEqual(result["event_details"][2]["status"], "PENDING")
        blocked = reentry_relapse_diagnostics(daily.assign(policy=None))
        self.assertEqual(blocked["status"], "DATA_BLOCKED")
        self.assertEqual(
            reentry_relapse_diagnostics(daily.drop(columns="policy"))["status"],
            "DATA_BLOCKED",
        )

        exact = daily.copy()
        exact["nq_close"] = 100.0
        exact.loc[2, "nq_close"] = 97.0
        exact["policy"] = [{"phase": "AT_TARGET"} for _ in dates]
        exact.at[0, "policy"] = {"phase": "REENTRY_DIRECT"}
        self.assertEqual(
            reentry_relapse_diagnostics(exact)["event_details"][0]["status"],
            "FAIL",
        )

    def test_reentry_completed_event_is_future_append_invariant(self) -> None:
        dates = pd.bdate_range("2026-01-02", periods=12)
        policies = [{"phase": "AT_TARGET"} for _ in dates]
        policies[0] = {"phase": "REENTRY_DIRECT"}
        prefix = pd.DataFrame(
            {"date": dates, "nq_close": 100.0, "policy": policies}
        )
        future_dates = pd.bdate_range(dates[-1] + pd.offsets.BDay(), periods=10)
        full = pd.concat(
            [
                prefix,
                pd.DataFrame(
                    {
                        "date": future_dates,
                        "nq_close": 90.0,
                        "policy": [{"phase": "AT_TARGET"}] * 10,
                    }
                ),
            ],
            ignore_index=True,
        )
        self.assertEqual(
            reentry_relapse_diagnostics(prefix)["event_details"][0],
            reentry_relapse_diagnostics(full)["event_details"][0],
        )

    def test_coverage_status_and_yearly_auc_are_descriptive(self) -> None:
        coverage = coverage_status_summary(
            [10.0, np.nan, 20.0], ["FULL_STRICT", "NO_SCORE", "AS_AVAILABLE_STRICT"]
        )
        self.assertEqual(coverage["score_rows"], 2)
        self.assertEqual(coverage["no_score_rows"], 1)
        self.assertAlmostEqual(coverage["coverage_share"], 2 / 3)
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2026-01-02", "2026-01-05", "2027-01-04", "2027-01-05"]
                ),
                "drawdown_event": [0, 1, 0, 0],
                "candidate": [0.0, 1.0, 0.0, 1.0],
            }
        )
        yearly = yearly_auc_summary(frame, ["candidate"])
        self.assertEqual(yearly["years"]["2026"]["models"]["candidate"]["roc_auc"], 1.0)
        self.assertIsNone(
            yearly["years"]["2027"]["models"]["candidate"]["roc_auc"]
        )
        self.assertTrue(yearly["descriptive_only"])

    def test_policy_summary_stops_at_as_of(self) -> None:
        dates = pd.bdate_range("2026-01-02", periods=80)
        daily = pd.DataFrame(
            {
                "date": dates,
                "nq_close": 100 + np.arange(80, dtype=float),
                "roro_score": np.arange(80, dtype=float),
                "benchmark": np.arange(80, dtype=float),
                "recommended_long_pct": 80.0,
            }
        )
        cutoff = dates[49]
        report = validate_roro(
            daily,
            benchmark_columns=["benchmark"],
            long_column="recommended_long_pct",
            as_of=cutoff,
            horizon=2,
            bootstrap_samples=5,
        )
        self.assertEqual(report["policy_10bp"]["end"], cutoff.date().isoformat())
        revised = daily.copy()
        revised.loc[revised["date"] > cutoff, "nq_close"] *= 10
        repeated = validate_roro(
            revised,
            benchmark_columns=["benchmark"],
            long_column="recommended_long_pct",
            as_of=cutoff,
            horizon=2,
            bootstrap_samples=5,
        )
        self.assertEqual(report["policy_10bp"], repeated["policy_10bp"])

    def test_original_roro_is_reference_only_and_not_paired(self) -> None:
        dates = pd.bdate_range("2026-01-02", periods=120)
        daily = pd.DataFrame(
            {
                "date": dates,
                "nq_close": 100 + np.sin(np.arange(120) / 2) * 10,
                "roro_score": np.arange(120, dtype=float),
                "benchmark": np.arange(120, dtype=float)[::-1],
                RORO_TXT_ORIGINAL: np.arange(120, dtype=float),
            }
        )
        report = validate_roro(
            daily,
            benchmark_columns=["benchmark", RORO_TXT_ORIGINAL],
            horizon=2,
            bootstrap_samples=5,
        )
        self.assertEqual(report["models"][RORO_TXT_ORIGINAL]["role"], "REFERENCE_ONLY")
        self.assertNotIn(RORO_TXT_ORIGINAL, report["paired_comparisons"])
        self.assertEqual(set(report["paired_comparisons"]), {"benchmark"})

    def test_output_cannot_start_before_frozen_date(self) -> None:
        with self.assertRaisesRegex(ValueError, "2026-01-02"):
            build_causal_validation_frame(
                pd.DataFrame(
                    {
                        "date": pd.bdate_range("2025-12-01", periods=30),
                        "nq_close": np.arange(30) + 100,
                        "roro_score": np.arange(30),
                    }
                ),
                score_columns=["roro_score"],
                output_start="2025-12-01",
            )


if __name__ == "__main__":
    unittest.main()
