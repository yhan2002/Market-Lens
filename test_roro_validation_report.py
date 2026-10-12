from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from roro_policy import FAST_CONDITIONS, replay_roro_policy
from roro_validation_report import (
    A2D_COLUMN,
    CAUSAL_RORO_EW_COLUMN,
    COMPARATOR_COLUMN,
    HMM_CORE_COLUMN,
    MODEL_COLUMN,
    ORIGINAL_RORO_COLUMN,
    ORIGINAL_RORO_SHA256,
    RANKED_COMPARATOR_COLUMNS,
    RORO_DATA_STATUS_COLUMN,
    RORO_POLICY_COLUMN,
    TREND_VOL_COLUMN,
    VOLATILITY_COLUMN,
    VOLATILITY_POLICY_COLUMN,
    build_validation_daily,
    evaluate_policy_ablation,
    generate_report,
    write_report,
)


class RoroValidationReportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.publish = self.root / "publish.db"
        self.shadow = self.root / "shadow.db"
        self.publish.write_bytes(b"publish")
        self.shadow.write_bytes(b"shadow")
        self.dates = pd.bdate_range("2025-12-01", periods=80)

    def _prices(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"date": self.dates, "close": 100 + np.arange(len(self.dates))}
        )

    def _router(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"risk_score": pd.Series(range(len(self.dates)), index=self.dates)}
        )

    def _history(self) -> list[dict[str, object]]:
        return [
            {
                "date": day.date().isoformat(),
                "payload": {
                    "market_risk_score": number,
                    "recommended_long": 80,
                    "data_status": "AS_AVAILABLE_STRICT",
                    "macro_risk_score": 40,
                    "market_trend_score": 35,
                    "portfolio_internal_score": 45,
                    "trajectory_1d": 1,
                    "trajectory_5d": -2,
                    "trajectory_20d": 3,
                    "data_completeness": 80,
                    "axis_disagreement": 5,
                },
            }
            for number, day in enumerate(self.dates[-40:])
        ]

    def _additional(
        self, *, ew_available: bool = False
    ) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
        frame = pd.DataFrame(
            {
                "date": self.dates,
                HMM_CORE_COLUMN: np.linspace(20, 80, len(self.dates)),
                A2D_COLUMN: np.linspace(25, 75, len(self.dates)),
                VOLATILITY_COLUMN: np.linspace(10, 90, len(self.dates)),
                VOLATILITY_POLICY_COLUMN: np.linspace(10, 90, len(self.dates)),
                TREND_VOL_COLUMN: np.linspace(15, 85, len(self.dates)),
                CAUSAL_RORO_EW_COLUMN: np.linspace(30, 70, len(self.dates))
                if ew_available
                else np.nan,
                ORIGINAL_RORO_COLUMN: np.nan,
            }
        )
        metadata = {
            HMM_CORE_COLUMN: {"model_version": "core", "status": "ROLLBACK_COMPARATOR"},
            A2D_COLUMN: {
                "model_version": "a2d",
                "status": "MANUAL_OVERRIDE_ACTIVE_HISTORICAL_FAIL_COMPARATOR",
            },
            VOLATILITY_COLUMN: {"model_version": "vol", "status": "COMPARATOR"},
            TREND_VOL_COLUMN: {"model_version": "trend-vol", "status": "COMPARATOR"},
            CAUSAL_RORO_EW_COLUMN: {
                "model_version": "ew",
                "status": "COMPARATOR" if ew_available else "DATA_BLOCKED",
                "reason": None if ew_available else "missing inputs",
            },
            ORIGINAL_RORO_COLUMN: {
                "model_version": "original",
                "status": "REFERENCE_ONLY",
                "reason": "non-causal",
            },
        }
        return frame, metadata

    @patch("roro_validation_report.build_additional_comparators")
    @patch("roro_validation_report.build_router_scores_from_database")
    @patch("roro_validation_report.read_shadow_history")
    @patch("roro_validation_report.load_benchmark_from_database")
    def test_exact_date_join_never_fills_missing_roro(
        self, load_prices, read_history, load_router, load_additional
    ) -> None:
        load_prices.return_value = self._prices()
        read_history.return_value = self._history()[1:]
        load_router.return_value = self._router()
        load_additional.return_value = self._additional()
        daily, metadata = build_validation_daily(self.publish, self.shadow)
        first_shadow_date = self.dates[-40]
        row = daily.loc[daily["date"] == first_shadow_date].iloc[0]
        self.assertTrue(pd.isna(row[MODEL_COLUMN]))
        self.assertEqual(row[COMPARATOR_COLUMN], 40)
        self.assertEqual(row[HMM_CORE_COLUMN], self._additional()[0].iloc[40][HMM_CORE_COLUMN])
        self.assertEqual(metadata[CAUSAL_RORO_EW_COLUMN]["status"], "DATA_BLOCKED")
        self.assertTrue(pd.isna(row[RORO_DATA_STATUS_COLUMN]))
        self.assertTrue(pd.isna(row[RORO_POLICY_COLUMN]))
        next_row = daily.loc[daily["date"] == self.dates[-39]].iloc[0]
        self.assertEqual(next_row[RORO_DATA_STATUS_COLUMN], "AS_AVAILABLE_STRICT")
        self.assertEqual(len(daily), len(self.dates))

    @patch("roro_validation_report.build_additional_comparators")
    @patch("roro_validation_report.build_router_scores_from_database")
    @patch("roro_validation_report.read_shadow_history")
    @patch("roro_validation_report.load_benchmark_from_database")
    def test_report_is_insufficient_and_writes_audit_files(
        self, load_prices, read_history, load_router, load_additional
    ) -> None:
        load_prices.return_value = self._prices()
        read_history.return_value = self._history()
        load_router.return_value = self._router()
        load_additional.return_value = self._additional()
        report, labelled = generate_report(
            self.publish,
            self.shadow,
            as_of=self.dates[-1].date().isoformat(),
            bootstrap_samples=5,
        )
        self.assertEqual(report["status"], "INSUFFICIENT_EVIDENCE")
        self.assertFalse(report["overall_pass"])
        self.assertEqual(report["definition"]["output_start"], "2026-01-07")
        self.assertEqual(report["q20_readiness"]["status"], "INSUFFICIENT_EVIDENCE")
        self.assertIsNone(report["q20_readiness"]["probability"])
        self.assertEqual(
            report["blocked_comparators"][CAUSAL_RORO_EW_COLUMN]["status"],
            "DATA_BLOCKED",
        )
        self.assertIn(A2D_COLUMN, report["paired_comparisons"])
        self.assertIn("score_diagnostics", report)
        self.assertEqual(
            set(report["score_diagnostics"]["windows"]), {"63", "126", "252"}
        )
        self.assertEqual(report["policy_comparison_10bp"]["transaction_cost_bps"], 10)
        self.assertEqual(report["coverage_status"]["data_status"], "AVAILABLE")
        self.assertEqual(report["reentry_relapse_10d"]["status"], "DATA_BLOCKED")
        self.assertEqual(report["policy_ablation_10bp"]["status"], "DATA_BLOCKED")
        self.assertEqual(report["red_alert_onset"]["status"], "DATA_BLOCKED")
        self.assertTrue(report["yearly_auc"]["descriptive_only"])
        summaries = list(report["policy_comparison_10bp"]["models"].values())
        self.assertEqual(len({item["rows"] for item in summaries}), 1)
        self.assertEqual(len({item["start"] for item in summaries}), 1)
        self.assertEqual(len({item["end"] for item in summaries}), 1)
        self.assertEqual(report["provenance"]["as_of"], self.dates[-1].date().isoformat())
        self.assertEqual(report["provenance"]["roro_txt_sha256"], ORIGINAL_RORO_SHA256)
        paths = write_report(report, labelled, self.root / "out")
        self.assertTrue(all(Path(path).is_file() for path in paths.values()))
        markdown = Path(paths["markdown"]).read_text("utf-8")
        self.assertIn("INSUFFICIENT_EVIDENCE", markdown)
        self.assertIn("MANUAL_OVERRIDE_ACTIVE_HISTORICAL_FAIL_COMPARATOR", markdown)
        self.assertIn("Integer score path diagnostics", markdown)
        self.assertIn("Net CAGR", markdown)
        self.assertIn("Score and data-status coverage", markdown)
        self.assertIn("REENTRY_DIRECT 10-session relapse", markdown)
        self.assertIn("Score and Fast policy ablation", markdown)

    def test_exact_policy_ablation_replays_and_fails_closed(self) -> None:
        dates = pd.bdate_range("2026-01-07", periods=15)

        def snapshot(day: pd.Timestamp, *, triggered: int = 0) -> dict[str, object]:
            conditions = []
            for index, key in enumerate(FAST_CONDITIONS):
                status = "TRIGGERED" if index < triggered else "CLEAR"
                conditions.append(
                    {
                        "key": key,
                        "label": key,
                        "status": status,
                        "data_quality": "EXACT",
                        "as_of": day.date().isoformat(),
                    }
                )
            return {
                "as_of": day.date().isoformat(),
                "status": "TRIGGERED" if triggered >= 3 else "CLEAR",
                "count": triggered,
                "trigger_count": triggered,
                "required": 3,
                "total": 8,
                "complete": True,
                "primary_driver": "FAST" if triggered else "CLEAR",
                "conditions": conditions,
            }

        decisions = [
            {
                "as_of": day.date().isoformat(),
                "risk_score": 66 if index < 4 else 50,
                "fast_snapshot": snapshot(day, triggered=3 if index == 1 else 0),
            }
            for index, day in enumerate(dates)
        ]
        combined = replay_roro_policy(decisions)
        daily = pd.DataFrame(
            {
                "date": dates,
                "nq_close": 100 + np.arange(len(dates), dtype=float),
                MODEL_COLUMN: [decision["risk_score"] for decision in decisions],
                "recommended_long_pct": [
                    item["recommended_long"] for item in combined
                ],
                RORO_POLICY_COLUMN: combined,
            }
        )
        result = evaluate_policy_ablation(
            daily, as_of=dates[-1].date().isoformat()
        )
        self.assertEqual(result["status"], "AVAILABLE")
        self.assertTrue(result["combined_replay_matches_stored"])
        self.assertEqual(
            set(result["models"]), {"score_only", "overlay_only", "combined"}
        )
        self.assertEqual(
            len({summary["rows"] for summary in result["models"].values()}), 1
        )

        blocked = evaluate_policy_ablation(
            daily.assign(**{RORO_POLICY_COLUMN: None}),
            as_of=dates[-1].date().isoformat(),
        )
        self.assertEqual(blocked["status"], "DATA_BLOCKED")
        tampered = daily.copy()
        tampered.loc[3, "recommended_long_pct"] += 1
        with self.assertRaisesRegex(ValueError, "stored Long"):
            evaluate_policy_ablation(
                tampered, as_of=dates[-1].date().isoformat()
            )

    @patch("roro_validation_report.build_additional_comparators")
    @patch("roro_validation_report.build_router_scores_from_database")
    @patch("roro_validation_report.read_shadow_history")
    @patch("roro_validation_report.load_benchmark_from_database")
    def test_causal_ew_is_ranked_only_when_available(
        self, load_prices, read_history, load_router, load_additional
    ) -> None:
        self.assertIn(CAUSAL_RORO_EW_COLUMN, RANKED_COMPARATOR_COLUMNS)
        load_prices.return_value = self._prices()
        read_history.return_value = self._history()
        load_router.return_value = self._router()
        load_additional.return_value = self._additional(ew_available=True)
        report, _ = generate_report(
            self.publish,
            self.shadow,
            as_of=self.dates[-1].date().isoformat(),
            bootstrap_samples=5,
        )
        self.assertIn(CAUSAL_RORO_EW_COLUMN, report["paired_comparisons"])
        self.assertNotIn(CAUSAL_RORO_EW_COLUMN, report.get("blocked_comparators", {}))


if __name__ == "__main__":
    unittest.main()
