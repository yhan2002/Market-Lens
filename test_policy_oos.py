import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from policy_oos import (
    BENCHMARK,
    HMM_MODEL_VERSION,
    MIN_OBSERVATIONS,
    V3_COST_SCENARIOS_BPS,
    _validate_inputs,
    align_cash_returns,
    build_evaluation_frame,
    decisions_frame,
    evaluate,
    performance_metrics,
    rerisk_failure_summary,
    sha256_file,
    write_outputs,
)
from risk_policy import (
    POLICY_VERSION,
    RULE_STRESS_SCORE_V3_POLICY_VERSION,
    evaluate_risk_policy,
)


def sample_frames(count: int = MIN_OBSERVATIONS + 1):
    dates = pd.bdate_range("2020-01-01", periods=count)
    prices = pd.DataFrame(
        {"date": dates, "close": 100 * np.cumprod(np.full(count, 1.001))}
    )
    probabilities = pd.DataFrame(
        {
            "model_version": HMM_MODEL_VERSION,
            "date": dates,
            "p_bull": 1.0,
            "p_sideways": 0.0,
            "p_bear": 0.0,
        }
    )
    return prices, probabilities


def fast_overlay(unavailable: int = 0) -> dict[str, object]:
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
    return {
        "conditions": [
            {
                "key": key,
                "label": key,
                "status": "UNAVAILABLE" if index < unavailable else "CLEAR",
                "data_quality": "UNAVAILABLE" if index < unavailable else "EXACT",
                "detail": "test",
            }
            for index, key in enumerate(keys)
        ]
    }


def replay_records(
    probabilities: pd.DataFrame,
    *,
    unavailable: int = 0,
    policy_version: str = POLICY_VERSION,
    slow_risk_score: int | None = None,
) -> list[dict[str, object]]:
    state = None
    records = []
    for row in probabilities.sort_values("date").itertuples(index=False):
        as_of = pd.Timestamp(row.date).date().isoformat()
        values = {"BULL": row.p_bull, "SIDEWAYS": row.p_sideways, "BEAR": row.p_bear}
        regime = {
            "date": as_of,
            "model_version": HMM_MODEL_VERSION,
            "state": max(values, key=values.get),
            "p_bull": row.p_bull,
            "p_sideways": row.p_sideways,
            "p_bear": row.p_bear,
        }
        policy = evaluate_risk_policy(
            regime,
            fast_overlay(unavailable),
            None,
            as_of=as_of,
            previous_state=state,
            slow_risk_score=slow_risk_score,
            policy_version=policy_version,
        )
        if policy["decision_status"] == "OK":
            state = policy["policy_state"]
        records.append({"as_of": as_of, "regime": regime, "risk_policy": policy})
    return records


def create_database(path: Path, count: int = MIN_OBSERVATIONS + 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices, probabilities = sample_frames(count)
    with closing(sqlite3.connect(path)) as connection:
        connection.execute(
            "CREATE TABLE daily_prices(ticker TEXT, date TEXT, close REAL, adj_close REAL)"
        )
        connection.execute(
            """
            CREATE TABLE regime_model_probabilities(
                model_version TEXT, date TEXT, p_bull REAL,
                p_sideways REAL, p_bear REAL
            )
            """
        )
        connection.executemany(
            "INSERT INTO daily_prices VALUES (?, ?, ?, ?)",
            [
                (BENCHMARK, row.date.date().isoformat(), row.close, row.close)
                for row in prices.itertuples(index=False)
            ],
        )
        connection.executemany(
            "INSERT INTO regime_model_probabilities VALUES (?, ?, ?, ?, ?)",
            [
                (
                    row.model_version,
                    row.date.date().isoformat(),
                    row.p_bull,
                    row.p_sideways,
                    row.p_bear,
                )
                for row in probabilities.itertuples(index=False)
            ],
        )
        connection.commit()
    return prices, probabilities


class PolicyOosTest(unittest.TestCase):
    def test_cash_return_uses_only_rate_available_at_holding_start(self):
        dates = pd.Series(pd.to_datetime([
            "2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"
        ]))
        rates = pd.DataFrame(
            {
                "observation_date": ["2026-01-01", "2026-01-05"],
                "annual_yield_pct": [3.65, 7.30],
                "available_at": ["2026-01-02T00:00:00Z", "2026-01-06T00:00:00Z"],
                "source": ["ALFRED:DGS3MO:OUTPUT3"] * 2,
            }
        )
        cash = align_cash_returns(dates, rates)
        self.assertAlmostEqual(cash.loc[1, "cash_return"], 0.0365 * 3 / 365)
        self.assertAlmostEqual(cash.loc[2, "cash_return"], 0.0365 / 365)
        self.assertAlmostEqual(cash.loc[3, "cash_return"], 0.0730 / 365)

        missing = rates.iloc[[1]].copy()
        with self.assertRaisesRegex(ValueError, "coverage is missing"):
            align_cash_returns(dates, missing)

    def test_v3_cost_set_and_cash_contribution_are_explicit(self):
        prices, probabilities = sample_frames(8)
        decisions = decisions_frame(replay_records(probabilities))
        rates = pd.DataFrame(
            {
                "observation_date": ["2019-12-31"],
                "annual_yield_pct": [3.65],
                "available_at": ["2019-12-31T00:00:00Z"],
                "source": ["ALFRED:DGS3MO:OUTPUT3"],
            }
        )
        with patch("policy_oos.MIN_OBSERVATIONS", 1):
            frame = build_evaluation_frame(
                prices,
                probabilities,
                decisions,
                cash_rates=rates,
                cost_scenarios_bps=V3_COST_SCENARIOS_BPS,
            )
        expected = (
            frame["applied_exposure"] * frame["benchmark_return"]
            + (1 - frame["applied_exposure"]) * frame["cash_return"]
        )
        np.testing.assert_allclose(frame["gross_return"], expected)
        self.assertIn("net_10bps_return", frame)
        self.assertNotIn("net_25bps_return", frame)

    def test_same_engine_decision_applies_only_to_next_nq_bar(self):
        prices, probabilities = sample_frames(8)
        probabilities.loc[3, ["p_bull", "p_sideways", "p_bear"]] = [0.0, 0.0, 1.0]
        records = replay_records(probabilities)
        decisions = decisions_frame(records)
        with patch("policy_oos.MIN_OBSERVATIONS", 1):
            frame = build_evaluation_frame(prices, probabilities, decisions)

        bear_date = probabilities.loc[3, "date"]
        bear_row = frame.loc[frame["date"].eq(bear_date)].iloc[0]
        next_row = frame.loc[frame["date"].gt(bear_date)].iloc[0]
        self.assertEqual(bear_row["applied_exposure"], 0.95)
        self.assertEqual(next_row["applied_exposure"], 0.40)
        self.assertEqual(next_row["signal_date"], bear_date)

    def test_missing_decision_day_keeps_return_and_prior_exposure(self):
        prices, probabilities = sample_frames(MIN_OBSERVATIONS + 2)
        records = replay_records(probabilities)
        decisions = decisions_frame(records).drop(index=10).reset_index(drop=True)
        missing_date = probabilities.loc[10, "date"]
        frame = build_evaluation_frame(prices, probabilities, decisions)

        self.assertIn(missing_date, set(frame["date"]))
        self.assertIn(prices.loc[11, "date"], set(frame["date"]))
        missing_row = frame.loc[frame["date"].eq(missing_date)].iloc[0]
        self.assertEqual(missing_row["applied_exposure"], 0.95)
        np.testing.assert_allclose(
            frame["net_10bps_return"],
            frame["gross_return"] - frame["turnover"] * 10 / 10_000,
        )

    def test_future_decisions_do_not_rewrite_past_returns(self):
        prices, probabilities = sample_frames(MIN_OBSERVATIONS + 5)
        changed = probabilities.copy()
        changed.loc[changed.index >= 900, ["p_bull", "p_sideways", "p_bear"]] = [
            0.1,
            0.2,
            0.7,
        ]
        original = build_evaluation_frame(
            prices, probabilities, decisions_frame(replay_records(probabilities))
        )
        revised = build_evaluation_frame(
            prices, changed, decisions_frame(replay_records(changed))
        )
        pd.testing.assert_series_equal(
            original.loc[:898, "gross_return"],
            revised.loc[:898, "gross_return"],
        )

    def test_initial_turnover_is_measured_from_fully_invested_overlay(self):
        prices, probabilities = sample_frames(MIN_OBSERVATIONS + 1)
        frame = build_evaluation_frame(
            prices, probabilities, decisions_frame(replay_records(probabilities))
        )
        self.assertAlmostEqual(frame.iloc[0]["turnover"], 0.05)

    def test_validation_rejects_duplicates_bad_probability_and_wrong_model(self):
        prices, probabilities = sample_frames()
        duplicated = pd.concat([prices, prices.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate price date"):
            _validate_inputs(duplicated, probabilities)

        invalid = probabilities.copy()
        invalid.loc[0, "p_bear"] = 0.3
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            _validate_inputs(prices, invalid)

        wrong = probabilities.copy()
        wrong["model_version"] = "another-model"
        with self.assertRaisesRegex(ValueError, "model-version mismatch"):
            _validate_inputs(prices, wrong)
        _, candidate = _validate_inputs(prices, wrong, "another-model")
        self.assertTrue(candidate["model_version"].eq("another-model").all())

    def test_mdd_includes_initial_capital(self):
        metrics = performance_metrics(pd.Series([-0.10, 0.01]))
        self.assertAlmostEqual(metrics["max_drawdown"], -0.10)

    def test_rerisk_failure_uses_next_ten_bars_and_bear_recurrence(self):
        dates = pd.bdate_range("2026-01-02", periods=12)
        decisions = pd.DataFrame(
            {
                "date": dates,
                "recovery_phase": ["RECOVERY_STEP"] + ["AT_TARGET"] * 11,
                "episode_started_as_of": ["2025-12-20"] * 12,
                "p_bull": [0.7] * 12,
                "p_sideways": [0.2] * 12,
                "p_bear": [0.1] * 12,
            }
        )
        prices = pd.DataFrame(
            {
                "date": dates,
                "close": [100, 99, 98, 96, 97, 98, 99, 100, 101, 102, 103, 104],
            }
        )
        result = rerisk_failure_summary(decisions, prices)
        self.assertEqual(result["evaluated_events"], 1)
        self.assertEqual(result["failures"], 1)
        self.assertEqual(result["status"], "NOT_EVALUATED")
        self.assertLessEqual(result["details"][0]["minimum_10bar_return"], -0.03)

    def test_database_is_read_only_and_outputs_record_same_engine(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "fixture.db"
            _, probabilities = create_database(database)
            records = replay_records(probabilities, unavailable=3)
            before = sha256_file(database)

            with patch("policy_oos.replay_policy_history", return_value=records):
                messages = []
                report, frame = evaluate(database, progress=messages.append)
            paths = write_outputs(report, frame, root / "outputs")

            self.assertEqual(before, sha256_file(database))
            self.assertEqual(len(messages), 4)
            self.assertIn("동일 비중 규칙", messages[1])
            self.assertEqual(report["scope"], "SEQUENTIAL_POLICY_AS_AVAILABLE")
            self.assertEqual(report["model_version"], POLICY_VERSION)
            self.assertEqual(
                report["model_id"],
                f"{POLICY_VERSION}--nq-core-v1-sequential-policy",
            )
            self.assertFalse(report["gates"]["full_pdf_policy_evaluated"])
            self.assertEqual(report["full_pdf_gate"]["status"], "NOT_EVALUATED")
            self.assertEqual(report["excluded_components"], [])
            self.assertEqual(report["observations"], MIN_OBSERVATIONS)
            self.assertEqual(report["coverage"]["decision_days"], MIN_OBSERVATIONS + 1)
            self.assertEqual(report["coverage"]["fast_complete_days"], 0)
            self.assertEqual(set(report["scenarios"]), {"gross", "cost_10bps", "cost_25bps"})
            self.assertEqual(report["provenance"]["database"]["sha256"], before)
            self.assertEqual(
                set(report["provenance"]["source_sha256"]),
                {
                    "policy_oos.py",
                    "risk_policy.py",
                    "site_publisher.py",
                    "risk_signals.py",
                    "MODEL_SPEC_v1.md",
                    "MODEL_SPEC_ROUTER_STATE_v2.md",
                },
            )
            parsed = json.loads(paths["json"].read_text(encoding="utf-8"))
            self.assertIn(
                parsed["evaluation_status"],
                {"AS_AVAILABLE_HISTORICAL_PASS", "AS_AVAILABLE_HISTORICAL_FAIL"},
            )
            self.assertIn(parsed["historical_metric_gate"]["status"], {"PASS", "FAIL"})
            self.assertTrue(paths["markdown"].is_file())
            presentation = paths["presentation"].read_text(encoding="utf-8")
            self.assertIn("동일 규칙 과거 성과", presentation)
            self.assertIn("가용 데이터 기준 참고 결과", presentation)
            self.assertIn("거래비용 10bp", presentation)
            daily = pd.read_csv(paths["csv"])
            self.assertEqual(len(daily), MIN_OBSERVATIONS)
            self.assertIn("gross_drawdown", daily)

    def test_v3_oos_uses_v2_score_pit_cash_and_only_ten_bps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "fixture.db"
            _, probabilities = create_database(database)
            with closing(sqlite3.connect(database)) as connection:
                connection.execute(
                    "CREATE TABLE macro_observations("
                    "series TEXT,date TEXT,value REAL,available_at TEXT,source TEXT)"
                )
                connection.execute(
                    "INSERT INTO macro_observations VALUES (?,?,?,?,?)",
                    (
                        "CASH_3M",
                        "2019-12-31",
                        3.65,
                        "2019-12-31",
                        "ALFRED:DGS3MO:OUTPUT3",
                    ),
                )
                connection.commit()
            records = replay_records(
                probabilities,
                policy_version=RULE_STRESS_SCORE_V3_POLICY_VERSION,
                slow_risk_score=20,
            )
            scores = pd.DataFrame(
                {"risk_score": 20}, index=pd.DatetimeIndex(probabilities["date"])
            )
            before = sha256_file(database)

            with (
                patch("policy_oos.replay_policy_history", return_value=records),
                patch(
                    "router_score_v2.build_router_scores_from_database",
                    return_value=scores,
                ),
            ):
                report, frame = evaluate(database, use_router_score_v3=True)
            paths = write_outputs(report, frame, root / "v3")

            self.assertEqual(before, sha256_file(database))
            self.assertEqual(report["policy_version"], RULE_STRESS_SCORE_V3_POLICY_VERSION)
            self.assertEqual(
                report["score_model_version"], "nq-rule-stress-score-v2-w756-m252"
            )
            self.assertEqual(set(report["scenarios"]), {"gross", "cost_10bps"})
            self.assertEqual(report["design"]["transaction_cost_bps"], [0, 10])
            self.assertEqual(report["design"]["cash_return"]["source_series"], "DGS3MO")
            self.assertIn("macro_data.py", report["provenance"]["source_sha256"])
            self.assertIn("cash_contribution", frame)
            self.assertNotIn("net_25bps_return", frame)
            self.assertIn(
                "3개월 미국채 금리",
                paths["presentation"].read_text(encoding="utf-8"),
            )
            self.assertIn(
                "PIT CASH_3M",
                paths["markdown"].read_text(encoding="utf-8"),
            )

    def test_candidate_hmm_version_is_loaded_replayed_and_reported(self):
        candidate_version = "hmm3-sticky-nq-core-v1-diag-mc10-shadow-a1-test"
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "fixture.db"
            _, probabilities = create_database(database)
            probabilities = probabilities.assign(model_version=candidate_version)
            with closing(sqlite3.connect(database)) as connection:
                connection.execute(
                    "UPDATE regime_model_probabilities SET model_version=?",
                    (candidate_version,),
                )
                connection.commit()

            with patch(
                "policy_oos.replay_policy_history",
                return_value=replay_records(probabilities, unavailable=3),
            ) as replay:
                report, _ = evaluate(
                    database,
                    hmm_model_version=candidate_version,
                )

            self.assertEqual(replay.call_args.kwargs["model_version"], candidate_version)
            self.assertEqual(report["hmm_model_version"], candidate_version)
            self.assertIn(candidate_version, report["model_id"])

    def test_minimum_observations_is_frozen(self):
        prices, probabilities = sample_frames(MIN_OBSERVATIONS)
        decisions = decisions_frame(replay_records(probabilities))
        with self.assertRaisesRegex(ValueError, "at least 1000"):
            build_evaluation_frame(prices, probabilities, decisions)


if __name__ == "__main__":
    unittest.main()
