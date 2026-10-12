import json
import io
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from roro_q20 import (
    DRAWDOWN_THRESHOLD,
    HORIZON,
    INSUFFICIENT_EVIDENCE,
    MIN_COMPLETE_LABELS,
    MIN_INDEPENDENT_EVENTS,
    MODEL_VERSION as Q20_MODEL_VERSION,
    READINESS_METADATA_FIELDS,
)
from roro_policy import POLICY_VERSION
from roro_shadow import (
    BREADTH_REPAIR_SERIES_VERSION,
    MODEL_VERSIONS,
    POLICY_AUDIT_FIELDS,
    Q20_SERIES_VERSION,
    R5_SERIES_VERSION,
    SERIES_VERSION,
    persist_shadow_frame,
    read_shadow_history,
    model_versions_for_series,
)
from roro_price_observations import CONTRACT_VERSION as PRICE_OBSERVATION_MODEL_VERSION
from contextlib import closing, redirect_stdout

from roro_site_publisher import _nasdaq100_closes, build_and_publish, main, upload_benchmark_prices, upload_snapshots, web_snapshot


def row() -> dict:
    return {
        "market_risk_score": 44, "risk_regime": "YELLOW", "recommended_long": 80,
        "trajectory_1d": 1.0, "trajectory_5d": -3.0, "trajectory_20d": None,
        "macro_risk_score": None, "market_trend_score": 30.0,
        "portfolio_internal_score": 58.0, "confidence": 31,
        "data_completeness": 60.0, "data_status": "CONDITIONAL_EVENT_PIT",
        "trend_phase": "TRANSITION", "turn_evidence": [],
        "checklist_available_count": 4, "bottom_confirmation_count": 2,
        "top_confirmation_count": 2, "portfolio_correlation_20": 0.25,
        "portfolio_correlation_trajectory_5d": -0.02, "nq_beta_60": 1.3,
        "effective_positions": 8.5, "breadth_above_ma20": 0.7,
        "negative_participation_20": 0.3, "portfolio_drawdown_60": -0.04,
        "price_weight_coverage": 1.0, "history_weight_coverage_20d": 0.9,
        "history_weight_coverage_60d": 0.9, "data_quality": "WEIGHT_PROXY_ADJ_CLOSE",
        "risk_drivers": ["portfolio_internal"], "relief_drivers": ["market_trend"],
        "q20": {
            "model_version": Q20_MODEL_VERSION,
            "status": INSUFFICIENT_EVIDENCE,
            "probability": None,
            "fit_allowed": False,
            "horizon_trading_days": HORIZON,
            "drawdown_threshold": DRAWDOWN_THRESHOLD,
            "feature_rows": 1,
            "complete_feature_labels": 0,
            "independent_events": 0,
            "minimum_complete_labels": MIN_COMPLETE_LABELS,
            "minimum_independent_events": MIN_INDEPENDENT_EVENTS,
        },
        "policy": {
            "model_version": POLICY_VERSION,
            "score_based_long": 80,
            "raw_target_long": 80,
            "recommended_long": 80,
            "fast_status": "INCOMPLETE",
            "fast_trigger_count": 0,
            "fast_available_count": 6,
            "active_fast_conditions": [],
            "unavailable_fast_conditions": [
                "jpy_3d_drop",
                "portfolio_correlation",
            ],
            "phase": "COLD_START",
            "episode_status": "INACTIVE",
            "clear_streak": 0,
            "primary_driver": "RORO SCORE",
        },
    }


class FakeResponse:
    def __init__(self, payload): self.payload = payload
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def read(self): return json.dumps(self.payload).encode()


class RoroSitePublisherTest(unittest.TestCase):
    def test_repair_uses_actual_series_and_blocks_partial_upload_without_paid_key(self):
        with (
            patch("roro_site_publisher.load_api_key") as key,
            patch("roro_site_publisher.build_shadow_series", return_value={"as_of": "2026-01-08", "series_version": BREADTH_REPAIR_SERIES_VERSION}) as build,
            patch("roro_site_publisher.read_shadow_history", return_value=[{"date": "2026-01-08", "payload": {"macro_data_status": "FULL", "breadth_risk_percentile": None}}]) as history,
            patch("roro_site_publisher.web_snapshot", return_value={"as_of": "2026-01-08"}),
            patch("roro_site_publisher.upload_snapshots") as upload,
        ):
            options = dict(price_observation_database="prices.db", breadth_observation_database="breadth.db",
                           as_of="2026-01-08", latest_only=True, require_as_of=True,
                           repair_missing_breadth=True, no_narrative=True)
            with self.assertRaisesRegex(RuntimeError, "시장 참여 폭 입력 미완료"):
                build_and_publish("publish.db", "private.db", "shadow.db", config={"site_url": "https://example.test", "upload_token": "test"}, **options)
            upload.assert_not_called()
            history.return_value[0]["payload"]["breadth_risk_percentile"] = 50.0
            build_and_publish("publish.db", "private.db", "shadow.db", **options)
            self.assertEqual(history.call_args.kwargs["series_version"], BREADTH_REPAIR_SERIES_VERSION)
            self.assertTrue(build.call_args.kwargs["require_complete_breadth"])
            self.assertTrue(build.call_args.kwargs["repair_missing_breadth"])
            key.assert_not_called()

    @patch("roro_site_publisher.web_snapshot", side_effect=lambda row: {"as_of": row["date"]})
    @patch("roro_site_publisher.read_shadow_history")
    @patch("roro_site_publisher.build_shadow_series")
    @patch("roro_site_publisher.load_api_key", return_value=None)
    def test_latest_only_reads_effective_cutoff_not_requested_tail(
        self, _load_key, build, read_history, _snapshot
    ):
        build.return_value = {
            "as_of": "2026-01-08",
            "effective_as_of": "2026-01-08",
            "requested_as_of": "2026-01-09",
            "pending": True,
        }
        read_history.return_value = [{"date": "2026-01-08"}]

        result, snapshots, uploaded = build_and_publish(
            "publish.db",
            "private.db",
            "shadow.db",
            price_observation_database="prices.db",
            breadth_observation_database="breadth.db",
            as_of="2026-01-09",
            latest_only=True,
        )

        self.assertTrue(result["pending"])
        self.assertEqual(snapshots, [{"as_of": "2026-01-08"}])
        self.assertIsNone(uploaded)
        read_history.assert_called_once_with(
            "shadow.db",
            series_version=SERIES_VERSION,
            start="2026-01-08",
            end="2026-01-08",
        )
        with self.assertRaisesRegex(RuntimeError, "입력 미완료"):
            build_and_publish(
                "publish.db",
                "private.db",
                "shadow.db",
                price_observation_database="prices.db",
                breadth_observation_database="breadth.db",
                as_of="2026-01-09",
                latest_only=True,
                require_as_of=True,
            )

        build.return_value = {
            "as_of": "2026-01-09",
            "effective_as_of": "2026-01-09",
            "requested_as_of": "2026-01-09",
            "pending": False,
        }
        read_history.return_value = [
            {"date": "2026-01-09", "payload": {"macro_data_status": "NO_SCORE"}}
        ]
        with self.assertRaisesRegex(RuntimeError, "MACRO 입력 미완료"):
            build_and_publish(
                "publish.db",
                "private.db",
                "shadow.db",
                price_observation_database="prices.db",
                breadth_observation_database="breadth.db",
                as_of="2026-01-09",
                latest_only=True,
                require_as_of=True,
            )

    def test_web_snapshot_is_stable_and_aggregate_only(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "shadow.db"
            persist_shadow_frame(target, pd.DataFrame([row()], index=pd.to_datetime(["2026-01-07"])), as_of="2026-01-07")
            observation = read_shadow_history(target)[0]
            snapshot = web_snapshot(observation)
            self.assertEqual(
                set(snapshot),
                {
                    "schema_version",
                    "series_version",
                    "as_of",
                    "generated_at",
                    "input_cutoff",
                    "provenance_hash",
                    "model_versions",
                    "payload",
                },
            )
            self.assertEqual(snapshot["schema_version"], 4)
            self.assertEqual(snapshot["series_version"], SERIES_VERSION)
            self.assertEqual(snapshot["input_cutoff"], snapshot["as_of"])
            self.assertEqual(snapshot["model_versions"], model_versions_for_series(SERIES_VERSION))
            self.assertEqual(snapshot["model_versions"]["q20"], Q20_MODEL_VERSION)
            self.assertEqual(snapshot["model_versions"]["policy"], POLICY_VERSION)
            self.assertEqual(
                snapshot["model_versions"]["price_observations"],
                PRICE_OBSERVATION_MODEL_VERSION,
            )
            self.assertEqual(
                set(snapshot["payload"]["q20"]), set(READINESS_METADATA_FIELDS)
            )
            self.assertEqual(
                set(snapshot["payload"]["policy"]), set(POLICY_AUDIT_FIELDS)
            )
            self.assertEqual(
                snapshot["payload"]["recommended_long"],
                snapshot["payload"]["policy"]["recommended_long"],
            )
            self.assertNotIn("ticker", json.dumps(snapshot).lower())
            self.assertEqual(snapshot, web_snapshot(observation))
            observation["series_version"] = BREADTH_REPAIR_SERIES_VERSION
            self.assertEqual(web_snapshot(observation)["series_version"], BREADTH_REPAIR_SERIES_VERSION)

    @patch("roro_site_publisher.urllib.request.urlopen")
    def test_uploads_one_atomic_date_per_request_without_logging_tokens(self, urlopen):
        urlopen.side_effect = [FakeResponse({"rows": 1, "inserted": 1, "idempotent": 0}), FakeResponse({"rows": 1, "inserted": 1, "idempotent": 0})]
        snapshots = [
            {"series_version": SERIES_VERSION, "as_of": "2026-01-07"},
            {"series_version": SERIES_VERSION, "as_of": "2026-01-08"},
        ]
        result = upload_snapshots(snapshots, "https://example.test", "secret", "bypass")
        self.assertEqual(result["rows"], 2)
        self.assertEqual(urlopen.call_count, 2)
        for call in urlopen.call_args_list:
            request = call.args[0]
            self.assertEqual(request.full_url, "https://example.test/api/roro")
            self.assertEqual(request.headers["Authorization"], "Bearer secret")
            self.assertEqual(len(json.loads(request.data)), 1)

    @patch("roro_site_publisher.urllib.request.urlopen")
    def test_uploads_cash_index_closes_with_explicit_identity(self, urlopen):
        urlopen.return_value = FakeResponse({"rows": 2, "updated": 3})
        result = upload_benchmark_prices([
            {"as_of": "2026-01-07", "close": 21000.0, "previous_as_of": None, "previous_close": None},
            {"as_of": "2026-01-08", "close": 21100.0, "previous_as_of": "2026-01-07", "previous_close": 21000.0},
        ], "https://example.test", "secret")

        self.assertEqual(result, {"rows": 2, "updated": 3})
        body = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(body["benchmark_ticker"], "^NDX")
        self.assertEqual(body["benchmark_prices"][0]["close"], 21000.0)

    def test_cash_index_uses_raw_close_and_its_own_previous_session(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "prices.db"
            with closing(sqlite3.connect(database)) as connection:
                connection.executescript("""
                    CREATE TABLE daily_prices(ticker TEXT, date TEXT, close REAL, adj_close REAL);
                    INSERT INTO daily_prices VALUES
                    ('NQ=F','2026-09-14',30000,30001),
                    ('^NDX','2026-09-10',100,999), ('^NDX','2026-09-11',101,999),
                    ('^NDX','2026-09-14',102,999), ('SPY','2026-09-15',200,200),
                    ('^NDX','2026-09-16',103,999), ('^NDX','2026-09-17',900,999);
                """)
            prices = _nasdaq100_closes(database, ["2026-09-10", "2026-09-14", "2026-09-15", "2026-09-16"])
            self.assertEqual([p["as_of"] for p in prices], ["2026-09-10", "2026-09-14", "2026-09-16"])
            self.assertEqual(prices[1], {"as_of": "2026-09-14", "close": 102.0,
                                         "previous_as_of": "2026-09-11", "previous_close": 101.0})
            self.assertIsNone(prices[2]["previous_close"])

    @patch("roro_site_publisher.urllib.request.urlopen")
    def test_invalid_cash_prices_fail_before_any_upload(self, urlopen):
        price = {"as_of": "2026-09-14", "close": 100, "previous_as_of": "2026-09-11", "previous_close": 101}
        for invalid in ({**price, "close": True}, {**price, "close": float("nan")},
                        {**price, "previous_as_of": "2026-09-14"}, {**price, "as_of": "2026-02-30"}):
            with self.assertRaises(ValueError):
                upload_benchmark_prices([price, invalid], "https://example.test", "secret")
        urlopen.assert_not_called()

    def test_missing_web_field_fails_closed(self):
        observation = {
            "series_version": SERIES_VERSION, "date": "2026-01-07", "input_cutoff": "2026-01-07",
            "created_at": "2026-01-08T00:00:00Z", "provenance_hash": "a" * 64,
            "payload": {"market_risk_score": 1},
        }
        with self.assertRaisesRegex(ValueError, "missing RORO web fields"):
            web_snapshot(observation)

    def test_extended_q20_contract_fails_closed(self):
        payload = row()
        payload["q20"] = {**payload["q20"], "future_probability": 0.5}
        observation = {
            "series_version": SERIES_VERSION,
            "date": "2026-01-07",
            "input_cutoff": "2026-01-07",
            "created_at": "2026-01-08T00:00:00Z",
            "provenance_hash": "a" * 64,
            "payload": payload,
        }
        with self.assertRaisesRegex(ValueError, "exact frozen contract"):
            web_snapshot(observation)

    def test_extended_policy_contract_and_top_level_mismatch_fail_closed(self):
        payload = row()
        payload["policy"] = {**payload["policy"], "future_state": "LEAK"}
        observation = {
            "series_version": SERIES_VERSION,
            "date": "2026-01-07",
            "input_cutoff": "2026-01-07",
            "created_at": "2026-01-08T00:00:00Z",
            "provenance_hash": "a" * 64,
            "payload": payload,
        }
        with self.assertRaisesRegex(ValueError, "exact frozen contract"):
            web_snapshot(observation)
        payload = row()
        payload["recommended_long"] = 79
        observation["payload"] = payload
        with self.assertRaisesRegex(ValueError, "must equal"):
            web_snapshot(observation)

        payload = row()
        payload["policy"]["score_based_long"] = 25
        payload["policy"]["raw_target_long"] = 25
        payload["policy"]["recommended_long"] = 25
        payload["recommended_long"] = 25
        observation["payload"] = payload
        with self.assertRaisesRegex(ValueError, "market_risk_score"):
            web_snapshot(observation)

    def test_older_envelope_cannot_be_mislabeled_as_current_series(self):
        observation = {
            "series_version": Q20_SERIES_VERSION,
            "date": "2026-01-07",
            "input_cutoff": "2026-01-07",
            "created_at": "2026-01-08T00:00:00Z",
            "provenance_hash": "a" * 64,
            "payload": row(),
        }
        with self.assertRaisesRegex(ValueError, "current"):
            web_snapshot(observation)
        observation["series_version"] = R5_SERIES_VERSION
        with self.assertRaisesRegex(ValueError, "current"):
            web_snapshot(observation)

    @patch("roro_site_publisher.build_and_publish")
    @patch("roro_site_publisher.argparse.ArgumentParser.parse_args")
    def test_cli_skips_external_upload_without_separate_consent(self, parse_args, build):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text('{"site_url":"https://example.test","upload_token":"secret"}')
            parse_args.return_value = type("Args", (), {
                "publish_database": root / "publish.db", "private_database": root / "private.db",
                "shadow_database": root / "shadow.db",
                "price_observation_database": root / "price_observations.db", "config": config,
                "breadth_observation_database": root / "breadth_observations.db",
                "consent": root / "missing-consent.json", "as_of": "2026-01-07",
                "latest_only": True, "no_upload": False, "output": root / "payload.json",
            })()
            build.return_value = ({"rows_seen": 1}, [], None)
            output = io.StringIO()
            with redirect_stdout(output):
                main()
            self.assertIsNone(build.call_args.kwargs["config"])
            self.assertIn("명시 동의", output.getvalue())


if __name__ == "__main__":
    unittest.main()
