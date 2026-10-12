from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from roro_shadow import (
    BREADTH_REPAIR_SERIES_VERSION,
    ImmutableSeriesConflict,
    LEGACY_SERIES_VERSION,
    NO_PHASE_SERIES_VERSION,
    POLICY_AUDIT_FIELDS,
    PORTFOLIO_MODEL_VERSION,
    Q20_MODEL_VERSIONS,
    Q20_SERIES_VERSION,
    R6_SERIES_VERSION,
    R7_SERIES_VERSION,
    R10_SERIES_VERSION,
    R11_SERIES_VERSION,
    R12_SERIES_VERSION,
    R15_SERIES_VERSION,
    R5_MODEL_VERSIONS,
    R5_NO_PHASE_SERIES_VERSION,
    R5_SERIES_VERSION,
    SERIES_VERSION,
    TABLE,
    _build_roro_fast_snapshot,
    _build_policy_history,
    _canonical_json,
    _canonical_provenance_encoding,
    _apply_stored_score_trajectories,
    _latest_ready_cutoff,
    _provenance,
    build_shadow_series,
    latest_shadow,
    main,
    model_versions_for_series,
    persist_shadow_frame,
    read_shadow_history,
)
from roro_price_observations import CONTRACT_VERSION as PRICE_OBSERVATION_MODEL_VERSION
from roro_price_observations import SOURCE as PRICE_OBSERVATION_SOURCE
from roro_price_observations import persist_price_observations
from roro_q20 import INSUFFICIENT_EVIDENCE, MODEL_VERSION as Q20_MODEL_VERSION
from roro_q20 import READINESS_METADATA_FIELDS
from roro_policy import FAST_CONDITIONS, POLICY_VERSION, long_anchor


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _axis_frame(score: float = 40.0) -> pd.DataFrame:
    index = pd.DatetimeIndex(
        ["2026-01-06", "2026-01-07", "2026-01-08"], name="date"
    )
    return pd.DataFrame(
        {
            "market_risk_score": pd.Series(
                [np.nan, 40.0, score], index=index, dtype="Float64"
            ),
            "data_status": [
                "DATA_BLOCKED",
                "CONDITIONAL_EVENT_PIT",
                "CONDITIONAL_EVENT_PIT",
            ],
            "risk_regime": ["NO_SCORE", "YELLOW", "YELLOW"],
            "recommended_long": pd.Series(
                [pd.NA, 84, long_anchor(int(score))], index=index, dtype="Int64"
            ),
            "market_trend_score": [35.0, 38.0, 39.0],
            "portfolio_internal_score": [np.nan, 42.0, 43.0],
            "risk_drivers": [(), ("portfolio_internal",), ("portfolio_internal",)],
            "axis_completeness": [
                {"macro_risk": 0.0, "market_trend": 50.0, "portfolio_internal": 0.0},
                {"macro_risk": 40.0, "market_trend": 50.0, "portfolio_internal": 100.0},
                {"macro_risk": 40.0, "market_trend": 50.0, "portfolio_internal": 100.0},
            ],
            "trajectory_5d": [np.nan, -2.0, -3.0],
            "credit_contribution": [np.nan, -2.0, -1.5],
            "credit_score_trajectory_5d": [np.nan, -1.0, -0.5],
            "available_components": [(), ("credit",), ("credit",)],
            "component_risk_drivers": [(), (), ()],
            "component_relief_drivers": [(), ("credit",), ("credit",)],
            # Deliberately private-looking upstream fields must never be persisted.
            "ticker": ["SECRET_A", "SECRET_B", "SECRET_C"],
            "holding_name": ["Private One", "Private Two", "Private Three"],
        },
        index=index,
    )


def _q20() -> dict[str, object]:
    return {
        "model_version": Q20_MODEL_VERSION,
        "status": INSUFFICIENT_EVIDENCE,
        "probability": None,
        "fit_allowed": False,
        "horizon_trading_days": 20,
        "drawdown_threshold": -0.05,
        "feature_rows": 0,
        "complete_feature_labels": 0,
        "independent_events": 0,
        "minimum_complete_labels": 252,
        "minimum_independent_events": 20,
    }


def _fast(as_of: str, *, triggered: int = 0, unavailable: int = 0) -> dict[str, object]:
    conditions = []
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
                "label": key,
                "status": status,
                "detail": status,
                "data_quality": "UNAVAILABLE" if status == "UNAVAILABLE" else "EXACT",
                "as_of": as_of,
            }
        )
    complete = unavailable == 0
    return {
        "as_of": as_of,
        "status": "TRIGGERED" if triggered >= 3 else "CLEAR" if complete else "INCOMPLETE",
        "count": triggered,
        "trigger_count": triggered,
        "required": 3,
        "total": 8,
        "complete": complete,
        "primary_driver": "FAST TRIGGER" if triggered else "FAST CLEAR",
        "advisory": "test",
        "conditions": conditions,
    }


def _portfolio_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "portfolio_correlation_20": [0.30, 0.28, 0.25],
            "portfolio_correlation_trajectory_5d": [-0.01, -0.02, -0.03],
            "nq_beta_60": [1.10, 1.08, 1.05],
            "effective_positions": [9.0, 9.5, 10.0],
            "breadth_above_ma20": [0.55, 0.60, 0.65],
            "negative_participation_20": [0.45, 0.40, 0.35],
            "portfolio_drawdown_60": [-0.08, -0.07, -0.06],
            "price_weight_coverage": [0.96, 0.97, 0.98],
            "history_weight_coverage_20d": [0.90, 0.91, 0.92],
            "history_weight_coverage_60d": [0.86, 0.87, 0.88],
            "data_quality": ["QC", "QC", "QC"],
            "ticker": ["PRIVATE_A", "PRIVATE_B", "PRIVATE_C"],
        },
        index=index,
    )


def _phase_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "vix_curve_available": [False, True, True],
            "bottom_vix_curve": pd.Series(
                [pd.NA, True, True], index=index, dtype="boolean"
            ),
            "checklist_available_count": [3, 5, 6],
            "checklist_required_count": [2, 3, 4],
            "checklist_eligible": [False, True, True],
            "bottom_confirmation_count": [2, 4, 5],
            "bottom_checklist_pass": [False, True, True],
            "bottom_confirmed_evidence": [[], ["VIX_CURVE_5D_DOWN"], ["VIX_CURVE_5D_DOWN"]],
            "missing_evidence": [["VIX_CURVE"], [], []],
            "trend_phase": ["INSUFFICIENT_EVIDENCE", "TRANSITION", "UPTREND"],
            "turn_evidence": [[], [], ["RISK_SCORE_AT_MOST_33", "NQ_ABOVE_MA20"]],
        },
        index=index,
    )


class RoroShadowPersistenceTest(unittest.TestCase):
    def setUp(self) -> None:
        start = patch("roro_shadow.SHADOW_START", pd.Timestamp("2026-01-07"))
        start.start()
        self.addCleanup(start.stop)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.publish = self.root / "publish.db"
        self.private = self.root / "private.db"
        self.price_observations = self.root / "price_observations.db"
        self.target = self.root / "shadow" / "roro_shadow.db"
        for path, value in ((self.publish, "PUBLIC"), (self.private, "PRIVATE")):
            with closing(sqlite3.connect(path)) as connection:
                connection.execute("CREATE TABLE sentinel(value TEXT)")
                connection.execute("INSERT INTO sentinel VALUES(?)", (value,))
                connection.commit()
        persist_price_observations(
            self.price_observations,
            [
                {
                    "contract_version": PRICE_OBSERVATION_MODEL_VERSION,
                    "source": PRICE_OBSERVATION_SOURCE,
                    "ticker": ticker,
                    "raw_timestamp_utc": raw_timestamp,
                    "source_timezone": "America/New_York",
                    "observation_session_date": "2026-01-05",
                    "available_at": raw_timestamp,
                    "close": close,
                    "adj_close": close,
                    "fetched_at": "2026-01-06T00:00:00Z",
                }
                for ticker, raw_timestamp, close in (
                    ("JPY=X", "2026-01-05T22:00:00Z", 156.0),
                    ("GLD", "2026-01-05T21:00:00Z", 410.0),
                )
            ],
        )

    def test_latest_trajectories_use_immutable_published_scores(self) -> None:
        dates = pd.date_range("2026-01-01", periods=20)
        history = [
            {"date": day.date().isoformat(), "payload": {"market_risk_score": score}}
            for day, score in zip(dates, range(31, 51), strict=True)
        ]
        frame = pd.DataFrame(
            {
                "market_risk_score": [50.0],
                "trajectory_1d": [-1.0],
                "trajectory_5d": [-2.0],
                "trajectory_20d": [-3.0],
            },
            index=pd.DatetimeIndex(["2026-01-21"]),
        )

        corrected = _apply_stored_score_trajectories(frame, history)

        self.assertEqual(corrected.iloc[0]["trajectory_1d"], 0.0)
        self.assertEqual(corrected.iloc[0]["trajectory_5d"], 4.0)
        self.assertEqual(corrected.iloc[0]["trajectory_20d"], 19.0)

    def _build(
        self,
        frame: pd.DataFrame,
        *,
        no_phase: bool = False,
        latest_only: bool = False,
        portfolio: pd.DataFrame | None = None,
        as_of: str = "2026-01-08",
        narrative_api_key: str | None = None,
        **options,
    ) -> dict[str, object]:
        placeholder = pd.DataFrame(
            {"nq": np.linspace(100.0, 102.0, len(frame))}, index=frame.index
        )
        portfolio = _portfolio_frame(frame.index) if portfolio is None else portfolio
        with (
            patch("roro_shadow.load_roro_panel", return_value=placeholder) as load_panel,
            patch("roro_shadow.build_roro_scores", return_value=placeholder),
            patch(
                "roro_shadow.load_portfolio_internal_series",
                return_value=portfolio,
            ),
            patch("roro_shadow.build_roro_axis_scores", return_value=frame),
            patch(
                "roro_shadow.build_trend_phase",
                side_effect=lambda axis, *_: _phase_frame(frame.index).loc[axis.index],
            ),
        ):
            result = build_shadow_series(
                self.publish,
                self.private,
                self.target,
                price_observation_database=self.price_observations,
                as_of=as_of,
                no_phase=no_phase,
                latest_only=latest_only,
                narrative_api_key=narrative_api_key,
                **options,
            )
        load_panel.assert_called_once_with(
            self.publish,
            as_of=as_of,
            price_observation_database=self.price_observations,
            **({"breadth_observation_database": options["breadth_observation_database"]}
               if options.get("breadth_observation_database") is not None else {}),
        )
        return result

    def test_sources_remain_unchanged_and_target_is_separate(self) -> None:
        before = (_digest(self.publish), _digest(self.private))
        result = self._build(_axis_frame())

        self.assertEqual((_digest(self.publish), _digest(self.private)), before)
        self.assertTrue(self.target.is_file())
        self.assertNotEqual(self.target.resolve(), self.publish.resolve())
        self.assertNotEqual(self.target.resolve(), self.private.resolve())
        self.assertEqual(result["inserted"], 2)

    def test_every_r5_row_has_causal_q20_and_policy_audits(self) -> None:
        self._build(_axis_frame())
        history = read_shadow_history(self.target)

        self.assertEqual(len(history), 2)
        for observation in history:
            q20 = observation["payload"]["q20"]
            self.assertEqual(set(q20), set(READINESS_METADATA_FIELDS))
            self.assertEqual(q20["model_version"], Q20_MODEL_VERSION)
            self.assertEqual(q20["status"], INSUFFICIENT_EVIDENCE)
            self.assertIsNone(q20["probability"])
            self.assertFalse(q20["fit_allowed"])
            policy = observation["payload"]["policy"]
            self.assertEqual(set(policy), set(POLICY_AUDIT_FIELDS))
            self.assertEqual(policy["model_version"], POLICY_VERSION)
            self.assertEqual(
                observation["payload"]["recommended_long"],
                policy["recommended_long"],
            )

    def test_identity_fields_are_filtered_and_schema_is_aggregate_only(self) -> None:
        self._build(_axis_frame())
        with closing(sqlite3.connect(self.target)) as connection:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            columns = {
                row[1] for row in connection.execute(f"PRAGMA table_info({TABLE})")
            }
            payloads = [
                row[0]
                for row in connection.execute(
                    f"SELECT payload_json FROM {TABLE} ORDER BY date"
                )
            ]

        self.assertEqual(tables, {TABLE})
        self.assertFalse(columns & {"ticker", "security", "holding_name", "quantity"})
        self.assertNotIn("SECRET", "".join(payloads))
        self.assertNotIn("holding_name", "".join(payloads))
        latest = json.loads(payloads[-1])
        self.assertIn("market_risk_score", latest)
        self.assertIn("credit_score_trajectory_5d", latest)
        self.assertIn("component_relief_drivers", latest)
        self.assertEqual(latest["trend_phase"], "UPTREND")
        self.assertEqual(latest["checklist_available_count"], 6)
        self.assertEqual(latest["portfolio_correlation_20"], 0.25)
        self.assertEqual(latest["history_weight_coverage_20d"], 0.92)
        self.assertEqual(latest["history_weight_coverage_60d"], 0.88)
        self.assertNotIn("ticker", latest)
        self.assertIn("q20", latest)
        self.assertIn("policy", latest)

    def test_identical_is_idempotent_and_changed_key_conflicts(self) -> None:
        first = self._build(_axis_frame())
        created_at = latest_shadow(self.target)["created_at"]
        second = self._build(_axis_frame())

        self.assertEqual(first["inserted"], 2)
        self.assertEqual(second["inserted"], 0)
        self.assertEqual(second["idempotent"], 2)
        self.assertEqual(latest_shadow(self.target)["created_at"], created_at)
        with self.assertRaisesRegex(ImmutableSeriesConflict, "2026-01-08"):
            self._build(_axis_frame(score=41.0))
        self.assertEqual(latest_shadow(self.target)["payload"]["market_risk_score"], 40.0)

    def test_rows_are_write_protected_and_remain_readable(self) -> None:
        self._build(_axis_frame())
        with closing(sqlite3.connect(self.target)) as connection:
            triggers = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='trigger'"
                )
            }
            self.assertEqual(
                triggers, {f"{TABLE}_no_update", f"{TABLE}_no_delete"}
            )
            for statement in (
                f"UPDATE {TABLE} SET input_cutoff='2099-01-01'",
                f"DELETE FROM {TABLE} WHERE date='2026-01-08'",
            ):
                with self.subTest(statement=statement):
                    with self.assertRaisesRegex(
                        sqlite3.DatabaseError, "immutable RORO SHADOW series"
                    ):
                        connection.execute(statement)
                    connection.rollback()

        self.assertEqual(len(read_shadow_history(self.target)), 2)

    def test_reader_rejects_tampered_payload_and_metadata(self) -> None:
        self._build(_axis_frame())
        with closing(sqlite3.connect(self.target)) as connection:
            connection.execute(f"DROP TRIGGER {TABLE}_no_update")
            connection.execute(
                f"UPDATE {TABLE} SET payload_json=? WHERE series_version=? AND date=?",
                ('{"market_risk_score":0}', SERIES_VERSION, "2026-01-08"),
            )
            connection.commit()
        with self.assertRaisesRegex(ValueError, "provenance|canonical"):
            latest_shadow(self.target)

        self.target.unlink()
        self._build(_axis_frame())
        with closing(sqlite3.connect(self.target)) as connection:
            connection.execute(f"DROP TRIGGER {TABLE}_no_update")
            connection.execute(
                f"UPDATE {TABLE} SET input_cutoff=? WHERE series_version=? AND date=?",
                ("2026-01-07", SERIES_VERSION, "2026-01-08"),
            )
            connection.commit()
        with self.assertRaisesRegex(ValueError, "metadata mismatch"):
            read_shadow_history(self.target)

    def test_r6_cross_language_provenance_vector_and_legacy_branch(self) -> None:
        vector = {
            "int": 1,
            "float": 1.0,
            "negative_zero": -0.0,
            "korean": "리스크",
            "nested": [True, None, [2.5, "끝"]],
        }
        expected = (
            "o5{s5:666c6f6174d3ff0000000000000s3:696e74d3ff0000000000000"
            "s6:6b6f7265616es9:eba6acec8aa4ed81acs13:6e656761746976655f7a65726f"
            "d8000000000000000s6:6e6573746564a3[b1na2[d4004000000000000s3:eb819d]]}"
        )
        self.assertEqual(_canonical_provenance_encoding(vector), expected)
        self.assertEqual(_canonical_provenance_encoding(1), _canonical_provenance_encoding(1.0))
        self.assertEqual(_canonical_provenance_encoding(-0.0), "d8000000000000000")
        self.assertEqual(
            hashlib.sha256(expected.encode("utf-8")).hexdigest(),
            "b64e38bcfa88f0e3d6df4f7a6aa48eff0f84d1ff8c59a2f3906b15f37e3cf51b",
        )

        payload_json = _canonical_json({"value": 1})
        legacy_material = {
            "series_version": R5_SERIES_VERSION,
            "date": "2026-01-08",
            "input_cutoff": "2026-01-08",
            "model_versions": model_versions_for_series(R5_SERIES_VERSION),
            "payload": {"value": 1},
        }
        self.assertEqual(
            _provenance("2026-01-08", payload_json, R5_SERIES_VERSION),
            hashlib.sha256(_canonical_json(legacy_material).encode("utf-8")).hexdigest(),
        )

    def test_reader_rejects_invalid_created_at_timestamp(self) -> None:
        self._build(_axis_frame())
        invalid_values = (
            "bad",
            "2026-01-08",
            "2026-02-30T00:00:00Z",
            "2026-01-08T00:00:00+00:00",
        )
        with closing(sqlite3.connect(self.target)) as connection:
            connection.execute(f"DROP TRIGGER {TABLE}_no_update")
            original = connection.execute(
                f"SELECT created_at FROM {TABLE} WHERE series_version=? AND date=?",
                (SERIES_VERSION, "2026-01-08"),
            ).fetchone()[0]
            for invalid in invalid_values:
                with self.subTest(created_at=invalid):
                    connection.execute(
                        f"UPDATE {TABLE} SET created_at=? WHERE series_version=? AND date=?",
                        (invalid, SERIES_VERSION, "2026-01-08"),
                    )
                    connection.commit()
                    with self.assertRaisesRegex(ValueError, "created_at.*UTC ISO"):
                        latest_shadow(self.target)
                    connection.execute(
                        f"UPDATE {TABLE} SET created_at=? WHERE series_version=? AND date=?",
                        (original, SERIES_VERSION, "2026-01-08"),
                    )
                    connection.commit()

    def test_latest_and_bounded_history(self) -> None:
        self._build(_axis_frame())
        history = read_shadow_history(
            self.target, start="2026-01-08", end="2026-01-08"
        )

        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["series_version"], SERIES_VERSION)
        self.assertEqual(history[0]["date"], "2026-01-08")
        self.assertEqual(history[0]["input_cutoff"], "2026-01-08")
        self.assertEqual(history[0]["portfolio_model_version"], PORTFOLIO_MODEL_VERSION)
        self.assertEqual(latest_shadow(self.target)["date"], "2026-01-08")
        self.assertEqual(
            len(latest_shadow(self.target)["provenance_hash"]),
            64,
        )

    def test_r3_rows_remain_unchanged_and_explicitly_readable(self) -> None:
        persist_shadow_frame(
            self.target,
            _axis_frame(),
            as_of="2026-01-08",
            series_version=LEGACY_SERIES_VERSION,
        )
        before = read_shadow_history(
            self.target, series_version=LEGACY_SERIES_VERSION
        )

        self._build(_axis_frame())

        after = read_shadow_history(
            self.target, series_version=LEGACY_SERIES_VERSION
        )
        self.assertEqual(after, before)
        self.assertEqual(len(read_shadow_history(self.target)), 2)

    def test_r4_rows_remain_unchanged_and_exclude_r5_policy(self) -> None:
        frame = _axis_frame().assign(q20=[_q20(), _q20(), _q20()])
        persist_shadow_frame(
            self.target,
            frame,
            as_of="2026-01-08",
            series_version=Q20_SERIES_VERSION,
        )
        before = read_shadow_history(
            self.target, series_version=Q20_SERIES_VERSION
        )

        self._build(_axis_frame())

        after = read_shadow_history(
            self.target, series_version=Q20_SERIES_VERSION
        )
        self.assertEqual(after, before)
        self.assertEqual(
            Q20_MODEL_VERSIONS,
            model_versions_for_series(Q20_SERIES_VERSION),
        )
        self.assertTrue(all("policy" not in row["payload"] for row in after))

    def test_r5_rows_remain_unchanged_and_explicitly_readable(self) -> None:
        frame = _axis_frame().assign(
            q20=[_q20(), _q20(), _q20()],
            policy=[
                {
                    "model_version": POLICY_VERSION,
                    "score_based_long": 84,
                    "raw_target_long": 84,
                    "recommended_long": 84,
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
                }
            ]
            * 3,
        )
        persist_shadow_frame(
            self.target,
            frame,
            as_of="2026-01-08",
            series_version=R5_SERIES_VERSION,
        )
        before = read_shadow_history(
            self.target, series_version=R5_SERIES_VERSION
        )

        self._build(_axis_frame())

        after = read_shadow_history(
            self.target, series_version=R5_SERIES_VERSION
        )
        self.assertEqual(after, before)
        self.assertEqual(
            R5_MODEL_VERSIONS,
            model_versions_for_series(R5_SERIES_VERSION),
        )
        self.assertEqual(
            R5_MODEL_VERSIONS,
            model_versions_for_series(R5_NO_PHASE_SERIES_VERSION),
        )
        self.assertNotIn("price_observations", R5_MODEL_VERSIONS)
        self.assertTrue(R6_SERIES_VERSION.endswith("shadow-r6-q20-policy-fxgold"))
        self.assertTrue(R7_SERIES_VERSION.endswith("shadow-r7-credit-move"))
        self.assertTrue(R10_SERIES_VERSION.endswith("shadow-r10-ndth-breadth"))
        self.assertTrue(R11_SERIES_VERSION.endswith("shadow-r11-ndth-move-reviewed"))
        self.assertTrue(R12_SERIES_VERSION.endswith("shadow-r12-reviewed-move-priority"))
        self.assertTrue(R15_SERIES_VERSION.endswith("shadow-r15-yahoo-usdjpy-fast"))
        self.assertTrue(SERIES_VERSION.endswith("shadow-r21-breadth-correction"))

    def test_unknown_series_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown RORO SHADOW series"):
            persist_shadow_frame(
                self.target,
                _axis_frame(),
                as_of="2026-01-08",
                series_version="arbitrary-series",
            )
        with self.assertRaisesRegex(ValueError, "unknown RORO SHADOW series"):
            read_shadow_history(self.target, series_version="arbitrary-series")
        with self.assertRaisesRegex(ValueError, "unknown RORO SHADOW series"):
            latest_shadow(self.target, series_version="arbitrary-series")

    def test_target_cannot_alias_a_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "separate"):
            build_shadow_series(
                self.publish,
                self.private,
                self.publish,
                price_observation_database=self.price_observations,
                as_of="2026-01-08",
                no_phase=True,
            )

    def test_r6_requires_a_nonempty_price_observation_database(self) -> None:
        with self.assertRaises(FileNotFoundError):
            build_shadow_series(
                self.publish,
                self.private,
                self.target,
                price_observation_database=self.root / "missing.db",
                as_of="2026-01-08",
            )
        empty = self.root / "empty_observations.db"
        persist_price_observations(empty, [])
        with self.assertRaisesRegex(ValueError, "no frozen-contract observations"):
            build_shadow_series(
                self.publish,
                self.private,
                self.target,
                price_observation_database=empty,
                as_of="2026-01-08",
            )

    def test_no_phase_uses_a_separate_diagnostic_series(self) -> None:
        result = self._build(_axis_frame(), no_phase=True)
        self.assertEqual(result["series_version"], NO_PHASE_SERIES_VERSION)
        self.assertFalse(result["phase_included"])
        latest = latest_shadow(self.target, series_version=NO_PHASE_SERIES_VERSION)
        self.assertNotIn("trend_phase", latest["payload"])
        self.assertIn("portfolio_correlation_20", latest["payload"])

    def test_latest_only_persists_last_row_and_reuses_immutable_date(self) -> None:
        first = self._build(_axis_frame(), latest_only=True)
        self.assertEqual(first["rows_seen"], 1)
        self.assertEqual(first["inserted"], 1)
        self.assertTrue(first["latest_only"])
        history = read_shadow_history(self.target)
        self.assertEqual([row["date"] for row in history], ["2026-01-08"])
        second = self._build(_axis_frame(score=41.0), latest_only=True)
        self.assertTrue(second["reused_existing"])
        self.assertEqual(latest_shadow(self.target)["payload"]["market_risk_score"], 40.0)

    def test_latest_only_uses_last_model_ready_date_when_tail_is_incomplete(self) -> None:
        frame = _axis_frame()
        frame.loc[pd.Timestamp("2026-01-08"), "market_risk_score"] = pd.NA
        frame.loc[pd.Timestamp("2026-01-08"), "recommended_long"] = pd.NA
        frame.loc[pd.Timestamp("2026-01-08"), "portfolio_internal_score"] = np.nan

        result = self._build(frame, latest_only=True)

        self.assertEqual(result["requested_as_of"], "2026-01-08")
        self.assertEqual(result["effective_as_of"], "2026-01-07")
        self.assertEqual(result["as_of"], "2026-01-07")
        self.assertTrue(result["pending"])
        self.assertTrue(result["warnings"])
        self.assertEqual([row["date"] for row in read_shadow_history(self.target)], ["2026-01-07"])

    def test_breadth_gate_fails_before_persistence_or_narrative(self) -> None:
        options = dict(latest_only=True, require_complete_breadth=True,
                       breadth_observation_database=self.price_observations)
        with (
            patch("roro_shadow._validate_breadth_observation_database", return_value=self.price_observations),
            patch("roro_shadow.breadth_readiness", return_value={"status": "DELAYED", "message": "시장 참여 폭 수집 지연"}) as readiness,
            patch("roro_shadow.persist_shadow_frame") as persist,
            patch("roro_narrative.generate_summary") as narrative,
        ):
            with self.assertRaisesRegex(RuntimeError, "수집 지연"):
                self._build(_axis_frame(), **options)
            readiness.return_value = {"status": "READY"}
            with self.assertRaisesRegex(RuntimeError, "점수 미완료"):
                self._build(_axis_frame(), **options)
            frame = _axis_frame()
            frame["breadth_risk_percentile"] = 50.0
            frame.loc[frame.index[-1], "portfolio_internal_score"] = np.nan
            with self.assertRaisesRegex(RuntimeError, "생성 가능일 2026-01-07"):
                self._build(frame, **options)
            persist.assert_not_called()
            narrative.assert_not_called()
        self.assertFalse(self.target.exists())

    def test_late_breadth_creates_separate_repair_once_and_preserves_original(self) -> None:
        self._build(_axis_frame(), latest_only=True)
        original = read_shadow_history(self.target)
        frame = _axis_frame(score=45.0)
        frame["breadth_risk_percentile"] = 50.0
        frame["market_data_status"] = "FULL"
        frame["available_components"] = [("credit", "market_breadth")] * len(frame)
        with (
            patch("roro_shadow._validate_breadth_observation_database", return_value=self.price_observations),
            patch("roro_shadow.breadth_readiness", return_value={"status": "READY"}),
            patch("roro_narrative.generate_summary", return_value="corrected") as narrative,
        ):
            options = dict(latest_only=True, require_complete_breadth=True,
                           breadth_observation_database=self.price_observations)
            with self.assertRaisesRegex(RuntimeError, "repair-missing-breadth"):
                self._build(frame, **options)
            first = self._build(frame, repair_missing_breadth=True, **options)
            second = self._build(frame, repair_missing_breadth=True, **options)
            narrative.assert_called_once()
        self.assertEqual(first["series_version"], BREADTH_REPAIR_SERIES_VERSION)
        self.assertEqual(first["inserted"], 1)
        self.assertTrue(second["reused_existing"])
        self.assertEqual(second["series_version"], BREADTH_REPAIR_SERIES_VERSION)
        self.assertEqual(original, read_shadow_history(self.target))
        corrected = read_shadow_history(self.target, series_version=BREADTH_REPAIR_SERIES_VERSION)
        self.assertEqual(corrected[0]["payload"]["breadth_risk_percentile"], 50.0)
        self.assertEqual(corrected[0]["payload"]["market_risk_score"], 45.0)

    def test_ready_cutoff_requires_exact_market_and_portfolio_axes(self) -> None:
        frame = _axis_frame()
        frame.loc[pd.Timestamp("2026-01-08"), "portfolio_internal_score"] = np.nan
        self.assertEqual(_latest_ready_cutoff(frame, "2026-01-08"), "2026-01-07")
        frame.loc[pd.Timestamp("2026-01-08"), "portfolio_internal_score"] = 43.0
        frame.loc[pd.Timestamp("2026-01-08"), "market_trend_score"] = np.nan
        self.assertEqual(_latest_ready_cutoff(frame, "2026-01-08"), "2026-01-07")

    def test_ready_cutoff_stops_before_an_interior_gap(self) -> None:
        index = pd.DatetimeIndex(
            ["2026-01-07", "2026-01-08", "2026-01-09"], name="date"
        )
        frame = pd.DataFrame(
            {
                "market_risk_score": [40.0, np.nan, 50.0],
                "market_trend_score": [35.0, 36.0, 37.0],
                "portfolio_internal_score": [42.0, 43.0, 44.0],
            },
            index=index,
        )
        self.assertEqual(_latest_ready_cutoff(frame, "2026-01-09"), "2026-01-07")

    def test_latest_only_reuses_existing_row_without_regenerating_narrative(self) -> None:
        with patch("roro_narrative.generate_summary", return_value="first summary") as generate:
            first = self._build(
                _axis_frame(), latest_only=True, narrative_api_key="test-key"
            )
            second = self._build(
                _axis_frame(), latest_only=True, narrative_api_key="test-key"
            )

        self.assertFalse(first["reused_existing"])
        self.assertTrue(second["reused_existing"])
        self.assertEqual(second["idempotent"], 1)
        generate.assert_called_once()
        self.assertEqual(
            latest_shadow(self.target)["payload"]["policy"]["primary_driver"],
            "first summary",
        )

    def test_portfolio_aggregate_join_is_exact_date_without_fill(self) -> None:
        frame = _axis_frame()
        portfolio = _portfolio_frame(frame.index).drop(frame.index[-1])
        self._build(frame, portfolio=portfolio)
        payload = latest_shadow(self.target)["payload"]
        self.assertIsNone(payload["portfolio_correlation_20"])
        self.assertIsNone(payload["history_weight_coverage_20d"])
        self.assertIsNone(payload["data_quality"])

    def test_policy_history_is_prefix_causal_when_future_dates_append(self) -> None:
        dates = pd.date_range("2026-01-07", periods=5, freq="B", name="date")

        def frames(scores: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
            index = dates[: len(scores)]
            panel = pd.DataFrame({"nq": np.arange(len(index)) + 100.0}, index=index)
            payload = pd.DataFrame(
                {
                    "market_risk_score": pd.Series(scores, index=index, dtype="Int64"),
                    "recommended_long": pd.Series(
                        [long_anchor(score) for score in scores], index=index, dtype="Int64"
                    ),
                },
                index=index,
            )
            return panel, payload

        prefix_panel, prefix_payload = frames([66, 60, 58])
        full_panel, full_payload = frames([66, 60, 58, 90, 20])
        with patch(
            "roro_shadow.build_fast_signal_snapshot",
            side_effect=lambda _, day: _fast(day),
        ):
            prefix = _build_policy_history(
                self.publish, prefix_panel, prefix_payload, as_of="2026-01-09"
            )
            full = _build_policy_history(
                self.publish, full_panel, full_payload, as_of="2026-01-13"
            )
        self.assertEqual(
            prefix.loc[:, ["recommended_long", "policy"]].to_dict("records"),
            full.iloc[: len(prefix)].loc[:, ["recommended_long", "policy"]].to_dict("records"),
        )

    def test_missing_fast_blocks_hold_increase_but_allow_risk_cut(self) -> None:
        index = pd.DatetimeIndex(["2026-01-07", "2026-01-08", "2026-01-09"], name="date")
        panel = pd.DataFrame({"nq": [100.0, 101.0, 99.0]}, index=index)
        scores = [66, 33, 100]
        payload = pd.DataFrame(
            {
                "market_risk_score": pd.Series(scores, index=index, dtype="Int64"),
                "recommended_long": pd.Series(
                    [long_anchor(score) for score in scores], index=index, dtype="Int64"
                ),
            },
            index=index,
        )
        with patch(
            "roro_shadow.build_fast_signal_snapshot",
            side_effect=lambda _, day: _fast(day, unavailable=4),
        ):
            result = _build_policy_history(
                self.publish, panel, payload, as_of="2026-01-09"
            )
        self.assertEqual(result["recommended_long"].tolist(), [60, 60, 25])
        self.assertEqual(result.iloc[1]["policy"]["phase"], "RECOVERY_HOLD")
        self.assertEqual(result.iloc[1]["policy"]["fast_available_count"], 2)
        self.assertEqual(result.iloc[2]["policy"]["phase"], "RISK_REDUCTION")

    def test_non_auditable_fast_conditions_are_forced_unavailable(self) -> None:
        with patch(
            "roro_shadow.build_fast_signal_snapshot",
            return_value=_fast("2026-01-07"),
        ):
            snapshot = _build_roro_fast_snapshot(self.publish, "2026-01-07")
        by_key = {item["key"]: item for item in snapshot["conditions"]}
        for key in ("jpy_3d_drop", "portfolio_correlation"):
            self.assertEqual(by_key[key]["status"], "UNAVAILABLE")
            self.assertEqual(by_key[key]["data_quality"], "UNAVAILABLE")
        self.assertEqual(snapshot["status"], "INCOMPLETE")
        self.assertFalse(snapshot["complete"])

    def test_policy_calendar_gap_duplicate_and_non_midnight_fail_closed(self) -> None:
        index = pd.DatetimeIndex(["2026-01-07", "2026-01-08", "2026-01-09"], name="date")
        panel = pd.DataFrame({"nq": [100.0, 101.0, 102.0]}, index=index)
        payload = pd.DataFrame(
            {"market_risk_score": [66, 60], "recommended_long": [60, 65]},
            index=index[[0, 2]],
        )
        with self.assertRaisesRegex(ValueError, "every NQ decision date"):
            _build_policy_history(self.publish, panel, payload, as_of="2026-01-09")
        duplicate = pd.concat([panel, panel.iloc[[-1]]])
        with self.assertRaisesRegex(ValueError, "unique and increasing"):
            _build_policy_history(self.publish, duplicate, payload, as_of="2026-01-09")
        intraday = panel.copy()
        intraday.index = pd.DatetimeIndex(
            ["2026-01-07", "2026-01-08 12:00", "2026-01-09"], name="date"
        )
        with self.assertRaisesRegex(ValueError, "exact calendar dates"):
            _build_policy_history(self.publish, intraday, payload, as_of="2026-01-09")

    def test_build_cli_defaults_to_phase_and_supports_diagnostics(self) -> None:
        expected = {
            "series_version": SERIES_VERSION,
            "as_of": "2026-01-08",
            "rows_seen": 2,
            "inserted": 2,
            "idempotent": 0,
        }
        output = io.StringIO()
        with (
            patch("roro_shadow.build_shadow_series", return_value=expected) as build,
            redirect_stdout(output),
        ):
            main(
                [
                    "build",
                    str(self.publish),
                    str(self.private),
                    str(self.target),
                    "--price-observation-database",
                    str(self.price_observations),
                    "--as-of",
                    "2026-01-08",
                ]
            )

        build.assert_called_once_with(
            str(self.publish),
            str(self.private),
            str(self.target),
            price_observation_database=str(self.price_observations),
            as_of="2026-01-08",
            no_phase=False,
            latest_only=False,
        )
        self.assertEqual(json.loads(output.getvalue()), expected)

        with (
            patch("roro_shadow.build_shadow_series", return_value=expected) as build,
            redirect_stdout(io.StringIO()),
        ):
            main(
                [
                    "build",
                    str(self.publish),
                    str(self.private),
                    str(self.target),
                    "--price-observation-database",
                    str(self.price_observations),
                    "--as-of",
                    "2026-01-08",
                    "--no-phase",
                    "--latest-only",
                ]
            )
        build.assert_called_once_with(
            str(self.publish),
            str(self.private),
            str(self.target),
            price_observation_database=str(self.price_observations),
            as_of="2026-01-08",
            no_phase=True,
            latest_only=True,
        )


if __name__ == "__main__":
    unittest.main()
