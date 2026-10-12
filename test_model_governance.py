import tempfile
import unittest
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

from model_governance import (
    manual_activate_model,
    new_registry,
    promote_model,
    read_registry,
    record_shadow_day,
    register_validation,
    rollback_model,
    seed_operational_models,
    start_shadow,
    write_registry,
)


SOURCE_HASH = "b" * 64
DB_HASH = "d" * 64


def report(
    model_version: str,
    role: str,
    generated_at: datetime,
    *,
    passed: bool = True,
    data_cutoff: str = "2026-08-07",
) -> dict[str, object]:
    return {
        "role": role,
        "model_version": model_version,
        "status": "HISTORICAL_PASS" if passed else "HISTORICAL_FAIL",
        "design": {"features": ["a", "b"], "target": "frozen"},
        "gates": {"auc": passed, "calibration": passed},
        "provenance": {
            "generated_at_utc": generated_at.isoformat(),
            "data_cutoff": data_cutoff,
            "database": {"name": "test.db", "sha256": DB_HASH},
            "source_sha256": {"MODEL_SPEC_v1.md": SOURCE_HASH},
        },
    }


class ModelGovernanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.t0 = datetime(2026, 8, 10, 0, 0, tzinfo=timezone.utc)
        self.sources = {"MODEL_SPEC_v1.md": SOURCE_HASH}
        self.registry = seed_operational_models(
            new_registry(),
            "hmm-live-v1",
            ["hmm-rollback-v2", "hmm-rollback-v1"],
            source_fingerprints=self.sources,
            now=self.t0,
        )

    def _start_shadow(self, role: str = "Q20", model_id: str = "q20-v1") -> dict:
        original = report(model_id, role, self.t0)
        registered = register_validation(
            self.registry,
            original,
            validation_sha256="a" * 64,
            source_fingerprints=self.sources,
            now=self.t0,
        )
        clean_time = self.t0 + timedelta(hours=25)
        clean = report(model_id, role, clean_time, data_cutoff="2026-08-08")
        return start_shadow(
            registered,
            model_id,
            clean,
            validation_sha256="c" * 64,
            source_fingerprints=self.sources,
            now=clean_time,
        )

    def _ready_shadow(self, role: str = "Q20", model_id: str = "q20-v1") -> dict:
        registry = self._start_shadow(role, model_id)
        cursor = date(2026, 8, 11)
        recorded = 0
        while recorded < 126:
            if cursor.weekday() < 5:
                event = "DE_RISK" if recorded == 0 else "REENTRY" if recorded == 125 else "NONE"
                registry = record_shadow_day(
                    registry,
                    model_id,
                    cursor,
                    event=event,
                    passed=True,
                    evidence_sha256=f"{recorded:064x}",
                    evidence_as_of=cursor,
                    now=datetime.combine(cursor, time(23), tzinfo=timezone.utc),
                )
                recorded += 1
            cursor += timedelta(days=1)
        return registry

    def test_schema_v2_seed_and_v1_migration_are_role_isolated(self) -> None:
        self.assertEqual(self.registry["schema_version"], 2)
        current = self.registry["roles"]["CURRENT_REGIME"]
        self.assertEqual(current["active_model_id"], "hmm-live-v1")
        self.assertEqual(len(current["rollback_model_ids"]), 2)
        self.assertIsNone(self.registry["roles"]["Q20"]["active_model_id"])

        legacy = {
            "schema_version": 1,
            "active_model_id": "hmm-v1",
            "rollback_model_ids": ["q20-v0", "hmm-v0"],
            "models": {
                "hmm-v1": {"design": {"model": "HMM"}},
                "q20-v0": {"design": {"target": "liquidation"}},
                "hmm-v0": {"design": {"model": "HMM"}},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            write_registry(path, legacy)
            migrated = read_registry(path)
        self.assertEqual(migrated["schema_version"], 2)
        self.assertEqual(migrated["roles"]["CURRENT_REGIME"]["active_model_id"], "hmm-v1")
        self.assertEqual(migrated["roles"]["Q20"]["rollback_model_ids"], ["q20-v0"])

    def test_registration_requires_role_and_provenance_and_supports_safe_supersede(self) -> None:
        missing_role = report("bad-role-v1", "Q20", self.t0)
        del missing_role["role"]
        with self.assertRaisesRegex(ValueError, "role must be"):
            register_validation(
                self.registry,
                missing_role,
                validation_sha256="9" * 64,
                source_fingerprints=self.sources,
                now=self.t0,
            )

        missing = report("bad-v1", "Q20", self.t0)
        del missing["provenance"]["database"]
        with self.assertRaisesRegex(ValueError, "database hash"):
            register_validation(
                self.registry,
                missing,
                validation_sha256="a" * 64,
                source_fingerprints=self.sources,
                now=self.t0,
            )

        failed = report("q20-rejected", "Q20", self.t0, passed=False)
        first = register_validation(
            self.registry,
            failed,
            validation_sha256="a" * 64,
            source_fingerprints=self.sources,
            now=self.t0,
        )
        first["models"]["q20-rejected"]["model_name"] = ""
        replacement = report(
            "q20-rejected", "Q20", self.t0 + timedelta(hours=1), passed=False
        )
        replacement["model"] = "Canonical model"
        replacement["design"]["validation_protocol"] = "corrected-v2"
        second = register_validation(
            first,
            replacement,
            validation_sha256="c" * 64,
            source_fingerprints=self.sources,
            now=self.t0 + timedelta(hours=1),
            supersede_rejected=True,
        )
        candidate = second["models"]["q20-rejected"]
        self.assertEqual(candidate["status"], "REJECTED")
        self.assertEqual(len(candidate["validation_history"]), 1)
        different_name = report(
            "q20-rejected", "Q20", self.t0 + timedelta(hours=2), passed=False
        )
        different_name["model"] = "Different model"
        with self.assertRaisesRegex(ValueError, "supersede requires"):
            register_validation(
                second,
                different_name,
                validation_sha256="d" * 64,
                source_fingerprints=self.sources,
                now=self.t0 + timedelta(hours=2),
                supersede_rejected=True,
            )
        passing = report("q20-rejected", "Q20", self.t0 + timedelta(hours=2))
        with self.assertRaisesRegex(ValueError, "supersede requires"):
            register_validation(
                second,
                passing,
                validation_sha256="e" * 64,
                source_fingerprints=self.sources,
                now=self.t0 + timedelta(hours=2),
                supersede_rejected=True,
            )

    def test_clean_rerun_must_be_new_and_after_cooling(self) -> None:
        original = report("q20-v1", "Q20", self.t0)
        registered = register_validation(
            self.registry,
            original,
            validation_sha256="a" * 64,
            source_fingerprints=self.sources,
            now=self.t0,
        )
        clean_time = self.t0 + timedelta(hours=25)
        clean = report("q20-v1", "Q20", clean_time)
        with self.assertRaisesRegex(ValueError, "different validation hash"):
            start_shadow(
                registered,
                "q20-v1",
                clean,
                validation_sha256="a" * 64,
                source_fingerprints=self.sources,
                now=clean_time,
            )
        too_old = report("q20-v1", "Q20", self.t0 + timedelta(hours=23))
        with self.assertRaisesRegex(ValueError, "generated after cooling-off"):
            start_shadow(
                registered,
                "q20-v1",
                too_old,
                validation_sha256="c" * 64,
                source_fingerprints=self.sources,
                now=clean_time,
            )

    def test_shadow_rejects_missing_weekend_future_and_mismatched_evidence(self) -> None:
        shadow = self._start_shadow()
        now = datetime(2026, 8, 17, 23, tzinfo=timezone.utc)
        common = {"event": "NONE", "passed": True, "now": now}
        with self.assertRaisesRegex(ValueError, "evidence"):
            record_shadow_day(shadow, "q20-v1", "2026-08-17", **common)
        with self.assertRaisesRegex(ValueError, "weekday"):
            record_shadow_day(
                shadow,
                "q20-v1",
                "2026-08-15",
                evidence_sha256="e" * 64,
                evidence_as_of="2026-08-15",
                **common,
            )
        with self.assertRaisesRegex(ValueError, "future"):
            record_shadow_day(
                shadow,
                "q20-v1",
                "2026-08-18",
                evidence_sha256="e" * 64,
                evidence_as_of="2026-08-18",
                **common,
            )
        with self.assertRaisesRegex(ValueError, "must equal"):
            record_shadow_day(
                shadow,
                "q20-v1",
                "2026-08-17",
                evidence_sha256="e" * 64,
                evidence_as_of="2026-08-14",
                **common,
            )

    def test_shadow_expires_at_252_days_without_both_cycles(self) -> None:
        registry = self._start_shadow()
        cursor = date(2026, 8, 11)
        recorded = 0
        while recorded < 252:
            if cursor.weekday() < 5:
                registry = record_shadow_day(
                    registry,
                    "q20-v1",
                    cursor,
                    event="NONE",
                    passed=True,
                    evidence_sha256=f"{recorded:064x}",
                    evidence_as_of=cursor,
                    now=datetime.combine(cursor, time(23), tzinfo=timezone.utc),
                )
                recorded += 1
            cursor += timedelta(days=1)

        candidate = registry["models"]["q20-v1"]
        self.assertEqual(candidate["status"], "SHADOW_FAILED")
        self.assertEqual(candidate["shadow_failure_reason"], "SHADOW_WINDOW_EXPIRED")
        with self.assertRaisesRegex(ValueError, "not in SHADOW"):
            record_shadow_day(
                registry,
                "q20-v1",
                cursor,
                event="REENTRY",
                passed=True,
                evidence_sha256="f" * 64,
                evidence_as_of=cursor,
                now=datetime.combine(cursor, time(23), tzinfo=timezone.utc),
            )

    def test_promotion_is_role_isolated_and_source_drift_is_blocked(self) -> None:
        ready = self._ready_shadow()
        with self.assertRaisesRegex(ValueError, "source fingerprints"):
            promote_model(
                ready,
                "q20-v1",
                source_fingerprints={"MODEL_SPEC_v1.md": "f" * 64},
            )
        promoted = promote_model(
            ready,
            "q20-v1",
            source_fingerprints=self.sources,
            now=datetime(2027, 3, 1, tzinfo=timezone.utc),
        )
        self.assertEqual(promoted["roles"]["Q20"]["active_model_id"], "q20-v1")
        self.assertEqual(
            promoted["roles"]["CURRENT_REGIME"]["active_model_id"], "hmm-live-v1"
        )
        with self.assertRaisesRegex(ValueError, "rollback target"):
            rollback_model(
                promoted, "hmm-rollback-v2", role="Q20", now=self.t0
            )
        rolled_back = rollback_model(
            promoted,
            "hmm-rollback-v2",
            role="CURRENT_REGIME",
            now=datetime(2027, 3, 2, tzinfo=timezone.utc),
        )
        self.assertEqual(
            rolled_back["roles"]["CURRENT_REGIME"]["active_model_id"],
            "hmm-rollback-v2",
        )
        self.assertEqual(rolled_back["roles"]["Q20"]["active_model_id"], "q20-v1")

    def test_manual_activation_is_explicit_current_regime_override(self) -> None:
        failed = report("hmm-rejected-v2", "CURRENT_REGIME", self.t0, passed=False)
        registered = register_validation(
            self.registry,
            failed,
            validation_sha256="a" * 64,
            source_fingerprints=self.sources,
            now=self.t0,
        )
        original_validation = registered["models"]["hmm-rejected-v2"][
            "registered_validation"
        ].copy()

        with self.assertRaisesRegex(ValueError, "not in SHADOW"):
            promote_model(
                registered,
                "hmm-rejected-v2",
                source_fingerprints=self.sources,
                now=self.t0,
            )
        with self.assertRaisesRegex(ValueError, "reason and approved_by"):
            manual_activate_model(
                registered,
                "hmm-rejected-v2",
                reason=" ",
                approved_by="user",
                now=self.t0,
            )

        activated = manual_activate_model(
            registered,
            "hmm-rejected-v2",
            reason="Explicit deployment instruction despite failed historical gates",
            approved_by="user",
            now=self.t0 + timedelta(hours=1),
        )
        current = activated["roles"]["CURRENT_REGIME"]
        candidate = activated["models"]["hmm-rejected-v2"]
        self.assertEqual(current["active_model_id"], "hmm-rejected-v2")
        self.assertEqual(
            current["rollback_model_ids"], ["hmm-live-v1", "hmm-rollback-v2"]
        )
        self.assertEqual(candidate["status"], "MANUAL_OVERRIDE_ACTIVE")
        self.assertEqual(candidate["activation_reason"], "MANUAL_OVERRIDE")
        self.assertEqual(candidate["manual_override"]["approved_by"], "user")
        self.assertEqual(candidate["manual_override"]["previous_status"], "REJECTED")
        self.assertEqual(candidate["registered_validation"], original_validation)
        self.assertEqual(
            candidate["registered_validation"]["status"], "HISTORICAL_FAIL"
        )
        self.assertEqual(activated["models"]["hmm-live-v1"]["status"], "ROLLBACK")

        rolled_back = rollback_model(
            activated,
            "hmm-live-v1",
            role="CURRENT_REGIME",
            now=self.t0 + timedelta(hours=2),
        )
        self.assertEqual(
            rolled_back["roles"]["CURRENT_REGIME"]["active_model_id"], "hmm-live-v1"
        )
        self.assertEqual(
            rolled_back["roles"]["CURRENT_REGIME"]["rollback_model_ids"],
            ["hmm-rejected-v2", "hmm-rollback-v2"],
        )
        self.assertEqual(
            rolled_back["models"]["hmm-rejected-v2"]["manual_override"],
            candidate["manual_override"],
        )

    def test_manual_activation_rejects_other_roles_and_non_rejected_models(self) -> None:
        failed_q20 = report("q20-rejected", "Q20", self.t0, passed=False)
        registered = register_validation(
            self.registry,
            failed_q20,
            validation_sha256="a" * 64,
            source_fingerprints=self.sources,
            now=self.t0,
        )
        with self.assertRaisesRegex(ValueError, "limited to CURRENT_REGIME"):
            manual_activate_model(
                registered,
                "q20-rejected",
                reason="explicit",
                approved_by="user",
                now=self.t0,
            )
        with self.assertRaisesRegex(ValueError, "requires a REJECTED"):
            manual_activate_model(
                self.registry,
                "hmm-live-v1",
                reason="explicit",
                approved_by="user",
                now=self.t0,
            )

    def test_atomic_registry_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            write_registry(path, self.registry)
            restored = read_registry(path)
        self.assertEqual(restored, self.registry)


if __name__ == "__main__":
    unittest.main()
