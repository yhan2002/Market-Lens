"""Local, role-isolated model promotion registry.

Registry approval is an audit decision only. It never edits live model code.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Mapping


NOTICE = "Registry approval does not mutate live model code."
ROLES = ("CURRENT_REGIME", "Q20", "POLICY")
EVENTS = {"NONE", "DE_RISK", "REENTRY"}
SHADOW_DAYS_REQUIRED = 126
SHADOW_DAYS_MAX = 252


def _empty_role_slot() -> dict[str, object]:
    return {"active_model_id": None, "rollback_model_ids": []}


def new_registry() -> dict[str, object]:
    return {
        "schema_version": 2,
        "notice": NOTICE,
        "roles": {role: _empty_role_slot() for role in ROLES},
        "models": {},
    }


def _normalize_role(value: object) -> str:
    if not isinstance(value, str) or value not in ROLES:
        raise ValueError(f"role must be one of {list(ROLES)}")
    return value


def _infer_legacy_role(model_id: str, candidate: Mapping[str, object]) -> str:
    explicit = candidate.get("role")
    if explicit in ROLES:
        return str(explicit)
    design = candidate.get("design")
    design = design if isinstance(design, Mapping) else {}
    design_role = design.get("role")
    if design_role in ROLES:
        return str(design_role)
    text = " ".join(
        str(value).lower()
        for value in (model_id, candidate.get("model_version"), design_role, design)
    )
    if "q20" in text or "liquidation" in text:
        return "Q20"
    if "policy" in text or "exposure" in text:
        return "POLICY"
    return "CURRENT_REGIME"


def _migrate_v1(registry: Mapping[str, object]) -> dict[str, object]:
    result = new_registry()
    raw_models = registry.get("models")
    if not isinstance(raw_models, Mapping):
        raise ValueError("registry models must be an object")
    models = result["models"]
    assert isinstance(models, dict)
    for model_id, raw_candidate in raw_models.items():
        if not isinstance(raw_candidate, Mapping):
            raise ValueError("candidate entry must be an object")
        candidate = copy.deepcopy(dict(raw_candidate))
        candidate["role"] = _infer_legacy_role(str(model_id), candidate)
        models[str(model_id)] = candidate

    roles = result["roles"]
    assert isinstance(roles, dict)
    active = registry.get("active_model_id")
    if isinstance(active, str) and active in models:
        role = str(models[active]["role"])
        roles[role]["active_model_id"] = active
    raw_rollbacks = registry.get("rollback_model_ids")
    if isinstance(raw_rollbacks, list):
        for model_id in raw_rollbacks:
            if not isinstance(model_id, str) or model_id not in models:
                continue
            role = str(models[model_id]["role"])
            slot = roles[role]
            if (
                model_id != slot["active_model_id"]
                and model_id not in slot["rollback_model_ids"]
                and len(slot["rollback_model_ids"]) < 2
            ):
                slot["rollback_model_ids"].append(model_id)
    return result


def _registry_copy(registry: Mapping[str, object] | None) -> dict[str, object]:
    if not registry:
        return new_registry()
    version = registry.get("schema_version", 1)
    if version == 1:
        result = _migrate_v1(registry)
    elif version == 2:
        result = copy.deepcopy(dict(registry))
    else:
        raise ValueError(f"unsupported registry schema_version: {version}")
    result["schema_version"] = 2
    result["notice"] = NOTICE
    models = result.setdefault("models", {})
    roles = result.setdefault("roles", {})
    if not isinstance(models, dict) or not isinstance(roles, dict):
        raise ValueError("registry models and roles must be objects")
    for model_id, candidate in models.items():
        if not isinstance(candidate, dict):
            raise ValueError("candidate entry must be an object")
        candidate["role"] = _normalize_role(
            candidate.get("role") or _infer_legacy_role(str(model_id), candidate)
        )
    for role in ROLES:
        slot = roles.setdefault(role, _empty_role_slot())
        if not isinstance(slot, dict):
            raise ValueError(f"registry role slot must be an object: {role}")
        slot.setdefault("active_model_id", None)
        slot.setdefault("rollback_model_ids", [])
        active = slot["active_model_id"]
        rollbacks = slot["rollback_model_ids"]
        if active is not None and (
            not isinstance(active, str)
            or active not in models
            or models[active]["role"] != role
        ):
            raise ValueError(f"active model is outside role {role}")
        if (
            not isinstance(rollbacks, list)
            or len(rollbacks) > 2
            or len(set(rollbacks)) != len(rollbacks)
            or active in rollbacks
            or any(
                not isinstance(item, str)
                or item not in models
                or models[item]["role"] != role
                for item in rollbacks
            )
        ):
            raise ValueError(f"invalid rollback slots for role {role}")
    return result


def _utc(now: datetime | None) -> datetime:
    value = now or datetime.now(timezone.utc)
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc)


def _timestamp(now: datetime | None) -> str:
    return _utc(now).isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as error:
        raise ValueError("timestamp must be ISO-8601 with timezone") from error
    return _utc(parsed)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_files(paths: Iterable[Path]) -> dict[str, str]:
    fingerprints: dict[str, str] = {}
    for path in paths:
        resolved = path.resolve(strict=True)
        fingerprints[str(resolved)] = sha256_file(resolved)
    if not fingerprints:
        raise ValueError("at least one source fingerprint is required")
    return fingerprints


def _normalized_hash(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a SHA256 hex digest")
    normalized = value.lower()
    if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
        raise ValueError(f"{label} must be a SHA256 hex digest")
    return normalized


def _normalized_fingerprints(values: Mapping[str, str]) -> dict[str, str]:
    if not values:
        raise ValueError("at least one source fingerprint is required")
    return {
        str(name): _normalized_hash(value, f"source fingerprint {name}")
        for name, value in sorted(values.items())
    }


def _fingerprints_by_name(values: Mapping[str, str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for path, digest in _normalized_fingerprints(values).items():
        name = Path(path).name
        if name in result:
            raise ValueError(f"duplicate source basename: {name}")
        result[name] = digest
    return result


def _trading_date(value: object, label: str = "trading_date") -> str:
    if isinstance(value, datetime):
        raise ValueError(f"{label} must not include a time")
    if isinstance(value, date):
        return value.isoformat()
    if not isinstance(value, str):
        raise ValueError(f"{label} must be ISO YYYY-MM-DD")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{label} must be ISO YYYY-MM-DD") from error
    if parsed.isoformat() != value:
        raise ValueError(f"{label} must be ISO YYYY-MM-DD")
    return value


def _report_identity(report: Mapping[str, object]) -> dict[str, object]:
    role = _normalize_role(report.get("role"))
    definition = report.get("definition")
    definition = definition if isinstance(definition, Mapping) else {}
    version = report.get("model_version") or definition.get("model_version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("validation report requires model_version")
    model_id = report.get("model_id") or version
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("validation report requires a valid model_id")
    design = report.get("design") or definition
    if not isinstance(design, Mapping) or not design:
        raise ValueError("validation report requires a non-empty design")
    model_name = report.get("model") or design.get("model") or ""
    if not isinstance(model_name, str):
        raise ValueError("model must be a string")
    design_copy = copy.deepcopy(dict(design))
    return {
        "role": role,
        "model_id": model_id.strip(),
        "model_version": version.strip(),
        "model_name": model_name,
        "design": design_copy,
        "design_sha256": _canonical_sha256(design_copy),
    }


def _report_provenance(
    report: Mapping[str, object],
    source_fingerprints: Mapping[str, str],
    now: datetime,
) -> dict[str, object]:
    raw = report.get("provenance")
    if not isinstance(raw, Mapping):
        raise ValueError("validation report requires provenance")
    generated_at = _parse_timestamp(raw.get("generated_at_utc"))
    if generated_at > now:
        raise ValueError("report generated_at_utc cannot be in the future")
    data_cutoff = _trading_date(raw.get("data_cutoff"), "data_cutoff")
    if date.fromisoformat(data_cutoff) > now.date():
        raise ValueError("report data_cutoff cannot be in the future")
    database = raw.get("database")
    if not isinstance(database, Mapping):
        raise ValueError("validation report requires provenance database hash")
    database_copy = copy.deepcopy(dict(database))
    database_copy["sha256"] = _normalized_hash(
        database.get("sha256"), "provenance database sha256"
    )
    report_sources = raw.get("source_sha256")
    if not isinstance(report_sources, Mapping):
        raise ValueError("validation report requires provenance source hashes")
    normalized_report_sources = _normalized_fingerprints(report_sources)
    if _fingerprints_by_name(normalized_report_sources) != _fingerprints_by_name(
        source_fingerprints
    ):
        raise ValueError("report provenance source hashes do not match current sources")
    return {
        "generated_at_utc": _timestamp(generated_at),
        "data_cutoff": data_cutoff,
        "database": database_copy,
        "source_sha256": normalized_report_sources,
    }


def _gates(report: Mapping[str, object]) -> tuple[dict[str, bool], bool]:
    raw = report.get("gates")
    if not isinstance(raw, Mapping) or not raw:
        return {}, False
    gates = {str(name): value for name, value in raw.items()}
    valid = all(type(value) is bool for value in gates.values())
    return gates, valid and all(gates.values())


def _candidate(registry: Mapping[str, object], model_id: str) -> dict[str, object]:
    models = registry.get("models")
    if not isinstance(models, dict) or model_id not in models:
        raise ValueError(f"unknown model: {model_id}")
    candidate = models[model_id]
    if not isinstance(candidate, dict):
        raise ValueError("candidate entry must be an object")
    return candidate


def _role_slot(registry: Mapping[str, object], role: str) -> dict[str, object]:
    normalized = _normalize_role(role)
    roles = registry.get("roles")
    if not isinstance(roles, dict) or not isinstance(roles.get(normalized), dict):
        raise ValueError(f"missing registry role slot: {normalized}")
    return roles[normalized]


def seed_operational_models(
    registry: Mapping[str, object] | None,
    active_model_id: str,
    rollback_model_ids: Iterable[str],
    *,
    source_fingerprints: Mapping[str, str],
    now: datetime | None = None,
) -> dict[str, object]:
    """Seed CURRENT_REGIME only, with one provisional and two legacy slots."""

    result = _registry_copy(registry)
    models = result["models"]
    roles = result["roles"]
    assert isinstance(models, dict) and isinstance(roles, dict)
    if models or any(
        slot["active_model_id"] is not None or slot["rollback_model_ids"]
        for slot in roles.values()
    ):
        raise ValueError("operational seed requires an empty registry")
    active = active_model_id.strip()
    rollbacks = [str(value).strip() for value in rollback_model_ids]
    if not active or len(rollbacks) != 2 or len(set(rollbacks)) != 2 or active in rollbacks:
        raise ValueError("seed requires one active model and two distinct rollback models")
    created_at = _timestamp(now)
    fingerprints = _normalized_fingerprints(source_fingerprints)
    for model_id, status in (
        (active, "PROVISIONAL"),
        *((item, "ROLLBACK") for item in rollbacks),
    ):
        design = {"role": "CURRENT_REGIME_DISPLAY_ONLY", "bootstrap_seed": True}
        models[model_id] = {
            "role": "CURRENT_REGIME",
            "model_id": model_id,
            "model_version": model_id,
            "model_name": "Sticky Gaussian HMM",
            "design": design,
            "design_sha256": _canonical_sha256(design),
            "status": status,
            "created_at": created_at,
            "registered_validation": {
                "status": "LEGACY_OPERATIONAL_BASELINE",
                "gates": {},
                "sha256": None,
                "source_fingerprints": fingerprints,
                "source": "verified database inventory",
            },
            "validation_history": [],
            "shadow": None,
        }
    roles["CURRENT_REGIME"]["active_model_id"] = active
    roles["CURRENT_REGIME"]["rollback_model_ids"] = rollbacks
    return result


def register_validation(
    registry: Mapping[str, object] | None,
    report: Mapping[str, object],
    *,
    validation_sha256: str,
    source_fingerprints: Mapping[str, str],
    now: datetime | None = None,
    validation_source: str | None = None,
    supersede_rejected: bool = False,
) -> dict[str, object]:
    """Register a frozen role-specific candidate."""

    result = _registry_copy(registry)
    identity = _report_identity(report)
    model_id = str(identity["model_id"])
    models = result["models"]
    assert isinstance(models, dict)
    created = _utc(now)
    sources = _normalized_fingerprints(source_fingerprints)
    provenance = _report_provenance(report, sources, created)
    validation_hash = _normalized_hash(validation_sha256, "validation_sha256")
    gates, gates_pass = _gates(report)
    historical_pass = report.get("status") == "HISTORICAL_PASS" and gates_pass
    previous = models.get(model_id)
    history: list[object] = []
    if previous is not None:
        if not supersede_rejected:
            raise ValueError(f"model already registered: {model_id}")
        if not isinstance(previous, Mapping):
            raise ValueError("candidate entry must be an object")
        same_core_identity = all(
            previous.get(key) == identity[key]
            for key in ("role", "model_id", "model_version")
        )
        previous_name = previous.get("model_name")
        same_name = previous_name == identity["model_name"] or previous_name in (None, "")
        if (
            previous.get("status") != "REJECTED"
            or historical_pass
            or not same_core_identity
            or not same_name
        ):
            raise ValueError(
                "supersede requires same-role same-identity REJECTED validations"
            )
        old_history = previous.get("validation_history")
        history = copy.deepcopy(old_history) if isinstance(old_history, list) else []
        history.append(copy.deepcopy(previous.get("registered_validation")))

    candidate = {
        **identity,
        "status": "COOLING_OFF" if historical_pass else "REJECTED",
        "created_at": _timestamp(created),
        "eligible_after": _timestamp(created + timedelta(hours=24))
        if historical_pass
        else None,
        "registered_validation": {
            "status": report.get("status"),
            "gates": gates,
            "sha256": validation_hash,
            "source_fingerprints": sources,
            "source": validation_source,
            "provenance": provenance,
        },
        "validation_history": history,
        "rejection_reason": None
        if historical_pass
        else "Historical status and every gate must pass",
        "shadow": None,
    }
    models[model_id] = candidate
    return result


def start_shadow(
    registry: Mapping[str, object],
    model_id: str,
    clean_report: Mapping[str, object],
    *,
    validation_sha256: str,
    source_fingerprints: Mapping[str, str],
    now: datetime | None = None,
    validation_source: str | None = None,
) -> dict[str, object]:
    """Start shadow after 24 hours and a distinct, clean rerun."""

    result = _registry_copy(registry)
    candidate = _candidate(result, model_id)
    if candidate.get("status") != "COOLING_OFF":
        raise ValueError("model is not in COOLING_OFF")
    current_time = _utc(now)
    eligible_after = candidate.get("eligible_after")
    if not isinstance(eligible_after, str) or current_time < _parse_timestamp(eligible_after):
        raise ValueError("24-hour cooling-off period has not elapsed")

    identity = _report_identity(clean_report)
    for key in ("role", "model_id", "model_version", "model_name", "design_sha256"):
        if identity[key] != candidate.get(key):
            raise ValueError(f"clean rerun {key} does not match registered candidate")
    gates, gates_pass = _gates(clean_report)
    if clean_report.get("status") != "HISTORICAL_PASS" or not gates_pass:
        raise ValueError("clean rerun must pass every historical gate")
    normalized_sources = _normalized_fingerprints(source_fingerprints)
    provenance = _report_provenance(clean_report, normalized_sources, current_time)
    registered = candidate.get("registered_validation")
    if not isinstance(registered, Mapping):
        raise ValueError("registered validation is missing")
    if normalized_sources != registered.get("source_fingerprints"):
        raise ValueError("clean rerun source fingerprints changed")
    registered_provenance = registered.get("provenance")
    if not isinstance(registered_provenance, Mapping):
        raise ValueError("registered provenance is missing")
    if provenance["source_sha256"] != registered_provenance.get("source_sha256"):
        raise ValueError("clean rerun provenance sources changed")
    clean_hash = _normalized_hash(validation_sha256, "validation_sha256")
    if clean_hash == registered.get("sha256"):
        raise ValueError("clean rerun must use a different validation hash")
    generated_at = _parse_timestamp(str(provenance["generated_at_utc"]))
    if generated_at < _parse_timestamp(eligible_after) or generated_at > current_time:
        raise ValueError("clean rerun report must be generated after cooling-off and before now")
    if date.fromisoformat(str(provenance["data_cutoff"])) < date.fromisoformat(
        str(registered_provenance.get("data_cutoff"))
    ):
        raise ValueError("clean rerun data_cutoff cannot move backward")

    candidate["status"] = "SHADOW"
    candidate["clean_rerun_validation"] = {
        "status": clean_report.get("status"),
        "gates": gates,
        "sha256": clean_hash,
        "source_fingerprints": normalized_sources,
        "source": validation_source,
        "provenance": provenance,
        "validated_at": _timestamp(current_time),
    }
    candidate["shadow"] = {
        "started_at": _timestamp(current_time),
        "start_date": current_time.date().isoformat(),
        "records": [],
    }
    return result


def record_shadow_day(
    registry: Mapping[str, object],
    model_id: str,
    trading_date: str | date,
    *,
    event: str,
    passed: bool,
    evidence_sha256: str | None = None,
    evidence_as_of: str | date | None = None,
    now: datetime | None = None,
) -> dict[str, object]:
    result = _registry_copy(registry)
    candidate = _candidate(result, model_id)
    if candidate.get("status") != "SHADOW":
        raise ValueError("model is not in SHADOW")
    if event not in EVENTS:
        raise ValueError(f"event must be one of {sorted(EVENTS)}")
    if type(passed) is not bool:
        raise ValueError("passed must be a boolean")
    if evidence_sha256 is None or evidence_as_of is None:
        raise ValueError("shadow evidence SHA256 and as_of are required")
    evidence_hash = _normalized_hash(evidence_sha256, "evidence_sha256")
    normalized_date = _trading_date(trading_date)
    normalized_as_of = _trading_date(evidence_as_of, "evidence_as_of")
    if normalized_as_of != normalized_date:
        raise ValueError("evidence_as_of must equal trading_date")
    trading_day = date.fromisoformat(normalized_date)
    if trading_day.weekday() >= 5:
        raise ValueError("shadow trading_date must be a weekday")
    current_time = _utc(now)
    shadow = candidate.get("shadow")
    if not isinstance(shadow, dict) or not isinstance(shadow.get("records"), list):
        raise ValueError("shadow state is missing")
    start_date = _trading_date(shadow.get("start_date"), "shadow start_date")
    if trading_day < date.fromisoformat(start_date):
        raise ValueError("shadow trading_date precedes shadow start date")
    if trading_day > current_time.date():
        raise ValueError("shadow trading_date cannot be in the future")
    records = shadow["records"]
    if len(records) >= SHADOW_DAYS_MAX:
        raise ValueError(f"shadow window is limited to {SHADOW_DAYS_MAX} trading dates")
    if records and normalized_date <= records[-1].get("trading_date", ""):
        raise ValueError("shadow trading dates must be strictly increasing and unique")
    records.append(
        {
            "trading_date": normalized_date,
            "event": event,
            "passed": passed,
            "evidence_sha256": evidence_hash,
            "evidence_as_of": normalized_as_of,
            "recorded_at": _timestamp(current_time),
        }
    )
    if not passed:
        candidate["status"] = "SHADOW_FAILED"
        candidate["shadow_failed_at"] = _timestamp(current_time)
        candidate["shadow_failure_reason"] = "DAILY_EVIDENCE_FAILED"
    elif len(records) == SHADOW_DAYS_MAX:
        events = {record.get("event") for record in records}
        if not {"DE_RISK", "REENTRY"}.issubset(events):
            candidate["status"] = "SHADOW_FAILED"
            candidate["shadow_failed_at"] = _timestamp(current_time)
            candidate["shadow_failure_reason"] = "SHADOW_WINDOW_EXPIRED"
    return result


def promote_model(
    registry: Mapping[str, object],
    model_id: str,
    *,
    source_fingerprints: Mapping[str, str],
    now: datetime | None = None,
) -> dict[str, object]:
    result = _registry_copy(registry)
    candidate = _candidate(result, model_id)
    if candidate.get("status") != "SHADOW":
        raise ValueError("model is not in SHADOW")
    clean = candidate.get("clean_rerun_validation")
    if not isinstance(clean, Mapping) or _normalized_fingerprints(
        source_fingerprints
    ) != clean.get("source_fingerprints"):
        raise ValueError("current source fingerprints differ from clean rerun")
    shadow = candidate.get("shadow")
    records = shadow.get("records") if isinstance(shadow, Mapping) else None
    if not isinstance(records, list):
        raise ValueError("shadow records are missing")
    dates = [record.get("trading_date") for record in records]
    if dates != sorted(set(dates)):
        raise ValueError("shadow trading dates are not strictly increasing and unique")
    if len(records) < SHADOW_DAYS_REQUIRED or not all(
        record.get("passed") is True for record in records
    ):
        raise ValueError(f"at least {SHADOW_DAYS_REQUIRED} passed shadow dates are required")
    if len(records) > SHADOW_DAYS_MAX:
        raise ValueError(f"shadow window exceeds {SHADOW_DAYS_MAX} trading dates")
    events = {record.get("event") for record in records}
    if not {"DE_RISK", "REENTRY"}.issubset(events):
        raise ValueError("shadow requires at least one DE_RISK and one REENTRY event")

    role = str(candidate["role"])
    slot = _role_slot(result, role)
    previous_active = slot.get("active_model_id")
    if previous_active == model_id:
        raise ValueError("model is already active")
    rollbacks = [item for item in list(slot.get("rollback_model_ids") or []) if item != model_id]
    if isinstance(previous_active, str):
        previous = _candidate(result, previous_active)
        if previous.get("role") != role:
            raise ValueError("active model role mismatch")
        previous["status"] = "ROLLBACK"
        rollbacks = [previous_active] + [item for item in rollbacks if item != previous_active]
    candidate["status"] = "APPROVED"
    candidate["approved_at"] = _timestamp(now)
    slot["active_model_id"] = model_id
    slot["rollback_model_ids"] = rollbacks[:2]
    return result


def manual_activate_model(
    registry: Mapping[str, object],
    model_id: str,
    *,
    reason: str,
    approved_by: str,
    now: datetime | None = None,
) -> dict[str, object]:
    """Activate a rejected CURRENT_REGIME candidate under explicit human override."""

    result = _registry_copy(registry)
    candidate = _candidate(result, model_id)
    if candidate.get("role") != "CURRENT_REGIME":
        raise ValueError("manual activation is limited to CURRENT_REGIME")
    if candidate.get("status") != "REJECTED":
        raise ValueError("manual activation requires a REJECTED candidate")
    normalized_reason = reason.strip() if isinstance(reason, str) else ""
    normalized_approver = approved_by.strip() if isinstance(approved_by, str) else ""
    if not normalized_reason or not normalized_approver:
        raise ValueError("manual activation requires reason and approved_by")

    slot = _role_slot(result, "CURRENT_REGIME")
    previous_active = slot.get("active_model_id")
    if previous_active == model_id:
        raise ValueError("model is already active")
    rollbacks = [item for item in list(slot.get("rollback_model_ids") or []) if item != model_id]
    if isinstance(previous_active, str):
        previous = _candidate(result, previous_active)
        previous["status"] = "ROLLBACK"
        rollbacks = [previous_active] + [item for item in rollbacks if item != previous_active]

    activated_at = _timestamp(now)
    candidate["status"] = "MANUAL_OVERRIDE_ACTIVE"
    candidate["activation_reason"] = "MANUAL_OVERRIDE"
    candidate["manual_override"] = {
        "approved_by": normalized_approver,
        "reason": normalized_reason,
        "activated_at": activated_at,
        "previous_status": "REJECTED",
    }
    slot["active_model_id"] = model_id
    slot["rollback_model_ids"] = rollbacks[:2]
    return result


def rollback_model(
    registry: Mapping[str, object],
    model_id: str | None = None,
    *,
    role: str,
    now: datetime | None = None,
) -> dict[str, object]:
    result = _registry_copy(registry)
    slot = _role_slot(result, role)
    active_id = slot.get("active_model_id")
    if not isinstance(active_id, str):
        raise ValueError(f"there is no active model for role {role}")
    rollbacks = list(slot.get("rollback_model_ids") or [])
    target_id = model_id or (rollbacks[0] if rollbacks else None)
    if target_id not in rollbacks:
        raise ValueError("rollback target is not in the retained role rollback slots")
    active = _candidate(result, active_id)
    target = _candidate(result, str(target_id))
    if active.get("role") != role or target.get("role") != role:
        raise ValueError("rollback cannot cross model roles")
    active["status"] = "ROLLBACK"
    target["status"] = "APPROVED"
    target["approved_at"] = _timestamp(now)
    target["activation_reason"] = "ROLLBACK"
    remaining = [item for item in rollbacks if item != target_id]
    slot["active_model_id"] = target_id
    slot["rollback_model_ids"] = ([active_id] + remaining)[:2]
    return result


def write_registry(path: Path, registry: Mapping[str, object]) -> None:
    """Atomically replace a registry with a flushed UTF-8 JSON file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(registry, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def read_registry(path: Path) -> dict[str, object]:
    if not path.exists():
        return new_registry()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("registry root must be an object")
    return _registry_copy(loaded)


def _load_json(path: Path, label: str) -> dict[str, object]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{label} root must be an object")
    return loaded


def _sources(values: list[Path]) -> dict[str, str]:
    return fingerprint_files(values)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=NOTICE)
    commands = parser.add_subparsers(dest="command", required=True)

    seed = commands.add_parser("seed-live")
    seed.add_argument("registry", type=Path)
    seed.add_argument("--active-model", required=True)
    seed.add_argument("--rollback-model", action="append", required=True)
    seed.add_argument("--source", type=Path, action="append", required=True)

    register = commands.add_parser("register")
    register.add_argument("registry", type=Path)
    register.add_argument("report", type=Path)
    register.add_argument("--source", type=Path, action="append", required=True)
    register.add_argument("--supersede-rejected", action="store_true")

    shadow = commands.add_parser("start-shadow")
    shadow.add_argument("registry", type=Path)
    shadow.add_argument("model_id")
    shadow.add_argument("report", type=Path)
    shadow.add_argument("--source", type=Path, action="append", required=True)

    record = commands.add_parser("record")
    record.add_argument("registry", type=Path)
    record.add_argument("model_id")
    record.add_argument("trading_date")
    record.add_argument("--event", choices=sorted(EVENTS), required=True)
    record.add_argument("--result", choices=("PASS", "FAIL"), required=True)
    record.add_argument("--evidence", type=Path, required=True)

    promote = commands.add_parser("promote")
    promote.add_argument("registry", type=Path)
    promote.add_argument("model_id")
    promote.add_argument("--source", type=Path, action="append", required=True)

    manual_activate = commands.add_parser("manual-activate")
    manual_activate.add_argument("registry", type=Path)
    manual_activate.add_argument("model_id")
    manual_activate.add_argument("--reason", required=True)
    manual_activate.add_argument("--approved-by", required=True)

    rollback = commands.add_parser("rollback")
    rollback.add_argument("registry", type=Path)
    rollback.add_argument("--role", choices=ROLES, required=True)
    rollback.add_argument("--model-id")

    show = commands.add_parser("show")
    show.add_argument("registry", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "show":
        print(json.dumps(read_registry(args.registry), ensure_ascii=False, indent=2))
        return
    registry = read_registry(args.registry)
    if args.command == "seed-live":
        updated = seed_operational_models(
            registry,
            args.active_model,
            args.rollback_model,
            source_fingerprints=_sources(args.source),
        )
        model_id = args.active_model
    elif args.command in {"register", "start-shadow"}:
        report = _load_json(args.report, "validation report")
        kwargs = {
            "validation_sha256": sha256_file(args.report),
            "source_fingerprints": _sources(args.source),
            "validation_source": str(args.report.resolve()),
        }
        if args.command == "register":
            updated = register_validation(
                registry,
                report,
                supersede_rejected=args.supersede_rejected,
                **kwargs,
            )
            model_id = str(_report_identity(report)["model_id"])
        else:
            updated = start_shadow(registry, args.model_id, report, **kwargs)
            model_id = args.model_id
    elif args.command == "record":
        evidence = _load_json(args.evidence, "evidence")
        updated = record_shadow_day(
            registry,
            args.model_id,
            args.trading_date,
            event=args.event,
            passed=args.result == "PASS",
            evidence_sha256=sha256_file(args.evidence),
            evidence_as_of=evidence.get("as_of"),
        )
        model_id = args.model_id
    elif args.command == "promote":
        updated = promote_model(
            registry,
            args.model_id,
            source_fingerprints=_sources(args.source),
        )
        model_id = args.model_id
    elif args.command == "manual-activate":
        updated = manual_activate_model(
            registry,
            args.model_id,
            reason=args.reason,
            approved_by=args.approved_by,
        )
        model_id = args.model_id
    else:
        updated = rollback_model(
            registry, args.model_id, role=args.role
        )
        model_id = str(_role_slot(updated, args.role)["active_model_id"])
    write_registry(args.registry, updated)
    status = _candidate(updated, model_id)["status"]
    print(f"model_id={model_id} role={_candidate(updated, model_id)['role']} status={status}")
    print(NOTICE)


if __name__ == "__main__":
    main()
