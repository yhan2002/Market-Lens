"""Pure, versioned RORO score and Fast-overlay exposure policy."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from datetime import date
from typing import Any


POLICY_VERSION = "roro-policy-v1-score-fast3of8-confirm3-shadow"
STATE_VERSION = 1
FAST_TRIGGER_REQUIRED = 3
FAST_TRIGGER_TOTAL = 8
FAST_AVAILABLE_FOR_RECOVERY = 5
FAST_CUT_PERCENTAGE_POINTS = 20
RECOVERY_CONFIRM_DAYS = 3
MIN_LONG = 25
LONG_ANCHORS = ((0, 100), (33, 90), (66, 60), (100, 25))

MODE_SCORE_ONLY = "score_only"
MODE_OVERLAY_ONLY = "overlay_only"
MODE_COMBINED = "combined"
MODES = frozenset((MODE_SCORE_ONLY, MODE_OVERLAY_ONLY, MODE_COMBINED))

FAST_CONDITIONS = (
    "vix_1d_spike",
    "vix_term_inversion",
    "nq_1d_drop",
    "move_1d_spike",
    "hy_oas_widening",
    "jpy_3d_drop",
    "breadth_1d_collapse",
    "portfolio_correlation",
)

_FAST_STATUSES = frozenset(("TRIGGERED", "CLEAR", "UNAVAILABLE"))
_FAST_QUALITIES = frozenset(("EXACT", "PUBLIC_LIVE", "UNAVAILABLE"))
_STATE_KEYS = frozenset(
    (
        "state_version",
        "policy_version",
        "mode",
        "last_as_of",
        "recommended_long",
        "raw_target_long",
        "policy_risk_score",
        "episode_active",
        "episode_started_as_of",
        "clear_streak",
    )
)


def _iso_day(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be an ISO date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an ISO date") from error
    if parsed.isoformat() != value:
        raise ValueError(f"{name} must be an ISO date")
    return value


def _integer(value: object, name: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer from {low} to {high}")
    return value


def long_anchor(risk_score: int) -> int:
    """Map a 0--100 RORO risk score to the frozen natural-number Long anchor."""

    score = _integer(risk_score, "risk_score", 0, 100)
    for (left_score, left_long), (right_score, right_long) in zip(
        LONG_ANCHORS, LONG_ANCHORS[1:]
    ):
        if score <= right_score:
            weight = (score - left_score) / (right_score - left_score)
            return math.floor(left_long + weight * (right_long - left_long) + 0.5)
    return LONG_ANCHORS[-1][1]


def _normalise_fast(snapshot: Mapping[str, object] | None, as_of: str) -> dict[str, Any]:
    if snapshot is None:
        return {
            "status": "INCOMPLETE",
            "trigger_count": 0,
            "available_count": 0,
            "primary_driver": "FAST DATA UNAVAILABLE",
            "active_triggers": [],
            "unavailable_conditions": list(FAST_CONDITIONS),
        }
    if not isinstance(snapshot, Mapping):
        raise ValueError("fast_snapshot must be a mapping or None")
    if _iso_day(snapshot.get("as_of"), "fast_snapshot.as_of") != as_of:
        raise ValueError("fast_snapshot.as_of must match the decision date")

    raw_conditions = snapshot.get("conditions")
    if not isinstance(raw_conditions, (list, tuple)) or len(raw_conditions) != FAST_TRIGGER_TOTAL:
        raise ValueError("fast_snapshot must contain exactly eight conditions")

    statuses: dict[str, str] = {}
    labels: dict[str, str] = {}
    for item in raw_conditions:
        if not isinstance(item, Mapping):
            raise ValueError("each Fast condition must be a mapping")
        key = item.get("key")
        label = item.get("label")
        status = item.get("status")
        quality = item.get("data_quality")
        if not isinstance(key, str) or key not in FAST_CONDITIONS or key in statuses:
            raise ValueError("Fast condition keys must match the frozen eight-condition contract")
        if not isinstance(label, str) or not label.strip():
            raise ValueError("Fast condition labels must be non-empty strings")
        if status not in _FAST_STATUSES or quality not in _FAST_QUALITIES:
            raise ValueError("Fast condition status or data_quality is invalid")
        if (status == "UNAVAILABLE") != (quality == "UNAVAILABLE"):
            raise ValueError("Fast condition status and data_quality disagree")
        condition_as_of = _iso_day(item.get("as_of"), f"Fast condition {key}.as_of")
        if condition_as_of > as_of:
            raise ValueError("Fast conditions cannot use a future date")
        statuses[key] = status
        labels[key] = label.strip()
    if set(statuses) != set(FAST_CONDITIONS):
        raise ValueError("Fast condition keys must match the frozen eight-condition contract")

    active = [key for key in FAST_CONDITIONS if statuses[key] == "TRIGGERED"]
    unavailable = [key for key in FAST_CONDITIONS if statuses[key] == "UNAVAILABLE"]
    available_count = FAST_TRIGGER_TOTAL - len(unavailable)
    complete = not unavailable
    derived_status = (
        "TRIGGERED"
        if len(active) >= FAST_TRIGGER_REQUIRED
        else "CLEAR"
        if complete
        else "INCOMPLETE"
    )
    expected_scalars = {
        "count": len(active),
        "trigger_count": len(active),
        "required": FAST_TRIGGER_REQUIRED,
        "total": FAST_TRIGGER_TOTAL,
        "complete": complete,
        "status": derived_status,
    }
    for key, expected in expected_scalars.items():
        if snapshot.get(key) != expected:
            raise ValueError(f"fast_snapshot.{key} disagrees with its conditions")
    source_driver = snapshot.get("primary_driver")
    if not isinstance(source_driver, str) or not source_driver.strip():
        raise ValueError("fast_snapshot.primary_driver must be a non-empty string")
    return {
        "status": derived_status,
        "trigger_count": len(active),
        "available_count": available_count,
        "primary_driver": source_driver.strip(),
        "active_triggers": active,
        "unavailable_conditions": unavailable,
    }


def _previous_state(value: object, *, as_of: str, mode: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != _STATE_KEYS:
        raise ValueError("previous_state must match the exact RORO policy-state contract")
    if value.get("state_version") != STATE_VERSION:
        raise ValueError("previous_state state version mismatch")
    if value.get("policy_version") != POLICY_VERSION:
        raise ValueError("previous_state policy version mismatch")
    if value.get("mode") != mode:
        raise ValueError("previous_state ablation mode mismatch")
    last_as_of = _iso_day(value.get("last_as_of"), "previous_state.last_as_of")
    if last_as_of >= as_of:
        raise ValueError("previous_state must precede the decision date")
    recommended = _integer(value.get("recommended_long"), "recommended_long", MIN_LONG, 100)
    raw = _integer(value.get("raw_target_long"), "raw_target_long", MIN_LONG, 100)
    if recommended > raw:
        raise ValueError("previous_state recommended_long cannot exceed raw_target_long")
    policy_score = _integer(value.get("policy_risk_score"), "policy_risk_score", 0, 100)
    episode_active = value.get("episode_active")
    if not isinstance(episode_active, bool):
        raise ValueError("previous_state episode_active must be boolean")
    started = value.get("episode_started_as_of")
    if episode_active:
        started = _iso_day(started, "previous_state.episode_started_as_of")
        if started > last_as_of:
            raise ValueError("previous_state episode starts after its last date")
    elif started is not None:
        raise ValueError("inactive previous_state cannot have an episode start")
    streak = _integer(value.get("clear_streak"), "clear_streak", 0, RECOVERY_CONFIRM_DAYS)
    if not episode_active and streak:
        raise ValueError("inactive previous_state cannot have a clear streak")
    return {
        "last_as_of": last_as_of,
        "recommended_long": recommended,
        "raw_target_long": raw,
        "policy_risk_score": policy_score,
        "episode_active": episode_active,
        "episode_started_as_of": started,
        "clear_streak": streak,
    }


def _state(
    *,
    as_of: str,
    mode: str,
    recommended_long: int,
    raw_target_long: int,
    policy_risk_score: int,
    episode_active: bool,
    episode_started_as_of: str | None,
    clear_streak: int,
) -> dict[str, object]:
    return {
        "state_version": STATE_VERSION,
        "policy_version": POLICY_VERSION,
        "mode": mode,
        "last_as_of": as_of,
        "recommended_long": recommended_long,
        "raw_target_long": raw_target_long,
        "policy_risk_score": policy_risk_score,
        "episode_active": episode_active,
        "episode_started_as_of": episode_started_as_of,
        "clear_streak": clear_streak,
    }


def decide_roro_policy(
    as_of: str,
    risk_score: int,
    fast_snapshot: Mapping[str, object] | None,
    previous_state: Mapping[str, object] | None = None,
    *,
    mode: str = MODE_COMBINED,
) -> dict[str, Any]:
    """Fold one decision day into a new immutable-by-contract policy-state row."""

    decision_date = _iso_day(as_of, "as_of")
    score = _integer(risk_score, "risk_score", 0, 100)
    if mode not in MODES:
        raise ValueError(f"mode must be one of {sorted(MODES)}")
    fast = _normalise_fast(fast_snapshot, decision_date)
    previous = _previous_state(previous_state, as_of=decision_date, mode=mode)

    use_score = mode != MODE_OVERLAY_ONLY
    use_fast = mode != MODE_SCORE_ONLY
    policy_score = score if use_score else 0
    base_long = long_anchor(policy_score)
    fast_triggered = use_fast and fast["trigger_count"] >= FAST_TRIGGER_REQUIRED
    adjustment = -FAST_CUT_PERCENTAGE_POINTS if fast_triggered else 0
    raw_target = max(MIN_LONG, base_long + adjustment)
    fast_recovery_ready = not use_fast or (
        not fast_triggered and fast["available_count"] >= FAST_AVAILABLE_FOR_RECOVERY
    )

    if previous is None:
        recommended = raw_target
        episode_active = fast_triggered
        episode_started = decision_date if episode_active else None
        clear_streak = 0
        phase = "COLD_START"
        primary_driver = fast["primary_driver"] if fast_triggered else "RORO SCORE"
    else:
        prior_long = previous["recommended_long"]
        worsening = policy_score > previous["policy_risk_score"]
        episode_active = previous["episode_active"]
        episode_started = previous["episode_started_as_of"]
        clear_streak = previous["clear_streak"]

        if raw_target < prior_long:
            recommended = raw_target
            episode_active = True
            episode_started = episode_started or decision_date
            clear_streak = 0
            phase = "RISK_REDUCTION"
            primary_driver = fast["primary_driver"] if fast_triggered else "RORO SCORE"
        else:
            increase_pending = raw_target > prior_long
            new_risk_event = fast_triggered or worsening
            if increase_pending or new_risk_event:
                episode_active = True
                episode_started = episode_started or decision_date
            if episode_active:
                clear_streak = (
                    min(clear_streak + 1, RECOVERY_CONFIRM_DAYS)
                    if fast_recovery_ready and not worsening
                    else 0
                )
            else:
                clear_streak = 0

            if increase_pending and clear_streak >= RECOVERY_CONFIRM_DAYS:
                recommended = raw_target
                episode_active = False
                episode_started = None
                clear_streak = 0
                phase = "REENTRY_DIRECT"
                primary_driver = "RECOVERY CONFIRMED"
            elif increase_pending:
                recommended = prior_long
                phase = "RECOVERY_HOLD"
                primary_driver = (
                    "FAST DATA BELOW 5/8"
                    if use_fast and fast["available_count"] < FAST_AVAILABLE_FOR_RECOVERY
                    else fast["primary_driver"]
                    if fast_triggered
                    else "3-DAY RECOVERY CONFIRMATION"
                )
            else:
                recommended = raw_target
                phase = "RECOVERY_CONFIRM" if episode_active else "AT_TARGET"
                primary_driver = (
                    fast["primary_driver"] if fast_triggered else "RORO SCORE"
                )

    policy_state = _state(
        as_of=decision_date,
        mode=mode,
        recommended_long=recommended,
        raw_target_long=raw_target,
        policy_risk_score=policy_score,
        episode_active=episode_active,
        episode_started_as_of=episode_started,
        clear_streak=clear_streak,
    )
    return {
        "model_version": POLICY_VERSION,
        "as_of": decision_date,
        "mode": mode,
        "risk_score": score,
        "base_long": base_long,
        "raw_target_long": raw_target,
        "fast_adjustment": adjustment,
        "recommended_long": recommended,
        "phase": phase,
        "episode_status": "ACTIVE" if episode_active else "INACTIVE",
        "clear_streak": clear_streak,
        "fast_status": fast["status"],
        "fast_trigger_count": fast["trigger_count"],
        "fast_available_count": fast["available_count"],
        "fast_overlay_applied": fast_triggered,
        "active_fast_conditions": list(fast["active_triggers"]),
        "unavailable_fast_conditions": list(fast["unavailable_conditions"]),
        "primary_driver": primary_driver,
        "policy_state": policy_state,
    }


def replay_roro_policy(
    decisions: Iterable[Mapping[str, object]],
    *,
    mode: str = MODE_COMBINED,
    initial_state: Mapping[str, object] | None = None,
) -> list[dict[str, Any]]:
    """Replay score-only, overlay-only, or combined with the same reducer."""

    state = initial_state
    results: list[dict[str, Any]] = []
    for decision in decisions:
        if not isinstance(decision, Mapping):
            raise ValueError("each replay decision must be a mapping")
        result = decide_roro_policy(
            decision.get("as_of"),
            decision.get("risk_score"),
            decision.get("fast_snapshot"),
            state,
            mode=mode,
        )
        results.append(result)
        state = result["policy_state"]
    return results
