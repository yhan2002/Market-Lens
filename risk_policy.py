"""Pure router-state policy engine for long-only exposure advisories."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from datetime import date
from typing import Any


POLICY_VERSION = "risk-policy-v5.1-posterior-weighted-candidate"
RULE_SCORE_POLICY_VERSION = "risk-policy-v6-nq-rule-score-shadow"
RULE_STRESS_SCORE_POLICY_VERSION = "risk-policy-v7-nq-rule-stress-score-shadow"
RULE_STRESS_SCORE_V3_POLICY_VERSION = (
    "risk-policy-v8-nq-rule-stress-score-exposure-v3-shadow"
)
RULE_STRESS_SCORE_V3_LIVE_POLICY_VERSION = (
    "risk-policy-v8.1-nq-rule-stress-score-exposure-v3-live"
)
_RULE_SCORE_POLICY_VERSIONS = {
    RULE_SCORE_POLICY_VERSION,
    RULE_STRESS_SCORE_POLICY_VERSION,
    RULE_STRESS_SCORE_V3_POLICY_VERSION,
    RULE_STRESS_SCORE_V3_LIVE_POLICY_VERSION,
}
_RULE_STRESS_POLICY_VERSIONS = {
    RULE_STRESS_SCORE_POLICY_VERSION,
    RULE_STRESS_SCORE_V3_POLICY_VERSION,
    RULE_STRESS_SCORE_V3_LIVE_POLICY_VERSION,
}
POLICY_STATE_VERSION = 1
REENTRY_LADDER = [0.25, 0.40, 0.60, 0.80, 1.00]
RISK_OFF_PEAK_MIN = 0.40
RISK_OFF_RETREAT_MIN = 0.05
FAST_TRIGGER_REQUIRED = 3
FAST_TRIGGER_TOTAL = 8
FAST_CUT = 0.20
FAST_AVAILABLE_FOR_RECOVERY = 5
RECOVERY_CONFIRM_DAYS = 3

REGIME_STATE_SCORES = {
    "BULL": 10,
    "SIDEWAYS": 50,
    "BEAR": 80,
}

_EXPOSURE_ANCHORS = (
    (0, 100),
    (20, 90),
    (40, 75),
    (60, 60),
    (80, 40),
    (100, 25),
)
_V3_EXPOSURE_ANCHORS = (
    (0, 100),
    (20, 100),
    (40, 90),
    (60, 80),
    (80, 40),
    (100, 25),
)

_FAST_STATUSES = {"TRIGGERED", "CLEAR", "UNAVAILABLE"}
_REENTRY_STATUSES = {"CONFIRMED", "NOT_CONFIRMED", "UNAVAILABLE"}
_DATA_QUALITY = {"EXACT", "PUBLIC_LIVE", "UNAVAILABLE"}
_FAST_CHANNELS = (
    ("vix_1d_spike", "VIX Daily Jump"),
    ("vix_term_inversion", "VIX Backwardation"),
    ("nq_1d_drop", "NQ Daily Loss"),
    ("move_1d_spike", "MOVE Daily Jump"),
    ("hy_oas_widening", "HY Spread Jump"),
    ("jpy_3d_drop", "USDJPY 3-Day Drop"),
    ("breadth_1d_collapse", "Breadth Shock"),
    ("portfolio_correlation", "Portfolio Correlation"),
)
_REENTRY_CHANNELS = (
    ("vix_term_structure", "VIX Curve Normalized"),
    ("hy_spread_stabilized", "HY Spread Stabilized"),
    ("nq_above_5dma", "NQ Above 5DMA"),
    ("breadth_divergence", "Breadth Divergence"),
)


def _probability(value: object, name: str, *, optional: bool = False) -> float | None:
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return result


def _probabilities(
    p_bull: object, p_sideways: object, p_bear: object
) -> tuple[float, float, float]:
    values = (
        _probability(p_bull, "p_bull"),
        _probability(p_sideways, "p_sideways"),
        _probability(p_bear, "p_bear"),
    )
    assert all(value is not None for value in values)
    bull, sideways, bear = values
    if not math.isclose(bull + sideways + bear, 1.0, abs_tol=1e-6):
        raise ValueError("p_bull, p_sideways, and p_bear must sum to 1")
    return bull, sideways, bear


def _round_half_up(value: float) -> int:
    return math.floor(value + 0.5)


def _regime_state(
    p_bull: float,
    p_sideways: float,
    p_bear: float,
    stated: object = None,
) -> str:
    probabilities = {"BULL": p_bull, "SIDEWAYS": p_sideways, "BEAR": p_bear}
    inferred = max(probabilities, key=probabilities.get)
    if stated is not None:
        state = str(stated).strip().upper()
        if state != inferred:
            raise ValueError("regime.state must match the largest HMM posterior")
    return inferred


def _exposure_from_score(score: int, *, policy_version: str = POLICY_VERSION) -> float:
    anchors = (
        _V3_EXPOSURE_ANCHORS
        if policy_version
        in {
            RULE_STRESS_SCORE_V3_POLICY_VERSION,
            RULE_STRESS_SCORE_V3_LIVE_POLICY_VERSION,
        }
        else _EXPOSURE_ANCHORS
    )
    for (left_score, left_exposure), (right_score, right_exposure) in zip(
        anchors, anchors[1:]
    ):
        if score <= right_score:
            weight = (score - left_score) / (right_score - left_score)
            target = left_exposure + weight * (right_exposure - left_exposure)
            return _round_half_up(target) / 100
    return anchors[-1][1] / 100


def _weighted_risk_score(p_bull: float, p_sideways: float, p_bear: float) -> int:
    return _round_half_up(
        p_bull * REGIME_STATE_SCORES["BULL"]
        + p_sideways * REGIME_STATE_SCORES["SIDEWAYS"]
        + p_bear * REGIME_STATE_SCORES["BEAR"]
    )


def _score_override(value: object, policy_version: str) -> int | None:
    if value is None:
        if policy_version != POLICY_VERSION:
            raise ValueError("a rule-score policy requires its explicit risk score")
        return None
    if policy_version not in _RULE_SCORE_POLICY_VERSIONS:
        raise ValueError("a score override requires a supported rule-score policy version")
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 100:
        raise ValueError("slow_risk_score must be an integer from 0 to 100")
    return value


def _next_reentry_rung(previous: float) -> float:
    return next((rung for rung in REENTRY_LADDER if rung > previous), 1.0)


def exposure_from_probabilities(
    p_bull: float, p_sideways: float, p_bear: float
) -> float:
    """Map the full HMM posterior to an integer-percentage exposure."""
    bull, sideways, bear = _probabilities(p_bull, p_sideways, p_bear)
    return _exposure_from_score(_weighted_risk_score(bull, sideways, bear))


def _conditions(
    conditions: Iterable[Mapping[str, object]],
    *,
    expected: int,
    statuses: set[str],
    kind: str,
) -> list[dict[str, str]]:
    checked: list[dict[str, str]] = []
    keys: set[str] = set()
    for condition in conditions:
        if not isinstance(condition, Mapping):
            raise ValueError(f"each {kind} condition must be a mapping")
        key = str(condition.get("key", "")).strip()
        label = str(condition.get("label", "")).strip()
        status = str(condition.get("status", "")).strip().upper()
        detail = str(condition.get("detail", "")).strip()
        if not key or not label or status not in statuses or key in keys:
            raise ValueError(f"invalid or duplicate {kind} condition")
        keys.add(key)
        checked.append({"key": key, "label": label, "status": status, "detail": detail})
    if len(checked) != expected:
        raise ValueError(f"exactly {expected} unique {kind} conditions are required")
    return checked


def _normalise_fast_overlay(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("fast_overlay must be a mapping")
    raw_conditions = value.get("conditions")
    if not isinstance(raw_conditions, (list, tuple)):
        raise ValueError("fast_overlay.conditions must be a list")
    conditions = _conditions(
        raw_conditions,
        expected=FAST_TRIGGER_TOTAL,
        statuses=_FAST_STATUSES,
        kind="fast-overlay",
    )
    if {item["key"] for item in conditions} != {key for key, _ in _FAST_CHANNELS}:
        raise ValueError("fast-overlay conditions must match the PDF channels")
    source_by_key = {str(item.get("key", "")).strip(): item for item in raw_conditions}
    for condition in conditions:
        source = source_by_key[condition["key"]]
        quality = str(source.get("data_quality", "")).strip().upper()
        if quality not in _DATA_QUALITY:
            raise ValueError("invalid fast-overlay data_quality")
        if (condition["status"] == "UNAVAILABLE") != (quality == "UNAVAILABLE"):
            raise ValueError("fast-overlay status and data_quality disagree")
        condition["data_quality"] = quality

    trigger_count = sum(item["status"] == "TRIGGERED" for item in conditions)
    complete = all(item["status"] != "UNAVAILABLE" for item in conditions)
    status = (
        "TRIGGERED"
        if trigger_count >= FAST_TRIGGER_REQUIRED
        else "CLEAR"
        if complete
        else "INCOMPLETE"
    )
    return {
        "status": status,
        "complete": complete,
        "trigger_count": trigger_count,
        "required": FAST_TRIGGER_REQUIRED,
        "total": FAST_TRIGGER_TOTAL,
        "conditions": conditions,
    }


def _unavailable_fast_overlay() -> dict[str, Any]:
    return {
        "status": "INCOMPLETE",
        "complete": False,
        "trigger_count": 0,
        "required": FAST_TRIGGER_REQUIRED,
        "total": FAST_TRIGGER_TOTAL,
        "conditions": [
            {
                "key": key,
                "label": label,
                "status": "UNAVAILABLE",
                "data_quality": "UNAVAILABLE",
                "detail": "",
                "data_quality": "UNAVAILABLE",
            }
            for key, label in _FAST_CHANNELS
        ],
    }


def evaluate_reentry(
    current_p_bear: float | None,
    previous_p_bear: float | None,
    conditions: Iterable[Mapping[str, object]],
    *,
    recent_peak_p_bear: float | None = None,
) -> dict[str, Any]:
    """Evaluate the PDF 3-of-4 gate after a meaningful Risk-Off peak."""
    raw_conditions = list(conditions)
    current = _probability(current_p_bear, "current_p_bear", optional=True)
    previous = _probability(previous_p_bear, "previous_p_bear", optional=True)
    recent_peak = _probability(
        recent_peak_p_bear, "recent_peak_p_bear", optional=True
    )
    checked = _conditions(
        raw_conditions,
        expected=4,
        statuses=_REENTRY_STATUSES,
        kind="re-entry",
    )
    if {item["key"] for item in checked} != {key for key, _ in _REENTRY_CHANNELS}:
        raise ValueError("re-entry conditions must match the PDF channels")
    source_by_key = {str(item.get("key", "")).strip(): item for item in raw_conditions}
    for condition in checked:
        source = source_by_key[condition["key"]]
        quality = str(source.get("data_quality", "")).strip().upper()
        if quality not in _DATA_QUALITY or (
            (condition["status"] == "UNAVAILABLE") != (quality == "UNAVAILABLE")
        ):
            raise ValueError("re-entry status and data_quality disagree")
        condition["data_quality"] = quality
    has_history = current is not None and previous is not None and recent_peak is not None
    falling = current < previous if has_history else None
    delta = current - previous if has_history else None
    peak_seen = recent_peak >= RISK_OFF_PEAK_MIN if has_history else None
    retreated = recent_peak - current >= RISK_OFF_RETREAT_MIN if has_history else None
    confirmed = sum(item["status"] == "CONFIRMED" for item in checked)
    complete = all(item["status"] != "UNAVAILABLE" for item in checked)

    if not has_history:
        status = "NO_HISTORY"
    elif not peak_seen:
        status = "WAIT_RISK_OFF_PEAK"
    elif not falling or not retreated:
        status = "WAIT_RISK_OFF_TURN"
    elif not complete or confirmed < 3:
        status = "WAIT_CONFIRMATIONS"
    else:
        status = "ELIGIBLE_ONE_STEP"

    return {
        "status": status,
        "current_p_bear": current,
        "previous_p_bear": previous,
        "recent_peak_p_bear": recent_peak,
        "p_bear_falling": falling,
        "delta_p_bear": delta,
        "risk_off_peak_seen": peak_seen,
        "risk_off_peak_min": RISK_OFF_PEAK_MIN,
        "confirmed": confirmed,
        "required": 3,
        "total": 4,
        "conditions": checked,
        "ladder": REENTRY_LADDER.copy(),
        "advisory_only": True,
    }


def _unevaluated_reentry(status: str = "NOT_EVALUATED") -> dict[str, Any]:
    return {
        "status": status,
        "current_p_bear": None,
        "previous_p_bear": None,
        "recent_peak_p_bear": None,
        "p_bear_falling": None,
        "delta_p_bear": None,
        "risk_off_peak_seen": None,
        "risk_off_peak_min": RISK_OFF_PEAK_MIN,
        "confirmed": 0,
        "required": 3,
        "total": 4,
        "conditions": [
            {
                "key": key,
                "label": label,
                "status": "UNAVAILABLE",
                "detail": "",
            }
            for key, label in _REENTRY_CHANNELS
        ],
        "ladder": REENTRY_LADDER.copy(),
        "advisory_only": True,
    }


def _normalise_reentry(
    value: object, *, episode_peak_p_bear: float | None = None
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return _unevaluated_reentry()
    raw_conditions = value.get("conditions")
    if not isinstance(raw_conditions, (list, tuple)):
        return _unevaluated_reentry("INCOMPLETE")
    try:
        current = _probability(value.get("current_p_bear"), "current_p_bear", optional=True)
        previous = _probability(
            value.get("previous_p_bear"), "previous_p_bear", optional=True
        )
        peak = _probability(
            episode_peak_p_bear
            if episode_peak_p_bear is not None
            else value.get("recent_peak_p_bear"),
            "recent_peak_p_bear",
            optional=True,
        )
    except ValueError:
        return _unevaluated_reentry("INCOMPLETE")
    try:
        derived = evaluate_reentry(
            current,
            previous,
            raw_conditions,
            recent_peak_p_bear=peak,
        )
    except ValueError:
        return _unevaluated_reentry("INCOMPLETE")
    # The publisher uses this marker to make same-day reruns idempotent.  It is
    # accepted only when the underlying inputs independently pass the gate.
    if (
        str(value.get("status", "")).strip().upper() == "WAIT_NEXT_TRADING_DAY"
        and derived["status"] == "ELIGIBLE_ONE_STEP"
    ):
        derived["status"] = "WAIT_NEXT_TRADING_DAY"
    return derived


def _exposure(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("exposure must be a 1%-grid value from 0.25 to 1.00")
    exposure = float(value)
    percent = round(exposure * 100)
    if (
        not math.isfinite(exposure)
        or not 25 <= percent <= 100
        or not math.isclose(exposure * 100, percent, abs_tol=1e-9)
    ):
        raise ValueError("exposure must be a 1%-grid value from 0.25 to 1.00")
    return percent / 100


def _iso_date(value: object, name: str) -> str:
    text = str(value)
    try:
        parsed = date.fromisoformat(text)
    except ValueError as error:
        raise ValueError(f"{name} must be an ISO date") from error
    if parsed.isoformat() != text:
        raise ValueError(f"{name} must be an ISO date")
    return text


def _regime_model_version(regime: Mapping[str, object]) -> str:
    value = regime.get("model_version")
    if not isinstance(value, str) or not value.strip() or len(value) > 200:
        raise ValueError("regime.model_version must be a non-empty string")
    return value.strip()


def _previous_state(
    value: object,
    *,
    as_of: str,
    regime_model_version: str,
    policy_version: str = POLICY_VERSION,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("previous_state must be a mapping")
    if value.get("state_version") != POLICY_STATE_VERSION:
        raise ValueError("previous_state version mismatch")
    if value.get("policy_version") != policy_version:
        raise ValueError("previous_state policy version mismatch")
    if value.get("regime_model_version") != regime_model_version:
        raise ValueError("previous_state regime model version mismatch")
    last_as_of = _iso_date(value.get("last_as_of"), "previous_state.last_as_of")
    if last_as_of > as_of:
        raise ValueError("previous_state cannot be from the future")
    exposure = _exposure(value.get("exposure"))
    raw_target = _exposure(value.get("raw_target"))
    streak = value.get("recovery_clear_streak")
    if isinstance(streak, bool) or not isinstance(streak, int) or not 0 <= streak <= 10_000:
        raise ValueError("previous_state recovery streak is invalid")
    peak = _probability(
        value.get("episode_peak_p_bear"),
        "previous_state.episode_peak_p_bear",
        optional=True,
    )
    started = value.get("episode_started_as_of")
    if started is not None:
        started = _iso_date(started, "previous_state.episode_started_as_of")
        if started > last_as_of:
            raise ValueError("previous_state episode starts after its last date")
    if (peak is None) != (started is None):
        raise ValueError("previous_state episode fields disagree")
    return {
        "state_version": POLICY_STATE_VERSION,
        "policy_version": policy_version,
        "regime_model_version": regime_model_version,
        "last_as_of": last_as_of,
        "exposure": exposure,
        "raw_target": raw_target,
        "recovery_clear_streak": streak,
        "episode_peak_p_bear": peak,
        "episode_started_as_of": started,
    }


def _policy_state(
    *,
    as_of: str,
    regime_model_version: str,
    exposure: float,
    raw_target: float,
    recovery_clear_streak: int,
    episode_peak_p_bear: float | None,
    episode_started_as_of: str | None,
    policy_version: str = POLICY_VERSION,
) -> dict[str, Any]:
    return {
        "state_version": POLICY_STATE_VERSION,
        "policy_version": policy_version,
        "regime_model_version": regime_model_version,
        "last_as_of": as_of,
        "exposure": exposure,
        "raw_target": raw_target,
        "recovery_clear_streak": recovery_clear_streak,
        "episode_peak_p_bear": episode_peak_p_bear,
        "episode_started_as_of": episode_started_as_of,
    }


def _no_decision(
    missing_channels: list[str],
    fast_overlay: Mapping[str, object] | None,
    reentry: Mapping[str, object],
    previous_state: Mapping[str, object] | None = None,
    policy_version: str = POLICY_VERSION,
) -> dict[str, Any]:
    overlay = dict(fast_overlay) if fast_overlay is not None else _unavailable_fast_overlay()
    conditions = overlay.get("conditions")
    available_count = sum(
        1
        for condition in conditions
        if isinstance(condition, Mapping) and condition.get("status") != "UNAVAILABLE"
    ) if isinstance(conditions, list) else 0
    return {
        "model_version": policy_version,
        "decision_status": "NO_DECISION",
        "slow_risk_score": None,
        "risk_score": None,
        "base_exposure": None,
        "fast_adjustment": None,
        "raw_recommended_exposure": None,
        "previous_recommended_exposure": None,
        "recommended_exposure": None,
        "risk_band": None,
        "primary_driver_code": "NO_DECISION",
        "primary_driver": "INPUT CHECK",
        "active_triggers": [],
        "missing_channels": missing_channels,
        "fast_available_count": available_count,
        "recovery_phase": "NO_DECISION",
        "fast_overlay": overlay,
        "reentry": dict(reentry),
        "policy_state": dict(previous_state) if previous_state is not None else None,
        "advisory_only": True,
    }


def step_risk_policy(
    as_of: str,
    regime: Mapping[str, object] | None,
    fast_overlay: Mapping[str, object] | None,
    reentry: Mapping[str, object] | None,
    *,
    previous_state: Mapping[str, object] | None = None,
    slow_risk_score: int | None = None,
    policy_version: str = POLICY_VERSION,
) -> dict[str, Any]:
    """Apply one deterministic policy transition without performing I/O."""
    override = _score_override(slow_risk_score, policy_version)
    try:
        decision_date = _iso_date(as_of, "as_of")
    except ValueError:
        return _no_decision(
            ["AS_OF"], None, _unevaluated_reentry(), policy_version=policy_version
        )

    normalised_reentry = _normalise_reentry(reentry)
    missing: list[str] = []
    probabilities: tuple[float, float, float] | None = None
    regime_state: str | None = None
    regime_version: str | None = None
    normalised_fast: dict[str, Any] | None = None
    previous: dict[str, Any] | None = None

    try:
        if not isinstance(regime, Mapping):
            raise ValueError("regime must be a mapping")
        probabilities = _probabilities(
            regime.get("p_bull"), regime.get("p_sideways"), regime.get("p_bear")
        )
        regime_state = _regime_state(*probabilities, stated=regime.get("state"))
        regime_version = _regime_model_version(regime)
        regime_date = regime.get("date")
        if regime_date is not None and _iso_date(regime_date, "regime.date") != decision_date:
            raise ValueError("regime.date must match as_of")
    except ValueError:
        missing.append("HMM_REGIME")
    try:
        normalised_fast = _normalise_fast_overlay(fast_overlay)
    except ValueError:
        missing.append("FAST_OVERLAY")
    try:
        if regime_version is None:
            raise ValueError("regime version unavailable")
        previous = _previous_state(
            previous_state,
            as_of=decision_date,
            regime_model_version=regime_version,
            policy_version=policy_version,
        )
    except ValueError:
        missing.append("PREVIOUS_STATE")
    if missing:
        return _no_decision(
            missing,
            normalised_fast,
            normalised_reentry,
            previous,
            policy_version=policy_version,
        )

    assert (
        probabilities is not None
        and regime_state is not None
        and regime_version is not None
        and normalised_fast is not None
    )
    bull, sideways, bear = probabilities
    slow_score = override if override is not None else _weighted_risk_score(bull, sideways, bear)
    base = _exposure_from_score(slow_score, policy_version=policy_version)
    triggered = normalised_fast["trigger_count"] >= FAST_TRIGGER_REQUIRED
    adjustment = -FAST_CUT if triggered else 0.0
    raw = max(0.25, round(base + adjustment, 2))
    risk_score = slow_score
    active_triggers = [
        item["key"]
        for item in normalised_fast["conditions"]
        if item["status"] == "TRIGGERED"
    ]
    unavailable = [
        item["key"]
        for item in normalised_fast["conditions"]
        if item["status"] == "UNAVAILABLE"
    ]
    available_count = FAST_TRIGGER_TOTAL - len(unavailable)
    fast_recovery_ready = (
        not triggered and available_count >= FAST_AVAILABLE_FOR_RECOVERY
    )

    maximum_target = (
        0.90
        if policy_version in _RULE_STRESS_POLICY_VERSIONS
        else _exposure_from_score(
            0 if override is not None else REGIME_STATE_SCORES["BULL"]
        )
    )
    prior_exposure = previous["exposure"] if previous is not None else None
    prior_raw = previous["raw_target"] if previous is not None else None
    episode_peak = previous["episode_peak_p_bear"] if previous is not None else None
    episode_started = previous["episode_started_as_of"] if previous is not None else None
    clear_streak = previous["recovery_clear_streak"] if previous is not None else 0

    if previous is None:
        recommended = raw
        phase = "COLD_START"
        driver_code, driver = (
            ("FAST_RISK_OVERLAY", "FAST RISK OVERLAY")
            if triggered
            else ("REGIME_SEVERITY", "REGIME SEVERITY")
        )
        if raw < maximum_target:
            episode_peak = bear
            episode_started = decision_date
        clear_streak = 0
    else:
        same_day = previous["last_as_of"] == decision_date
        if episode_peak is not None:
            episode_peak = max(episode_peak, bear)

        if raw < prior_exposure:
            recommended = raw
            phase = "RISK_REDUCTION"
            driver_code, driver = (
                ("FAST_RISK_OVERLAY", "FAST RISK OVERLAY")
                if triggered
                else ("REGIME_SEVERITY", "REGIME SEVERITY")
            )
            episode_peak = max(episode_peak if episode_peak is not None else bear, bear)
            episode_started = episode_started or decision_date
            clear_streak = 0
        elif raw <= prior_exposure:
            recommended = raw
            phase = "AT_TARGET"
            driver_code, driver = (
                ("FAST_RISK_OVERLAY", "FAST RISK OVERLAY")
                if triggered
                else ("REGIME_SEVERITY", "REGIME SEVERITY")
            )
            clear_streak = 0
            if raw < maximum_target and episode_peak is None:
                episode_peak = bear
                episode_started = decision_date
        elif (
            policy_version in _RULE_STRESS_POLICY_VERSIONS
            and episode_peak is None
            and episode_started is None
        ):
            recommended = raw
            phase = "AT_TARGET"
            driver_code, driver = "REGIME_SEVERITY", "REGIME SEVERITY"
            clear_streak = 0
        else:
            reentry_bear = normalised_reentry.get("current_p_bear")
            if reentry_bear is not None and not math.isclose(
                float(reentry_bear), bear, abs_tol=1e-9
            ):
                return _no_decision(
                    ["REENTRY"],
                    normalised_fast,
                    normalised_reentry,
                    previous,
                    policy_version=policy_version,
                )
            if episode_peak is None:
                episode_peak = bear
                episode_started = decision_date
            normalised_reentry = _normalise_reentry(
                reentry, episode_peak_p_bear=episode_peak
            )
            target_stable = (
                raw >= prior_raw - 1e-9
                if policy_version in _RULE_STRESS_POLICY_VERSIONS
                else math.isclose(raw, prior_raw, abs_tol=1e-9)
            )
            if not same_day and fast_recovery_ready and target_stable:
                clear_streak += 1
            elif not same_day and fast_recovery_ready:
                clear_streak = 1
            else:
                clear_streak = 0

            bottom_fishing_ready = (
                not same_day
                and not triggered
                and normalised_reentry["status"] == "ELIGIBLE_ONE_STEP"
            )
            normal_recovery_ready = (
                not same_day and clear_streak >= RECOVERY_CONFIRM_DAYS
            )
            if bottom_fishing_ready or normal_recovery_ready:
                recommended = min(raw, _next_reentry_rung(prior_exposure))
                phase = "RECOVERY_STEP"
                if bottom_fishing_ready:
                    driver_code, driver = "BOTTOM_FISHING", "BOTTOM FISHING STEP"
                else:
                    driver_code, driver = "STEPWISE_REENTRY", "STEPWISE RE-ENTRY"
            else:
                recommended = prior_exposure
                phase = "RECOVERY_WAIT"
                driver_code, driver = "RECOVERY_HOLD", "RECOVERY CONFIRMATION"

    episode_complete = (
        not triggered
        and math.isclose(recommended, raw, abs_tol=1e-9)
        and raw >= maximum_target
        if policy_version in _RULE_STRESS_POLICY_VERSIONS
        else recommended >= maximum_target and raw >= maximum_target
    )
    if episode_complete:
        episode_peak = None
        episode_started = None
        clear_streak = 0

    next_state = _policy_state(
        as_of=decision_date,
        regime_model_version=regime_version,
        exposure=recommended,
        raw_target=raw,
        recovery_clear_streak=clear_streak,
        episode_peak_p_bear=episode_peak,
        episode_started_as_of=episode_started,
        policy_version=policy_version,
    )

    risk_band = "GREEN" if recommended >= 0.90 else "YELLOW" if recommended >= 0.60 else "RED"
    return {
        "model_version": policy_version,
        "decision_status": "OK",
        "slow_risk_score": slow_score,
        "risk_score": risk_score,
        "base_exposure": base,
        "fast_adjustment": adjustment,
        "raw_recommended_exposure": raw,
        "previous_recommended_exposure": prior_exposure,
        "recommended_exposure": recommended,
        "risk_band": risk_band,
        "primary_driver_code": driver_code,
        "primary_driver": driver,
        "active_triggers": active_triggers,
        "missing_channels": unavailable,
        "fast_available_count": available_count,
        "recovery_phase": phase,
        "fast_overlay": normalised_fast,
        "reentry": normalised_reentry,
        "policy_state": next_state,
        "advisory_only": True,
    }


def evaluate_risk_policy(
    regime: Mapping[str, object] | None,
    fast_overlay: Mapping[str, object] | None,
    reentry: Mapping[str, object] | None,
    *,
    as_of: str,
    previous_state: Mapping[str, object] | None = None,
    slow_risk_score: int | None = None,
    policy_version: str = POLICY_VERSION,
) -> dict[str, Any]:
    """Compatibility name for the single production policy transition."""
    return step_risk_policy(
        as_of,
        regime,
        fast_overlay,
        reentry,
        previous_state=previous_state,
        slow_risk_score=slow_risk_score,
        policy_version=policy_version,
    )
