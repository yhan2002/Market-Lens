export type RegimeSnapshot = {
  date: string;
  state: "BULL" | "SIDEWAYS" | "BEAR";
  model_version: string;
  p_bull: number;
  p_sideways: number;
  p_bear: number;
};

export type RiskSnapshot = {
  date: string;
  model_version: string;
  risk_band: string;
  score: number | null;
  score_1d?: number | null;
  score_5d?: number | null;
  at_risk?: boolean;
  score_is_probability?: boolean;
  vix_level_rank?: number | null;
  vix_acceleration_rank?: number | null;
  credit_risk_rank?: number | null;
};

export type SectorRow = {
  ticker: string;
  sector: string;
  quadrant: string;
  quadrant_transition: string;
  tail_heading: string;
  rs_ratio: number;
  rs_momentum: number;
  delta_rs_ratio: number;
  delta_rs_momentum: number;
  shadow_score: number;
  forecast_rank: number;
  forecast_score: number;
  predicted_excess_20d: number;
  signal: string;
};

export type ReentryCondition = {
  key: string;
  label: string;
  status: "CONFIRMED" | "NOT_CONFIRMED" | "UNAVAILABLE";
  data_quality?: "EXACT" | "PUBLIC_LIVE" | "UNAVAILABLE";
  detail: string;
};

export type ReentryPolicy = {
  status: string;
  current_p_bear?: number | null;
  previous_p_bear?: number | null;
  recent_peak_p_bear?: number | null;
  risk_off_peak_seen?: boolean | null;
  risk_off_peak_min?: number;
  confirmed: number;
  required: number;
  total: number;
  p_bear_falling: boolean | null;
  delta_p_bear: number | null;
  ladder: number[];
  conditions: ReentryCondition[];
  advisory_only: boolean;
};

export type LegacyRiskPolicySnapshot = {
  model_version: string;
  decision_status: "OK" | "NO_DECISION";
  market_health_score: number | null;
  health_band: "GREEN" | "YELLOW" | "RED" | null;
  base_exposure: number | null;
  hmm_cap: number | null;
  fast_overlay_cap: number | null;
  raw_recommended_exposure?: number | null;
  previous_recommended_exposure?: number | null;
  recommended_exposure: number | null;
  primary_driver_code: string;
  primary_driver: string;
  active_triggers: string[];
  missing_channels?: string[];
  score_method: string;
  score_coverage: string | Record<string, string | number | boolean | null | string[]>;
  advisory_only: boolean;
  reentry: ReentryPolicy;
};

export type FastOverlayCondition = {
  key: string;
  label: string;
  status: "TRIGGERED" | "CLEAR" | "UNAVAILABLE";
  data_quality: "EXACT" | "PUBLIC_LIVE" | "UNAVAILABLE";
  detail?: string;
};

export type FastOverlayPolicy = {
  status: "TRIGGERED" | "CLEAR" | "INCOMPLETE";
  trigger_count: number;
  required: 3;
  total: 8;
  complete: boolean;
  conditions: FastOverlayCondition[];
};

export type RiskPolicySnapshotV2 = {
  model_version: string;
  decision_status: "OK" | "NO_DECISION";
  slow_risk_score: number | null;
  risk_score: number | null;
  risk_score_delta?: number | null;
  risk_trend?: "UP" | "DOWN" | "FLAT" | null;
  base_exposure: number | null;
  fast_adjustment: number | null;
  raw_recommended_exposure: number | null;
  previous_recommended_exposure: number | null;
  recommended_exposure: number | null;
  risk_band: "GREEN" | "YELLOW" | "RED" | null;
  primary_driver_code: string;
  primary_driver: string;
  active_triggers: string[];
  missing_channels: string[];
  fast_overlay: FastOverlayPolicy;
  reentry: ReentryPolicy;
  advisory_only: boolean;
};

export type PolicyState = {
  state_version: 1;
  policy_version: string;
  regime_model_version: string;
  last_as_of: string;
  exposure: number;
  raw_target: number;
  recovery_clear_streak: number;
  episode_peak_p_bear: number | null;
  episode_started_as_of: string | null;
};

export type RiskPolicySnapshotV3 = RiskPolicySnapshotV2 & {
  fast_available_count: number;
  recovery_phase: "COLD_START" | "RISK_REDUCTION" | "AT_TARGET" | "RECOVERY_WAIT" | "RECOVERY_STEP" | "NO_DECISION";
  policy_state: PolicyState | null;
};

export type RiskPolicySnapshot = LegacyRiskPolicySnapshot | RiskPolicySnapshotV2 | RiskPolicySnapshotV3;

type DashboardBase = {
  generated_at: string;
  as_of: string;
  regime: RegimeSnapshot;
  crash_guard: RiskSnapshot | null;
  drawdown_onset: RiskSnapshot | null;
  sector: {
    rrg_as_of: string;
    forecast_as_of: string;
    target: string;
    training_weeks: number;
    rows: SectorRow[];
  };
  notices: string[];
};

export type DashboardSnapshotV1 = DashboardBase & {
  schema_version: 1;
  risk_policy?: LegacyRiskPolicySnapshot;
};

export type DashboardSnapshotV2 = DashboardBase & {
  schema_version: 2;
  risk_policy: RiskPolicySnapshotV2;
};

export type DashboardSnapshotV3 = DashboardBase & {
  schema_version: 3;
  history: {
    series_version: string;
    build_mode: "LIVE" | "BACKFILL";
    seed_mode: "STRICT_PRIOR" | "FULL_REPLAY";
    parent_as_of: string | null;
  };
  risk_policy: RiskPolicySnapshotV3;
};

export type DashboardSnapshot = DashboardSnapshotV1 | DashboardSnapshotV2 | DashboardSnapshotV3;

function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}

function hasOnlyKeys(value: Record<string, unknown>, allowed: readonly string[]) {
  return Object.keys(value).every((key) => allowed.includes(key));
}

function isShortString(value: unknown, max = 500): value is string {
  return typeof value === "string" && value.length <= max;
}

function isIsoDate(value: unknown): value is string {
  if (typeof value !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(value)) return false;
  const parsed = new Date(`${value}T00:00:00Z`);
  return !Number.isNaN(parsed.valueOf()) && parsed.toISOString().slice(0, 10) === value;
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isNumberInRange(value: unknown, min: number, max: number): value is number {
  return isFiniteNumber(value) && value >= min && value <= max;
}

function isIntegerInRange(value: unknown, min: number, max: number): value is number {
  return Number.isInteger(value) && isNumberInRange(value, min, max);
}

function isPercentGrid(value: unknown, min = 0.25, max = 1): value is number {
  return isNumberInRange(value, min, max) && Math.abs(value * 100 - Math.round(value * 100)) < 1e-8;
}

function isNullablePercentGrid(value: unknown, min = 0.25, max = 1): value is number | null {
  return value === null || isPercentGrid(value, min, max);
}

function isNullableNumberInRange(value: unknown, min: number, max: number): value is number | null {
  return value === null || isNumberInRange(value, min, max);
}

function isStringList(value: unknown, maxItems = 100): value is string[] {
  return Array.isArray(value) && value.length <= maxItems && value.every((item) => isShortString(item));
}

function isRegime(value: unknown): value is RegimeSnapshot {
  if (!isRecord(value)) return false;
  return (
    hasOnlyKeys(value, ["date", "state", "model_version", "p_bull", "p_sideways", "p_bear"]) &&
    isShortString(value.date, 32) &&
    ["BULL", "SIDEWAYS", "BEAR"].includes(String(value.state)) &&
    isShortString(value.model_version, 200) &&
    isNumberInRange(value.p_bull, 0, 1) &&
    isNumberInRange(value.p_sideways, 0, 1) &&
    isNumberInRange(value.p_bear, 0, 1)
  );
}

function isRiskSnapshot(value: unknown): value is RiskSnapshot | null {
  if (value === null) return true;
  if (!isRecord(value)) return false;
  return (
    hasOnlyKeys(value, [
      "date", "model_version", "risk_band", "score", "score_1d", "score_5d",
      "at_risk", "score_is_probability", "vix_level_rank", "vix_acceleration_rank",
      "credit_risk_rank",
    ]) &&
    isShortString(value.date, 32) &&
    isShortString(value.model_version, 200) &&
    isShortString(value.risk_band, 100) &&
    isNullableNumberInRange(value.score, -1_000_000, 1_000_000) &&
    (value.score_1d === undefined || isNullableNumberInRange(value.score_1d, -1_000_000, 1_000_000)) &&
    (value.score_5d === undefined || isNullableNumberInRange(value.score_5d, -1_000_000, 1_000_000)) &&
    (value.at_risk === undefined || typeof value.at_risk === "boolean") &&
    (value.score_is_probability === undefined || typeof value.score_is_probability === "boolean") &&
    (value.vix_level_rank === undefined || isNullableNumberInRange(value.vix_level_rank, -1_000_000, 1_000_000)) &&
    (value.vix_acceleration_rank === undefined || isNullableNumberInRange(value.vix_acceleration_rank, -1_000_000, 1_000_000)) &&
    (value.credit_risk_rank === undefined || isNullableNumberInRange(value.credit_risk_rank, -1_000_000, 1_000_000))
  );
}

function isSectorRow(value: unknown): value is SectorRow {
  if (!isRecord(value)) return false;
  if (!hasOnlyKeys(value, [
    "ticker", "sector", "quadrant", "quadrant_transition", "tail_heading", "rs_ratio",
    "rs_momentum", "delta_rs_ratio", "delta_rs_momentum", "shadow_score", "forecast_rank",
    "forecast_score", "predicted_excess_20d", "signal",
  ])) return false;
  const strings = [
    value.ticker,
    value.sector,
    value.quadrant,
    value.quadrant_transition,
    value.tail_heading,
    value.signal,
  ];
  const numbers = [
    value.rs_ratio,
    value.rs_momentum,
    value.delta_rs_ratio,
    value.delta_rs_momentum,
    value.shadow_score,
    value.forecast_rank,
    value.forecast_score,
    value.predicted_excess_20d,
  ];
  return strings.every((item) => isShortString(item, 200)) && numbers.every(isFiniteNumber);
}

function isScoreCoverage(value: unknown): value is LegacyRiskPolicySnapshot["score_coverage"] {
  if (isShortString(value, 2_000)) return true;
  if (!isRecord(value) || Object.keys(value).length > 24) return false;
  return Object.entries(value).every(([key, item]) => {
    if (!isShortString(key, 100)) return false;
    if (item === null || typeof item === "boolean" || isFiniteNumber(item) || isShortString(item, 500)) return true;
    return isStringList(item, 24);
  });
}

function isReentryCondition(value: unknown): value is ReentryCondition {
  if (!isRecord(value)) return false;
  const status = value.status;
  const quality = value.data_quality;
  return (
    hasOnlyKeys(value, ["key", "label", "status", "data_quality", "detail"]) &&
    isShortString(value.key, 100) &&
    value.key.length > 0 &&
    isShortString(value.label, 200) &&
    value.label.length > 0 &&
    ["CONFIRMED", "NOT_CONFIRMED", "UNAVAILABLE"].includes(String(status)) &&
    (quality === undefined || (
      ["EXACT", "PUBLIC_LIVE", "UNAVAILABLE"].includes(String(quality)) &&
      ((status === "UNAVAILABLE") === (quality === "UNAVAILABLE"))
    )) &&
    isShortString(value.detail, 1_000)
  );
}

function isReentryPolicy(value: unknown): value is ReentryPolicy {
  if (!isRecord(value)) return false;
  if (!hasOnlyKeys(value, [
    "status", "current_p_bear", "previous_p_bear", "recent_peak_p_bear", "risk_off_peak_seen",
    "risk_off_peak_min", "p_bear_falling", "delta_p_bear",
    "confirmed", "required", "total", "conditions", "ladder", "advisory_only",
  ])) return false;
  const confirmed = value.confirmed;
  const required = value.required;
  const total = value.total;
  if (
    !Number.isInteger(confirmed) ||
    !Number.isInteger(required) ||
    !Number.isInteger(total) ||
    !isFiniteNumber(confirmed) ||
    !isFiniteNumber(required) ||
    !isFiniteNumber(total) ||
    confirmed < 0 ||
    required < 0 ||
    total < 1 ||
    total > 8 ||
    confirmed > total ||
    required > total
  ) return false;
  if (!Array.isArray(value.ladder) || value.ladder.length < 1 || value.ladder.length > 8) return false;
  if (!value.ladder.every((item) => isNumberInRange(item, 0, 1))) return false;
  if (!Array.isArray(value.conditions) || value.conditions.length !== total) return false;
  if (!value.conditions.every(isReentryCondition)) return false;
  const conditionKeys = value.conditions.map((condition) => condition.key);
  return (
    new Set(conditionKeys).size === conditionKeys.length &&
    isShortString(value.status, 100) &&
    value.status.length > 0 &&
    (value.current_p_bear === undefined || isNullableNumberInRange(value.current_p_bear, 0, 1)) &&
    (value.previous_p_bear === undefined || isNullableNumberInRange(value.previous_p_bear, 0, 1)) &&
    (value.recent_peak_p_bear === undefined || isNullableNumberInRange(value.recent_peak_p_bear, 0, 1)) &&
    (value.risk_off_peak_seen === undefined || value.risk_off_peak_seen === null || typeof value.risk_off_peak_seen === "boolean") &&
    (value.risk_off_peak_min === undefined || isNumberInRange(value.risk_off_peak_min, 0, 1)) &&
    (value.p_bear_falling === null || typeof value.p_bear_falling === "boolean") &&
    isNullableNumberInRange(value.delta_p_bear, -1, 1) &&
    value.advisory_only === true
  );
}

function isLegacyRiskPolicy(value: unknown): value is LegacyRiskPolicySnapshot {
  if (!isRecord(value)) return false;
  if (!hasOnlyKeys(value, [
    "model_version", "decision_status", "market_health_score", "health_band", "base_exposure",
    "hmm_cap", "fast_overlay_cap", "raw_recommended_exposure", "previous_recommended_exposure",
    "recommended_exposure", "primary_driver_code", "primary_driver",
    "active_triggers", "missing_channels", "score_method", "score_coverage", "reentry", "advisory_only",
  ])) return false;
  const decisionStatus = value.decision_status;
  if (decisionStatus !== "OK" && decisionStatus !== "NO_DECISION") return false;
  const score = value.market_health_score;
  const exposures = [value.base_exposure, value.hmm_cap, value.fast_overlay_cap, value.recommended_exposure];
  const band = value.health_band;
  const valuesAreValid = (
    isNullableNumberInRange(score, 0, 100) &&
    exposures.every((item) => isNullableNumberInRange(item, 0, 1)) &&
    (value.raw_recommended_exposure === undefined || isNullableNumberInRange(value.raw_recommended_exposure, 0, 1)) &&
    (value.previous_recommended_exposure === undefined || isNullableNumberInRange(value.previous_recommended_exposure, 0, 1)) &&
    (band === null || ["GREEN", "YELLOW", "RED"].includes(String(band)))
  );
  const decisionIsConsistent = decisionStatus === "OK"
    ? score !== null && exposures.every((item) => item !== null) && band !== null
    : score === null && exposures.every((item) => item === null) && band === null;
  return (
    valuesAreValid &&
    decisionIsConsistent &&
    isShortString(value.model_version, 200) &&
    value.model_version.length > 0 &&
    isShortString(value.primary_driver_code, 100) &&
    isShortString(value.primary_driver, 500) &&
    isStringList(value.active_triggers, 32) &&
    (value.missing_channels === undefined || isStringList(value.missing_channels, 32)) &&
    isShortString(value.score_method, 200) &&
    isScoreCoverage(value.score_coverage) &&
    value.advisory_only === true &&
    isReentryPolicy(value.reentry)
  );
}

function isFastOverlayCondition(value: unknown): value is FastOverlayCondition {
  if (!isRecord(value) || !hasOnlyKeys(value, ["key", "label", "status", "data_quality", "detail"])) return false;
  const status = value.status;
  const quality = value.data_quality;
  return (
    isShortString(value.key, 100) &&
    value.key.length > 0 &&
    isShortString(value.label, 200) &&
    value.label.length > 0 &&
    ["TRIGGERED", "CLEAR", "UNAVAILABLE"].includes(String(status)) &&
    ["EXACT", "PUBLIC_LIVE", "UNAVAILABLE"].includes(String(quality)) &&
    (status === "UNAVAILABLE") === (quality === "UNAVAILABLE") &&
    (value.detail === undefined || isShortString(value.detail, 1_000))
  );
}

function isFastOverlayPolicy(value: unknown): value is FastOverlayPolicy {
  if (!isRecord(value) || !hasOnlyKeys(value, [
    "status", "trigger_count", "required", "total", "complete", "conditions",
  ])) return false;
  if (
    !["TRIGGERED", "CLEAR", "INCOMPLETE"].includes(String(value.status)) ||
    !isIntegerInRange(value.trigger_count, 0, 8) ||
    value.required !== 3 ||
    value.total !== 8 ||
    typeof value.complete !== "boolean" ||
    !Array.isArray(value.conditions) ||
    value.conditions.length !== 8 ||
    !value.conditions.every(isFastOverlayCondition)
  ) return false;
  const conditions = value.conditions as FastOverlayCondition[];
  const keys = conditions.map((condition) => condition.key);
  const triggerCount = conditions.filter((condition) => condition.status === "TRIGGERED").length;
  const complete = conditions.every((condition) => condition.status !== "UNAVAILABLE");
  const expectedStatus = triggerCount >= 3 ? "TRIGGERED" : complete ? "CLEAR" : "INCOMPLETE";
  return (
    new Set(keys).size === keys.length &&
    value.trigger_count === triggerCount &&
    value.complete === complete &&
    value.status === expectedStatus
  );
}

function isV2ReentryPolicy(value: unknown): value is ReentryPolicy {
  return (
    isReentryPolicy(value) &&
    value.required === 3 &&
    value.total === 4 &&
    value.ladder.every((item) => isPercentGrid(item))
  );
}

function isRiskPolicyV2(value: unknown): value is RiskPolicySnapshotV2 {
  if (!isRecord(value) || !hasOnlyKeys(value, [
    "model_version", "decision_status", "slow_risk_score", "risk_score", "risk_score_delta",
    "risk_trend", "base_exposure", "fast_adjustment", "raw_recommended_exposure",
    "previous_recommended_exposure", "recommended_exposure", "risk_band",
    "primary_driver_code", "primary_driver", "active_triggers", "missing_channels",
    "fast_overlay", "reentry", "advisory_only",
  ])) return false;
  const decisionStatus = value.decision_status;
  if (decisionStatus !== "OK" && decisionStatus !== "NO_DECISION") return false;
  const slowScore = value.slow_risk_score;
  const riskScore = value.risk_score;
  const band = value.risk_band;
  const exposures = [value.base_exposure, value.raw_recommended_exposure, value.recommended_exposure];
  const scoresAreValid = (
    (slowScore === null || isIntegerInRange(slowScore, 0, 100)) &&
    (riskScore === null || isIntegerInRange(riskScore, 0, 100)) &&
    (value.risk_score_delta === undefined || value.risk_score_delta === null || isNumberInRange(value.risk_score_delta, -100, 100)) &&
    (value.risk_trend === undefined || value.risk_trend === null || ["UP", "DOWN", "FLAT"].includes(String(value.risk_trend)))
  );
  const allocationValuesAreValid = (
    exposures.every((item) => isNullablePercentGrid(item)) &&
    isNullablePercentGrid(value.previous_recommended_exposure) &&
    (value.fast_adjustment === null || isPercentGrid(value.fast_adjustment, -0.75, 0)) &&
    (band === null || ["GREEN", "YELLOW", "RED"].includes(String(band)))
  );
  const decisionIsConsistent = decisionStatus === "OK"
    ? slowScore !== null && riskScore !== null && exposures.every((item) => item !== null) && value.fast_adjustment !== null && band !== null
    : slowScore === null && riskScore === null && exposures.every((item) => item === null) && value.fast_adjustment === null && band === null;
  return (
    scoresAreValid &&
    allocationValuesAreValid &&
    decisionIsConsistent &&
    isShortString(value.model_version, 200) &&
    value.model_version.length > 0 &&
    isShortString(value.primary_driver_code, 100) &&
    isShortString(value.primary_driver, 500) &&
    isStringList(value.active_triggers, 32) &&
    isStringList(value.missing_channels, 32) &&
    isFastOverlayPolicy(value.fast_overlay) &&
    isV2ReentryPolicy(value.reentry) &&
    value.advisory_only === true
  );
}

function isPolicyState(value: unknown): value is PolicyState {
  if (!isRecord(value) || !hasOnlyKeys(value, [
    "state_version", "policy_version", "regime_model_version", "last_as_of",
    "exposure", "raw_target", "recovery_clear_streak", "episode_peak_p_bear",
    "episode_started_as_of",
  ])) return false;
  const episodeFieldsAgree = (value.episode_peak_p_bear === null) === (value.episode_started_as_of === null);
  return (
    value.state_version === 1 &&
    isShortString(value.policy_version, 200) && value.policy_version.length > 0 &&
    isShortString(value.regime_model_version, 200) && value.regime_model_version.length > 0 &&
    isIsoDate(value.last_as_of) &&
    isPercentGrid(value.exposure) &&
    isPercentGrid(value.raw_target) &&
    isIntegerInRange(value.recovery_clear_streak, 0, 10_000) &&
    (value.episode_peak_p_bear === null || isNumberInRange(value.episode_peak_p_bear, 0, 1)) &&
    (value.episode_started_as_of === null || isIsoDate(value.episode_started_as_of)) &&
    episodeFieldsAgree
  );
}

function isRiskPolicyV3(value: unknown): value is RiskPolicySnapshotV3 {
  if (!isRecord(value) || !hasOnlyKeys(value, [
    "model_version", "decision_status", "slow_risk_score", "risk_score", "risk_score_delta",
    "risk_trend", "base_exposure", "fast_adjustment", "raw_recommended_exposure",
    "previous_recommended_exposure", "recommended_exposure", "risk_band",
    "primary_driver_code", "primary_driver", "active_triggers", "missing_channels",
    "fast_available_count", "recovery_phase", "fast_overlay", "reentry", "policy_state",
    "advisory_only",
  ])) return false;
  const base = { ...value };
  delete base.fast_available_count;
  delete base.recovery_phase;
  delete base.policy_state;
  if (!isRiskPolicyV2(base)) return false;
  const policy = value as unknown as RiskPolicySnapshotV3;
  const availableCount = policy.fast_overlay.conditions.filter(
    (condition) => condition.status !== "UNAVAILABLE",
  ).length;
  const validState = policy.policy_state === null || isPolicyState(policy.policy_state);
  if (!validState || policy.fast_available_count !== availableCount) return false;
  if (policy.decision_status === "NO_DECISION") {
    return policy.recovery_phase === "NO_DECISION" && (
      policy.policy_state === null || policy.policy_state.policy_version === policy.model_version
    );
  }
  return (
    policy.recovery_phase !== "NO_DECISION" &&
    policy.policy_state !== null &&
    policy.policy_state.policy_version === policy.model_version &&
    policy.policy_state.exposure === policy.recommended_exposure &&
    policy.policy_state.raw_target === policy.raw_recommended_exposure
  );
}

function isHistory(value: unknown): value is DashboardSnapshotV3["history"] {
  if (!isRecord(value) || !hasOnlyKeys(value, [
    "series_version", "build_mode", "seed_mode", "parent_as_of",
  ])) return false;
  return (
    isShortString(value.series_version, 500) && value.series_version.length > 0 &&
    ["LIVE", "BACKFILL"].includes(String(value.build_mode)) &&
    ["STRICT_PRIOR", "FULL_REPLAY"].includes(String(value.seed_mode)) &&
    (value.parent_as_of === null || isIsoDate(value.parent_as_of))
  );
}

function hasValidDashboardBase(value: Record<string, unknown>) {
  return (
    isShortString(value.generated_at, 100) &&
    isShortString(value.as_of, 32) &&
    isRegime(value.regime) &&
    isRiskSnapshot(value.crash_guard) &&
    isRiskSnapshot(value.drawdown_onset) &&
    isRecord(value.sector) &&
    hasOnlyKeys(value.sector, ["rrg_as_of", "forecast_as_of", "target", "training_weeks", "rows"]) &&
    isShortString(value.sector.rrg_as_of, 32) &&
    isShortString(value.sector.forecast_as_of, 32) &&
    isShortString(value.sector.target, 500) &&
    Number.isInteger(value.sector.training_weeks) &&
    isFiniteNumber(value.sector.training_weeks) &&
    value.sector.training_weeks >= 0 &&
    Array.isArray(value.sector.rows) &&
    value.sector.rows.length <= 100 &&
    value.sector.rows.every(isSectorRow) &&
    isStringList(value.notices)
  );
}

export function isDashboardSnapshot(value: unknown): value is DashboardSnapshot {
  if (!isRecord(value) || ![1, 2, 3].includes(Number(value.schema_version))) return false;
  const allowed = [
    "schema_version", "generated_at", "as_of", "regime", "crash_guard",
    "drawdown_onset", "risk_policy", "sector", "notices",
    ...(value.schema_version === 3 ? ["history"] : []),
  ];
  if (!hasOnlyKeys(value, allowed) || !hasValidDashboardBase(value)) return false;
  if (value.schema_version === 1) {
    return value.risk_policy === undefined || isLegacyRiskPolicy(value.risk_policy);
  }
  if (value.schema_version === 2) return isRiskPolicyV2(value.risk_policy);
  if (!isHistory(value.history) || !isRiskPolicyV3(value.risk_policy)) return false;
  const snapshot = value as unknown as DashboardSnapshotV3;
  const state = snapshot.risk_policy.policy_state;
  return (
    isIsoDate(snapshot.as_of) && snapshot.regime.date === snapshot.as_of &&
    (snapshot.history.parent_as_of === null || snapshot.history.parent_as_of < snapshot.as_of) &&
    (state === null || (
      state.regime_model_version === snapshot.regime.model_version &&
      (state.episode_started_as_of === null || state.episode_started_as_of <= state.last_as_of) &&
      (snapshot.risk_policy.decision_status === "OK"
        ? state.last_as_of === snapshot.as_of
        : state.last_as_of < snapshot.as_of)
    ))
  );
}
