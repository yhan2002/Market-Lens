type RoroModelVersionsV1 = {
  roro: "roro-ew-v1-w756-m252-u20";
  axis: "roro-3axis-v1-w756-m252-p126-u20";
  portfolio: "portfolio-internals-nq-v2-c85-incomplete-snapshot-qc";
  phase: "roro-phase-v1";
};

export type RoroQ20 = {
  model_version: "roro-q20-logit-v1-frozen8";
  status: "INSUFFICIENT_EVIDENCE" | "READY_FOR_WALK_FORWARD_FIT";
  probability: number | null;
  fit_allowed: boolean;
  horizon_trading_days: 20;
  drawdown_threshold: -0.05;
  feature_rows: number;
  complete_feature_labels: number;
  independent_events: number;
  minimum_complete_labels: 252;
  minimum_independent_events: 20;
};

export type RoroPolicy = {
  model_version: "roro-policy-v1-score-fast3of8-confirm3-shadow";
  score_based_long: number;
  raw_target_long: number;
  recommended_long: number;
  fast_status: "TRIGGERED" | "CLEAR" | "INCOMPLETE";
  fast_trigger_count: number;
  fast_available_count: number;
  active_fast_conditions: string[];
  unavailable_fast_conditions: string[];
  phase: "COLD_START" | "RISK_REDUCTION" | "RECOVERY_HOLD" | "REENTRY_DIRECT" | "RECOVERY_CONFIRM" | "AT_TARGET";
  episode_status: "ACTIVE" | "INACTIVE";
  clear_streak: number;
  primary_driver: string;
};

export type RoroComponentName =
  | "credit"
  | "rates_volatility"
  | "equity_volatility"
  | "fx_carry"
  | "safe_haven"
  | "equity_trend"
  | "market_breadth"
  | "portfolio_correlation"
  | "portfolio_beta"
  | "portfolio_concentration"
  | "portfolio_participation"
  | "portfolio_drawdown";

type RoroComponentDiagnostics = {
  [Key in `${RoroComponentName}_contribution`]: number | null;
} & {
  [Key in `${RoroComponentName}_score_trajectory_${1 | 5 | 20}d`]: number | null;
};

export type RoroPayloadV1 = RoroComponentDiagnostics & {
  credit_score: number | null;
  rates_volatility_score: number | null;
  equity_volatility_score: number | null;
  fx_carry_score: number | null;
  safe_haven_score: number | null;
  equity_trend_score: number | null;
  breadth_risk_percentile: number | null;
  portfolio_correlation_score: number | null;
  portfolio_beta_score: number | null;
  portfolio_concentration_score: number | null;
  portfolio_participation_score: number | null;
  portfolio_drawdown_score: number | null;
  market_risk_score: number;
  risk_regime: "GREEN" | "YELLOW" | "RED";
  recommended_long: number;
  trajectory_1d: number | null;
  trajectory_5d: number | null;
  trajectory_20d: number | null;
  macro_risk_score: number | null;
  market_trend_score: number;
  portfolio_internal_score: number;
  confidence: number;
  data_completeness: number;
  axis_disagreement: number;
  axis_agreement: number;
  data_status: string;
  trend_phase: string;
  turn_evidence: string[];
  checklist_available_count: number;
  bottom_confirmation_count: number;
  top_confirmation_count: number;
  portfolio_correlation_20: number | null;
  portfolio_correlation_trajectory_5d: number | null;
  nq_beta_60: number | null;
  effective_positions: number | null;
  breadth_above_ma20: number | null;
  negative_participation_20: number | null;
  portfolio_drawdown_60: number | null;
  price_weight_coverage: number | null;
  history_weight_coverage_20d: number | null;
  history_weight_coverage_60d: number | null;
  data_quality: string | null;
  risk_drivers: string[];
  relief_drivers: string[];
  available_components: RoroComponentName[];
  component_risk_drivers: RoroComponentName[];
  component_relief_drivers: RoroComponentName[];
};

type RoroEnvelope = {
  series_version: string;
  as_of: string;
  generated_at: string;
  input_cutoff: string;
  provenance_hash: string;
};

export type RoroShadowSnapshotV1 = RoroEnvelope & {
  schema_version: 1;
  model_versions: RoroModelVersionsV1;
  payload: RoroPayloadV1;
};

export type RoroShadowSnapshotV2 = RoroEnvelope & {
  schema_version: 2;
  model_versions: RoroModelVersionsV1 & { q20: string };
  payload: RoroPayloadV1 & { q20: RoroQ20 };
};

export type RoroShadowSnapshotV3 = RoroEnvelope & {
  schema_version: 3;
  model_versions: RoroModelVersionsV1 & { q20: string; policy: string };
  payload: RoroPayloadV1 & { q20: RoroQ20; policy: RoroPolicy };
};

export type RoroShadowSnapshotV4 = RoroEnvelope & {
  schema_version: 4;
  model_versions: RoroModelVersionsV1 & {
    q20: string;
    policy: string;
    price_observations: "yahoo-session-v1-jpyx-gld-close";
    market_breadth?: "external-breadth-v1-ndth-200dma";
  };
  payload: RoroPayloadV1 & { q20: RoroQ20; policy: RoroPolicy };
};

export type RoroShadowSnapshot = RoroShadowSnapshotV1 | RoroShadowSnapshotV2 | RoroShadowSnapshotV3 | RoroShadowSnapshotV4;

const ROOT_KEYS = ["schema_version", "series_version", "as_of", "generated_at", "input_cutoff", "provenance_hash", "model_versions", "payload"] as const;
const MODEL_V1_KEYS = ["roro", "axis", "portfolio", "phase"] as const;
const MODEL_V2_KEYS = [...MODEL_V1_KEYS, "q20"] as const;
const MODEL_V3_KEYS = [...MODEL_V2_KEYS, "policy"] as const;
const MODEL_V4_KEYS = [...MODEL_V3_KEYS, "price_observations"] as const;
const MODEL_V10_KEYS = [...MODEL_V4_KEYS, "market_breadth"] as const;
const Q20_KEYS = [
  "model_version", "status", "probability", "fit_allowed", "horizon_trading_days",
  "drawdown_threshold", "feature_rows", "complete_feature_labels", "independent_events",
  "minimum_complete_labels", "minimum_independent_events",
] as const;
const POLICY_KEYS = [
  "model_version", "score_based_long", "raw_target_long", "recommended_long",
  "fast_status", "fast_trigger_count", "fast_available_count",
  "active_fast_conditions", "unavailable_fast_conditions", "phase",
  "episode_status", "clear_streak", "primary_driver",
] as const;
const FAST_CONDITION_KEYS = [
  "vix_1d_spike", "vix_term_inversion", "nq_1d_drop", "move_1d_spike",
  "hy_oas_widening", "jpy_3d_drop", "breadth_1d_collapse", "portfolio_correlation",
] as const;
const COMPONENT_NAMES = [
  "credit", "rates_volatility", "equity_volatility", "fx_carry", "safe_haven",
  "equity_trend", "market_breadth", "portfolio_correlation", "portfolio_beta",
  "portfolio_concentration", "portfolio_participation", "portfolio_drawdown",
] as const;
const COMPONENT_SCORE_FIELDS = {
  credit: "credit_score",
  rates_volatility: "rates_volatility_score",
  equity_volatility: "equity_volatility_score",
  fx_carry: "fx_carry_score",
  safe_haven: "safe_haven_score",
  equity_trend: "equity_trend_score",
  market_breadth: "breadth_risk_percentile",
  portfolio_correlation: "portfolio_correlation_score",
  portfolio_beta: "portfolio_beta_score",
  portfolio_concentration: "portfolio_concentration_score",
  portfolio_participation: "portfolio_participation_score",
  portfolio_drawdown: "portfolio_drawdown_score",
} as const;
export const BREADTH_BASE_SERIES = "roro-3axis-v1-w756-m252-p126-u20__roro-phase-v1__p-c85-isqc__shadow-r21-breadth-correction";
export const BREADTH_REPAIR_SERIES = "roro-3axis-v1-w756-m252-p126-u20__roro-phase-v1__p-c85-isqc__shadow-r22-late-breadth-repair";

// Only replace matching dates. A correction must not hide a newer normal observation.
export function applyBreadthRepairs<T extends { asOfKey: number }>(rows: T[], repairs: T[]): T[] {
  const byDate = new Map(repairs.map((row) => [row.asOfKey, row]));
  return rows.map((row) => byDate.get(row.asOfKey) ?? row);
}

const SERIES_BY_SCHEMA = {
  1: "roro-3axis-v1-w756-m252-p126-u20__roro-phase-v1__p-c85-isqc__shadow-r3",
  2: "roro-3axis-v1-w756-m252-p126-u20__roro-phase-v1__p-c85-isqc__shadow-r4-q20",
  3: "roro-3axis-v1-w756-m252-p126-u20__roro-phase-v1__p-c85-isqc__shadow-r5-q20-policy",
  4: [
    "roro-3axis-v1-w756-m252-p126-u20__roro-phase-v1__p-c85-isqc__shadow-r6-q20-policy-fxgold",
    "roro-3axis-v1-w756-m252-p126-u20__roro-phase-v1__p-c85-isqc__shadow-r7-credit-move",
    "roro-3axis-v1-w756-m252-p126-u20__roro-phase-v1__p-c85-isqc__shadow-r8-narrative",
    "roro-3axis-v1-w756-m252-p126-u20__roro-phase-v1__p-c85-isqc__shadow-r9-market-narrative",
    "roro-3axis-v1-w756-m252-p126-u20__roro-phase-v1__p-c85-isqc__shadow-r10-ndth-breadth",
    "roro-3axis-v1-w756-m252-p126-u20__roro-phase-v1__p-c85-isqc__shadow-r11-ndth-move-reviewed",
    "roro-3axis-v1-w756-m252-p126-u20__roro-phase-v1__p-c85-isqc__shadow-r12-reviewed-move-priority",
    "roro-3axis-v1-w756-m252-p126-u20__roro-phase-v1__p-c85-isqc__shadow-r13-revised-hy-history",
    "roro-3axis-v1-w756-m252-p126-u20__roro-phase-v1__p-c85-isqc__shadow-r14-fred-usdjpy-fast",
    "roro-3axis-v1-w756-m252-p126-u20__roro-phase-v1__p-c85-isqc__shadow-r15-yahoo-usdjpy-fast",
    "roro-3axis-v1-w756-m252-p126-u20__roro-phase-v1__p-c85-isqc__shadow-r16-nyclose-plus1h",
    "roro-3axis-v1-w756-m252-p126-u20__roro-phase-v1__p-c85-isqc__shadow-r17-exact-market-inputs",
    "roro-3axis-v1-w756-m252-p126-u20__roro-phase-v1__p-c85-isqc__shadow-r18-history-from-20251008",
    "roro-3axis-v1-w756-m252-p126-u20__roro-phase-v1__p-c85-isqc__shadow-r19-holdings-history-recalc",
    BREADTH_BASE_SERIES,
    BREADTH_REPAIR_SERIES,
  ],
} as const;

export function roroHistoryFallbackSeries(activeSeries: string): string[] {
  const series = SERIES_BY_SCHEMA[4] as readonly string[];
  const activeIndex = series.indexOf(activeSeries);
  return activeIndex < 1 ? [] : series.slice(0, activeIndex).reverse();
}

export function trailingDifference(values: readonly (number | null)[]): (number | null)[] {
  return values.map((value, index) => {
    const previous = values[index - 1];
    return finite(value) && finite(previous) ? value - previous : null;
  });
}
const PRICE_OBSERVATIONS_CONTRACT = "yahoo-session-v1-jpyx-gld-close" as const;
const BASE_MODEL_VERSIONS = {
  roro: "roro-ew-v1-w756-m252-u20",
  axis: "roro-3axis-v1-w756-m252-p126-u20",
  portfolio: "portfolio-internals-nq-v2-c85-incomplete-snapshot-qc",
  phase: "roro-phase-v1",
} as const;
const FORBIDDEN_IDENTITY_KEYS = new Set([
  "ticker", "tickers", "symbol", "symbols", "isin", "cusip", "sedol",
  "security", "security_id", "security_name", "company", "company_name", "name",
  "quantity", "position", "positions", "holding", "holdings", "daily_holdings",
  "market_value", "market_value_base",
]);

function object(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function onlyKeys(value: Record<string, unknown>, keys: readonly string[]) {
  const actual = Object.keys(value);
  return actual.length === keys.length && actual.every((key) => keys.includes(key));
}

function finite(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function optionalFinite(value: unknown): value is number | null {
  return value === null || finite(value);
}

function bounded(value: unknown, low = 0, high = 100): value is number {
  return finite(value) && value >= low && value <= high;
}

function optionalBounded(value: unknown, low: number, high: number): value is number | null {
  return value === null || bounded(value, low, high);
}

function nonNegativeInteger(value: unknown): value is number {
  return Number.isInteger(value) && Number(value) >= 0;
}

function integerBetween(value: unknown, low: number, high: number): value is number {
  return Number.isInteger(value) && Number(value) >= low && Number(value) <= high;
}

function scoreBasedLong(riskScore: number) {
  const anchors = [[0, 100], [33, 90], [66, 60], [100, 25]] as const;
  for (let index = 1; index < anchors.length; index += 1) {
    const [leftScore, leftLong] = anchors[index - 1];
    const [rightScore, rightLong] = anchors[index];
    if (riskScore <= rightScore) {
      const weight = (riskScore - leftScore) / (rightScore - leftScore);
      return Math.floor(leftLong + weight * (rightLong - leftLong) + 0.5);
    }
  }
  return 25;
}

function stringList(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((item) => typeof item === "string");
}

function isoTimestamp(value: unknown): value is string {
  if (typeof value !== "string" || value.length > 100 ||
      !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$/.test(value)) return false;
  const parsed = new Date(value);
  return !Number.isNaN(parsed.valueOf()) && parsed.toISOString().slice(0, 19) === value.slice(0, 19);
}

function containsForbiddenIdentity(value: unknown, seen = new WeakSet<object>()): boolean {
  if (typeof value !== "object" || value === null || seen.has(value)) return false;
  seen.add(value);
  if (Array.isArray(value)) return value.some((item) => containsForbiddenIdentity(item, seen));
  return Object.entries(value).some(([key, item]) => {
    const normalized = key.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "");
    return FORBIDDEN_IDENTITY_KEYS.has(normalized) || containsForbiddenIdentity(item, seen);
  });
}

function validModels(value: Record<string, unknown>) {
  return value.roro === BASE_MODEL_VERSIONS.roro && value.axis === BASE_MODEL_VERSIONS.axis &&
    value.portfolio === BASE_MODEL_VERSIONS.portfolio && value.phase === BASE_MODEL_VERSIONS.phase;
}

function bytesHex(bytes: Uint8Array) {
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
}

function compareCodePoints(left: string, right: string) {
  const a = Array.from(left, (character) => character.codePointAt(0) as number);
  const b = Array.from(right, (character) => character.codePointAt(0) as number);
  for (let index = 0; index < Math.min(a.length, b.length); index += 1) {
    if (a[index] !== b[index]) return a[index] - b[index];
  }
  return a.length - b.length;
}

export function canonicalRoroProvenance(value: unknown): string {
  if (value === null) return "n";
  if (typeof value === "boolean") return value ? "b1" : "b0";
  if (typeof value === "number") {
    if (!Number.isFinite(value)) throw new TypeError("non-finite provenance number");
    const bytes = new Uint8Array(8);
    new DataView(bytes.buffer).setFloat64(0, value, false);
    return `d${bytesHex(bytes)}`;
  }
  if (typeof value === "string") {
    const bytes = new TextEncoder().encode(value);
    return `s${bytes.length}:${bytesHex(bytes)}`;
  }
  if (Array.isArray(value)) {
    return `a${value.length}[${value.map(canonicalRoroProvenance).join("")}]`;
  }
  if (object(value)) {
    const keys = Object.keys(value).sort(compareCodePoints);
    return `o${keys.length}{${keys.map((key) =>
      canonicalRoroProvenance(key) + canonicalRoroProvenance(value[key])).join("")}}`;
  }
  throw new TypeError(`unsupported provenance value: ${typeof value}`);
}

export async function roroProvenanceHash(snapshot: RoroShadowSnapshotV4): Promise<string> {
  const material = {
    series_version: snapshot.series_version,
    date: snapshot.as_of,
    input_cutoff: snapshot.input_cutoff,
    model_versions: snapshot.model_versions,
    payload: snapshot.payload,
  };
  const digest = await crypto.subtle.digest(
    "SHA-256",
    new TextEncoder().encode(canonicalRoroProvenance(material)),
  );
  return bytesHex(new Uint8Array(digest));
}

export async function verifyRoroProvenance(snapshot: RoroShadowSnapshot): Promise<boolean> {
  return snapshot.schema_version === 4 &&
    snapshot.provenance_hash === await roroProvenanceHash(snapshot);
}

function validPayload(payload: Record<string, unknown>) {
  const componentLists = [
    payload.available_components,
    payload.component_risk_drivers,
    payload.component_relief_drivers,
  ];
  const componentsValid = COMPONENT_NAMES.every((component) =>
    optionalBounded(payload[COMPONENT_SCORE_FIELDS[component]], 0, 100) &&
    optionalFinite(payload[`${component}_contribution`]) &&
    ([1, 5, 20] as const).every((lag) =>
      optionalFinite(payload[`${component}_score_trajectory_${lag}d`]),
    ),
  ) && componentLists.every((items) =>
    stringList(items) && new Set(items).size === items.length &&
    items.every((item) => COMPONENT_NAMES.includes(item as RoroComponentName)),
  );
  return (
    bounded(payload.market_risk_score) && ["GREEN", "YELLOW", "RED"].includes(String(payload.risk_regime)) &&
    bounded(payload.recommended_long) &&
    optionalFinite(payload.trajectory_1d) && optionalFinite(payload.trajectory_5d) && optionalFinite(payload.trajectory_20d) &&
    optionalBounded(payload.macro_risk_score, 0, 100) && bounded(payload.market_trend_score) && bounded(payload.portfolio_internal_score) &&
    bounded(payload.confidence) && bounded(payload.data_completeness) &&
    bounded(payload.axis_disagreement) && bounded(payload.axis_agreement) &&
    typeof payload.data_status === "string" &&
    typeof payload.trend_phase === "string" && stringList(payload.turn_evidence) &&
    finite(payload.checklist_available_count) && finite(payload.bottom_confirmation_count) && finite(payload.top_confirmation_count) &&
    optionalBounded(payload.portfolio_correlation_20, -1, 1) && optionalFinite(payload.portfolio_correlation_trajectory_5d) &&
    optionalFinite(payload.nq_beta_60) && (payload.effective_positions === null || finite(payload.effective_positions) && payload.effective_positions > 0) &&
    optionalBounded(payload.breadth_above_ma20, 0, 1) && optionalBounded(payload.negative_participation_20, 0, 1) &&
    optionalBounded(payload.portfolio_drawdown_60, -1, 0) && optionalBounded(payload.price_weight_coverage, 0, 1 + 1e-9) &&
    (payload.data_quality === null || typeof payload.data_quality === "string") &&
    optionalBounded(payload.history_weight_coverage_20d, 0, 1 + 1e-9) && optionalBounded(payload.history_weight_coverage_60d, 0, 1 + 1e-9) &&
    stringList(payload.risk_drivers) && stringList(payload.relief_drivers) && componentsValid
  );
}

function validQ20(value: unknown): value is RoroQ20 {
  if (!object(value) || !onlyKeys(value, Q20_KEYS)) return false;
  if (value.model_version !== "roro-q20-logit-v1-frozen8" ||
      !["INSUFFICIENT_EVIDENCE", "READY_FOR_WALK_FORWARD_FIT"].includes(String(value.status)) ||
      value.probability !== null || typeof value.fit_allowed !== "boolean" ||
      value.horizon_trading_days !== 20 || value.drawdown_threshold !== -0.05 ||
      !nonNegativeInteger(value.feature_rows) || !nonNegativeInteger(value.complete_feature_labels) ||
      !nonNegativeInteger(value.independent_events) || value.minimum_complete_labels !== 252 ||
      value.minimum_independent_events !== 20 || value.complete_feature_labels > value.feature_rows ||
      value.independent_events > value.complete_feature_labels) return false;
  const fitAllowed = value.complete_feature_labels >= 252 && value.independent_events >= 20;
  return value.fit_allowed === fitAllowed &&
    value.status === (fitAllowed ? "READY_FOR_WALK_FORWARD_FIT" : "INSUFFICIENT_EVIDENCE");
}

function validPolicy(value: unknown): value is RoroPolicy {
  if (!object(value) || !onlyKeys(value, POLICY_KEYS) ||
      value.model_version !== "roro-policy-v1-score-fast3of8-confirm3-shadow" ||
      !integerBetween(value.score_based_long, 25, 100) ||
      !integerBetween(value.raw_target_long, 25, 100) ||
      !integerBetween(value.recommended_long, 25, 100) ||
      !integerBetween(value.fast_trigger_count, 0, FAST_CONDITION_KEYS.length) ||
      !integerBetween(value.fast_available_count, 0, FAST_CONDITION_KEYS.length) ||
      !integerBetween(value.clear_streak, 0, 3) ||
      !stringList(value.active_fast_conditions) || !stringList(value.unavailable_fast_conditions) ||
      !["COLD_START", "RISK_REDUCTION", "RECOVERY_HOLD", "REENTRY_DIRECT", "RECOVERY_CONFIRM", "AT_TARGET"].includes(String(value.phase)) ||
      !["ACTIVE", "INACTIVE"].includes(String(value.episode_status)) ||
      typeof value.primary_driver !== "string" || !value.primary_driver.trim()) return false;

  const active = value.active_fast_conditions;
  const unavailable = value.unavailable_fast_conditions;
  if (new Set(active).size !== active.length || new Set(unavailable).size !== unavailable.length ||
      !active.every((key) => FAST_CONDITION_KEYS.includes(key as typeof FAST_CONDITION_KEYS[number])) ||
      !unavailable.every((key) => FAST_CONDITION_KEYS.includes(key as typeof FAST_CONDITION_KEYS[number])) ||
      active.some((key) => unavailable.includes(key)) ||
      value.fast_trigger_count !== active.length ||
      value.fast_available_count !== FAST_CONDITION_KEYS.length - unavailable.length) return false;

  const expectedStatus = active.length >= 3
    ? "TRIGGERED"
    : unavailable.length === 0 ? "CLEAR" : "INCOMPLETE";
  const expectedRaw = active.length >= 3
    ? Math.max(25, value.score_based_long - 20)
    : value.score_based_long;
  return value.fast_status === expectedStatus && value.raw_target_long === expectedRaw &&
    value.recommended_long <= value.raw_target_long &&
    (value.episode_status === "ACTIVE" || value.clear_streak === 0);
}

export function isRoroShadowSnapshot(value: unknown): value is RoroShadowSnapshot {
  if (!object(value) || !onlyKeys(value, ROOT_KEYS) || !object(value.model_versions) || !object(value.payload) ||
      typeof value.series_version !== "string" || !value.series_version || !isoTimestamp(value.generated_at) ||
      typeof value.input_cutoff !== "string" || typeof value.as_of !== "string" ||
      !/^\d{4}-\d{2}-\d{2}$/.test(value.as_of) || typeof value.provenance_hash !== "string" ||
      !/^[a-f0-9]{64}$/.test(value.provenance_hash) || containsForbiddenIdentity(value.payload) || !validPayload(value.payload) ||
      !validModels(value.model_versions)) return false;
  if (value.series_version === BREADTH_REPAIR_SERIES &&
      (!finite(value.payload.breadth_risk_percentile) || value.payload.market_data_status !== "FULL" ||
       !(value.payload.available_components as string[]).includes("market_breadth"))) return false;
  if (value.schema_version === 1) return value.series_version === SERIES_BY_SCHEMA[1] &&
    onlyKeys(value.model_versions, MODEL_V1_KEYS);
  if (value.schema_version === 2) return value.series_version === SERIES_BY_SCHEMA[2] &&
    onlyKeys(value.model_versions, MODEL_V2_KEYS) &&
    typeof value.model_versions.q20 === "string" && validQ20(value.payload.q20) &&
    value.model_versions.q20 === value.payload.q20.model_version;
  if (value.schema_version !== 3 && value.schema_version !== 4) return false;
  const policyModelsValid = value.schema_version === 3
    ? value.series_version === SERIES_BY_SCHEMA[3] && onlyKeys(value.model_versions, MODEL_V3_KEYS)
    : SERIES_BY_SCHEMA[4].includes(value.series_version as typeof SERIES_BY_SCHEMA[4][number]) &&
        (value.series_version.endsWith("shadow-r10-ndth-breadth") || value.series_version.endsWith("shadow-r11-ndth-move-reviewed") || value.series_version.endsWith("shadow-r12-reviewed-move-priority") || value.series_version.endsWith("shadow-r13-revised-hy-history") || value.series_version.endsWith("shadow-r14-fred-usdjpy-fast") || value.series_version.endsWith("shadow-r15-yahoo-usdjpy-fast") || value.series_version.endsWith("shadow-r16-nyclose-plus1h") || value.series_version.endsWith("shadow-r17-exact-market-inputs") || value.series_version.endsWith("shadow-r18-history-from-20251008") || value.series_version.endsWith("shadow-r19-holdings-history-recalc") || value.series_version.endsWith("shadow-r21-breadth-correction") || value.series_version === BREADTH_REPAIR_SERIES
        ? onlyKeys(value.model_versions, MODEL_V10_KEYS) && value.model_versions.market_breadth === "external-breadth-v1-ndth-200dma"
        : onlyKeys(value.model_versions, MODEL_V4_KEYS)) &&
      value.model_versions.price_observations === PRICE_OBSERVATIONS_CONTRACT;
  return policyModelsValid &&
    typeof value.model_versions.q20 === "string" && typeof value.model_versions.policy === "string" &&
    validQ20(value.payload.q20) && validPolicy(value.payload.policy) &&
    value.payload.policy.unavailable_fast_conditions.includes("portfolio_correlation") &&
    !value.payload.policy.active_fast_conditions.includes("portfolio_correlation") &&
    integerBetween(value.payload.market_risk_score, 0, 100) &&
    value.payload.policy.score_based_long === scoreBasedLong(value.payload.market_risk_score) &&
    value.model_versions.q20 === value.payload.q20.model_version &&
    value.model_versions.policy === value.payload.policy.model_version &&
    value.payload.recommended_long === value.payload.policy.recommended_long;
}
