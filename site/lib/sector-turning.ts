import { canonicalRoroProvenance } from "./roro.ts";

export const CURRENT_SECTOR_TURNING_SCHEMA = 1 as const;
export const CURRENT_SECTOR_TURNING_SERIES = "sector-turning-attention-gauge-v0-pit-partial";
export const CURRENT_SECTOR_TURNING_MODEL = "BETA_BREADTH_SHALLOW_BOOST";
export const CURRENT_ATTENTION_THRESHOLD_MODEL = "attention-proxy-v0-causal252-m60-b90-a6-c4";

export type TurningAlertLevel = "NORMAL" | "WATCH" | "WARNING" | "PRIORITY";

export type TurningGaugeReading = {
  score: number;
  threshold: number;
  alert_margin: number;
  level: TurningAlertLevel;
  confirmation_score: number | null;
  noise_score: number | null;
  proxy_status: "DYNAMIC" | "BASE_FALLBACK";
};

export type SectorTurningRow = {
  sector: string;
  sector_name: string;
  attention_score: number | null;
  attention_band: "LOW" | "NORMAL" | "HIGH" | "UNAVAILABLE";
  rotation_out: TurningGaugeReading | null;
  rotation_in: TurningGaugeReading | null;
};

export type SectorTurningSnapshot = {
  schema_version: typeof CURRENT_SECTOR_TURNING_SCHEMA;
  series_version: typeof CURRENT_SECTOR_TURNING_SERIES;
  as_of: string;
  generated_at: string;
  input_cutoff: string;
  provenance_hash: string;
  model_versions: {
    turning_model: typeof CURRENT_SECTOR_TURNING_MODEL;
    attention_threshold: typeof CURRENT_ATTENTION_THRESHOLD_MODEL;
  };
  payload: {
    status: "RESEARCH_SHADOW_ONLY";
    pit_status: "PIT_PARTIAL/CURRENT_REVISED_HISTORY_ONLY";
    source_model: typeof CURRENT_SECTOR_TURNING_MODEL;
    threshold_model: typeof CURRENT_ATTENTION_THRESHOLD_MODEL;
    base_threshold: 90;
    priority_threshold: 95;
    threshold_floor: 85;
    threshold_ceiling: 95;
    feature_timing: "EOD_T_INPUT; ACTION_NO_EARLIER_THAN_T_PLUS_1";
    attention_proxy: string;
    rows: SectorTurningRow[];
  };
};

function object(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function onlyKeys(value: Record<string, unknown>, keys: readonly string[]) {
  const actual = Object.keys(value).sort();
  const expected = [...keys].sort();
  return actual.length === expected.length && actual.every((key, index) => key === expected[index]);
}

function bounded(value: unknown, low: number, high: number): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= low && value <= high;
}

function shortString(value: unknown, maximum = 500): value is string {
  return typeof value === "string" && value.length > 0 && value.length <= maximum;
}

function isoDate(value: unknown): value is string {
  if (typeof value !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(value)) return false;
  const parsed = new Date(`${value}T00:00:00Z`);
  return !Number.isNaN(parsed.valueOf()) && parsed.toISOString().slice(0, 10) === value;
}

function isoTimestamp(value: unknown): value is string {
  if (typeof value !== "string" || value.length > 100 ||
      !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$/.test(value)) return false;
  const parsed = new Date(value);
  return !Number.isNaN(parsed.valueOf()) && parsed.toISOString().slice(0, 19) === value.slice(0, 19);
}

function expectedLevel(score: number, threshold: number): TurningAlertLevel {
  if (score >= Math.max(95, threshold)) return "PRIORITY";
  if (score >= threshold) return "WARNING";
  if (score >= Math.max(0, threshold - 10)) return "WATCH";
  return "NORMAL";
}

function isGaugeReading(value: unknown): value is TurningGaugeReading {
  if (!object(value) || !onlyKeys(value, [
    "score", "threshold", "alert_margin", "level", "confirmation_score", "noise_score",
    "proxy_status",
  ]) || !bounded(value.score, 0, 100) || !bounded(value.threshold, 85, 95) ||
      !bounded(value.alert_margin, -95, 15) ||
      Math.abs(value.alert_margin - (value.score - value.threshold)) > 1e-3 ||
      expectedLevel(value.score, value.threshold) !== value.level ||
      !["DYNAMIC", "BASE_FALLBACK"].includes(String(value.proxy_status))) return false;
  const optionalScore = (candidate: unknown) => candidate === null || bounded(candidate, 0, 100);
  if (!optionalScore(value.confirmation_score) || !optionalScore(value.noise_score)) return false;
  return value.proxy_status === "DYNAMIC"
    ? value.confirmation_score !== null && value.noise_score !== null
    : value.threshold === 90;
}

function expectedAttentionBand(score: number | null) {
  if (score === null) return "UNAVAILABLE";
  if (score < 100 / 3) return "LOW";
  return score >= 200 / 3 ? "HIGH" : "NORMAL";
}

function isSectorTurningRow(value: unknown): value is SectorTurningRow {
  if (!object(value) || !onlyKeys(value, [
    "sector", "sector_name", "attention_score", "attention_band", "rotation_out", "rotation_in",
  ]) || !/^\d{4}$/.test(String(value.sector)) || !shortString(value.sector_name, 100) ||
      !(value.attention_score === null || bounded(value.attention_score, 0, 100)) ||
      expectedAttentionBand(value.attention_score as number | null) !== value.attention_band) return false;
  const validReading = (candidate: unknown) => candidate === null || isGaugeReading(candidate);
  return validReading(value.rotation_out) && validReading(value.rotation_in) &&
    (value.rotation_out !== null || value.rotation_in !== null);
}

export function isSectorTurningSnapshot(value: unknown): value is SectorTurningSnapshot {
  if (!object(value) || !onlyKeys(value, [
    "schema_version", "series_version", "as_of", "generated_at", "input_cutoff",
    "provenance_hash", "model_versions", "payload",
  ]) || value.schema_version !== CURRENT_SECTOR_TURNING_SCHEMA ||
      value.series_version !== CURRENT_SECTOR_TURNING_SERIES || !isoDate(value.as_of) ||
      !isoTimestamp(value.generated_at) || value.input_cutoff !== value.as_of ||
      typeof value.provenance_hash !== "string" || !/^[a-f0-9]{64}$/.test(value.provenance_hash) ||
      !object(value.model_versions) || !object(value.payload)) return false;
  const models = value.model_versions;
  const payload = value.payload;
  if (!onlyKeys(models, ["turning_model", "attention_threshold"]) ||
      models.turning_model !== CURRENT_SECTOR_TURNING_MODEL ||
      models.attention_threshold !== CURRENT_ATTENTION_THRESHOLD_MODEL ||
      !onlyKeys(payload, [
        "status", "pit_status", "source_model", "threshold_model", "base_threshold",
        "priority_threshold", "threshold_floor", "threshold_ceiling", "feature_timing",
        "attention_proxy", "rows",
      ])) return false;
  const rows = payload.rows;
  return payload.status === "RESEARCH_SHADOW_ONLY" &&
    payload.pit_status === "PIT_PARTIAL/CURRENT_REVISED_HISTORY_ONLY" &&
    payload.source_model === CURRENT_SECTOR_TURNING_MODEL &&
    payload.threshold_model === CURRENT_ATTENTION_THRESHOLD_MODEL &&
    payload.base_threshold === 90 && payload.priority_threshold === 95 &&
    payload.threshold_floor === 85 && payload.threshold_ceiling === 95 &&
    payload.feature_timing === "EOD_T_INPUT; ACTION_NO_EARLIER_THAN_T_PLUS_1" &&
    shortString(payload.attention_proxy) && Array.isArray(rows) &&
    rows.length >= 2 && rows.length <= 24 && rows.every(isSectorTurningRow) &&
    new Set(rows.map((row) => row.sector)).size === rows.length;
}

export async function sectorTurningProvenanceHash(snapshot: SectorTurningSnapshot) {
  const material = {
    series_version: snapshot.series_version,
    date: snapshot.as_of,
    input_cutoff: snapshot.input_cutoff,
    model_versions: snapshot.model_versions,
    payload: snapshot.payload,
  };
  const digest = await crypto.subtle.digest(
    "SHA-256", new TextEncoder().encode(canonicalRoroProvenance(material)),
  );
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

export async function verifySectorTurningProvenance(snapshot: SectorTurningSnapshot) {
  return snapshot.provenance_hash === await sectorTurningProvenanceHash(snapshot);
}
