import { canonicalRoroProvenance } from "./roro";

export const SECTOR_RORO_TICKERS = [
  "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY",
] as const;

export const CURRENT_SECTOR_RORO_SCHEMA = 2 as const;
export const CURRENT_SECTOR_RORO_MODEL = "sector-roro-context-v1-spy-rel-ridge-a10-m156-w20";
export const CURRENT_RORO_SERIES =
  "roro-3axis-v1-w756-m252-p126-u20__roro-phase-v1__p-c85-isqc__shadow-r21-breadth-correction";
export const CURRENT_SECTOR_RORO_SERIES =
  `${CURRENT_SECTOR_RORO_MODEL}__${CURRENT_RORO_SERIES}__immutable-r1`;
const CURRENT_CONTEXT_TYPE = "RORO_EXACT_DATE_BETA_FIT";

const SECTOR_NAMES: Record<(typeof SECTOR_RORO_TICKERS)[number], string> = {
  XLB: "Materials",
  XLC: "Communication Services",
  XLE: "Energy",
  XLF: "Financials",
  XLI: "Industrials",
  XLK: "Information Technology",
  XLP: "Consumer Staples",
  XLRE: "Real Estate",
  XLU: "Utilities",
  XLV: "Health Care",
  XLY: "Consumer Discretionary",
};

export type SectorRoroRow = {
  sector: (typeof SECTOR_RORO_TICKERS)[number];
  sector_name: string;
  rrg_score: number;
  predicted_excess_20d: number;
  forecast_score: number;
  forecast_rank: number;
  signal: string;
  quadrant: "LEADING" | "WEAKENING" | "LAGGING" | "IMPROVING";
  quadrant_transition: string;
  tail_heading: string;
  delta_rs_ratio: number;
  delta_rs_momentum: number;
  sector_beta_60: number;
  beta_rank: number;
  roro_fit: number;
  alpha_positive: boolean;
  alpha_top3: boolean;
  rrg_supportive: boolean;
  roro_context: "FIT" | "RORO_MISMATCH";
  alignment: "ALIGNED" | "DIVERGENCE";
  sector_status: "SELECT" | "WATCH" | "AVOID";
};

export type RotationEvent = "EARLY BUILD" | "ROTATION IN" | "ROTATION OUT" | "—";

export function rotationEvent(row: Pick<SectorRoroRow, "quadrant" | "quadrant_transition" | "tail_heading">): RotationEvent {
  const [before, after] = row.quadrant_transition.split("->");
  const strong = (quadrant: string) => quadrant === "LEADING" || quadrant === "IMPROVING";
  const weak = (quadrant: string) => quadrant === "LAGGING" || quadrant === "WEAKENING";
  if (weak(before) && strong(after)) return "ROTATION IN";
  if (strong(before) && weak(after)) return "ROTATION OUT";
  if (weak(row.quadrant) && ["NE_STRENGTHENING", "NW_EARLY_RECOVERY"].includes(row.tail_heading)) return "EARLY BUILD";
  return "—";
}

type SectorRoroShadowSnapshotBase = {
  schema_version: 1 | typeof CURRENT_SECTOR_RORO_SCHEMA;
  series_version: string;
  as_of: string;
  generated_at: string;
  input_cutoff: string;
  provenance_hash: string;
  model_versions: {
    sector_roro: string;
    roro: string;
  };
  payload: {
    forecast_target: string;
    training_rows: number;
    training_weeks: number;
    training_start: string;
    training_end: string;
    latest_training_label_end: string;
    alpha: number;
    top_k: number;
    context_type: string;
    status_contract: string;
    relative_benchmark: "SPY";
    market_context: "NQ=F";
    market_risk_score: number;
    risk_regime: "GREEN" | "YELLOW" | "RED";
    roro_trajectory_5d: number | null;
    roro_data_status: string;
    rows: SectorRoroRow[];
  };
};

export type LegacySectorRoroShadowSnapshot = SectorRoroShadowSnapshotBase & {
  schema_version: 1;
};

export type SectorRoroShadowSnapshotV2 = SectorRoroShadowSnapshotBase & {
  schema_version: typeof CURRENT_SECTOR_RORO_SCHEMA;
};

export type SectorRoroShadowSnapshot =
  LegacySectorRoroShadowSnapshot | SectorRoroShadowSnapshotV2;

function object(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function onlyKeys(value: Record<string, unknown>, keys: readonly string[]) {
  const actual = Object.keys(value).sort();
  const expected = [...keys].sort();
  return actual.length === expected.length && actual.every((key, index) => key === expected[index]);
}

function finite(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function bounded(value: unknown, low: number, high: number): value is number {
  return finite(value) && value >= low && value <= high;
}

function integer(value: unknown, low: number, high: number): value is number {
  return bounded(value, low, high) && Number.isInteger(value);
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

function isSectorRoroRow(value: unknown): value is SectorRoroRow {
  if (!object(value) || !onlyKeys(value, [
    "sector", "sector_name", "rrg_score", "predicted_excess_20d", "forecast_score",
    "forecast_rank", "signal", "quadrant", "quadrant_transition", "tail_heading",
    "delta_rs_ratio", "delta_rs_momentum", "sector_beta_60", "beta_rank", "roro_fit",
    "alpha_positive", "alpha_top3", "rrg_supportive", "roro_context", "alignment",
    "sector_status",
  ])) return false;
  const sector = value.sector as SectorRoroRow["sector"];
  return (
    SECTOR_RORO_TICKERS.includes(sector) && value.sector_name === SECTOR_NAMES[sector] &&
    bounded(value.rrg_score, 0, 1) &&
    finite(value.predicted_excess_20d) && bounded(value.forecast_score, 0, 1) &&
    integer(value.forecast_rank, 1, 11) &&
    ["MODEL_TOP3", "POSITIVE_WATCH", "NEGATIVE"].includes(String(value.signal)) &&
    ["LEADING", "WEAKENING", "LAGGING", "IMPROVING"].includes(String(value.quadrant)) &&
    shortString(value.quadrant_transition, 100) && shortString(value.tail_heading, 100) &&
    finite(value.delta_rs_ratio) && finite(value.delta_rs_momentum) &&
    finite(value.sector_beta_60) && bounded(value.beta_rank, 0, 1) && bounded(value.roro_fit, 0, 1) &&
    typeof value.alpha_positive === "boolean" && typeof value.alpha_top3 === "boolean" &&
    typeof value.rrg_supportive === "boolean" && ["FIT", "RORO_MISMATCH"].includes(String(value.roro_context)) &&
    ["ALIGNED", "DIVERGENCE"].includes(String(value.alignment)) &&
    ["SELECT", "WATCH", "AVOID"].includes(String(value.sector_status))
  );
}

export function isSectorRoroShadowSnapshot(value: unknown): value is SectorRoroShadowSnapshot {
  if (!object(value) || !onlyKeys(value, [
    "schema_version", "series_version", "as_of", "generated_at", "input_cutoff",
    "provenance_hash", "model_versions", "payload",
  ]) || (value.schema_version !== 1 && value.schema_version !== CURRENT_SECTOR_RORO_SCHEMA) ||
      !object(value.model_versions) || !object(value.payload)) return false;
  const models = value.model_versions;
  const payload = value.payload;
  if (!onlyKeys(models, ["sector_roro", "roro"]) || !onlyKeys(payload, [
    "forecast_target", "training_rows", "training_weeks", "training_start", "training_end",
    "latest_training_label_end", "alpha", "top_k", "context_type", "status_contract",
    "relative_benchmark", "market_context", "market_risk_score", "risk_regime",
    "roro_trajectory_5d", "roro_data_status", "rows",
  ])) return false;
  const rows = payload.rows;
  const valid = (
    shortString(value.series_version) && isoDate(value.as_of) && value.as_of >= "2026-01-07" &&
    isoTimestamp(value.generated_at) && value.input_cutoff === value.as_of &&
    typeof value.provenance_hash === "string" && /^[a-f0-9]{64}$/.test(value.provenance_hash) &&
    shortString(models.sector_roro) && shortString(models.roro) &&
    shortString(payload.forecast_target) && integer(payload.training_rows, 1, 10_000_000) &&
    integer(payload.training_weeks, 1, 100_000) &&
    isoDate(payload.training_start) && isoDate(payload.training_end) && isoDate(payload.latest_training_label_end) &&
    payload.training_start <= payload.training_end && payload.training_end < value.as_of &&
    payload.latest_training_label_end < value.as_of &&
    finite(payload.alpha) && payload.alpha >= 0 && integer(payload.top_k, 1, 11) &&
    shortString(payload.context_type) && payload.status_contract === "SELECT/WATCH/AVOID; no forced Top3 fill" &&
    payload.relative_benchmark === "SPY" && payload.market_context === "NQ=F" &&
    bounded(payload.market_risk_score, 0, 100) && ["GREEN", "YELLOW", "RED"].includes(String(payload.risk_regime)) &&
    (payload.roro_trajectory_5d === null || finite(payload.roro_trajectory_5d)) && shortString(payload.roro_data_status) &&
    Array.isArray(rows) && rows.length === SECTOR_RORO_TICKERS.length && rows.every(isSectorRoroRow) &&
    new Set(rows.map((row) => row.sector)).size === SECTOR_RORO_TICKERS.length &&
    new Set(rows.map((row) => row.forecast_rank)).size === SECTOR_RORO_TICKERS.length
  );
  if (!valid || value.schema_version === 1) return valid;
  return (
    value.series_version === CURRENT_SECTOR_RORO_SERIES &&
    models.sector_roro === CURRENT_SECTOR_RORO_MODEL &&
    models.roro === CURRENT_RORO_SERIES &&
    payload.alpha === 10 && payload.top_k === 3 &&
    payload.context_type === CURRENT_CONTEXT_TYPE
  );
}

export function isCurrentSectorRoroShadowSnapshot(
  value: unknown,
): value is SectorRoroShadowSnapshotV2 {
  return isSectorRoroShadowSnapshot(value) &&
    value.schema_version === CURRENT_SECTOR_RORO_SCHEMA;
}

export async function sectorRoroProvenanceHash(
  snapshot: SectorRoroShadowSnapshotV2,
): Promise<string> {
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
  return Array.from(new Uint8Array(digest), (byte) =>
    byte.toString(16).padStart(2, "0")).join("");
}

export async function verifySectorRoroProvenance(
  snapshot: SectorRoroShadowSnapshot,
): Promise<boolean> {
  return snapshot.schema_version === CURRENT_SECTOR_RORO_SCHEMA &&
    snapshot.provenance_hash === await sectorRoroProvenanceHash(snapshot);
}
