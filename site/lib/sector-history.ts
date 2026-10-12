import { SECTOR_RORO_TICKERS } from "./sector-roro";

export const SECTOR_HISTORY_SCHEMA = 1 as const;
export const SECTOR_HISTORY_SERIES =
  "sector-roro-context-v1-spy-rel-ridge-a10-m156-w20__weekly-research-2025-r2";

export type SectorHistoryRow = {
  decision_date: string;
  sector: (typeof SECTOR_RORO_TICKERS)[number];
  sector_name: string;
  forecast_rank: number;
  predicted_excess_20d: number;
  actual_excess_20d: number | null;
  alpha_hit: boolean | null;
  quadrant: "LEADING" | "WEAKENING" | "LAGGING" | "IMPROVING";
  quadrant_transition: string;
  tail_heading: string;
  roro_fit: number | null;
  roro_context: "N/A" | "FIT" | "RORO_MISMATCH";
  fit_change: "N/A" | "—" | "MISMATCH → FIT" | "FIT → MISMATCH";
  sector_status: "N/A" | "SELECT" | "WATCH" | "AVOID";
};

export type SectorHistory = {
  schema_version: typeof SECTOR_HISTORY_SCHEMA;
  series_version: typeof SECTOR_HISTORY_SERIES;
  model_version: string;
  start: string;
  as_of: string;
  generated_at: string;
  cadence: "WEEKLY";
  rows: SectorHistoryRow[];
};

function object(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function finite(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isoDate(value: unknown): value is string {
  if (typeof value !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(value)) return false;
  const parsed = new Date(`${value}T00:00:00Z`);
  return !Number.isNaN(parsed.valueOf()) && parsed.toISOString().slice(0, 10) === value;
}

function validRow(value: unknown): value is SectorHistoryRow {
  if (!object(value)) return false;
  return isoDate(value.decision_date) && value.decision_date >= "2025-01-01" &&
    SECTOR_RORO_TICKERS.includes(value.sector as SectorHistoryRow["sector"]) &&
    typeof value.sector_name === "string" && value.sector_name.length > 0 && value.sector_name.length <= 100 &&
    Number.isInteger(value.forecast_rank) && Number(value.forecast_rank) >= 1 && Number(value.forecast_rank) <= 11 &&
    finite(value.predicted_excess_20d) && (value.actual_excess_20d === null || finite(value.actual_excess_20d)) &&
    (value.alpha_hit === null || typeof value.alpha_hit === "boolean") &&
    ["LEADING", "WEAKENING", "LAGGING", "IMPROVING"].includes(String(value.quadrant)) &&
    typeof value.quadrant_transition === "string" && value.quadrant_transition.length <= 100 &&
    typeof value.tail_heading === "string" && value.tail_heading.length <= 100 &&
    (value.roro_fit === null || (finite(value.roro_fit) && value.roro_fit >= 0 && value.roro_fit <= 1)) &&
    ["N/A", "FIT", "RORO_MISMATCH"].includes(String(value.roro_context)) &&
    ["N/A", "—", "MISMATCH → FIT", "FIT → MISMATCH"].includes(String(value.fit_change)) &&
    ["N/A", "SELECT", "WATCH", "AVOID"].includes(String(value.sector_status));
}

export function isSectorHistory(value: unknown): value is SectorHistory {
  if (!object(value) || value.schema_version !== SECTOR_HISTORY_SCHEMA ||
      value.series_version !== SECTOR_HISTORY_SERIES || !isoDate(value.start) || !isoDate(value.as_of) ||
      value.start < "2025-01-01" || value.start > value.as_of || value.cadence !== "WEEKLY" ||
      typeof value.model_version !== "string" || value.model_version.length === 0 ||
      typeof value.generated_at !== "string" || Number.isNaN(new Date(value.generated_at).valueOf()) ||
      !Array.isArray(value.rows) || value.rows.length < 11 || value.rows.length > 2_000 ||
      !value.rows.every(validRow)) return false;
  const rows = value.rows as SectorHistoryRow[];
  if (rows[0].decision_date !== value.start || rows.at(-1)?.decision_date !== value.as_of) return false;
  const byDate = new Map<string, SectorHistoryRow[]>();
  for (const row of rows) byDate.set(row.decision_date, [...(byDate.get(row.decision_date) ?? []), row]);
  return [...byDate.values()].every((group) =>
    group.length === SECTOR_RORO_TICKERS.length &&
    new Set(group.map((row) => row.sector)).size === SECTOR_RORO_TICKERS.length &&
    new Set(group.map((row) => row.forecast_rank)).size === SECTOR_RORO_TICKERS.length
  );
}
