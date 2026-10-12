import { ROTATION_SECTORS } from './rotation-entry';

export const SECTOR_TRANSITION_SERIES = 'sector-transition-price-multiscale-v1';

export type SectorTransitionRow = {
  ticker: string;
  sector: string;
  state: 'UP_ACTIVE' | 'DOWN_ACTIVE';
  direction: 'ROTATION_IN' | 'ROTATION_OUT' | null;
  score: number | null;
  status: 'SCORED' | 'N/A_COOLDOWN_OR_NOT_SCORABLE';
};

export type SectorTransitionSnapshot = {
  schema_version: 1;
  series_version: typeof SECTOR_TRANSITION_SERIES;
  as_of: string;
  score_semantics: 'CAUSAL_TRAILING_PERCENTILE';
  high_threshold: 80;
  rows: SectorTransitionRow[];
};

export function isSectorTransitionSnapshot(value: unknown): value is SectorTransitionSnapshot {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false;
  const v = value as SectorTransitionSnapshot;
  if (v.schema_version !== 1 || v.series_version !== SECTOR_TRANSITION_SERIES ||
      !/^\d{4}-\d{2}-\d{2}$/.test(v.as_of) || !Number.isFinite(Date.parse(v.as_of)) ||
      v.score_semantics !== 'CAUSAL_TRAILING_PERCENTILE' || v.high_threshold !== 80 ||
      !Array.isArray(v.rows) || v.rows.length !== ROTATION_SECTORS.length) return false;
  const seen = new Set<string>();
  return v.rows.every(row => {
    if (!row || typeof row !== 'object' || !ROTATION_SECTORS.includes(row.ticker) || seen.has(row.ticker) ||
        typeof row.sector !== 'string' || !['UP_ACTIVE','DOWN_ACTIVE'].includes(row.state) ||
        !['SCORED','N/A_COOLDOWN_OR_NOT_SCORABLE'].includes(row.status)) return false;
    seen.add(row.ticker);
    if (row.status === 'SCORED') return ['ROTATION_IN','ROTATION_OUT'].includes(String(row.direction)) &&
      typeof row.score === 'number' && Number.isFinite(row.score) && row.score >= 0 && row.score <= 100;
    return row.direction === null && row.score === null;
  });
}
