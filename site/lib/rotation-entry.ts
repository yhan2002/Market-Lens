export const ROTATION_ENTRY_SERIES = 'entry-rank5-change-top3-h10-p5-forward-v1';
export const ROTATION_SECTORS = ['XLB','XLC','XLE','XLF','XLI','XLK','XLP','XLRE','XLU','XLV','XLY'];
export type RotationEntry = {
  series_version: typeof ROTATION_ENTRY_SERIES;
  as_of: string; recorded_at: string;
  phase: 'HISTORICAL_REPLAY' | 'BASELINE' | 'ACTIVE' | 'EXPIRED_STATE_ONLY';
  alert: boolean; threshold: number; nominated: string; nearest4: string; score: number;
  ranks: Record<string, number>; rank_change5: Record<string, number>;
  days_since_last_alert: number | null;
};
export function isRotationEntry(value: unknown): value is RotationEntry {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false;
  const v = value as RotationEntry;
  const keys = ['series_version','as_of','recorded_at','phase','alert','threshold','nominated','nearest4','score','ranks','rank_change5','days_since_last_alert'];
  if (Object.keys(v).length !== keys.length || !keys.every(k => k in v)) return false;
  if (v.series_version !== ROTATION_ENTRY_SERIES || !/^\d{4}-\d{2}-\d{2}$/.test(v.as_of) ||
      !Number.isFinite(Date.parse(v.as_of)) || new Date(v.as_of).toISOString().slice(0,10) !== v.as_of ||
      !Number.isFinite(Date.parse(v.recorded_at)) || v.threshold !== 6 || typeof v.alert !== 'boolean' ||
      !['HISTORICAL_REPLAY','BASELINE','ACTIVE','EXPIRED_STATE_ONLY'].includes(v.phase)) return false;
  const validMap = (m: Record<string, number>) => m && Object.keys(m).length === 11 && ROTATION_SECTORS.every(t => Number.isInteger(m[t]));
  if (!validMap(v.ranks) || !validMap(v.rank_change5) || new Set(Object.values(v.ranks)).size !== 11 ||
      ROTATION_SECTORS.some(t => v.ranks[t] < 1 || v.ranks[t] > 11 || Math.abs(v.rank_change5[t]) > 10)) return false;
  const previous = ROTATION_SECTORS.map(t => v.ranks[t] + v.rank_change5[t]);
  if (new Set(previous).size !== 11 || previous.some(r => r < 1 || r > 11)) return false;
  const outside = ROTATION_SECTORS.filter(t => v.ranks[t] > 3);
  const nominee = [...outside].sort((a,b) => v.rank_change5[b]-v.rank_change5[a] || a.localeCompare(b))[0];
  if (v.nominated !== nominee || v.score !== v.rank_change5[nominee] || v.ranks[v.nearest4] !== 4) return false;
  const spacing = v.days_since_last_alert;
  if (spacing !== null && (!Number.isInteger(spacing) || spacing < 1)) return false;
  if (v.phase === 'HISTORICAL_REPLAY') return v.as_of >= '2026-06-01' && v.as_of <= '2026-09-04' &&
    v.alert === (v.score >= 6 && (spacing === null || spacing >= 10));
  if (v.phase === 'BASELINE') return v.as_of === '2026-09-08' && !v.alert && spacing === null;
  if (v.as_of <= '2026-09-08' || (v.phase === 'ACTIVE') !== v.as_of.startsWith('2026-')) return false;
  return v.alert === (v.phase === 'ACTIVE' && v.score >= 6 && (spacing === null || spacing >= 10));
}
