import { ROTATION_SECTORS } from './rotation-entry.ts';

export const HYBRID_SERIES = 'price-hybrid-boost-in-extra-out-2026-v1';
export type HybridRow = {ticker:string; sector:string; status:'READY'|'MISSING_FEATURES'; direction:'IN'|'OUT'|null; model?:string; signal:'IN'|'OUT'|'NONE'|null; score:number|null; threshold:number|null};
export type HybridSnapshot = {series_version:string; as_of:string; decision_date:string|null; latest_price_date:string|null; status:string; rows:HybridRow[]};
const record = (v:unknown):v is Record<string,unknown> => !!v && typeof v==='object' && !Array.isArray(v);
const day = (v:unknown):v is string => typeof v==='string' && /^\d{4}-\d{2}-\d{2}$/.test(v) && Number.isFinite(Date.parse(v)) && new Date(v).toISOString().slice(0,10)===v;
const unit = (v:unknown):v is number => typeof v==='number' && Number.isFinite(v) && v>=0 && v<=1;
export function isHybridSnapshot(v:unknown):v is HybridSnapshot {
  if (!record(v) || v.series_version!==HYBRID_SERIES || !day(v.as_of) || !Array.isArray(v.rows)) return false;
  if (v.decision_date!==null && (!day(v.decision_date) || v.decision_date>v.as_of)) return false;
  if (v.latest_price_date!==null && (!day(v.latest_price_date) || v.latest_price_date>v.as_of)) return false;
  const ready = v.status==='READY' || v.status==='PARTIAL';
  if (!ready) return ['ERROR','MODEL_NOT_READY','NO_PRICES','NO_BENCHMARK','NO_COMPLETED_WEEK','STALE_PRICES','MODEL_YEAR_MISMATCH','INSUFFICIENT_HISTORY'].includes(String(v.status)) && v.rows.length===0;
  if (!day(v.decision_date) || !day(v.latest_price_date) || v.decision_date>v.latest_price_date || !v.decision_date.startsWith('2026-') || (Date.parse(v.as_of)-Date.parse(v.decision_date))/86400000>10) return false;
  if (v.rows.length!==11 || new Set(v.rows.map(r=>r?.ticker)).size!==11) return false;
  if (!v.rows.every(r=>{
    if (!record(r) || !ROTATION_SECTORS.includes(r.ticker as typeof ROTATION_SECTORS[number]) || typeof r.sector!=='string') return false;
    if (r.status==='MISSING_FEATURES') return r.signal===null && r.score===null && r.threshold===null && r.direction===null;
    if (r.status!=='READY' || !unit(r.score) || !unit(r.threshold) || !['IN','OUT'].includes(String(r.direction))) return false;
    return r.model===(r.direction==='IN'?'PRICE_BOOST':'PRICE_EXTRA') && r.signal===(r.score>=r.threshold?r.direction:'NONE');
  })) return false;
  return (v.status==='READY')===v.rows.every(r=>r.status==='READY');
}
