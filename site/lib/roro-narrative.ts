export type RoroNarrative = {
  as_of: string; series_version: string; provenance_hash: string;
  summary: string; model: string; generated_at: string;
};

export function isRoroNarrative(value: unknown): value is RoroNarrative {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const v = value as Record<string, unknown>;
  const keys = ["as_of", "series_version", "provenance_hash", "summary", "model", "generated_at"];
  if (Object.keys(v).length !== keys.length || !keys.every(key => typeof v[key] === "string")) return false;
  const date = v.as_of as string;
  return /^\d{4}-\d{2}-\d{2}$/.test(date) && Number.isFinite(Date.parse(date)) &&
    new Date(date).toISOString().slice(0, 10) === date &&
    /^[a-f0-9]{64}$/.test(v.provenance_hash as string) &&
    (v.series_version as string).length > 0 && (v.series_version as string).length <= 500 &&
    (v.summary as string).trim().length > 0 && [...v.summary as string].length <= 180 &&
    !/[\x00-\x1f\x7f]/.test(v.summary as string) &&
    /^[a-zA-Z0-9._:/-]{1,120}$/.test(v.model as string) &&
    /^\d{4}-\d{2}-\d{2}T/.test(v.generated_at as string) && Number.isFinite(Date.parse(v.generated_at as string));
}
