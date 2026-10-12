import { env } from "cloudflare:workers";
import { and, desc, eq, lte } from "drizzle-orm";
import { getDb } from "../../../db";
import { roroShadowSnapshots } from "../../../db/schema";
import { applyBreadthRepairs, BREADTH_BASE_SERIES, BREADTH_REPAIR_SERIES, isRoroShadowSnapshot, roroHistoryFallbackSeries, verifyRoroProvenance } from "../../../lib/roro";
import { isRoroNarrative } from "../../../lib/roro-narrative";
import { benchmarkPrices, CHART_BENCHMARK } from "../../../lib/chart-benchmark";

export const dynamic = "force-dynamic";

async function ensureTable() {
  await env.DB.prepare(`
    CREATE TABLE IF NOT EXISTS roro_shadow_snapshots (
      id TEXT PRIMARY KEY,
      series_version TEXT NOT NULL,
      as_of_key INTEGER NOT NULL,
      payload TEXT NOT NULL,
      benchmark_close REAL,
      generated_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    )
  `).run();
  await env.DB.prepare(`
    CREATE UNIQUE INDEX IF NOT EXISTS uq_roro_shadow_series_date_v1
    ON roro_shadow_snapshots(series_version, as_of_key)
  `).run();
}

function bearer(request: Request) {
  const header = request.headers.get("authorization") ?? "";
  return header.startsWith("Bearer ") ? header.slice(7) : "";
}

function allowed(request: Request, mode: "view" | "upload") {
  const token = bearer(request);
  const upload = env.UPLOAD_TOKEN as string | undefined;
  const view = env.VIEW_TOKEN as string | undefined;
  return Boolean(token && upload && (mode === "upload" ? token === upload : token === upload || token === view));
}

function dateKey(value: string): number | null {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) return null;
  const parsed = new Date(`${value}T00:00:00Z`);
  return !Number.isNaN(parsed.valueOf()) && parsed.toISOString().slice(0, 10) === value
    ? Number(value.replaceAll("-", "")) : null;
}

function validSeries(value: unknown): value is string {
  return typeof value === "string" && value.length > 0 && value.length <= 500 &&
    [...value].every((character) => character.charCodeAt(0) >= 32);
}

function rowId(series: string, key: number) {
  return `${series}:${key}`;
}

function readRows(series: string, requestedKey: number | null, limit: number) {
  return getDb().select().from(roroShadowSnapshots)
    .where(requestedKey === null
      ? eq(roroShadowSnapshots.seriesVersion, series)
      : and(eq(roroShadowSnapshots.seriesVersion, series), lte(roroShadowSnapshots.asOfKey, requestedKey)))
    .orderBy(desc(roroShadowSnapshots.asOfKey)).limit(limit);
}

export async function GET(request: Request) {
  if (!allowed(request, "view")) return Response.json({ error: "unauthorized" }, { status: 401 });
  const url = new URL(request.url);
  const requestedDate = url.searchParams.get("date");
  const requestedKey = requestedDate === null ? null : dateKey(requestedDate);
  if (requestedDate !== null && requestedKey === null) return Response.json({ error: "invalid date" }, { status: 400 });
  const historyParam = url.searchParams.get("history");
  const historyLimit = historyParam === null ? 1 : Number(historyParam);
  if (!Number.isInteger(historyLimit) || historyLimit < 1 || historyLimit > 40) {
    return Response.json({ error: "history must be an integer from 1 to 40" }, { status: 400 });
  }
  const requestedSeries = url.searchParams.get("series");
  const configuredSeries = env.ACTIVE_RORO_SERIES as string | undefined;
  if (requestedSeries !== null && bearer(request) !== (env.UPLOAD_TOKEN as string | undefined)) {
    return Response.json({ error: "series override requires upload authorization" }, { status: 403 });
  }
  const series = requestedSeries ?? configuredSeries;
  if (!validSeries(series)) return Response.json({ error: "active RORO series is not configured" }, { status: 503 });
  await ensureTable();
  let rows = await readRows(series, requestedKey, historyLimit);
  if (requestedSeries === null && series === BREADTH_BASE_SERIES) {
    rows = applyBreadthRepairs(rows, await readRows(BREADTH_REPAIR_SERIES, requestedKey, historyLimit));
  }
  const fallbackSeries = requestedSeries === null ? roroHistoryFallbackSeries(series) : [];
  if (historyParam !== null && rows.length < historyLimit) {
    const byDate = new Map(rows.map((row) => [row.asOfKey, row]));
    for (const previousSeries of fallbackSeries) {
      const previousRows = await readRows(previousSeries, requestedKey, historyLimit);
      for (const row of previousRows) if (!byDate.has(row.asOfKey)) byDate.set(row.asOfKey, row);
      if (byDate.size >= historyLimit) break;
    }
    rows = [...byDate.values()].sort((left, right) => right.asOfKey - left.asOfKey).slice(0, historyLimit);
  } else if (!rows.length && fallbackSeries.length) {
    for (const previousSeries of fallbackSeries) {
      rows = await readRows(previousSeries, requestedKey, 1);
      if (rows.length) break;
    }
  }
  if (!rows.length) return Response.json({ error: "no RORO SHADOW snapshot" }, { status: 404 });
  if (historyParam !== null) {
    type Price = { as_of_key: number; close: number; previous_as_of: string | null; previous_close: number | null };
    const prices = new Map<number, Price>();
    try {
      const result = await env.DB.prepare(`SELECT as_of_key, close, previous_as_of, previous_close
        FROM nasdaq100_chart_prices WHERE as_of_key IN (${rows.map(() => "?").join(",")})`)
        .bind(...rows.map((row) => row.asOfKey)).all<Price>();
      for (const price of result.results) prices.set(price.as_of_key, price);
    } catch (error) {
      console.error("Nasdaq100 chart prices unavailable", error);
      // Keep risk history available; never fall back to legacy NQ closes.
    }
    return Response.json(rows.reverse().map((row) => ({
      snapshot: JSON.parse(row.payload), benchmark_ticker: CHART_BENCHMARK,
      benchmark_close: prices.get(row.asOfKey)?.close ?? null,
      benchmark_previous_as_of: prices.get(row.asOfKey)?.previous_as_of ?? null,
      benchmark_previous_close: prices.get(row.asOfKey)?.previous_close ?? null,
    })), {
      headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" },
    });
  }
  if (url.searchParams.get("include_narrative") === "1") {
    const snapshot = JSON.parse(rows[0].payload);
    const stored = await env.DB.prepare("SELECT payload FROM roro_narratives WHERE id=? AND provenance_hash=?")
      .bind(rows[0].id, snapshot.provenance_hash).first<{payload: string}>();
    return Response.json({ snapshot, narrative: stored ? JSON.parse(stored.payload) : null },
      { headers: { "cache-control": "no-store" } });
  }
  return new Response(rows[0].payload, { headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" } });
}

export async function POST(request: Request) {
  if (!allowed(request, "upload")) return Response.json({ error: "unauthorized" }, { status: 401 });
  const candidate: unknown = await request.json();
  if (candidate && typeof candidate === "object" && "narrative" in candidate) {
    const narrative = candidate.narrative;
    if (Object.keys(candidate).length !== 1 || !isRoroNarrative(narrative)) {
      return Response.json({ error: "invalid narrative" }, { status: 400 });
    }
    const id = rowId(narrative.series_version, dateKey(narrative.as_of)!);
    const stored = await env.DB.prepare("SELECT payload FROM roro_shadow_snapshots WHERE id=?")
      .bind(id).first<{payload: string}>();
    if (!stored || JSON.parse(stored.payload).provenance_hash !== narrative.provenance_hash) {
      return Response.json({ error: "exact snapshot provenance required" }, { status: 409 });
    }
    // Narrative is separate: never rewrite a score snapshot or its provenance.
    await env.DB.prepare(`INSERT INTO roro_narratives (id, provenance_hash, payload) VALUES (?, ?, ?)
      ON CONFLICT(id) DO UPDATE SET provenance_hash=excluded.provenance_hash, payload=excluded.payload`)
      .bind(id, narrative.provenance_hash, JSON.stringify(narrative)).run();
    return Response.json({ ok: true, as_of: narrative.as_of });
  }
  const prices = benchmarkPrices(candidate);
  if (prices) {
    try {
      const results = await env.DB.batch(prices.map((price) => env.DB.prepare(`
        INSERT INTO nasdaq100_chart_prices (as_of_key, close, previous_as_of, previous_close, updated_at)
        VALUES (?, ?, ?, ?, ?) ON CONFLICT(as_of_key) DO UPDATE SET close=excluded.close,
        previous_as_of=excluded.previous_as_of, previous_close=excluded.previous_close, updated_at=excluded.updated_at
      `).bind(dateKey(price.as_of), price.close, price.previous_as_of, price.previous_close, new Date().toISOString())));
      return Response.json({ ok: true, rows: prices.length, updated: results.reduce<number>((total, result) => total + Number((result as { meta?: { changes?: number } }).meta?.changes ?? 0), 0) });
    } catch (error) {
      console.error("Nasdaq100 chart upload failed", error);
      return Response.json({ error: "Nasdaq100 chart storage unavailable; retry after migration" }, { status: 503 });
    }
  }
  const payload = Array.isArray(candidate) && candidate.length === 1 ? candidate[0] : candidate;
  if (!isRoroShadowSnapshot(payload)) {
    return Response.json({ error: "invalid RORO SHADOW payload" }, { status: 400 });
  }
  if (payload.schema_version !== 4) {
    return Response.json({ error: "only schema v4 uploads are accepted" }, { status: 400 });
  }
  const snapshot = payload;
  const series = snapshot.series_version;
  if (!validSeries(series)) {
    return Response.json({ error: "one valid series required" }, { status: 400 });
  }
  const key = dateKey(snapshot.as_of);
  if (key === null || snapshot.input_cutoff !== snapshot.as_of) {
    return Response.json({ error: "invalid causal date" }, { status: 400 });
  }
  if (!await verifyRoroProvenance(snapshot)) {
    return Response.json({ error: "invalid RORO provenance hash" }, { status: 400 });
  }
  const serialized = JSON.stringify(snapshot);
  if (serialized.length > 100_000) return Response.json({ error: "RORO snapshot too large" }, { status: 413 });
  await ensureTable();
  if (series === BREADTH_REPAIR_SERIES) {
    const original = await env.DB.prepare("SELECT payload FROM roro_shadow_snapshots WHERE id=?")
      .bind(rowId(BREADTH_BASE_SERIES, key)).first<{ payload: string }>();
    if (!original || JSON.parse(original.payload).payload.breadth_risk_percentile != null) {
      return Response.json({ error: "breadth repair requires an original missing-breadth snapshot" }, { status: 409 });
    }
  }
  const id = rowId(series, key);
  const existing = await getDb().select().from(roroShadowSnapshots)
    .where(eq(roroShadowSnapshots.id, id)).limit(1);
  if (existing.length) {
    if (existing[0].payload !== serialized) {
      return Response.json({ error: "immutable RORO history conflict", as_of: snapshot.as_of }, { status: 409 });
    }
    return Response.json({ ok: true, series, rows: 1, inserted: 0, idempotent: 1, as_of: snapshot.as_of });
  }
  const now = new Date().toISOString();
  const result = await env.DB.prepare(`
    INSERT OR IGNORE INTO roro_shadow_snapshots
      (id, series_version, as_of_key, payload, generated_at, updated_at)
    VALUES (?, ?, ?, ?, ?, ?)
  `).bind(id, series, key, serialized, snapshot.generated_at, now).run();
  const stored = await getDb().select().from(roroShadowSnapshots)
    .where(and(eq(roroShadowSnapshots.id, id), eq(roroShadowSnapshots.seriesVersion, series))).limit(1);
  if (!stored.length || stored[0].payload !== serialized) {
    return Response.json({ error: "concurrent RORO conflict" }, { status: 409 });
  }
  const inserted = Number(result.meta?.changes ?? 0) === 1;
  return Response.json({
    ok: true,
    series,
    rows: 1,
    inserted: inserted ? 1 : 0,
    idempotent: inserted ? 0 : 1,
    as_of: snapshot.as_of,
  });
}
