import { env } from "cloudflare:workers";
import { and, desc, eq, lte } from "drizzle-orm";
import { getDb } from "../../../db";
import { sectorRoroShadowSnapshots } from "../../../db/schema";
import {
  isCurrentSectorRoroShadowSnapshot,
  verifySectorRoroProvenance,
} from "../../../lib/sector-roro";

export const dynamic = "force-dynamic";

async function ensureTable() {
  await env.DB.prepare(`
    CREATE TABLE IF NOT EXISTS sector_roro_shadow_snapshots (
      id TEXT PRIMARY KEY,
      series_version TEXT NOT NULL,
      as_of_key INTEGER NOT NULL,
      payload TEXT NOT NULL,
      generated_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    )
  `).run();
  await env.DB.prepare(`
    CREATE UNIQUE INDEX IF NOT EXISTS uq_sector_roro_shadow_series_date_v1
    ON sector_roro_shadow_snapshots(series_version, as_of_key)
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

export async function GET(request: Request) {
  if (!allowed(request, "view")) return Response.json({ error: "unauthorized" }, { status: 401 });
  const url = new URL(request.url);
  const requestedDate = url.searchParams.get("date");
  const requestedKey = requestedDate === null ? null : dateKey(requestedDate);
  if (requestedDate !== null && requestedKey === null) return Response.json({ error: "invalid date" }, { status: 400 });
  const historyParam = url.searchParams.get("history");
  if (historyParam !== null && historyParam !== "2") {
    return Response.json({ error: "history must be 2" }, { status: 400 });
  }
  const requestedSeries = url.searchParams.get("series");
  const configuredSeries = env.ACTIVE_SECTOR_RORO_SERIES as string | undefined;
  if (requestedSeries !== null && bearer(request) !== (env.UPLOAD_TOKEN as string | undefined)) {
    return Response.json({ error: "series override requires upload authorization" }, { status: 403 });
  }
  const series = requestedSeries ?? configuredSeries;
  if (!validSeries(series)) return Response.json({ error: "active Sector RORO series is not configured" }, { status: 503 });
  await ensureTable();
  const rows = await getDb().select().from(sectorRoroShadowSnapshots)
    .where(requestedKey === null
      ? eq(sectorRoroShadowSnapshots.seriesVersion, series)
      : and(eq(sectorRoroShadowSnapshots.seriesVersion, series), lte(sectorRoroShadowSnapshots.asOfKey, requestedKey)))
    .orderBy(desc(sectorRoroShadowSnapshots.asOfKey)).limit(historyParam === null ? 1 : 2);
  if (!rows.length) return Response.json({ error: "no Sector RORO SHADOW snapshot" }, { status: 404 });
  if (historyParam !== null) {
    return new Response(`[${rows.reverse().map((row) => row.payload).join(",")}]`, {
      headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" },
    });
  }
  return new Response(rows[0].payload, { headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" } });
}

export async function POST(request: Request) {
  if (!allowed(request, "upload")) return Response.json({ error: "unauthorized" }, { status: 401 });
  const candidate: unknown = await request.json();
  const payload = Array.isArray(candidate) && candidate.length === 1 ? candidate[0] : candidate;
  if (!isCurrentSectorRoroShadowSnapshot(payload)) {
    return Response.json({ error: "invalid Sector RORO SHADOW payload" }, { status: 400 });
  }
  const snapshot = payload;
  if (!await verifySectorRoroProvenance(snapshot)) {
    return Response.json({ error: "invalid Sector RORO SHADOW provenance" }, { status: 400 });
  }
  const series = snapshot.series_version;
  if (!validSeries(series)) {
    return Response.json({ error: "one valid series required" }, { status: 400 });
  }
  await ensureTable();
  const key = dateKey(snapshot.as_of);
  if (key === null) return Response.json({ error: "invalid causal date" }, { status: 400 });
  const serialized = JSON.stringify(snapshot);
  if (serialized.length > 250_000) return Response.json({ error: "Sector RORO snapshot too large" }, { status: 413 });
  const id = rowId(series, key);
  const existing = await getDb().select().from(sectorRoroShadowSnapshots)
    .where(eq(sectorRoroShadowSnapshots.id, id)).limit(1);
  if (existing.length) {
    if (existing[0].payload !== serialized) {
      return Response.json({ error: "immutable Sector RORO history conflict", as_of: snapshot.as_of }, { status: 409 });
    }
    return Response.json({ ok: true, series, dates: 1, rows: 11, inserted: 0, idempotent: 11, as_of: snapshot.as_of });
  }
  const now = new Date().toISOString();
  const result = await env.DB.prepare(`
    INSERT OR IGNORE INTO sector_roro_shadow_snapshots
      (id, series_version, as_of_key, payload, generated_at, updated_at)
    VALUES (?, ?, ?, ?, ?, ?)
  `).bind(id, series, key, serialized, snapshot.generated_at, now).run();
  const stored = await getDb().select().from(sectorRoroShadowSnapshots)
    .where(and(eq(sectorRoroShadowSnapshots.id, id), eq(sectorRoroShadowSnapshots.seriesVersion, series))).limit(1);
  if (!stored.length || stored[0].payload !== serialized) {
    return Response.json({ error: "concurrent Sector RORO conflict" }, { status: 409 });
  }
  const inserted = Number(result.meta?.changes ?? 0) === 1;
  return Response.json({
    ok: true,
    series,
    dates: 1,
    rows: 11,
    inserted: inserted ? 11 : 0,
    idempotent: inserted ? 0 : 11,
    as_of: snapshot.as_of,
  });
}
