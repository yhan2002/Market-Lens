import { env } from "cloudflare:workers";
import { and, desc, eq, lte } from "drizzle-orm";
import { getDb } from "../../../db";
import { sectorTurningGaugeSnapshots } from "../../../db/schema";
import {
  CURRENT_SECTOR_TURNING_SERIES,
  isSectorTurningSnapshot,
  verifySectorTurningProvenance,
} from "../../../lib/sector-turning";

export const dynamic = "force-dynamic";

async function ensureTable() {
  await env.DB.prepare(`
    CREATE TABLE IF NOT EXISTS sector_turning_gauge_snapshots (
      id TEXT PRIMARY KEY,
      series_version TEXT NOT NULL,
      as_of_key INTEGER NOT NULL,
      payload TEXT NOT NULL,
      generated_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    )
  `).run();
  await env.DB.prepare(`
    CREATE UNIQUE INDEX IF NOT EXISTS uq_sector_turning_gauge_series_date_v1
    ON sector_turning_gauge_snapshots(series_version, as_of_key)
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

export async function GET(request: Request) {
  if (!allowed(request, "view")) return Response.json({ error: "unauthorized" }, { status: 401 });
  const url = new URL(request.url);
  const requestedDate = url.searchParams.get("date");
  const requestedKey = requestedDate === null ? null : dateKey(requestedDate);
  if (requestedDate !== null && requestedKey === null) {
    return Response.json({ error: "invalid date" }, { status: 400 });
  }
  const requestedSeries = url.searchParams.get("series");
  if (requestedSeries !== null && bearer(request) !== (env.UPLOAD_TOKEN as string | undefined)) {
    return Response.json({ error: "series override requires upload authorization" }, { status: 403 });
  }
  const configured = env.ACTIVE_SECTOR_TURNING_SERIES as string | undefined;
  const series = requestedSeries ?? configured ?? CURRENT_SECTOR_TURNING_SERIES;
  if (!validSeries(series)) return Response.json({ error: "invalid Sector Turning series" }, { status: 503 });
  await ensureTable();
  const rows = await getDb().select().from(sectorTurningGaugeSnapshots)
    .where(requestedKey === null
      ? eq(sectorTurningGaugeSnapshots.seriesVersion, series)
      : and(eq(sectorTurningGaugeSnapshots.seriesVersion, series), lte(sectorTurningGaugeSnapshots.asOfKey, requestedKey)))
    .orderBy(desc(sectorTurningGaugeSnapshots.asOfKey)).limit(1);
  if (!rows.length) return Response.json({ error: "no Sector Turning gauge snapshot" }, { status: 404 });
  return new Response(rows[0].payload, {
    headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" },
  });
}

export async function POST(request: Request) {
  if (!allowed(request, "upload")) return Response.json({ error: "unauthorized" }, { status: 401 });
  const candidate: unknown = await request.json();
  const payload = Array.isArray(candidate) && candidate.length === 1 ? candidate[0] : candidate;
  if (!isSectorTurningSnapshot(payload)) {
    return Response.json({ error: "invalid Sector Turning gauge payload" }, { status: 400 });
  }
  const snapshot = payload;
  if (!await verifySectorTurningProvenance(snapshot)) {
    return Response.json({ error: "invalid Sector Turning gauge provenance" }, { status: 400 });
  }
  await ensureTable();
  const key = dateKey(snapshot.as_of);
  if (key === null) return Response.json({ error: "invalid causal date" }, { status: 400 });
  const serialized = JSON.stringify(snapshot);
  if (serialized.length > 250_000) return Response.json({ error: "Sector Turning snapshot too large" }, { status: 413 });
  const id = `${snapshot.series_version}:${key}`;
  const existing = await getDb().select().from(sectorTurningGaugeSnapshots)
    .where(eq(sectorTurningGaugeSnapshots.id, id)).limit(1);
  if (existing.length) {
    if (existing[0].payload !== serialized) {
      return Response.json({ error: "immutable Sector Turning history conflict", as_of: snapshot.as_of }, { status: 409 });
    }
    return Response.json({ ok: true, series: snapshot.series_version, dates: 1, rows: snapshot.payload.rows.length, inserted: 0 });
  }
  const now = new Date().toISOString();
  await env.DB.prepare(`
    INSERT INTO sector_turning_gauge_snapshots
      (id, series_version, as_of_key, payload, generated_at, updated_at)
    VALUES (?, ?, ?, ?, ?, ?)
  `).bind(id, snapshot.series_version, key, serialized, snapshot.generated_at, now).run();
  return Response.json({
    ok: true,
    series: snapshot.series_version,
    dates: 1,
    rows: snapshot.payload.rows.length,
    inserted: snapshot.payload.rows.length,
    as_of: snapshot.as_of,
  });
}
