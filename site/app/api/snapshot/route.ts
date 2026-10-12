import { env } from "cloudflare:workers";
import { and, desc, eq, gt, lt, lte } from "drizzle-orm";
import { getDb } from "../../../db";
import { dashboardPolicySnapshots } from "../../../db/schema";
import {
  isDashboardSnapshot,
  type DashboardSnapshotV3,
  type PolicyState,
} from "../../../lib/snapshot";

export const dynamic = "force-dynamic";

async function ensureTable() {
  await env.DB.prepare(`
    CREATE TABLE IF NOT EXISTS dashboard_policy_snapshots (
      id TEXT PRIMARY KEY,
      series_version TEXT NOT NULL,
      as_of_key INTEGER NOT NULL,
      payload TEXT NOT NULL,
      generated_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    )
  `).run();
  await env.DB.prepare(`
    CREATE UNIQUE INDEX IF NOT EXISTS uq_dashboard_policy_series_date_v1
    ON dashboard_policy_snapshots(series_version, as_of_key)
  `).run();
}

function bearer(request: Request): string {
  const header = request.headers.get("authorization") ?? "";
  return header.startsWith("Bearer ") ? header.slice(7) : "";
}

function allowed(request: Request, mode: "view" | "upload"): boolean {
  const token = bearer(request);
  const uploadToken = env.UPLOAD_TOKEN as string | undefined;
  const viewToken = env.VIEW_TOKEN as string | undefined;
  if (!token || !uploadToken) return false;
  return mode === "upload" ? token === uploadToken : token === uploadToken || token === viewToken;
}

function dateKey(value: string): number | null {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) return null;
  const parsed = new Date(`${value}T00:00:00Z`);
  if (Number.isNaN(parsed.valueOf()) || parsed.toISOString().slice(0, 10) !== value) return null;
  return Number(value.replaceAll("-", ""));
}

function validSeries(value: unknown): value is string {
  return typeof value === "string" && value.length > 0 && value.length <= 500 &&
    [...value].every((character) => character.charCodeAt(0) >= 32);
}

function selectedSeries(request: Request): string | null {
  const requested = new URL(request.url).searchParams.get("series");
  if (requested !== null) return validSeries(requested) ? requested : null;
  const active = env.ACTIVE_POLICY_SERIES as string | undefined;
  return validSeries(active) ? active : null;
}

function legacySeries(request: Request, active: string): string | null {
  if (new URL(request.url).searchParams.has("series")) return null;
  const legacy = env.LEGACY_POLICY_SERIES as string | undefined;
  return validSeries(legacy) && legacy !== active ? legacy : null;
}

function rowId(series: string, asOfKey: number): string {
  return `${series}:${asOfKey}`;
}

function parseV3(payload: string): DashboardSnapshotV3 | null {
  try {
    const parsed: unknown = JSON.parse(payload);
    return isDashboardSnapshot(parsed) && parsed.schema_version === 3 ? parsed : null;
  } catch {
    return null;
  }
}

function exposurePercent(value: number | null): number | null {
  return value === null ? null : Math.round(value * 100);
}

function samePolicyState(left: PolicyState | null, right: PolicyState | null): boolean {
  if (left === null || right === null) return left === right;
  return (
    left.state_version === right.state_version &&
    left.policy_version === right.policy_version &&
    left.regime_model_version === right.regime_model_version &&
    left.last_as_of === right.last_as_of &&
    left.exposure === right.exposure &&
    left.raw_target === right.raw_target &&
    left.recovery_clear_streak === right.recovery_clear_streak &&
    left.episode_peak_p_bear === right.episode_peak_p_bear &&
    left.episode_started_as_of === right.episode_started_as_of
  );
}

export async function GET(request: Request) {
  if (!allowed(request, "view")) {
    return Response.json({ error: "unauthorized" }, { status: 401 });
  }
  const series = selectedSeries(request);
  if (series === null) {
    return Response.json({ error: "active policy series is not configured" }, { status: 503 });
  }
  const requestedDate = new URL(request.url).searchParams.get("date");
  const requestedKey = requestedDate === null ? null : dateKey(requestedDate);
  if (requestedDate !== null && requestedKey === null) {
    return Response.json({ error: "invalid date" }, { status: 400 });
  }
  await ensureTable();
  let rows = await getDb()
    .select()
    .from(dashboardPolicySnapshots)
    .where(
      requestedKey === null
        ? eq(dashboardPolicySnapshots.seriesVersion, series)
        : and(
            eq(dashboardPolicySnapshots.seriesVersion, series),
            lte(dashboardPolicySnapshots.asOfKey, requestedKey),
          ),
    )
    .orderBy(desc(dashboardPolicySnapshots.asOfKey))
    .limit(1);
  const fallback = requestedKey === null ? null : legacySeries(request, series);
  if (!rows.length && fallback !== null && requestedKey !== null) {
    rows = await getDb()
      .select()
      .from(dashboardPolicySnapshots)
      .where(and(
        eq(dashboardPolicySnapshots.seriesVersion, fallback),
        lte(dashboardPolicySnapshots.asOfKey, requestedKey),
      ))
      .orderBy(desc(dashboardPolicySnapshots.asOfKey))
      .limit(1);
  }
  if (!rows.length) return Response.json({ error: "no snapshot in active series" }, { status: 404 });
  return new Response(rows[0].payload, {
    headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" },
  });
}

export async function POST(request: Request) {
  if (!allowed(request, "upload")) {
    return Response.json({ error: "unauthorized" }, { status: 401 });
  }
  const payload: unknown = await request.json();
  if (!isDashboardSnapshot(payload) || payload.schema_version !== 3) {
    return Response.json({ error: "schema v3 snapshot required" }, { status: 400 });
  }
  const series = payload.history.series_version;
  const snapshotKey = dateKey(payload.as_of);
  if (!validSeries(series) || snapshotKey === null) {
    return Response.json({ error: "invalid snapshot series or date" }, { status: 400 });
  }
  const serialized = JSON.stringify(payload);
  if (serialized.length > 250_000) {
    return Response.json({ error: "snapshot too large" }, { status: 413 });
  }
  await ensureTable();

  const existing = await getDb()
    .select()
    .from(dashboardPolicySnapshots)
    .where(eq(dashboardPolicySnapshots.id, rowId(series, snapshotKey)))
    .limit(1);
  if (existing.length) {
    if (existing[0].payload !== serialized) {
      return Response.json({ error: "immutable history: snapshot already exists" }, { status: 409 });
    }
    return Response.json({ ok: true, as_of: payload.as_of, series, idempotent: true });
  }

  const later = await getDb()
    .select()
    .from(dashboardPolicySnapshots)
    .where(and(
      eq(dashboardPolicySnapshots.seriesVersion, series),
      gt(dashboardPolicySnapshots.asOfKey, snapshotKey),
    ))
    .orderBy(desc(dashboardPolicySnapshots.asOfKey))
    .limit(1);
  if (later.length) {
    return Response.json({ error: "immutable history: later snapshot exists" }, { status: 409 });
  }

  const priorRows = await getDb()
    .select()
    .from(dashboardPolicySnapshots)
    .where(and(
      eq(dashboardPolicySnapshots.seriesVersion, series),
      lt(dashboardPolicySnapshots.asOfKey, snapshotKey),
    ))
    .orderBy(desc(dashboardPolicySnapshots.asOfKey))
    .limit(1);
  if (!priorRows.length) {
    if (payload.history.seed_mode !== "FULL_REPLAY") {
      return Response.json({ error: "first series snapshot must be full replay" }, { status: 409 });
    }
  } else {
    if (payload.history.seed_mode !== "STRICT_PRIOR") {
      return Response.json({ error: "only the first series snapshot may use full replay" }, { status: 409 });
    }
    const prior = parseV3(priorRows[0].payload);
    if (prior === null || payload.history.parent_as_of !== prior.as_of) {
      return Response.json({ error: "policy history parent mismatch" }, { status: 409 });
    }
    const priorState = prior.risk_policy.policy_state;
    if (payload.risk_policy.decision_status === "NO_DECISION") {
      if (!samePolicyState(payload.risk_policy.policy_state, priorState)) {
        return Response.json({ error: "no-decision policy state mismatch" }, { status: 409 });
      }
    } else {
      const priorExposure = exposurePercent(priorState?.exposure ?? null);
      const currentPrevious = exposurePercent(payload.risk_policy.previous_recommended_exposure);
      if (priorExposure !== currentPrevious) {
        return Response.json({ error: "policy history exposure mismatch" }, { status: 409 });
      }
    }
  }

  const now = new Date().toISOString();
  await getDb()
    .insert(dashboardPolicySnapshots)
    .values({
      id: rowId(series, snapshotKey),
      seriesVersion: series,
      asOfKey: snapshotKey,
      payload: serialized,
      generatedAt: payload.generated_at,
      updatedAt: now,
    })
    .onConflictDoNothing({ target: dashboardPolicySnapshots.id });
  const stored = await getDb()
    .select()
    .from(dashboardPolicySnapshots)
    .where(eq(dashboardPolicySnapshots.id, rowId(series, snapshotKey)))
    .limit(1);
  if (!stored.length || stored[0].payload !== serialized) {
    return Response.json({ error: "concurrent snapshot conflict" }, { status: 409 });
  }
  return Response.json({ ok: true, as_of: payload.as_of, series, updated_at: now });
}
