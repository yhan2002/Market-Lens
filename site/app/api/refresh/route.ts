import { env } from "cloudflare:workers";

export const dynamic = "force-dynamic";

type RefreshStatus = "PENDING" | "RUNNING" | "SUCCEEDED" | "FAILED";

type RefreshRequest = {
  id: number;
  status: RefreshStatus;
  requested_at: string;
  started_at: string | null;
  finished_at: string | null;
  as_of: string | null;
  detail: string | null;
};

async function ensureTable() {
  const db = env.DB;
  await db.batch([
    db.prepare(`
      CREATE TABLE IF NOT EXISTS dashboard_refresh_requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        status TEXT NOT NULL,
        requested_at TEXT NOT NULL,
        started_at TEXT,
        finished_at TEXT,
        as_of TEXT,
        detail TEXT
      )
    `),
    db.prepare(`
      CREATE INDEX IF NOT EXISTS idx_dashboard_refresh_requests_status_id
      ON dashboard_refresh_requests(status, id)
    `),
  ]);
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

function validDate(value: unknown): value is string {
  if (typeof value !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(value)) return false;
  const parsed = new Date(`${value}T00:00:00Z`);
  return !Number.isNaN(parsed.valueOf()) && parsed.toISOString().slice(0, 10) === value;
}

async function latestRequest(id?: number): Promise<RefreshRequest | null> {
  if (id !== undefined) {
    return env.DB.prepare(`
      SELECT id, status, requested_at, started_at, finished_at, as_of, detail
      FROM dashboard_refresh_requests
      WHERE id = ?
    `).bind(id).first<RefreshRequest>();
  }
  return env.DB.prepare(`
    SELECT id, status, requested_at, started_at, finished_at, as_of, detail
    FROM dashboard_refresh_requests
    ORDER BY id DESC
    LIMIT 1
  `).first<RefreshRequest>();
}

async function requestRefresh(): Promise<RefreshRequest | null> {
  const now = new Date().toISOString();
  const created = await env.DB.prepare(`
    INSERT INTO dashboard_refresh_requests(status, requested_at)
    SELECT 'PENDING', ?
    WHERE NOT EXISTS (
      SELECT 1 FROM dashboard_refresh_requests WHERE status IN ('PENDING', 'RUNNING')
    )
    RETURNING id, status, requested_at, started_at, finished_at, as_of, detail
  `).bind(now).first<RefreshRequest>();
  return created ?? latestRequest();
}

async function claimRefresh(): Promise<RefreshRequest | null> {
  const now = new Date();
  const cutoff = new Date(now.valueOf() - 2 * 60 * 60 * 1000).toISOString();
  await env.DB.prepare(`
    UPDATE dashboard_refresh_requests
    SET status = 'FAILED', finished_at = ?, detail = '업데이트 시간이 초과되었습니다.'
    WHERE status = 'RUNNING' AND started_at IS NOT NULL AND started_at < ?
  `).bind(now.toISOString(), cutoff).run();
  return env.DB.prepare(`
    UPDATE dashboard_refresh_requests
    SET status = 'RUNNING', started_at = ?, finished_at = NULL, detail = NULL
    WHERE id = (
      SELECT id FROM dashboard_refresh_requests
      WHERE status = 'PENDING'
      ORDER BY id
      LIMIT 1
    ) AND status = 'PENDING'
    RETURNING id, status, requested_at, started_at, finished_at, as_of, detail
  `).bind(now.toISOString()).first<RefreshRequest>();
}

async function finishRefresh(
  id: number,
  status: "SUCCEEDED" | "FAILED",
  asOf: string | null,
  detail: string | null,
): Promise<RefreshRequest | null> {
  return env.DB.prepare(`
    UPDATE dashboard_refresh_requests
    SET status = ?, finished_at = ?, as_of = ?, detail = ?
    WHERE id = ? AND status = 'RUNNING'
    RETURNING id, status, requested_at, started_at, finished_at, as_of, detail
  `).bind(status, new Date().toISOString(), asOf, detail, id).first<RefreshRequest>();
}

export async function GET(request: Request) {
  if (!allowed(request, "view")) {
    return Response.json({ error: "unauthorized" }, { status: 401 });
  }
  const rawId = new URL(request.url).searchParams.get("id");
  const id = rawId === null ? undefined : Number(rawId);
  if (rawId !== null && (!Number.isSafeInteger(id) || (id ?? 0) < 1)) {
    return Response.json({ error: "invalid request id" }, { status: 400 });
  }
  await ensureTable();
  const row = await latestRequest(id);
  return row
    ? Response.json(row, { headers: { "cache-control": "no-store" } })
    : Response.json({ error: "no refresh request" }, { status: 404 });
}

export async function POST(request: Request) {
  if (!allowed(request, "view")) {
    return Response.json({ error: "unauthorized" }, { status: 401 });
  }
  let parsed: unknown;
  try {
    parsed = await request.json();
  } catch {
    return Response.json({ error: "invalid json" }, { status: 400 });
  }
  if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
    return Response.json({ error: "invalid json object" }, { status: 400 });
  }
  const payload = parsed as Record<string, unknown>;
  const action = typeof payload.action === "string" ? payload.action : "request";
  await ensureTable();

  if (action === "request") {
    const row = await requestRefresh();
    return row
      ? Response.json(row, { status: row.status === "PENDING" ? 202 : 200 })
      : Response.json({ error: "refresh request unavailable" }, { status: 503 });
  }
  if (!allowed(request, "upload")) {
    return Response.json({ error: "unauthorized" }, { status: 401 });
  }
  if (action === "claim") {
    const row = await claimRefresh();
    return row ? Response.json(row) : new Response(null, { status: 204 });
  }
  if (action === "finish") {
    const id = payload.id;
    const status = payload.status;
    const asOf = payload.as_of == null ? null : payload.as_of;
    const detail = payload.detail == null ? null : payload.detail;
    if (
      typeof id !== "number" || !Number.isSafeInteger(id) || id < 1 ||
      (status !== "SUCCEEDED" && status !== "FAILED") ||
      (asOf !== null && !validDate(asOf)) ||
      (detail !== null && (typeof detail !== "string" || detail.length > 1000))
    ) {
      return Response.json({ error: "invalid finish payload" }, { status: 400 });
    }
    const row = await finishRefresh(id, status, asOf, detail);
    return row
      ? Response.json(row)
      : Response.json({ error: "refresh request is not running" }, { status: 409 });
  }
  return Response.json({ error: "invalid action" }, { status: 400 });
}
