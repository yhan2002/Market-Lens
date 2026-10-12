import { env } from "cloudflare:workers";
import { isSectorHistory } from "../../../lib/sector-history";

export const dynamic = "force-dynamic";

async function ensureTable() {
  await env.DB.prepare(`
    CREATE TABLE IF NOT EXISTS sector_research_history (
      id INTEGER PRIMARY KEY,
      payload TEXT NOT NULL,
      generated_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    )
  `).run();
}

function allowed(request: Request, upload = false) {
  const header = request.headers.get("authorization") ?? "";
  const token = header.startsWith("Bearer ") ? header.slice(7) : "";
  const uploadToken = env.UPLOAD_TOKEN as string | undefined;
  const viewToken = env.VIEW_TOKEN as string | undefined;
  return Boolean(token && uploadToken && (upload ? token === uploadToken : token === uploadToken || token === viewToken));
}

export async function GET(request: Request) {
  if (!allowed(request)) return Response.json({ error: "unauthorized" }, { status: 401 });
  await ensureTable();
  const row = await env.DB.prepare("SELECT payload FROM sector_research_history WHERE id=1").first<{ payload: string }>();
  if (!row) return Response.json({ error: "no Sector research history" }, { status: 404 });
  return new Response(row.payload, { headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" } });
}

export async function POST(request: Request) {
  if (!allowed(request, true)) return Response.json({ error: "unauthorized" }, { status: 401 });
  const payload: unknown = await request.json();
  if (!isSectorHistory(payload)) return Response.json({ error: "invalid Sector research history" }, { status: 400 });
  const serialized = JSON.stringify(payload);
  if (serialized.length > 1_500_000) return Response.json({ error: "Sector research history too large" }, { status: 413 });
  await ensureTable();
  const now = new Date().toISOString();
  await env.DB.prepare(`
    INSERT INTO sector_research_history (id, payload, generated_at, updated_at)
    VALUES (1, ?, ?, ?)
    ON CONFLICT(id) DO UPDATE SET payload=excluded.payload, generated_at=excluded.generated_at, updated_at=excluded.updated_at
  `).bind(serialized, payload.generated_at, now).run();
  return Response.json({ ok: true, weeks: new Set(payload.rows.map((row) => row.decision_date)).size, rows: payload.rows.length, as_of: payload.as_of });
}
