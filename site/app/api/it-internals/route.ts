import { env } from "cloudflare:workers";
import { isItInternalsHistory } from "../../../lib/it-internals";

export const dynamic = "force-dynamic";

function allowed(request: Request, upload = false) {
  const header = request.headers.get("authorization") ?? "";
  const token = header.startsWith("Bearer ") ? header.slice(7) : "";
  const uploadToken = env.UPLOAD_TOKEN as string | undefined;
  const viewToken = env.VIEW_TOKEN as string | undefined;
  return Boolean(token && uploadToken && (upload ? token === uploadToken : token === uploadToken || token === viewToken));
}

export async function GET(request: Request) {
  if (!allowed(request)) return Response.json({ error: "unauthorized" }, { status: 401 });
  const row = await env.DB.prepare("SELECT payload FROM it_internals_history WHERE id=1").first<{ payload: string }>();
  if (!row) return Response.json({ error: "no IT internals history" }, { status: 404 });
  return new Response(row.payload, { headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" } });
}

export async function POST(request: Request) {
  if (!allowed(request, true)) return Response.json({ error: "unauthorized" }, { status: 401 });
  const payload: unknown = await request.json();
  if (!isItInternalsHistory(payload)) return Response.json({ error: "invalid IT internals history" }, { status: 400 });
  const serialized = JSON.stringify(payload);
  if (serialized.length > 1_250_000) return Response.json({ error: "IT internals history too large" }, { status: 413 });
  const now = new Date().toISOString();
  await env.DB.prepare(`
    INSERT INTO it_internals_history (id, payload, generated_at, updated_at)
    VALUES (1, ?, ?, ?)
    ON CONFLICT(id) DO UPDATE SET payload=excluded.payload, generated_at=excluded.generated_at, updated_at=excluded.updated_at
  `).bind(serialized, payload.generated_at, now).run();
  return Response.json({ ok: true, observations: payload.rows.length, as_of: payload.as_of });
}
