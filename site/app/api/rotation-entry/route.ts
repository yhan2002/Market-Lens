import { env } from 'cloudflare:workers';
import { isRotationEntry } from '../../../lib/rotation-entry';

export const dynamic = 'force-dynamic';
function allowed(request: Request, upload = false) {
  const token = request.headers.get('authorization')?.replace(/^Bearer /, '');
  return Boolean(token && env.UPLOAD_TOKEN && (token === env.UPLOAD_TOKEN || (!upload && token === env.VIEW_TOKEN)));
}
export async function GET(request: Request) {
  if (!allowed(request)) return Response.json({error:'unauthorized'}, {status:401});
  const date = new URL(request.url).searchParams.get('date') ?? '9999-12-31';
  if (!/^\d{4}-\d{2}-\d{2}$/.test(date) || !Number.isFinite(Date.parse(date)) || new Date(date).toISOString().slice(0,10) !== date)
    return Response.json({error:'invalid date'}, {status:400});
  const row = await env.DB.prepare('SELECT payload FROM rotation_entry_snapshots WHERE as_of <= ? ORDER BY as_of DESC LIMIT 1').bind(date).first<{payload:string}>();
  if (!row) return Response.json({error:'no rotation entry observation'}, {status:404});
  return new Response(row.payload, {headers:{'content-type':'application/json', 'cache-control':'no-store'}});
}
export async function POST(request: Request) {
  if (!allowed(request, true)) return Response.json({error:'unauthorized'}, {status:401});
  const raw = await request.text();
  if (raw.length > 10000) return Response.json({error:'too large'}, {status:413});
  let payload: unknown;
  try { payload = JSON.parse(raw); } catch { return Response.json({error:'invalid JSON'}, {status:400}); }
  if (!isRotationEntry(payload)) return Response.json({error:'invalid rotation entry'}, {status:400});
  await env.DB.prepare('INSERT INTO rotation_entry_snapshots (as_of, payload) VALUES (?, ?) ON CONFLICT(as_of) DO NOTHING').bind(payload.as_of, JSON.stringify(payload)).run();
  const row = await env.DB.prepare('SELECT payload FROM rotation_entry_snapshots WHERE as_of = ?').bind(payload.as_of).first<{payload:string}>();
  const stored = JSON.parse(row!.payload);
  if (Object.keys(payload).some(k => JSON.stringify(stored[k]) !== JSON.stringify(payload[k as keyof typeof payload])))
    return Response.json({error:'immutable observation conflict'}, {status:409});
  return Response.json({ok:true, as_of:payload.as_of});
}
