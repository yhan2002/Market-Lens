import { env } from 'cloudflare:workers';
import { isHybridSnapshot } from '../../../lib/sector-hybrid';
export const dynamic = 'force-dynamic';
function allowed(request:Request, upload=false) {
  const token=request.headers.get('authorization')?.replace(/^Bearer /,'');
  return Boolean(token && env.UPLOAD_TOKEN && (token===env.UPLOAD_TOKEN || (!upload && token===env.VIEW_TOKEN)));
}
export async function GET(request:Request) {
  if (!allowed(request)) return Response.json({error:'unauthorized'},{status:401});
  const date=new URL(request.url).searchParams.get('date')??'9999-12-31';
  if (!/^\d{4}-\d{2}-\d{2}$/.test(date) || !Number.isFinite(Date.parse(date)) || new Date(date).toISOString().slice(0,10)!==date) return Response.json({error:'invalid date'},{status:400});
  try {
    const row=await env.DB.prepare('SELECT payload FROM sector_hybrid_snapshots WHERE as_of <= ? ORDER BY as_of DESC LIMIT 1').bind(date).first<{payload:string}>();
    if (!row) return Response.json({error:'no hybrid observation'},{status:404});
    return new Response(row.payload,{headers:{'content-type':'application/json','cache-control':'no-store'}});
  } catch { return Response.json({error:'hybrid unavailable'},{status:503}); }
}
export async function POST(request:Request) {
  if (!allowed(request,true)) return Response.json({error:'unauthorized'},{status:401});
  const raw=await request.text();
  if(raw.length>20000) return Response.json({error:'too large'},{status:413});
  let payload:unknown;
  try {payload=JSON.parse(raw);} catch {return Response.json({error:'invalid JSON'},{status:400});}
  if (!isHybridSnapshot(payload)) return Response.json({error:'invalid hybrid snapshot'},{status:400});
  await env.DB.prepare('INSERT INTO sector_hybrid_snapshots (as_of,payload) VALUES (?,?) ON CONFLICT(as_of) DO UPDATE SET payload=excluded.payload').bind(payload.as_of,JSON.stringify(payload)).run();
  return Response.json({ok:true,as_of:payload.as_of});
}
