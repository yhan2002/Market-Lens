import {validSync,saveSync,articlePage} from './news-data.mjs';
const enc = new TextEncoder();
const hex = b => [...new Uint8Array(b)].map(v=>v.toString(16).padStart(2,'0')).join('');
const bytes = h => Uint8Array.from(h.match(/../g)||[],v=>parseInt(v,16));
const headers = {'cache-control':'no-store','x-content-type-options':'nosniff','referrer-policy':'no-referrer'};
const json = (v,status=200) => Response.json(v,{status,headers});
async function digest(v) { return hex(await crypto.subtle.digest('SHA-256',v)); }
async function key(secret) { return crypto.subtle.importKey('raw',enc.encode(secret),{name:'HMAC',hash:'SHA-256'},false,['sign','verify']); }
export async function signedSession(secret,expiry) {
  return expiry+'.'+hex(await crypto.subtle.sign('HMAC',await key(secret),enc.encode(String(expiry))));
}
export async function sessionValid(cookie,secret,now=Date.now()) {
  const value=/(?:^|;\s*)news_session=(\d{13}\.[a-f0-9]{64})(?:;|$)/.exec(cookie||'')?.[1];
  if(!value)return false;
  const [expiry,signature]=value.split('.');
  if(Number(expiry)<=now || Number(expiry)>now+8*3600*1000)return false;
  return crypto.subtle.verify('HMAC',await key(secret),bytes(signature),enc.encode(expiry));
}
async function limitedBody(request,limit) {
  if(Number(request.headers.get('content-length'))>limit)throw new Error('size');
  const reader=request.body?.getReader(); if(!reader)return '';
  let size=0;const chunks=[];
  while(true){const {value,done}=await reader.read();if(done)break;size+=value.length;if(size>limit){await reader.cancel();throw new Error('size');}chunks.push(value);}
  const all=new Uint8Array(size);let offset=0;for(const c of chunks){all.set(c,offset);offset+=c.length;}
  return new TextDecoder().decode(all);
}
export function validUpload(value) {
  return value && typeof value.generated_at==='string' && /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}\+00:00$/.test(value.generated_at) && Number.isFinite(Date.parse(value.generated_at)) &&
    Array.isArray(value.snapshots) && value.snapshots.length===5 &&
    new Set(value.snapshots.map(s=>s.hours)).size===5 && value.snapshots.every(s=>
      [0,24,72,168,720].includes(s.hours) && typeof s.body==='string' && s.body.length>0 &&
      s.body.length<=4000000 && /^[A-Za-z0-9+/]+={0,2}$/.test(s.body));
}
function loginPage(auth) {
  return `<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Market Lens News · 로그인</title><style>body{background:#0b141a;color:#dce7ed;font:16px system-ui;display:grid;place-items:center;min-height:95vh}main{width:min(360px,85vw)}input,button{box-sizing:border-box;width:100%;padding:14px;margin:8px 0;font:inherit;border-radius:6px;border:1px solid #34444f}button{background:#e4b656;color:#17140c;cursor:pointer}small{color:#9fb0bc}</style><main><small>Market Lens RESEARCH</small><h1>News Intelligence</h1><form><label for="password">공유 비밀번호</label><input id="password" type="password" autocomplete="current-password" required><button>로그인</button></form><p id="status" role="status"></p></main><script>
const form=document.querySelector('form');form.onsubmit=async e=>{e.preventDefault();const button=form.querySelector('button');button.disabled=true;document.querySelector('#status').textContent='확인 중…';try{const password=document.querySelector('#password').value;const key=await crypto.subtle.importKey('raw',new TextEncoder().encode(password),'PBKDF2',false,['deriveBits']);const salt=Uint8Array.from('${auth.salt}'.match(/../g),x=>parseInt(x,16));const bits=await crypto.subtle.deriveBits({name:'PBKDF2',hash:'SHA-256',salt,iterations:${auth.iterations}},key,256);const proof=Array.from(new Uint8Array(bits),x=>x.toString(16).padStart(2,'0')).join('');const r=await fetch('/news/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({proof})});if(r.ok)location.replace('/news/');else document.querySelector('#status').textContent=r.status===429?'잠시 후 다시 시도해주세요.':'비밀번호를 확인해주세요.';}catch{document.querySelector('#status').textContent='연결에 실패했습니다. 다시 시도해주세요.';}finally{button.disabled=false;}};
</script></html>`;
}

export async function newsRoute(request,env) {
  const url=new URL(request.url); const path=url.pathname;
  if(path!='/news' && !path.startsWith('/news/'))return null;
  try {
    if(path==='/news/api/sync') {
      if(!env.NEWS_UPLOAD_TOKEN || request.headers.get('authorization')!=='Bearer '+env.NEWS_UPLOAD_TOKEN)return json({error:'unauthorized'},401);
      if(request.method==='GET')return json({protocol:2,removals:true});
      if(request.method!=='POST')return json({error:'method'},405);
      const value=JSON.parse(await limitedBody(request,2000000));
      if(!validSync(value))return json({error:'invalid article batch'},400);
      await saveSync(env.DB,value);
      return json({ok:true});
    }
    if(path==='/news/api/upload') {
      if(request.method!=='POST')return json({error:'method'},405);
      if(!env.NEWS_UPLOAD_TOKEN || request.headers.get('authorization')!=='Bearer '+env.NEWS_UPLOAD_TOKEN)return json({error:'unauthorized'},401);
      const value=JSON.parse(await limitedBody(request,20100000));
      if(!validUpload(value))return json({error:'invalid snapshot'},400);
      // One batch replaces all periods atomically. Other applications' tables are untouched.
      const statements=[env.DB.prepare('DELETE FROM news_snapshots WHERE generated_at <= ?').bind(value.generated_at)];
      for(const s of value.snapshots)for(let offset=0;offset<s.body.length;offset+=500000) {
        statements.push(env.DB.prepare('INSERT INTO news_snapshots (hours,part,body,generated_at) SELECT ?,?,?,? WHERE NOT EXISTS (SELECT 1 FROM news_snapshots WHERE generated_at > ?) ON CONFLICT(hours,part) DO NOTHING').bind(s.hours,offset/500000,s.body.slice(offset,offset+500000),value.generated_at,value.generated_at));
      }
      await env.DB.batch(statements);
      return json({ok:true});
    }
    if(!env.NEWS_AUTH || !env.NEWS_SESSION_KEY)return json({error:'뉴스 접속 설정 준비 중'},503);
    const auth=JSON.parse(env.NEWS_AUTH);
    if(!/^[a-f0-9]{32}$/.test(auth.salt) || auth.iterations!==600000 || !/^[a-f0-9]{64}$/.test(auth.verifier))throw new Error('auth');
    if(path==='/news/login') {
      if(request.method==='GET')return new Response(loginPage(auth),{headers:{...headers,'content-type':'text/html; charset=utf-8','content-security-policy':"default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; frame-ancestors 'none'; base-uri 'none'"}});
      if(request.method!=='POST')return json({error:'method'},405);
      if(request.headers.get('origin')!==url.origin)return json({error:'origin'},403);
      const bucket=Math.floor(Date.now()/60000);
      // Small-team global bucket: bounds password attempts without unbounded per-IP storage.
      const row=await env.DB.prepare('INSERT INTO news_login_rate (id,bucket,count) VALUES (1,?,1) ON CONFLICT(id) DO UPDATE SET count=CASE WHEN bucket=excluded.bucket THEN count+1 ELSE 1 END,bucket=excluded.bucket RETURNING count').bind(bucket).first();
      if(row.count>10)return json({error:'rate limited'},429);
      const input=JSON.parse(await limitedBody(request,1024));
      if(!/^[a-f0-9]{64}$/.test(input.proof||'') || await digest(bytes(input.proof))!==auth.verifier)return json({error:'unauthorized'},401);
      const session=await signedSession(env.NEWS_SESSION_KEY+auth.verifier,Date.now()+8*3600*1000);
      return new Response('{}',{headers:{...headers,'content-type':'application/json','set-cookie':`news_session=${session}; Path=/news; Secure; HttpOnly; SameSite=Strict; Max-Age=28800`}});
    }
    if(path==='/news/logout' && request.method==='POST') {
      if(request.headers.get('origin')!==url.origin)return json({error:'origin'},403);
      return new Response(null,{status:303,headers:{...headers,location:'/news/login','set-cookie':'news_session=; Path=/news; Secure; HttpOnly; SameSite=Strict; Max-Age=0'}});
    }
    if(!await sessionValid(request.headers.get('cookie'),env.NEWS_SESSION_KEY+auth.verifier)) {
      return path.startsWith('/news/api/')?json({error:'unauthorized'},401):new Response(null,{status:303,headers:{...headers,location:'/news/login'}});
    }
    if(request.method!=='GET')return json({error:'method'},405);
    if(path==='/news/api/articles') {
      const page=await articlePage(env.DB,url);
      return json(page,page.status||200);
    }
    if(path==='/news/api/dashboard') {
      const hours=Number(url.searchParams.get('hours')??24);
      if(![0,24,72,168,720].includes(hours))return json({error:'invalid period'},400);
      const catalog=await env.DB.prepare('SELECT updated_at,body FROM news_catalog WHERE hours=?').bind(hours).first();
      if(catalog)return json({...JSON.parse(catalog.body),protocol:2,revision:catalog.updated_at});
      const {results}=await env.DB.prepare('SELECT body,generated_at FROM news_snapshots WHERE hours=? ORDER BY part').bind(hours).all();
      if(!results.length)return json({error:'아직 업로드된 뉴스가 없습니다.'},503);
      const zipped=Uint8Array.from(atob(results.map(r=>r.body).join('')),c=>c.charCodeAt(0));
      return new Response(new Blob([zipped]).stream().pipeThrough(new DecompressionStream('gzip')),{headers:{...headers,'content-type':'application/json; charset=utf-8','x-news-generated-at':results[0].generated_at}});
    }
    if(path==='/news' || path==='/news/') {
      const asset=await env.ASSETS.fetch(new Request(new URL('/news-static/index.html',url)));
      return new Response(asset.body,{status:asset.status,headers:{...headers,'content-type':'text/html; charset=utf-8'}});
    }
    return json({error:'not found'},404);
  } catch { return json({error:'뉴스 서비스 처리 실패. 관리자에게 문의해주세요.'},503); }
}
