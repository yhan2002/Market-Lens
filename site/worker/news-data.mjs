// Article rows are independent of the selected display period. Keep all dates in UTC.
const periods = [0,24,72,168,720];
const stamp = v => typeof v==='string' && /^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{6}\+00:00$/.test(v) && Number.isFinite(Date.parse(v));
const optionalDate = v => v===null || typeof v==='string' && Number.isFinite(Date.parse(v));
export function validSync(v) {
  return v && stamp(v.generated_at) && Array.isArray(v.articles) && v.articles.length<=50 &&
    new Set(v.articles.map(a=>a.id)).size===v.articles.length && v.articles.every(a=>
      typeof a.id==='string' && /^[a-zA-Z0-9_-]{1,80}$/.test(a.id) &&
      Number.isSafeInteger(a.published_at) && a.published_at>=0 &&
      optionalDate(a.first_collected_at) && optionalDate(a.analysis_updated_at) &&
      typeof a.body==='string' && a.body.length<=250000 && (()=>{try{
        const row=JSON.parse(a.body);return row.id===a.id && Date.parse(row.published_at)===a.published_at && typeof row.title==='string';
      }catch{return false;}})()) &&
    (v.removed_ids===undefined || Array.isArray(v.removed_ids) && v.removed_ids.length<=20000 &&
      new Set(v.removed_ids).size===v.removed_ids.length &&
      v.removed_ids.every(id=>typeof id==='string' && /^[a-zA-Z0-9_-]{1,80}$/.test(id) && !v.articles.some(a=>a.id===id)) &&
      (!v.removed_ids.length || Array.isArray(v.catalogs))) &&
    (v.catalogs===undefined || Array.isArray(v.catalogs) && v.catalogs.length===5 &&
      new Set(v.catalogs.map(c=>c.hours)).size===5 && v.catalogs.every(c=>periods.includes(c.hours) &&
        typeof c.body==='string' && c.body.length<=250000 && (()=>{try{
          const data=JSON.parse(c.body);return data.hours===c.hours && data.as_of===v.generated_at && !data.articles && !data.issues;
        }catch{return false;}})()));
}

export async function saveSync(db,v) {
  const statements=v.articles.map(a=>db.prepare('INSERT INTO news_articles (id,published_at,first_collected_at,analysis_updated_at,updated_at,body) VALUES (?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET published_at=excluded.published_at,first_collected_at=COALESCE(news_articles.first_collected_at,excluded.first_collected_at),analysis_updated_at=excluded.analysis_updated_at,updated_at=excluded.updated_at,body=excluded.body WHERE excluded.updated_at>=news_articles.updated_at')
    .bind(a.id,a.published_at,a.first_collected_at,a.analysis_updated_at,v.generated_at,a.body));
  if(v.removed_ids?.length)statements.push(db.prepare('DELETE FROM news_articles WHERE id IN (SELECT value FROM json_each(?)) AND updated_at<=? AND NOT EXISTS (SELECT 1 FROM news_catalog WHERE updated_at>?)')
    .bind(JSON.stringify(v.removed_ids),v.generated_at,v.generated_at));
  for(const c of v.catalogs||[]) statements.push(db.prepare('INSERT INTO news_catalog (hours,updated_at,body) VALUES (?,?,?) ON CONFLICT(hours) DO UPDATE SET updated_at=excluded.updated_at,body=excluded.body WHERE excluded.updated_at>=news_catalog.updated_at').bind(c.hours,v.generated_at,c.body));
  if(statements.length)await db.batch(statements);
}

export async function articlePage(db,url) {
  const hours=Number(url.searchParams.get('hours')??24);
  if(!periods.includes(hours))return {status:400,error:'invalid period'};
  const catalog=await db.prepare('SELECT updated_at,body FROM news_catalog WHERE hours=?').bind(hours).first();
  if(!catalog)return {status:503,error:'기사별 데이터 동기화 대기 중'};
  if(url.searchParams.get('revision')!==catalog.updated_at)return {status:409,error:'데이터가 갱신되었습니다. 다시 조회해주세요.'};
  const end=Date.parse(catalog.updated_at),start=hours?end-hours*3600000:0;
  let cursor=null;
  if(url.searchParams.has('cursor')) {
    try {cursor=JSON.parse(url.searchParams.get('cursor'));}catch{return {status:400,error:'invalid cursor'};}
    if(!Array.isArray(cursor)||cursor.length!==2||!Number.isSafeInteger(cursor[0])||typeof cursor[1]!=='string'||cursor[1].length>80)return {status:400,error:'invalid cursor'};
  }
  const sql='SELECT id,published_at,body FROM news_articles WHERE published_at>=? AND published_at<=?'+
    (cursor?' AND (published_at<? OR (published_at=? AND id<?))':'')+' ORDER BY published_at DESC,id DESC LIMIT 51';
  const values=cursor?[start,end,cursor[0],cursor[0],cursor[1]]:[start,end];
  const {results}=await db.prepare(sql).bind(...values).all();
  const page=results.slice(0,50),last=page.at(-1);
  return {articles:page.map(r=>JSON.parse(r.body)),next:results.length>50?JSON.stringify([last.published_at,last.id]):null,revision:catalog.updated_at};
}
