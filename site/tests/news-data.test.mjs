import assert from 'node:assert/strict';
import {DatabaseSync} from 'node:sqlite';
import {readFileSync} from 'node:fs';
import {validSync,saveSync,articlePage} from '../worker/news-data.mjs';
import {newsRoute,signedSession} from '../worker/news.mjs';
import '../public/news-static/time-data.js';

const sqlite=new DatabaseSync(':memory:');
sqlite.exec(readFileSync(new URL('../drizzle/0013_damp_mole_man.sql',import.meta.url),'utf8'));
const db={prepare(sql){return {bind(...args){const s=sqlite.prepare(sql);return {
 async all(){return {results:s.all(...args)};},async first(){return s.get(...args);},async run(){return s.run(...args);}};}};},
 async batch(statements){sqlite.exec('BEGIN');try{for(const s of statements)await s.run();sqlite.exec('COMMIT');}catch(e){sqlite.exec('ROLLBACK');throw e;}}};
const revision='2026-09-16T09:00:00.000000+00:00',at=Date.parse(revision)-1000;
const rows=Array.from({length:53},(_,i)=>{const id=String(i).padStart(3,'0');return {id,published_at:at,first_collected_at:null,analysis_updated_at:revision,
 body:JSON.stringify({id,published_at:new Date(at).toISOString(),title:'Company signs contract '+i})};});
for(const batch of [rows.slice(0,50),rows.slice(50)]){const v={generated_at:revision,articles:batch};assert(validSync(v));await saveSync(db,v);}
const catalogs=[0,24,72,168,720].map(hours=>({hours,body:JSON.stringify({hours,as_of:revision,summary:{count:53}})}));
await saveSync(db,{generated_at:revision,articles:[],catalogs});
const url=new URL('https://example.com/news/api/articles?hours=24&revision='+encodeURIComponent(revision));
const first=await articlePage(db,url);assert.equal(first.articles.length,50);url.searchParams.set('cursor',first.next);
const last=await articlePage(db,url);assert.equal(last.articles.length,3);assert.equal(last.next,null);
assert.equal(new Set([...first.articles,...last.articles].map(a=>a.id)).size,53);
await saveSync(db,{generated_at:'2000-01-01T00:00:00.000000+00:00',articles:[{...rows[0],body:JSON.stringify({id:'000',title:'stale'})}]});
assert(!sqlite.prepare('SELECT body FROM news_articles WHERE id=?').get('000').body.includes('stale'));
for(const ms of [at-25*3600000,at+3600000]){
 const a={...rows[0],id:String(ms),published_at:ms,body:JSON.stringify({id:String(ms),published_at:new Date(ms).toISOString(),title:'Outside current window'})};
 await saveSync(db,{generated_at:revision,articles:[a]});
}
url.searchParams.delete('cursor');assert.equal((await articlePage(db,url)).articles.length,50);
url.searchParams.set('hours','0');const history=await articlePage(db,url);url.searchParams.set('cursor',history.next);assert.equal((await articlePage(db,url)).articles.length,4);
url.searchParams.set('hours','24');url.searchParams.set('revision','old');assert.equal((await articlePage(db,url)).status,409);
assert(!validSync({generated_at:revision,articles:[{...rows[0],published_at:'not-time'}]}));
assert(!validSync({generated_at:revision,articles:rows}));
const env={DB:db,NEWS_AUTH:JSON.stringify({salt:'ab'.repeat(16),iterations:600000,verifier:'cd'.repeat(32)}),NEWS_SESSION_KEY:'test',NEWS_UPLOAD_TOKEN:'test-upload'};
assert.equal((await newsRoute(new Request('https://example.com/news/api/articles'),env)).status,401);
assert.equal((await newsRoute(new Request('https://example.com/news/api/sync'),env)).status,401);
const token=await signedSession('test'+'cd'.repeat(32),Date.now()+10000);
const response=await newsRoute(new Request('https://example.com/news/api/dashboard?hours=24',{headers:{cookie:'news_session='+token}}),env);
assert.equal((await response.json()).protocol,2);
const article={id:'a',published_at:revision,title:'A',summary:'Summary',source:'Source',importance:{score:75,reasons:['계약'],high:true},links:[],themes:{},sentiment:{label:'positive'},macro_events:{CPI:['CPI']},issue_groups:{24:'group'}};
const issues=CSQTimeData.issues([article,{...article,id:'b'}],24);
assert.equal(issues.length,1);assert.equal(issues[0].article_ids.length,2);assert.equal(issues[0].in_scope,true);
const catalog={protocol:2,revision,hours:24,summary:{count:1}};
globalThis.fetch=async address=>Response.json(address.includes('/articles?')?{articles:[article],next:null,revision}:catalog);
const loaded=await CSQTimeData.load(catalog,null);assert.equal(loaded.articles.length,1);
globalThis.fetch=()=>{throw Error('Unchanged catalog must reuse loaded data');};assert.equal(await CSQTimeData.load(catalog,loaded),loaded);
// Removed duplicate + catalog switch happen in one batch; stale cleanup cannot erase new data.
const cleanupRevision='2026-09-16T09:01:00.000000+00:00';
const cleanupCatalogs=catalogs.map(c=>({hours:c.hours,body:JSON.stringify({hours:c.hours,as_of:cleanupRevision,summary:{count:52}})}));
const cleanup={generated_at:cleanupRevision,articles:[],catalogs:cleanupCatalogs,removed_ids:['000']};
assert(validSync(cleanup));assert(!validSync({...cleanup,catalogs:undefined}));
assert(!validSync({...cleanup,removed_ids:['000','000']}));
await saveSync(db,cleanup);
assert.equal(sqlite.prepare('SELECT COUNT(*) n FROM news_articles WHERE id=?').get('000').n,0);
assert.equal(sqlite.prepare('SELECT updated_at FROM news_catalog WHERE hours=24').get().updated_at,cleanupRevision);
url.searchParams.set('hours','24');url.searchParams.set('revision',cleanupRevision);url.searchParams.delete('cursor');
const cleanedFirst=await articlePage(db,url);url.searchParams.set('cursor',cleanedFirst.next);
const cleanedLast=await articlePage(db,url);
assert.equal(cleanedFirst.articles.length+cleanedLast.articles.length,52);
await saveSync(db,{generated_at:revision,articles:[],catalogs,removed_ids:['001']});
assert.equal(sqlite.prepare('SELECT COUNT(*) n FROM news_articles WHERE id=?').get('001').n,1);
await saveSync(db,cleanup); // Safe retry after lost acknowledgement.
assert.equal(sqlite.prepare('SELECT COUNT(*) n FROM news_articles WHERE id=?').get('001').n,1);
sqlite.close();
console.log('PASS: actual SQLite migration/upsert, stale-write protection, time ranges, tied-time pagination, auth, grouped network and cache reuse. No network.');
