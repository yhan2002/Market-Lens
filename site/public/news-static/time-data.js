"use strict";
globalThis.CSQTimeData = {
 issues(articles,hours) {
  const groups=new Map();
  for(const a of articles){const id=a.issue_groups?.[String(hours)]||a.id;if(!groups.has(id))groups.set(id,[]);groups.get(id).push(a);}
  const issues=[];
  for(const [id,items] of groups){
   items.sort((a,b)=>a.published_at.localeCompare(b.published_at));
   const lead=items.reduce((a,b)=>b.importance.score>a.importance.score||b.importance.score===a.importance.score&&b.published_at>a.published_at?b:a);
   const links=new Map(),themes=new Map(),sentiments={};
   for(const a of items){
    sentiments[a.sentiment.label]=(sentiments[a.sentiment.label]||0)+1;
    for(const link of a.links){if(!links.has(link.symbol))links.set(link.symbol,{symbol:link.symbol,name:link.name,evidence:new Set(),article_ids:[]});const e=links.get(link.symbol);link.evidence.forEach(v=>e.evidence.add(v));e.article_ids.push(a.id);}
    for(const [name,words] of Object.entries(a.themes)){if(!themes.has(name))themes.set(name,{name,evidence:new Set(),article_ids:[]});const e=themes.get(name);words.forEach(v=>e.evidence.add(v));e.article_ids.push(a.id);}
   }
   const edges=[...links.values()].map(e=>({...e,evidence:[...e.evidence].sort()}));
   const transactions=items.map(a=>a.transaction).filter(Boolean);
   issues.push({id,title:lead.title,summary:lead.summary,transaction:transactions.find(t=>t.amount)||transactions[0]||null,
    in_scope:!!edges.length||items.some(a=>Object.keys(a.macro_events||{}).length),score:lead.importance.score,reasons:lead.importance.reasons,
    level:lead.importance.level||'본문 평가 대기',basis:lead.importance.basis||'제목 규칙 · 임시 선별',
    category:lead.importance.reasons.find(r=>r!=='미확정 보도')||'일반 뉴스',
    status:items.some(a=>a.importance.reasons.includes('미확정 보도'))?'미확정 보도':items.some(a=>a.official_source)?'공식 출처 포함':'언론 보도 · 확정 여부 미판정',
    article_ids:items.map(a=>a.id),first_at:items[0].published_at,last_at:items.at(-1).published_at,
    sources:[...new Set(items.map(a=>a.source))].sort(),sentiments,links:edges,
    themes:[...themes.values()].map(e=>({...e,evidence:[...e.evidence].sort()}))});
  }
  return issues.sort((a,b)=>b.score-a.score||b.last_at.localeCompare(a.last_at));
 },
 async load(catalog,previous,{signal,base='/api',progress=()=>{}}={}) {
  if(catalog.protocol!==2)return catalog;
  if(previous?.revision===catalog.revision&&previous.hours===catalog.hours)return previous;
  const articles=[];const ids=new Set();let cursor=null;const cursors=new Set();
  do {
   const params=new URLSearchParams({hours:String(catalog.hours),revision:catalog.revision});if(cursor)params.set('cursor',cursor);
   const response=await fetch(`${base}/articles?${params}`,{signal});const page=await response.json();
   if(!response.ok)throw new Error(page.error||'기사 조회 실패');
   if(page.revision!==catalog.revision||!Array.isArray(page.articles))throw new Error('기사 데이터 버전 불일치');
   for(const a of page.articles){if(ids.has(a.id))throw new Error('중복 페이지 감지');ids.add(a.id);articles.push(a);}
   progress(articles.length);cursor=page.next;
   if(cursor){if(cursors.has(cursor))throw new Error('반복 페이지 감지');cursors.add(cursor);}
  }while(cursor);
  const check=await fetch(`${base}/dashboard?hours=${catalog.hours}`,{signal});
  if(!check.ok||(await check.json()).revision!==catalog.revision||articles.length!==catalog.summary.count)throw new Error('동기화 중입니다. 잠시 후 다시 조회해주세요.');
  return {...catalog,articles,issues:this.issues(articles,catalog.hours)};
 }
};
