"use strict";
const $=s=>document.querySelector(s);
function applyTheme(value){
 const theme=value==='light'?'light':'dark',light=theme==='light';
 document.documentElement.dataset.theme=theme;
 const button=$('#theme-toggle');button.textContent=light?'☾ 다크':'☀ 라이트';
 button.setAttribute('aria-label',light?'다크 모드로 전환':'라이트 모드로 전환');
 $('meta[name="theme-color"]')?.setAttribute('content',light?'#f3f5f8':'#0b1017');
 try{localStorage.setItem('csq-news-theme',theme);}catch{}
}
function restoreTheme(){let theme='dark';try{theme=localStorage.getItem('csq-news-theme');}catch{}applyTheme(theme);}
restoreTheme();
$('#theme-toggle').addEventListener('click',()=>applyTheme(document.documentElement.dataset.theme==='light'?'dark':'light'));
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const labels={positive:'긍정',neutral:'중립',negative:'부정'};
const date=v=>new Date(v).toLocaleString('ko-KR',{timeZone:'Asia/Seoul',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',hour12:false});
const signed=n=>`${n>0?'+':''}${n.toFixed(2)}`,tone=n=>n>.15?'positive':n<-.15?'negative':'neutral';
function sentimentTag(a){
 const s=a.sentiment;
 if(!s||!Number.isFinite(s.score)||!Object.hasOwn(labels,s.label))return '<span class="sent-tag">센티 미평가</span>';
 return `<span class="sent-tag ${s.label}" title="기사 센티 점수: -1(부정) ~ +1(긍정) · 중요도와 별개">센티 ${labels[s.label]} ${signed(s.score)}</span>`;
}
const empty=s=>`<div class="empty">${esc(s)}</div>`;
let selectionSource='auto';
let data,selectedIssue='',selectedEdge='',graphMode='flow',activeView='network',requestController,archiveIssue='',archiveStock='',archivePage=0,issueStock='',supported=new Set();
function safeURL(v){try{const u=new URL(v);return ['http:','https:'].includes(u.protocol)&&!u.username&&!u.password?u.href:'';}catch{return '';}}
const sourceLink=(a,title=a.title)=>safeURL(a.url)?`<a href="${esc(safeURL(a.url))}" target="_blank" rel="noopener noreferrer">${esc(title)} ↗</a>`:esc(title);
const currentIssue=()=>data?.issues.find(i=>i.id===selectedIssue);
function visibleIssues(){const term=$('#issue-search').value.trim().toLowerCase(),cat=$('#issue-category').value;return (data?.issues||[]).filter(i=>($('#issue-level').value!=='high'||(i.score>=70&&i.in_scope))&&(cat==='all'||i.category===cat)&&(!issueStock||i.links.some(s=>s.symbol===issueStock))&&(!term||`${i.title} ${i.links.map(s=>`${s.symbol} ${s.name}`).join(' ')} ${i.category}`.toLowerCase().includes(term))).sort((a,b)=>($('#issue-sort').value==='latest'?0:(b.score||0)-(a.score||0))||String(b.last_at||'').localeCompare(String(a.last_at||'')));}
async function load(){requestController?.abort();const controller=new AbortController();requestController=controller;$('#reload').disabled=true;$('#content').setAttribute('aria-busy','true');try{
 const r=await fetch(`/news/api/dashboard?mode=${$('#mode').value}&hours=${$('#hours').value}`,{signal:controller.signal});if(r.status===401){window.location.replace('/news/login');return;}let payload=await r.json();if(!r.ok)throw new Error(payload.error||'데이터 조회 실패');
 payload=await CSQTimeData.load(payload,data,{signal:controller.signal,base:'/news/api',progress:n=>{$('#status').textContent=`선택 기간 기사 불러오는 중 · ${n}건`;}});
 const previous=currentIssue();data=payload;data.issues||=[];if(previous&&!currentIssue())selectedIssue=data.issues.find(i=>i.article_ids.some(id=>previous.article_ids.includes(id)))?.id||'';
 supported=new Set(data.universe?.symbols||data.stocks.map(s=>s.symbol));$('#error').hidden=true;$('#content').hidden=false;
 const cat=$('#issue-category').value,cats=[...new Set(data.issues.map(i=>i.category))];$('#issue-category').innerHTML='<option value="all">모든 사건</option>'+cats.map(c=>`<option value="${esc(c)}">${esc(c)}</option>`).join('');$('#issue-category').value=cats.includes(cat)?cat:'all';
 const macro=$('#macro-filter').value;$('#macro-filter').innerHTML='<option value="all">모든 뉴스</option><option value="macro">주요 지표</option>'+(data.indicators||[]).map(i=>`<option value="${esc(i.id)}">${esc(i.name)}</option>`).join('');$('#macro-filter').value=['all','macro',...(data.indicators||[]).map(i=>i.id)].includes(macro)?macro:'all';render();
 }catch(e){if(e.name==='AbortError')return;$('#error').textContent=`갱신 실패: ${e.message}${data?' · 마지막 데이터를 유지합니다.':''}`;$('#error').hidden=false;}finally{if(requestController===controller){requestController=undefined;$('#reload').disabled=false;$('#content').setAttribute('aria-busy','false');}}}
function render(){const m=data.metadata;$('#clock').textContent=date(data.as_of)+' KST';$('#status').textContent=data.mode==='demo'?'DEMO · 모든 기사와 수치는 합성 예시입니다.':`저장 데이터 · 최근 수집 ${m.last_collected?date(m.last_collected):'없음'} KST · Market Lens UNIV ${data.stocks.length}종목 · 화면 1분 자동 갱신`;
 const p=data.processing,q=p?.qwen;if(p&&data.mode!=='demo')$('#status').textContent+=` · 선택 기간 Qwen 성공 ${q.done||0} / 대기 ${q.pending||0} / 재시도 대기 ${q.retry_wait||0} / 한도 도달 ${q.exhausted||0} · Luna 검토 ${p.luna_reviewed}건 (본문 요약 ${p.luna_bodies}건) · 마지막 ${p.luna_last?date(p.luna_last):'없음'}`;
 if(m.errors?.length){$('#error').innerHTML=`<details><summary>${m.errors.some(e=>/HTTP Error 50[0234]/.test(e))?'일부 피드 수집 지연':'일부 데이터 확인 필요'} · ${m.errors.length}건 <span>상세 보기</span></summary><p>수집에 성공한 저장 뉴스는 계속 표시됩니다. 아래 피드의 최신 뉴스는 누락됐을 수 있습니다.</p><ul>${m.errors.map(e=>`<li>${esc(e)}</li>`).join('')}</ul></details>`;$('#error').hidden=false;}
 const stats=[['수집 뉴스',data.articles.length,'선택 기간 전체 기사'],['우선 검토 후보',data.issues.filter(i=>(i.score>=70&&i.in_scope)).length,'제목 선별 포함 · 최종 평가와 구분'],['직접 연결 기업',data.summary.active_stocks,`Market Lens ${data.stocks.length}종목 중`],['유사 보도 묶음',data.issues.filter(i=>i.article_ids.length>1).length,'원문은 모두 보존']];
 $('#stats').innerHTML=stats.map(([name,value,note],index)=>`<div class="stat"><span class="stat-label">${name}</span><strong class="${index===1?'amber-text':''}">${value.toLocaleString()}</strong><small>${note}</small></div>`).join('');renderIssues();renderStocks();renderSectors();renderFeed();}
function renderIssues(){const issues=visibleIssues();if(!issues.some(i=>i.id===selectedIssue)){selectedIssue=issues[0]?.id||'';selectionSource='auto';if(activeView!=='network'){selectedEdge='';graphMode='flow';}}$('#issue-count').textContent=`${issues.length} 이슈`;
 $('#issues').innerHTML=issues.map((i,index)=>`<button class="issue ${i.id===selectedIssue?'active':''}" data-issue="${esc(i.id)}" aria-pressed="${i.id===selectedIssue}"><span class="issue-top"><span class="micro">${String(index+1).padStart(2,'0')} / ${esc(i.category)}</span><b class="score" title="${esc(i.basis)}">${esc(i.level==='중요'||i.level==='필수 확인'?i.level:'검토 후보')}</b></span><strong class="issue-event-name">${esc(i.transaction?`${i.transaction.buyer} · ${i.category}`:(i.links.slice(0,2).map(l=>l.symbol).join(" · ")||"기업 미연결")+" · "+i.category)}</strong><span class="issue-subtitle">${esc(i.title)}</span><span class="issue-bottom"><span>${esc(i.links.slice(0,3).map(s=>s.symbol).join(' · ')||'거시 / 기업 미연결')}</span><span>${i.article_ids.length} 보도</span></span><span class="issue-time">${i.id===selectedIssue?'<b class="selected-label">선택한 사건</b> · ':''}${date(i.last_at)}${i.status==='미확정 보도'?' · 미확정':''}</span></button>`).join('')||empty('해당 조건의 이슈가 없습니다. 모든 이슈로 전환하거나 기간을 넓혀보세요.');renderFocus();}
function relatedIssues(i){
 const symbols=new Set(i.links.map(l=>l.symbol)),own=new Set(i.article_ids);
 return (data?.issues||[]).filter(other=>other.id!==i.id&&!other.article_ids.some(id=>own.has(id)))
 .map(other=>({issue:other,shared:other.links.filter(l=>symbols.has(l.symbol))}))
 .filter(r=>r.shared.length).sort((a,b)=>(b.issue.score||0)-(a.issue.score||0)||String(b.issue.last_at||'').localeCompare(String(a.issue.last_at||'')));
}
function edgeRecords(i){
 const edges=[...i.links.map(s=>({...s,key:`stock:${s.symbol}`,label:s.symbol,type:'stock'})),...i.themes.map(s=>({...s,key:`theme:${s.name}`,label:s.name,type:'theme'}))];
 if(i.transaction)for(const role of ['buyer','asset','seller'])edges.push({key:`role:${role}`,label:i.transaction[role],type:'role',description:{buyer:'인수자',asset:'인수 대상',seller:'매도자'}[role],evidence:[i.transaction.evidence],article_ids:[i.transaction.article_id]});
 for(const r of relatedIssues(i))for(const link of r.shared)edges.push({key:`shared:${r.issue.id}:${link.symbol}`,type:'shared',label:link.symbol,evidence:[link.symbol],article_ids:[...new Set([...i.links.find(l=>l.symbol===link.symbol).article_ids,...link.article_ids])],description:'두 이슈의 공통 언급 기업'});
 return edges;
}
function graphNode({x,y,w=220,h=100,label,role,note='',kind='',action='',key=''}){
 const attrs=action?`data-${action}="${esc(key)}" tabindex="0" role="button" aria-label="${esc(role+': '+label+' · 근거 보기')}"`:'';
 return `<g class="node ${kind} ${selectedEdge===key?'selected':''}" ${attrs}><title>${esc(label)}</title><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="10"/><foreignObject x="${x+14}" y="${y+10}" width="${w-28}" height="${h-20}" pointer-events="none"><div xmlns="http://www.w3.org/1999/xhtml" class="graph-card-text"><span class="graph-card-role">${esc(role)}</span><strong>${esc(label)}</strong>${note?`<small>${esc(note)}</small>`:''}</div></foreignObject></g>`;
}
function graphWire(x1,y1,x2,y2,key='',kind='',label=''){
 const path=`M${x1},${y1} C${(x1+x2)/2},${y1} ${(x1+x2)/2},${y2} ${x2},${y2}`;
 return `<g ${key?`data-edge="${esc(key)}" tabindex="0" role="button" aria-label="${esc(label||'연결 근거 보기')}"`:''}><path class="wire ${kind}" d="${path}" ${kind==='directed'?'marker-end="url(#arrow)"':''}/>${key?`<path class="link-hit" d="${path}"/>`:''}${label?`<text class="edge-label" text-anchor="middle" x="${(x1+x2)/2}" y="${(y1+y2)/2-12}">${esc(label)}</text>`:''}</g>`;
}

let networkKind='all',networkEdges=[];
let semanticNodes=[],networkView={x:0,y:0,w:1040,h:680};
function meaningGraph(issues){
 const nodes=new Map(),edges=new Map(),ids=new Set(issues.flatMap(i=>i.article_ids));
 const add=(key,label,type,id)=>{if(networkKind!=='all'&&networkKind!==type)return false;if(!nodes.has(key))nodes.set(key,{key,label,type,article_ids:new Set()});nodes.get(key).article_ids.add(id);return true;};
 for(const a of data.articles.filter(a=>ids.has(a.id)&&a.semantic_status==='grounded')){
  for(const r of a.semantic_relations||[]){
   const left='company:'+r.source_key,right='company:'+r.target_key;
   const l=add(left,r.source_key===r.source.toLowerCase()?r.source:r.source_key,'company',a.id),rr=add(right,r.target_key===r.target.toLowerCase()?r.target:r.target_key,'company',a.id);
   if(l&&rr){const pair=['competition','partnership'].includes(r.kind)?[left,right].sort():[left,right];const key='net:meaning:'+r.kind+':'+pair.join('|');if(!edges.has(key))edges.set(key,{key,left:nodes.get(pair[0]),right:nodes.get(pair[1]),kind:r.kind,article_ids:[],quotes:[]});const e=edges.get(key);if(!e.article_ids.includes(a.id))e.article_ids.push(a.id);e.quotes.push(r.quote);}
  }
  for(const [label,quotes] of Object.entries(a.themes||{}))add('theme:'+label,label,'theme',a.id);
 }
 return {nodes:[...nodes.values()].map(n=>({...n,article_ids:[...n.article_ids]})),edges:[...edges.values()]};
}
let networkMeaning=false;
function semanticGraph(issues){
 if(networkMeaning)return meaningGraph(issues);
 const nodes=new Map(),edges=new Map();
 for(const i of issues)for(const [type,items] of [['company',i.links||[]],['theme',i.themes||[]]])for(const item of items){
  if(networkKind!=='all'&&networkKind!==type)continue;
  const label=item.symbol||item.name,key=`${type}:${label}`;
  if(!nodes.has(key))nodes.set(key,{key,label,type,article_ids:new Set()});
  for(const id of item.article_ids||[])if(i.article_ids.includes(id))nodes.get(key).article_ids.add(id);
 }
 const list=[...nodes.values()].sort((a,b)=>a.key.localeCompare(b.key));
 for(let a=0;a<list.length;a++)for(let b=a+1;b<list.length;b++){
  const ids=[...list[a].article_ids].filter(id=>list[b].article_ids.has(id));
  if(ids.length)edges.set(`${a}:${b}`,{key:`net:${list[a].key}|${list[b].key}`,left:list[a],right:list[b],article_ids:ids});
 }
 return {nodes:list.map(n=>({...n,article_ids:[...n.article_ids]})),edges:[...edges.values()]};
}
function semanticPositions(nodes,edges){
 const points=nodes.map((n,i)=>({...n,x:520+320*Math.cos(i*2.39996),y:340+320*Math.sin(i*2.39996),r:Math.max(n.type==='theme'?34:22,Math.min(40,18+Math.sqrt(n.article_ids.length)*2))})),byKey=new Map(points.map(p=>[p.key,p]));
 // ponytail: bounded O(n²) layout; use a spatial index if the universe grows to thousands.
 for(let step=0;step<100;step++){
  const forces=points.map(p=>({x:(520-p.x)*.008,y:(340-p.y)*.008}));
  for(let a=0;a<points.length;a++)for(let b=a+1;b<points.length;b++){
   const dx=points[a].x-points[b].x,dy=points[a].y-points[b].y,d=Math.max(1,Math.hypot(dx,dy)),f=Math.min(24,3200/(d*d)+Math.max(0,points[a].r+points[b].r+55-d)*.18);
   forces[a].x+=dx/d*f;forces[a].y+=dy/d*f;forces[b].x-=dx/d*f;forces[b].y-=dy/d*f;
  }
  for(const e of edges){const a=byKey.get(e.left.key),b=byKey.get(e.right.key),dx=b.x-a.x,dy=b.y-a.y,d=Math.max(1,Math.hypot(dx,dy)),f=(d-230)*.006;const ai=points.indexOf(a),bi=points.indexOf(b);forces[ai].x+=dx/d*f;forces[ai].y+=dy/d*f;forces[bi].x-=dx/d*f;forces[bi].y-=dy/d*f;}
  points.forEach((p,i)=>{p.x+=Math.max(-20,Math.min(20,forces[i].x));p.y+=Math.max(-20,Math.min(20,forces[i].y));});
 }
 return points;
}
function fitNetwork(){
 if(!semanticNodes.length){networkView={x:0,y:0,w:1040,h:680};return;}
 const x=Math.min(...semanticNodes.map(n=>n.x-n.r))-35,y=Math.min(...semanticNodes.map(n=>n.y-n.r))-35;
 networkView={x,y,w:Math.max(100,Math.max(...semanticNodes.map(n=>n.x+n.r))-x+35),h:Math.max(100,Math.max(...semanticNodes.map(n=>n.y+n.r))-y+35)};
}
function updateNetworkView(){document.querySelector('.semantic-svg')?.setAttribute('viewBox',`${networkView.x} ${networkView.y} ${networkView.w} ${networkView.h}`);}
function zoomNetwork(factor,x=.5,y=.5){
 const w=Math.max(120,Math.min(12000,networkView.w*factor)),ratio=w/networkView.w,h=networkView.h*ratio;
 networkView={x:networkView.x+(networkView.w-w)*x,y:networkView.y+(networkView.h-h)*y,w,h};updateNetworkView();
}
let semanticSignature='';
function renderEventMap(){
 const graph=semanticGraph(activeView==='network'?networkIssues():visibleIssues());networkEdges=graph.edges;
 const signature=JSON.stringify([networkMeaning,networkKind,graph.nodes,graph.edges.map(e=>[e.key,e.article_ids])]);
 if(signature!==semanticSignature){semanticNodes=semanticPositions(graph.nodes,graph.edges);semanticSignature=signature;fitNetwork();}
 const byKey=new Map(semanticNodes.map(n=>[n.key,n])),chosen=selectedEdge.replace('net:node:','');
 const wires=networkEdges.map(e=>{const a=byKey.get(e.left.key),b=byKey.get(e.right.key),near=e.key===selectedEdge||[a.key,b.key].includes(chosen);return `<g data-edge="${esc(e.key)}" tabindex="0" role="button" aria-label="${esc(a.label+' · '+b.label)} ${esc(e.kind||'공동 언급')} · 근거 ${e.article_ids.length}건"><path class="event-wire ${near?'chosen':'near'}" style="stroke-width:${Math.min(6,1+Math.log2(e.article_ids.length+1))};opacity:${selectedEdge&&!near?.08:.4}" d="M${a.x} ${a.y} Q${(a.x+b.x)/2+20} ${(a.y+b.y)/2-20} ${b.x} ${b.y}"/><path class="link-hit" d="M${a.x} ${a.y} Q${(a.x+b.x)/2+20} ${(a.y+b.y)/2-20} ${b.x} ${b.y}"/></g>`;}).join('');
 const nodes=semanticNodes.map(n=>`<g class="semantic-node ${n.type}" data-edge="net:node:${esc(n.key)}" tabindex="0" role="button" aria-label="${esc(n.label)} 관련 기사 ${n.article_ids.length}건"><circle cx="${n.x}" cy="${n.y}" r="${n.r}" class="${chosen===n.key?'chosen':''}"/><foreignObject x="${n.x-n.r*.8}" y="${n.y-22}" width="${n.r*1.6}" height="44" pointer-events="none"><div xmlns="http://www.w3.org/1999/xhtml" class="semantic-label ${n.type}">${n.type==='theme'?esc(n.label).split('·').join('<br>'):esc(n.label)}</div></foreignObject></g>`).join('');
 $('#map-tabs').innerHTML='';
 $('#surface').innerHTML=`<div class="event-map"><div class="event-toolbar"><label>관계 <select id="network-meaning"><option value="meaning" ${networkMeaning?'selected':''}>AI 관계 · 시험</option><option value="mention" ${!networkMeaning?'selected':''}>공동 언급</option></select></label><label>표시 <select id="network-kind">${[['all','기업 + 테마'],['company','기업'],['theme','테마']].map(([v,t])=>`<option value="${v}" ${v===networkKind?'selected':''}>${t}</option>`).join('')}</select></label><span>${semanticNodes.length}개 단위 · ${networkEdges.length}개 연결</span><div><button data-zoom="out" aria-label="지도 축소">−</button><button data-zoom="fit">전체 맞춤</button><button data-zoom="in" aria-label="지도 확대">+</button></div></div><svg class="semantic-svg event-svg" viewBox="${networkView.x} ${networkView.y} ${networkView.w} ${networkView.h}" aria-label="기업과 테마의 공동 보도 연결망" role="group">${wires}${nodes}</svg>${!semanticNodes.length?'<p class="flow-note">선택 기간에 근거 검증을 통과한 의미 판정이 없습니다. 공동 언급 보기에서 기존 기사를 확인할 수 있습니다.</p>':''}<div class="event-legend">파랑: 기업 · 보라: 테마 · 원 크기: 기사 수 · 선 굵기: 근거 기사 수</div></div>`;
 $('#keyword-links').innerHTML='<span>휠: 확대·축소 · 드래그: 이동 · 원/선: 근거 기사</span><span>'+(networkMeaning?'AI 시험 판정 · 인용·기업·관계 표현 대조 · 의미 오류 가능':'공동 언급 · 의미·인과관계 미판정')+'</span>';
}
function renderEventEvidence(){
 const node=semanticNodes.find(n=>`net:node:${n.key}`===selectedEdge),edge=networkEdges.find(e=>e.key===selectedEdge),ids=node?.article_ids||edge?.article_ids;
 if(!ids){selectedEdge='';renderEvidence();return;}
 const articles=ids.map(id=>data.articles.find(a=>a.id===id)).filter(Boolean).sort((a,b)=>(b.importance?.score||0)-(a.importance?.score||0)||String(b.published_at).localeCompare(String(a.published_at)));
 $('#edge-evidence').innerHTML=`<span class="eyebrow">SOURCE EVIDENCE</span><h3>${esc(node?.label||edge.left.label+' ↔ '+edge.right.label)}</h3><p>${node?'이 단위의 근거 기사':edge.kind?({competition:'경쟁',acquisition:'인수자 → 인수 대상',supply:'공급자 → 고객',partnership:'협력',regulation:'규제 주체 → 대상'}[edge.kind]):'공동 언급 · 의미 관계 미판정'} · ${articles.length}건 · 중요도순</p>${edge?.quotes?`<blockquote>${esc([...new Set(edge.quotes)].join(' / '))}</blockquote>`:''}${articles.map(a=>`<article class="relation-source"><div class="article-meta"><small>${date(a.published_at)} · ${esc(a.source)}</small>${sentimentTag(a)}</div><h3>${sourceLink(a)}</h3><p>${esc(a.summary||'요약 없음')}</p><button class="story-link" data-open-story="${esc(a.id)}">사건 흐름 보기 →</button></article>`).join('')}`;
}
document.addEventListener('change',e=>{if(e.target.id==='network-meaning'){networkMeaning=e.target.value==='meaning';selectedEdge='';renderFocus();}if(e.target.id==='network-kind'){networkKind=e.target.value;selectedEdge='';renderFocus();}});
document.addEventListener('click',e=>{const button=e.target.closest('[data-zoom]');if(!button)return;if(button.dataset.zoom==='fit'){fitNetwork();updateNetworkView();}else zoomNetwork(button.dataset.zoom==='in'?.8:1.25);});
document.addEventListener('wheel',e=>{const svg=e.target.closest('.semantic-svg');if(!svg)return;e.preventDefault();const point=svg.createSVGPoint();point.x=e.clientX;point.y=e.clientY;const local=point.matrixTransform(svg.getScreenCTM().inverse());zoomNetwork(Math.exp(Math.max(-200,Math.min(200,e.deltaY))*.002),(local.x-networkView.x)/networkView.w,(local.y-networkView.y)/networkView.h);},{passive:false});
let networkDrag;
document.addEventListener('pointerdown',e=>{const svg=e.target.closest('.semantic-svg');if(!svg||e.target.closest('[data-edge]')||e.button!==0)return;networkDrag={x:e.clientX,y:e.clientY,svg};svg.setPointerCapture(e.pointerId);});
document.addEventListener('pointermove',e=>{if(!networkDrag)return;const {svg,x,y}=networkDrag,m=svg.getScreenCTM();networkView.x-=(e.clientX-x)/m.a;networkView.y-=(e.clientY-y)/m.d;networkDrag.x=e.clientX;networkDrag.y=e.clientY;updateNetworkView();});
for(const event of ['pointerup','pointercancel','lostpointercapture'])document.addEventListener(event,()=>{networkDrag=null;});

function flowArticles(i){
 return [...new Set(i.article_ids)].map(id=>data.articles.find(a=>a.id===id)).filter(Boolean).sort((a,b)=>String(a.published_at||'').localeCompare(String(b.published_at||'')));
}
function groupedFlowArticles(i){
 const groups=[],seen=new Map();
 for(const a of flowArticles(i)){const key=JSON.stringify([a.title?.trim().toLowerCase(),a.summary?.trim().toLowerCase()]);if(!a.title||!seen.has(key)){const group={lead:a,duplicates:[]};groups.push(group);if(a.title)seen.set(key,group);}else seen.get(key).duplicates.push(a);}
 return groups;
}
function renderFlow(i){
 const articles=flowArticles(i),groups=groupedFlowArticles(i),related=relatedIssues(i);let lastDay='';
 $('#map-tabs').innerHTML='<button data-graph="network" class="quiet">홈에서 관계 보기 ↗</button>';
 $('#surface').innerHTML=`<div class="story-flow"><div class="flow-heading"><strong>시간순 보도 <span>${articles.length}건</span></strong><small>과거 → 최근 · 한국시간</small></div><ol class="flow-list">${groups.map((g,n)=>{const a=g.lead,day=a.published_at?date(a.published_at).slice(0,7):'시각 미확인',dayLabel=day!==lastDay?`<div class="flow-day">${esc(day)}</div>`:'';lastDay=day;return `<li>${dayLabel}<div class="flow-step">${String(n+1).padStart(2,'0')}</div><article class="flow-card"><div class="flow-meta"><time>${a.published_at?date(a.published_at)+' KST':'게시 시각 미확인'}</time><span>${n===0?'수집 범위 내 첫 보도':'추가 보도'}${a.official_source?' · 공식 출처':''}</span></div><h3>${sourceLink(a)}</h3><p class="flow-summary">${esc(a.assessment?.summary_text||a.summary||'요약이 없습니다. 원문을 확인하세요.')}</p><div class="flow-meta"><span>${esc(a.source)}</span><span>${esc(a.importance?.level||'본문 평가 대기')}</span></div><details class="flow-proof"><summary>분석 근거·전체 요약</summary><p>${esc(a.summary||'제공된 요약 없음')}</p><p>평가 범위: ${esc(a.importance?.basis||'미평가')}<br>기사 감성: ${esc(labels[a.sentiment?.label]||'미확인')}</p>${a.assessment?.evidence?.length?`<p>${esc(a.assessment.evidence.join(' / '))}</p>`:''}</details>${g.duplicates.length?`<details class="flow-duplicates"><summary>동일 내용 보도 ${g.duplicates.length}건 더 보기</summary>${g.duplicates.map(d=>`<p><time>${date(d.published_at)}</time> · ${esc(d.source)}<br>${sourceLink(d)}</p>`).join('')}</details>`:''}</article></li>`;}).join('')}</ol>${articles.length===1?'<p class="flow-note">현재 확인된 보도는 1건입니다. 후속 보도가 수집되면 이어서 표시합니다.</p>':''}<details class="flow-related"><summary>같은 기업의 다른 사건 · ${related.length}건</summary><p class="flow-note">공통 기업이 등장하는 별개 사건이며, 이 사건의 후속 보도로 판정한 관계는 아닙니다.</p>${related.map(r=>`<article class="related-story"><div class="related-story-meta"><span>${esc(r.shared.map(c=>c.symbol).join(' · '))}</span><span>${esc(r.issue.level||'검토 후보')}</span></div><button class="related-story-title" data-related="${esc(r.issue.id)}">${esc(r.issue.title)}</button><time>${date(r.issue.last_at)}</time></article>`).join('')}</details></div>`;
 $('#keyword-links').innerHTML='<span>새로 수집된 보도가 반드시 새로운 사실을 뜻하지는 않습니다.</span>';
}

function renderNetwork(i){
 if(graphMode==='flow'){renderFlow(i);return;}
 if(graphMode==='network'){renderEventMap(i);return;}
 const t=i.transaction,related=relatedIssues(i);let nodes='',wires='',caption='',height=480;
 const preferred=new Set(related.map(r=>r.shared[0].symbol));
 const companies=[...i.links].sort((a,b)=>Number(preferred.has(b.symbol))-Number(preferred.has(a.symbol))||related.filter(r=>r.shared.some(l=>l.symbol===b.symbol)).length-related.filter(r=>r.shared.some(l=>l.symbol===a.symbol)).length).slice(0,4);
 if(graphMode==='structure'&&t){
  nodes=graphNode({x:35,y:160,w:250,h:145,label:t.buyer,role:'인수자',note:'거래 당사자',action:'edge',key:'role:buyer'})+graphNode({x:375,y:145,w:250,h:180,label:t.asset,role:'인수 대상',note:t.amount?`거래금액 ${t.amount}`:'금액: 제목에 미기재',kind:'focus',action:'edge',key:'role:asset'})+graphNode({x:715,y:160,w:250,h:145,label:t.seller,role:'매도자',note:'거래 당사자',action:'edge',key:'role:seller'});
  wires=graphWire(285,230,375,230,'role:buyer','directed','인수')+graphWire(715,230,625,230,'role:seller','directed','매각');
  caption='기사 제목에 명시된 거래 구조입니다. 각 당사자를 누르면 해당 원문을 확인합니다.';
 }else if(!(graphMode==='related'&&related.length)){
  height=Math.max(480,companies.length*120+90);
  nodes=graphNode({x:65,y:height/2-80,w:325,h:160,label:i.title,role:'지금 보는 뉴스',note:i.category,kind:'focus'});
  companies.forEach((c,n)=>{const y=height/2-(companies.length*112-12)/2+n*112;nodes+=graphNode({x:650,y,w:275,h:100,label:c.name||c.symbol,role:c.symbol,note:`직접 언급 · 근거 ${c.article_ids.length}건`,action:'edge',key:`stock:${c.symbol}`});wires+=graphWire(390,height/2,650,y+50,`stock:${c.symbol}`,'',`${c.article_ids.length}건 직접 언급`);});
  caption=companies.length?`기업 ${companies.length}/${i.links.length}개 표시 · 기사에서 직접 확인되는 연결입니다. ${graphMode==='related'?'선택 기간에 같은 기업의 다른 이슈가 없습니다.':'거래 역할이 명시되지 않은 기사는 인수자·수혜자로 추정하지 않습니다.'}`:'이 기사에서 직접 연결된 Market Lens 기업을 확인하지 못했습니다. 원문에서 사건 내용을 확인하세요.';
 }
 if(graphMode==='related'&&related.length){
 $('#surface').innerHTML=`<div class="related-feed"><div class="related-feed-heading"><strong>같은 기업의 다른 뉴스 ${related.length}건</strong><span>중요도 점수 높은 순 · 동점은 최신순<br>미평가 뉴스는 임시 선별 점수 기준</span></div>${related.map((r,n)=>`<article class="related-story"><div class="related-story-meta"><span>${n+1} · ${esc(r.issue.category)}</span><span>${esc(r.issue.level||'본문 평가 대기')} · ${r.issue.score||0}점</span></div><button class="related-story-title" data-related="${esc(r.issue.id)}" title="${esc(r.issue.title)}">${esc(r.issue.title)}</button><div class="related-story-links"><span>공통 기업</span>${r.shared.map(c=>`<button data-edge="shared:${esc(r.issue.id)}:${esc(c.symbol)}" aria-label="${esc(c.symbol)} 연결 근거 보기" aria-pressed="${selectedEdge===`shared:${r.issue.id}:${c.symbol}`}">${esc(c.symbol)} · 근거 보기</button>`).join('')}<time>${date(r.issue.last_at)}</time></div></article>`).join('')}</div>`;
 }else $('#surface').innerHTML=`<div class="network-viewport"><svg class="network-svg" viewBox="0 0 1000 ${height}" role="group" aria-label="${graphMode==='related'?'공통 기업으로 연결한 연관 이슈':'기사 근거의 사건 구조'}"><defs><marker id="arrow" markerWidth="7" markerHeight="7" refX="7" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7" fill="#b4c9e3"/></marker></defs><text x="35" y="28" class="column-label">${graphMode==='related'?'① 지금 보는 뉴스 → ② 공통 기업 → ③ 다른 뉴스':'기사에 명시된 관계 · 기업을 눌러 원문 확인'}</text>${wires}${nodes}</svg><div class="network-caption">${caption}</div></div>`;
 $('#map-tabs').innerHTML=`<button data-graph="flow" aria-pressed="false">사건 흐름</button><button data-graph="structure" aria-pressed="${graphMode==='structure'}">${t?'누가 무엇을 인수하나':'이 뉴스에 나온 기업'}</button><button data-graph="related" aria-pressed="${graphMode==='related'}">같은 기업의 다른 뉴스 <span>${related.length}</span></button>`;
 $('#keyword-links').innerHTML=`<span>키워드 참고</span>${i.themes.map(t=>`<button data-edge="theme:${esc(t.name)}">${esc(t.evidence.join(' · '))}</button>`).join('')||'<span>일치 없음</span>'}<span class="keyword-note">단어 일치 · 사업 관계 미확인</span>`;
}
function renderFocus(){if(activeView==='network'){renderGlobalNetwork();return;}const i=currentIssue();if(!i){$('#map-title').textContent='이슈를 선택하세요';$('#surface').innerHTML=empty('조건에 맞는 이슈가 없습니다. 왼쪽 필터를 변경하세요.');$('#map-tabs').innerHTML='';$('#keyword-links').innerHTML='';$('#edge-evidence').innerHTML='';$('#dossier').innerHTML=empty('왼쪽에서 이슈를 선택하세요.');$('#timeline').innerHTML='';return;}
 $('#map-title').innerHTML=`<div class="selection-context"><span>${selectionSource==='auto'?'자동 선택 · 처음 표시된 목록의 첫 뉴스':selectionSource==='related'?'다른 뉴스에서 이동':'목록에서 직접 선택'}</span><span>${i.article_ids.length}개 보도 · ${i.sources.length}개 출처</span></div><h3>${esc(i.title)}</h3><div class="selection-meta"><span class="badge">${esc(i.category)}</span><strong>${esc(i.transaction?`${i.transaction.buyer} → ${i.transaction.asset}`:(i.links.map(l=>l.symbol).join(' · ')||'기업 미연결'))}</strong></div>`;
 renderNetwork(i);renderEvidence();
 const total=i.article_ids.length,distribution=Object.entries(labels).map(([key,label])=>`<span class="${key}" style="flex:${i.sentiments[key]||0}" title="${label} ${i.sentiments[key]||0}건"></span>`).join('');
 $('#dossier').innerHTML=`<div class="dossier-category"><span class="eyebrow">${esc(i.category)}</span><span class="badge ${i.status==='미확정 보도'?'caution':''}">${esc(i.status)}</span></div><h3>${esc(i.title)}</h3><div class="dossier-metrics"><div><small>중요도</small><strong class="amber-text">${esc(i.level||"본문 평가 대기")}</strong></div><div><small>보도 / 출처</small><strong>${total}<em>/ ${i.sources.length}</em></strong></div></div><h4>왜 주목하나</h4><p>${esc(i.reasons.join(' · ')||'중요 사건 키워드 없음')}<br><span class="muted">${esc(i.basis||"제목 규칙 · 임시 선별")} · 수익률 예측 아님</span></p><h4>기사 논조</h4><div class="sentiment-bar">${distribution}</div><div class="sentiment-labels">${Object.entries(labels).map(([key,label])=>`<span class="${key}">${label} ${i.sentiments[key]||0}</span>`).join('')}</div><h4>제공된 기사 요약</h4><p class="source-summary">${esc(i.summary||'피드 요약이 없습니다. 원문을 확인하세요.')}</p><h4>관련 종목</h4><div class="chips">${i.links.map(s=>`<button data-edge="stock:${esc(s.symbol)}" class="chip">${esc(s.symbol)}</button>`).join('')||'<span class="muted">직접 언급 없음</span>'}</div><button class="primary full" id="show-issue-articles">이 이슈의 원문 ${total}건 보기 ↓</button>`;
 if(selectedEdge)$('#dossier').innerHTML=$('#edge-evidence').innerHTML;
 if(activeView==='radar'&&graphMode==='flow'){
  $('#map-title').innerHTML+=`<div class="reader-summary"><span class="reader-summary-label">지금 알아야 할 내용</span><p>${esc(i.summary||i.title)}</p><div class="reader-facts"><span>중요도 <b>${esc(i.level||'본문 평가 대기')}</b></span><span>보도 상태 <b>${esc(i.status||'미확인')}</b></span><span>최근 보도 <b>${date(i.last_at)}</b></span></div></div>`;
  $('#surface').innerHTML+=`<details class="reader-details"><summary>사건 분석 상세·감성·원문</summary>${$('#dossier').innerHTML.replace(/<button class="primary full" id="show-issue-articles">[\s\S]*?<\/button>/,'')}</details>`;
  $('#edge-evidence').innerHTML='';
 }

 const items=i.article_ids.map(id=>data.articles.find(a=>a.id===id)).filter(Boolean).sort((a,b)=>a.published_at.localeCompare(b.published_at));
 $('#timeline').innerHTML=items.map((a,n)=>`<article class="timeline-entry"><span class="timeline-dot"></span><time>${date(a.published_at)} KST</time><span class="micro">${n===0?'수집 범위 내 최초':'추가 / 유사 보도'} · ${esc(a.source)}</span><h3>${sourceLink(a)}</h3><span class="${a.sentiment.label}">${labels[a.sentiment.label]} · 기사 논조</span></article>`).join('');}
function renderEvidence(){if(selectedEdge.startsWith('net:')){renderEventEvidence();return;}const i=currentIssue(),e=i&&edgeRecords(i).find(e=>e.key===selectedEdge);$('#edge-evidence').innerHTML=e?`<span class="eyebrow">SOURCE EVIDENCE</span><strong>${esc(e.label)} · ${esc(e.description||({stock:'기업 직접 언급',theme:'키워드 일치 · 사업 관계 미확인'}[e.type]))}</strong><p>${esc(e.type==='role'?e.evidence.join(' · '):'일치 표현: '+e.evidence.join(' · '))}</p><div>${e.article_ids.map(id=>data.articles.find(a=>a.id===id)).filter(Boolean).map(a=>`<p>${sourceLink(a)} · ${sentimentTag(a)}</p>`).join('')}</div>`:'<div class="evidence-empty"><b>↗</b><div><strong>연결의 근거를 확인하세요</strong><p>기업·연결선을 선택하면 실제 기사가 여기에 표시됩니다.</p></div></div>';}
function sortedStocks(stocks,articles,key,direction){
 const high={};for(const a of articles)if(a.importance?.high)for(const symbol of new Set(a.links.map(s=>s.symbol)))high[symbol]=(high[symbol]||0)+1;
 const value=s=>key==='high'?(high[s.symbol]||0):key==='sentiment'?(s.count?s.sentiment:null):key==='market_cap'?s.market_cap:s.count;
 return stocks.map(s=>({...s,high:high[s.symbol]||0})).sort((a,b)=>{const x=value(a),y=value(b),xm=!Number.isFinite(x),ym=!Number.isFinite(y);return xm!==ym?(xm?1:-1):(!xm?(x-y)*(direction==='asc'?1:-1):0)||a.symbol.localeCompare(b.symbol);});
}
function marketCapLabel(value){return Number.isFinite(value)&&value>0?'$'+(value/1e9).toLocaleString('en-US',{maximumFractionDigits:2})+'B':'—';}
function renderStocks(){
 const term=$('#stock-search').value.trim().toLocaleLowerCase();
 const matches=data.stocks.filter(s=>`${s.symbol} ${s.name}`.toLocaleLowerCase().includes(term));
 const rows=sortedStocks(matches,data.articles,$('#stock-sort').value,$('#stock-direction').value);
 $('#universe-count').textContent=`${rows.length} / ${data.stocks.length} 종목`;
 $('#stock-table').innerHTML=rows.map(s=>`<tr><td><strong>${esc(s.symbol)}</strong><small>${esc(s.name)}</small></td><td title="${esc(s.market_cap_as_of?'시총 확인: '+date(s.market_cap_as_of)+' KST':'시총 미수집')}">${marketCapLabel(s.market_cap)}</td><td>${s.count}</td><td class="amber-text">${s.high}</td><td class="${tone(s.sentiment)}">${s.count?signed(s.sentiment):'—'}</td><td><button class="quiet" data-stock="${esc(s.symbol)}">이슈 보기 ↗</button></td></tr>`).join('')||'<tr><td colspan="6">검색 결과가 없습니다. 티커 또는 종목명을 확인하세요.</td></tr>';
}
for(const id of ['stock-sort','stock-direction'])$(`#${id}`).addEventListener('change',()=>{if(data)renderStocks();});
$('#stock-search').addEventListener('input',()=>{if(data)renderStocks();});
function renderFeed(){if(!data)return;const term=$('#search').value.trim().toLowerCase(),macro=$('#macro-filter').value,sent=$('#sentiment-filter').value,focus=data.issues.find(i=>i.id===archiveIssue);
 const rows=data.articles.filter(a=>(!focus||focus.article_ids.includes(a.id))&&(!archiveStock||a.links.some(s=>s.symbol===archiveStock))&&($('#importance-filter').value!=='high'||a.importance?.high)&&(sent==='all'||a.sentiment.label===sent)&&(macro==='all'||(macro==='macro'?Object.keys(a.macro_events||{}).length>0:Object.hasOwn(a.macro_events||{},macro)))&&(!term||`${a.title} ${a.summary} ${a.links.map(s=>s.symbol).join(' ')}`.toLowerCase().includes(term)));
 const size=30,pages=Math.max(1,Math.ceil(rows.length/size));archivePage=Math.min(archivePage,pages-1);$('#feed-title').textContent=focus?'선택 이슈의 원문':archiveStock?`${archiveStock} 원문`:'전체 뉴스 원문';$('#selection').textContent=`${rows.length.toLocaleString()}건 · ${archivePage+1}/${pages} 페이지`;
 $('#articles').innerHTML=rows.slice(archivePage*size,(archivePage+1)*size).map(a=>`<article class="article"><div class="article-meta"><span>${esc(a.source)}${a.official_source?' · 공식 출처':''}</span><time>${date(a.published_at)} KST</time></div><h3>${sourceLink(a)}</h3><p>${esc(a.summary)||'피드 요약 없음'}</p><div class="chips"><span class="badge ${a.importance?.high?'caution':''}">${esc(a.importance?.level||"본문 평가 대기")}</span>${a.links.map(s=>`<button data-stock="${esc(s.symbol)}" class="chip">${esc(s.symbol)}</button>`).join('')}${sentimentTag(a)}</div><details><summary>분석 근거</summary><p>평가 범위: ${esc(a.importance?.basis||"제목 규칙")}<br>원문 근거: ${esc((a.assessment?.evidence||[]).join(" / ")||"본문 평가 대기")}<br>중요도: ${esc((a.importance?.reasons||[]).join(' · ')||'일반 뉴스')}<br>기업 언급: ${esc(a.links.map(s=>`${s.symbol}: ${s.evidence.join(', ')}`).join(' / ')||'없음')}<br>조회 피드: ${esc((a.feed_symbols||[]).join(', '))} (직접 언급 근거 아님)<br>분석 엔진: ${esc(a.sentiment.engine)} · 제목 + 피드 요약</p></details></article>`).join('')||empty('조건에 맞는 뉴스가 없습니다.');$('#archive-pages').innerHTML=`<button class="quiet" data-page="prev" ${archivePage===0?'disabled':''}>← 이전</button><span>${archivePage+1} / ${pages}</span><button class="quiet" data-page="next" ${archivePage+1>=pages?'disabled':''}>다음 →</button>`;}
const sectorNames={'Technology':'기술','Finance':'금융','Financials':'금융','Health Care':'헬스케어','Healthcare':'헬스케어','Consumer Discretionary':'경기소비재','Consumer Staples':'필수소비재','Industrials':'산업재','Energy':'에너지','Utilities':'유틸리티','Real Estate':'부동산','Basic Materials':'기초소재','Telecommunications':'통신','Communication Services':'커뮤니케이션','Miscellaneous':'기타','Unclassified':'미분류'};
let selectedSector='';
function sectorRows(stocks,articles){
 const bySymbol=new Map(stocks.map(s=>[s.symbol,s.sector||'Unclassified'])),groups=new Map();
 for(const sector of bySymbol.values())if(!groups.has(sector))groups.set(sector,{sector,total:0,count:0,high:0,article_ids:[]});
 for(const a of articles){
  const score=a.sentiment?.score;if(!Number.isFinite(score))continue;
  for(const sector of new Set((a.links||[]).map(l=>bySymbol.get(l.symbol)).filter(Boolean))){const g=groups.get(sector);if(g.article_ids.includes(a.id))continue;g.article_ids.push(a.id);g.total+=score;g.count++;if(a.importance?.high)g.high++;}
 }
 return [...groups.values()].map(g=>({...g,average:g.count?g.total/g.count:null}));
}
function renderSectors(){
 if(!data)return;const key=$('#sector-sort').value||'total',direction=$('#sector-direction').value==='asc'?1:-1;
 const rows=sectorRows(data.stocks,data.articles).sort((a,b)=>Number(!a.count)-Number(!b.count)||((a[key]||0)-(b[key]||0))*direction||a.sector.localeCompare(b.sector));
 const max=Math.max(1,...rows.map(g=>Math.abs(g.total)));
 $('#sector-table').innerHTML=rows.map(g=>`<tr class="${g.sector===selectedSector?'sector-selected':''}"><td><button class="sector-pick" data-sector="${esc(g.sector)}" aria-pressed="${g.sector===selectedSector}">${esc(sectorNames[g.sector]||g.sector)}<small>${esc(g.sector)}</small></button></td><td class="sector-number ${tone(g.total)}">${g.count?signed(g.total):'—'}<div class="sector-meter"><i style="width:${Math.abs(g.total)/max*100}%"></i></div></td><td>${g.count}</td><td>${g.high}</td><td class="${tone(g.average||0)}">${g.count?signed(g.average):'—'}</td></tr>`).join('');
 $('#sector-reset').textContent=selectedSector?'선택 해제 · 전체 보기':'전체 섹터 보는 중';
}
function selectSector(sector){selectedSector=sector===selectedSector?'':sector;selectedEdge='';$('#network-search').value='';renderSectors();renderGlobalNetwork();}
for(const id of ['sector-sort','sector-direction'])$(`#${id}`).addEventListener('change',renderSectors);
$('#sector-reset').addEventListener('click',()=>{selectedSector='';selectedEdge='';renderSectors();renderGlobalNetwork();});
document.addEventListener('click',e=>{const button=e.target.closest('[data-sector]');if(button)selectSector(button.dataset.sector);});
function networkIssues(){
 const term=$('#network-search').value.trim().toLowerCase();
 const symbols=new Set((data?.stocks||[]).filter(s=>(s.sector||'Unclassified')===selectedSector).map(s=>s.symbol));
 const ids=new Set((data?.articles||[]).filter(a=>a.links.some(l=>symbols.has(l.symbol))).map(a=>a.id));
 return (data?.issues||[]).filter(i=>(!selectedSector||i.article_ids.some(id=>ids.has(id)))&&(!term||`${i.title} ${i.links.map(l=>`${l.symbol} ${l.name}`).join(' ')} ${(i.themes||[]).map(t=>t.name).join(' ')}`.toLowerCase().includes(term)));
}

function renderGlobalNetwork(){
 if(!data)return;
 renderEventMap();renderEventEvidence();
 $('#map-heading').textContent=selectedSector?(sectorNames[selectedSector]||selectedSector)+' · 연결망':'전체 연결망';
 $('#dossier').innerHTML=selectedEdge?$('#edge-evidence').innerHTML:`<div class="network-intro"><span class="eyebrow">EXPLORE THE CONNECTIONS</span><h3>연결의 근거를 확인하세요</h3><p>원을 선택하면 관련 기사를, 선을 선택하면 두 단위의 연결 근거를 확인할 수 있습니다.</p><div class="network-totals"><span><b>${semanticNodes.length}</b> 기업·테마</span><span><b>${networkEdges.length}</b> 연결</span></div><h4>기사 수가 많은 단위</h4><div class="network-directory">${[...semanticNodes].sort((a,b)=>b.article_ids.length-a.article_ids.length).slice(0,10).map(n=>`<button data-edge="net:node:${esc(n.key)}"><span>${esc(n.label)}</span><small>${n.article_ids.length}건 ↗</small></button>`).join('')||'<p>검색어를 지우거나 기간을 넓혀보세요.</p>'}</div><p class="network-note">공동 언급은 인과관계가 아닙니다. AI 관계는 원문 근거를 함께 확인하세요.</p></div>`;
 if(selectedSector&&!selectedEdge){const ids=new Set(sectorRows(data.stocks,data.articles).find(g=>g.sector===selectedSector)?.article_ids||[]);$('#dossier').innerHTML=`<div class="sector-news"><h3>${esc(sectorNames[selectedSector]||selectedSector)} · 관련 뉴스 ${ids.size}건</h3>${data.articles.filter(a=>ids.has(a.id)).sort((a,b)=>String(b.published_at).localeCompare(String(a.published_at))).map(a=>`<button class="sector-news-item" data-open-story="${esc(a.id)}"><time>${date(a.published_at)}</time><strong>${esc(a.title)}</strong><span>기사 보기 →</span></button>`).join('')||empty('선택 기간에 해당 섹터 뉴스가 없습니다.')}</div>`;}
}
function changeView(view){
 if(!['network','radar','stocks','method'].includes(view))return;
 if($('#map-dialog').open)$('#map-dialog').close();
 if(activeView!==view)selectedEdge='';
 activeView=view;document.body.dataset.workspace=view;
 for(const v of ['radar','stocks','method'])$(`#${v}-view`).hidden=v==='radar'?!['radar','network'].includes(view):v!==view;
 const copy={network:['오늘의 뉴스 한눈에','왼쪽에서 섹터를 선택하고, 오른쪽에서 기업 관계를 확인하세요.'],radar:['사건 흐름','사건을 선택하고, 시간순으로 보도와 근거를 읽으세요.'],stocks:['종목 유니버스','관심 기업에서 시작해 관련 뉴스를 탐색하세요.'],method:['분석 기준','데이터 범위와 자동 판정의 근거를 확인하세요.']};
 $('#page-title').textContent=copy[view][0];$('#page-description').textContent=copy[view][1];
 $('#map-heading').textContent=view==='network'?'기업·테마 지도':'선택 사건의 보도';$('#dossier-heading').textContent=view==='network'?'연결 근거':'사건 상세';
 document.querySelectorAll('.nav-item').forEach(b=>{b.classList.toggle('active',b.dataset.view===view);if(b.dataset.view===view)b.setAttribute('aria-current','page');else b.removeAttribute('aria-current');});
 if(data&&['network','radar'].includes(view))renderFocus();
}
function openStory(articleId){
 const issue=data?.issues.find(i=>i.article_ids.includes(articleId));if(!issue)return;
 selectedIssue=issue.id;selectionSource='related';graphMode='flow';issueStock='';
 $('#issue-search').value='';$('#issue-level').value='all';$('#issue-category').value='all';
 changeView('radar');renderIssues();$('#map-heading').setAttribute('tabindex','-1');$('#map-heading').focus();
}
$('#network-search').addEventListener('input',()=>{selectedEdge='';renderGlobalNetwork();});
$('#network-reset').addEventListener('click',()=>{$('#network-search').value='';selectedSector='';selectedEdge='';renderSectors();renderGlobalNetwork();});

function handleTarget(t){if(t.dataset.openStory){openStory(t.dataset.openStory);return;}if(t.dataset.graph==='network'){const symbol=currentIssue()?.links[0]?.symbol||'';selectedSector='';$('#network-search').value=symbol;changeView('network');return;}if(t.dataset.graph){graphMode=t.dataset.graph;selectedEdge='';renderFocus();return;}if(t.dataset.related){selectedIssue=t.dataset.related;selectionSource='related';selectedEdge='';issueStock='';$('#issue-search').value='';$('#issue-level').value='all';$('#issue-category').value='all';renderIssues();return;}if(t.dataset.view)return changeView(t.dataset.view);if(t.dataset.issue){selectedIssue=t.dataset.issue;selectionSource='manual';selectedEdge='';renderIssues();}if(t.dataset.edge){selectedEdge=t.dataset.edge;renderFocus();}if(t.dataset.stock){issueStock=t.dataset.stock;changeView('radar');$('#issue-search').value=t.dataset.stock;$('#issue-level').value='all';$('#issue-category').value='all';archiveStock=t.dataset.stock;archiveIssue='';archivePage=0;renderIssues();renderFeed();}}
function resetArchiveFilters(){$('#search').value='';for(const id of ['macro-filter','importance-filter','sentiment-filter'])$(`#${id}`).value='all';}
 document.addEventListener('click',e=>{const t=e.target.closest('[data-view],[data-issue],[data-edge],[data-stock],[data-graph],[data-related],[data-open-story]');if(t)handleTarget(t);if(e.target.closest('#reset-map')){selectedEdge='';renderFocus();}if(e.target.closest('#show-issue-articles')){archiveIssue=selectedIssue;archiveStock='';archivePage=0;resetArchiveFilters();renderFeed();$('.archive').scrollIntoView({behavior:'smooth'});}if(e.target.closest('#archive-reset')){archiveIssue='';archiveStock='';archivePage=0;resetArchiveFilters();renderFeed();}const page=e.target.closest('[data-page]');if(page&&!page.disabled){archivePage+=page.dataset.page==='next'?1:-1;renderFeed();}});
 document.addEventListener('keydown',e=>{const g=e.target.closest('g[data-edge],g[data-related],g[data-issue]');if(g&&['Enter',' '].includes(e.key)){e.preventDefault();const key=g.dataset.edge;handleTarget(g);if(key)[...document.querySelectorAll('g[data-edge]')].find(n=>n.dataset.edge===key)?.focus();}});
for(const id of ['issue-search','issue-level','issue-category','issue-sort'])$(`#${id}`).addEventListener(id==='issue-search'?'input':'change',()=>{if(id==='issue-search')issueStock='';selectedEdge='';renderIssues();});
for(const id of ['search','macro-filter','importance-filter','sentiment-filter'])$(`#${id}`).addEventListener(id==='search'?'input':'change',()=>{archivePage=0;renderFeed();});
$('#mode').addEventListener('change',()=>{selectedIssue='';selectedEdge='';archiveIssue='';archivePage=0;load();});$('#hours').addEventListener('change',()=>{archivePage=0;load();});$('#reload').addEventListener('click',load);
let mapOrigin;
function toggleMapSize(){
 const dialog=$('#map-dialog'),panel=$('.map-panel'),button=$('#expand-map');
 if(dialog.open){dialog.close();return;}
 mapOrigin={parent:panel.parentNode,next:panel.nextSibling};
 dialog.append(panel);button.textContent='↙ 축소';button.setAttribute('aria-expanded','true');
 dialog.showModal();button.focus();
}
$('#expand-map').addEventListener('click',toggleMapSize);
$('#map-dialog').addEventListener('close',()=>{
 if(!mapOrigin)return;
 mapOrigin.parent.insertBefore($('.map-panel'),mapOrigin.next);mapOrigin=undefined;
 const button=$('#expand-map');button.textContent='⛶ 확대';button.setAttribute('aria-expanded','false');button.focus();
});
$('#setup').addEventListener('click',()=>{updateCommand();$('#setup-dialog').showModal();});$('#close-dialog').addEventListener('click',()=>$('#setup-dialog').close());
for(const id of ['ticker-input','engine-input','scope-input'])$(`#${id}`).addEventListener(id==='ticker-input'?'input':'change',updateCommand);
$('#copy-command').addEventListener('click',async()=>{try{await navigator.clipboard.writeText($('#collect-command').textContent);$('#command-feedback').textContent='복사했습니다. 프로젝트 PowerShell에서 실행하세요.';}catch{$('#command-feedback').textContent='위 명령을 직접 복사하세요.';}});
function refreshSaved(){if(!document.hidden&&$('#mode').value==='saved'&&!requestController&&!$('#setup-dialog').open)load();}
setInterval(refreshSaved,60000);document.addEventListener('visibilitychange',refreshSaved);window.addEventListener('pageshow',e=>{if(e.persisted)load();});load();

function updateCommand() {
  const input = $("#ticker-input").value.trim().toUpperCase();
  const raw = input === "CSQ_UNIV" ? [...supported] : input.split(",").map((part) => part.trim()).filter(Boolean);
  const symbols = [...new Set(raw)], invalid = symbols.filter((symbol) => !supported.has(symbol));
  const scope = $("#scope-input").value, macro = scope !== "stocks", macroOnly = scope === "macro";
  $("#ticker-input").disabled = macroOnly;
  const macroCount = macro ? (data?.macro_sources?.length || 0) : 0;
  const requests = (macroOnly ? 0 : symbols.length) + macroCount;
  const valid = (macroOnly || (symbols.length > 0 && !invalid.length)) && (!macro || macroCount > 0);
  $("#copy-command").disabled = !valid;
  $("#command-scope").textContent = valid ? `${macroOnly ? 0 : symbols.length}종목 + 지표 ${macroCount}피드 · 최대 ${requests * 20}건 · RSS·Atom ${requests}회 · 유료 API 없음` : `피드 정보를 불러온 뒤 지원 종목을 입력하세요: ${[...supported].join(", ")}`;
  $("#collect-command").textContent = valid ? `.\\run.ps1 collect ${macroOnly ? "--macro-only" : `--tickers '${(input === "CSQ_UNIV" ? "CSQ_UNIV" : symbols.join(","))}'${macro ? " --macro" : ""}`} --limit 20 --engine ${$("#engine-input").value}` : "미국 커버리지에 등록된 티커를 쉼표로 구분해 입력하세요.";
  $("#command-feedback").textContent = "";
}
