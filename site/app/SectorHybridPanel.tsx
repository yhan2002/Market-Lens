import type { HybridSnapshot } from '../lib/sector-hybrid';

const NAMES:Record<string,string>={XLB:'소재',XLC:'커뮤니케이션',XLE:'에너지',XLF:'금융',XLI:'산업재',XLK:'기술',XLP:'필수소비재',XLRE:'부동산',XLU:'유틸리티',XLV:'헬스케어',XLY:'경기소비재'};
export default function SectorHybridPanel({snapshot:s,date}:{snapshot:HybridSnapshot|null;date:string}) {
  const usable=s && s.as_of<=date && s.decision_date && ['READY','PARTIAL'].includes(s.status) && (Date.parse(date)-Date.parse(s.decision_date))/86400000<=10;
  if(!usable || !s) return <section className="rotation-card rotation-empty"><h3>섹터 순환 판단을 확인할 수 없습니다</h3><p>선택일에 유효한 새 엔진 기록이 없습니다. 최신화 상태와 기준일을 확인하세요. 데이터 미확인은 ‘신호 없음’이 아닙니다.</p></section>;
  const alerts=s.rows.filter(r=>r.signal==='IN'||r.signal==='OUT');
  return <section className="rotation-card hybrid-panel" aria-label="가격 기반 섹터 순환">
    <div className="rotation-topline"><span className="rotation-status">{s.status==='PARTIAL'?'일부 섹터 데이터 미확인':alerts.length?`IN ${alerts.filter(r=>r.signal==='IN').length} · OUT ${alerts.filter(r=>r.signal==='OUT').length}`:'이번 판단주 신호 없음'}</span><p>주간 판단 <time>{s.decision_date}</time> · 가격 확인 {s.latest_price_date}</p></div>
    <div className="hybrid-summary">{(['IN','OUT'] as const).map(direction=><div key={direction} className={`hybrid-callout ${direction.toLowerCase()}`}><span>{direction==='IN'?'상승 전환 감지':'하락 전환 감지'} · {direction}</span><strong>{alerts.filter(r=>r.signal===direction).map(r=>r.ticker).join(' · ')||'신호 없음'}</strong><p>{direction==='IN'?'가격 부스팅':'가격 Extra Trees'} · 완료된 주간 기준</p></div>)}</div>
    <div className="table-wrap" role="region" aria-label="11개 섹터 전환 판단" tabIndex={0}><table><thead><tr><th>섹터</th><th>현재 판단</th><th>검사 방향</th><th>모델 점수</th><th>경보 기준</th></tr></thead><tbody>{s.rows.map(r=><tr key={r.ticker}><th scope="row">{r.ticker} <span>{NAMES[r.ticker]}</span></th><td><span className={`hybrid-badge ${r.signal?.toLowerCase()??''}`}>{r.status!=='READY'?'N/A':r.signal==='NONE'?'신호 없음':r.signal}</span></td><td>{r.direction==='IN'?'상승 전환':r.direction==='OUT'?'하락 전환':'—'}</td><td>{r.score===null?'—':(r.score*100).toFixed(1)}</td><td>{r.threshold===null?'—':(r.threshold*100).toFixed(1)}</td></tr>)}</tbody></table></div>
    <div className="rotation-method"><p>IN은 상승 전환, OUT은 하락 전환 경보입니다. 최근 60일 수익률이 양수이면 OUT, 그 외에는 IN을 검사합니다. ‘신호 없음’은 강세·약세 유지 판정이나 매수·매도 권고가 아닙니다.</p><p>점수는 0~100으로 환산한 모델 출력이며 전환 확률이 아닙니다. 최신화할 때 저장 모델로 계산하되, 신호는 완료된 주간 기준으로 유지합니다.</p></div>
    <details className="rotation-detail"><summary><span>모델과 해석 기준</span><small>IN 부스팅 · OUT Extra Trees</small></summary><div className="rotation-method"><p>11개 ETF의 조정가격에서 수익률·변동성·낙폭·반등폭·이동평균·상대강도를 계산합니다. 검증 구간에서 고정한 방향별 기준을 넘으면 경보를 냅니다. 자동 재학습하지 않습니다.</p><p>경보는 반대 방향의 변동성 장벽에 먼저 도달할 조건을 학습한 결과로 정확한 고점·저점을 뜻하지 않습니다. 과거 결과를 보고 선택한 결합안이며, 독립된 신규 기간 검증은 아직 완료되지 않았습니다.</p></div></details>
  </section>;
}
