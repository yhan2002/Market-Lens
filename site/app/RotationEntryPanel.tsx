import { ROTATION_SECTORS, type RotationEntry } from '../lib/rotation-entry';

const NAMES: Record<string, string> = {XLB:'소재',XLC:'커뮤니케이션',XLE:'에너지',XLF:'금융',XLI:'산업재',XLK:'기술',XLP:'필수소비재',XLRE:'부동산',XLU:'유틸리티',XLV:'헬스케어',XLY:'경기소비재'};
const signed = (n: number) => `${n > 0 ? '+' : ''}${n}`;

export default function RotationEntryPanel({snapshot: s, date}: {snapshot: RotationEntry | null; date: string}) {
  if (!s || s.as_of > date) return <section className="rotation-card rotation-empty"><h3>조회 가능한 순환 기록이 없습니다</h3><p>선택한 기준일 이후의 경보는 표시하지 않습니다.</p></section>;
  const cooldown = s.days_since_last_alert !== null && s.days_since_last_alert < 10;
  const historical = s.phase === 'HISTORICAL_REPLAY';
  const label = historical ? (s.alert ? '과거 재현 경보' : '과거 순위 재현') : s.phase === 'BASELINE' ? '전진 관측 대기' : s.phase === 'EXPIRED_STATE_ONLY' ? '경보 중단 · 설정 검토 필요' : s.alert ? '새 진입 경보' : cooldown ? '경보 간격 대기' : '새 진입 경보 없음';
  const ordered = [...ROTATION_SECTORS].sort((a,b) => s.ranks[a]-s.ranks[b]);
  return <section className="rotation-card" aria-label="섹터 순환 진입 경보">
    <div className="rotation-topline"><span className={`rotation-status${s.alert ? ' is-alert' : ''}`}>{label}</span><p>미국 종가 <time dateTime={s.as_of}>{s.as_of}</time>{s.as_of < date && <span className="rotation-stale">선택일보다 이전 기록</span>}</p></div>
    <div className="rotation-layout">
      <div className="rotation-leadership"><h3>현재 주도 섹터</h3><p>최근 20거래일 SPY 상대수익률 상위 3개</p>
        <ol className="leader-cards">{ordered.slice(0,3).map(t => <li key={t}>
          <div className="leader-heading"><span className="leader-position">0{s.ranks[t]}</span><span className={`rank-move ${s.rank_change5[t]>0 ? 'up' : s.rank_change5[t]<0 ? 'down' : ''}`}>{s.rank_change5[t] === 0 ? '변동 없음' : `${s.rank_change5[t]>0 ? '↑' : '↓'} ${Math.abs(s.rank_change5[t])}단계`}</span></div>
          <strong>{t}</strong><span className="sector-name">{NAMES[t]}</span><small>5거래일 전 {s.ranks[t]+s.rank_change5[t]}위</small>
        </li>)}</ol>
      </div>
      <aside className={`rotation-nominee${s.alert ? ' is-alert' : ''}`} aria-label="다음 진입 감시 후보">
        <span className="section-label">{historical ? (s.alert ? '과거 재현 경보 대상' : '과거 재현 감시 후보') : s.alert ? '진입 경보 대상' : '참고 · 진입 감시 후보'}</span><div className="nominee-identity"><h3>{s.nominated}</h3><span>{NAMES[s.nominated]}</span></div>
        <div className="nominee-score"><strong>{signed(s.score)}</strong><span>5거래일 순위 개선<small>경보 기준 +{s.threshold}단계</small></span></div>
        <dl><div><dt>현재 순위</dt><dd>{s.ranks[s.nominated]}위</dd></div><div><dt>현재 4위 비교 대상</dt><dd>{s.nearest4}</dd></div></dl>
        <p>{historical ? '해당 날짜까지의 가격만 사용해 같은 규칙을 재현한 연구용 결과입니다.' : s.phase === 'BASELINE' ? '시작 기준 상태입니다. 신규 경보로 집계하지 않습니다.' : s.phase === 'EXPIRED_STATE_ONLY' ? '설정 검토 전까지 상태만 기록합니다.' : cooldown ? `다음 경보까지 최소 ${10-s.days_since_last_alert!}거래일 남았습니다.` : s.alert ? '향후 10거래일 내 주도군 진입을 관찰합니다.' : '감시 후보 선정은 경보 발생을 의미하지 않습니다.'}</p>
      </aside>
    </div>
    <div className="rotation-method" aria-label="순위 읽는 법">
      <p><strong>순위 기준</strong> 최근 20거래일 수익률이 높은 섹터부터 1~11위입니다. 배당·분할 등을 반영한 조정종가를 사용하며, 팩터 점수나 거래량은 반영하지 않습니다.</p>
      <p><strong>순위 개선</strong> 5거래일 전 7위 → 현재 4위라면 ‘+3단계’입니다. 수익률 +3%라는 뜻은 아닙니다.</p>
      <p><strong>경보 기준 +{s.threshold}</strong> 5거래일 동안 순위가 {s.threshold}계단 이상 오른 경우입니다(예: 10위 → 4위). 전진 관측 중인 해의 상위 3개 밖 감시 후보에 적용하며, 직전 경보가 있다면 최소 10거래일이 지나야 합니다.</p>
    </div>
    <details className="rotation-detail"><summary><span>전체 11개 섹터 순위</span><small>현재 순위 · 5거래일 변화</small></summary>
      <div className="table-wrap rotation-rank-table" tabIndex={0} role="region" aria-label="전체 섹터 순위표"><table><thead><tr><th scope="col">섹터</th><th scope="col">현재</th><th scope="col">5거래일 전</th><th scope="col">순위 개선</th><th scope="col">상태</th></tr></thead><tbody>
        {ordered.map(t => <tr key={t} className={s.ranks[t]<=3 ? 'leader' : ''}><th scope="row"><b>{t}</b><span>{NAMES[t]}</span></th><td>{s.ranks[t]}위</td><td>{s.ranks[t]+s.rank_change5[t]}위</td><td>{signed(s.rank_change5[t])}</td><td>{s.ranks[t]<=3 ? '현재 주도' : t===s.nominated ? (s.alert ? '진입 경보' : '감시 후보') : '관찰'}</td></tr>)}
      </tbody></table></div>
    </details>
    <details className="rotation-detail"><summary><span>경보 기준과 검증 결과</span><small>고정 규칙 · 전진 관측</small></summary><div className="rotation-method">
      <p>상대수익률은 각 섹터의 20거래일 로그수익률에서 SPY(S&P 500 ETF)의 같은 기간 로그수익률을 뺀 값입니다. 모든 섹터에 같은 SPY 값을 빼므로 순위는 섹터 자체 수익률 순위와 같습니다.</p>
      <p>주도군 밖에서 5거래일 순위 개선이 가장 큰 섹터를 감시합니다. 경보는 향후 10거래일 내 상위 3개 진입 후 5거래일 유지를 관찰하며, 경보 간 최소 간격은 10거래일입니다.</p>
      <p>+6은 과거 보정 구간에서 경보 빈도를 연 6회 안팎으로 맞추도록 선택한 2026년 고정 기준입니다. 수익률이나 예측 확률을 뜻하지 않으며, 향후 경보 횟수와 적중을 보장하지 않습니다.</p>
      <p>과거 22회 경보 중 지목 섹터 진입 15회(68.2%), 현재 4위 비교 11회(50.0%). 차이의 주 검증 구간은 0~35.5%p로 우위 확정 단계는 아닙니다. 2026-09-08 이후 고정 규칙으로 전진 관측합니다. 수익률·매매 비중 예측과 이탈 경보는 포함하지 않습니다.</p>
    </div></details>
  </section>;
}
