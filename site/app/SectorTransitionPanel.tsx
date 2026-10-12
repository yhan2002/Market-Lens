import type { SectorTransitionSnapshot } from '../lib/sector-transition';

const direction = (value: 'ROTATION_IN' | 'ROTATION_OUT' | null) =>
  value === 'ROTATION_IN' ? '하락 종료' : value === 'ROTATION_OUT' ? '상승 종료' : 'N/A';
const level = (score: number | null) => score === null ? 'N/A' : score >= 80 ? '높음' : score >= 60 ? '관찰' : '보통';

export default function SectorTransitionPanel({snapshot: s}: {snapshot: SectorTransitionSnapshot | null}) {
  if (!s) return null;
  const ranked = [...s.rows].sort((a,b) => (b.score ?? -1) - (a.score ?? -1));
  return <section className="transition-card" aria-label="섹터 추세 전환 가능성">
    <div className="transition-head"><div><span className="section-label">TREND EXHAUSTION</span><h3>추세 전환 가능성</h3></div><time dateTime={s.as_of}>{s.as_of}</time></div>
    <p className="transition-intro">향후 10거래일 안에 현재 상대추세가 끝날 가능성을 각 섹터의 과거 대비 점수로 표시합니다.</p>
    <div className="transition-leaders">{ranked.slice(0,3).map(row => <article key={row.ticker} className={row.score !== null && row.score >= s.high_threshold ? 'is-high' : ''}>
      <div><b>{row.ticker}</b><span>{row.sector}</span></div><strong>{row.score === null ? 'N/A' : row.score.toFixed(0)}</strong>
      <small>{direction(row.direction)} · {level(row.score)}</small>
    </article>)}</div>
    <details className="rotation-detail"><summary><span>전체 11개 섹터</span><small>과거 대비 0–100점</small></summary>
      <div className="table-wrap transition-table"><table><thead><tr><th>섹터</th><th>현재 추세</th><th>예상 전환</th><th>점수</th><th>상태</th></tr></thead><tbody>
        {ranked.map(row => <tr key={row.ticker}><th><b>{row.ticker}</b><span>{row.sector}</span></th><td>{row.state === 'UP_ACTIVE' ? '상승' : '하락'}</td><td>{direction(row.direction)}</td><td><b>{row.score === null ? 'N/A' : row.score.toFixed(1)}</b></td><td>{level(row.score)}</td></tr>)}
      </tbody></table></div>
    </details>
    <p className="transition-note">80점 이상은 우선 점검 구간입니다. 점수는 실제 확률이 아니라 해당 섹터의 과거 신호 대비 백분위입니다.</p>
  </section>;
}
