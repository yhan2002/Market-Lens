import { itSpreadHistory, pairRelationship, pairSpreadPercentile, type ItInternalsRow, type PairPriceRow } from "../lib/it-internals";

export default function ItSpreadChart({ rows, date, prices = [], first = "HW", second = "SW", firstTicker = "SOXX", secondTicker = "IGV" }: {
  rows: ItInternalsRow[]; date: string; prices?: PairPriceRow[];
  first?: string; second?: string; firstTicker?: string; secondTicker?: string;
}) {
  const history = itSpreadHistory(rows, date, 504);
  const last = history.at(-1);
  if (!last) return null;
  const relation = pairRelationship(prices, date);
  const longRelation = pairRelationship(prices, date, 1_260);
  const longSpread = pairSpreadPercentile(rows, date);
  const fmt = (value: number | null, unit = "%p") => value === null ? "N/A" : `${value >= 0 ? "+" : ""}${value.toFixed(2)}${unit}`;
  const charts = [false, true].map((standardized) => {
    const points = standardized ? history.slice(-63) : history;
    const bounds = standardized ? points.flatMap((p) => p.z === null ? [] : [p.z]) : points.flatMap((p) => [p.spread, ...(p.mean !== null && p.sd !== null ? [p.mean - 2 * p.sd, p.mean + 2 * p.sd] : [])]);
    const extent = Math.max(standardized ? 3 : 1, ...bounds.map(Math.abs)) * 1.12;
    const x = (i: number) => 64 + i / Math.max(1, points.length - 1) * 640;
    const y = (v: number) => 132 - v / extent * 102;
    const line = (values: (number | null)[]) => {
      let connected = false;
      return values.map((value, i) => {
        if (value === null) { connected = false; return ""; }
        const command = connected ? "L" : "M"; connected = true;
        return `${command}${x(i)},${y(value)}`;
      }).join(" ");
    };
    const bands = (sigma: number) => points.slice(1).map((p, i) => {
      const a = points[i];
      if (a.mean === null || a.sd === null || p.mean === null || p.sd === null) return null;
      return <path key={i} d={`M${x(i)},${y(a.mean + sigma * a.sd)} L${x(i + 1)},${y(p.mean + sigma * p.sd)} L${x(i + 1)},${y(p.mean - sigma * p.sd)} L${x(i)},${y(a.mean - sigma * a.sd)} Z`} fill="#42a5d9" opacity={sigma === 2 ? 0.10 : 0.17} />;
    });
    return <div key={String(standardized)} className="it-spread-plot">
      <h4>{standardized ? "과거 3개월 대비 위치 (Z-score)" : `${first} − ${second} 20일 수익률 스프레드 · 최근 2년`}</h4>
      <svg viewBox="0 0 760 280" role="img" aria-label={standardized ? "직전 63거래일 평균 대비 표준편차 거리" : "스프레드와 직전 63거래일 평균 및 표준편차 구간"}>
        {!standardized && <>{bands(2)}{bands(1)}</>}
        {(standardized ? [-2, -1, 0, 1, 2] : [-extent * 0.8, 0, extent * 0.8]).map((tick) => <g key={tick}>
          <line x1="64" x2="704" y1={y(tick)} y2={y(tick)} stroke="currentColor" opacity={tick === 0 ? 0.55 : 0.2} strokeDasharray="5 5" />
          <text x="56" y={y(tick) + 4} textAnchor="end" fill="currentColor" fontSize="12">{tick.toFixed(1)}{standardized ? "σ" : "%p"}</text>
        </g>)}
        {!standardized && <path d={line(points.map((p) => p.mean))} fill="none" stroke="#b98820" strokeWidth="1.8" strokeDasharray="5 4" />}
        <path d={line(points.map((p) => standardized ? p.z : p.spread))} fill="none" stroke={standardized ? "#b56ade" : "#2199d4"} strokeWidth="2.5" />
        {points.map((p, i) => { const value = standardized ? p.z : p.spread; return value === null ? null : <circle key={p.date} cx={x(i)} cy={y(value)} r={standardized || i === points.length - 1 ? 3 : 1} fill={standardized ? "#b56ade" : "#2199d4"}><title>{p.date} · {fmt(value, standardized ? "σ" : "%p")} · 평균 {fmt(p.mean)} · 표준편차 {p.sd?.toFixed(2) ?? "N/A"}%p</title></circle>; })}
        {[0, Math.floor((points.length - 1) / 2), points.length - 1].filter((v, i, a) => a.indexOf(v) === i).map((i) => <text key={i} x={x(i)} y="264" textAnchor="middle" fill="currentColor" fontSize="12">{points[i].date}</text>)}
      </svg>
      <p>{standardized ? `최근 63거래일 · 양수: 평소보다 ${first} 쪽 · 음수: 평소보다 ${second} 쪽` : "최근 504거래일 · 파랑: 스프레드 · 금색 점선: 3개월 평균 · 음영: ±1σ / ±2σ"}</p>
    </div>;
  });
  return <div className="it-spread-history">
    <div className="it-spread-stats"><span>스프레드 <b>{fmt(last.spread)}</b></span><span>직전 3개월 평균 <b>{fmt(last.mean)}</b></span><span>표준편차 <b>{last.sd === null ? "N/A" : `${last.sd.toFixed(2)}%p`}</b></span><span>평균 대비 <b>{fmt(last.z, "σ")}</b></span></div>
    <div className="it-spread-stats pair-relationship">
      <span>63일 상관계수 r <b>{fmt(relation.correlation, "")}</b></span>
      <span>회귀계수 β ({firstTicker} / {secondTicker}) <b>{fmt(relation.beta, "")}</b></span>
      <span>설명력 R² <b>{relation.r2 === null ? "N/A" : `${(relation.r2 * 100).toFixed(1)}%`}</b></span>
      <span>표본 <b>{relation.n}/63일</b></span>
    </div>
    <div className="it-spread-stats pair-relationship long-relationship">
      <span>5년 스프레드 백분위 <b>{longSpread.percentile === null ? "N/A" : `${(longSpread.percentile * 100).toFixed(1)}%`}</b></span>
      <span>5년 상관계수 r <b>{fmt(longRelation.correlation, "")}</b></span>
      <span>5년 회귀계수 β ({firstTicker} / {secondTicker}) <b>{fmt(longRelation.beta, "")}</b></span>
      <span>5년 설명력 R² <b>{longRelation.r2 === null ? "N/A" : `${(longRelation.r2 * 100).toFixed(1)}%`}</b></span>
      <span>표본 <b>{longRelation.n}/1,260일</b></span>
    </div>
    <div className="it-spread-grid">{charts}</div>
    <p className="it-spread-note">{first}는 {firstTicker}, {second}는 {secondTicker} 기준. 스프레드 = {firstTicker} 20일 수익률 − {secondTicker} 20일 수익률. 각 날짜 직전 63거래일의 표본 표준편차로 계산하며, 이력이 부족하거나 표준편차가 0이면 Z-score는 N/A입니다. ±σ 구간은 과거 분산을 보여주며 추세 전환 확률을 뜻하지 않습니다.</p>
    <p className="it-spread-note">상관·회귀는 판단일까지 최근 63개 및 5년(1,260개) 공통 거래일의 일간 수정주가 수익률 기준입니다. r은 동행 정도(−1~+1), β는 {secondTicker}가 1% 움직일 때 {firstTicker}의 평균적인 동반 움직임(%), R²는 과거 동행 설명력입니다. 5년 스프레드 백분위는 현재 20일 스프레드보다 낮거나 같은 관측치의 비중입니다. 절편을 포함한 단순 회귀이며 인과관계·자금 이동·향후 전환을 보장하지 않습니다. 표본이 부족하거나 변동이 없으면 N/A입니다.</p>
  </div>;
}
