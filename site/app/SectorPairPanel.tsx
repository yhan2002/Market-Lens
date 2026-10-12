import { useMemo, useState } from "react";
import { pairRotationRows, rankedSectorPairs, sectorPairPrices, widestSectorPair, SECTOR_PAIR_NAMES, type SectorPrices, type PairPriceRow } from "../lib/it-internals";
import ItSpreadChart from "./ItSpreadChart";

const pct = (v: number) => `${v >= 0 ? "+" : ""}${(v * 100).toFixed(2)}%p`;

export default function SectorPairPanel({ prices: legacyPrices, sectors, selectedDate }: { prices?: PairPriceRow[]; sectors?: SectorPrices; selectedDate: string }) {
  const latest = (sectors ? sectors.rows.at(-1)?.[0] : legacyPrices?.at(-1)?.[0]) ?? "";
  const [requested, setRequested] = useState(selectedDate || latest);
  const automaticPair = useMemo(() => sectors ? widestSectorPair(sectors, requested) : null, [sectors, requested]);
  const [selectedPair, setSelectedPair] = useState<[string, string] | null>(null);
  const [first, second] = selectedPair ?? [automaticPair?.first ?? "XLC", automaticPair?.second ?? "XLE"];
  const prices = useMemo(() => sectors ? sectorPairPrices(sectors, first, second) : legacyPrices, [sectors, first, second, legacyPrices]);
  const rows = useMemo(() => pairRotationRows(prices ?? [], requested), [prices, requested]);
  const row = rows.at(-1), prior = rows.at(-2);
  const firstDate = prices?.[60]?.[0] ?? latest;
  const ranked = useMemo(() => sectors ? rankedSectorPairs(sectors, requested) : [], [sectors, requested]);
  const STATES = ["동반 시장 상회", "동반 시장 하회", `${first} 우위`, `${second} 우위`];
  return <section className="sector-history-panel" aria-label="11개 섹터 페어 일간 비교">
    <div className="sector-history-head">
      <div><span>SECTOR PAIR · DAILY CHECK</span><h3>{SECTOR_PAIR_NAMES[first]} vs {SECTOR_PAIR_NAMES[second]}</h3><small>{first}와 {second}의 SPY 대비 상대수익률 · 조정종가 기준</small></div>
      {latest && <label>판단일<input type="date" min={firstDate} max={latest} value={requested} onChange={(e) => setRequested(e.target.value)} /></label>}
    </div>
    {sectors ? <>
      <div className="it-spread-stats">
        <label>섹터 A <select value={first} onChange={(e) => setSelectedPair([e.target.value, second])}>{sectors.tickers.map((ticker) => <option key={ticker} value={ticker} disabled={ticker === second}>{ticker} · {SECTOR_PAIR_NAMES[ticker]}</option>)}</select></label>
        <label>섹터 B <select value={second} onChange={(e) => setSelectedPair([first, e.target.value])}>{sectors.tickers.map((ticker) => <option key={ticker} value={ticker} disabled={ticker === first}>{ticker} · {SECTOR_PAIR_NAMES[ticker]}</option>)}</select></label>
        <button type="button" disabled={!automaticPair || !selectedPair} onClick={() => setSelectedPair(null)}>최대 스프레드 자동 선택</button>
      </div>
      <p className="it-spread-note">{selectedPair ? "직접 선택" : automaticPair ? `자동 선택 · ${automaticPair.date} 기준 20거래일 수익률 최고(A) − 최저(B) · 격차 ${pct(automaticPair.spread)}` : "20거래일 이력 부족 · 기본 XLC/XLE 표시"}</p>
      <p className="it-spread-note">|Z-score| 상위 2개 · {ranked[0]?.date ?? "기준일 데이터 없음"} · 유효 {ranked.length}/55 페어 · 직전 63거래일 기준 · 역방향 중복 제외, 동일 섹터 중복 허용</p>
      <div className="it-spread-stats">{ranked.slice(0, 2).map((pair, i) => <button type="button" key={`${pair.first}-${pair.second}`} onClick={() => setSelectedPair([pair.first, pair.second])}>
        {i + 1}. {pair.first} − {pair.second} · Z {pair.z >= 0 ? "+" : ""}{pair.z.toFixed(2)}σ · 스프레드 {pct(pair.spread / 100)} · 상세 보기
      </button>)}</div>
      {!ranked.length && <p className="it-spread-note">이력이 부족하거나 표준편차가 0인 페어는 순위에서 제외됩니다.</p>}
    </> : <p className="it-spread-note">11개 섹터 가격 갱신 전에는 기존 XLC/XLE 비교를 표시합니다.</p>}
    {!row ? <p>이 날짜의 비교 데이터가 없습니다. 기존 종가 자료로 일간 비교를 생성한 뒤 표시됩니다.</p> : <>
      <p className="it-spread-note">기준일 {row[0]}{requested > latest ? ` · 요청일 ${requested} 종가 미확보` : ""}</p>
      <div className="table-wrap"><table>
        <thead><tr><th>SECTOR</th><th>PROXY</th><th>20D vs SPY</th><th>60D vs SPY</th><th>20D LEADER</th><th>STATE</th><th>STATE CHANGE</th></tr></thead>
        <tbody>{[first, second].map((ticker, i) => {
          const name = SECTOR_PAIR_NAMES[ticker];
          const leading = i === 0 ? row[7] > 0 : row[7] < 0;
          const r20 = i === 0 ? row[3] : row[4], r60 = i === 0 ? row[5] : row[6];
          return <tr key={name} className={leading ? "leader" : ""}>
            <td><b>{name}</b></td><td><b>{ticker}</b></td>
            <td className={r20 >= 0 ? "positive" : "negative"}>{pct(r20)}</td>
            <td className={r60 >= 0 ? "positive" : "negative"}>{pct(r60)}</td>
            <td><b className={`model-status ${leading ? "status-select" : "status-watch"}`}>{row[7] === 0 ? "TIE" : leading ? "LEAD" : "TRAIL"}</b></td>
            <td><b className="rrg-align align-early">{STATES[row[1]]}</b><small className="sector-fit-score">{row[1] === row[2] ? "20D · 60D CONFIRMED" : "20D · 60D DIVERGENCE"}</small></td>
            <td>{!prior ? "N/A" : prior[1] === row[1] ? "—" : `${STATES[prior[1]]} → ${STATES[row[1]]}`}</td>
          </tr>;
        })}</tbody>
      </table></div>
      <ItSpreadChart rows={rows} date={row[0]} prices={prices} first={SECTOR_PAIR_NAMES[first]} second={SECTOR_PAIR_NAMES[second]} firstTicker={first} secondTicker={second} />
    </>}
  </section>;
}
