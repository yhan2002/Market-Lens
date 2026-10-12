from __future__ import annotations

import argparse
import json
import math
import sqlite3
from datetime import datetime
from pathlib import Path

from crash_guard import _load_prices, build_crash_guard_scores
from regime_router_app import HMM_MODEL_VERSION
from research_drawdown_onset import SHADOW_MODEL_VERSION


def _number(value: object) -> float | None:
    if value is None:
        return None
    number = float(value)
    return round(number, 6) if math.isfinite(number) else None


def build_payload(database: Path) -> dict[str, object]:
    with sqlite3.connect(database) as connection:
        regimes = connection.execute(
            """
            SELECT date, p_bull, p_sideways, p_bear
            FROM regime_model_probabilities
            WHERE model_version=? ORDER BY date
            """,
            (HMM_MODEL_VERSION,),
        ).fetchall()
        hazards = connection.execute(
            """
            SELECT date, score, risk_band
            FROM drawdown_hazard_scores
            WHERE model_version=? ORDER BY date
            """,
            (SHADOW_MODEL_VERSION,),
        ).fetchall()
        price_dates = [
            row[0]
            for row in connection.execute(
                "SELECT date FROM daily_prices WHERE ticker='SPY' ORDER BY date"
            )
        ]

    if not regimes or not price_dates:
        raise ValueError("HTML 조회판에 필요한 HMM 또는 SPY 데이터가 없습니다.")

    prices = _load_prices(database, price_dates[-1])
    guards = build_crash_guard_scores(prices)
    guard_rows = [
        [
            str(index.date()),
            _number(row.score_1d),
            _number(row.score_5d),
            _number(row.score),
            str(row.risk_band),
        ]
        for index, row in guards.iterrows()
    ]

    return {
        "generatedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
        "latestDate": price_dates[-1],
        "prices": price_dates,
        "regimes": [
            [date, _number(bull), _number(sideways), _number(bear)]
            for date, bull, sideways, bear in regimes
        ],
        "guards": guard_rows,
        "hazards": [[date, _number(score), band] for date, score, band in hazards],
    }


HTML = r'''<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Regime Router 조회</title>
<style>
:root{color-scheme:light dark;--bg:#f3f6fa;--panel:#fff;--text:#18212f;--muted:#667085;--line:#d9e0e8;--good:#16794b;--warn:#a05a00;--bad:#b42318;--blue:#2563eb}
@media(prefers-color-scheme:dark){:root{--bg:#10151d;--panel:#18202b;--text:#eef2f6;--muted:#aab4c0;--line:#344050;--good:#5dd39e;--warn:#ffbd66;--bad:#ff7b72;--blue:#7aa2ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:"Malgun Gothic","Apple SD Gothic Neo",sans-serif}.shell{max-width:860px;margin:0 auto;padding:28px 18px 48px}header{margin-bottom:20px}h1{margin:0 0 6px;font-size:28px;font-weight:700}.sub{margin:0;color:var(--muted);font-size:14px}.controls{display:flex;gap:10px;align-items:end;flex-wrap:wrap;margin:20px 0}.field{display:grid;gap:6px}.field label{font-size:13px;color:var(--muted)}input,button{font:inherit;border:1px solid var(--line);border-radius:8px;padding:10px 12px;background:var(--panel);color:var(--text)}button{cursor:pointer;font-weight:600}button.primary{background:var(--blue);border-color:var(--blue);color:#fff}.panel{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:22px}.headline{margin:0 0 18px;font-size:22px}.tone-normal{color:var(--good)}.tone-caution{color:var(--warn)}.tone-alert{color:var(--bad)}.tone-neutral{color:var(--text)}.meta{display:flex;gap:14px;flex-wrap:wrap;color:var(--muted);font-size:13px;margin-bottom:20px}.section{border-top:1px solid var(--line);padding-top:18px;margin-top:18px}.section h2{font-size:15px;margin:0 0 12px}.prob{display:grid;grid-template-columns:48px 1fr 58px;gap:10px;align-items:center;margin:9px 0}.track{height:9px;background:var(--bg);border-radius:99px;overflow:hidden}.bar{height:100%;background:var(--blue)}.prob strong,.scores strong{font-variant-numeric:tabular-nums;text-align:right}.scores{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.score{background:var(--bg);border-radius:9px;padding:12px}.score span{display:block;color:var(--muted);font-size:12px;margin-bottom:4px}.copy{margin:10px 0 0;color:var(--muted);line-height:1.65}.notice{margin-top:18px;padding-top:16px;border-top:1px solid var(--line);color:var(--muted);font-size:13px;line-height:1.6}.error{color:var(--bad);font-weight:600}.foot{margin-top:12px;color:var(--muted);font-size:12px}@media(max-width:560px){.scores{grid-template-columns:1fr}.panel{padding:17px}.headline{font-size:19px}}
</style>
</head>
<body>
<main class="shell">
<header><h1>Regime Router</h1><p class="sub">시장 국면 · Crash Guard · Drawdown Onset 조회 전용</p></header>
<div class="controls">
  <div class="field"><label for="date">확인 날짜</label><input id="date" type="date"></div>
  <button id="show" class="primary" type="button">상태 확인</button>
  <button id="latest" type="button">최신일</button>
</div>
<section class="panel" aria-live="polite">
  <h2 id="headline" class="headline">불러오는 중…</h2>
  <div id="content"></div>
</section>
<p id="foot" class="foot"></p>
</main>
<script>
const DATA=__DATA__;
const stateLabels={BULL:"상승장",SIDEWAYS:"횡보장",BEAR:"하락장"};
const onsetCopy={ALERT_SHADOW:["조기경보: 위험 축소 검토","다음 거래일에 위험 축소 여부를 검토하세요. 자동 매도 신호는 아닙니다.","alert"],NORMAL_SHADOW:["조기경보 없음","현재 조기경보 조건에 해당하지 않습니다. HMM 국면과 함께 참고하세요.","normal"],DORMANT:["Onset 적용 밖","SPY가 252일 고점 대비 -3% 밖에 있어 Onset 적용 구간이 아닙니다. Crash Guard는 계속 감시합니다.","caution"],WARMUP:["조기경보 준비 중","점수를 만들 과거 데이터가 아직 부족합니다. HMM 국면만 참고하세요.","neutral"]};
const guardCopy={CRASH_WATCH_SHADOW:["급락 주의","1일 급변 또는 5일 취약성이 높은 구간입니다. 다음 거래일 위험 축소나 헤지 여부를 검토하세요. 자동 주문 신호는 아닙니다.","alert"],VULNERABLE_SHADOW:["급락 취약성 증가","급락 취약성이 평소보다 높습니다. 익스포저와 대응 계획을 점검하세요. 자동 주문 신호는 아닙니다.","caution"],NORMAL_SHADOW:["상시 감시 정상","현재 급락 취약성 순위가 경계 구간보다 낮습니다. 갑작스러운 충격이 없다는 보장은 아닙니다.","normal"],WARMUP:["상시 감시 준비 중","Crash Guard 계산에 필요한 과거 데이터가 아직 부족합니다.","neutral"]};
function prior(rows,date){let lo=0,hi=rows.length-1,found=null;while(lo<=hi){const mid=(lo+hi)>>1;if(rows[mid][0]<=date){found=rows[mid];lo=mid+1}else hi=mid-1}return found}
function pct(v){return `${(v*100).toFixed(1)}%`}
function score(v){return v==null?"계산 전":v.toFixed(3)}
function esc(v){return String(v).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]))}
function prob(label,value){return `<div class="prob"><span>${label}</span><div class="track"><div class="bar" style="width:${Math.max(0,Math.min(100,value*100))}%"></div></div><strong>${pct(value)}</strong></div>`}
function render(){
  const requested=document.getElementById("date").value;
  const headline=document.getElementById("headline"),content=document.getElementById("content");
  if(!requested){headline.textContent="날짜를 선택하세요";headline.className="headline tone-neutral";content.innerHTML="";return}
  const r=prior(DATA.regimes,requested),g=prior(DATA.guards,requested),h=prior(DATA.hazards,requested),p=prior(DATA.prices.map(d=>[d]),requested);
  if(!r){headline.textContent="조회 가능한 HMM 결과가 없습니다";headline.className="headline tone-alert";content.innerHTML=`<p class="error">${esc(requested)} 이전 결과가 없습니다.</p>`;return}
  const probs={BULL:r[1],SIDEWAYS:r[2],BEAR:r[3]},state=Object.keys(probs).reduce((a,b)=>probs[a]>=probs[b]?a:b);
  const onset=h?(onsetCopy[h[2]]||[h[2],"알 수 없는 경보 상태입니다.","neutral"]):["조기경보 미계산","Drawdown Onset 조기경보가 아직 계산되지 않았습니다.","neutral"];
  const guard=g?(guardCopy[g[4]]||[g[4],"알 수 없는 Crash Guard 상태입니다.","neutral"]):["계산 결과 없음","Crash Guard 결과가 없습니다.","neutral"];
  let title,tone;
  if(g&&g[4]==="CRASH_WATCH_SHADOW"&&h&&h[2]==="ALERT_SHADOW"){title=`${stateLabels[state]} · 복합 경보: 위험 축소 검토`;tone="alert"}
  else if(g&&g[4]==="CRASH_WATCH_SHADOW"){title=`${stateLabels[state]} · 급락 주의`;tone="alert"}
  else if(h&&h[2]==="ALERT_SHADOW"){title=`${stateLabels[state]} · ${onset[0]}`;tone="alert"}
  else if(g&&g[4]==="VULNERABLE_SHADOW"){title=`${stateLabels[state]} · 급락 취약성 증가`;tone="caution"}
  else{title=`${stateLabels[state]} · ${onset[0]}`;tone=onset[2]}
  headline.textContent=title;headline.className=`headline tone-${tone}`;
  content.innerHTML=`
    <div class="meta"><span>요청일 ${esc(requested)}</span><span>HMM ${r[0]}</span><span>시장가격 ${p?p[0]:"없음"}</span></div>
    <div class="section"><h2>시장 국면</h2>${prob("상승",r[1])}${prob("횡보",r[2])}${prob("하락",r[3])}</div>
    <div class="section"><h2>Crash Guard · ${esc(guard[0])}</h2><div class="scores"><div class="score"><span>1일 급변</span><strong>${g?score(g[1]):"계산 전"}</strong></div><div class="score"><span>5일 취약성</span><strong>${g?score(g[2]):"계산 전"}</strong></div><div class="score"><span>종합</span><strong>${g?score(g[3]):"계산 전"}</strong></div></div><p class="copy">계산일 ${g?g[0]:"없음"}<br>${esc(guard[1])}</p></div>
    <div class="section"><h2>Drawdown Onset · ${esc(onset[0])}</h2><div class="scores"><div class="score"><span>Hazard 점수</span><strong>${h?score(h[1]):"계산 전"}</strong></div><div class="score"><span>경보 기준</span><strong>0.720</strong></div><div class="score"><span>계산일</span><strong>${h?h[0]:"없음"}</strong></div></div><p class="copy">${esc(onset[1])}</p></div>
    <div class="notice">Shadow 점수는 하락 확률이 아니라 과거 데이터 내 위험 순위입니다. 자동 주문 또는 매도 신호가 아니며 판단 보조용입니다.</div>`;
}
const dateInput=document.getElementById("date");dateInput.min=DATA.regimes[0][0];dateInput.max=DATA.latestDate;dateInput.value=DATA.latestDate;
document.getElementById("show").addEventListener("click",render);document.getElementById("latest").addEventListener("click",()=>{dateInput.value=DATA.latestDate;render()});dateInput.addEventListener("change",render);
document.getElementById("foot").textContent=`포함 데이터 최신일 ${DATA.latestDate} · 생성 ${DATA.generatedAt.replace("T"," ")}`;
render();
</script>
</body>
</html>'''


def write_viewer(database: Path, output: Path) -> None:
    payload = build_payload(database)
    document = HTML.replace(
        "__DATA__", json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )
    forbidden = ("daily_holdings", "inferred_actions", "portfolio_nav", "universe_config")
    if any(value in document for value in forbidden):
        raise ValueError("민감 데이터 식별자가 HTML에 포함되었습니다.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a standalone offline status viewer")
    parser.add_argument("database", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    write_viewer(args.database, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
