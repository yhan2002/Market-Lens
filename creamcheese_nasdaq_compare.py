"""Research-only Nasdaq rule transplant versus saved HMM; no operational writes."""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/creamcheese_nasdaq_compare_20260914"
BREADTH_URL = "https://historyofmarket.com/api/ndx/breadth.json"
MODEL = "hmm3-sticky-nq-independent-a2d-logvol-v1-diag-mc25-k25-m252-w1260-r20-s42-n3"
END = "2026-09-11"
NAMES = ["CC_SHORT_50_PROXY", "CC_COMBINED_PROXY", "HMM", "MA200", "BUY_HOLD"]
COLORS = {"GREEN": 0, "YELLOW": 1, "RED": 2, "BLACK": 3}
EXPOSURE = {"GREEN": 1.0, "YELLOW": 0.5, "RED": 0.0, "BLACK": 0.0}
COST = 0.001  # 10bp per one-way change of 100% exposure.
SPEC = {
    "status": "RESEARCH_ONLY_APPROXIMATE_TRANSPLANT_NOT_ORIGINAL_STRATEGY",
    "benchmark": "NQ=F adjusted continuous price; synthetic unlevered returns",
    "hmm_version": MODEL,
    "no_hmm_refit_or_parameter_search": True,
    "short": "NQ>MA20 and below-MA50 share<=50% -> GREEN; NQ<MA20 and below-MA50 share>=60% -> RED; else YELLOW",
    "short_approximations": "MA50 breadth substitutes unavailable MA20 breadth; unidentified crash branch omitted",
    "long": "BLACK if 252-session drawdown>=30% and below-MA200 share>=80%; RED if drawdown>=20% or (NQ<MA200 and below-MA200>=60%); GREEN if NQ>MA200, MA50 five-session slope>=0 and below-MA200<=50%; else YELLOW",
    "long_approximations": "Entire long rule is a frozen economic proxy, not a recovered original formula; missing NHNL/ADR/distribution-day branches omitted",
    "relaxation": "NQ above MA50 for three consecutive NQ sessions: BLACK->RED, RED->YELLOW before confirmation",
    "confirmation": "Improve after 3 matching candidates; worsening YELLOW after 1, RED after 2, BLACK after 1; missing observation holds state but resets pending streak",
    "fusion": "Long RED/BLACK overrides; both GREEN -> GREEN; otherwise YELLOW",
    "breadth_timing": "Each source observation is available on the next NQ session; no input interpolation/forward-fill",
    "missing_signal_policy": "No new order if an input is missing; previously known target exposure remains unchanged",
    "execution": "Signal at close t, rebalance at close t+1, first earned return close t+1 to close t+2",
    "cost": "10bp per one-way 100% exposure change; entry and final liquidation included; cash yield=0",
    "turns": "Existing directional-change label, fixed +/-0.05 log-price barrier (~5%), five-session cooldown; unfinished final swing excluded",
    "capture": "Strict prior capture: direction-matched exposure change 1..10 NQ sessions before pivot; additional early confirmation: pivot..pivot+5; one alert matches at most one event",
    "ci": "500 paired moving-block resamples, 40 sessions; Wilson 95% intervals for descriptive event recall",
    "pit": "Current-constituent breadth and revised prices; historical vintages not verified. Lag does not cure survivorship/revision bias.",
    "not_included": "Futures margin/roll implementation, financing, realistic cash returns, equal-alert-budget tuning, original site stock-selection strategy",
    "attribution": "History of Market · 美股编年史 (historyofmarket.com), data CC BY 4.0",
}


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def save_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(json_ready(value), handle, ensure_ascii=False, indent=2, allow_nan=False)


def load_inputs() -> tuple[pd.DataFrame, dict]:
    snapshot = json.loads((OUT / "breadth_snapshot.json").read_text(encoding="utf-8"))
    raw = pd.DataFrame(snapshot["payload"]["series"])
    if not {"date", "pct50", "pct200"}.issubset(raw):
        raise ValueError("Required source breadth fields absent")
    raw["date"] = pd.to_datetime(raw["date"], errors="raise")
    if raw.date.duplicated().any() or not raw.date.is_monotonic_increasing:
        raise ValueError("Source dates must be unique and ordered")
    for column in ("pct50", "pct200"):
        raw[column] = pd.to_numeric(raw[column], errors="raise")
        if not raw[column].between(0, 100).all():
            raise ValueError(f"Invalid percentage: {column}")
    db_path = ROOT / "regime_router_publish.db"
    with closing(sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        price = pd.read_sql_query("SELECT date,adj_close AS close FROM daily_prices WHERE ticker='NQ=F' AND date<=? ORDER BY date", db, params=(END,), parse_dates=["date"])
        hmm = pd.read_sql_query("SELECT date,p_bull,p_sideways,p_bear FROM regime_model_probabilities WHERE model_version=? AND date<=? ORDER BY date", db, params=(MODEL, END), parse_dates=["date"])
    if price.empty or hmm.empty or price.date.duplicated().any() or hmm.date.duplicated().any():
        raise ValueError("Empty or duplicate NQ/HMM inputs")
    if not np.isfinite(price.close).all() or not price.close.gt(0).all():
        raise ValueError("Nonpositive/nonfinite NQ price")
    probabilities = hmm[["p_bull", "p_sideways", "p_bear"]]
    if not np.isfinite(probabilities).all().all() or not probabilities.ge(0).all().all() or not probabilities.le(1).all().all() or not np.allclose(probabilities.sum(axis=1), 1):
        raise ValueError("Invalid HMM probabilities")
    frame = price.set_index("date").join(hmm.set_index("date"))
    frame["b50"] = np.nan
    frame["b200"] = np.nan
    frame["breadth_observation_date"] = pd.NaT
    for row in raw.itertuples(index=False):
        position = int(frame.index.searchsorted(row.date, side="right"))
        if position < len(frame):
            frame.iloc[position, frame.columns.get_loc("b50")] = 1 - row.pct50 / 100
            frame.iloc[position, frame.columns.get_loc("b200")] = 1 - row.pct200 / 100
            frame.iloc[position, frame.columns.get_loc("breadth_observation_date")] = row.date
    observed = frame.breadth_observation_date.notna()
    if not (frame.loc[observed, "breadth_observation_date"] < frame.index[observed]).all():
        raise ValueError("Breadth information date is not strictly prior")
    audit = {"source_url": snapshot["url"], "source_sha256": snapshot["sha256"],
             "source_metadata": {k: v for k, v in snapshot["payload"].items() if k != "series"},
             "source_rows": len(raw), "source_start": raw.date.min(), "source_end": raw.date.max(),
             "nq_rows": len(price), "hmm_rows": len(hmm), "breadth_joined_rows": int(observed.sum()),
             "source_csv_or_db_modified": False}
    return frame, audit


def confirm(values: pd.Series) -> pd.Series:
    state, pending, streak = None, None, 0
    output = []
    for candidate in values:
        if pd.isna(candidate):
            pending, streak = None, 0
            output.append(np.nan)
            continue
        if state is None or candidate == state:
            state, pending, streak = candidate, None, 0
        else:
            streak = streak + 1 if pending == candidate else 1
            pending = candidate
            needed = 3 if COLORS[candidate] < COLORS[state] else {"YELLOW": 1, "RED": 2, "BLACK": 1}[candidate]
            if streak >= needed:
                state, pending, streak = candidate, None, 0
        output.append(state)
    return pd.Series(output, index=values.index, dtype=object)


def make_signals(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    from creamcheese_signal_reverse import fuse
    f = frame.copy()
    for window in (20, 50, 200):
        f[f"ma{window}"] = f.close.rolling(window, min_periods=window).mean()
    f["dd252"] = 1 - f.close / f.close.rolling(252, min_periods=252).max()
    slope = f.ma50.pct_change(5, fill_method=None)
    short = pd.Series(np.select([f.close.gt(f.ma20) & f.b50.le(.5),
                                 f.close.lt(f.ma20) & f.b50.ge(.6)], ["GREEN", "RED"], default="YELLOW"), index=f.index)
    short.loc[f[["ma20", "b50"]].isna().any(axis=1)] = np.nan
    long = pd.Series(np.select([
        f.dd252.ge(.30) & f.b200.ge(.80),
        f.dd252.ge(.20) | (f.close.lt(f.ma200) & f.b200.ge(.60)),
        f.close.gt(f.ma200) & slope.ge(0) & f.b200.le(.50),
    ], ["BLACK", "RED", "GREEN"], default="YELLOW"), index=f.index)
    relax = f.close.gt(f.ma50).rolling(3, min_periods=3).sum().eq(3)
    long.loc[relax & long.eq("RED")] = "YELLOW"
    long.loc[relax & long.eq("BLACK")] = "RED"
    long.loc[f[["ma200", "b200", "dd252"]].isna().any(axis=1) | slope.isna()] = np.nan
    signals = pd.DataFrame(index=f.index)
    signals["CC_SHORT_50_PROXY"] = confirm(short)
    confirmed_long = confirm(long)
    combined = fuse(signals.CC_SHORT_50_PROXY.fillna("NA"), confirmed_long.fillna("NA"))
    signals["CC_COMBINED_PROXY"] = pd.Series(combined, index=f.index).replace("NA", np.nan)
    signals.loc[short.isna() | long.isna(), "CC_COMBINED_PROXY"] = np.nan
    probabilities = f[["p_bull", "p_sideways", "p_bear"]]
    hmm_valid = probabilities.notna().all(axis=1)
    signals["HMM"] = pd.Series(index=f.index, dtype=object)
    signals.loc[hmm_valid, "HMM"] = probabilities.loc[hmm_valid].idxmax(axis=1).map({"p_bull": "GREEN", "p_sideways": "YELLOW", "p_bear": "RED"})
    signals["MA200"] = pd.Series(np.where(f.close.ge(f.ma200), "GREEN", "RED"), index=f.index)
    signals.loc[f.ma200.isna(), "MA200"] = np.nan
    signals["BUY_HOLD"] = "GREEN"
    f["cc_short_candidate"] = short
    f["cc_long_candidate"] = long
    return f, signals


def equity_statistics(net: np.ndarray) -> dict:
    wealth = np.cumprod(1 + net)
    peak = np.maximum.accumulate(np.r_[1., wealth])[1:]
    years = len(net) / 252
    sd = np.std(net, ddof=1)
    return {"total_return": wealth[-1] - 1, "cagr": wealth[-1] ** (1 / years) - 1,
            "max_drawdown": np.min(wealth / peak - 1),
            "sharpe_cash0": np.mean(net) / sd * np.sqrt(252) if sd > 0 else None}


def strategy_returns(price: pd.Series, signals: pd.DataFrame, start: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    # Holding the last known target is an order policy, not a fill of price/breadth inputs.
    targets = signals.apply(lambda s: s.map(EXPOSURE)).ffill().fillna(0.)
    executed = targets.shift(1).fillna(0.)
    held = executed.shift(1).fillna(0.)
    turnover = (executed - held).abs()
    gross = held.mul(price.pct_change(fill_method=None), axis=0)
    index = price.index[price.index >= start]
    gross, held, turnover = gross.loc[index], held.loc[index], turnover.loc[index].copy()
    turnover.iloc[0] += held.iloc[0]  # Opening portfolio at the start of this evaluation window.
    turnover.iloc[-1] += executed.loc[index[-1]]  # Liquidate all strategies at the common endpoint.
    net = gross - COST * turnover
    if net.isna().any().any():
        raise ValueError("Return alignment contains missing observations")
    result = {}
    for name in signals:
        result[name] = {**equity_statistics(net[name].to_numpy()),
                        "gross_total_return": float((1 + gross[name]).prod() - 1),
                        "average_exposure": held[name].mean(),
                        "one_way_turnover": turnover[name].sum(),
                        "rebalance_days": int(turnover[name].gt(0).sum()),
                        "n": len(index)}
    return net, result


def turn_capture(price: pd.Series, signals: pd.DataFrame, start: pd.Timestamp) -> tuple[dict, pd.DataFrame]:
    from sector_turning_pooled_shadow import directional_change_labels, _wilson_interval
    labels = directional_change_labels(price, minimum_barrier=.05, maximum_barrier=.05, cooldown_sessions=5)
    events = labels.loc[labels.event.ne(""), ["event", "pivot_position", "confirmation_position", "pivot_date", "confirmation_date"]].copy()
    first = int(price.index.searchsorted(start))
    events = events.loc[events.pivot_position.ge(first + 10)].reset_index(drop=True)
    targets = signals.apply(lambda s: s.map(EXPOSURE)).ffill()
    report, event_rows = {}, []
    for name in signals:
        change = targets[name].diff()
        report[name] = {}
        for kind, comparison in [("TOP_END", change.lt(0)), ("BOTTOM_END", change.gt(0))]:
            candidates = set(np.flatnonzero(comparison.to_numpy() & (price.index >= start)))
            available = candidates.copy()
            eligible = events.loc[events.event.eq(kind)]
            prior = near = 0
            lead = []
            for event in eligible.itertuples(index=False):
                pivot = int(event.pivot_position)
                options = [p for p in available if pivot - 10 <= p < pivot]
                if options:
                    selected, label = max(options), "PRIOR"
                    prior += 1
                    lead.append(pivot - selected)
                else:
                    options = [p for p in available if pivot <= p <= min(pivot + 5, int(event.confirmation_position))]
                    selected, label = (min(options), "NEAR") if options else (None, "MISSED")
                    near += int(selected is not None)
                if selected is not None:
                    available.remove(selected)
                event_rows.append({"model": name, "event": kind, "pivot_date": event.pivot_date,
                                   "confirmation_date": event.confirmation_date,
                                   "matched_alert_date": price.index[selected] if selected is not None else None,
                                   "match": label, "lead_to_pivot": pivot - selected if selected is not None else None})
            count = len(eligible)
            lo, hi = _wilson_interval(prior, count)
            report[name][kind] = {"events": count, "prior_hits": prior,
                "prior_recall": prior / count if count else None,
                "prior_recall_wilson95": [lo, hi], "near_hits": near,
                "prior_or_near_recall": (prior + near) / count if count else None,
                "median_prior_lead_sessions": np.median(lead) if lead else None,
                "alerts": len(candidates), "unmatched_alerts": len(available),
                "matched_share_of_alerts": (prior + near) / len(candidates) if candidates else None}
    return report, pd.DataFrame(event_rows)


def paired_intervals(frame: pd.DataFrame, signals: pd.DataFrame, net: pd.DataFrame, start: pd.Timestamp) -> dict:
    from sklearn.metrics import roc_auc_score
    from validate_hmm import _future_drawdown, _moving_block_indices
    future = pd.Series(_future_drawdown(frame.close.to_numpy(), 20), index=frame.index)
    risk = signals.apply(lambda s: s.map(COLORS)).astype(float)
    risk["HMM"] = frame.p_bear
    valid = future.notna() & risk[NAMES[:3]].notna().all(axis=1) & (frame.index >= start)
    y = future.loc[valid].le(-.05).astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        raise ValueError("Mature drawdown targets do not contain both classes")
    x = risk.loc[valid, NAMES[:3]]
    auc = {name: roc_auc_score(y, x[name]) for name in x}
    rng = np.random.default_rng(42)
    values = {name: {"auc_delta": [], "annual_mean_return_delta": []} for name in NAMES[:2]}
    for _ in range(500):
        # Identical sampled dates for each paired comparison; no random train/test split.
        a = _moving_block_indices(len(y), block_days=40, rng=rng)
        r = _moving_block_indices(len(net), block_days=40, rng=rng)
        for name in values:
            if len(np.unique(y[a])) == 2:
                values[name]["auc_delta"].append(roc_auc_score(y[a], x[name].to_numpy()[a]) - roc_auc_score(y[a], x.HMM.to_numpy()[a]))
            values[name]["annual_mean_return_delta"].append((net[name].to_numpy()[r] - net.HMM.to_numpy()[r]).mean() * 252)
    return {"target": "next 20 NQ sessions minimum close/current close - 1 <= -5%", "n": len(y),
            "target_events": int(y.sum()), "start": x.index.min(), "end": x.index.max(), "auc": auc,
            "vs_hmm": {name: {"auc_delta": auc[name] - auc["HMM"],
                "auc_delta_ci95": np.quantile(v["auc_delta"], [.025, .975]).tolist(),
                "annual_mean_return_delta": (net[name] - net.HMM).mean() * 252,
                "annual_mean_return_delta_ci95": np.quantile(v["annual_mean_return_delta"], [.025, .975]).tolist()}
                for name, v in values.items()}}


def report_text(result: dict) -> str:
    s = result["performance"]
    lines = ["# 크림치즈 논리 나스닥 이식 vs 기존 HMM", "",
             "원본 크림치즈 전략의 성과가 아니라 **고정 근사 규칙의 연구 결과**다. 나스닥 수익률로 규칙을 최적화하지 않았고 HMM도 재학습하지 않았다.", "",
             f"- 기간: {result['start']}~{result['end']} ({result['n_sessions']} NQ 거래일)",
             "- 공통 비중: GREEN 100%, YELLOW 50%, RED/BLACK 0%. 신호 다음 거래일 종가 체결, 그 다음 수익률부터 반영.",
             "- 편도 10bp, 최초 진입·최종 청산 비용 포함. 현금 이자 0%. 연속선물 가격 기반 가상 비중 수익률이며 실제 선물계좌 수익률이 아니다.",
             "- 웹·운영 DB·모델 변경 없음. 아래 숫자는 저장된 현재 HMM 버전의 과거 결과와 비교한 것으로 당시 실제 배포 성과가 아니다.", "",
             "## 1. 동일 기간 운용 성과", "",
             "| 전략 | 누적 순수익률 | 연환산 | 최대낙폭 | Sharpe | 평균 비중 | 비중 조정일 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name in NAMES:
        row = s[name]
        lines.append(f"| {name} | {row['total_return']:.1%} | {row['cagr']:.1%} | {row['max_drawdown']:.1%} | {row['sharpe_cash0']:.2f} | {row['average_exposure']:.1%} | {row['rebalance_days']} |")
    lines += ["", "CC_SHORT_50_PROXY: 단기 근사형. CC_COMBINED_PROXY: 단기·장기 결합 근사형. MA200: 200일선 위 100%, 아래 0%. BUY_HOLD: 단순 보유.",
              "", "## 2. 추세 종료 사전·초기 포착", "",
              "고점·저점은 기존 directional-change 함수를 이용해 약 5%의 반대 움직임으로 확정했다. 미확정 마지막 움직임은 제외. 사전 포착은 극점 **1~10거래일 전**, 추가 초기 포착은 극점 당일~5거래일 후이면서 반전 확정일까지다. 경보는 비중 확대(IN)·축소(OUT) 시점이며 하나의 경보를 여러 전환에 중복 연결하지 않는다.", "",
              "| 전략 | 방향 | 전환 수 | 사전 포착 | 사전+초기 | 사전 포착률 95% 구간 | 사전 리드 중앙값 | 경보 수 | 미연결 경보 |",
              "|---|---|---:|---:|---:|---|---:|---:|---:|"]
    for name in NAMES[:-1]:
        for kind, label in [("TOP_END", "상승 종료/OUT"), ("BOTTOM_END", "하락 종료/IN")]:
            row = result["turns"][name][kind]
            lo, hi = row["prior_recall_wilson95"]
            ci = f"{lo:.1%}~{hi:.1%}" if lo is not None else "N/A"
            lines.append(f"| {name} | {label} | {row['events']} | {row['prior_hits']} | {row['prior_hits'] + row['near_hits']} | {ci} | {row['median_prior_lead_sessions']} | {row['alerts']} | {row['unmatched_alerts']} |")
    lines += ["", "경보 수가 다르므로 포착 건수만으로 우위를 판단하면 안 된다. 미연결 경보는 정해진 매칭 창 안에서 전환에 연결되지 않은 경우이며, 반드시 이후 수익이 나빴다는 뜻은 아니다. Wilson 구간은 소표본·사건 의존성을 완전히 반영하지 못하는 참고 구간이다.", "",
              "## 3. 이후 하락 구분력과 차이의 불확실성", "",
              "AUC 대상은 향후 20거래일 내 현재 종가 대비 -5% 이상 하락이다. 추세 전환 포착률과는 다른 지표다. 최근 20거래일은 라벨 미성숙으로 제외했다.", ""]
    paired = result["paired"]
    lines.append(f"공통 성숙 표본: {paired['start']}~{paired['end']}, {paired['n']}일, 하락 해당일 {paired['target_events']}일(독립 사건 수 아님).")
    for name, value in paired["auc"].items():
        lines.append(f"- {name} AUC: {value:.3f}")
    for name, v in paired["vs_hmm"].items():
        al, ah = v["auc_delta_ci95"]
        rl, rh = v["annual_mean_return_delta_ci95"]
        lines.append(f"- {name} − HMM: ΔAUC {v['auc_delta']:+.3f} (95% {al:+.3f}~{ah:+.3f}); 연환산 일평균 순수익률 차이 {v['annual_mean_return_delta']:+.1%}p (95% {rl:+.1%}p~{rh:+.1%}p). 이 차이는 CAGR 차이가 아니다.")
    lines += ["", "40거래일 이동블록을 두 전략에 동일하게 적용한 500회 bootstrap. 신뢰구간에 0이 포함되면 이 자료에서 우위가 확인됐다고 말하지 않는다.", "",
              "## 4. 근사·시점 한계", "",
              "- 원본 단기 MA20 종목 비율 대신 **MA50 비율**을 썼다. 반응 속도가 달라진다. 원본 crash 지표는 정의가 없어 제외했다.",
              "- 장기 판정은 MA200·MA50 기울기·252일 고점 대비 낙폭·MA200 breadth를 이용한 **새 근사 규칙**이다. 원본 NHNL·ADR·분배일 조건은 포함하지 않았다. BLACK/완화 조건도 검증된 원본 전체 공식이 아니다.",
              "- 연속 확인은 개선 3회, RED 악화 2회 등 기존 역산의 대표 설정을 고정했다. 단기·장기 합성은 장기 위험 우선 규칙이다.",
              "- 참여도는 현 구성종목으로 재계산된 자료: 생존편향·과거 수정 위험. 다음 NQ 거래일에 사용했어도 PIT 문제가 해소되지는 않는다.",
              "- 가격·참여도 입력의 보간/forward-fill 없음. 자료가 없는 날 신규 주문을 내지 않고 기존 목표비중만 유지한다. 동일 경보 예산으로 임계값을 튜닝하지 않았다.",
              "- HMM 매개변수 자체가 이전 연구에서 선택됐으므로 이 비교는 완전히 새로운 미사용 외표본 검증이 아니다.",
              "- [참여도 자료: History of Market · 美股编年史](https://historyofmarket.com/api/ndx/breadth.json), CC BY 4.0.",
              "- [크림치즈 공개 원본](https://github.com/jangdt/CREAMCHEESE-HOUSE-SITE/blob/816dc4439508a45c25d18438db6697b6aca8b66d/index.html).", "",
              "전체 규칙·출처·지문은 spec.json/results.json, 날짜별 신호·비중은 daily_results.json, 전환별 매칭은 turn_matches.json에 기록했다.", ""]
    return "\n".join(lines)


def plot_result(signals: pd.DataFrame, net: pd.DataFrame) -> None:
    from PIL import Image, ImageDraw, ImageFont
    canvas = Image.new("RGB", (1600, 1000), "white")
    draw = ImageDraw.Draw(canvas)
    font_path = "C:/Windows/Fonts/malgun.ttf"
    fonts = {size: ImageFont.truetype(font_path, size) for size in (17, 19, 22, 32)}
    palette = ["#009E73", "#D55E00", "#0072B2", "#777777", "#333333"]
    titles = ["단기 근사형", "단기·장기 결합", "기존 HMM", "200일선", "단순 보유"]
    left, right, top, bottom = 220, 1460, 160, 510
    equity = (1 + net).cumprod() * 100
    lo = np.floor(min(100, equity.min().min()) / 20) * 20
    hi = np.ceil(equity.max().max() / 20) * 20
    times = net.index.asi8
    x = left + (times - times[0]) / (times[-1] - times[0]) * (right - left)
    draw.text((45, 25), "나스닥: 크림치즈 논리 근사형 vs 기존 HMM", font=fonts[32], fill="#172B3A")
    draw.text((45, 78), f"{net.index[0]:%Y-%m-%d} ~ {net.index[-1]:%Y-%m-%d}  |  동일 비중 규칙·체결 지연·비용  |  연구용", font=fonts[19], fill="#536776")
    for i, (title, color) in enumerate(zip(titles, palette)):
        xpos = 220 + i * 245
        draw.line((xpos, 130, xpos + 25, 130), fill=color, width=4)
        draw.text((xpos + 33, 117), title, font=fonts[19], fill=color)
    draw.text((45, 168), "누적 순수익", font=fonts[22], fill="#172B3A")
    draw.text((45, 201), "초기자산 = 100", font=fonts[17], fill="#536776")
    for tick in np.arange(lo, hi + 1, 20):
        y = bottom - (tick - lo) / (hi - lo) * (bottom - top)
        draw.line((left, y, right, y), fill="#E1E7EC", width=1)
        draw.text((left - 15, y), f"{tick:.0f}", anchor="rm", font=fonts[17], fill="#536776")
    for name, color in zip(NAMES, palette):
        y = bottom - (equity[name].to_numpy() - lo) / (hi - lo) * (bottom - top)
        draw.line(list(zip(x, y)), fill=color, width=3)
    for position in np.linspace(0, len(net) - 1, 5).astype(int):
        draw.text((x[position], bottom + 15), f"{net.index[position]:%Y-%m-%d}", anchor="mt", font=fonts[17], fill="#536776")
    draw.text((45, 577), "목표 비중 변화", font=fonts[22], fill="#172B3A")
    draw.text((220, 577), "각 행 위 100% · 가운데 50% · 아래 0%   (신호 다음 거래일 종가에 실행)", font=fonts[19], fill="#536776")
    for i, (name, color, title) in enumerate(zip(NAMES[:3], palette, titles)):
        upper, lower = 632 + i * 90, 697 + i * 90
        draw.text((left - 20, (upper + lower) / 2), title, anchor="rm", font=fonts[19], fill=color)
        for value in (0, .5, 1):
            y = lower - value * (lower - upper)
            draw.line((left, y, right, y), fill="#E1E7EC", width=1)
        target = signals[name].map(EXPOSURE).ffill().reindex(net.index).to_numpy()
        y = lower - target * (lower - upper)
        points = [(x[0], y[0])]
        for k in range(1, len(x)):
            points.extend([(x[k], y[k - 1]), (x[k], y[k])])
        draw.line(points, fill=color, width=2)
    for position in np.linspace(0, len(net) - 1, 5).astype(int):
        draw.text((x[position], 890), f"{net.index[position]:%Y-%m-%d}", anchor="mt", font=fonts[17], fill="#536776")
    draw.text((45, 942), "원본 크림치즈 전략 아님 · 50일 breadth 대체 및 장기 규칙 추정 · 현 구성종목 편향/PIT 미검증", font=fonts[17], fill="#536776")
    draw.text((45, 970), "Breadth: History of Market (historyofmarket.com), CC BY 4.0  |  NQ 연속가격 기반, 실제 선물계좌 성과 아님", font=fonts[17], fill="#536776")
    canvas.save(OUT / "comparison.png")


def run() -> None:
    if (OUT / "results.json").exists() or (OUT / "spec.json").exists():
        raise FileExistsError("Refusing to overwrite a prior experiment/specification")
    save_json(OUT / "spec.json", SPEC)  # Freeze rules before seeing any performance.
    raw, audit = load_inputs()
    frame, signals = make_signals(raw)
    valid = signals[NAMES[:3]].notna().all(axis=1)
    if not valid.any():
        raise ValueError("No common signal observations")
    start = frame.index[int(np.flatnonzero(valid.to_numpy())[0]) + 2]
    net, performance = strategy_returns(frame.close, signals, start)
    turns, matches = turn_capture(frame.close, signals, start)
    paired = paired_intervals(frame, signals, net, start)
    recent_start = max(start, pd.Timestamp("2025-05-23"))
    _, recent = strategy_returns(frame.close, signals, recent_start)
    result = json_ready({"spec": SPEC, "audit": audit, "start": start, "end": frame.index.max(),
              "n_sessions": len(net), "common_signal_days": int(valid.loc[start:].sum()),
              "missing_breadth_days": int(frame.loc[start:, "b50"].isna().sum()),
              "performance": performance, "recent_start": recent_start, "recent_performance": recent,
              "turns": turns, "paired": paired,
              "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    save_json(OUT / "results.json", result)
    daily = frame[["close", "b50", "b200", "breadth_observation_date", "p_bear", "cc_short_candidate", "cc_long_candidate"]].join(signals.add_prefix("signal_")).join(net.add_prefix("net_")).loc[start:]
    # JSON artifacts avoid changing any source CSV/DB; timestamps serialized explicitly.
    (OUT / "daily_results.json").write_text(daily.reset_index().to_json(orient="records", date_format="iso", indent=2), encoding="utf-8")
    (OUT / "turn_matches.json").write_text(matches.to_json(orient="records", date_format="iso", indent=2), encoding="utf-8")
    (OUT / "REPORT.md").write_text(report_text(result), encoding="utf-8")
    try:
        plot_result(signals, net)
    except ModuleNotFoundError as error:
        print(f"Metrics saved; optional chart unavailable: {error}. Use --render with a Pillow runtime.")
    print(json.dumps({"start": result["start"], "end": result["end"], "performance": performance,
                      "turns": turns, "paired": paired}, default=json_ready, ensure_ascii=True, indent=2))


def self_test() -> None:
    from creamcheese_signal_reverse import fuse
    from validate_hmm import _future_drawdown
    dates = pd.bdate_range("2020-01-01", periods=320)
    seq = pd.Series(["GREEN", "RED", np.nan, "RED", "RED", "GREEN", "GREEN", "GREEN"])
    assert confirm(seq).fillna("NA").tolist() == ["GREEN", "GREEN", "NA", "GREEN", "RED", "RED", "RED", "GREEN"]
    assert fuse(["RED", "GREEN"], ["GREEN", "BLACK"]).tolist() == ["YELLOW", "BLACK"]
    price = pd.Series(np.linspace(100., 150., len(dates)), index=dates)
    f = pd.DataFrame({"close": price, "b50": .4, "b200": .4, "p_bull": .8, "p_sideways": .1, "p_bear": .1})
    a, sig = make_signals(f)
    altered = f.copy()
    altered.loc[dates[-20]:, "close"] *= .5
    _, sig2 = make_signals(altered)
    pd.testing.assert_frame_equal(sig.iloc[:-20], sig2.iloc[:-20])
    tiny = pd.Series([100., 100., 110., 121., 133.1], index=dates[:5])
    signals = pd.DataFrame({"TEST": ["RED", "GREEN", "GREEN", "GREEN", "GREEN"]}, index=tiny.index)
    net, _ = strategy_returns(tiny, signals, tiny.index[1])
    assert net.TEST.iloc[0] == 0  # At t1 the t1 signal is not tradable yet.
    assert np.isclose(net.TEST.iloc[1], -COST)  # Rebalance t2 close; no t1->t2 gain.
    assert np.isclose(net.TEST.iloc[2], .1)  # First gain t2->t3.
    assert np.isclose(equity_statistics(np.array([-.1, .1]))["max_drawdown"], -.1)
    assert np.isnan(_future_drawdown(tiny.to_numpy(), 2)[-2:]).all()
    print("PASS: prefix causality, missing-streak reset, long precedence, execution lag, fees, initial drawdown, immature labels")


def render_saved() -> None:
    """Render the completed run without loading source DBs or rerunning any test."""
    daily = pd.read_json(OUT / "daily_results.json").set_index("date")
    daily.index = pd.to_datetime(daily.index)
    signals = daily[[f"signal_{name}" for name in NAMES]].rename(columns=lambda c: c.removeprefix("signal_"))
    net = daily[[f"net_{name}" for name in NAMES]].rename(columns=lambda c: c.removeprefix("net_"))
    plot_result(signals, net)
    print("Saved comparison.png from completed results; no backtest rerun")


def fetch_breadth() -> None:
    path = OUT / "breadth_snapshot.json"
    if path.exists():
        raise FileExistsError("Immutable research snapshot already exists")
    with urlopen(Request(BREADTH_URL, headers={"User-Agent": "MarketLens-Research"}), timeout=30) as response:
        raw = response.read(4_000_001)
    if len(raw) > 4_000_000:
        raise ValueError("Unexpectedly large breadth response")
    payload = json.loads(raw)
    if not isinstance(payload.get("series"), list) or not payload["series"]:
        raise ValueError("Missing breadth series")
    OUT.mkdir(parents=True, exist_ok=True)
    saved = {"url": BREADTH_URL, "retrieved_utc": datetime.now(timezone.utc).isoformat(),
             "sha256": hashlib.sha256(raw).hexdigest(), "payload": payload}
    with path.open("x", encoding="utf-8") as handle:
        json.dump(saved, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"rows": len(payload["series"]), "fields": list(payload["series"][0]),
                      "metadata": {k: v for k, v in payload.items() if k != "series"}}, ensure_ascii=True)[:6000])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if args.fetch:
        fetch_breadth()
    if args.self_test:
        self_test()
    if args.run:
        run()
    if args.render:
        render_saved()


if __name__ == "__main__":
    main()
