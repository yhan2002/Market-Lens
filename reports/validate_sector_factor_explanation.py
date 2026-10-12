"""Sector-level contemporaneous Market Lens attribution using local data only."""
import argparse
import json
import sqlite3

import numpy as np
import pandas as pd

from validate_market_lens_factors import ROOT, FACTORS, load, shapley_r2, self_test
from validate_nasdaq_ratio import hac, holm

OUT = ROOT / "output/analysis/sector_factor_explanation_20260909"
SECTORS = {"XLK": "정보기술", "XLC": "커뮤니케이션", "XLY": "경기소비재", "XLP": "필수소비재",
           "XLF": "금융", "XLI": "산업재", "XLE": "에너지", "XLB": "소재",
           "XLV": "헬스케어", "XLU": "유틸리티", "XLRE": "부동산"}
FIVE = ["equity_volatility_score", "credit_score", "rates_volatility_score",
        "breadth_risk_percentile", "fx_carry_score"]


def r_squared(y, x):
    a = np.column_stack([np.ones(len(y)), x])
    residual = y - a @ np.linalg.lstsq(a, y, rcond=None)[0]
    return 1 - residual @ residual / ((y - y.mean()) ** 2).sum()


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    scores, _ = load()
    with sqlite3.connect((ROOT / "regime_router_publish.db").resolve().as_uri() + "?mode=ro", uri=True) as c:
        tickers = [*SECTORS, "SPY"]
        rows = pd.read_sql_query(
            "SELECT date,ticker,adj_close FROM daily_prices WHERE ticker IN (" + ",".join("?" for _ in tickers) + ") AND date<=? ORDER BY date",
            c, params=[*tickers, "2026-09-08"], parse_dates=["date"])
    prices = rows.pivot(index="date", columns="ticker", values="adj_close").loc[scores.index.min():scores.index.max()]
    assert prices.index.equals(scores.index), "Trading calendars differ; review before differencing"
    assert prices.notna().all().all() and (prices > 0).all().all()
    prices.to_csv(OUT / "sector_adjusted_prices.csv")
    models, factors = [], []
    for h in [1, 5, 20]:
        # Keep missing sessions during differencing; then use the same sample for every sector/model.
        frame = scores[FACTORS].diff(h).join(np.log(prices).diff(h)).dropna()
        for name, columns in [("five", FIVE), ("seven", FACTORS)]:
            z = (frame[columns] - frame[columns].mean()) / frame[columns].std()
            for ticker, label in SECTORS.items():
                y = frame[ticker].to_numpy()
                shares, r2 = shapley_r2(z, y)
                beta, se, p = hac(y, z, max(20, 2*h))
                spy_r2 = r_squared(y, frame[["SPY"]])
                combined_r2 = r_squared(y, np.column_stack([frame.SPY, z]))
                model = {"model": name, "horizon": h, "ticker": ticker, "sector": label,
                         "n": len(frame), "start": str(frame.index.min().date()), "end": str(frame.index.max().date()),
                         "r2": r2, "adjusted_r2": 1-(1-r2)*(len(frame)-1)/(len(frame)-len(columns)-1),
                         "unexplained": 1-r2, "spy_r2": spy_r2,
                         "incremental_r2_above_spy": combined_r2-spy_r2}
                for period, sub in [("early", frame.loc[:"2026-03-31"]), ("late", frame.loc["2026-04-01":])]:
                    model[period + "_r2"] = r_squared(sub[ticker].to_numpy(), sub[columns])
                assert np.isclose(shares.sum()+model["unexplained"], 1)
                models.append(model)
                for i, col in enumerate(columns):
                    factors.append({"model": name, "horizon": h, "ticker": ticker, "sector": label,
                                    "factor": col, "r2_contribution": shares[i], "share_of_explained": shares[i]/r2,
                                    "beta_1sd": beta[i+1], "ci_low": beta[i+1]-1.96*se[i+1],
                                    "ci_high": beta[i+1]+1.96*se[i+1], "p_hac": p[i+1]})
    m, f = pd.DataFrame(models), pd.DataFrame(factors)
    f["p_holm"] = holm(f.p_hac)
    m.to_csv(OUT / "model_statistics.csv", index=False)
    f.to_csv(OUT / "factor_statistics.csv", index=False)
    table = f[(f.model == "five") & (f.horizon == 5)].pivot(index="ticker", columns="factor", values="r2_contribution")[FIVE]
    table = table.join(m[(m.model == "five") & (m.horizon == 5)].set_index("ticker")[["r2", "unexplained", "adjusted_r2", "incremental_r2_above_spy"]]).sort_values("r2", ascending=False)
    table.to_csv(OUT / "five_day_summary.csv")
    meta = {"source": "r19 Market Lens score stores and regime_router_publish.db adjusted sector closes",
            "scope": "11 sector ETFs; contemporaneous 1/5/20-session log returns vs same-horizon score changes",
            "score_sessions": len(scores), "missing_handling": "no fill; common complete sample after differencing",
            "attribution": "exact LMG/Shapley R2; shares of total variance, not causal or portfolio weights",
            "inference": "HAC Bartlett maxlags=max(20,2*h); Holm over 396 coefficient tests",
            "limitations": "230-session reconstructed history; in-sample; no future direction or trading backtest"}
    (OUT / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 섹터별 Market Lens 팩터 설명력", "", "5개 팩터의 같은 기간 점수 변화로 섹터 ETF의 조정종가 로그 수익률을 설명했다. 아래 숫자는 전체 수익률 분산에 대한 Shapley 배분(%)이며 원인이나 운용 비중이 아니다.", "",
             "- 원자료 기간: 2025-10-08~2026-09-08, 230거래일. 5일 분석은 결측·차분 제외 후 222개 공통 관측치(2025-10-15~2026-09-08).",
             "- 팩터: VIX·기간구조, HY OAS 신용, MOVE, 시장 폭, 엔 캐리. NQ 추세·금/NQ 상대강도를 더한 7팩터는 별도 민감도 분석이다.",
             "- 모든 섹터·두 모형은 각 기간별 동일 표본을 사용했다. 조정 종가는 배당·분할을 반영하며 결측을 채우지 않았다.", "",
             "| 섹터 | VIX | 신용 | MOVE | 시장 폭 | 엔 | 합계 R² | 미설명 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for t, row in table.iterrows():
        lines.append("| " + SECTORS[t] + " (" + t + ") | " + " | ".join(f"{100*row[k]:.1f}" for k in FIVE + ["r2", "unexplained"]) + " |")
    lines += ["", "반올림으로 합계가 다를 수 있다. 예를 들어 VIX 20%는 모형이 설명한 부분의 20%가 아니라 해당 섹터 전체 수익률 분산의 20%를 배분했다는 뜻이다.", "",
              "## 기간·모형 민감도", "", "| 섹터 | 1일 5팩터 R² | 5일 5팩터 R² | 20일 5팩터 R² | 5일 7팩터 R² | 5일 SPY 단독 R² | SPY에 5팩터 추가 ΔR² |", "|---|---:|---:|---:|---:|---:|---:|"]
    for ticker in table.index:
        a = m[(m.ticker == ticker) & (m.model == "five")].set_index("horizon")
        b = m[(m.ticker == ticker) & (m.model == "seven") & (m.horizon == 5)].iloc[0]
        values = [a.loc[h, "r2"] for h in [1,5,20]] + [b.r2, a.loc[5,"spy_r2"], a.loc[5,"incremental_r2_above_spy"]]
        lines.append("| " + SECTORS[ticker] + " | " + " | ".join(f"{v*100:.1f}" for v in values) + " |")
    lines += ["", "SPY 비교는 같은 시점의 시장 수익률을 통제하는 진단이다. ΔR²는 분산 설명력의 퍼센트포인트 증가이며, 미래에 얻을 예측 개선이 아니다. 섹터 ETF는 S&P 시장의 구성요소이므로 SPY 단독 설명력도 구조적으로 높을 수 있다.", "",
              "## 해석", "", "- 5일 기준 정보기술·경기소비재·커뮤니케이션의 설명력이 가장 높고 VIX 블록의 배분이 크다. 부동산·에너지·산업재는 MOVE의 배분이 가장 크다.",
              "- 396개 계수 검정 보정 후 5팩터·5일 모형에서 유의한 음의 계수는 VIX: XLK·XLC·XLY, MOVE: XLK·XLC·XLI·XLV·XLU·XLRE이다. 해당 점수 상승이 같은 기간 섹터 수익률 하락과 연관됐다. 에너지는 MOVE 배분이 크지만 독립 계수의 보정 후 유의성이 남지는 않았다.",
              "- 정보기술의 SPY 단독 R²는 75.4%로 5팩터의 51.4%보다 높으며, SPY에 팩터를 추가해도 증가는 1.25%p다. 부동산·에너지는 추가 설명력이 각각 23.32%p·16.45%p로 상대적으로 크다.",
              "- 기간별 안정성에 한계가 있다. 에너지의 5팩터 R²는 2026-04-01 이전 10.7%에서 이후 36.5%, 소재는 9.7%에서 38.2%로 바뀌었다. 각 구간 재적합 결과이며 표본 밖 성과가 아니다.", "",
              "## 검정과 한계", "", "계수의 1표준편차 효과, 95% HAC 구간, 396개 검정의 Holm 보정 p값은 factor_statistics.csv에 저장했다. Shapley 비중이 양수라고 그 팩터의 독립적 계수가 유의하다는 뜻은 아니다. 비중 자체의 신뢰구간은 계산하지 않았다.",
              "", "현재 표본에 맞춘 동행 설명력이다. 사후 재구성된 약 11개월 자료이며, 중첩 5·20일 관측치가 독립 표본은 아니다. 표본 밖 선행 예측·인과관계·최적 배분을 입증하지 않는다. 원자재·유가·금리 수준 등 Market Lens에 없는 원천을 추가한 모형도 아니다. 따라서 낮은 설명력은 해당 섹터를 설명하는 주요 요인이 이 팩터 집합 밖에 있을 가능성을 뜻할 뿐, 그 원인까지 특정하지 않는다.",
              "", "## 재현", "", "```powershell", "& '.\\.venv\\Scripts\\python.exe' reports/validate_sector_factor_explanation.py --self-test", "& '.\\.venv\\Scripts\\python.exe' reports/validate_sector_factor_explanation.py", "```", "", "추가 다운로드 없이 로컬 DB를 읽기 전용으로 사용한다. 기존 Shapley·HAC 계산을 재사용했으며 날짜 정렬, 공통 표본, 양수 가격, 비중 합계 검사를 수행했다. 전체 수치는 model_statistics.csv 및 factor_statistics.csv, 핵심 표는 five_day_summary.csv에 있다."]
    (OUT / "report_ko.md").write_text("\n".join(lines), encoding="utf-8")
    print((table*100).round(2).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    self_test() if args.self_test else run()
