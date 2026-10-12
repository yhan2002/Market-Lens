"""Predeclared local experiments A/B/C; no downloads, production writes or tuning."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from roro_data import load_roro_panel
from validate_nasdaq_ratio import hac, holm

OUT = ROOT / "output/analysis/leading_research_20260909"
NQ = ["nq5", "nq20", "vol20", "dd60"]
VIX = NQ + ["vix", "curve"]
MOVE = VIX + ["move", "move5"]


def panel():
    p = load_roro_panel(ROOT / "regime_router_publish.db", as_of="2026-09-08",
                        price_observation_database=ROOT / "outputs/roro_shadow/roro_price_observations.db",
                        breadth_observation_database=ROOT / "outputs/roro_shadow/ndth_breadth_observations.db")
    assert p.index.is_unique and p.index.is_monotonic_increasing
    q = np.log(p.nq)
    x = pd.DataFrame({"nq5": q.diff(5), "nq20": q.diff(20), "vol20": q.diff().rolling(20).std(),
                      "dd60": p.nq / p.nq.rolling(60).max() - 1,
                      "vix": np.log(p.vix), "curve": np.log(p.vix / p.vix3m),
                      "move": np.log(p.move), "move5": np.log(p.move).diff(5),
                      "vix5": np.log(p.vix).diff(5), "credit": np.log(p.hy_oas),
                      "credit5": p.hy_oas.diff(5), "breadth": 1-p.breadth_200dma,
                      "breadth5": -p.breadth_200dma.diff(5)}, index=p.index)
    x["move_downtrend"] = x.move5 * (x.nq20 < 0)
    return p, q, x.replace([np.inf, -np.inf], np.nan)


def labels(q, target):
    h = 5 if target == "direction5" else 10
    if target == "direction5":
        value = q.shift(-h)-q
        y = (value > 0).astype(float).where(value.notna())
    else:
        trough = pd.concat([q.shift(-k) for k in range(1, h+1)], axis=1).min(axis=1, skipna=False)-q
        y = (trough <= np.log(.97)).astype(float).where(trough.notna())
    return y.rename("y"), pd.Series(q.index, index=q.index).shift(-h).rename("end")


def classify(frame, models, start_year, min_train):
    rows = []
    for year in range(start_year, 2027):
        start = pd.Timestamp(year, 1, 1)
        train = frame.loc[frame.end < start]
        test = frame.loc[frame.index.year == year]
        if len(train) < min_train or test.empty or train.y.nunique() < 2:
            continue
        result = test[["y"]].copy()
        result["prior"] = train.y.mean()
        for name, cols in models.items():
            model = make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=2000))
            model.fit(train[cols], train.y)
            result[name] = model.predict_proba(test[cols])[:, 1]
            result[name+"_alert"] = result[name] >= np.quantile(model.predict_proba(train[cols])[:, 1], .8)
        result["train_n"] = len(train)
        rows.append(result)
    return pd.concat(rows)


def metric(y, prob, alert=None):
    y, prob = np.asarray(y), np.asarray(prob)
    result = {"n": len(y), "rate": y.mean(), "brier": np.mean((prob-y)**2),
              "logloss": log_loss(y, prob, labels=[0,1]),
              "auc": roc_auc_score(y, prob) if len(np.unique(y))==2 else np.nan,
              "ap": average_precision_score(y, prob) if y.sum() else np.nan,
              "accuracy": ((prob>=.5)==y).mean(), "majority_accuracy": max(y.mean(),1-y.mean())}
    if alert is not None:
        alert = np.asarray(alert).astype(bool)
        result.update({"alert_rate": alert.mean(), "precision": y[alert].mean() if alert.any() else np.nan,
                       "recall": (y*alert).sum()/y.sum() if y.sum() else np.nan})
    return result


def loss_test(y, base, candidate):
    diff = (np.asarray(base)-y)**2 - (np.asarray(candidate)-y)**2
    b, s, p = hac(diff, np.empty((len(diff), 0)), 20)
    return {"brier_improvement": b[0], "ci_low": b[0]-1.96*s[0], "ci_high": b[0]+1.96*s[0], "p_hac": p[0]}


def market(which):
    p, q, x = panel()
    if which == "A":
        models = {"nq": NQ, "nq_vix": VIX, "move": MOVE, "interaction": MOVE+["move_downtrend"]}
        start, minimum, reference = 2015, 756, "nq_vix"
        periods = {"development": ("2015", "2019"), "historical_confirmation": ("2020", "2024"), "recent_reused": ("2025", "2026")}
        candidates = ["move", "interaction"]
    else:
        models = {"move": MOVE, "credit": MOVE+["credit", "credit5"],
                  "breadth": MOVE+["breadth", "breadth5"], "joint": MOVE+["credit", "credit5", "breadth", "breadth5"]}
        start, minimum, reference = 2025, 252, "move"
        periods = {"recent_exploratory": ("2025", "2026")}
        candidates = ["credit", "breadth", "joint"]
    allcols = list(dict.fromkeys(c for cols in models.values() for c in cols))
    metrics, tests, saved = [], [], []
    for target in ["direction5", "downside10"]:
        y, end = labels(q, target)
        frame = x[allcols].join([y,end]).dropna()
        pred = classify(frame, models, start, minimum)
        saved.append(pred.assign(target=target))
        for period, (a,b) in periods.items():
            sub = pred.loc[a:b]
            for name in ["prior", *models]:
                alert = sub[name+"_alert"] if name != "prior" else None
                metrics.append({"target": target, "period": period, "model": name, **metric(sub.y,sub[name],alert)})
            for candidate in candidates:
                tests.append({"target": target, "period": period, "candidate": candidate, "reference": reference,
                              **loss_test(sub.y.to_numpy(), sub[reference], sub[candidate])})
        for year, sub in pred.groupby(pred.index.year):
            for name in ["prior", *models]:
                metrics.append({"target": target, "period": str(year), "model": name, **metric(sub.y,sub[name])})
    t = pd.DataFrame(tests)
    t["p_holm"] = np.nan
    confirm = t.period.eq("historical_confirmation" if which=="A" else "recent_exploratory")
    t.loc[confirm, "p_holm"] = holm(t.loc[confirm,"p_hac"])
    pd.concat(saved).to_csv(OUT / f"{which}_predictions.csv")
    pd.DataFrame(metrics).to_csv(OUT / f"{which}_metrics.csv", index=False)
    t.to_csv(OUT / f"{which}_tests.csv", index=False)
    p.to_csv(OUT / "source_panel.csv")
    print(f"EXPERIMENT {which}")
    print(pd.DataFrame(metrics).loc[lambda d: d.period.isin(periods)].to_string(index=False))
    print(t.to_string(index=False))


def sector():
    import sqlite3
    from validate_sector_factor_explanation import SECTORS
    _, _, x = panel()
    with sqlite3.connect((ROOT / "regime_router_publish.db").resolve().as_uri()+"?mode=ro",uri=True) as c:
        ticks = [*SECTORS,"SPY"]
        rows = pd.read_sql_query("SELECT date,ticker,adj_close FROM daily_prices WHERE ticker IN ("+",".join("?" for _ in ticks)+") AND date<='2026-09-08' ORDER BY date", c, params=ticks, parse_dates=["date"])
    prices = rows.pivot(index="date",columns="ticker",values="adj_close").dropna()
    q = np.log(prices)
    end = pd.Series(q.index,index=q.index).shift(-20)
    features = ["rel20","rel60","vol20"]
    models = {"momentum": features, "macro": features+["move5","vix5","curve"]}
    saved = []
    for ticker in SECTORS:
        relative = q[ticker]-q.SPY
        f = pd.DataFrame({"rel20":relative.diff(20),"rel60":relative.diff(60),"vol20":q[ticker].diff().rolling(20).std(),
                          "y":relative.shift(-20)-relative,"sector_return":np.expm1(q[ticker].shift(-20)-q[ticker]),
                          "spy_return":np.expm1(q.SPY.shift(-20)-q.SPY),"end":end}).join(x[["move5","vix5","curve"]]).dropna()
        for year in range(2021,2027):
            train = f.loc[f.end < pd.Timestamp(year,1,1)]
            test = f.loc[f.index.year==year]
            if len(train)<504 or test.empty:
                continue
            result = test[["y","sector_return","spy_return"]].copy()
            result["zero"] = 0.
            for name, cols in models.items():
                model = make_pipeline(StandardScaler(),Ridge(alpha=1.))
                model.fit(train[cols], train.y)
                result[name] = model.predict(test[cols])
            saved.append(result.assign(ticker=ticker))
    pred = pd.concat(saved).reset_index()
    pred = pred.rename(columns={pred.columns[0]:"date"})
    gooddates = pred.groupby("date").ticker.nunique().loc[lambda s:s==len(SECTORS)].index
    pred = pred[pred.date.isin(gooddates)].copy()
    pred.to_csv(OUT/"C_predictions.csv",index=False)
    metrics = []
    for ticker, sub in pred.groupby("ticker"):
        mse0 = np.mean(sub.y**2)
        mse1 = np.mean((sub.y-sub.momentum)**2)
        mse2 = np.mean((sub.y-sub.macro)**2)
        metrics.append({"ticker":ticker,"n":len(sub),"momentum_r2_zero":1-mse1/mse0,"macro_r2_zero":1-mse2/mse0,
                        "macro_improvement":1-mse2/mse1,**loss_test(sub.y.to_numpy(),sub.momentum,sub.macro)})
    stats = pd.DataFrame(metrics)
    stats = stats.rename(columns={"brier_improvement": "mse_improvement"})
    stats["p_holm"] = holm(stats.p_hac)
    stats.to_csv(OUT/"C_sector_metrics.csv",index=False)
    ranks = []
    for date, sub in pred.groupby("date"):
        ranks.append({"date":date,**{name:sub[name].corr(sub.y,method="spearman") for name in models}})
    pd.DataFrame(ranks).to_csv(OUT/"C_rank_ic.csv",index=False)
    # Anchor to the original price calendar, not a compressed complete-case calendar.
    anchor = prices.index.searchsorted(pred.date.min())
    rebalance = prices.index[anchor::20]
    holdings = {name:{} for name in models}
    blocks = []
    for date in rebalance:
        sub = pred[pred.date==date].set_index("ticker")
        if len(sub)!=len(SECTORS):
            # Do not bridge missing forecasts into an invented holding period.
            holdings = {name:{} for name in models}
            continue
        for name in models:
            top = sub.nlargest(3,name)
            weights = {t:1/3 for t in top.index}
            turnover = sum(abs(weights.get(t,0)-holdings[name].get(t,0)) for t in set(weights)|set(holdings[name]))
            cost = turnover*.001
            gross = top.sector_return.mean()-top.spy_return.iloc[0]
            blocks.append({"date":date,"model":name,"gross_excess":gross,"turnover":turnover,"cost":cost,"net_excess":gross-cost})
            holdings[name]=weights
    pd.DataFrame(blocks).to_csv(OUT/"C_nonoverlap_blocks.csv",index=False)
    print(stats.to_string(index=False))
    print("MEAN DAILY RANK IC",pd.DataFrame(ranks)[list(models)].mean().to_dict())
    print("NONOVERLAP TOP3 BLOCKS",pd.DataFrame(blocks).groupby("model")[['gross_excess','net_excess','turnover']].mean().to_string())


def self_test():
    q = pd.Series(np.log([100]*10+[96]+[100]*11),index=pd.bdate_range("2020-01-01",periods=22))
    y,end = labels(q,"downside10")
    assert y.iloc[0]==1 and y.iloc[10]==0 and y.iloc[-10:].isna().all()
    assert end.iloc[0]==q.index[10]
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2010-01-01","2016-12-31")
    f = pd.DataFrame({"x":rng.normal(size=len(dates)),"y":rng.integers(0,2,len(dates)),"end":dates+pd.offsets.BDay(10)},index=dates)
    before = classify(f,{"m":["x"]},2015,756)
    f.loc[f.end >= "2015-01-01","y"] = 1-f.loc[f.end >= "2015-01-01","y"]
    after = classify(f,{"m":["x"]},2015,756)
    assert np.allclose(before.loc["2015","m"],after.loc["2015","m"])
    print("PASS: barrier endpoint, incomplete labels, purged train-only classification")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment",choices=["A","B","C"])
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.self_test:self_test()
    elif args.experiment=="C":sector()
    elif args.experiment:market(args.experiment)
    else:parser.error("choose --experiment A/B/C or --self-test")
