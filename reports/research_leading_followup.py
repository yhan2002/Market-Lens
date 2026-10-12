"""Explicitly exploratory follow-up D; fixed settings, local data only."""
import argparse
import sqlite3

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from research_leading_improvements import ROOT, OUT, panel, labels, metric, loss_test
from validate_nasdaq_ratio import holm


def risk():
    _, q, x = panel()
    y,end=labels(q,"downside10")
    probability=pd.Series(2*norm.cdf(np.log(.97)/(x.vol20*np.sqrt(10))), index=x.index).clip(1e-6,1-1e-6)
    x["barrier_logit"]=logit(probability)
    models={"calibrated":["barrier_logit","nq20"],"calibrated_macro":["barrier_logit","nq20","curve","move5"]}
    f=x[["barrier_logit","nq20","curve","move5"]].join([y,end]).assign(analytic=probability).dropna()
    saved=[]
    for start in pd.date_range("2015-01-01","2026-07-01",freq="QS"):
        train=f.loc[f.end<start].tail(756)
        test=f.loc[(f.index>=start)&(f.index<start+pd.offsets.QuarterBegin(startingMonth=1))]
        if len(train)<504 or test.empty:continue
        result=test[["y","analytic"]].copy()
        result["recent_prior"]=train.y.mean()
        for name,cols in models.items():
            model=make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=2000))
            model.fit(train[cols],train.y)
            result[name]=model.predict_proba(test[cols])[:,1]
        saved.append(result)
    pred=pd.concat(saved)
    a=pd.read_csv(OUT/"A_predictions.csv",index_col=0,parse_dates=True)
    a=a.loc[a.target.eq("downside10"),["nq","nq_vix"]]
    pred=pred.join(a,how="inner").dropna()
    assert pred.index.is_unique
    pred.to_csv(OUT/"D_risk_predictions.csv")
    results,tests=[],[]
    for period,(start,stop) in {"development_reused":("2015","2019"),"historical_reused":("2020","2024"),"recent_reused":("2025","2026")}.items():
        sub=pred.loc[start:stop]
        for name in ["nq","nq_vix","recent_prior","analytic",*models]:
            results.append({"period":period,"model":name,**metric(sub.y,sub[name])})
        for reference in ["nq","nq_vix"]:
            for candidate in ["analytic",*models]:
                tests.append({"period":period,"candidate":candidate,"reference":reference,**loss_test(sub.y.to_numpy(),sub[reference],sub[candidate])})
    t=pd.DataFrame(tests);t["p_holm_exploratory"]=holm(t.p_hac)
    pd.DataFrame(results).to_csv(OUT/"D_risk_metrics.csv",index=False)
    t.to_csv(OUT/"D_risk_tests.csv",index=False)
    print(pd.DataFrame(results).to_string(index=False));print(t.to_string(index=False))


def portfolio_blocks(prices, signals, name, lag, offset=0):
    """20-session close-to-close blocks; drifting weights and entry/exit costs."""
    # Both latency scenarios start together, once yesterday's forecast exists.
    start=prices.index.searchsorted(signals.index.min())+1+offset
    previous={},None
    records=[]
    for position in range(start,len(prices)-20,20):
        if position<lag:continue
        entry,exitday=prices.index[position],prices.index[position+20]
        signalday=prices.index[position-lag]
        common_signals=signals.reindex([entry,prices.index[position-1]])
        if common_signals.isna().any().any():
            if records and previous[1] is not None:
                records[-1]["cost"]+=.001;records[-1]["net_excess"]-=.001
            previous={},None
            continue
        top=signals.loc[signalday].nlargest(3).index
        new={t:1/3 for t in top}
        old,oldentry=previous
        if oldentry is not None:
            amounts={t:w*prices.loc[entry,t]/prices.loc[oldentry,t] for t,w in old.items()}
            total=sum(amounts.values());old={t:v/total for t,v in amounts.items()}
        turnover=sum(abs(new.get(t,0)-old.get(t,0)) for t in set(new)|set(old))
        cost=.001*turnover
        gross=float((prices.loc[exitday,top]/prices.loc[entry,top]-1).mean())
        benchmark=float(prices.loc[exitday,"SPY"]/prices.loc[entry,"SPY"]-1)
        records.append({"date":entry,"exit":exitday,"signal_date":signalday,"model":name,"lag":lag,
                        "gross_return":gross,"spy_return":benchmark,"gross_excess":gross-benchmark,
                        "turnover":turnover,"cost":cost,"net_excess":gross-benchmark-cost})
        previous=new,entry
    if records and previous[1] is not None:
        records[-1]["cost"]+=.001;records[-1]["net_excess"]-=.001
    return pd.DataFrame(records)


def sector():
    from validate_sector_factor_explanation import SECTORS
    with sqlite3.connect((ROOT/"regime_router_publish.db").resolve().as_uri()+"?mode=ro",uri=True) as c:
        ticks=[*SECTORS,"SPY"]
        rows=pd.read_sql_query("SELECT date,ticker,adj_close FROM daily_prices WHERE ticker IN ("+",".join("?" for _ in ticks)+") AND date<='2026-09-08' ORDER BY date",c,params=ticks,parse_dates=["date"])
    prices=rows.pivot(index="date",columns="ticker",values="adj_close").dropna()
    relative=np.log(prices[list(SECTORS)]).sub(np.log(prices.SPY),axis=0)
    c=pd.read_csv(OUT/"C_predictions.csv",parse_dates=["date"])
    methods={name:c.pivot(index="date",columns="ticker",values=name) for name in ["momentum","macro"]}
    methods["raw60"]=relative.diff(60).reindex(methods["momentum"].index)
    truth=c.pivot(index="date",columns="ticker",values="y")
    ranks=pd.DataFrame({name:signal.corrwith(truth,axis=1,method="spearman") for name,signal in methods.items()})
    ranks.to_csv(OUT/"D_sector_rank_ic.csv")
    blocks=pd.concat([portfolio_blocks(prices,signal,name,lag) for name,signal in methods.items() for lag in [0,1]])
    blocks.to_csv(OUT/"D_sector_blocks.csv",index=False)
    summary=blocks.groupby(["model","lag"]).agg(n=("net_excess","size"),gross_excess=("gross_excess","mean"),net_excess=("net_excess","mean"),cost=("cost","mean"))
    summary.to_csv(OUT/"D_sector_summary.csv")
    print("RANK IC",ranks.mean().to_dict());print(summary.to_string())


def self_test():
    dates=pd.bdate_range("2020-01-01",periods=61)
    prices=pd.DataFrame({"A":100.,"B":100.,"C":100.,"SPY":100.},index=dates)
    s=pd.DataFrame({"A":3.,"B":2.,"C":1.},index=dates)
    b=portfolio_blocks(prices,s,"check",0)
    assert len(b)==2 and np.isclose(b.cost.sum(),.002)
    assert np.isclose(b.net_excess.sum(),-.002)
    assert (b.exit.shift().dropna().to_numpy()==b.date.iloc[1:].to_numpy()).all()
    b=portfolio_blocks(prices,s,"check",1)
    assert (b.signal_date<b.date).all()
    assert b.date.equals(portfolio_blocks(prices,s,"check",0).date)
    print("PASS: independent calendar blocks, cost entry/exit, lagged signal")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part",choices=["risk","sector"])
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if args.self_test:self_test()
    elif args.part=="risk":risk()
    elif args.part=="sector":sector()
    else:parser.error("choose --part or --self-test")
