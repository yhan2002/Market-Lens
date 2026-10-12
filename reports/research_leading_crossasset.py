"""Exploratory E: fixed cross-asset inputs and one small nonlinear model."""
import os
import sqlite3
os.environ.setdefault('LOKY_MAX_CPU_COUNT','1')
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from research_leading_improvements import ROOT, OUT, VIX, MOVE, panel, labels, metric, loss_test
from validate_nasdaq_ratio import holm


def run():
    _,q,x=panel()
    with sqlite3.connect((ROOT/'regime_router_publish.db').resolve().as_uri()+'?mode=ro',uri=True) as c:
        d=pd.read_sql_query("SELECT date,ticker,adj_close FROM daily_prices WHERE ticker IN ('IEF','RSP','SPY') AND date<='2026-09-08' ORDER BY date",c,parse_dates=['date'])
    p=d.pivot(index='date',columns='ticker',values='adj_close').reindex(q.index)
    cached=pd.read_pickle(ROOT/'output/analysis/nasdaq_ratio_20260909/yahoo_daily.pkl')['Adj Close'].reindex(q.index)
    x['rates5']=np.log(p.IEF).diff(5)
    x['credit_ratio5']=np.log(cached.HYG/cached.LQD).diff(5)
    x['participation5']=np.log(p.RSP/p.SPY).diff(5)
    extended=MOVE+['rates5','credit_ratio5','participation5']
    predictions=[];results=[];tests=[]
    for target in ['direction5','downside10']:
        y,end=labels(q,target)
        f=x[extended].join([y,end]).dropna()
        rows=[]
        for year in range(2015,2027):
            train=f.loc[f.end<pd.Timestamp(year,1,1)].tail(1260)
            test=f.loc[f.index.year==year]
            if len(train)<756 or test.empty:continue
            assert train.end.max()<test.index.min()
            r=test[['y']].copy();r['prior']=train.y.mean()
            for nonlinear in [False,True]:
                for name,cols in [('base',VIX),('crossasset',extended)]:
                    model=HistGradientBoostingClassifier(max_iter=100,learning_rate=.05,max_depth=2,min_samples_leaf=100,l2_regularization=10,early_stopping=False,random_state=42) if nonlinear else make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=2000))
                    model.fit(train[cols],train.y)
                    r[('tree_' if nonlinear else 'linear_')+name]=model.predict_proba(test[cols])[:,1]
            rows.append(r)
        pred=pd.concat(rows);predictions.append(pred.assign(target=target))
        for period,(a,b) in {'development_reused':('2015','2019'),'historical_reused':('2020','2024'),'recent_reused':('2025','2026')}.items():
            sub=pred.loc[a:b]
            for name in ['prior','linear_base','linear_crossasset','tree_base','tree_crossasset']:
                results.append({'target':target,'period':period,'model':name,**metric(sub.y,sub[name])})
            for name in ['linear_crossasset','tree_base','tree_crossasset']:
                tests.append({'target':target,'period':period,'model':name,**loss_test(sub.y.to_numpy(),sub.linear_base,sub[name])})
        for year,sub in pred.groupby(pred.index.year):
            for name in ['prior','linear_base','linear_crossasset','tree_base','tree_crossasset']:
                results.append({'target':target,'period':str(year),'model':name,**metric(sub.y,sub[name])})
    pd.concat(predictions).to_csv(OUT/'E_predictions.csv')
    pd.DataFrame(results).to_csv(OUT/'E_metrics.csv',index=False)
    t=pd.DataFrame(tests);t['p_holm_exploratory']=holm(t.p_hac)
    t.to_csv(OUT/'E_tests.csv',index=False)
    print(pd.DataFrame(results).loc[lambda d:~d.period.str.fullmatch(r'\d{4}')].to_string(index=False));print(t.to_string(index=False))


if __name__=='__main__':run()
