"""Exploratory F: reuse existing sector option transforms, next-session inputs."""
import json
import sqlite3
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from research_leading_improvements import ROOT, OUT, loss_test
from research_leading_followup import portfolio_blocks
from validate_sector_factor_explanation import SECTORS
from validate_nasdaq_ratio import holm
from sector_turning_leading_confirmation_experiment import load_leading_features, HISTORY, ETF_TO_SECTORS

BASE=['rel20','rel60','vol20']
OPTION=['option_iv_rel_z252','option_term_rel_z252','option_oi_rel_change_5']


def run():
    option,metadata=load_leading_features(HISTORY)
    with sqlite3.connect((ROOT/'regime_router_publish.db').resolve().as_uri()+'?mode=ro',uri=True) as c:
        ticks=[*SECTORS,'SPY']
        raw=pd.read_sql_query("SELECT date,ticker,adj_close FROM daily_prices WHERE ticker IN ("+','.join('?' for _ in ticks)+") AND date<='2026-09-08' ORDER BY date",c,params=ticks,parse_dates=['date'])
    prices=raw.pivot(index='date',columns='ticker',values='adj_close').dropna()
    q=np.log(prices)
    end=pd.Series(q.index,index=q.index).shift(-20)
    panels=[]
    for ticker in SECTORS:
        o=option.loc[option.sector.eq(ETF_TO_SECTORS[ticker][0]),['date',*OPTION]].set_index('date')
        assert o.index.is_unique
        o['source_date']=o.index
        o=o.reindex(prices.index).shift(1)
        relative=q[ticker]-q.SPY
        f=pd.DataFrame({'rel20':relative.diff(20),'rel60':relative.diff(60),'vol20':q[ticker].diff().rolling(20).std(),
                        'y':relative.shift(-20)-relative,'end':end}).join(o).dropna()
        assert (f.source_date<f.index).all()
        panels.append(f.assign(ticker=ticker))
    frame=pd.concat(panels).reset_index(names='date')
    complete=frame.groupby('date').ticker.nunique().loc[lambda s:s==11].index
    frame=frame[frame.date.isin(complete)].copy()
    dummies=pd.get_dummies(frame.ticker,prefix='sector',dtype=float)
    frame=pd.concat([frame,dummies],axis=1)
    dummycols=list(dummies.columns)
    saved=[]
    for year in range(2021,2027):
        train=frame[frame.end<pd.Timestamp(year,1,1)]
        test=frame[frame.date.dt.year.eq(year)].copy()
        if test.empty or train.groupby('ticker').size().min()<504:continue
        assert train.end.max()<test.date.min()
        for option_on in [False,True]:
            cols=BASE+(OPTION if option_on else [])
            name='options' if option_on else 'base'
            pool=make_pipeline(StandardScaler(),Ridge(alpha=1.))
            pool.fit(train[cols+dummycols],train.y)
            test['pooled_'+name]=pool.predict(test[cols+dummycols])
            for ticker,sub in train.groupby('ticker'):
                model=make_pipeline(StandardScaler(),Ridge(alpha=1.))
                model.fit(sub[cols],sub.y)
                mask=test.ticker.eq(ticker)
                test.loc[mask,'individual_'+name]=model.predict(test.loc[mask,cols])
        saved.append(test)
    pred=pd.concat(saved).sort_values(['date','ticker'])
    methods=['individual_base','individual_options','pooled_base','pooled_options']
    assert pred[methods].notna().all().all()
    pred[['date','ticker','source_date','y',*methods]].to_csv(OUT/'F_predictions.csv',index=False)
    stats=[]
    for ticker,sub in pred.groupby('ticker'):
        for kind in ['individual','pooled']:
            base=sub[kind+'_base'];candidate=sub[kind+'_options']
            d=loss_test(sub.y.to_numpy(),base,candidate)
            d['mse_improvement']=d.pop('brier_improvement')
            stats.append({'ticker':ticker,'kind':kind,'n':len(sub),'relative_mse_improvement':d['mse_improvement']/np.mean((sub.y-base)**2),**d})
    stats=pd.DataFrame(stats);stats['p_holm']=holm(stats.p_hac)
    stats.to_csv(OUT/'F_tests.csv',index=False)
    truth=pred.pivot(index='date',columns='ticker',values='y')
    signals={name:pred.pivot(index='date',columns='ticker',values=name) for name in methods}
    ranks=pd.DataFrame({name:s.corrwith(truth,axis=1,method='spearman') for name,s in signals.items()})
    ranks.to_csv(OUT/'F_rank_ic.csv')
    blocks=pd.concat([portfolio_blocks(prices,s,name,lag) for name,s in signals.items() for lag in [0,1]])
    blocks.to_csv(OUT/'F_blocks.csv',index=False)
    summary=blocks.groupby(['model','lag']).agg(n=('net_excess','size'),gross_excess=('gross_excess','mean'),net_excess=('net_excess','mean'),cost=('cost','mean'))
    summary.to_csv(OUT/'F_summary.csv')
    metadata.update({'pit_status':'PIT_NOT_VERIFIED','extra_session_delay':1,'shared_sample_n':int(pred.date.nunique()),'start':str(pred.date.min()),'end':str(pred.date.max())})
    (OUT/'F_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(stats.to_string(index=False));print('RANK IC',ranks.mean().to_dict());print(summary.to_string())


if __name__=='__main__':run()
