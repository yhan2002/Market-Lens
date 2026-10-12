"""S: fixed rates/oil feature increment using locally cached revised history."""
import json
import hashlib
import numpy as np
import pandas as pd
from research_leading_improvements import ROOT,OUT,panel,labels,classify,metric,loss_test,NQ,VIX
from validate_nasdaq_ratio import holm

SOURCE = ROOT/'outputs/bloomberg_turning_macro_staging_20260902_212424/bloomberg_turning_macro_20260902_212424/macro_history.csv'


def inputs(raw,index):
    p = raw.pivot(index='date',columns='series',values='value').reindex(index)
    assert p.BRENT_SPOT.dropna().gt(0).all()
    f = pd.DataFrame({'real5':p.UST_REAL_10Y.diff(5),'curve5':(p.UST_10Y-p.UST_2Y).diff(5),
                      'be5':p.BREAKEVEN_10Y.diff(5),'brent5':np.log(p.BRENT_SPOT).diff(5)})
    return pd.concat([f.shift(lag).add_suffix(f'_lag{lag}') for lag in [1,5]],axis=1)


def run():
    _,q,x = panel()
    raw = pd.read_csv(SOURCE,parse_dates=['date'])
    assert not raw.duplicated(['date','series']).any()
    assert set(raw.pit_status)=={'CURRENT_REVISED_HISTORY_ONLY'}
    f = inputs(raw,x.index)
    cutoff = x.index[len(x)//2]
    pd.testing.assert_frame_equal(f.loc[:cutoff],inputs(raw[raw.date<=cutoff],x.index[x.index<=cutoff]))
    x = x.join(f)
    models = {'nq':NQ,'nq_vix':VIX,**{f'rates_lag{lag}':VIX+[c for c in f if c.endswith(f'lag{lag}')] for lag in [1,5]}}
    metrics,tests,saved = [],[],[]
    for target in ['direction5','downside10']:
        y,end = labels(q,target)
        frame = x[VIX+list(f.columns)].join([y,end]).dropna()
        pred = classify(frame,models,2015,756)
        saved.append(pred.assign(target=target))
        for period,a,b in [('development_reused','2015','2019'),('primary_reused','2020','2026'),('historical_reused','2020','2024'),('recent_reused','2025','2026')]:
            sub = pred.loc[a:b]
            for name in ['prior',*models]:
                metrics.append({'target':target,'period':period,'model':name,**metric(sub.y,sub[name])})
            if period=='primary_reused':
                for candidate in ['rates_lag1','rates_lag5']:
                    tests.append({'target':target,'candidate':candidate,**loss_test(sub.y.to_numpy(),sub.nq_vix,sub[candidate])})
    t = pd.DataFrame(tests);t['p_holm_4']=holm(t.p_hac)
    t.to_csv(OUT/'S_tests.csv',index=False)
    m=pd.DataFrame(metrics);m.to_csv(OUT/'S_metrics.csv',index=False)
    pd.concat(saved).to_csv(OUT/'S_predictions.csv')
    (OUT/'S_metadata.json').write_text(json.dumps({'source':str(SOURCE),'sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'coverage':raw.groupby('series').date.agg(['min','max','count']).astype(str).to_dict(),
        'pit_status':'CURRENT_REVISED_HISTORY_ONLY; publication dates not supplied; delays are sensitivity only'},indent=2),encoding='utf-8')
    print(m[m.period.eq('primary_reused')].to_string(index=False));print(t.to_string(index=False))


if __name__=='__main__':run()
