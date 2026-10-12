"""V: weekly, delayed NTM outlook changes; economic observations never used."""
import json
import numpy as np
import pandas as pd
from research_leading_improvements import OUT,panel,labels,classify,metric,VIX
from validate_nasdaq_ratio import hac,holm


def outlook(raw):
    e=raw[raw.data_group.eq('EARNINGS')&raw.target_period.eq('1TY')]
    p=e.pivot(index='observation_date',columns=['security','field_id'],values='value').sort_index()
    assert p.notna().all().all() and p.gt(0).all().all()
    assert p.index.to_series().diff().dropna().eq(pd.Timedelta(days=7)).all()
    changes=np.log(p).diff(4)
    return pd.DataFrame({'eps4':changes[('NDX Index','BE008')],'sales4':changes[('NDX Index','BE001')],
                         'relative_eps4':changes[('NDX Index','BE008')]-changes[('SPX Index','BE008')],
                         'relative_sales4':changes[('NDX Index','BE001')]-changes[('SPX Index','BE001')]})


def run():
    source=OUT/'earnings_econ_full_received/history.csv'
    raw=pd.read_csv(source,parse_dates=['observation_date'])
    f=outlook(raw)
    cutoff=f.index[len(f)//2]
    pd.testing.assert_frame_equal(f.loc[:cutoff],outlook(raw[raw.observation_date<=cutoff]))
    _,q,x=panel()
    endpoints=pd.Series(x.index,index=x.index).groupby(x.index.to_period('W-FRI')).max()
    allcols=[]
    for lag in [1,2]:
        part=f.add_suffix(f'_lag{lag}').copy()
        part['source_date']=part.index
        part['decision_date']=(part.index.to_period('W-FRI')+lag).map(endpoints)
        part=part.dropna(subset=['decision_date']).set_index('decision_date')
        assert part.index.is_unique and (part.source_date<part.index).all()
        cols=[c for c in part if c!='source_date'];allcols+=cols
        x=x.join(part[cols])
    models={'nq_vix':VIX,**{f'outlook_lag{lag}':VIX+[c for c in allcols if c.endswith(f'lag{lag}')] for lag in [1,2]}}
    metrics,tests,saved=[],[],[]
    for target in ['direction5','downside10']:
        y,end=labels(q,target)
        frame=x[VIX+allcols].join([y,end]).dropna()
        pred=classify(frame,models,2015,156)
        saved.append(pred.assign(target=target))
        for period,a,b in [('development_reused','2015','2019'),('primary_reused','2020','2026'),('historical_reused','2020','2024'),('recent_reused','2025','2026')]:
            sub=pred.loc[a:b]
            for name in ['prior',*models]:
                metrics.append({'target':target,'period':period,'model':name,**metric(sub.y,sub[name])})
            if period=='primary_reused':
                for name in ['outlook_lag1','outlook_lag2']:
                    diff=(sub.y-sub.nq_vix)**2-(sub.y-sub[name])**2
                    beta,se,p=hac(diff.to_numpy(),np.empty((len(diff),0)),4)
                    tests.append({'target':target,'candidate':name,'n_weeks':len(sub),'brier_improvement':beta[0],
                                  'ci_low':beta[0]-1.96*se[0],'ci_high':beta[0]+1.96*se[0],'p_hac4':p[0]})
    t=pd.DataFrame(tests);t['p_holm4']=holm(t.p_hac4)
    t.to_csv(OUT/'V_tests.csv',index=False)
    m=pd.DataFrame(metrics);m.to_csv(OUT/'V_metrics.csv',index=False)
    pd.concat(saved).to_csv(OUT/'V_predictions.csv')
    (OUT/'V_metadata.json').write_text(json.dumps({'source':str(source),'features':allcols,
        'source_weeks':len(f),'pit_status':'NOT_VERIFIED','economic_series_used':False,
        'interpretation':'Changes in rolling NTM index outlooks, not pure revision breadth'},indent=2),encoding='utf-8')
    print(m[m.period.eq('primary_reused')].to_string(index=False));print(t.to_string(index=False))


if __name__=='__main__':run()
