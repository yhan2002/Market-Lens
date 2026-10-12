"""Q: cached MSCI proxy increment with explicit cross-market input delay."""
import json
import numpy as np
import pandas as pd
from research_leading_improvements import ROOT, OUT, panel, labels, classify, metric, loss_test, NQ, VIX
from research_msci_rotation import CACHE, TICKERS, features
from validate_nasdaq_ratio import holm


def run():
    _,q,x = panel()
    proxy = pd.read_pickle(CACHE)['Adj Close'][TICKERS]
    spy = pd.read_pickle(ROOT/'output/analysis/nasdaq_ratio_20260909/yahoo_daily.pkl')['Adj Close']['SPY']
    f = features(proxy.join(spy.rename('SPY')))
    cols = list(f.columns)
    f['source_date'] = f.index
    f = f.reindex(x.index).shift(1)
    x = x.join(f)
    models = {'nq':NQ,'nq_vix':VIX,'nq_msci':NQ+cols,'nq_vix_msci':VIX+cols}
    metrics,tests,saved = [],[],[]
    for target in ['direction5','downside10']:
        y,end = labels(q,target)
        frame = x[VIX+cols+['source_date']].join([y,end]).dropna()
        assert (frame.source_date < frame.index).all()
        pred = classify(frame,models,2017,756)
        assert pred[list(models)].ge(0).all().all() and pred[list(models)].le(1).all().all()
        saved.append(pred.assign(target=target))
        for period,start,stop in [('development_reused','2017','2019'),('primary_reused','2020','2026'),
                                   ('historical_reused','2020','2024'),('recent_reused','2025','2026')]:
            sub = pred.loc[start:stop]
            for name in ['prior',*models]:
                metrics.append({'target':target,'period':period,'model':name,**metric(sub.y,sub[name])})
            if period=='primary_reused':
                for base,candidate in [('nq','nq_msci'),('nq_vix','nq_vix_msci')]:
                    tests.append({'target':target,'base':base,'candidate':candidate,
                                  **loss_test(sub.y.to_numpy(),sub[base],sub[candidate])})
    tests = pd.DataFrame(tests);tests['p_holm_4'] = holm(tests.p_hac)
    tests.to_csv(OUT/'Q_tests.csv',index=False)
    metrics = pd.DataFrame(metrics);metrics.to_csv(OUT/'Q_metrics.csv',index=False)
    pred = pd.concat(saved);pred.to_csv(OUT/'Q_predictions.csv')
    (OUT/'Q_metadata.json').write_text(json.dumps({'features':cols,'delay_nq_sessions':1,
        'first_prediction':str(pred.index.min()),'last_prediction':str(pred.index.max()),
        'pit_status':'ETF proxies,current adjusted price histories,original vintages not verified'},indent=2),encoding='utf-8')
    print(metrics[metrics.period.eq('primary_reused')].to_string(index=False));print(tests.to_string(index=False))


if __name__=='__main__':run()
