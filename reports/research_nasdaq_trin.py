"""T: indirect industry TRIN inputs, no fill and next-session availability assumption."""
import json
import numpy as np
import pandas as pd
from research_leading_improvements import ROOT,OUT,panel,labels,classify,metric,loss_test,VIX
from validate_nasdaq_ratio import holm


def run():
    raw = pd.read_csv(ROOT/'outputs/bloomberg_sector_option_full_staging_20260904_184703/merged_history.csv',dtype={'code':str},parse_dates=['date'])
    raw = raw[raw.field_id.eq('IX116')]
    assert not raw.duplicated(['code','date']).any() and raw.value.gt(0).all()
    _,q,x = panel()
    codes = ['1010','2010','4510','5510']
    p = raw.pivot(index='date',columns='code',values='value')[codes].reindex(x.index)
    f = np.log(p).shift(1).add_prefix('trin_')
    x = x.join(f)
    models = {'nq_vix':VIX,'trin':VIX+list(f)}
    results,tests,saved = [],[],[]
    for target in ['direction5','downside10']:
        y,end=labels(q,target)
        frame=x[VIX+list(f)].join([y,end]).dropna()
        pred=classify(frame,models,2021,504)
        saved.append(pred.assign(target=target))
        tests.append({'target':target,'n':len(pred),**loss_test(pred.y.to_numpy(),pred.nq_vix,pred.trin)})
        for period,sub in [('all',pred),*[(str(year),v) for year,v in pred.groupby(pred.index.year)]]:
            for name in ['prior',*models]:
                results.append({'target':target,'period':period,'model':name,**metric(sub.y,sub[name])})
    t=pd.DataFrame(tests);t['p_holm_2']=holm(t.p_hac)
    t.to_csv(OUT/'T_tests.csv',index=False)
    m=pd.DataFrame(results);m.to_csv(OUT/'T_metrics.csv',index=False)
    s=pd.concat(saved);s.to_csv(OUT/'T_predictions.csv')
    meta={'codes':codes,'raw_coverage':raw.groupby('code').date.agg(['min','max','count']).astype(str).to_dict(),
          'all_four_quote_days':int(p.notna().all(axis=1).sum()),'oos_start':str(s.index.min()),'oos_end':str(s.index.max()),
          'scope':'Four industry TRINs,not Nasdaq-wide breadth','pit_status':'publication vintage and constituent basis unverified'}
    (OUT/'T_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(m[m.period.eq('all')].to_string(index=False));print(t.to_string(index=False));print(json.dumps(meta,indent=2))


if __name__=='__main__':run()
