"""R: saved Nasdaq baseline skill, calibration and alert audit."""
import pandas as pd
from research_leading_improvements import OUT, metric, loss_test
from validate_nasdaq_ratio import holm


def run():
    tests,annual,calibration,metrics = [],[],[],[]
    for source in ['A','Q']:
        saved = pd.read_csv(OUT/f'{source}_predictions.csv',parse_dates=['date'])
        saved = saved[saved.date.ge('2020-01-01')]
        assert not saved.duplicated(['date','target']).any()
        for target,sub in saved.groupby('target'):
            tests.append({'source':source,'target':target,'n':len(sub),
                          **loss_test(sub.y.to_numpy(),sub.prior,sub.nq)})
            for name in ['prior','nq']:
                alert = sub.nq_alert if name=='nq' else None
                metrics.append({'source':source,'target':target,'model':name,**metric(sub.y,sub[name],alert)})
            for year,s in sub.groupby(sub.date.dt.year):
                annual.append({'source':source,'target':target,'year':year,'n':len(s),
                               'brier_improvement':((s.prior-s.y)**2-(s.nq-s.y)**2).mean()})
            if target=='downside10':
                bins = pd.cut(sub.nq,[0,.2,.4,.6,.8,1],include_lowest=True)
                for label,s in sub.groupby(bins,observed=True):
                    calibration.append({'source':source,'bin':str(label),'n':len(s),
                                        'mean_probability':s.nq.mean(),'event_rate':s.y.mean()})
    tests = pd.DataFrame(tests);tests['p_holm_4'] = holm(tests.p_hac)
    for name,frame in [('tests',tests),('annual',annual),('calibration',calibration),('metrics',metrics)]:
        pd.DataFrame(frame).to_csv(OUT/f'R_{name}.csv',index=False)
    print(tests.to_string(index=False));print(pd.DataFrame(metrics).to_string(index=False))


if __name__=='__main__':run()
