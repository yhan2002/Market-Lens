"""Audit a previously frozen onset rule; no new parameter search."""
import argparse
import json
import sqlite3
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from research_leading_improvements import ROOT, OUT
from research_drawdown_onset import REQUIRED_TICKERS, build_frame, fixed_core_rank_score, causal_trailing_percentile, event_metrics


def campaigns(frame, valid, score, cutoff, begin, finish):
    rows=[];until=-1
    eligible=frame.index[valid & frame.date.between(begin,finish)]
    if len(eligible)==0:raise ValueError('No eligible rows')
    onsets=frame.loc[frame.onset & frame.date.between(begin,finish),'_position'].astype(int).tolist()
    for i in eligible:
        position=int(frame.loc[i,'_position'])
        if position<=until or score.loc[i]<cutoff:continue
        until=position+20
        hits=[event for event in onsets if 2<=event-position<=20]
        rows.append({'date':str(frame.loc[i,'date'].date()),'position':position,'expiry':until,
                     'hit':bool(hits),'first_event':min(hits) if hits else None,'signal_lead':min(hits)-position if hits else None})
    issued=pd.DataFrame(rows)
    # Keep all observable eligible onsets in the denominator, including ones missed during cooldown.
    possible=[event for event in onsets if any(2<=event-int(frame.loc[i,'_position'])<=20 for i in eligible)]
    hits=[event for event in possible if any(2<=event-int(r['position'])<=20 for r in rows)]
    years=(pd.Timestamp(finish)-pd.Timestamp(begin)).days/365.25
    result={'eligible_onsets':len(possible),'matched_onsets':len(hits),'recall':len(hits)/len(possible) if possible else None,
            'campaigns':len(rows),'false_campaigns':sum(not r['hit'] for r in rows),
            'false_per_year':sum(not r['hit'] for r in rows)/years,
            'precision':sum(r['hit'] for r in rows)/len(rows) if rows else None,
            'median_action_lead':float(np.median([r['signal_lead']-1 for r in rows if r['hit']])) if hits else None}
    return result,issued


def run(valid_observations=False):
    with sqlite3.connect((ROOT/'regime_router_publish.db').resolve().as_uri()+'?mode=ro',uri=True) as c:
        raw=pd.read_sql_query('SELECT date,ticker,adj_close FROM daily_prices WHERE ticker IN ('+','.join('?' for _ in REQUIRED_TICKERS)+") AND date<='2026-09-08' ORDER BY date",c,params=REQUIRED_TICKERS,parse_dates=['date'])
    prices=raw.pivot(index='date',columns='ticker',values='adj_close')
    prices=prices.loc[prices['NQ=F'].notna()]
    frame=build_frame(prices)
    if valid_observations:
        # Separate sensitivity variant: 252 actual VIX quotes, never filled prices.
        lv=np.log(prices['^VIX'].dropna())
        z=(lv-lv.rolling(252).mean())/lv.rolling(252).std().replace(0,np.nan)
        frame['vix_z252']=z.reindex(prices.index).to_numpy()
    # Historical helper defaults may carry missing prices; the audited credit input does not.
    frame['hyg_ief_20']=(prices.HYG/prices.IEF).pct_change(20,fill_method=None).to_numpy()
    scores={'frozen_core':fixed_core_rank_score(frame)}
    cutoffs={'frozen_core':.72,'baseline_vix':.60,'baseline_vol':.72,'baseline_neg_ma200':.83}
    for name in list(cutoffs)[1:]:scores[name]=causal_trailing_percentile(frame[name].where(frame.at_risk))
    valid=frame.target.notna()
    for s in scores.values():valid &= s.notna()
    begin='2020-01-01';finish=str(frame.date.iloc[-1].date())
    cutoff_pos=len(frame)-21
    valid &= frame._position.le(cutoff_pos)
    mask=valid & frame.date.between(begin,finish)
    events=frame.loc[frame.onset & frame.date.between(begin,finish),'_position'].astype(int).tolist()
    results=[];allissues=[]
    for name,score in scores.items():
        old=event_metrics(frame.loc[mask],events,score,cutoffs[name])
        result,issues=campaigns(frame,valid,score,cutoffs[name],begin,finish)
        results.append({'model':name,'cutoff':cutoffs[name],'n':int(mask.sum()),
                        'auc':roc_auc_score(frame.loc[mask,'target'],score.loc[mask]),
                        'ap':average_precision_score(frame.loc[mask,'target'],score.loc[mask]),
                        'old_recall':old['onset_recall'],'old_false_per_year':old['false_alert_clusters_per_year'],
                        'old_alert_day_rate':old['alert_day_rate'],**result})
        allissues.append(issues.assign(model=name))
    prefix='I' if valid_observations else 'H'
    pd.DataFrame(results).to_csv(OUT/f'{prefix}_onset_audit.csv',index=False)
    pd.concat(allissues).to_csv(OUT/f'{prefix}_campaigns.csv',index=False)
    frame.assign(**scores).to_csv(OUT/f'{prefix}_frame.csv',index=False)
    fresh=frame.loc[frame.onset & frame.date.gt('2026-08-04'),['date','drawdown_252']]
    fresh.to_csv(OUT/f'{prefix}_new_onsets.csv',index=False)
    print(pd.DataFrame(results).to_string(index=False));print('NEW ONSETS',fresh.to_string(index=False))


def self_test():
    f=pd.DataFrame({'date':pd.bdate_range('2020-01-01',periods=100),'_position':np.arange(100),'onset':False})
    f.loc[[35,62],'onset']=True
    valid=pd.Series(True,index=f.index)
    score=pd.Series(1.,index=f.index)
    result,issued=campaigns(f,valid,score,.72,'2020-01-01','2020-06-01')
    assert issued.position.tolist()==[0,21,42,63,84]
    assert result['matched_onsets']==2
    assert issued.expiry.tolist()==[20,41,62,83,104]
    # A one-day warning cannot count as actionable after next-close entry.
    f.onset=False;f.loc[1,'onset']=True
    result,_=campaigns(f,valid,score,.72,'2020-01-01','2020-06-01')
    assert result['matched_onsets']==0
    print('PASS: fixed expiry, non-extending cooldown, next-close action lead')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--self-test',action='store_true');parser.add_argument('--valid-observations',action='store_true');args=parser.parse_args()
    self_test() if args.self_test else run(args.valid_observations)
