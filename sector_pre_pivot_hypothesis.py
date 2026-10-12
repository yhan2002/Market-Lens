"""Frozen IS/OOS test of absolute-sector pre-pivot signatures. Local research only."""
import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sector_absolute_turn_prediction import features, ABS_FEATURES, PATH_FEATURES, TREND_FEATURES
from sector_rotation import SECTOR_ETFS
from sector_turning_pooled_shadow import _labels_with_calendar_gaps, _direction_frame
from sector_turning_sector_macro_boost import build_factor_features, RAW_MACRO_FEATURES

ROOT=Path(__file__).resolve().parent
PRICE=ABS_FEATURES+PATH_FEATURES+TREND_FEATURES
MACRO=list(RAW_MACRO_FEATURES)
SPLIT=pd.Timestamp('2023-01-01')
VOLUME=['relative_volume','volume_change5','signed_volume5','close_location5']
OPTIONS=['iv30','iv_change5','iv_term','iv_term_change5','put_call_oi','put_call_change5']
DIVERGENCE=['low_iv_relief','high_iv_stress','low_volume_relief','high_volume_fading']


def extreme_responses(price,iv,signed_volume):
    """Compare today's response with the prior 20-session price extreme, not a future pivot."""
    out=pd.DataFrame(index=price.index)
    for side,fn,columns in [('low',np.argmin,DIVERGENCE[::2]),
                            ('high',np.argmax,DIVERGENCE[1::2])]:
        offsets=price.shift(1).rolling(20).apply(fn,raw=True)
        positions=np.arange(len(price))-20+offsets.fillna(0).to_numpy().astype(int)
        positions=np.clip(positions,0,len(price)-1)
        prior_price=pd.Series(price.to_numpy()[positions],index=price.index)
        breakout=price.lt(prior_price) if side=='low' else price.gt(prior_price)
        for response,col in zip((iv,signed_volume),columns):
            prior=pd.Series(response.to_numpy()[positions],index=price.index)
            delta=response-prior
            if col in ('low_iv_relief','high_volume_fading'):delta=-delta
            valid=offsets.notna() & price.notna() & response.notna() & prior.notna()
            out[col]=delta.where(breakout,0).where(valid)
    return out.reindex(columns=DIVERGENCE)


def liquidity_features(raw,calendar,option_history):
    """Close-known volume proxies, previous-session options; no fill or put-call IV skew."""
    if raw.duplicated(['ticker','date']).any() or option_history.duplicated(['code','field_id','date']).any():
        raise ValueError('duplicate input keys')
    parts=[]
    for ticker in SECTOR_ETFS:
        p=raw.loc[raw.ticker.eq(ticker)].set_index('date').reindex(calendar)
        v=p.volume.where(p.volume>0)
        f=pd.DataFrame(index=calendar)
        f['relative_volume']=v/v.shift(1).rolling(20).mean()
        f['volume_change5']=np.log(v.rolling(5).mean()/v.shift(5).rolling(5).mean())
        sign=np.sign(p.adj_close.diff())
        f['signed_volume5']=(sign*v).rolling(5).sum()/v.rolling(5).sum()
        span=(p.high-p.low).replace(0,np.nan)
        f['close_location5']=((2*p.close-p.high-p.low)/span).rolling(5).mean()
        op=option_history.loc[option_history.code.eq(ticker)].pivot(index='date',columns='field_id',values='value')
        op=op.reindex(calendar).shift(1)
        if not {'RK324','RK293','PQ950'}.issubset(op.columns):raise ValueError('missing option fields')
        f['iv30']=op.RK324
        f['iv_change5']=op.RK324.diff(5)
        f['iv_term']=op.RK324/op.RK293.where(op.RK293>0)-1
        f['iv_term_change5']=f.iv_term.diff(5)
        f['put_call_oi']=op.PQ950
        f['put_call_change5']=op.PQ950.diff(5)
        # Align all divergence inputs to t-1, matching the option observation date.
        f[DIVERGENCE]=extreme_responses(p.adj_close.shift(1),f.iv30,f.signed_volume5.shift(1))
        f['date']=calendar;f['ticker']=ticker;parts.append(f.reset_index(drop=True))
    return pd.concat(parts,ignore_index=True)


def recall_curve_metrics(y,score):
    """OOS ranking diagnostic only: interpolation is NOT an executable alert rule."""
    fpr,tpr,_=roc_curve(y,score,drop_intermediate=False)
    # Use the least false positives at each attainable recall before interpolation.
    curve=pd.DataFrame({'tpr':tpr,'fpr':fpr}).groupby('tpr').fpr.min()
    false=float(np.interp(.60,curve.index,curve.values));base=float(np.mean(y))
    return dict(fpr_at_60_recall=false,precision_at_60_recall=(.60*base)/(.60*base+false*(1-base)))


def matched_effects(train,columns):
    """Same sector and IS-only trend/volatility quintiles; controls have no confirmation."""
    frame=train.copy()
    for col in ('ret20','vol20'):
        edges=np.unique(np.quantile(frame[col],np.linspace(0,1,6)))
        frame[col+'_bin']=np.searchsorted(edges[1:-1],frame[col],side='right')
    keys=['sector','ret20_bin','vol20_bin']
    controls=frame.loc[frame.confirmation_target.eq(0)].groupby(keys)[columns].mean()
    cases=frame.loc[frame.target.eq(1)].join(controls,on=keys,rsuffix='_control')
    scale=frame[columns].std().replace(0,np.nan)
    rows=[]
    for col in columns:
        delta=(cases[col]-cases[col+'_control'])/scale[col]
        for band,mask in [('ALL',pd.Series(True,index=cases.index)),
                          ('1-5',cases.lead.between(1,5)),('6-10',cases.lead.between(6,10)),
                          ('11-20',cases.lead.between(11,20))]:
            valid=delta.loc[mask].dropna()
            rows.append(dict(feature=col,lead_band=band,matched_rows=len(valid),
                             standardized_difference=float(valid.mean())))
    return pd.DataFrame(rows)


def summarize(rows,score):
    y=rows.target.to_numpy();alert=score>=.5
    positives=rows.loc[rows.target.eq(1)].copy()
    positives['alert']=alert[rows.target.eq(1)]
    events=positives.groupby(['sector','target_event_confirmation_date']).alert.any()
    hits=positives.loc[positives.alert]
    lead=hits.groupby(['sector','target_event_confirmation_date']).lead.max()
    return dict(**recall_curve_metrics(y,score),rows=len(rows),base_rate=float(y.mean()),auc=float(roc_auc_score(y,score)),
                pr_auc=float(average_precision_score(y,score)),alerts=int(alert.sum()),
                precision=float(y[alert].mean()) if alert.any() else None,
                sample_recall=float(y[alert].sum()/y.sum()),eligible_events=len(events),
                event_recall=float(events.mean()),median_lead=float(lead.median()) if len(lead) else None)


def block_delta(rows,a,b,metric='auc'):
    # Calendar blocks keep contemporaneous sectors together; 300 replicates is exploratory.
    groups=[g.index.to_numpy() for _,g in rows.groupby(rows.date.dt.to_period('Q'))]
    rng=np.random.default_rng(42);values=[]
    for _ in range(300):
        ix=np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))])
        y=rows.loc[ix,'target']
        if y.nunique()==2:
            if metric=='auc':
                values.append(roc_auc_score(y,rows.loc[ix,b])-roc_auc_score(y,rows.loc[ix,a]))
            elif metric=='fpr60':
                values.append(recall_curve_metrics(y,rows.loc[ix,b])['fpr_at_60_recall']-
                              recall_curve_metrics(y,rows.loc[ix,a])['fpr_at_60_recall'])
            else:raise ValueError('unknown metric')
    return np.quantile(values,[.025,.975]).tolist()


def curve_audit(directory):
    """Evaluate saved scores without retraining or changing any decision threshold."""
    target=directory/'false_positive_ci.json'
    if target.exists():raise FileExistsError(target)
    predictions=pd.read_csv(directory/'predictions.csv',parse_dates=['date'])
    output=[]
    for direction,rows in predictions.groupby('direction'):
        rows=rows.reset_index(drop=True)
        for suffix in ('','_TREE'):
            reference='PRICE'+suffix
            candidates=[c for c in rows if c.startswith('PRICE_') and
                        c.endswith('_TREE')==bool(suffix) and c!=reference]
            for candidate in candidates:
                output.append(dict(direction=direction,comparison=candidate+' minus '+reference,
                                   fpr60_delta_ci95=block_delta(rows,reference,candidate,'fpr60')))
    target.write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(output))


def run(output,volume_options=False,divergence=False):
    volume_options=volume_options or divergence
    if output.exists():raise FileExistsError(output)
    with sqlite3.connect((ROOT/'regime_router_publish.db').as_uri()+'?mode=ro',uri=True) as con:
        names=['SPY',*SECTOR_ETFS]
        raw=pd.read_sql_query('SELECT date,ticker,open,high,low,close,adj_close,volume FROM daily_prices WHERE ticker IN ('+
                             ','.join('?' for _ in names)+')',con,params=names,parse_dates=['date'])
    if raw.duplicated(['date','ticker']).any():raise ValueError('duplicate prices')
    prices=raw.pivot(index='date',columns='ticker',values='adj_close').sort_index()
    feature=features(prices)
    if volume_options:
        op=pd.read_csv(ROOT/'outputs/bloomberg_sector_option_full_staging_20260904_184703/merged_history.csv',parse_dates=['date'])
        feature=feature.merge(liquidity_features(raw,prices.index,op),on=['date','ticker'],validate='one_to_one')
        extra=VOLUME+OPTIONS+(DIVERGENCE if divergence else [])
    else:
        macro_path=ROOT/'outputs/bloomberg_turning_macro_staging_20260902_212424/bloomberg_turning_macro_20260902_212424/macro_history.csv'
        factors=build_factor_features(macro_path,ROOT/'regime_router_publish.db')
        aligned=factors.reindex(prices.index)[MACRO].shift(1)
        aligned.index.name='date'
        feature=feature.merge(aligned.reset_index(),on='date',how='left',validate='many_to_one')
        extra=MACRO
    parts=[]
    for ticker in SECTOR_ETFS:
        labels=_labels_with_calendar_gaps(prices[ticker],horizon=20)
        labels['date']=prices.index;labels['sector']=ticker;labels['_position']=np.arange(len(prices))
        parts.append(labels.reset_index(drop=True))
    base=pd.concat(parts,ignore_index=True).merge(feature,left_on=['date','sector'],right_on=['date','ticker'],validate='one_to_one')
    predictions=[];metrics=[];effects=[];audits=[];intervals=[]
    for direction in ('ROTATION_IN','ROTATION_OUT'):
        frame=_direction_frame(base,direction,target_mode='PRE_PIVOT')
        prefix='bottom' if direction=='ROTATION_IN' else 'top'
        frame['confirmation_target']=frame[prefix+'_target']
        frame['lead']=frame.target_event_pivot_position-frame._position
        frame=frame.loc[frame.at_risk & frame.target.notna()].replace([np.inf,-np.inf],np.nan)
        before=len(frame);frame=frame.dropna(subset=PRICE+extra).copy()
        train=frame.loc[(frame.date<SPLIT)&(frame.label_end_date<SPLIT)].copy()
        test=frame.loc[frame.date>=SPLIT].copy().reset_index(drop=True)
        assert train.label_end_date.max()<test.date.min()
        assert (train.loc[train.target.eq(1),'lead']>0).all()
        effect=matched_effects(train,PRICE+extra);effect['direction']=direction;effects.append(effect)
        ranking=effect.loc[effect.lead_band.eq('ALL')].set_index('feature').standardized_difference.abs()
        selected_price=ranking.reindex(PRICE).dropna().nlargest(6).index.tolist()
        selected_macro=ranking.reindex(extra).dropna().nlargest(6).index.tolist()
        sets={'PRICE':selected_price,'MACRO':selected_macro,'COMBINED':selected_price+selected_macro,
              'PRICE_TREE':selected_price,'COMBINED_TREE':selected_price+selected_macro}
        if volume_options:
            sets={'PRICE':selected_price,'PRICE_VOLUME':selected_price+VOLUME,
                  'PRICE_OPTIONS':selected_price+OPTIONS,'PRICE_BOTH':selected_price+VOLUME+OPTIONS}
            sets.update({name+'_TREE':cols for name,cols in list(sets.items())})
        if divergence:
            sets={'PRICE':selected_price,'PRICE_BOTH':selected_price+VOLUME+OPTIONS,
                  'PRICE_DIVERGENCE':selected_price+DIVERGENCE,
                  'PRICE_BOTH_DIVERGENCE':selected_price+VOLUME+OPTIONS+DIVERGENCE}
            sets.update({name+'_TREE':cols for name,cols in list(sets.items())})
        sector_cols=[]
        for ticker in SECTOR_ETFS:
            col='sector_'+ticker;sector_cols.append(col)
            train[col]=train.sector.eq(ticker).astype(float);test[col]=test.sector.eq(ticker).astype(float)
        saved=test[['date','sector','target','lead','target_event_confirmation_date','label_end_date']].copy()
        for name,cols in sets.items():
            use=cols+sector_cols
            if name.endswith('_TREE'):
                model=HistGradientBoostingClassifier(max_depth=2,max_iter=80,min_samples_leaf=60,
                    l2_regularization=2,learning_rate=.05,early_stopping=False,random_state=42)
                y=train.target.to_numpy();weight=np.where(y==1,len(y)/(2*y.sum()),len(y)/(2*(len(y)-y.sum())))
                model.fit(train[use],y,sample_weight=weight)
            else:
                model=make_pipeline(StandardScaler(),LogisticRegression(C=.1,class_weight='balanced',max_iter=2000,random_state=42))
                model.fit(train[use],train.target)
            score=model.predict_proba(test[use])[:,1];saved[name]=score
            metrics.append(dict(direction=direction,model=name,period='ALL',**summarize(test,score)))
            for year,group in test.groupby(test.date.dt.year):
                if group.target.nunique()==2:
                    metrics.append(dict(direction=direction,model=name,period=str(year),**summarize(group,score[group.index])))
            for sector,group in test.groupby('sector'):
                if group.target.nunique()==2:
                    metrics.append(dict(direction=direction,model=name,period=sector,**summarize(group,score[group.index])))
        for name in sets:
            reference='PRICE_TREE' if name.endswith('_TREE') else 'PRICE'
            if name!=reference:
                intervals.append(dict(direction=direction,comparison=name+' minus '+reference,
                                      auc_delta_ci95=block_delta(saved,reference,name)))
        if divergence:
            for suffix in ('','_TREE'):
                reference='PRICE_BOTH'+suffix;candidate='PRICE_BOTH_DIVERGENCE'+suffix
                intervals.append(dict(direction=direction,comparison=candidate+' minus '+reference,
                                      auc_delta_ci95=block_delta(saved,reference,candidate),
                                      fpr60_delta_ci95=block_delta(saved,reference,candidate,'fpr60')))
        saved['direction']=direction;predictions.append(saved)
        audits.append(dict(direction=direction,train_rows=len(train),test_rows=len(test),
                           train_start=str(train.date.min()),train_end=str(train.date.max()),
                           latest_train_label_end=str(train.label_end_date.max()),test_start=str(test.date.min()),
                           test_end=str(test.date.max()),missing_feature_rows=before-len(frame),selected=sets,
                           all_oos_confirmed_events=int(((base.event==('BOTTOM_END' if prefix=='bottom' else 'TOP_END')) &
                               base.date.between(test.date.min(),test.date.max())).sum())))
        print(direction, 'finished',flush=True)
    output.mkdir(parents=True)
    pd.concat(predictions).to_csv(output/'predictions.csv',index=False)
    pd.DataFrame(metrics).to_csv(output/'metrics.csv',index=False)
    pd.concat(effects).to_csv(output/'is_matched_effects.csv',index=False)
    (output/'audit.json').write_text(json.dumps(dict(folds=audits,comparisons=intervals,
        limits=['absolute adjusted ETF prices; revised vintage, macro PIT unverified',
                'volume/option extension uses prior-session options and close-known volume proxies; not true buy/sell order flow',
                'OOS recall-60 curve metrics are retrospective interpolated diagnostics, not deployable cutoffs',
                'divergence compares t-1 price/IV/signed-volume with prior20 price extreme; exploratory fixed features, not future pivots',
                'pre-pivot AND confirmation within 20 sessions; slow confirmations excluded',
                'fixed IS before 2023, OOS historically reviewed, not pristine holdout',
                'IS effect size selection only, overlapping samples not independent',
                'balanced learner scores are not calibrated probabilities; fixed .5 alert cutoff',
                'event recall denominator includes only events with eligible complete pre-pivot rows',
                'controls are same-sector trend/volatility bins without next20 confirmation, not causal identification',
                'quarter-block 300-replicate intervals exploratory, not multiplicity adjusted',
                'frozen model tests signature persistence, not a periodically retrained strategy'],
        operational_changes=False),indent=2),encoding='utf-8')
    print(pd.DataFrame(metrics).query("period=='ALL'").to_string(index=False))
    print(json.dumps(intervals))


def self_test():
    dates=pd.bdate_range('2020-01-01',periods=260)
    p=pd.Series(100*np.exp(.07*np.sin(np.arange(260)/15)),index=dates)
    a=_labels_with_calendar_gaps(p,horizon=20)
    q=p.copy();q.iloc[-1]*=1.5;b=_labels_with_calendar_gaps(q,horizon=20)
    pd.testing.assert_frame_equal(a.iloc[:-1][['state','event','pivot_date']],b.iloc[:-1][['state','event','pivot_date']])
    toy=pd.DataFrame({'sector':['X']*20,'ret20':np.arange(20),'vol20':np.arange(20),
        'target':np.tile([0,1],10),'confirmation_target':np.tile([0,1],10),
        'lead':5,'x':np.tile([0,1],10)})
    effect=matched_effects(toy,['x'])
    assert effect.iloc[0].matched_rows==10 and effect.iloc[0].standardized_difference>0
    assert recall_curve_metrics([0,0,1,1],[.1,.2,.8,.9])['fpr_at_60_recall']==0
    raw=pd.concat([pd.DataFrame({'ticker':t,'date':dates,'open':p.values,'high':p.values+1,
        'low':p.values-1,'close':p.values,'adj_close':p.values,'volume':np.arange(260)+100}) for t in SECTOR_ETFS])
    op=pd.concat([pd.DataFrame({'code':t,'date':dates,'field_id':fld,'value':np.arange(260)+100.})
                  for t in SECTOR_ETFS for fld in ['RK324','RK293','PQ950']])
    f=liquidity_features(raw,dates,op)
    op.loc[op.date.eq(dates[-1]),'value']*=2
    pd.testing.assert_frame_equal(f,liquidity_features(raw,dates,op))
    raw.loc[raw.date.eq(dates[-1]),'volume']*=2
    g=liquidity_features(raw,dates,op)
    pd.testing.assert_frame_equal(f.loc[f.date<dates[-1]],g.loc[g.date<dates[-1]])
    # Prefix invariance: no later observation can change an earlier extreme response.
    z=extreme_responses(p,p*.2,p.diff())
    assert z.columns.tolist()==DIVERGENCE
    pd.testing.assert_frame_equal(z.iloc[:200],extreme_responses(p.iloc[:200],p.iloc[:200]*.2,p.diff().iloc[:200]))
    falling=pd.Series(np.arange(30.,0.,-1))
    d=extreme_responses(falling,falling,pd.Series(np.arange(30.)))
    assert d.loc[25,'low_iv_relief']==1 and d.loc[25,'low_volume_relief']==1
    assert d.loc[25,'high_iv_stress']==0 and d.loc[25,'high_volume_fading']==0
    rising=-falling
    d=extreme_responses(rising,rising,falling)
    assert d.loc[25,'high_iv_stress']==1 and d.loc[25,'high_volume_fading']==1
    missing=rising.copy();missing.iloc[24]=np.nan
    assert extreme_responses(missing,rising,falling).loc[25].isna().all()
    print('self-test PASS: causal event history, matched IS controls')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--volume-options',action='store_true')
    parser.add_argument('--divergence',action='store_true',help='Compare fixed price-extreme response features on common rows')
    parser.add_argument('--curve-audit',type=Path)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/pre_pivot_hypothesis_20260921')
    args=parser.parse_args()
    if args.self_test:self_test()
    elif args.curve_audit:curve_audit(args.curve_audit)
    else:run(args.output_dir,args.volume_options,args.divergence)
