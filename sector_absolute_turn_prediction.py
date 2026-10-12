"""Purged quarterly walk-forward predictions for 11-ETF absolute reversals.

Research only. Existing labels are 20-session path outcomes, not structural peaks.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier

from sector_rotation import SECTOR_ETFS, build_rrg_signals

ROOT=Path(__file__).resolve().parent
ABS_FEATURES=['ret1','ret5','ret20','ret60','vol20','drawdown20','rebound20','speed_change']
RRG_FEATURES=['rrg_ratio','rrg_momentum','rrg_delta5']
PATH_FEATURES=['return5_risk','return20_risk','drawdown_risk','rebound_risk','speed_risk','vol_ratio']
TREND_FEATURES=['ma_gap_short','ma_gap_long','ma_slope','ma_gap_change']
SHIFT_FEATURES=['cusum_up','cusum_down']
MODELS={'ABS_LOGIT':ABS_FEATURES,'ABS_RRG_LOGIT':ABS_FEATURES+RRG_FEATURES,
        'PATH_BOOST':ABS_FEATURES+RRG_FEATURES+PATH_FEATURES,
        'TREND_BOOST':ABS_FEATURES+RRG_FEATURES+PATH_FEATURES+TREND_FEATURES,
        'SHIFT_BOOST':ABS_FEATURES+RRG_FEATURES+PATH_FEATURES+TREND_FEATURES+SHIFT_FEATURES}
MODELS.update({f'FRESH_{name}':MODELS[name] for name in ('TREND_BOOST','SHIFT_BOOST')})


def cumulative_shift(innovations):
    """One-sided CUSUM states, not probabilities or forecasts. No retrospective writes."""
    values=np.asarray(innovations,dtype=float)
    if values.ndim!=1 or np.isinf(values).any():raise ValueError('invalid innovations')
    out=np.full((len(values),2),np.nan);up=down=0.0
    for i,z in enumerate(values):
        if np.isnan(z):
            up=down=0.0
            continue
        up=max(0.0,up+z-.5);down=max(0.0,down-z-.5)
        out[i]=up,down
        # ponytail: fixed k=.5/h=5 screening features, not fitted change probabilities.
        if up>=5:up=0.0
        if down>=5:down=0.0
    return out


def features(prices):
    rrg=build_rrg_signals(prices).sort_values(['ticker','date'])
    rrg['rrg_ratio']=rrg.rs_ratio-100
    rrg['rrg_momentum']=rrg.rs_momentum-100
    rrg['rrg_delta5']=rrg.groupby('ticker').rrg_momentum.diff(5)
    previous=rrg.groupby('ticker').rrg_momentum.shift(5)
    rrg['rrg_event']=np.select([(previous<0)&(rrg.rrg_momentum>=0),
                                (previous>=0)&(rrg.rrg_momentum<0)],[1,-1],default=0)
    parts=[]
    for ticker in SECTOR_ETFS:
        p=prices[ticker]; log=np.log(p)
        f=pd.DataFrame({'date':p.index,'ticker':ticker,'observed_price':p.to_numpy()})
        for lag in (1,5,20,60): f[f'ret{lag}']=log.diff(lag).to_numpy()
        f['vol20']=log.diff().rolling(20).std().to_numpy()
        f['drawdown20']=(p/p.rolling(20).max()-1).to_numpy()
        f['rebound20']=(p/p.rolling(20).min()-1).to_numpy()
        f['speed_change']=(log.diff(5)/5-log.diff(20)/20).to_numpy()
        sigma=log.diff().rolling(60).std().replace(0,np.nan)
        f['return5_risk']=(log.diff(5)/(sigma*np.sqrt(5))).to_numpy()
        f['return20_risk']=(log.diff(20)/(sigma*np.sqrt(20))).to_numpy()
        f['drawdown_risk']=f.drawdown20/(sigma*np.sqrt(20)).to_numpy()
        f['rebound_risk']=f.rebound20/(sigma*np.sqrt(20)).to_numpy()
        f['speed_risk']=f.speed_change/sigma.to_numpy()
        f['vol_ratio']=(log.diff().rolling(5).std()/sigma).to_numpy()
        # Fixed windows, current/past closes only; not an OUT veto or peak label.
        ma10=p.rolling(10).mean();ma40=p.rolling(40).mean();ma120=p.rolling(120).mean()
        gap=np.log(ma10/ma40)
        f['ma_gap_short']=(gap/(sigma*np.sqrt(40))).to_numpy()
        f['ma_gap_long']=(np.log(ma40/ma120)/(sigma*np.sqrt(120))).to_numpy()
        f['ma_slope']=(np.log(ma40).diff(5)/(sigma*np.sqrt(5))).to_numpy()
        f['ma_gap_change']=(gap.diff(5)/(sigma*np.sqrt(5))).to_numpy()
        daily=log.diff()
        innovations=(daily-daily.rolling(60).mean().shift(1))/sigma.shift(1)
        f[SHIFT_FEATURES]=cumulative_shift(innovations)
        parts.append(f)
    return pd.concat(parts,ignore_index=True).merge(
        rrg[['date','ticker',*RRG_FEATURES,'rrg_event']],on=['date','ticker'],how='left',validate='one_to_one')


def training_rows(frame,cutoff):
    return frame.loc[frame.date.lt(cutoff)&frame.label_end.notna()&frame.label_end.lt(cutoff)]


def load(args):
    labels=pd.read_csv(args.labels,parse_dates=['date','label_end'])
    if labels.duplicated(['date','ticker']).any() or set(labels.ticker)!=set(SECTOR_ETFS):
        raise ValueError('invalid labels universe/duplicate keys')
    with sqlite3.connect(args.database.resolve().as_uri()+'?mode=ro',uri=True) as c:
        names=['SPY',*SECTOR_ETFS];slots=','.join('?' for _ in names)
        raw=pd.read_sql_query(f'SELECT date,ticker,adj_close FROM daily_prices WHERE ticker IN ({slots})',c,params=names,parse_dates=['date'])
    if raw.duplicated(['date','ticker']).any(): raise ValueError('duplicate prices')
    prices=raw.pivot(index='date',columns='ticker',values='adj_close').sort_index()
    if (prices<=0).any().any() or np.isinf(prices.to_numpy()).any():raise ValueError('invalid prices')
    frame=labels.merge(features(prices),on=['date','ticker'],validate='one_to_one')
    if not np.allclose(frame.price,frame.observed_price,rtol=1e-8,atol=1e-8):
        raise ValueError('label/price vintage mismatch; regenerate labels in a new directory')
    # Weekly decisions match the existing table cadence. Incomplete final week omitted.
    dates=pd.Series(prices.index,index=prices.index)
    weekly=dates.groupby(dates.index.to_period('W-FRI')).max()
    weekly=weekly.loc[weekly.index.end_time.normalize()<=prices.index.max()]
    frame=frame.loc[frame.date.isin(weekly)&frame.trend.isin([-1,1])].copy()
    frame=frame.replace([np.inf,-np.inf],np.nan).dropna(subset=ABS_FEATURES+RRG_FEATURES+PATH_FEATURES+TREND_FEATURES+SHIFT_FEATURES)
    valid={'TOP_END','BOTTOM_END','UP_CONTINUES','DOWN_CONTINUES','UNCERTAIN','IMMATURE'}
    if not set(frame.target).issubset(valid):raise ValueError('invalid at-risk target')
    frame['y']=frame.target.isin(['TOP_END','BOTTOM_END']).astype(int)
    frame.loc[frame.target.eq('IMMATURE'),'y']=np.nan
    for ticker in SECTOR_ETFS:
        p=prices[ticker];entry=p.shift(-1)
        future=pd.concat([p.shift(-k)/entry-1 for k in range(2,22)],axis=1)
        complete=future.notna().all(axis=1)
        ix=frame.index[frame.ticker.eq(ticker)]; ds=frame.loc[ix,'date']
        frame.loc[ix,'trade_return20']=(p.shift(-21)/entry-1).where(complete).reindex(ds).to_numpy()
        frame.loc[ix,'potential_upside20']=future.max(axis=1).where(complete).reindex(ds).to_numpy()
        frame.loc[ix,'potential_downside20']=future.min(axis=1).where(complete).reindex(ds).to_numpy()
    return frame.sort_values(['date','ticker']).reset_index(drop=True)


def path_classes(target):
    """Retain ambiguous paths as their own class instead of pretending continuation."""
    mapped=target.map({'UNCERTAIN':0,'UP_CONTINUES':1,'DOWN_CONTINUES':1,
                       'TOP_END':2,'BOTTOM_END':2})
    if mapped.isna().any():raise ValueError('immature/unknown label supplied for training')
    return mapped.astype(int)


def wilson_lower(hits,n):
    if n==0:return 0.0
    p=hits/n;z=1.96
    return float((p+z*z/(2*n)-z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n))


def select_cutoff(scores,truth):
    """Fixed small grid, validation only. No forced quota or always-alert fallback."""
    scores=np.asarray(scores,dtype=float);truth=np.asarray(truth,dtype=float)
    if scores.ndim!=1 or truth.ndim!=1 or len(scores)!=len(truth) or not np.isfinite(scores).all() or not np.isin(truth,[0,1]).all() or not ((scores>=0)&(scores<=1)).all():
        raise ValueError('invalid validation scores/labels')
    positives=int(truth.sum());candidates=[];grid=[]
    base=float(truth.mean()) if len(truth) else None
    for cutoff in (.30,.40,.50,.60):
        selected=scores>=cutoff;n=int(selected.sum());hits=int(truth[selected].sum())
        precision=hits/n if n else 0;recall=hits/positives if positives else 0
        lower=wilson_lower(hits,n)
        reasons=[]
        if n<10:reasons.append('fewer_than_10_alerts')
        if recall<.15:reasons.append('recall_below_15pct')
        if precision<.50:reasons.append('precision_below_50pct')
        if base is None or lower<=base:reasons.append('lower_not_above_base')
        grid.append(dict(cutoff=cutoff,alerts=n,hits=hits,precision=precision,recall=recall,
                         wilson_lower_descriptive=lower,rejection_reasons=reasons))
        # Heuristic screening guard, not a significance test on dependent weekly labels.
        if not reasons:
            candidates.append((lower,precision,recall,cutoff,n))
    details={'validation_rows':len(truth),'validation_base_rate':base,'cutoff_grid':grid}
    if not candidates:return None,{'status':'ABSTAIN_VALIDATION_FAILED',**details}
    best=max(candidates)
    return best[3],{'status':'VALIDATION_PASSED_SCREEN_ONLY','cutoff':best[3],
                    'precision':best[1],'recall':best[2],'wilson_lower_descriptive':best[0],'alerts':best[4],**details}


def fit_fresh_candidate(train,test,columns,history,name,cutoff):
    """Refit on mature labels; screen using earlier out-of-sample scores only."""
    if set(path_classes(train.target))!={0,1,2}:
        return None,{'status':'ABSTAIN_INSUFFICIENT_CLASSES'}
    assert train.label_end.max()<cutoff
    past=training_rows(history,cutoff).dropna(subset=['y',name+'_score'])
    past=past.loc[past.date.ge(cutoff-pd.DateOffset(years=2))]
    threshold,info=select_cutoff(past[name+'_score'],past.y)
    if past.date.nunique()<52:
        threshold=None
        info['status']='ABSTAIN_OOS_WARMUP'
    model=HistGradientBoostingClassifier(max_iter=80,max_depth=2,min_samples_leaf=40,
        learning_rate=.05,l2_regularization=2,early_stopping=False,random_state=42)
    model.fit(train[columns],path_classes(train.target))
    slot=int(np.flatnonzero(model.classes_==2)[0])
    info.update(fit_rows=len(train),latest_fit_label_end=str(train.label_end.max()),
                validation_weeks=int(past.date.nunique()),
                latest_validation_label_end=str(past.label_end.max()),
                validation_source='prior quarterly OOS predictions, trailing 2 calendar years')
    return (model.predict_proba(test[columns]),slot,threshold,model.classes_),info


def fit_path_candidate(train,test,columns):
    dates=sorted(train.date.unique())
    if len(dates)<78:return None,{'status':'ABSTAIN_SHORT_HISTORY'}
    validation_start=pd.Timestamp(dates[-26])
    fit=training_rows(train,validation_start)
    validation=train.loc[train.date.ge(validation_start)]
    if fit.date.nunique()<52 or set(path_classes(fit.target))!={0,1,2}:
        return None,{'status':'ABSTAIN_INSUFFICIENT_CLASSES'}
    assert fit.label_end.max()<validation_start
    model=HistGradientBoostingClassifier(max_iter=80,max_depth=2,min_samples_leaf=40,
        learning_rate=.05,l2_regularization=2,early_stopping=False,random_state=42)
    model.fit(fit[columns],path_classes(fit.target))
    slot=int(np.flatnonzero(model.classes_==2)[0])
    threshold,info=select_cutoff(model.predict_proba(validation[columns])[:,slot],validation.y)
    info.update(fit_rows=len(fit),validation_start=str(validation_start),
                latest_fit_label_end=str(fit.label_end.max()),
                latest_validation_label_end=str(validation.label_end.max()))
    # Keep the same fitted model after validation; refitting would change its score scale.
    return (model.predict_proba(test[columns]),slot,threshold,model.classes_),info


def predict(frame,start):
    outputs=[]; audits=[]
    sector_columns=[f'sector_{t}' for t in SECTOR_ETFS]
    for ticker in SECTOR_ETFS:frame[f'sector_{ticker}']=frame.ticker.eq(ticker).astype(float)
    for trend in (1,-1):
        pool=frame.loc[frame.trend.eq(trend)].copy()
        evaluation=pool.loc[pool.date.ge(pd.Timestamp(start))]
        for quarter,test in evaluation.groupby(evaluation.date.dt.to_period('Q')):
            cutoff=test.date.min()
            train=training_rows(pool,cutoff).dropna(subset=['y'])
            if train.date.nunique()<104 or train.y.nunique()!=2:
                audits.append({'trend':trend,'quarter':str(quarter),'status':'SKIP_INSUFFICIENT_TRAINING','training_weeks':train.date.nunique()})
                continue
            assert train.label_end.max()<cutoff
            result=test[['date','ticker','trend','target','y','label_end','rrg_event',
                         'trade_return20','potential_upside20','potential_downside20']].copy()
            result['refit_date']=cutoff
            result['latest_train_label_end']=train.label_end.max()
            result['train_base_rate']=float(train.y.mean())
            result['RRG']=result.rrg_event.eq(-trend)
            result['NO_ALERT']=False
            for name,columns in MODELS.items():
                cols=columns+sector_columns
                if name.endswith('_BOOST'):
                    if name.startswith('FRESH_'):
                        prior=[r for r in outputs if r.trend.iloc[0]==trend]
                        history=pd.concat(prior,ignore_index=True) if prior else result.iloc[:0].assign(**{name+'_score':np.nan})
                        candidate,info=fit_fresh_candidate(train,test,cols,history,name,cutoff)
                    else:
                        candidate,info=fit_path_candidate(train,test,cols)
                    audits.append({'trend':trend,'quarter':str(quarter),'model':name,**info})
                    result[name]=False
                    result[name+'_score']=np.nan
                    result[name+'_cutoff']=np.nan
                    result[name+'_status']=info['status']
                    if candidate is not None:
                        probabilities,slot,threshold,classes=candidate
                        result[name+'_score']=probabilities[:,slot]
                        for value,label in ((0,'uncertain'),(1,'continuation')):
                            result[name+'_'+label]=probabilities[:,int(np.flatnonzero(classes==value)[0])]
                        if threshold is not None:
                            result[name+'_cutoff']=threshold
                            result[name]=result[name+'_score'].ge(threshold)
                    continue
                model=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=2000,random_state=42))
                model.fit(train[cols],train.y.astype(int))
                # Fixed cutoff, no test-period tuning or retrospective percentile mapping.
                result[name+'_score']=model.predict_proba(test[cols])[:,1]
                result[name]=result[name+'_score'].ge(.5)
            outputs.append(result)
            audits.append({'trend':trend,'quarter':str(quarter),'status':'FIT','training_weeks':train.date.nunique(),
                           'training_rows':len(train),'latest_label_end':str(train.label_end.max()),'cutoff':str(cutoff)})
    if not outputs:raise ValueError('no eligible walk-forward folds')
    return pd.concat(outputs,ignore_index=True),audits


def metrics(predictions):
    records=[]
    for trend,pool in predictions.groupby('trend'):
        mature=pool.loc[pool.y.notna()]
        periods={'ALL':mature,**{str(y):g for y,g in mature.groupby(mature.date.dt.year)}}
        for period,rows in periods.items():
            for name in ['NO_ALERT','RRG',*MODELS]:
                selected=rows.loc[rows[name]]; positives=int(rows.y.sum())
                continuing=rows.target.eq('UP_CONTINUES' if trend==1 else 'DOWN_CONTINUES')
                scored=rows.loc[rows[name+'_score'].notna()] if name in MODELS else rows.iloc[:0]
                records.append(dict(direction='OUT' if trend==1 else 'IN',period=period,policy=name,
                    eligible_days=len(rows),positive_days=positives,alerts=len(selected),
                    score_coverage=len(scored) if name in MODELS else None,
                    abstained_days=int(rows[name+'_status'].str.startswith('ABSTAIN').sum()) if name+'_status' in rows else 0,
                    false_discovery_fraction=float(1-selected.y.mean()) if len(selected) else None,
                    positive_day_capture=float(selected.y.sum()/positives) if positives else None,
                    continuation_false_alarm_rate=float(rows.loc[continuing,name].mean()) if continuing.any() else None,
                    mean_trade_signed_return=float((selected.trade_return20*(-trend)).mean()) if selected.trade_return20.notna().any() else None,
                    mean_potential_missed_upside_OUT=float(selected.potential_upside20.clip(lower=0).mean()) if trend==1 and selected.potential_upside20.notna().any() else None,
                    mean_potential_further_drop_IN=float((-selected.potential_downside20).clip(lower=0).mean()) if trend==-1 and selected.potential_downside20.notna().any() else None,
                    roc_auc=float(roc_auc_score(scored.y,scored[name+'_score'])) if scored.y.nunique()==2 else None,
                    pr_auc=float(average_precision_score(scored.y,scored[name+'_score'])) if scored.y.nunique()==2 else None))
    return pd.DataFrame(records)


def score_diagnostics(predictions):
    """Fixed score bins on held-out predictions; diagnostic only, never tunes alerts."""
    records=[]
    for trend,pool in predictions.groupby('trend'):
        for name in MODELS:
            rows=pool.loc[pool.y.notna()&pool[name+'_score'].notna()].copy()
            rows['score']=rows[name+'_score']
            if not rows.score.between(0,1).all():raise ValueError('invalid prediction probability')
            rows['score_band']=pd.cut(rows.score,[0,.2,.4,.6,.8,1],include_lowest=True)
            for band,group in rows.groupby('score_band',observed=True):
                records.append(dict(direction='OUT' if trend==1 else 'IN',policy=name,
                    score_band=str(band),samples=len(group),mean_score=group.score.mean(),
                    realized_reversal_rate=group.y.mean(),
                    brier=((group.score-group.y)**2).mean(),
                    train_base_brier=((group.train_base_rate-group.y)**2).mean()))
    return pd.DataFrame(records)


def self_test():
    from sector_absolute_turn_labels import self_test as label_test
    label_test()
    dates=pd.bdate_range('2020-01-01',periods=180)
    prices=pd.DataFrame({t:100*np.exp(np.arange(180)*.001+np.sin(np.arange(180)/8)*.02) for t in ['SPY',*SECTOR_ETFS]},index=dates)
    a=features(prices)
    prices.iloc[-1]*=2
    b=features(prices)
    cols=ABS_FEATURES+RRG_FEATURES+PATH_FEATURES+TREND_FEATURES+SHIFT_FEATURES
    pd.testing.assert_frame_equal(a.loc[a.date<dates[-1],cols],b.loc[b.date<dates[-1],cols])
    scaled=features(prices*10)
    np.testing.assert_allclose(b[TREND_FEATURES],scaled[TREND_FEATURES],atol=1e-10,equal_nan=True)
    xlk=b.loc[b.ticker.eq('XLK')].set_index('date')
    assert xlk.ma_gap_long.iloc[:119].isna().all()
    assert xlk[TREND_FEATURES].iloc[-1].notna().all()
    assert set(MODELS['TREND_BOOST'])-set(MODELS['PATH_BOOST'])==set(TREND_FEATURES)
    np.testing.assert_allclose(b[SHIFT_FEATURES],scaled[SHIFT_FEATURES],atol=1e-9,equal_nan=True)
    sequence=np.array([2.,2.,2.,2.,2.,np.nan,-2.,-2.])
    states=cumulative_shift(sequence)
    np.testing.assert_allclose(states[:5,0],[1.5,3.,4.5,6.,1.5])
    assert np.isnan(states[5]).all() and states[6,1]==1.5
    np.testing.assert_allclose(cumulative_shift(sequence[:4]),states[:4])
    np.testing.assert_allclose(cumulative_shift(-sequence),states[:,::-1],equal_nan=True)
    toy=pd.DataFrame({'date':pd.to_datetime(['2020-01-01']*3),
                      'label_end':pd.to_datetime(['2020-02-01','2020-02-02',None])})
    assert len(training_rows(toy,pd.Timestamp('2020-02-02')))==1
    sample=pd.DataFrame({'trend':[1]*3,'date':pd.to_datetime(['2024-01-05']*3),'target':['TOP_END','UP_CONTINUES','UNCERTAIN'],
                         'y':[1.,0.,0.],'trade_return20':[-.1,.1,.01],'potential_upside20':[0,.1,.01],
                         'potential_downside20':[-.1,0,0],'RRG':[True,True,False],'NO_ALERT':[False]*3})
    for name in MODELS:sample[name]=[True,False,False];sample[name+'_score']=[.8,.2,.1]
    sample['train_base_rate']=.3
    diagnostic=score_diagnostics(sample)
    assert diagnostic.groupby('policy').samples.sum().eq(3).all()
    assert np.isclose((diagnostic.loc[diagnostic.policy.eq('ABS_LOGIT'),'brier']*
                       diagnostic.loc[diagnostic.policy.eq('ABS_LOGIT'),'samples']).sum()/3,.03)
    result=metrics(sample)
    assert result.loc[result.policy.eq('NO_ALERT'),'false_discovery_fraction'].isna().all()
    assert result.loc[result.policy.eq('RRG'),'false_discovery_fraction'].eq(.5).all()
    assert result.loc[result.policy.eq('ABS_LOGIT'),'positive_day_capture'].eq(1).all()
    assert path_classes(pd.Series(['UNCERTAIN','UP_CONTINUES','TOP_END'])).tolist()==[0,1,2]
    assert select_cutoff(np.full(50,.1),np.r_[np.ones(10),np.zeros(40)])[0] is None
    scores=np.r_[np.full(15,.7),np.full(35,.1)]
    truth=np.r_[np.ones(15),np.zeros(35)]
    threshold,info=select_cutoff(scores,truth)
    assert threshold is not None and info['precision']==1
    assert len(info['cutoff_grid'])==4
    assert select_cutoff([],[])[0] is None
    try:select_cutoff([.2],[.5])
    except ValueError:pass
    else:raise AssertionError('fractional label accepted')
    # Small synthetic fit verifies class mapping and chronological validation isolation.
    rng=np.random.default_rng(7)
    dates=pd.bdate_range('2010-01-01',periods=900)[::5]
    train=pd.DataFrame({'date':dates,'label_end':dates+pd.Timedelta(days=28),
                         'target':np.resize(['UNCERTAIN','UP_CONTINUES','TOP_END'],len(dates)),
                         'x':rng.normal(size=len(dates))})
    train['y']=train.target.eq('TOP_END').astype(int)
    test=pd.DataFrame({'x':[0.,1.]})
    candidate,info=fit_path_candidate(train,test,['x'])
    assert candidate is not None and candidate[0].shape==(2,3)
    assert pd.Timestamp(info['latest_fit_label_end'])<pd.Timestamp(info['validation_start'])
    assert np.allclose(candidate[0].sum(axis=1),1)
    cutoff=train.label_end.max()+pd.Timedelta(days=1)
    history=train.copy();history['FRESH_TREND_BOOST_score']=.2
    fresh,info=fit_fresh_candidate(train,test,['x'],history,'FRESH_TREND_BOOST',cutoff)
    assert pd.Timestamp(info['latest_fit_label_end'])==train.label_end.max()
    future=history.iloc[-1:].copy();future['date']=cutoff;future['label_end']=cutoff+pd.Timedelta(days=28)
    future['FRESH_TREND_BOOST_score']=1.;future['y']=1
    again,check=fit_fresh_candidate(train,test,['x'],pd.concat([history,future]),'FRESH_TREND_BOOST',cutoff)
    assert info==check
    np.testing.assert_allclose(fresh[0],again[0])
    print('prediction self-test PASS: causal features, strict label purge, metrics, zero-alert handling')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database',type=Path,default=ROOT/'regime_router_publish.db')
    parser.add_argument('--labels',type=Path,default=ROOT/'outputs/absolute_turn_labels_20260921/labels.csv')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/absolute_turn_prediction_v5_20260921')
    parser.add_argument('--start',default='2020-01-01')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:self_test()
    else:
        if args.output_dir.exists():raise FileExistsError(args.output_dir)
        frame=load(args); predictions,folds=predict(frame,args.start); result=metrics(predictions)
        args.output_dir.mkdir(parents=True,exist_ok=False)
        predictions.to_csv(args.output_dir/'predictions.csv',index=False)
        result.to_csv(args.output_dir/'metrics.csv',index=False)
        score_diagnostics(predictions).to_csv(args.output_dir/'score_diagnostics.csv',index=False)
        (args.output_dir/'audit.json').write_text(json.dumps({'folds':folds,'feature_sets':MODELS,
            'cutoff':'logit .5; boosts validation-only .30/.40/.50/.60 or abstain',
            'fresh_ablation':'FRESH arms fit all labels matured before the quarter. Threshold screening uses only earlier OOS predictions matured before cutoff, trailing 2 calendar years, minimum 52 observed weeks. Initial quarters score but abstain. Same learner/features/labels; jointly changes fit recency and validation source.',
            'shift_ablation':'TREND_BOOST vs SHIFT_BOOST: only two CUSUM inputs added; standardized innovations use previous 60-session mean/std; k=.5, reset h=5 fixed, not optimized. No alert veto.',
            'trend_ablation':'PATH_BOOST vs TREND_BOOST: identical learner, dates and validation rule; added volatility-scaled MA10/40/120 gaps, MA40 slope and short-gap change. Fixed windows, not optimized.',
            'models':'original logits plus depth-2 three-class boost; past 26 observed weeks validation, purged fit; scores uncalibrated',
            'limits':['20-session reversals, NOT established structural tops/bottoms',
                      'UNCERTAIN negative for original logits, separate class for boosts; evaluation still counts it as not-reversal',
                      'original boosts retain pre-validation fit; FRESH scores from different quarterly fits may drift and are not calibrated probabilities',
                      'validation Wilson bound is a heuristic guard, not independent-sample statistical proof',
                      'weekly sample capture is NOT unique-event or pre-pivot recall',
                      'same dates and at-risk absolute trend states for all policies; not full unconstrained RRG performance',
                      'overlapping targets, no IID significance claim or promotion gate',
                      'current adjusted-price vintage, PIT unverified; historically reviewed period',
                      'no execution/cost model; potential missed upside is not strategy P&L'],
            'operational_changes':False},ensure_ascii=False,indent=2),encoding='utf-8')
        print(result.loc[result.period.eq('ALL')].to_string(index=False))
