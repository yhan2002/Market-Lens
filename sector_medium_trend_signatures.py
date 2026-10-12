"""Local descriptive event study, not a predictor: 63/126-session trend signatures."""
import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sector_rotation import SECTOR_ETFS

ROOT=Path(__file__).resolve().parent
FACTORS=['MTUM','QUAL','VLUE','USMV']
SPLIT=pd.Timestamp('2023-01-01')
OFFSETS=(-20,-10,0)


def signatures(prices,ticker):
    """Inputs available at each close; no future values, fills or full-sample scaling."""
    logs=np.log(prices)
    relative={f'{t}_SPY':logs[t]-logs.SPY for t in FACTORS}
    relative.update(MTUM_VLUE=logs.MTUM-logs.VLUE,QUAL_USMV=logs.QUAL-logs.USMV,
                    SECTOR_SPY=logs[ticker]-logs.SPY)
    return pd.DataFrame({f'{name}_{h}':v.diff(h) for name,v in relative.items() for h in (20,63)})


def outcome(path,prior,vol):
    """A broad reversal/continuation window, never an exact historical extremum."""
    if len(path)!=126 or not np.isfinite(path).all():return 'IMMATURE'
    if not np.isfinite(prior) or not np.isfinite(vol) or vol<=0:return 'NO_TREND'
    if abs(prior)<max(.025,.5*vol*np.sqrt(63)):return 'NO_TREND'
    direction=1 if prior>0 else -1
    barriers=np.maximum([.05,.08],.75*vol*np.sqrt([63,126]))
    for sign,label in [(-direction,'REVERSAL'),(direction,'CONTINUATION')]:
        signed=path*sign
        if (signed[[62,125]]>=barriers).all() and (signed>0).mean()>=.60:
            return label
    return 'AMBIGUOUS'


def load_prices():
    factor_path=ROOT/'output/analysis/leading_research_20260909/J_msci_proxy_adjusted.csv'
    factor=pd.read_csv(factor_path,parse_dates=['Date']).set_index('Date')[FACTORS]
    if not factor.index.is_unique:raise ValueError('duplicate factor dates')
    with sqlite3.connect((ROOT/'regime_router_publish.db').as_uri()+'?mode=ro',uri=True) as con:
        names=['SPY',*SECTOR_ETFS]
        raw=pd.read_sql_query('SELECT date,ticker,adj_close FROM daily_prices WHERE ticker IN ('+
                             ','.join('?' for _ in names)+')',con,params=names,parse_dates=['date'])
    if raw.duplicated(['date','ticker']).any():raise ValueError('duplicate sector prices')
    prices=raw.pivot(index='date',columns='ticker',values='adj_close').sort_index()
    prices=prices.loc[prices.SPY.notna()].join(factor)
    if (prices<=0).any().any() or np.isinf(prices.to_numpy()).any():raise ValueError('invalid prices')
    return prices,factor


def run(output):
    if output.exists():raise FileExistsError(output)
    prices,factor=load_prices()
    # Preserve exchange sessions and missing values; never compress horizons across gaps.
    calendar=prices.index;dates=pd.Series(calendar,index=calendar)
    weekly=set(dates.groupby(calendar.to_period('W-FRI')).max())
    records=[];pairs=[];contrasts=[];coverage=[]
    for ticker in SECTOR_ETFS:
        log=np.log(prices[ticker]);ret=log.diff(63);vol=log.diff().shift(1).rolling(63).std()
        feat=signatures(prices,ticker)
        scale=feat.loc[feat.index<SPLIT].std().replace(0,np.nan)
        rows=[]
        for pos,date in enumerate(calendar):
            if date not in weekly or pos<84:continue
            path=(log.iloc[pos+1:pos+127]-log.iloc[pos]).to_numpy()
            label=outcome(path,ret.iloc[pos],vol.iloc[pos])
            label_end=calendar[pos+126] if pos+126<len(calendar) else pd.NaT
            partition='IS' if pd.notna(label_end) and label_end<SPLIT else ('REVIEWED_LATER' if date>=SPLIT else 'PURGED')
            usable=all(feat.iloc[pos+offset].notna().all() for offset in OFFSETS)
            rows.append(dict(sector=ticker,date=date,position=pos,target=label,prior63=ret.iloc[pos],
                vol63=vol.iloc[pos],direction='OUT' if ret.iloc[pos]>0 else 'IN',
                label_end=label_end,partition=partition,features_complete=usable,
                future63=np.expm1(path[62]) if len(path)>62 else np.nan,
                future126=np.expm1(path[125]) if len(path)==126 else np.nan))
        frame=pd.DataFrame(rows);records.append(frame)
        history=frame.loc[frame.partition.eq('IS')]
        sd=history.prior63.std()
        for partition in ('IS','REVIEWED_LATER'):
            pool=frame.loc[frame.partition.eq(partition)&frame.features_complete].copy()
            for direction in ('IN','OUT'):
                candidates=pool.loc[pool.direction.eq(direction)&pool.target.eq('REVERSAL')]
                controls=pool.loc[pool.direction.eq(direction)&pool.target.eq('CONTINUATION')]
                last=-1000;used=[];event_count=0;matched=0
                for case in candidates.itertuples():
                    if case.position-last<=126:continue
                    last=case.position;event_count+=1
                    controls_ok=controls.loc[(controls.position-case.position).abs().between(127,504)].copy()
                    for position in used:
                        controls_ok=controls_ok.loc[(controls_ok.position-position).abs()>126]
                    if controls_ok.empty or not np.isfinite(sd) or sd<=0:continue
                    dr=(controls_ok.prior63-case.prior63).abs()/sd
                    dv=np.log(controls_ok.vol63/case.vol63).abs()
                    eligible=(dr<=1)&(dv<=.5)
                    if not eligible.any():continue
                    cost=(dr+dv/.5).loc[eligible]
                    control=controls_ok.loc[cost.idxmin()];used.append(int(control.position));matched+=1
                    pair_id=f'{ticker}_{direction}_{case.date:%Y%m%d}'
                    pairs.append(dict(pair_id=pair_id,sector=ticker,direction=direction,partition=partition,
                        case_date=case.date,control_date=control.date,case_end=case.label_end,control_end=control.label_end,
                        case_prior63=case.prior63,control_prior63=control.prior63,
                        case_vol63=case.vol63,control_vol63=control.vol63,match_distance=float(cost.min())))
                    for offset in OFFSETS:
                        a=feat.iloc[case.position+offset];b=feat.iloc[int(control.position)+offset]
                        for col in feat:
                            contrasts.append(dict(pair_id=pair_id,sector=ticker,direction=direction,partition=partition,
                                case_date=case.date,offset=offset,feature=col,case_value=a[col],control_value=b[col],
                                difference=a[col]-b[col],standardized_difference=(a[col]-b[col])/scale[col]))
                coverage.append(dict(sector=ticker,direction=direction,partition=partition,
                    candidate_weeks=len(candidates),spaced_events=event_count,matched_events=matched))
    detail=pd.DataFrame(contrasts)
    output.mkdir(parents=True)
    pd.concat(records).to_csv(output/'label_candidates.csv',index=False)
    pd.DataFrame(coverage).to_csv(output/'matching_coverage.csv',index=False)
    pd.DataFrame(pairs).to_csv(output/'matched_events.csv',index=False)
    detail.to_csv(output/'factor_contrasts.csv',index=False)
    if not detail.empty:
        summaries=[]
        for sector,sub in [('ALL',detail),*list(detail.groupby('sector'))]:
            g=sub.groupby(['partition','direction','offset','feature'])
            table=g.agg(pairs=('pair_id','nunique'),mean_case=('case_value','mean'),mean_control=('control_value','mean'),
                mean_difference=('difference','mean'),median_difference=('difference','median'),
                mean_standardized_difference=('standardized_difference','mean'),
                positive_fraction=('difference',lambda s:float((s>0).mean())))
            summaries.append(table.reset_index().assign(sector=sector))
        pd.concat(summaries).to_csv(output/'signature_summary.csv',index=False)
    meta=dict(horizons=[63,126],offsets=list(OFFSETS),thresholds_log_floor=[.05,.08],
        volatility_multiplier=.75,positive_path_fraction=.60,case_spacing_sessions=126,
        controls='same sector, prior direction, partition; 127..504 sessions away; prior63 caliper1 IS SD, log-vol ratio caliper0.5',
        factors=FACTORS,factor_last={c:str(factor[c].last_valid_index()) for c in FACTORS},
        labels='retrospective broad trend windows, not exact pivots, not predictions',
        inference='descriptive contrasts only; no significance claim, no model fitting',
        limitations=['hand-set label thresholds need visual review before modeling',
            'strict sustained paths exclude choppy/ambiguous reversals; unmatched events reported',
            '126-session same-direction event thinning can miss closely spaced regimes',
            'later period previously reviewed; not pristine OOS; control matching is retrospective',
            'cross-sector correlated events and imperfect matching; not causal',
            'current revised adjusted prices; factor ETFs are proxies, not pure factors'],
        operational_changes=False)
    (output/'audit.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(pd.DataFrame(coverage).groupby(['partition','direction'])[['spaced_events','matched_events']].sum().to_string())
    print(output)


def validation_masks(frame,year):
    cutoff=pd.Timestamp(year,1,1);validation_start=pd.Timestamp(year-1,1,1)
    return (frame.label_end.lt(validation_start),
            frame.date.ge(validation_start)&frame.label_end.lt(cutoff),
            frame.date.dt.year.eq(year))


def validation_screen(score,truth):
    from sector_absolute_turn_prediction import select_cutoff
    if len(truth)==0 or pd.Series(truth).nunique()<2:
        return None,dict(status='ABSTAIN_VALIDATION_SINGLE_CLASS',validation_rows=len(truth))
    return select_cutoff(score,truth)


def validate(output):
    """Prepare sequential predictions from all mature trend windows, not matched cases."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score,average_precision_score
    if output.exists():raise FileExistsError(output)
    source=ROOT/'outputs/medium_trend_signatures_20260921/label_candidates.csv'
    labels=pd.read_csv(source,parse_dates=['date','label_end'])
    if labels.duplicated(['sector','date']).any():raise ValueError('duplicate labels')
    prices,_=load_prices();logs=np.log(prices)
    parts=[]
    for ticker in SECTOR_ETFS:
        f=signatures(prices,ticker)
        f['SPY20']=logs.SPY.diff(20);f['SPY63']=logs.SPY.diff(63)
        f['check_prior63']=logs[ticker].diff(63)
        f['check_vol63']=logs[ticker].diff().shift(1).rolling(63).std()
        f['date']=f.index;f['sector']=ticker;parts.append(f.reset_index(drop=True))
    frame=labels.merge(pd.concat(parts),on=['date','sector'],validate='one_to_one')
    for saved,current in [('prior63','check_prior63'),('vol63','check_vol63')]:
        if not np.allclose(frame[saved],frame[current],equal_nan=True):
            raise ValueError('Price vintage differs from saved labels; stop before validation')
    frame=frame.loc[frame.target.isin(['REVERSAL','CONTINUATION','AMBIGUOUS'])].copy()
    frame['y']=frame.target.eq('REVERSAL').astype(int)
    base=['prior63','vol63','SPY20','SPY63','SECTOR_SPY_20']
    factors=[f'{t}_SPY_{h}' for t in FACTORS for h in (20,63)]
    sets={'PRICE_MARKET':base,'LOW_VOL':base+['USMV_SPY_63'],
          'LOW_VOL_PAIR':base+['USMV_SPY_63','QUAL_USMV_63'],
          'ALL_FACTORS':base+factors}
    required=list(dict.fromkeys(sum(sets.values(),[])))
    frame=frame.replace([np.inf,-np.inf],np.nan).dropna(subset=required+['label_end'])
    sector_cols=[]
    for ticker in SECTOR_ETFS:
        col='sector_'+ticker;frame[col]=frame.sector.eq(ticker).astype(int);sector_cols.append(col)
    predictions=[];audits=[];metrics=[]
    for direction in ('IN','OUT'):
        part=frame.loc[frame.direction.eq(direction)].sort_values(['date','sector'])
        for year in (2023,2024,2025,2026):
            fm,vm,tm=validation_masks(part,year)
            fit=part.loc[fm];validation=part.loc[vm];test=part.loc[tm]
            if test.empty:continue
            if fit.y.nunique()<2 or len(fit)<100:
                audits.append(dict(direction=direction,year=year,status='SKIP_INSUFFICIENT_CLASSES',
                    fit_rows=len(fit),validation_rows=len(validation),test_rows=len(test)))
                continue
            assert fit.label_end.max()<pd.Timestamp(year-1,1,1)
            if not validation.empty:assert validation.label_end.max()<test.date.min()
            for name,cols in sets.items():
                model=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=2000,
                    class_weight='balanced',random_state=42))
                model.fit(fit[cols+sector_cols],fit.y)
                validation_score=model.predict_proba(validation[cols+sector_cols])[:,1] if len(validation) else np.array([])
                threshold,screen=validation_screen(validation_score,validation.y)
                score=model.predict_proba(test[cols+sector_cols])[:,1]
                row=test[['date','sector','direction','target','y','label_end','future63','future126']].copy()
                row['model']=name;row['year']=year;row['score']=score
                row['threshold']=threshold if threshold is not None else np.nan
                row['alert']=score>=threshold if threshold is not None else False
                predictions.append(row)
                audits.append(dict(direction=direction,year=year,model=name,fit_rows=len(fit),
                    validation_rows=len(validation),test_rows=len(test),
                    latest_fit_label_end=str(fit.label_end.max()),validation_start=str(validation.date.min()),
                    latest_validation_label_end=str(validation.label_end.max()),test_start=str(test.date.min()),
                    threshold=threshold,screen=screen))
    if not predictions:raise ValueError('No eligible chronological folds')
    result=pd.concat(predictions,ignore_index=True)
    for (direction,model),group in result.groupby(['direction','model']):
        for period,g in [('ALL',group),*list(group.groupby('year'))]:
            selected=g.loc[g.alert];wrong=selected.loc[selected.y.eq(0)]
            metrics.append(dict(direction=direction,model=model,period=str(period),rows=len(g),
                positive_rows=int(g.y.sum()),base_rate=float(g.y.mean()),
                auc=float(roc_auc_score(g.y,g.score)) if g.y.nunique()==2 else None,
                pr_auc=float(average_precision_score(g.y,g.score)) if g.y.sum() else None,
                alerts=len(selected),precision=float(selected.y.mean()) if len(selected) else None,
                recall=float(selected.y.sum()/g.y.sum()) if g.y.sum() else None,
                fpr=float(len(wrong)/(g.y.eq(0).sum())) if g.y.eq(0).any() else None,
                wrong_continuation=int(wrong.target.eq('CONTINUATION').sum()),
                wrong_ambiguous=int(wrong.target.eq('AMBIGUOUS').sum()),
                mean_return63=float(selected.future63.mean()) if len(selected) else None,
                mean_return126=float(selected.future126.mean()) if len(selected) else None))
    output.mkdir(parents=True)
    result.to_csv(output/'predictions.csv',index=False)
    pd.DataFrame(metrics).to_csv(output/'metrics.csv',index=False)
    (output/'audit.json').write_text(json.dumps(dict(folds=audits,models=sets,
        method='annual chronological test; previous calendar-year mature validation; purged earlier fit; no refit after screening',
        limits=['126-session labels overlap; weekly rows are not independent events',
            'AMBIGUOUS means not the specified sustained reversal, not guaranteed trend continuation',
            'fixed existing validation heuristic, not a significance test; abstention is permitted',
            'candidate factors chosen after historical review; not pristine out-of-sample evidence',
            'current adjusted history; PIT unverified; no trading costs or execution backtest',
            'raw post-signal returns are descriptive, not strategy profits',
            'no alert debouncing or hold-state simulation; all weekly alerts retained'],
        operational_changes=False),indent=2),encoding='utf-8')
    print(pd.DataFrame(metrics).query("period=='ALL'").to_string(index=False))
    print(output)


def self_test():
    up=np.linspace(.002,.20,126)
    assert outcome(up,-.10,.01)=='REVERSAL'
    assert outcome(up,.10,.01)=='CONTINUATION'
    assert outcome(-up,.10,.01)=='REVERSAL'
    assert outcome(up[:100],-.1,.01)=='IMMATURE'
    ix=pd.bdate_range('2020-01-01',periods=200)
    p=pd.DataFrame({t:np.exp(np.arange(200)*(.001+i*.0001)) for i,t in enumerate(['SPY','XLK',*FACTORS])},index=ix)
    a=signatures(p,'XLK');p.iloc[150:]*=2
    pd.testing.assert_frame_equal(a.iloc[:150],signatures(p,'XLK').iloc[:150])
    p.loc[ix[100],'MTUM']=np.nan
    assert pd.isna(signatures(p,'XLK').loc[ix[100],'MTUM_SPY_20'])
    toy=pd.DataFrame({'date':pd.to_datetime(['2020-01-01','2021-12-01','2022-01-01','2022-08-01','2023-01-01']),
        'label_end':pd.to_datetime(['2020-07-01','2022-06-01','2022-07-01','2023-02-01','2023-07-01'])})
    fit,val,test=validation_masks(toy,2023)
    assert fit.tolist()==[True,False,False,False,False]
    assert val.tolist()==[False,False,True,False,False]
    assert test.tolist()==[False,False,False,False,True]
    threshold,screen=validation_screen([.2,.3],[0,0])
    assert threshold is None and screen['status']=='ABSTAIN_VALIDATION_SINGLE_CLASS'
    print('PASS: medium-term outcome examples, causal inputs, missing values preserved')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--validate',action='store_true')
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    if args.self_test:self_test()
    elif args.validate:validate(args.output_dir or ROOT/'outputs/medium_trend_factor_validation_20260921')
    else:run(args.output_dir or ROOT/'outputs/medium_trend_signatures_20260921')
