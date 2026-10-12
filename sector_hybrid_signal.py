"""Local frozen IN/BOOST + OUT/EXTRA artifact export and price-only inference.

Export explicitly fits two selected historical folds. Snapshot NEVER fits or downloads.
Joblib artifacts must be locally generated and trusted; a checksum is not authentication.
"""
import argparse
import hashlib
import inspect
import json
import os
import pickle
import sqlite3
import tempfile
from datetime import datetime, timezone
from contextlib import closing
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from sector_absolute_turn_prediction import features, cumulative_shift, ABS_FEATURES, PATH_FEATURES, TREND_FEATURES, RRG_FEATURES
from sector_rotation import SECTOR_ETFS, build_rrg_signals, _rolling_zscore

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'outputs/variable_barrier_comparison_20260921'
DEFAULT_BUNDLE=ROOT/'outputs/sector_hybrid_models/2026'
PRICE_COLUMNS=ABS_FEATURES+PATH_FEATURES+TREND_FEATURES+RRG_FEATURES
COLUMNS=PRICE_COLUMNS+['sector_'+ticker for ticker in SECTOR_ETFS]
ROUTES={'IN':'BOOST','OUT':'EXTRA'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def feature_signature():
    source=''.join(inspect.getsource(fn) for fn in (features,cumulative_shift,build_rrg_signals,_rolling_zscore))
    return hashlib.sha256(source.encode()).hexdigest()


def write_json(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as stream:
            json.dump(payload,stream,ensure_ascii=False,allow_nan=False,indent=2)
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def export_bundle(source,destination,year):
    """Reproduce the selected evaluated year; refuse a silent new training specification."""
    from sector_variable_barrier_comparison import match_weights, make_candidate
    source=Path(source);destination=Path(destination)
    if destination.exists():raise FileExistsError(destination)
    frame=pd.read_csv(source/'research_frame.csv',parse_dates=['date','label_end'])
    previous=pd.read_csv(source/'predictions.csv',parse_dates=['date'])
    if frame.duplicated(['date','ticker']).any():raise ValueError('Duplicate research rows')
    saved_columns=json.loads((source/'audit.json').read_text(encoding='utf-8'))['sets']['PRICE']
    if saved_columns!=COLUMNS:raise ValueError('Research feature schema changed')
    validation_start=pd.Timestamp(year-2,1,1);start=pd.Timestamp(year,1,1)
    packs={};meta={}
    for direction,family in ROUTES.items():
        part=frame.loc[frame.direction.eq(direction)]
        fit=part.loc[part.label_end.lt(validation_start)].reset_index(drop=True)
        val=part.loc[part.date.ge(validation_start)&part.label_end.lt(start)]
        test=part.loc[part.date.dt.year.eq(year)]
        if len(fit)<100 or fit.y.nunique()!=2 or val.y.nunique()!=2 or test.empty:
            raise ValueError('Insufficient evaluated fold; run a separate approved study first')
        if any(not np.isfinite(g[COLUMNS].to_numpy()).all() for g in (fit,val,test)):
            raise ValueError('Invalid research features')
        if not fit.y.isin([0,1]).all() or not val.y.isin([0,1]).all():raise ValueError('Invalid labels')
        if not ((fit.date<fit.label_end).all() and (val.date<val.label_end).all()):
            raise ValueError('Invalid label maturity dates')
        weights,_=match_weights(fit);scaler=StandardScaler().fit(fit[COLUMNS])
        model=make_candidate(family)
        with threadpool_limits(limits=2):
            model.fit(scaler.transform(fit[COLUMNS]),fit.y,sample_weight=weights)
            threshold=float(np.quantile(model.predict_proba(scaler.transform(val[COLUMNS]))[:,1],.90))
            scores=model.predict_proba(scaler.transform(test[COLUMNS]))[:,1]
        check=test[['date','ticker']].copy();check['replayed']=scores
        old=previous.loc[previous.direction.eq(direction)&previous.model.eq('PRICE_'+family)&previous.year.eq(year)]
        merged=check.merge(old,on=['date','ticker'],validate='one_to_one')
        if len(merged)!=len(test) or len(old)!=len(test) or not np.allclose(merged.score,merged.replayed,rtol=1e-9,atol=1e-10):
            raise ValueError('Saved prediction parity failed; no bundle published')
        if not np.allclose(merged.threshold,threshold,rtol=1e-9,atol=1e-10):
            raise ValueError('Saved threshold parity failed')
        packs[direction]={'model':model,'scaler':scaler,'threshold':threshold}
        meta[direction]={'family':family,'threshold':threshold,'fit_rows':len(fit),'validation_rows':len(val),
            'fit_label_end':str(fit.label_end.max().date()),'validation_label_end':str(val.label_end.max().date()),
            'prediction_parity_rows':len(merged),'max_score_error':float(abs(merged.score-merged.replayed).max())}
    destination.mkdir(parents=True)
    for direction,pack in packs.items():
        path=destination/(direction+'.joblib');joblib.dump(pack,path)
        meta[direction]['sha256']=digest(path)
    manifest={'schema':1,'year':year,'columns':COLUMNS,'sklearn':sklearn.__version__,
        'feature_signature':feature_signature(),'created_at':datetime.now(timezone.utc).isoformat(),
        'research_frame_sha256':digest(source/'research_frame.csv'),'models':meta,
        'cadence':'WEEKLY_COMPLETED_WEEK','score_semantics':'weighted score, not calibrated probability',
        'selection':'post-research selected hybrid; not independently validated',
        'inference_scope':'price-only completeness; research evaluation used full-input intersection'}
    write_json(destination/'manifest.json',manifest)
    return manifest


def load_bundle(directory):
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if manifest['schema']!=1 or manifest['columns']!=COLUMNS:raise ValueError('Model schema mismatch')
    if manifest['sklearn']!=sklearn.__version__:raise ValueError('Scikit-learn version mismatch')
    if manifest['feature_signature']!=feature_signature():raise ValueError('Feature implementation changed')
    year=int(manifest['year'])
    packs={}
    for direction in ROUTES:
        path=directory/(direction+'.joblib');description=manifest['models'][direction]
        if (pd.Timestamp(description['fit_label_end'])>=pd.Timestamp(year-2,1,1) or
            pd.Timestamp(description['validation_label_end'])>=pd.Timestamp(year,1,1)):
            raise ValueError('Model maturity boundary mismatch')
        if description['family']!=ROUTES[direction] or digest(path)!=description['sha256']:
            raise ValueError('Model integrity mismatch')
        # Only load artifacts generated locally by export_bundle, never downloaded pickles.
        pack=joblib.load(path)
        if not np.isfinite(pack['threshold']) or not 0<=pack['threshold']<=1:
            raise ValueError('Invalid threshold')
        if pack['threshold']!=description['threshold']:raise ValueError('Threshold mismatch')
        packs[direction]=pack
    return manifest,packs


def completed_week(calendar,as_of):
    dates=pd.Series(pd.DatetimeIndex(calendar),index=pd.DatetimeIndex(calendar))
    dates=dates.loc[dates<=as_of]
    if dates.empty:return None
    groups=dates.groupby(dates.index.to_period('W-FRI')).max()
    groups=groups.loc[groups.index.end_time.normalize()<=as_of]
    return None if groups.empty else pd.Timestamp(groups.iloc[-1])


def score_rows(frame,packs):
    result=[]
    for ticker,name in SECTOR_ETFS.items():
        selected=frame.loc[frame.ticker.eq(ticker)]
        row={'ticker':ticker,'sector':name,'signal':None,'score':None,'threshold':None,'direction':None}
        if len(selected)!=1 or not np.isfinite(selected[PRICE_COLUMNS].to_numpy()).all():
            row['status']='MISSING_FEATURES';result.append(row);continue
        values=selected.copy()
        direction='OUT' if float(values.iloc[0].ret60)>0 else 'IN'
        for sector in SECTOR_ETFS:values['sector_'+sector]=float(sector==ticker)
        pack=packs[direction]
        with threadpool_limits(limits=2):score=float(pack['model'].predict_proba(pack['scaler'].transform(values[COLUMNS]))[0,1])
        if not np.isfinite(score) or not 0<=score<=1:raise ValueError('Invalid prediction')
        row.update(status='READY',direction=direction,model='PRICE_'+ROUTES[direction],score=score,
            threshold=pack['threshold'],signal=direction if score>=pack['threshold'] else 'NONE')
        result.append(row)
    return result


def build_snapshot(database,bundle=DEFAULT_BUNDLE,as_of=None):
    requested=pd.Timestamp(as_of or datetime.now().date()).normalize()
    if pd.isna(requested):raise ValueError('Invalid as-of date')
    base={'schema':1,'requested_as_of':str(requested.date()),'cadence':'WEEKLY_COMPLETED_WEEK',
        'generated_at':datetime.now(timezone.utc).isoformat(),'rows':[],
        'note':'IN=price boosting; OUT=price Extra Trees. Weighted scores, not probabilities. No automatic retraining.'}
    if not (Path(bundle)/'manifest.json').exists():return {**base,'status':'MODEL_NOT_READY'}
    manifest,packs=load_bundle(bundle)
    with closing(sqlite3.connect(Path(database).resolve().as_uri()+'?mode=ro',uri=True)) as con:
        names=['SPY',*SECTOR_ETFS]
        raw=pd.read_sql_query('SELECT date,ticker,adj_close FROM daily_prices WHERE date<=? AND ticker IN ('+
            ','.join('?' for _ in names)+')',con,params=[str(requested.date()),*names],parse_dates=['date'])
    if raw.duplicated(['date','ticker']).any():raise ValueError('Duplicate price keys')
    if raw.empty:return {**base,'status':'NO_PRICES'}
    prices=raw.pivot(index='date',columns='ticker',values='adj_close').sort_index().reindex(columns=names)
    observed=prices.stack().dropna()
    if not np.isfinite(observed).all() or observed.le(0).any():raise ValueError('Invalid price values')
    prices=prices.loc[prices.SPY.notna()]
    if prices.empty:return {**base,'status':'NO_BENCHMARK'}
    # Conservative holiday handling: only weeks closed by the latest observed SPY session.
    decision=completed_week(prices.index,min(requested,prices.index[-1]))
    if decision is None:return {**base,'status':'NO_COMPLETED_WEEK'}
    base.update(as_of=str(decision.date()),latest_price_date=str(prices.index[-1].date()),model_year=manifest['year'])
    if (requested-decision).days>10:return {**base,'status':'STALE_PRICES'}
    if decision.year!=manifest['year']:return {**base,'status':'MODEL_YEAR_MISMATCH'}
    # Truncate BEFORE building features, so even an as-of request cannot see later prices.
    prices=prices.loc[:decision]
    if len(prices)<150:return {**base,'status':'INSUFFICIENT_HISTORY'}
    frame=features(prices)
    rows=score_rows(frame.loc[frame.date.eq(decision)],packs)
    return {**base,'status':'READY' if all(r['status']=='READY' for r in rows) else 'PARTIAL','rows':rows}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    export=commands.add_parser('export',help='Explicit two-model refit and parity-checked persistence')
    export.add_argument('--source',type=Path,default=SOURCE);export.add_argument('--bundle',type=Path,default=DEFAULT_BUNDLE)
    export.add_argument('--year',type=int,default=2026)
    snap=commands.add_parser('snapshot',help='No fit/download; local JSON only')
    snap.add_argument('--database',type=Path,default=ROOT/'regime_router_publish.db')
    snap.add_argument('--bundle',type=Path,default=DEFAULT_BUNDLE);snap.add_argument('--as-of')
    snap.add_argument('--output',type=Path,default=ROOT/'outputs/sector_hybrid_latest.json')
    args=parser.parse_args()
    if args.command=='export':
        result=export_bundle(args.source,args.bundle,args.year)
        print(json.dumps({'status':'EXPORTED','bundle':str(args.bundle),'year':result['year']}))
    else:
        try:result=build_snapshot(args.database,args.bundle,args.as_of)
        except (ValueError,KeyError,TypeError,AttributeError,EOFError,pickle.UnpicklingError,OSError,sqlite3.Error) as error:
            result={'status':'ERROR','error_type':type(error).__name__,'message':str(error),'rows':[]}
        write_json(args.output,result)
        print(json.dumps({'status':result['status'],'output':str(args.output)}))
        if result['status'] not in ('READY','PARTIAL'):raise SystemExit(2)


if __name__=='__main__':main()
