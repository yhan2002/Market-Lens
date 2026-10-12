"""Y: predeclared leadership-event forecasts; local cache, no trading or download."""
import argparse
import hashlib
import json
import sqlite3

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from research_msci_rotation import CACHE, ROOT, OUT, TICKERS, features, residual_features
from validate_sector_factor_explanation import SECTORS
from rotation_events import (HORIZON, PERSIST, leadership, events_from_leaders,
                             make_labels, calibrate_threshold, alerts_from_scores, match_alerts)

BASE = ['rel5', 'rel20', 'rel60', 'vol20', 'rank', 'rank_change5', 'gap3']
MODEL_NAMES = ['persistence', 'calendar', 'sector_rule', 'sector', 'raw', 'residual']


def build_panel(prices, factor):
    relative20, full_leaders = leadership(prices)
    first = relative20.dropna().index.min()
    leaders = full_leaders.loc[first:]
    events = events_from_leaders(leaders)
    positions = pd.Series(np.arange(len(leaders)), index=leaders.index)
    ranks = relative20.rank(axis=1, ascending=False, method='first')
    third = relative20.apply(lambda r: r.nlargest(3).min() if r.notna().all() else np.nan, axis=1)
    logp = np.log(prices)
    label_end = pd.Series(leaders.index,index=leaders.index).shift(-(HORIZON+PERSIST-1))
    rows = []
    for kind in ['entry','exit']:
        labels, eligible = make_labels(leaders,events,kind)
        for ticker in leaders:
            rel = logp[ticker]-logp.SPY
            x = pd.DataFrame({'rel5':rel.diff(5), 'rel20':rel.diff(20), 'rel60':rel.diff(60),
                              'vol20':logp[ticker].diff().rolling(20).std(),
                              'rank':ranks[ticker], 'rank_change5':ranks[ticker].shift(5)-ranks[ticker],
                              'gap3':relative20[ticker]-third})
            x = x.join(factor).reindex(leaders.index)
            x['y'], x['eligible'], x['end'], x['pos'] = labels[ticker],eligible[ticker],label_end,positions
            rows.append(x.assign(ticker=ticker,kind=kind))
    frame = pd.concat(rows).reset_index(names='date').replace([np.inf,-np.inf],np.nan)
    # All sector/kind rows must be complete before defining a shared date calendar.
    frame = frame.dropna()
    dates = frame.groupby('date').size().loc[lambda s:s == 2*len(leaders.columns)].index
    frame = frame[frame.date.isin(dates)]
    return frame, leaders, relative20.loc[leaders.index], events


def fit_forecast(train, calibration, test, models):
    cal, pred = calibration.copy(), test.copy()
    audit, coef_rows = [], []
    for kind in ['entry','exit']:
        for ticker in sorted(train.ticker.unique()):
            sub = train[train.kind.eq(kind)&train.ticker.eq(ticker)&train.eligible]
            counts = sub.y.value_counts()
            fallback = len(sub)<126 or counts.get(0.,0)<5 or counts.get(1.,0)<5
            prior = (sub.y.sum()+1)/(len(sub)+2)
            audit.append({'kind':kind,'ticker':ticker,'train_n':len(sub),
                          'positive_n':int(sub.y.sum()),'fallback_prior':bool(fallback)})
            for name,cols in models.items():
                model = None
                if not fallback:
                    model = make_pipeline(StandardScaler(), LogisticRegression(C=.1,max_iter=2000))
                    model.fit(sub[cols],sub.y)
                    if model[-1].n_iter_.max()>=2000:
                        raise ValueError('Logistic fit reached iteration limit')
                    for feature,coef in zip(cols,model[-1].coef_[0]):
                        coef_rows.append({'kind':kind,'ticker':ticker,'model':name,
                                          'feature':feature,'standardized_log_odds_coefficient':coef})
                for data in [cal,pred]:
                    mask = data.kind.eq(kind)&data.ticker.eq(ticker)&data.eligible
                    if not mask.any():
                        continue
                    data.loc[mask,name] = (prior if model is None else
                                          model.predict_proba(data.loc[mask,cols])[:,1])
    for data in [cal,pred]:
        direction = data.kind.map({'entry':1.,'exit':-1.})
        # Rank improvement towards entry, or deterioration towards exit; causal only.
        data['sector_rule'] = direction*data.rank_change5
        data.loc[~data.eligible,'sector_rule'] = np.nan
        data['calendar'] = data.sector_rule  # Nominee only; timing is overridden below.
        data['persistence'] = np.nan  # Explicit no-change/no-warning benchmark.
        assert np.isfinite(data.loc[data.eligible,list(models)].to_numpy()).all()
    return cal,pred,audit,coef_rows


def daily_scores(pred, kind, model):
    sub = pred[pred.kind.eq(kind)]
    daily = sub[['date','pos']].drop_duplicates().set_index('date').sort_index()
    eligible = sub[sub.eligible & sub[model].notna()]
    best = eligible.sort_values(['date',model,'ticker'],ascending=[True,False,True]).drop_duplicates('date')
    daily = daily.join(best.set_index('date')[[model,'ticker']].rename(columns={model:'score','ticker':'nominated'}))
    if model == 'calendar':
        daily['score'] = np.where(daily.pos.mod(42).eq(0),1.,np.nan)
    return daily


def metrics(matched, events, kind, daily, year=None):
    positions = set(daily.pos.astype(int))
    groups = events[events.kind.eq(kind)].groupby('onset_pos').ticker.apply(set)
    # Recall requires a complete preceding forecast window; precision uses every
    # warning with a fully matured forward window, including evaluation edges.
    recall_events = {int(p) for p in groups.index if set(range(int(p)-HORIZON,int(p))) <= positions}
    hit_events = set(matched.loc[matched.hit,'event_pos'].astype(int)) if len(matched) else set()
    covered = {p+k for p in matched.decision_pos for k in range(1,HORIZON+1)}
    if year is not None:
        year_events = set(events.loc[pd.to_datetime(events.onset_date).dt.year.eq(year),'onset_pos'].astype(int))
        recall_events &= year_events
        matched = matched[pd.to_datetime(matched.decision_date).dt.year.eq(year)]
        reporting_daily = daily[daily.index.year == year]
    else:
        reporting_daily = daily
    hit_count = int(matched.hit.sum()) if len(matched) else 0
    correct = int(matched.candidate_hit.sum()) if len(matched) else 0
    nominee_hits = int(matched.nominee_horizon_hit.sum()) if len(matched) else 0
    months = len(reporting_daily)/21.
    valid_event_dates = {p for p in range(min(positions)+1,max(positions)+2)
                         if set(range(p-HORIZON,p)) <= positions}
    if year is not None:
        # Yearly coverage reports evaluated dates, preserving the global alarm history.
        valid_event_dates &= set(reporting_daily.pos)
    leads = matched.loc[matched.hit,'lead_days'] if len(matched) else pd.Series(dtype=float)
    return {'kind':kind,'sessions':len(reporting_daily),'event_groups':len(recall_events),
            'alerts':len(matched),'hits':hit_count,'precision':hit_count/len(matched) if len(matched) else np.nan,
            'recall':len(hit_events&recall_events)/len(recall_events) if recall_events else np.nan,
            'candidate_precision':correct/len(matched) if len(matched) else np.nan,
            'candidate_accuracy_on_hits':correct/hit_count if hit_count else np.nan,
            'nominee_within10_precision':nominee_hits/len(matched) if len(matched) else np.nan,
            'median_lead_days':leads.median(),'alerts_per_month':len(matched)/months,
            'false_alerts_per_month':(len(matched)-hit_count)/months,
            'warning_window_coverage':len(covered&valid_event_dates)/len(valid_event_dates) if valid_event_dates else np.nan}


def run():
    with sqlite3.connect((ROOT/'regime_router_publish.db').resolve().as_uri()+'?mode=ro',uri=True) as c:
        tickers = ['SPY',*sorted(SECTORS)]
        rows = pd.read_sql_query('SELECT date,ticker,adj_close FROM daily_prices WHERE ticker IN ('+
                                ','.join('?' for _ in tickers)+") AND date<='2026-09-08' ORDER BY date",
                                c,params=tickers,parse_dates=['date'])
    prices = rows.pivot(index='date',columns='ticker',values='adj_close').dropna()
    assert prices.index.is_unique and prices.index.is_monotonic_increasing and (prices>0).all().all()
    proxy = pd.read_pickle(CACHE)['Adj Close'][TICKERS]
    joined = prices.join(proxy)
    factor = features(joined).join(residual_features(joined))
    frame,leaders,relative20,events = build_panel(prices,factor)
    rawcols = [c for c in factor if '_rel' in c]
    residcols = [c for c in factor if '_resid' in c]
    models = {'sector':BASE,'raw':BASE+rawcols,'residual':BASE+residcols}
    saved,settings,splits,counts,coefficients,skipped = [],[],[],[],[],[]
    for year in range(2022,2027):
        cutoff, cal_start = pd.Timestamp(year,1,1), pd.Timestamp(year-1,1,1)
        train = frame[frame.end<cal_start]
        calibration = frame[frame.date.dt.year.eq(year-1)&(frame.end<cutoff)]
        test = frame[frame.date.dt.year.eq(year)]
        if train.date.nunique()<504 or calibration.date.nunique()<126 or test.empty:
            skipped.append({'year':year,'train_dates':train.date.nunique(),
                            'calibration_dates':calibration.date.nunique(),'test_dates':test.date.nunique()})
            print(f'{year}: skipped, insufficient train/calibration/test history',flush=True)
            continue
        assert train.end.max()<calibration.date.min() and calibration.end.max()<test.date.min()
        cal,pred,audit,coef = fit_forecast(train,calibration,test,models)
        # Keep the fitted model fixed after calibration so threshold scale is preserved.
        counts.extend(dict(a,year=year) for a in audit)
        coefficients.extend(dict(a,year=year) for a in coef)
        splits.append({'year':year,'train_dates':train.date.nunique(),
                       'train_last_label_end':str(train.end.max()),'calibration_dates':calibration.date.nunique(),
                       'calibration_start':str(calibration.date.min()),'calibration_last_label_end':str(calibration.end.max()),
                       'test_start':str(test.date.min()),'test_dates':test.date.nunique(),
                       'train_dates_sha256':hashlib.sha256(train.date.drop_duplicates().to_numpy().tobytes()).hexdigest()})
        for kind in ['entry','exit']:
            for name in MODEL_NAMES:
                daily = daily_scores(cal,kind,name)
                threshold = (np.inf if name=='persistence' else
                             (.5 if name=='calendar' else calibrate_threshold(daily)))
                settings.append({'year':year,'kind':kind,'model':name,'threshold':threshold,
                                 'calibration_alerts':len(alerts_from_scores(daily,threshold))})
        saved.append(pred[['date','pos','end','ticker','kind','y','eligible',*MODEL_NAMES]])
        print(f'{year}: train={train.date.nunique()}, calibration={calibration.date.nunique()}, test={test.date.nunique()}',flush=True)
    if not saved:
        raise ValueError('No outer year satisfies the fixed history requirements')
    pred = pd.concat(saved).sort_values(['date','kind','ticker'])
    settings = pd.DataFrame(settings)
    summaries,annual,alerts,sector_scores = [],[],[],[]
    for kind in ['entry','exit']:
        for name in MODEL_NAMES:
            daily = daily_scores(pred,kind,name)
            by_year = settings[settings.kind.eq(kind)&settings.model.eq(name)].set_index('year').threshold
            threshold = pd.Series(daily.index.year.map(by_year),index=daily.index)
            assert threshold.notna().all()
            issued = alerts_from_scores(daily,threshold)
            matched = match_alerts(issued,events,kind)
            alerts.append(matched.assign(model=name,kind=kind))
            summaries.append(dict(metrics(matched,events,kind,daily),model=name))
            for year in sorted(daily.index.year.unique()):
                annual.append(dict(metrics(matched,events,kind,daily,year=year),model=name,year=int(year)))
            if name in models:
                sub = pred[pred.kind.eq(kind)&pred.eligible]
                for ticker,group in sub.groupby('ticker'):
                    sector_scores.append({'kind':kind,'model':name,'ticker':ticker,'n':len(group),
                                          'positive_n':int(group.y.sum()),'brier':float(((group[name]-group.y)**2).mean())})
    summary = pd.DataFrame(summaries)
    all_alerts = pd.concat(alerts,ignore_index=True)
    OUT.mkdir(parents=True,exist_ok=True)
    for name,data in {'events':events,'predictions':pred,'thresholds':settings,'split_audit':pd.DataFrame(splits),
                      'fit_audit':pd.DataFrame(counts),'coefficients':pd.DataFrame(coefficients),
                      'alerts':all_alerts,'metrics':summary,'sector_brier':pd.DataFrame(sector_scores)}.items():
        data.to_csv(OUT/f'Y_{name}.csv',index=False)
    pd.DataFrame(annual).to_csv(OUT/'Y_annual_metrics.csv',index=False)
    current = pd.DataFrame({'relative20':relative20.iloc[-1],'leader':leaders.iloc[-1]})
    current.assign(date=leaders.index[-1]).to_csv(OUT/'Y_current_leadership.csv',index_label='ticker')
    meta = {'experiment':'Y leadership event forecasts','source':'local adjusted sector and factor ETF prices',
            'cache_sha256':hashlib.sha256(CACHE.read_bytes()).hexdigest(),
            'horizon':HORIZON,'persistence':PERSIST,'target_alerts_per_kind_year':6,
            'models':models,'logistic_C':.1,'oos_start':str(pred.date.min()),'oos_end':str(pred.date.max()),
            'oos_sessions':int(pred.date.nunique()),'event_definition':'top3 trailing20 relative return; previous5 opposite and next5 persistent',
            'skipped_years':skipped,
            'decision_time':'after common ETF close; future onset must be at least one trading session later',
            'threshold_selection':'prior-year alert-count budget only; no performance-based threshold tuning',
            'assessment':'descriptive exploratory event metrics; no independent confirmation or significance claim',
            'boundary':'recall includes only events with all prior10 dates evaluated; precision includes every matured alert',
            'pit_status':'current adjusted histories; historical vintages unverified',
            'scope':'entry/exit scored separately; simultaneous groups and kinds are not independent'}
    (OUT/'Y_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    report = ['# Y 로테이션 경보 결과','',
              '수익률 크기를 학습하지 않고 주도권 진입·이탈을 직접 예측했다. 아래는 탐색 결과이며 유의한 선행력 판정이 아니다.',
              '과거 보정 구간의 경보 예산은 같지만 표본외 실제 경보 빈도는 다를 수 있다. 진입·이탈 포착률을 합산하지 않는다.','',
              '```',summary.to_string(index=False),'```','',
              'precision: 경보 적중률; recall: 실제 사건 포착률; candidate_precision: 경보 중 섹터까지 맞힌 비율.',
              'candidate_precision은 최초 매칭 교체의 섹터 적중, nominee_within10_precision은 지목 섹터가 10일 안에 전환한 비율이다.',
              'lead_days는 확인일이 아닌 최초 시작일 기준이다. warning_window_coverage와 실제 경보 빈도를 함께 비교해야 한다.',
              '기존 20일 수익률 실험과 사건 정의가 다르며, 동일 역사 구간을 재사용했다. 운영 가중치는 변경하지 않았다.']
    (OUT/'Y_REPORT_KO.md').write_text('\n'.join(report),encoding='utf-8')
    print(summary.to_string(index=False))


def self_test():
    from rotation_events import self_test as check_events
    check_events()
    rng = np.random.default_rng(10)
    dates = pd.bdate_range('2019-01-01',periods=250)
    prices = pd.DataFrame(np.exp(np.cumsum(rng.normal(0,.01,(250,5)),axis=0)),
                          index=dates,columns=['SPY','a','b','c','d'])
    factors = pd.DataFrame({'f_rel20':rng.normal(size=250),'f_resid20':rng.normal(size=250)},index=dates)
    panel,leaders,_,_ = build_panel(prices,factors)
    end_dates = pd.Series(leaders.index,index=leaders.index).shift(-14)
    assert panel.end.equals(panel.date.map(end_dates).rename('end'))
    changed = prices.copy()
    changed.iloc[200:] *= np.exp(rng.normal(size=(50,5)))
    later,_,_,_ = build_panel(changed,factors)
    cols = ['date','ticker','kind',*BASE,'eligible']
    pd.testing.assert_frame_equal(panel.loc[panel.date<dates[200],cols].reset_index(drop=True),
                                  later.loc[later.date<dates[200],cols].reset_index(drop=True))
    pd.testing.assert_series_equal(panel.loc[panel.date<dates[186],'y'].reset_index(drop=True),
                                   later.loc[later.date<dates[186],'y'].reset_index(drop=True))
    rows = []
    for kind in ['entry','exit']:
        for ticker in ['a','b','c']:
            for i in range(180):
                row = dict(zip(BASE,rng.normal(size=len(BASE))))
                row.update(ticker=ticker,kind=kind,eligible=True,y=float(i%5==0),
                           date=pd.Timestamp('2020-01-01')+pd.offsets.BDay(i),pos=i)
                rows.append(row)
    train = pd.DataFrame(rows)
    cal,test = train.tail(30).copy(),train.tail(20).copy()
    models = {'sector':BASE,'raw':BASE,'residual':BASE}
    _,before,_,_ = fit_forecast(train,cal,test,models)
    test.y = 1-test.y
    cal.y = 1-cal.y
    _,after,_,_ = fit_forecast(train,cal,test,models)
    pd.testing.assert_frame_equal(before[list(models)],after[list(models)])
    scores = daily_scores(before,'exit','sector')
    assert len(scores) == before.date.nunique() and scores.score.notna().all()
    empty = match_alerts(alerts_from_scores(scores,np.inf),pd.DataFrame(
        columns=['kind','ticker','onset_pos','confirm_pos','onset_date','confirm_date']),'exit')
    zero = metrics(empty,pd.DataFrame(columns=['kind','ticker','onset_pos']),'exit',scores)
    assert zero['alerts']==0 and np.isnan(zero['precision'])
    print('Forecast: test-label independence, daily nomination, and no-alert metrics passed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test',action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        run()
