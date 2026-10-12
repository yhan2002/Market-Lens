"""Z: audit frozen Y entry warnings, no retraining, tuning, downloads or trading."""
import argparse
import hashlib
import json
import sqlite3

import numpy as np
import pandas as pd

from research_msci_rotation import ROOT, OUT
from validate_sector_factor_explanation import SECTORS
from research_rotation_signals import daily_scores
from rotation_events import HORIZON, leadership, events_from_leaders, make_labels, match_alerts
from rotation_validation_stats import random_schedules, block_intervals

SEED = 20260909


def day_lookup(pred, events, relative20):
    """Outcome lookup for every evaluated day, including days without warnings."""
    entry = pred[pred.kind.eq('entry')].copy()
    daily = daily_scores(entry,'entry','sector_rule')
    labels = entry.pivot(index='date',columns='ticker',values='y')
    eligible = entry.pivot(index='date',columns='ticker',values='eligible').astype(bool)
    relative = relative20.reindex(index=labels.index,columns=labels.columns)
    if relative.isna().any().any():
        raise ValueError('Price coverage does not match the frozen evaluation dates')
    # Highest current relative strength outside the top3 is the fixed rank4 baseline.
    nearest = relative.where(eligible).idxmax(axis=1)
    daily['nearest4'] = nearest.reindex(daily.index)
    for name,column in [('nominee','nominated'),('nearest4','nearest4')]:
        daily[name+'_hit'] = [float(labels.loc[d,t]) for d,t in zip(daily.index,daily[column])]
    onsets = sorted(set(events.loc[events.kind.eq('entry'),'onset_pos'].astype(int)))
    daily['market_hit'] = [float(any(1<=p-pos<=HORIZON for p in onsets)) for pos in daily.pos]
    daily['nominee_lead'] = np.nan
    by_sector = {t:g.onset_pos.to_numpy(dtype=int) for t,g in events[events.kind.eq('entry')].groupby('ticker')}
    for d,row in daily.iterrows():
        future = by_sector.get(row.nominated,np.array([],dtype=int))-int(row.pos)
        future = future[(future>=1)&(future<=HORIZON)]
        if len(future):
            daily.loc[d,'nominee_lead'] = int(future.min())
    np.testing.assert_array_equal(daily.nominee_lead.notna().astype(float),daily.nominee_hit)
    return daily


def load_frozen():
    pred = pd.read_csv(OUT/'Y_predictions.csv',parse_dates=['date','end'])
    warnings = pd.read_csv(OUT/'Y_alerts.csv',parse_dates=['decision_date'])
    warnings = warnings[warnings.kind.eq('entry')&warnings.model.eq('sector_rule')].sort_values('decision_pos')
    events = pd.read_csv(OUT/'Y_events.csv',parse_dates=['onset_date','confirm_date'])
    if warnings.empty or not warnings.decision_pos.is_unique:
        raise ValueError('Frozen entry warnings are missing or duplicated')
    if (warnings.decision_pos.diff().dropna()<HORIZON).any():
        raise ValueError('Frozen warning windows overlap')
    with sqlite3.connect((ROOT/'regime_router_publish.db').resolve().as_uri()+'?mode=ro',uri=True) as c:
        tickers = ['SPY',*sorted(SECTORS)]
        data = pd.read_sql_query('SELECT date,ticker,adj_close FROM daily_prices WHERE ticker IN ('+
                                ','.join('?' for _ in tickers)+") AND date<='2026-09-08' ORDER BY date",
                                c,params=tickers,parse_dates=['date'])
    prices = data.pivot(index='date',columns='ticker',values='adj_close').dropna()
    relative,full_leaders = leadership(prices)
    leaders = full_leaders.loc[relative.dropna().index.min():]
    actual_events = events_from_leaders(leaders)
    pd.testing.assert_frame_equal(events.reset_index(drop=True),actual_events.reset_index(drop=True),check_dtype=False)
    positions = pd.Series(np.arange(len(leaders)),index=leaders.index)
    assert (pred.pos == pred.date.map(positions)).all()
    for kind in ['entry','exit']:
        labels,eligible = make_labels(leaders,events,kind)
        sub = pred[pred.kind.eq(kind)]
        y = sub.pivot(index='date',columns='ticker',values='y')
        risk = sub.pivot(index='date',columns='ticker',values='eligible')
        pd.testing.assert_frame_equal(y,labels.reindex(index=y.index,columns=y.columns),check_names=False,check_dtype=False)
        pd.testing.assert_frame_equal(risk,eligible.reindex(index=risk.index,columns=risk.columns),check_names=False,check_dtype=False)
    daily = day_lookup(pred,events,relative)
    if not np.array_equal(np.diff(daily.pos),np.ones(len(daily)-1)):
        raise ValueError('Random schedule test requires a consecutive evaluated calendar; no dates are silently filled')
    selected = daily.loc[warnings.decision_date].copy()
    assert selected.pos.tolist() == warnings.decision_pos.tolist()
    assert selected.nominated.tolist() == warnings.nominated.tolist()
    np.testing.assert_array_equal(selected.nominee_hit.to_numpy(),warnings.nominee_horizon_hit.to_numpy(dtype=float))
    replay = match_alerts(warnings[['decision_date','decision_pos','score','nominated']],events,'entry')
    np.testing.assert_array_equal(replay.hit.to_numpy(),warnings.hit.to_numpy())
    np.testing.assert_array_equal(selected.market_hit.to_numpy(),warnings.hit.to_numpy(dtype=float))
    return daily,selected,warnings


def sensitivity(selected):
    rows = []
    groups = [('all','all',selected)]
    groups += [('leave_year_out',str(y),selected[selected.index.year!=y]) for y in sorted(selected.index.year.unique())]
    groups += [('leave_nominated_sector_out',t,selected[selected.nominated!=t]) for t in sorted(selected.nominated.unique())]
    for kind,excluded,sub in groups:
        rows.append({'check':kind,'excluded':excluded,'alerts':len(sub),
                     'nominee_precision':sub.nominee_hit.mean(),'nearest4_precision':sub.nearest4_hit.mean(),
                     'paired_difference':(sub.nominee_hit-sub.nearest4_hit).mean(),
                     'market_precision':sub.market_hit.mean(),'nominee_lead_median':sub.nominee_lead.median()})
    return pd.DataFrame(rows)


def run(draws=2000):
    if draws<100:
        raise ValueError('Use at least 100 draws; --self-test is the small synthetic check')
    daily,selected,warnings = load_frozen()
    outcomes = {'nominee_precision':selected.nominee_hit.to_numpy(),
                'nearest4_precision':selected.nearest4_hit.to_numpy(),
                'market_precision':selected.market_hit.to_numpy(),
                'paired_difference':(selected.nominee_hit-selected.nearest4_hit).to_numpy()}
    intervals = pd.concat([block_intervals(selected.pos.to_numpy(),daily.pos.to_numpy(),outcomes,
                                           draws=draws,block_length=length,seed=SEED)
                           for length in [63,126]],ignore_index=True)
    counts = pd.Series(selected.index.year).value_counts().sort_index().to_dict()
    schedules = random_schedules(daily.pos.to_numpy(),daily.index.year.to_numpy(),counts,draws=draws,seed=SEED)
    indices = schedules-int(daily.pos.iloc[0])
    placebo = pd.DataFrame({metric:daily[column].to_numpy()[indices].mean(axis=1)
                            for metric,column in [('nominee_precision','nominee_hit'),('market_precision','market_hit')]})
    placebo.insert(0,'draw',np.arange(draws))
    comparison = []
    for metric in ['nominee_precision','market_precision']:
        observed = float(np.mean(outcomes[metric]))
        values = placebo[metric].to_numpy()
        comparison.append({'metric':metric,'observed':observed,'placebo_mean':float(values.mean()),
                           'placebo_q025':float(np.quantile(values,.025)),'placebo_q975':float(np.quantile(values,.975)),
                           'placebo_tail_fraction':float((1+(values>=observed).sum())/(draws+1)),
                           'draws':draws,'alerts_per_schedule':len(selected)})
    comparison = pd.DataFrame(comparison)
    cases = selected.copy()
    cases['nominee_only_success'] = cases.nominee_hit.eq(1)&cases.nearest4_hit.eq(0)
    cases['nearest4_only_success'] = cases.nominee_hit.eq(0)&cases.nearest4_hit.eq(1)
    cases['same_nominee'] = cases.nominated.eq(cases.nearest4)
    timing = pd.DataFrame([{'minimum_lead_sessions':n,'successes':int(selected.nominee_lead.ge(n).sum()),
                           'success_fraction_all_alerts':float(selected.nominee_lead.ge(n).sum()/len(selected))}
                          for n in [1,2,3]])
    outputs = {'paired_cases':cases.reset_index(names='decision_date'),'block_intervals':intervals,
               'placebo_draws':placebo,'placebo_summary':comparison,'sensitivity':sensitivity(selected),
               'lead_margin':timing}
    for name,data in outputs.items():
        data.to_csv(OUT/f'Z_{name}.csv',index=False)
    files = ['Y_predictions.csv','Y_alerts.csv','Y_events.csv','Y_thresholds.csv','Y_metadata.json']
    sources = {name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in files}
    for name in ['validate_rotation_entry.py','rotation_validation_stats.py','rotation_events.py']:
        sources['reports/'+name] = hashlib.sha256((ROOT/'reports'/name).read_bytes()).hexdigest()
    meta = {'scope':'fixed Y entry sector_rule, retrospective diagnostics only',
            'rule_changed':False,'refitting':False,'oos_start':str(daily.index.min()),'oos_end':str(daily.index.max()),
            'alerts':len(selected),'annual_counts':{str(k):int(v) for k,v in counts.items()},'draws':draws,'seed':SEED,
            'primary':'same warning dates: original nominee minus current rank4 nominee',
            'block_lengths':[63,126],'primary_block_length':63,
            'placebo':'uniform cooldown-valid schedules conditioned on observed annual warning counts; nominee stays the causal daily rank-change choice',
            'inference_limit':'conditional historical uncertainty; placebo tail fraction is not a calibrated significance p-value; ignores prior strategy selection uncertainty',
            'source_sha256':sources,'production_changed':False}
    (OUT/'Z_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    summary = pd.DataFrame([{'alerts':len(selected),'nominee_hits':int(selected.nominee_hit.sum()),
                             'nearest4_hits':int(selected.nearest4_hit.sum()),
                             'same_nominee':int(cases.same_nominee.sum()),
                             'nominee_only_success':int(cases.nominee_only_success.sum()),
                             'nearest4_only_success':int(cases.nearest4_only_success.sum())}])
    report = ['# Z 고정 진입 경보 추가 검증','',
              'Y 경보 날짜·후보·임계값·사건 정의를 유지했다. 기존 표본의 추가 진단이며 새로운 독립 확증이 아니다.','',
              '## 같은 날짜의 현재 4위 후보와 비교','```',summary.to_string(index=False),'```','',
              '## 시간 블록 불확실성','```',intervals.to_string(index=False),'```','',
              '63거래일이 주 분석, 126일은 민감도다. 경보가 없는 블록도 포함한다. 소수 블록의 조건부 범위이며 과거 후보 선택의 불확실성은 포함하지 않는다.','',
              '## 같은 연도별 경보 횟수의 무작위 시점 비교','```',comparison.to_string(index=False),'```','',
              'placebo_tail_fraction은 조건부 무작위 시간표가 관측 정밀도 이상인 비율이다. 시간의 교환가능성이나 독립 확증을 증명하는 p값이 아니다.',
              '경보 시점만 바꾸고 각 날짜 후보 선택 규칙은 유지한다. 평가 기간 전체에서 연도별 건수와 10일 경보 간격을 보존한다.','',
              '## 지목 섹터 진입까지 남은 시간','```',timing.to_string(index=False),'```','',
              '최소 2일/3일 행은 원래 경보의 여유 시간 진단이다. 경보를 실제로 지연해 재실행하거나 예측 창을 늘린 결과가 아니다.',
              '연도/지목 섹터 제외 민감도는 Z_sensitivity.csv, 개별 사례는 Z_paired_cases.csv에 기록했다. 가격 패널에서 섹터를 제거해 사건을 재정의하지 않았다.']
    (OUT/'Z_REPORT_KO.md').write_text('\n'.join(report),encoding='utf-8')
    print('PAIRED NOMINEES\n'+summary.to_string(index=False))
    print('BLOCK INTERVALS\n'+intervals.to_string(index=False))
    print('TIMING PLACEBO\n'+comparison.to_string(index=False))
    print('LEAD MARGIN\n'+timing.to_string(index=False))


def self_test():
    from rotation_validation_stats import self_test as stats_test
    stats_test()
    dates = pd.bdate_range('2020-01-01',periods=30)
    rows = []
    for pos,date in enumerate(dates):
        for ticker,score,y in [('A',2.,float(0<=pos<8)),('B',1.,float(0<=pos<4))]:
            rows.append({'date':date,'pos':pos,'ticker':ticker,'kind':'entry','eligible':True,'y':y,'sector_rule':score})
    pred = pd.DataFrame(rows)
    events = pd.DataFrame([{'kind':'entry','ticker':'A','onset_pos':8},{'kind':'entry','ticker':'B','onset_pos':4}])
    relative = pd.DataFrame({'A':.01,'B':.02},index=dates)
    daily = day_lookup(pred,events,relative)
    assert daily.nominated.eq('A').all() and daily.nearest4.eq('B').all()
    assert daily.loc[dates[5],'nominee_hit']==1 and daily.loc[dates[5],'nearest4_hit']==0
    assert daily.loc[dates[5],'nominee_lead']==3
    changed = pred.copy()
    changed['y'] = np.where(changed.ticker.eq('A'),changed.pos.lt(9),changed.pos.lt(5)).astype(float)
    pd.testing.assert_series_equal(daily.nominated,day_lookup(changed,events.assign(onset_pos=events.onset_pos+1),relative).nominated)
    assert len(sensitivity(daily.iloc[:3]))==3
    print('Validation: fixed nominee lookup, rank4 comparator, paired outcomes and sensitivity passed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--draws',type=int,default=2000)
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        run(args.draws)
