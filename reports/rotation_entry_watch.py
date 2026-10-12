"""Manual, forward-only rotation observation journal; no downloads or trades."""
import argparse
import hashlib
import json
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from research_msci_rotation import ROOT, OUT
from validate_sector_factor_explanation import SECTORS
from rotation_events import leadership, events_from_leaders
from rotation_session_calendar import schedule, latest_completed, observation_window, CALENDAR_VERSION, SOURCE_URL

TARGET = ROOT/'outputs/rotation_entry_watch/watch.sqlite'
VERSION = 'entry-rank5-change-top3-h10-p5-forward-v1'


def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)


def code_hash():
    digest = hashlib.sha256()
    for name in ['rotation_entry_watch.py','rotation_events.py','rotation_session_calendar.py']:
        digest.update((ROOT/'reports'/name).read_bytes())
    return digest.hexdigest()


def open_store(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    c = sqlite3.connect(path)
    c.executescript('''
        CREATE TABLE IF NOT EXISTS specification (id INTEGER PRIMARY KEY CHECK(id=1), payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS observations (date TEXT PRIMARY KEY, recorded_at TEXT NOT NULL,
            role TEXT NOT NULL, alert INTEGER NOT NULL, payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS assessments (decision_date TEXT PRIMARY KEY,
            recorded_at TEXT NOT NULL, status TEXT NOT NULL, payload TEXT NOT NULL);
    ''')
    return c


def snapshot(prices):
    relative,leaders = leadership(prices)
    ranks = relative.rank(axis=1,method='first',ascending=False)
    change = ranks.iloc[-6]-ranks.iloc[-1]
    eligible = change.where(~leaders.iloc[-1]).dropna().sort_values(ascending=False,kind='stable')
    nominated = eligible.index[0]
    near = ranks.iloc[-1].where(~leaders.iloc[-1]).dropna().idxmin()
    third = relative.iloc[-1].nlargest(3).min()
    return {'date':prices.index[-1].strftime('%Y-%m-%d'), 'nominated':nominated,'nearest4':near,
            'score':float(eligible.iloc[0]),'nominated_rank':int(ranks[nominated].iloc[-1]),
            'nearest4_rank':int(ranks[near].iloc[-1]),
            'nominated_gap_to_top3':float(relative[nominated].iloc[-1]-third),
            'nearest4_gap_to_top3':float(relative[near].iloc[-1]-third),
            'leaders':{t:bool(v) for t,v in leaders.iloc[-1].items()},
            'ranks':{t:int(v) for t,v in ranks.iloc[-1].items()},
            'rank_change5':{t:float(v) for t,v in change.items()},
            'input_prices':{d.strftime('%Y-%m-%d'):{t:float(v) for t,v in row.items()} for d,row in prices.iterrows()}},leaders


def read_prices(now):
    cal = schedule()
    day = latest_completed(now)
    position = cal.index.get_loc(day)
    dates = cal.index[position-25:position+1]
    if len(dates)!=26:
        raise ValueError('Calendar does not cover the required 26 prior sessions')
    with sqlite3.connect((ROOT/'regime_router_publish.db').resolve().as_uri()+'?mode=ro',uri=True) as c:
        ticks = ['SPY',*sorted(SECTORS)]
        rows = pd.read_sql_query('SELECT date,ticker,adj_close FROM daily_prices WHERE ticker IN ('+
                                ','.join('?' for _ in ticks)+') AND date>=? AND date<=? ORDER BY date',c,
                                params=[*ticks,str(dates[0].date()),str(day.date())],parse_dates=['date'])
    prices = rows.pivot(index='date',columns='ticker',values='adj_close').reindex(index=dates,columns=ticks)
    if prices.isna().any().any() or not np.isfinite(prices.to_numpy()).all() or (prices<=0).any().any():
        raise ValueError(f'Complete adjusted prices for all 12 ETFs through {day.date()} are required; update local data first')
    return prices


def initialize(c,prices,now,threshold):
    with c:
        c.execute('BEGIN IMMEDIATE')
        existing = c.execute('SELECT payload FROM specification WHERE id=1').fetchone()
        if existing:
            return json.loads(existing[0])
        current,leaders = snapshot(prices)
        spec = {'version':VERSION,'frozen_at':now.isoformat(),'cutoff_date':current['date'],
                'signal_year':2026,'threshold':float(threshold),'horizon':10,'persistence':5,'cooldown':10,
                'calendar_version':CALENDAR_VERSION,'calendar_source':SOURCE_URL,'code_sha256':code_hash(),
                'context_input_prices':current['input_prices'],
                'policy':'newest completed session only, before next open; no historical alerts; manual invocation',
                'prior_research':'Y/Z inspected before freeze; only observations after cutoff are forward',
                'next_year':'record states and settle outstanding alerts; no new alerts without a reviewed specification'}
        c.execute('INSERT INTO specification VALUES (1,?)',(canonical(spec),))
        for day,row in leaders.tail(5).iterrows():
            c.execute('INSERT INTO observations VALUES (?,?,?,?,?)',
                      (day.strftime('%Y-%m-%d'),now.isoformat(),'context',0,
                       canonical({'leaders':{t:bool(v) for t,v in row.items()}})))
    return spec


def record(c,prices,now):
    with c:
        c.execute('BEGIN IMMEDIATE')
        row = c.execute('SELECT payload FROM specification WHERE id=1').fetchone()
        if row is None:
            raise ValueError('Initialize the frozen journal first with --initialize')
        spec = json.loads(row[0])
        if spec['code_sha256']!=code_hash() or spec['version']!=VERSION:
            raise ValueError('Frozen code/spec changed; review required, do not reset the existing journal')
        day = prices.index[-1]
        key = day.strftime('%Y-%m-%d')
        if c.execute('SELECT 1 FROM observations WHERE date=?',(key,)).fetchone():
            return {'status':'ALREADY_RECORDED','date':key}
        if key<=spec['cutoff_date'] or now<=pd.Timestamp(spec['frozen_at']):
            raise ValueError('Historical or pre-freeze observations cannot become forward predictions')
        if not observation_window(day,now):
            raise ValueError('Record after this session closes and before the next session opens')
        current,_ = snapshot(prices)
        cal = schedule().index
        last = c.execute('SELECT date FROM observations WHERE alert=1 ORDER BY date DESC LIMIT 1').fetchone()
        spacing = None if last is None else int(cal.get_loc(day)-cal.get_loc(pd.Timestamp(last[0])))
        valid_year = day.year==spec['signal_year']
        alert = bool(valid_year and current['score']>=spec['threshold'] and (spacing is None or spacing>=10))
        current.update(threshold=spec['threshold'],alert=alert,days_since_last_alert=spacing,
                       signal_status='ACTIVE' if valid_year else 'EXPIRED_STATE_ONLY',
                       input_sha256=hashlib.sha256(canonical(current['input_prices']).encode()).hexdigest(),
                       recorded_at=now.isoformat())
        c.execute('INSERT INTO observations VALUES (?,?,?,?,?)',(key,now.isoformat(),'live',int(alert),canonical(current)))
    return {k:v for k,v in current.items() if k not in ['input_prices','leaders','ranks','rank_change5']}


def assess(c,now):
    """Assess only first-recorded states, never revise them with fresh price history."""
    cal = schedule().index
    latest = latest_completed(now)
    results = []
    with c:
        c.execute('BEGIN IMMEDIATE')
        states = {pd.Timestamp(d):json.loads(payload)['leaders'] for d,payload in
                  c.execute('SELECT date,payload FROM observations')}
        pending = list(c.execute('SELECT date,payload FROM observations WHERE alert=1 AND date NOT IN '
                                 '(SELECT decision_date FROM assessments) ORDER BY date'))
        for key,payload in pending:
            day = pd.Timestamp(key)
            pos = cal.get_loc(day)
            if pos+14>=len(cal) or latest<cal[pos+14]:
                continue
            dates = cal[pos-4:pos+15]
            missing = [str(d.date()) for d in dates if d not in states]
            if missing:
                result = {'status':'UNEVALUABLE_MISSING_OBSERVATIONS','missing_dates':missing}
            else:
                observed = pd.DataFrame([states[d] for d in dates],index=dates)
                events = events_from_leaders(observed)
                events = events[events.kind.eq('entry') & events.onset_pos.between(5,14)]
                warning = json.loads(payload)
                first = events.onset_pos.min() if len(events) else None
                result = {'status':'MATURED','market_hit':bool(len(events)),
                          'market_lead':int(first-4) if first is not None else None}
                for name in ['nominated','nearest4']:
                    selected = events[events.ticker.eq(warning[name])]
                    result[name+'_hit'] = bool(len(selected))
                    result[name+'_lead'] = int(selected.onset_pos.min()-4) if len(selected) else None
                result['candidate_first_hit'] = bool(len(events) and warning['nominated'] in events.loc[events.onset_pos.eq(first),'ticker'].tolist())
            result['decision_date'] = key
            c.execute('INSERT OR IGNORE INTO assessments VALUES (?,?,?,?)',
                      (key,now.isoformat(),result['status'],canonical(result)))
            results.append(result)
    return results


def status(c):
    spec = c.execute('SELECT payload FROM specification WHERE id=1').fetchone()
    if spec is None:
        return {'status':'NOT_INITIALIZED'}
    frozen = json.loads(spec[0])
    return {'frozen_at':frozen['frozen_at'],'cutoff_date':frozen['cutoff_date'],'threshold':frozen['threshold'],
            'latest_live_session':c.execute("SELECT MAX(date) FROM observations WHERE role='live'").fetchone()[0],
            'live_observations':c.execute("SELECT COUNT(*) FROM observations WHERE role='live'").fetchone()[0],
            'alerts':c.execute('SELECT COUNT(*) FROM observations WHERE alert=1').fetchone()[0],
            'assessment_counts':dict(c.execute('SELECT status,COUNT(*) FROM assessments GROUP BY status'))}


def main():
    parser = argparse.ArgumentParser()
    flags = parser.add_mutually_exclusive_group()
    flags.add_argument('--initialize',action='store_true')
    flags.add_argument('--status',action='store_true')
    flags.add_argument('--self-test',action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    now = pd.Timestamp.now(tz='UTC')
    c = open_store(TARGET)
    try:
        if args.initialize:
            thresholds = pd.read_csv(OUT/'Y_thresholds.csv')
            threshold = thresholds.loc[thresholds.year.eq(2026)&thresholds.kind.eq('entry')&thresholds.model.eq('sector_rule'),'threshold']
            if len(threshold)!=1 or not np.isfinite(threshold.iloc[0]):
                raise ValueError('Exactly one finite frozen 2026 entry threshold is required')
            prices = read_prices(now)
            initialize(c,prices,pd.Timestamp.now(tz='UTC'),float(threshold.iloc[0]))
        elif not args.status:
            prices = read_prices(now)
            now = pd.Timestamp.now(tz='UTC')
            print(canonical(record(c,prices,now)))
            print(canonical({'completed':assess(c,now)}))
        print(canonical(status(c)))
    finally:
        c.close()


def self_test():
    from rotation_session_calendar import self_test as calendar_test
    calendar_test()
    cal = schedule()
    end = cal.index.get_loc(pd.Timestamp('2026-09-08'))
    rng = np.random.default_rng(13)
    all_prices = pd.DataFrame(np.exp(np.cumsum(rng.normal(0,.01,(70,12)),axis=0)),
                              index=cal.index[end-25:end+45],columns=['SPY',*sorted(SECTORS)])
    now = pd.Timestamp('2026-09-09T07:00:00Z')
    with tempfile.TemporaryDirectory() as temp:
        c = open_store(Path(temp)/'watch.sqlite')
        try:
            first = initialize(c,all_prices.iloc[:26],now,6.)
            assert status(c)['alerts']==0 and status(c)['live_observations']==0
            assert initialize(c,all_prices.iloc[1:27],now+pd.Timedelta(days=1),99.)==first
            data = all_prices.iloc[1:27]
            issued = cal.loc[data.index[-1],'close']+pd.Timedelta(hours=1)
            saved = record(c,data,issued)
            assert saved['date']=='2026-09-09'
            changed = data.copy(); changed.iloc[-1]*=2
            assert record(c,changed,issued)['status']=='ALREADY_RECORDED'
            future = all_prices.iloc[2:28]
            try:
                record(c,future,cal.loc[future.index[-1],'open'])
            except ValueError:
                pass
            else:
                raise AssertionError('Incomplete session accepted')
            assert status(c)['live_observations']==1
            # Fixture a prospective warning; missing subsequent recorded states
            # must produce UNEVALUABLE, never a hindsight-filled success/failure.
            with c:
                c.execute('UPDATE observations SET alert=1 WHERE date=?',(saved['date'],))
            settled = assess(c,cal.loc[all_prices.index[40],'close']+pd.Timedelta(hours=1))
            assert settled[0]['status']=='UNEVALUABLE_MISSING_OBSERVATIONS'
            assert assess(c,cal.loc[all_prices.index[41],'close']+pd.Timedelta(hours=1))==[]
        finally:
            c.close()
        c = open_store(Path(temp)/'matured.sqlite')
        try:
            dates = cal.index[end+1:end+20]
            with c:
                for i,day in enumerate(dates):
                    state = {t:t in ['XLC','XLK','XLV'] for t in sorted(SECTORS)}
                    if i>=7:
                        state['XLK'],state['XLB'] = False,True
                    payload = {'leaders':state,'nominated':'XLB','nearest4':'XLE'}
                    c.execute('INSERT INTO observations VALUES (?,?,?,?,?)',
                              (str(day.date()),now.isoformat(),'live',int(i==4),canonical(payload)))
            result = assess(c,cal.loc[dates[-1],'close']+pd.Timedelta(hours=1))[0]
            assert result['status']=='MATURED' and result['nominated_hit'] and not result['nearest4_hit']
            assert result['nominated_lead']==3 and result['market_lead']==3 and result['candidate_first_hit']
        finally:
            c.close()
    print('PASS: freeze, no backfill, immutable duplicate, incomplete-bar rejection, missing-state assessment')


if __name__ == '__main__':
    main()
