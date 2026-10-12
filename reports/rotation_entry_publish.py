"""Publish only aggregate, first-recorded sector rotation decisions."""
import argparse
import json
import sqlite3
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

import rotation_entry_watch as watch


HISTORY_START = pd.Timestamp('2026-06-01')
HISTORY_END = pd.Timestamp('2026-09-04')


def public_snapshot(c):
    spec = json.loads(c.execute('SELECT payload FROM specification WHERE id=1').fetchone()[0])
    if spec['code_sha256'] != watch.code_hash():
        raise ValueError('Frozen rotation code changed')
    row = c.execute("SELECT payload FROM observations WHERE role='live' ORDER BY date DESC LIMIT 1").fetchone()
    if row:
        current = json.loads(row[0])
        phase = current['signal_status']
    else:
        prices = pd.DataFrame.from_dict(spec['context_input_prices'], orient='index').sort_index()
        prices.index = pd.to_datetime(prices.index)
        current, _ = watch.snapshot(prices)
        current.update(alert=False, days_since_last_alert=None, recorded_at=spec['frozen_at'])
        phase = 'BASELINE'
    # Explicit allowlist: never publish source ETF prices or private holdings.
    return dict(
        series_version=watch.VERSION, as_of=current['date'], recorded_at=current['recorded_at'],
        phase=phase, alert=current['alert'], threshold=spec['threshold'],
        nominated=current['nominated'], nearest4=current['nearest4'], score=current['score'],
        ranks=current['ranks'], rank_change5=current['rank_change5'],
        days_since_last_alert=current['days_since_last_alert'],
    )


def historical_snapshots(start, end, recorded_at, threshold):
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    if start < HISTORY_START or end > HISTORY_END or start > end:
        raise ValueError(f'Historical replay must stay within {HISTORY_START.date()}..{HISTORY_END.date()}')
    calendar = watch.schedule()
    sessions = calendar.index[(calendar.index >= start) & (calendar.index <= end)]
    if len(sessions) == 0:
        raise ValueError('No historical replay sessions')
    first_pos = calendar.index.get_loc(sessions[0])
    required = calendar.index[first_pos-25:calendar.index.get_loc(sessions[-1])+1]
    if len(required) < len(sessions)+25:
        raise ValueError('Calendar does not cover 25 warm-up sessions')
    tickers = ['SPY', *sorted(watch.SECTORS)]
    with sqlite3.connect((watch.ROOT/'regime_router_publish.db').resolve().as_uri()+'?mode=ro', uri=True) as c:
        rows = pd.read_sql_query(
            'SELECT date,ticker,adj_close FROM daily_prices WHERE ticker IN ('+
            ','.join('?' for _ in tickers)+') AND date>=? AND date<=? ORDER BY date', c,
            params=[*tickers, str(required[0].date()), str(required[-1].date())], parse_dates=['date'])
    prices = rows.pivot(index='date', columns='ticker', values='adj_close').reindex(index=required, columns=tickers)
    if prices.isna().any().any() or not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise ValueError('Complete adjusted prices for all historical replay sessions are required')
    payloads, last_alert_pos = [], None
    for day in sessions:
        position = calendar.index.get_loc(day)
        current, _ = watch.snapshot(prices.loc[calendar.index[position-25:position+1]])
        spacing = None if last_alert_pos is None else int(position-last_alert_pos)
        alert = bool(current['score'] >= threshold and (spacing is None or spacing >= 10))
        if alert:
            last_alert_pos = position
        payloads.append(dict(
            series_version=watch.VERSION, as_of=current['date'], recorded_at=recorded_at,
            phase='HISTORICAL_REPLAY', alert=alert, threshold=threshold,
            nominated=current['nominated'], nearest4=current['nearest4'], score=current['score'],
            ranks=current['ranks'], rank_change5=current['rank_change5'],
            days_since_last_alert=spacing,
        ))
    return payloads


def upload(payload, endpoint, headers):
    encoded = watch.canonical(payload).encode('utf-8')
    request = urllib.request.Request(endpoint, data=encoded, headers=headers, method='POST')
    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.load(response)
    if not result.get('ok'):
        raise ValueError('Rotation upload was not acknowledged')
    request = urllib.request.Request(endpoint+'?date='+payload['as_of'], headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        if json.load(response) != payload:
            raise ValueError('Published rotation snapshot failed read-back verification')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--upload', action='store_true')
    parser.add_argument('--backfill-start')
    parser.add_argument('--backfill-end')
    parser.add_argument('--output', type=Path, default=watch.TARGET.parent/'backfill_payload.json')
    args = parser.parse_args()
    if args.record and args.backfill_start:
        parser.error('--record and --backfill-start are mutually exclusive')
    if args.backfill_end and not args.backfill_start:
        parser.error('--backfill-end requires --backfill-start')
    if not watch.TARGET.is_file():
        raise ValueError('Frozen journal must already exist; automatic initialization is forbidden')
    with watch.open_store(watch.TARGET) as c:
        if args.backfill_start:
            spec = json.loads(c.execute('SELECT payload FROM specification WHERE id=1').fetchone()[0])
            payloads = historical_snapshots(
                args.backfill_start, args.backfill_end or HISTORY_END,
                pd.Timestamp.now(tz='UTC').isoformat(), spec['threshold'])
            args.output.write_text(watch.canonical(payloads), encoding='utf-8')
            if args.upload:
                config = json.loads((watch.ROOT/'site_upload_config.json').read_text(encoding='utf-8-sig'))
                endpoint = config['site_url'].rstrip('/')+'/api/rotation-entry'
                headers = {'Authorization':'Bearer '+config['upload_token'], 'Content-Type':'application/json'}
                if config.get('sites_bypass_token'):
                    headers['OAI-Sites-Authorization'] = 'Bearer '+config['sites_bypass_token']
                for payload in payloads:
                    upload(payload, endpoint, headers)
            print(watch.canonical({'start':payloads[0]['as_of'], 'end':payloads[-1]['as_of'],
                                   'sessions':len(payloads), 'alerts':sum(p['alert'] for p in payloads),
                                   'uploaded':args.upload}))
            return
        if args.record:
            prices = watch.read_prices(pd.Timestamp.now(tz='UTC'))
            now = pd.Timestamp.now(tz='UTC')
            watch.record(c, prices, now)
            watch.assess(c, now)
        payload = public_snapshot(c)
    encoded = watch.canonical(payload).encode('utf-8')
    (watch.TARGET.parent/'site_payload.json').write_bytes(encoded)
    if args.upload:
        config = json.loads((watch.ROOT/'site_upload_config.json').read_text(encoding='utf-8-sig'))
        endpoint = config['site_url'].rstrip('/')+'/api/rotation-entry'
        headers = {'Authorization':'Bearer '+config['upload_token'], 'Content-Type':'application/json'}
        if config.get('sites_bypass_token'):
            headers['OAI-Sites-Authorization'] = 'Bearer '+config['sites_bypass_token']
        upload(payload, endpoint, headers)
    print(watch.canonical({'as_of':payload['as_of'], 'phase':payload['phase'],
                           'alert':payload['alert'], 'uploaded':args.upload}))


if __name__ == '__main__':
    main()
