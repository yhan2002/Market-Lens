"""Publish one saved-model weekly inference; no downloads or training."""
import argparse
import json
from datetime import date
from pathlib import Path
from sector_hybrid_signal import build_snapshot, write_json, ROOT, DEFAULT_BUNDLE

SERIES = 'price-hybrid-boost-in-extra-out-2026-v1'


def public_snapshot(database, as_of, bundle=DEFAULT_BUNDLE):
    try:
        result = build_snapshot(database, bundle, as_of)
    except Exception as error:
        # Publish an unavailable state instead of silently leaving an old healthy signal.
        result = {'status': 'ERROR', 'rows': []}
        print('Hybrid inference unavailable: ' + type(error).__name__)
    return {'series_version': SERIES, 'as_of': as_of,
            'decision_date': result.get('as_of'), 'latest_price_date': result.get('latest_price_date'),
            'status': result['status'], 'rows': result['rows']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--as-of', default=date.today().isoformat())
    parser.add_argument('--database', type=Path, default=ROOT/'regime_router_publish.db')
    parser.add_argument('--upload', action='store_true')
    args = parser.parse_args()
    as_of = date.fromisoformat(args.as_of).isoformat()
    payload = public_snapshot(args.database, as_of)
    write_json(ROOT/'outputs/sector_hybrid_site_latest.json', payload)
    if args.upload:
        # Match the existing authenticated snapshot transport.
        import urllib.request
        config = json.loads((ROOT/'site_upload_config.json').read_text(encoding='utf-8-sig'))
        headers = {'Authorization': 'Bearer '+config['upload_token'], 'Content-Type': 'application/json'}
        if config.get('sites_bypass_token'):
            headers['OAI-Sites-Authorization'] = 'Bearer '+config['sites_bypass_token']
        endpoint = config['site_url'].rstrip('/')+'/api/sector-hybrid'
        request = urllib.request.Request(endpoint, data=json.dumps(payload, allow_nan=False).encode(), headers=headers, method='POST')
        with urllib.request.urlopen(request, timeout=30) as response:
            if not json.load(response).get('ok'): raise RuntimeError('Hybrid upload not acknowledged')
        request = urllib.request.Request(endpoint+'?date='+as_of, headers=headers)
        with urllib.request.urlopen(request, timeout=30) as response:
            if json.load(response) != payload: raise RuntimeError('Hybrid read-back mismatch')
    print(json.dumps({'status':payload['status'], 'as_of':as_of, 'decision_date':payload['decision_date'], 'uploaded':args.upload}))
    if payload['status'] not in ('READY','PARTIAL'): raise SystemExit(2)


if __name__ == '__main__': main()
