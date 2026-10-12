"""Read-only, current-universe US stock discovery. No orders, training or uploads."""
import argparse
import csv
import html
import io
import json
import math
import os
import re
import sqlite3
import tempfile
import webbrowser
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from breadth_metrics import OPEN_DATA_SP500_URL, _parse_current_members
from sector_rotation import SECTOR_ETFS
from mp_pitch_evidence import load_mp, load_news

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'outputs/us_topdown'
METADATA_DB = ROOT.parent / 'QuantBot/QuantBotSystem_Beta/data/market_data.sqlite'
STOCK_DATABASE = METADATA_DB
MP_DATABASE = ROOT / 'regime_router_shadow.db'
NEWS_PROJECT = ROOT.parent / 'News Alarm'
SECTOR_NAMES = {name: ticker for ticker, name in SECTOR_ETFS.items()}
SECTOR_NAMES.update(dict(zip(
    ['Basic Materials', 'Consumer Cyclical', 'Consumer Defensive', 'Financial Services', 'Healthcare', 'Technology'],
    ['XLB', 'XLY', 'XLP', 'XLF', 'XLV', 'XLK'])))
STATUS = {'CANDIDATE': '검토 후보', 'WATCH': '관찰', 'EXTENDED': '급등·과열 주의',
          'UNMAPPED': '분류 없음', 'DATA_MISSING': '가격 결측', 'ILLIQUID': '유동성 미달',
          'HELD': 'MP 기존 보유', 'STALE': '가격 갱신 필요'}
SETUP = {'RECOVERY': '회복 관찰', 'TREND': '강세 지속', 'WEAK': '조건 미충족',
         'NOT_EVALUATED': '평가 불가'}


@contextmanager
def readonly(path):
    connection = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)
    try:
        connection.execute('PRAGMA query_only=ON')
        yield connection
    finally:
        connection.close()


def save(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + '.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(content)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def parse_classifications(payload):
    """The existing component source also carries names and sector classifications."""
    _parse_current_members(payload)
    records = {}
    for row in csv.DictReader(io.StringIO(payload.decode('utf-8-sig'))):
        ticker = row['Symbol'].strip().upper().replace('.', '-')
        sector = row.get('Sector', row.get('GICS Sector', '')).strip()
        if sector not in SECTOR_NAMES or not row.get('Security', row.get('Name', '')).strip():
            raise ValueError('Invalid company name or sector: ' + ticker)
        records[ticker] = {'name': row.get('Security', row.get('Name')), 'sector_etf': SECTOR_NAMES[sector]}
    return records


def refresh_classifications(path):
    # Explicit opt-in only; normal screen never accesses the network.
    request = Request(OPEN_DATA_SP500_URL, headers={'User-Agent': 'MarketLens-local-research/1.0'})
    with urlopen(request, timeout=30) as response:
        raw = response.read(2_000_001)
    if len(raw) > 2_000_000:
        raise ValueError('Classification response too large')
    payload = {'source': OPEN_DATA_SP500_URL, 'updated_at': datetime.now(timezone.utc).isoformat(),
               'records': parse_classifications(raw)}
    save(path, json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2))


def classifications(cache, metadata_db):
    if Path(cache).exists():
        payload = json.loads(Path(cache).read_text(encoding='utf-8'))
        records = payload['records']
        for ticker, row in records.items():
            if not re.fullmatch(r'[A-Z0-9][A-Z0-9.-]{0,14}', ticker) or row['sector_etf'] not in SECTOR_ETFS:
                raise ValueError('Invalid classification cache')
        return records, str(payload['source']), str(payload['updated_at'])
    records, stamps = {}, []
    if Path(metadata_db).exists():
        with readonly(metadata_db) as con:
            rows = con.execute("SELECT ticker, short_name, sector, updated_at FROM security_metadata "
                               "WHERE listing_country='US' AND quote_type='EQUITY'").fetchall()
        for ticker, name, sector, stamp in rows:
            if sector in SECTOR_NAMES and name:
                records[ticker] = {'name': name, 'sector_etf': SECTOR_NAMES[sector]}
                stamps.append(stamp)
    return records, 'QuantBot Yahoo sector proxy (공식 GICS 검증 아님)', min(stamps) if stamps else None


def load_prices(database, stock_database=None):
    """Return the SPY cutoff, current members and aligned matrices with source metadata.

    None preserves the Market Lens-only API. The CLI explicitly selects Beta by default.
    Each stock's entire series comes from the selected database, without fallback.
    """
    today = date.today().isoformat()
    with readonly(database) as con:
        last = con.execute("SELECT MAX(date) FROM daily_prices WHERE ticker='SPY' AND date<=?", (today,)).fetchone()[0]
        if not last:
            raise ValueError('SPY prices missing')
        tickers = [r[0] for r in con.execute(
            "SELECT DISTINCT ticker FROM sp500_membership WHERE effective_from<=? "
            "AND (effective_to IS NULL OR effective_to>=?) AND membership_as_of<=? AND available_at<=? ORDER BY ticker",
            (last, last, last, last))]
        if not tickers:
            raise ValueError('Current S&P500 universe missing')
        benchmarks = {'SPY'} | set(SECTOR_ETFS)
        symbols = sorted(benchmarks if stock_database is not None else set(tickers) | benchmarks)
        start = (date.fromisoformat(last) - timedelta(days=450)).isoformat()
        marks = ','.join('?' for _ in symbols)
        frame = pd.read_sql_query(
            f'SELECT ticker,date,adj_close,close,volume FROM daily_prices WHERE ticker IN ({marks}) AND date BETWEEN ? AND ?',
            con, params=[*symbols, start, last])
    if stock_database is not None:
        stock_symbols = sorted(set(tickers) - benchmarks)
        marks = ','.join('?' for _ in stock_symbols)
        with readonly(stock_database) as con:
            stocks = pd.read_sql_query(
                f'SELECT ticker,date,adj_close,close,volume FROM ohlcv WHERE ticker IN ({marks}) AND date BETWEEN ? AND ?',
                con, params=[*stock_symbols, start, last])
        frame = pd.concat([frame, stocks], ignore_index=True)
    if frame.duplicated(['ticker', 'date']).any():
        raise ValueError('Duplicate ticker/date prices')
    dates = pd.to_datetime(frame['date'], format='%Y-%m-%d', errors='coerce')
    if dates.isna().any() or not dates.dt.strftime('%Y-%m-%d').eq(frame['date']).all():
        raise ValueError('Invalid price date; expected YYYY-MM-DD')
    for col in ['adj_close', 'close', 'volume']:
        frame[col] = pd.to_numeric(frame[col], errors='coerce')
    calendar = sorted(frame.loc[frame.ticker.eq('SPY'), 'date'].unique())
    # No raw-close fallback and no forward fill: adjusted-return comparability matters.
    matrices = {col: frame.pivot(index='date', columns='ticker', values=col).reindex(
                    index=calendar, columns=sorted(set(tickers) | benchmarks))
                for col in ['adj_close', 'close', 'volume']}
    matrices['price_sources'] = {
        'benchmark': {'provider': 'Market Lens', 'database': str(Path(database).resolve()),
                      'table': 'daily_prices', 'symbols': sorted(benchmarks)},
        'stock': {'provider': 'QuantBot Beta' if stock_database is not None else 'Market Lens',
                  'database': str(Path(stock_database if stock_database is not None else database).resolve()),
                  'table': 'ohlcv' if stock_database is not None else 'daily_prices',
                  'return_field': 'adj_close', 'liquidity_fields': ['close', 'volume'],
                  'adjustment_provenance': 'UNVERIFIED',
                  'adjustment_note': ('Beta의 기존 수집기가 Adj Close 누락 시 Close를 adj_close에 저장했을 수 있습니다. '
                                      '저장된 adj_close의 조정 이력은 보증되지 않습니다.'
                                      if stock_database is not None else '저장된 adj_close의 조정 이력을 별도로 검증하지 않았습니다.')},
        'alignment': 'SPY sessions through as_of; whole stock series from one source; no fill or close fallback',
        'point_in_time': False}
    return last, tickers, matrices


def metrics(prices):
    values = prices.tail(126).to_numpy(dtype=float)
    if len(values) != 126 or not np.isfinite(values).all() or (values <= 0).any():
        return None
    returns = np.diff(values) / values[:-1]
    vol = float(np.std(returns[-20:], ddof=1))
    r20 = values[-1] / values[-21] - 1
    old20 = values[-6] / values[-26] - 1
    extension = (values[-1] / values[-20:].mean() - 1) / (vol * math.sqrt(20)) if vol > 0 else None
    return {'return5_pct': 100 * (values[-1] / values[-6] - 1), 'return20_pct': 100 * r20,
            'return60_pct': 100 * (values[-1] / values[-61] - 1),
            'previous20_pct': 100 * (values[-21] / values[-41] - 1),
            'change20_5d_pp': 100 * (r20 - old20), 'above_ma20': bool(values[-1] > values[-20:].mean()),
            'above_ma50': bool(values[-1] > values[-50:].mean()),
            'drawdown126_pct': 100 * (values[-1] / values.max() - 1), 'extension_sigma': extension}


def assign_ranks(rows, fields):
    if not rows:
        return
    frame = pd.DataFrame(rows)
    # ponytail: equal-weight descriptive ranks; validate out-of-sample before optimizing weights.
    scores = frame[fields].rank(pct=True).mean(axis=1) * 100
    for row, score in zip(rows, scores):
        row['score'] = float(score)
    rows.sort(key=lambda row: (-row['score'], row['ticker']))
    for i, row in enumerate(rows, 1):
        row['rank'] = i


def screen(as_of, tickers, matrices, mapping, held=None):
    prices = matrices['adj_close'].loc[:as_of]
    available = {ticker: metrics(prices[ticker]) for ticker in prices}
    spy = available.get('SPY')
    if not spy:
        raise ValueError('SPY requires 126 consecutive valid adjusted closes')
    stocks, sectors = [], []
    sources = matrices.get('price_sources', {'stock': {'provider': 'Supplied matrices'}})
    for ticker in tickers:
        meta, values = mapping.get(ticker, {}), available.get(ticker)
        stock_prices = prices.get(ticker, pd.Series(np.nan, index=prices.index))
        valid_prices = np.isfinite(stock_prices) & stock_prices.gt(0)
        last_price_date = str(stock_prices.index[valid_prices][-1]) if valid_prices.any() else None
        closes = matrices['close'].reindex(index=prices.index, columns=[ticker])[ticker].tail(20).to_numpy(dtype=float)
        volumes = matrices['volume'].reindex(index=prices.index, columns=[ticker])[ticker].tail(20).to_numpy(dtype=float)
        liquidity_sessions = int((np.isfinite(closes) & np.isfinite(volumes) & (closes > 0) & (volumes >= 0)).sum())
        valid_volume = liquidity_sessions == 20
        row = {'ticker': ticker, 'name': meta.get('name', ticker), 'sector_etf': meta.get('sector_etf'),
               'status': 'DATA_MISSING', 'setup': 'NOT_EVALUATED', 'setup_evaluated': False, 'score': None, 'rank': None,
               'price_source': sources['stock']['provider'], 'last_price_date': last_price_date,
               'missing_latest_sessions': int((prices.index > last_price_date).sum()) if last_price_date else len(prices),
               'valid_sessions126': int(valid_prices.tail(126).sum()), 'liquidity_valid_sessions20': liquidity_sessions,
               'fundamental_status': 'NOT_CONNECTED', 'mp_status': 'UNKNOWN' if held is None else 'HELD' if ticker in held else 'NOT_HELD',
               'reasons': [], 'news': []}
        if values:
            row.update(values)
            sector = available.get(row['sector_etf'])
            row['liquidity20_usd'] = float(np.mean(closes * volumes)) if valid_volume else None
            if sector:
                row['setup_evaluated'] = True
                row.update({f'relative{n}_sector_pp': values[f'return{n}_pct'] - sector[f'return{n}_pct'] for n in (5, 20, 60)})
                row['relative_acceleration_pp'] = values['change20_5d_pp'] - sector['change20_5d_pp']
                row['setup'] = ('RECOVERY' if values['previous20_pct'] < 0 < values['return20_pct'] and values['above_ma20'] and row['relative5_sector_pp'] > 0
                                else 'TREND' if values['return60_pct'] > 0 and values['above_ma50'] and row['relative20_sector_pp'] > 0 else 'WEAK')
                row['status'] = 'CANDIDATE' if row['setup'] != 'WEAK' else 'WATCH'
                row['reasons'] = [f"20일 섹터 대비 {row['relative20_sector_pp']:+.2f}%p", f"60일 섹터 대비 {row['relative60_sector_pp']:+.2f}%p"]
                if values['return20_pct'] >= 25 or (values['extension_sigma'] is not None and values['extension_sigma'] >= 2):
                    row['status'] = 'EXTENDED'
                if not valid_volume:
                    row['status'] = 'DATA_MISSING'
                    row['reasons'].append(f'최근 20세션 거래대금 유효값 {liquidity_sessions}/20')
                elif row['liquidity20_usd'] < 10_000_000:
                    row['status'] = 'ILLIQUID'
                if row['mp_status'] == 'HELD' and row['status'] in ('CANDIDATE', 'WATCH', 'EXTENDED'):
                    row['status'] = 'HELD'
            else:
                row['status'] = 'UNMAPPED' if not row['sector_etf'] else 'DATA_MISSING'
                row['reasons'].append('섹터 분류 없음' if not row['sector_etf'] else '비교 섹터 ETF의 126세션 가격 결측')
        else:
            row['reasons'].append(f"SPY 기준 최근 126세션 adj_close 유효값 {row['valid_sessions126']}/126; 보간·타 출처 보충 없음")
        stocks.append(row)
    for ticker, name in SECTOR_ETFS.items():
        values = available.get(ticker)
        if values is None:
            continue
        peers = [r for r in stocks if r['sector_etf'] == ticker]
        measured = [r for r in peers if 'above_ma50' in r]
        sectors.append({'ticker': ticker, 'name': name, 'relative20_pp': values['return20_pct'] - spy['return20_pct'],
                        'relative60_pp': values['return60_pct'] - spy['return60_pct'],
                        'breadth50_pct': 100 * sum(r['above_ma50'] for r in measured) / len(measured) if measured else None,
                        'peer_count': len(measured), 'classified_peer_count': len(peers), 'hybrid_signal': None})
        ranked = [r for r in peers if r['status'] in ('CANDIDATE', 'WATCH', 'EXTENDED', 'HELD')]
        assign_ranks(ranked, ['relative20_sector_pp', 'relative60_sector_pp', 'relative_acceleration_pp'])
    assign_ranks(sectors, ['relative20_pp', 'relative60_pp'])
    sector_ranks = {r['ticker']: r['rank'] for r in sectors}
    for row in stocks:
        row['sector_rank'] = sector_ranks.get(row['sector_etf'])
    stocks.sort(key=lambda r: (r['sector_rank'] or 99, r['rank'] or 999, r['ticker']))
    last200 = prices['SPY'].tail(200)
    above200 = bool(last200.iloc[-1] > last200.mean()) if len(last200) == 200 and np.isfinite(last200).all() and last200.gt(0).all() else None
    valid_stocks = [r for r in stocks if 'above_ma50' in r]
    market = {'return20_pct': spy['return20_pct'], 'return60_pct': spy['return60_pct'], 'above_ma200': above200,
              'trend': '강세 환경' if above200 and spy['return20_pct'] > 0 else '방어적 환경' if above200 is False and spy['return20_pct'] < 0 else '혼조 / 확인 필요',
              'breadth50_pct': 100 * sum(r['above_ma50'] for r in valid_stocks) / len(valid_stocks) if valid_stocks else None}
    coverage = {'universe_count': len(stocks), 'required_sessions': 126,
                'price_evaluated_count': len(valid_stocks), 'price_missing_count': len(stocks) - len(valid_stocks),
                'setup_evaluated_count': sum(r['setup_evaluated'] for r in stocks),
                'data_missing_count': sum(r['status'] == 'DATA_MISSING' for r in stocks),
                'latest_session_missing_count': sum(r['missing_latest_sessions'] > 0 for r in stocks),
                'latest_stock_date': max((r['last_price_date'] for r in stocks if r['last_price_date']), default=None)}
    return {'schema_version': 1, 'as_of': as_of, 'market': market, 'sectors': sectors, 'candidates': stocks,
            'price_sources': sources, 'price_coverage': coverage}


def attach_hybrid(result, path):
    if not Path(path).exists():
        result['warnings'].append('저장된 섹터 전환 신호 없음. 가격 순위만 계산.')
        return
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    decision = payload.get('decision_date')
    if payload.get('status') not in ('READY', 'PARTIAL') or not decision or not decision <= result['as_of'] or payload.get('as_of', '9999') > result['as_of']:
        result['warnings'].append('섹터 전환 신호의 상태/시점 불일치: 사용하지 않음.')
        return
    if (date.fromisoformat(result['as_of']) - date.fromisoformat(decision)).days > 10:
        result['warnings'].append('섹터 전환 신호가 10일 넘게 오래됨: 사용하지 않음.')
        return
    signals = {r['ticker']: r for r in payload.get('rows', []) if r.get('status') == 'READY'}
    for row in result['sectors']:
        row['hybrid_signal'] = signals.get(row['ticker'], {}).get('signal')
        row['hybrid_decision_date'] = decision


def num(value, suffix=''):
    return 'N/A' if value is None else f'{value:,.2f}{suffix}'


def attach_pitch_evidence(result, mp, news):
    """Evidence completeness is separate from investment scores and recommendations."""
    result['mp'] = {key: value for key, value in mp.items() if key != 'held'}
    result['news_context'] = {key: value for key, value in news.items() if key != 'by_ticker'}
    held = set(mp.get('held', []))
    sectors = {row['ticker']: row for row in result['sectors']}
    for row in result['candidates']:
        row['mp_status'] = ('HELD' if row['ticker'] in held else 'NOT_HELD'
                            if mp['status'] in ('READY', 'USER_LIST') else 'UNKNOWN')
        if row['mp_status'] == 'HELD' and row['status'] in ('CANDIDATE', 'WATCH', 'EXTENDED'):
            row['status'] = 'HELD'
        row['news'] = news.get('by_ticker', {}).get(row['ticker'], [])
        row['pitch_gaps'] = ['실적 성장·가이던스', '밸류에이션·동종기업 비교', '투자논리의 반증 조건']
        if row['mp_status'] == 'UNKNOWN':
            row['pitch_gaps'].insert(0, '최신 MP 보유 여부')
        if not row['news']:
            row['pitch_gaps'].append('검증된 기업 관련 기사·촉매 근거')
        row['sector_context'] = sectors.get(row['sector_etf'])
        row['pitch_status'] = ('EXISTING_HOLDING' if row['mp_status'] == 'HELD' else
                               'DUE_DILIGENCE' if row['status'] == 'CANDIDATE' and row['mp_status'] == 'NOT_HELD'
                               else 'WATCH')
    if mp['status'] not in ('READY', 'USER_LIST'):
        result['warnings'].append('MP 스냅샷이 없거나 오래되었거나 종목 매핑이 불완전합니다. 미확인 종목을 신규 편입으로 단정하지 않습니다.')
    if news.get('status') != 'READY':
        result['warnings'].append('뉴스 근거 연결 상태: ' + str(news.get('status')) + '. 기사 없음은 악재나 낮은 투자 점수를 뜻하지 않습니다.')


def pitch_details(row):
    esc = lambda value: html.escape(str(value))
    sector = row.get('sector_context') or {}
    articles = []
    for item in row.get('news', []):
        # Defense in depth: only ordinary web links can be opened from the local report.
        from urllib.parse import urlsplit
        url = urlsplit(item['url'])
        if url.scheme not in ('https', 'http') or not url.hostname or url.username or url.password:
            continue
        articles.append(f'<li><a href="{esc(item["url"])}" target="_blank" rel="noopener noreferrer">{esc(item["title"])}</a>'
                        f'<p>{esc(item["quote"])}</p><small>게시 {esc(item["published_at"])}<br>'
                        f'수집 {esc(item["collected_at"])}<br>관련성 평가 {esc(item["assessed_at"])}</small></li>')
    news = '<ul>' + ''.join(articles) + '</ul>' if articles else '<p>조건을 충족한 저장 기사 없음. 촉매 부재를 뜻하지 않습니다.</p>'
    gaps = ' · '.join(esc(value) for value in row.get('pitch_gaps', ['실적·가치·촉매 검증']))
    return (f'<details class="pitch"><summary>피칭 검토 근거 · 기사 {len(articles)}건</summary>'
            f'<b>가격상 관찰 근거</b><p>{esc(SETUP[row["setup"]])} · '
            f'20일 섹터 대비 {num(row.get("relative20_sector_pp"), "%p")}, '
            f'60일 {num(row.get("relative60_sector_pp"), "%p")}</p>'
            f'<small>주간 섹터 판단 {esc(sector.get("hybrid_signal") or "미확인")} · '
            f'{esc(sector.get("hybrid_decision_date") or "기준일 없음")}. 일간 재확인 신호가 아닙니다.</small>'
            f'<b>기업 관련 기사 원문 근거</b>{news}'
            f'<small>자동 기업 관련성 검증을 통과한 저장 기사입니다. 기사 논조·관련성이 주가 상승 근거를 보장하지 않습니다.</small>'
            f'<b>피칭 전 추가 확인</b><p>{gaps}</p>'
            f'<small>진입 시점: 일간 거래대금·과열만 확인. 장중 체결·호가·스캘핑 신호 미연결.</small></details>')


def render_report(result):
    esc = lambda v: html.escape(str(v))
    market, universe = result['market'], result['universe']
    sources, coverage = result.get('price_sources', {}), result.get('price_coverage', {})
    stock_source, benchmark_source = sources.get('stock', {}), sources.get('benchmark', {})
    sector_cards = ''.join(f'<article><b>{s["rank"]}. {esc(s["ticker"])}</b><p>{esc(s["name"])}</p>'
        f'<p>SPY 대비 20일 {num(s["relative20_pp"], "%p")}<br>60일 {num(s["relative60_pp"], "%p")}</p>'
        f'<small>50일선 상회 {num(s["breadth50_pct"], "%")} · 가격 평가 {s["peer_count"]}/{s.get("classified_peer_count", s["peer_count"])}개 기준<br>'
        f'주간 전환 {esc(s["hybrid_signal"] or "미확인")} · {esc(s.get("hybrid_decision_date", ""))}</small></article>' for s in result['sectors'])
    rows = []
    for row in result['candidates']:
        searchable = esc(f'{row["ticker"]} {row["name"]}').lower()
        rows.append(f'<tr data-sector="{esc(row["sector_etf"] or "UNKNOWN")}" data-status="{row["status"]}" data-mp="{row["mp_status"]}" data-search="{searchable}">'
            f'<td><b>{esc(row["ticker"])}</b><small>{esc(row["name"])}</small>'
            f'<small>{esc(row.get("price_source", "출처 미확인"))} · 최종 유효일 {esc(row.get("last_price_date") or "없음")}<br>'
            f'최근 결측 {row.get("missing_latest_sessions", "N/A")}세션 · 126세션 유효 {row.get("valid_sessions126", "N/A")}/126</small></td>'
            f'<td>{esc(row["sector_etf"] or "미분류")}<small>섹터 {row["sector_rank"] or "-"}위 / 종목 {row["rank"] or "-"}위</small></td>'
            f'<td><span class="tag {row["status"]}">{STATUS[row["status"]]}</span><small>{SETUP[row["setup"]]}</small></td>'
            f'<td>{num(row["score"])}</td><td>{num(row.get("relative20_sector_pp"), "%p")}</td><td>{num(row.get("relative60_sector_pp"), "%p")}</td>'
            f'<td>{num(row.get("return20_pct"), "%")}<small>126일 고점 대비 {num(row.get("drawdown126_pct"), "%")}</small></td>'
            f'<td>{num(row.get("liquidity20_usd", 0) / 1e6 if row.get("liquidity20_usd") is not None else None)} M$</td>'
            f'<td>{"미확인" if row["mp_status"] == "UNKNOWN" else "보유" if row["mp_status"] == "HELD" else "기준 명단 내 미보유"}'
            f'<small>실적·밸류에이션 미연결</small>{pitch_details(row)}</td></tr>')
    sector_options = ''.join(f'<option>{s}</option>' for s in SECTOR_ETFS) + '<option value="UNKNOWN">미분류</option>'
    warnings = ''.join(f'<li>{esc(w)}</li>' for w in result['warnings'])
    return f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Market Lens · MP 신규 편입 검토</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f3f6f5;color:#153c3e;font:15px/1.6 system-ui,"Malgun Gothic",sans-serif}}main{{max-width:1500px;margin:auto;padding:32px}}
h1{{font-size:28px;margin:5px 0}}h2{{font-size:19px;margin-top:30px}}p{{margin:10px 0}}small{{display:block;color:#637b7b;font-size:12px}}.overline{{letter-spacing:2px;font-size:11px;color:#567879}}
.box,article{{background:white;border:1px solid #d5e1de;border-radius:12px;padding:18px}}.warning{{border-left:4px solid #b48926;background:#fffbee;padding:14px 22px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px}}
.toolbar{{display:flex;gap:12px;align-items:end;flex-wrap:wrap;margin:12px 0}}label{{font-size:13px}}select,input{{display:block;border:1px solid #acc1bc;border-radius:6px;padding:9px;background:white;color:#153c3e;font:inherit}}.scroll{{overflow:auto}}table{{width:100%;border-collapse:collapse;white-space:nowrap;background:white}}td,th{{padding:13px 10px;text-align:left;border-bottom:1px solid #e4ece9}}th{{font-size:12px;background:#eaf1ee}}td{{font-size:13px}}td:first-child small{{max-width:190px;white-space:normal}}
.tag{{padding:3px 7px;border:1px solid #a6bab5;border-radius:5px;font-size:11px}}.CANDIDATE{{color:#087651;background:#e9f6ef}}.EXTENDED{{color:#935600;background:#fff5dc}}button{{border:1px solid #a6bab5;border-radius:6px;padding:9px;cursor:pointer;background:white}}summary{{cursor:pointer}}@media(max-width:700px){{main{{padding:16px}}h1{{font-size:22px}}}}
.pitch{{white-space:normal;min-width:270px;max-width:460px}}.pitch[open]{{padding:12px;background:#f3f6f5}}.pitch b{{display:block;margin-top:12px}}.pitch li{{margin:12px 0}}.pitch a{{color:#12666d;overflow-wrap:anywhere}}.pitch summary{{font-size:12px}}
</style><main><div class="overline">Market Lens · LOCAL RESEARCH WORKBENCH</div><h1>MP 신규 편입 검토 · 미국 개별주</h1>
<p>시장 환경 → 섹터 우선순위 → 같은 섹터 내 종목 비교 → 편입 검토</p><small>가격 기준 {esc(result['as_of'])} · 로컬 데이터 · 다운로드/주문/웹 게시 없음</small>
<div class="warning"><b>가격 기반 검토 후보이며 매수 추천·전환 확률이 아닙니다.</b><br>펀더멘털 미연결 · 기본 산식 성과 미검증 · 분류 {universe['mapped_count']}/{universe['count']}종목 (분류된 종목 안의 순위).<ul>{warnings}</ul></div>
<p>개별주 출처: {esc(stock_source.get('provider', '미확인'))} · SPY/11개 섹터 ETF: {esc(benchmark_source.get('provider', '미확인'))}<br>
126세션 가격 평가 {coverage.get('price_evaluated_count', 'N/A')}/{universe['count']}개 · 가격 결측 {coverage.get('price_missing_count', 'N/A')}개 · 섹터 대비 조건 평가 {coverage.get('setup_evaluated_count', 'N/A')}개 · 유동성/섹터 포함 데이터 결측 {coverage.get('data_missing_count', 'N/A')}개<br>
<small>개별주 최종 유효일 {esc(coverage.get('latest_stock_date') or '없음')} · 기준일 가격 결측 {coverage.get('latest_session_missing_count', 'N/A')}개. 종목별 마지막 유효일과 연속 결측은 아래 표에서 확인합니다.<br>{esc(stock_source.get('adjustment_note', '조정 이력 미검증'))}</small></p>
<p>MP 보유 기준 {esc(result.get('mp', {}).get('as_of') or '미확인')} · {esc(result.get('mp', {}).get('source') or '미연결')}<br>
<small>뉴스 근거 확인시각 {esc(result.get('news_context', {}).get('cutoff') or '미연결')} · 가격 기준일과 별개인 현재 검토자료입니다. 과거 백테스트에 소급 적용하지 않습니다.</small></p>
<h2>1. 시장 환경</h2><div class="box"><b>{esc(market['trend'])}</b> · SPY 20일 {num(market['return20_pct'], "%")} · 60일 {num(market['return60_pct'], "%")} · 현재 구성종목 50일선 상회 {num(market['breadth50_pct'], "%")}<small>현재 S&P500 구성 기준, 과거 PIT 아님. 시장 상태는 위험 참고이며 후보 자동 매수/제외 조건이 아닙니다. SS 계좌 위험점수는 사용하지 않습니다.</small></div>
<h2>2. 섹터 우선순위</h2><div class="grid">{sector_cards}</div><small>SPY 대비 20·60일 상대수익률의 횡단면 순위 평균. 주간 IN/OUT은 별도 참고 신호이며 이 순위에 합산하지 않습니다.</small>
<h2>3. 종목 검토</h2><div class="toolbar"><label>섹터<select id="sector"><option value="">전체</option>{sector_options}</select></label>
<label>분류<select id="status"><option value="NEW">MP 미보유 가격 후보</option><option value="CANDIDATE">가격 후보 · MP 미확인 포함</option><option value="">전체 · 제외/누락 포함</option>{''.join(f'<option value="{k}">{v}</option>' for k,v in STATUS.items() if k != 'CANDIDATE')}</select></label>
<label>종목/회사명<input id="query" type="search" placeholder="예: MSFT"></label><span id="count" role="status"></span><button onclick="window.print()">인쇄 / PDF</button></div>
<div class="scroll"><table><thead><tr><th>종목</th><th>Top-down 순서</th><th>판정</th><th>동일 섹터 점수</th><th>20일 vs 섹터</th><th>60일 vs 섹터</th><th>절대 수익률</th><th>20일 평균 거래대금</th><th>MP / 피칭 근거</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<h2>4. 최종 피칭 전 확인</h2><div class="box">가격 후보 → 실적 성장·가이던스 → 밸류에이션 → 뉴스 원문/촉매 → MP 중복·집중 위험 → 담당자 승인. 이 단계들이 채워지기 전 자동 편입하지 않습니다.</div>
<details class="box"><summary>산식 · 데이터 출처 · 한계</summary><p>초기 범위는 저장된 현재 S&P500이며 미국 상장 전체 종목은 아닙니다. 순서는 섹터 순위 → 섹터 내 점수입니다. 점수는 20일·60일 섹터 대비 수익률과 20일 상대수익률의 최근 5일 개선폭 백분위의 단순 평균이며 성공 확률이 아닙니다.</p>
<p>회복 관찰: 직전 20일 하락 후 최근 20일 상승, 20일선 상회, 최근 5일 섹터 초과. 강세 지속: 60일 상승, 50일선 상회, 20일 섹터 초과. 과열 주의: 20일 수익률 25% 이상 또는 20일선 이격 / (일간 변동성 × √20)이 2 이상. 후자는 정규분포 확률이 아닙니다. 유동성은 20일 평균 종가×거래량 1천만 달러 이상. 이 문턱은 검증 전 가설입니다.</p>
<p>126개 연속 조정종가 필요. 결측 보간 없음. 비교 상대 ETF는 섹터 proxy이며 시가총액·베타·업종 내 차이는 별도 통제하지 않았습니다. 분류/가격은 현재 수정 이력으로 생존편향 없는 과거 성과를 뜻하지 않습니다.</p>
<p>개별주: {esc(stock_source.get('database', '직접 제공'))} / {esc(stock_source.get('table', '미확인'))}. 각 종목의 전체 계산 구간을 같은 출처에서 읽으며 Market Lens의 과거 주가와 이어 붙이지 않습니다. 수익률에는 저장된 adj_close만 사용하고 close 대체·결측 보간을 하지 않습니다. 거래대금만 close×volume을 사용합니다.<br>
SPY/섹터 ETF·현재 구성종목·기준일: {esc(benchmark_source.get('database', '직접 제공'))}. SPY 거래일과 기준일 이하로 정렬합니다. 저장된 adj_close의 실제 조정 여부와 과거 PIT 데이터는 보증하지 않습니다.</p>
<p>분류 출처: {esc(universe['mapping_source'])}<br>분류 저장일: {esc(universe['mapping_as_of'])}<br>사용자 확인에 따라 SS 가공용에서 가져온 보유 스냅샷을 MP 기준으로 사용합니다. 보유 여부는 티커 기준이며 다른 주식 종류·ADR 등 동일 기업 중복은 추가 확인해야 합니다. 뉴스는 점수에 합산하지 않습니다. QuantBot의 기술적 점수는 재무 퀄리티가 아니며, 실적/가치 데이터는 아직 미연결입니다.</p></details>
</main><script>const rows=[...document.querySelectorAll('tbody tr')];const sector=document.getElementById('sector'),status=document.getElementById('status'),query=document.getElementById('query');function filter(){{let n=0;for(const r of rows){{const excluded=status.value==='NEW'?(r.dataset.status!=='CANDIDATE'||r.dataset.mp!=='NOT_HELD'):(status.value&&r.dataset.status!==status.value);r.hidden=!!((sector.value&&r.dataset.sector!==sector.value)||excluded||!r.dataset.search.includes(query.value.toLowerCase().trim()));if(!r.hidden)n++;}}document.getElementById('count').textContent=n+'개 표시';}}for(const el of [sector,status,query])el.addEventListener('input',filter);filter();</script></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, default=ROOT / 'regime_router_publish.db')
    stock_source = parser.add_mutually_exclusive_group()
    stock_source.add_argument('--stock-database', type=Path, default=STOCK_DATABASE,
                              help='Read-only Beta ohlcv database for the entire stock-price series')
    stock_source.add_argument('--market-lens-only', action='store_true',
                              help='Explicitly use --database daily_prices for stocks as well as benchmarks')
    parser.add_argument('--metadata-db', type=Path, default=METADATA_DB)
    parser.add_argument('--mapping', type=Path, default=OUTPUT / 'classifications.json')
    parser.add_argument('--hybrid', type=Path, default=ROOT / 'outputs/sector_hybrid_site_latest.json')
    mp_source = parser.add_mutually_exclusive_group()
    mp_source.add_argument('--mp-tickers', type=Path, help='Explicit MP ticker list overriding the SS-imported snapshot')
    mp_source.add_argument('--mp-database', type=Path, default=MP_DATABASE, help='Read-only SS-imported MP holdings database')
    parser.add_argument('--news-project', type=Path, default=NEWS_PROJECT)
    parser.add_argument('--skip-news', action='store_true', help='Skip local saved-news evidence reading')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--refresh-classifications', action='store_true', help='Opt-in: download one free public component CSV')
    parser.add_argument('--open', action='store_true')
    args = parser.parse_args()
    if args.refresh_classifications:
        refresh_classifications(args.mapping)
    mapping, source, stamp = classifications(args.mapping, args.metadata_db)
    as_of, tickers, matrices = load_prices(args.database, None if args.market_lens_only else args.stock_database)
    if args.mp_tickers:
        held = {line.strip().upper().replace('.', '-') for line in args.mp_tickers.read_text(encoding='utf-8-sig').splitlines() if line.strip()}
        if not held or any(not re.fullmatch(r'[A-Z0-9][A-Z0-9-]{0,14}', t) for t in held):
            raise ValueError('Invalid or empty MP ticker list')
        mp = {'source': '사용자 지정 MP 티커 명단', 'as_of': None, 'status': 'USER_LIST', 'held': sorted(held)}
    else:
        mp = load_mp(args.mp_database, as_of)
    result = screen(as_of, tickers, matrices, mapping, set(mp.get('held', [])))
    mapped = sum(t in mapping for t in tickers)
    result['universe'] = {'name': 'Current S&P500', 'count': len(tickers), 'mapped_count': mapped,
                          'mapping_source': source, 'mapping_as_of': stamp}
    result['warnings'] = ['실적·가치·촉매 검증 전 가격 스크리닝입니다. 성과 개선은 아직 검증하지 않았습니다.']
    if mapped < len(tickers):
        result['warnings'].append(f'분류 없는 {len(tickers) - mapped}개는 후보에서 제외. 전체 시장 최상위라는 뜻이 아닙니다.')
    if result['price_coverage']['price_missing_count']:
        result['warnings'].append(f"선택한 주가 출처의 126세션 요건 미충족 {result['price_coverage']['price_missing_count']}개는 평가 불가입니다. 다른 출처로 보충하지 않습니다.")
    if stamp and (date.today() - date.fromisoformat(stamp[:10])).days > 30:
        result['warnings'].append('분류 캐시가 30일 초과: 업종/상장 변경을 확인하세요.')
    stale = (date.today() - date.fromisoformat(as_of)).days > 4
    if stale:
        result['warnings'].append('가격이 4일 넘게 오래되어 검토 후보를 보류합니다. 기존 Market Lens 최신화 후 다시 실행하세요.')
        for row in result['candidates']:
            if row['status'] == 'CANDIDATE':
                row['status'] = 'STALE'
    if len(result['sectors']) != 11:
        result['warnings'].append('일부 섹터 ETF 가격 결측: 섹터 순위도 부분 집합 기준입니다.')
    attach_hybrid(result, args.hybrid)
    evidence_at = datetime.now(timezone.utc).isoformat()
    news = ({'status': 'SKIPPED', 'cutoff': evidence_at, 'by_ticker': {}} if args.skip_news else
            load_news(args.news_project, evidence_at, set(tickers)))
    attach_pitch_evidence(result, mp, news)
    result['status'] = 'PARTIAL'  # Fundamentals/MP validation are not a completed selection engine.
    result['generated_at'] = datetime.now(timezone.utc).isoformat()
    save(args.output / 'snapshot.json', json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
    save(args.output / 'index.html', render_report(result))
    print(json.dumps({'status': result['status'], 'as_of': as_of, 'universe': len(tickers), 'mapped': mapped,
                      'stock_price_source': result['price_sources']['stock']['provider'],
                      'price_coverage': result['price_coverage'],
                      'candidates': sum(r['status'] == 'CANDIDATE' for r in result['candidates']),
                      'new_mp_price_candidates': sum(r['pitch_status'] == 'DUE_DILIGENCE' for r in result['candidates']),
                      'report': str((args.output / 'index.html').resolve())}, ensure_ascii=False))
    if args.open:
        webbrowser.open((args.output / 'index.html').resolve().as_uri())


if __name__ == '__main__':
    main()
