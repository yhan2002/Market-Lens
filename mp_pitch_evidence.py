"""Read-only MP identity and grounded saved-news evidence; never fetch or infer."""
import importlib
import json
import math
import re
import sqlite3
import sys
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit


@contextmanager
def _readonly(path):
    connection = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)
    try:
        connection.execute('PRAGMA query_only=ON')
        yield connection
    finally:
        connection.close()


def load_mp(database, as_of):
    """SS imported holdings, including shorts; unknown mappings never imply absence."""
    result = dict(source='SS 가공용 보유 스냅샷', as_of=None, status='MISSING', held=[],
                  unmapped_positions=0, invalid_quantities=0, stale=False)
    if not Path(database).is_file():
        return dict(result, reason='MP holdings database missing')
    try:
        cutoff = date.fromisoformat(as_of)
        with _readonly(database) as connection:
            columns = {row[1] for row in connection.execute('PRAGMA table_info(daily_holdings)')}
            if not {'date', 'ticker', 'quantity'} <= columns:
                return dict(result, reason='daily_holdings schema missing')
            stamp = connection.execute('SELECT MAX(date) FROM daily_holdings WHERE date<=?',
                                       (cutoff.isoformat(),)).fetchone()[0]
            if stamp is None:
                return dict(result, reason='No holdings snapshot on or before cutoff')
            result.update(as_of=stamp, stale=(cutoff-date.fromisoformat(stamp)).days > 14)
            rows = connection.execute('SELECT ticker, quantity FROM daily_holdings WHERE date=?',
                                      (stamp,)).fetchall()
        held = set()
        for ticker, quantity in rows:
            valid_quantity = isinstance(quantity, (int, float)) and math.isfinite(quantity)
            if valid_quantity and quantity == 0:
                continue
            if not valid_quantity:
                result['invalid_quantities'] += 1
            if isinstance(ticker, str) and re.fullmatch(r'[A-Z0-9][A-Z0-9.\-]{0,20}', ticker.strip().upper()):
                if valid_quantity:
                    held.add(ticker.strip().upper().replace('.', '-'))
            else:
                result['unmapped_positions'] += 1
        result['held'] = sorted(held)
        result['status'] = ('INCOMPLETE' if result['unmapped_positions'] or result['invalid_quantities']
                            else 'STALE' if result['stale'] else 'READY')
        return result
    except (OSError, sqlite3.Error, ValueError, TypeError, OverflowError) as exc:
        return dict(result, status='ERROR', reason=type(exc).__name__)


def _timestamp(value):
    if not isinstance(value, str):
        raise ValueError('Timestamp missing')
    stamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError('Timezone missing')
    return stamp.astimezone(timezone.utc)


@contextmanager
def _news_helpers(project):
    """Reuse the sibling's readers without retaining ambiguous top-level imports."""
    names = ('universe', 'news', 'content_assessment', 'first_pass', 'company_links', 'relation_verifier')
    previous = set(sys.modules)
    for name in names:
        module = sys.modules.get(name)
        if module is not None and Path(getattr(module, '__file__', '')).resolve() != project / (name + '.py'):
            raise ImportError('Conflicting News Alarm dependency')
    sys.path.insert(0, str(project))
    try:
        ca, first_pass, links = (importlib.import_module(name) for name in
                                 ('content_assessment', 'first_pass', 'company_links'))
        for module in (ca, first_pass, links):
            if Path(module.__file__).resolve().parent != project:
                raise ImportError('Unexpected News Alarm dependency location')
        yield ca, first_pass, links
    finally:
        sys.path.remove(str(project))
        for name in names:
            if name not in previous:
                sys.modules.pop(name, None)


def load_news(project: Path, cutoff_iso: str, tickers: set[str]) -> dict:
    """Latest three saved, source-matched company quotes; not a catalyst ranking."""
    result = dict(source='News Alarm 저장 분석', status='MISSING', cutoff=cutoff_iso,
                  window_start=None, by_ticker={}, scanned_articles=0, excluded_articles=0,
                  capped=False, notes=[
                      '게시·최초 수집·평가 시각이 모두 기준 시각 이내인 저장 근거만 연결합니다.',
                      '현재 저장 본문과 평가 입력의 일치만 검사하며 과거 본문 vintage/PIT는 보장하지 않습니다.',
                      '기업 관련성 근거이며 호재·악재 점수 또는 주가 변화의 인과관계를 뜻하지 않습니다.'])
    project = Path(project).resolve()
    database, cache = project/'data/news.sqlite', project/'data/first-pass.sqlite'
    if not database.is_file() or not cache.is_file():
        return dict(result, reason='Saved news or first-pass database missing')
    if any(not (project/(name+'.py')).is_file() for name in
           ('content_assessment', 'first_pass', 'company_links', 'news', 'universe')):
        return dict(result, reason='News Alarm helper modules missing')
    try:
        cutoff = _timestamp(cutoff_iso)
        start = cutoff-timedelta(days=14)
        result.update(cutoff=cutoff.isoformat(), window_start=start.isoformat())
        wanted = {ticker.upper().replace('.', '-') for ticker in tickers}
        if not wanted:
            return dict(result, status='READY')
        with _readonly(database) as connection:
            # Bound materialized rows, not just rendered results. SQLite normalizes offsets.
            published = "CASE WHEN json_valid(body) THEN julianday(json_extract(body,'$.published_at')) END"
            rows = connection.execute(
                f'SELECT id, body FROM articles WHERE {published} BETWEEN julianday(?) AND julianday(?) '
                f'ORDER BY {published} DESC, id LIMIT 5001', (start.isoformat(), cutoff.isoformat())).fetchall()
        result['capped'] = len(rows) > 5000
        rows = rows[:5000]
        result['scanned_articles'] = len(rows)
        evidence = {}
        with _news_helpers(project) as (ca, first_pass, company_links), _readonly(cache) as connection:
            # Assert schema even if there are no recent articles.
            connection.execute('SELECT key, body FROM first_pass LIMIT 0')
            for article_id, body in rows:
                try:
                    article = json.loads(body)
                    article.pop('qwen', None)
                    if article.get('id') != article_id or not isinstance(article.get('title'), str):
                        raise ValueError('Invalid article identity')
                    published_at = _timestamp(article.get('published_at'))
                    collected_at = _timestamp(article.get('first_collected_at'))
                    url = urlsplit(article['url'])
                    if (url.scheme not in ('https', 'http') or not url.hostname or url.username or url.password
                            or any(ord(c) < 32 for c in article['url'])):
                        raise ValueError('Unsafe article URL')
                    saved = connection.execute('SELECT body FROM first_pass WHERE key=?',
                                               (ca.fingerprint(article),)).fetchone()
                    record = json.loads(saved[0]) if saved else {}
                    assessed_at = _timestamp(record.get('at'))
                    available_at = max(published_at, collected_at, assessed_at)
                    if not start <= published_at <= cutoff or available_at > cutoff:
                        raise ValueError('Unavailable at cutoff')
                    if record.get('status') != 'done' or not first_pass.current(record) or not first_pass.same_input(article, record):
                        raise ValueError('Stale or incomplete assessment')
                    companies = record.get('facts', {}).get('companies', {})
                    if companies.get('version') != 1 or not isinstance(companies.get('rows'), list):
                        raise ValueError('Grounded company assessment missing')
                    article['qwen'] = record
                    text, _ = first_pass.evidence_input(article)
                    for link in company_links.resolve(article):
                        ticker = link['symbol'].upper().replace('.', '-')
                        quotes = link.get('evidence', [])
                        if ticker not in wanted or link.get('basis') != 'model_company_relevance':
                            continue
                        quote = next((q for q in quotes if isinstance(q, str) and q.strip() and q in text), None)
                        if quote is None:
                            continue
                        item = dict(id=article_id, title=article['title'], url=article['url'], quote=quote,
                                    published_at=published_at.isoformat(), collected_at=collected_at.isoformat(),
                                    assessed_at=assessed_at.isoformat(), available_at=available_at.isoformat())
                        evidence.setdefault(ticker, {})[article_id] = item
                except (ValueError, TypeError, KeyError, AttributeError, IndexError, OverflowError):
                    result['excluded_articles'] += 1
        result['by_ticker'] = {ticker: sorted(items.values(), key=lambda item:
                              (item['published_at'], item['available_at'], item['id']), reverse=True)[:3]
                              for ticker, items in sorted(evidence.items())}
        result['status'] = 'PARTIAL' if result['capped'] or result['excluded_articles'] else 'READY'
        if result['capped']:
            result['notes'].append('최근 14일 조회가 5,000건 한도에 도달하여 일부 기사는 미검토 상태입니다.')
        return result
    except (OSError, sqlite3.Error, ImportError, ValueError, TypeError, AttributeError) as exc:
        return dict(result, status='ERROR', reason=type(exc).__name__)
