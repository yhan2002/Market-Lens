"""Synthetic SQLite fixtures, actual pure News Alarm helpers, no network or models."""
import hashlib
import importlib.util
import json
import sqlite3
import sys
import tempfile
import types
import unittest
from contextlib import closing, contextmanager
from pathlib import Path
from unittest.mock import patch

import mp_pitch_evidence as evidence


class MPChecks(unittest.TestCase):
    def test_snapshot_cutoff_shorts_unknown_and_staleness(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'mp.db'
            self.assertEqual(evidence.load_mp(path, '2026-09-29')['status'], 'MISSING')
            self.assertFalse(path.exists())
            with closing(sqlite3.connect(path)) as connection, connection:
                connection.execute('CREATE TABLE daily_holdings(date, ticker, quantity)')
                connection.executemany('INSERT INTO daily_holdings VALUES (?,?,?)', [
                    ('2026-09-25', 'AAA', 10), ('2026-09-25', 'BBB', -3),
                    ('2026-09-25', 'BRK.B', 2), ('2026-09-25', None, 0),
                    ('2026-09-30', 'FUTURE', 3)])
            before = path.read_bytes()
            result = evidence.load_mp(path, '2026-09-29')
            self.assertEqual((result['status'], result['as_of'], result['held']),
                             ('READY', '2026-09-25', ['AAA', 'BBB', 'BRK-B']))
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(evidence.load_mp(path, '2026-09-20')['status'], 'MISSING')
            self.assertEqual(evidence.load_mp(path, '2026-10-14')['status'], 'READY')
            self.assertEqual(evidence.load_mp(path, '2026-10-15')['status'], 'STALE')
            with closing(sqlite3.connect(path)) as connection, connection:
                connection.executemany('INSERT INTO daily_holdings VALUES (?,?,?)', [
                    ('2026-09-25', None, 1), ('2026-09-25', 'BADQ', float('inf')),
                    ('2026-09-25', None, 'broken')])
            result = evidence.load_mp(path, '2026-09-29')
            self.assertEqual(result['status'], 'INCOMPLETE')
            self.assertEqual(result['unmapped_positions'], 2)
            self.assertEqual(result['invalid_quantities'], 2)
            self.assertIn('AAA', result['held'])
            self.assertNotIn('BADQ', result['held'])
            self.assertNotIn('quantity', result)
            with evidence._readonly(path) as connection:
                with self.assertRaises(sqlite3.OperationalError):
                    connection.execute('DELETE FROM daily_holdings')

    def test_missing_table_and_invalid_input(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'mp.db'
            sqlite3.connect(path).close()
            self.assertEqual(evidence.load_mp(path, '2026-09-29')['status'], 'MISSING')
            self.assertEqual(evidence.load_mp(path, 'bad date')['status'], 'ERROR')


class NewsChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parent.parent/'News Alarm'
        if not all((source/(name+'.py')).is_file() for name in
                   ('content_assessment', 'first_pass', 'company_links')):
            raise unittest.SkipTest('Sibling News Alarm pure helpers unavailable')
        cls.news = types.ModuleType('news')
        cls.news.UNIVERSE = {'AAA': ('Example company', [], []), 'BBB': ('Other company', [], [])}
        cls.news.entities = lambda article: (_ for _ in ()).throw(AssertionError('Keyword fallback called'))

        def module(name):
            spec = importlib.util.spec_from_file_location('_pitch_test_'+name, source/(name+'.py'))
            out = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(out)
            return out

        cls.ca = module('content_assessment')
        with patch.dict(sys.modules, {'content_assessment': cls.ca, 'news': cls.news}):
            cls.first_pass = module('first_pass')
        cls.links = module('company_links')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/'data').mkdir()
        for name in ('content_assessment', 'first_pass', 'company_links', 'news', 'universe'):
            (self.root/(name+'.py')).touch()
        self.db = self.root/'data/news.sqlite'
        self.cache = self.root/'data/first-pass.sqlite'
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.execute('CREATE TABLE articles(id TEXT PRIMARY KEY, body TEXT)')
        with closing(sqlite3.connect(self.cache)) as connection, connection:
            connection.execute('CREATE TABLE first_pass(key TEXT PRIMARY KEY, body TEXT)')

    def add(self, article_id, *, article_changes=None, record_changes=None, save_record=True):
        article = dict(id=article_id, title='AAA announced a new contract. '+article_id,
                       summary='', url='https://example.com/'+article_id,
                       published_at='2026-09-28T10:00:00+00:00', first_collected_at='2026-09-28T10:05:00Z')
        article.update(article_changes or {})
        text = article['title']+'\n'+article['summary']
        record = dict(version=self.first_pass.VERSION, model=self.first_pass.MODEL,
                      status='done', coverage='제목·RSS 요약', at='2026-09-28T10:06:00Z',
                      text_hash=hashlib.sha256((article['published_at']+'\n'+text).encode()).hexdigest(),
                      facts={'companies': {'version': 1, 'rows': [
                          {'symbol': 'AAA', 'decision': 'relevant', 'quote': article['title']},
                          {'symbol': 'AAA', 'decision': 'relevant', 'quote': article['title']},
                          {'symbol': 'BBB', 'decision': 'unrelated', 'quote': article['title']}]}})
        record.update(record_changes or {})
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.execute('INSERT INTO articles VALUES (?,?)', (article_id, json.dumps(article)))
        if save_record:
            with closing(sqlite3.connect(self.cache)) as connection, connection:
                connection.execute('INSERT INTO first_pass VALUES (?,?)',
                                   (self.ca.fingerprint(article), json.dumps(record)))

    @contextmanager
    def helpers(self, project):
        self.assertEqual(project, self.root.resolve())
        with patch.object(self.ca, 'ROOT', self.root), patch.dict(sys.modules, {'news': self.news}):
            yield self.ca, self.first_pass, self.links

    def load(self, cutoff='2026-09-29T00:00:00Z'):
        with patch.object(evidence, '_news_helpers', self.helpers):
            return evidence.load_news(self.root, cutoff, {'AAA', 'BBB'})

    def test_grounded_latest_three_and_no_source_mutation(self):
        for day in range(24, 29):
            self.add(str(day), article_changes={'published_at': f'2026-09-{day}T10:00:00Z'})
        before = (self.db.read_bytes(), self.cache.read_bytes())
        result = self.load()
        self.assertEqual(result['status'], 'READY')
        self.assertEqual([item['id'] for item in result['by_ticker']['AAA']], ['28', '27', '26'])
        self.assertNotIn('BBB', result['by_ticker'])
        self.assertEqual(result['by_ticker']['AAA'][0]['available_at'], '2026-09-28T10:06:00+00:00')
        self.assertEqual(before, (self.db.read_bytes(), self.cache.read_bytes()))

    def test_current_hash_timestamps_safe_url_and_no_keyword_fallback(self):
        self.add('good')
        self.add('future-published', article_changes={'published_at': '2026-09-30T00:00:00Z'})
        self.add('old', article_changes={'published_at': '2026-09-01T00:00:00Z'})
        self.add('late-collection', article_changes={'first_collected_at': '2026-09-30T00:00:00Z'})
        self.add('late-assessment', record_changes={'at': '2026-09-30T00:00:00Z'})
        self.add('naive', article_changes={'published_at': '2026-09-28T00:00:00'})
        self.add('missing-collection', article_changes={'first_collected_at': None})
        self.add('stale-model', record_changes={'version': -1})
        self.add('changed-input', record_changes={'text_hash': 'wrong'})
        self.add('bad-url', article_changes={'url': 'javascript:alert(1)'})
        self.add('credentials', article_changes={'url': 'https://user:password@example.com/a'})
        self.add('no-company-review', record_changes={'facts': {}})
        self.add('empty-quote', record_changes={'facts': {'companies': {'version': 1, 'rows': [
            {'symbol': 'AAA', 'decision': 'relevant', 'quote': ''}]}}})
        self.add('raw-qwen-is-not-cache', save_record=False, article_changes={'qwen': {
            'status': 'done', 'facts': {'companies': {'version': 1, 'rows': []}}}})
        result = self.load()
        self.assertEqual(result['status'], 'PARTIAL')
        self.assertEqual([item['id'] for item in result['by_ticker']['AAA']], ['good'])
        self.assertEqual(result['scanned_articles'], 12)
        self.assertEqual(result['excluded_articles'], 10)
        self.assertEqual(self.load('2026-09-29')['status'], 'ERROR')

    def test_missing_dependencies_and_bounded_scan(self):
        missing = self.root/'missing'
        self.assertEqual(evidence.load_news(missing, '2026-09-29T00:00:00Z', {'AAA'})['status'], 'MISSING')
        self.assertFalse(missing.exists())
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.executemany('INSERT INTO articles VALUES (?,?)', [
                (str(i), json.dumps({'id': str(i), 'published_at': '2026-09-28T00:00:00Z'}))
                for i in range(5001)])
        result = self.load()
        self.assertEqual((result['status'], result['scanned_articles'], result['capped']), ('PARTIAL', 5000, True))
        self.assertTrue(any('5,000' in note for note in result['notes']))
        with patch.object(evidence, '_news_helpers', side_effect=ImportError('not displayed secret')):
            result = evidence.load_news(self.root, '2026-09-29T00:00:00Z', {'AAA'})
        self.assertEqual(result['reason'], 'ImportError')
        self.assertNotIn('secret', str(result))


if __name__ == '__main__':
    unittest.main()
