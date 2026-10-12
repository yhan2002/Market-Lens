"""Fast offline checks; synthetic prices and temporary files only."""
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

from sector_rotation import SECTOR_ETFS
from us_topdown_screener import attach_hybrid, attach_pitch_evidence, load_prices, pitch_details, readonly, render_report, screen


class TopdownChecks(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range('2025-01-01', periods=225).strftime('%Y-%m-%d')
        series = {'SPY': 100 * 1.0005 ** np.arange(225)}
        series.update({t: 100 * 1.001 ** np.arange(225) for t in SECTOR_ETFS})
        series.update(AAA=100 * 1.002 ** np.arange(225), BBB=100 * 1.0015 ** np.arange(225),
                      HOT=100 * 1.025 ** np.arange(225), UNKNOWN=100 * 1.002 ** np.arange(225))
        frame = pd.DataFrame(series, index=self.dates)
        self.matrices = {'adj_close': frame, 'close': frame.copy(), 'volume': frame * 0 + 1_000_000}
        self.mapping = {t: {'name': t, 'sector_etf': 'XLK'} for t in ('AAA', 'BBB', 'HOT')}

    def result(self, end=None, held=None):
        return screen(end or self.dates[-1], ['AAA', 'BBB', 'HOT', 'UNKNOWN'], self.matrices, self.mapping, held)

    def price_databases(self, folder):
        lens, beta = Path(folder) / 'lens.db', Path(folder) / 'beta.db'
        with closing(sqlite3.connect(lens)) as con, con:
            con.executescript('''CREATE TABLE sp500_membership(ticker,effective_from,effective_to,membership_as_of,available_at);
                CREATE TABLE daily_prices(ticker,date,adj_close,close,volume);''')
            con.executemany('INSERT INTO sp500_membership VALUES(?,?,NULL,?,?)',
                            [(t, self.dates[0], self.dates[0], self.dates[0]) for t in ['AAA', 'BBB', 'HOT', 'UNKNOWN']])
            con.executemany('INSERT INTO daily_prices VALUES(?,?,?,?,?)', [
                (ticker, day, float(self.matrices['adj_close'].loc[day, ticker]),
                 float(self.matrices['close'].loc[day, ticker]), 1_000_000)
                for ticker in self.matrices['adj_close'] for i, day in enumerate(self.dates)
                if ticker != 'AAA' or i < len(self.dates) - 7])
            con.execute("INSERT INTO daily_prices VALUES('SPY','2099-01-01',999999,999999,1000)")
        with closing(sqlite3.connect(beta)) as con, con:
            con.execute('CREATE TABLE ohlcv(ticker,date,adj_close,close,volume)')
            con.executemany('INSERT INTO ohlcv VALUES(?,?,?,?,?)', [
                ('AAA', day, 200 * 1.003 ** i, 220 * 1.003 ** i, 1_000_000)
                for i, day in enumerate(self.dates)])
            con.executemany('INSERT INTO ohlcv VALUES(?,?,?,?,?)', [
                ('BBB', day, 300., 300., 1_000_000) for day in self.dates[-20:]])
            con.executemany('INSERT INTO ohlcv VALUES(?,?,?,?,?)', [
                ('HOT', day, None, 300., 1_000_000) for day in self.dates])
            con.executemany('INSERT INTO ohlcv VALUES(?,?,?,?,?)', [
                ('SPY', self.dates[-1], 999999, 999999, 1_000_000),
                ('XLK', self.dates[-1], 999999, 999999, 1_000_000),
                ('NOT_MEMBER', self.dates[-1], 999999, 999999, 1_000_000),
                ('AAA', '2099-01-01', 999999, 999999, 1_000_000)])
        return lens, beta

    def test_candidates_exclusions_and_within_sector_rank(self):
        # Add realistic volatility so monotonic fixture is not flagged on zero variance.
        for t in ['AAA', 'BBB']:
            wave = 1 + .012 * np.sin(np.arange(225) * 1.7)
            self.matrices['adj_close'][t] *= wave
        rows = {r['ticker']: r for r in self.result()['candidates']}
        self.assertEqual(rows['UNKNOWN']['status'], 'UNMAPPED')
        self.assertEqual(rows['HOT']['status'], 'EXTENDED')
        self.assertEqual(rows['AAA']['status'], 'CANDIDATE')
        self.assertEqual(rows['AAA']['mp_status'], 'UNKNOWN')
        self.assertLess(rows['AAA']['rank'], rows['BBB']['rank'])
        self.assertEqual({r['ticker']: r for r in self.result(held={'AAA'})['candidates']}['AAA']['status'], 'HELD')

    def test_future_prices_are_excluded(self):
        end = self.dates[-10]
        before = self.result(end)
        for matrix in self.matrices.values():
            matrix.loc[matrix.index > end, 'AAA'] = 1e12
        self.assertEqual(before, self.result(end))

    def test_no_fill_and_missing_liquidity(self):
        self.matrices['adj_close'].loc[self.dates[-5], 'AAA'] = np.nan
        self.matrices['volume'].loc[self.dates[-5], 'BBB'] = np.nan
        rows = {r['ticker']: r for r in self.result()['candidates']}
        self.assertEqual(rows['AAA']['status'], 'DATA_MISSING')
        self.assertEqual(rows['AAA']['setup'], 'NOT_EVALUATED')
        self.assertFalse(rows['AAA']['setup_evaluated'])
        self.assertEqual(rows['BBB']['status'], 'DATA_MISSING')
        self.assertIsNone(rows['AAA']['score'])
        self.assertIsNone(rows['BBB']['score'])
        held_rows = {r['ticker']: r for r in self.result(held={'BBB'})['candidates']}
        self.assertEqual(held_rows['BBB']['status'], 'DATA_MISSING')
        self.assertEqual(held_rows['BBB']['mp_status'], 'HELD')
        self.assertIsNone(held_rows['BBB']['score'])

    def test_beta_full_series_and_lens_benchmarks_are_readonly(self):
        with tempfile.TemporaryDirectory() as folder:
            lens, beta = self.price_databases(folder)
            before = {path: path.read_bytes() for path in (lens, beta)}
            as_of, tickers, matrices = load_prices(lens, beta)
            self.assertEqual(as_of, self.dates[-1])
            self.assertEqual(tickers, ['AAA', 'BBB', 'HOT', 'UNKNOWN'])
            prices = matrices['adj_close']
            self.assertEqual(prices.loc[self.dates[0], 'AAA'], 200)
            self.assertAlmostEqual(prices.loc[as_of, 'AAA'], 200 * 1.003 ** 224)
            for ticker in ['SPY', *SECTOR_ETFS]:
                pd.testing.assert_series_equal(prices[ticker], self.matrices['adj_close'][ticker], check_names=False)
            self.assertNotIn('NOT_MEMBER', prices)
            self.assertNotIn('2099-01-01', prices.index)
            result = screen(as_of, tickers, matrices, self.mapping)
            self.assertEqual(result['price_sources']['stock']['provider'], 'QuantBot Beta')
            self.assertEqual(result['price_sources']['stock']['adjustment_provenance'], 'UNVERIFIED')
            self.assertIn('Close를 adj_close', result['price_sources']['stock']['adjustment_note'])
            self.assertFalse(result['price_sources']['point_in_time'])
            row = next(r for r in result['candidates'] if r['ticker'] == 'AAA')
            self.assertEqual(row['last_price_date'], as_of)
            self.assertEqual(row['missing_latest_sessions'], 0)
            self.assertTrue(row['setup_evaluated'])
            for path, table in [(lens, 'daily_prices'), (beta, 'ohlcv')]:
                with readonly(path) as con:
                    with self.assertRaises(sqlite3.OperationalError):
                        con.execute(f'UPDATE {table} SET volume=0')
                self.assertEqual(path.read_bytes(), before[path])
            self.assertEqual(set(Path(folder).iterdir()), {lens, beta})

    def test_beta_missing_history_never_stitches_or_uses_close(self):
        with tempfile.TemporaryDirectory() as folder:
            lens, beta = self.price_databases(folder)
            as_of, tickers, matrices = load_prices(lens, beta)
            self.assertTrue(matrices['adj_close']['BBB'].iloc[:-20].isna().all())
            self.assertTrue(matrices['adj_close']['HOT'].isna().all())
            self.assertTrue(matrices['adj_close']['UNKNOWN'].isna().all())
            result = screen(as_of, tickers, matrices, self.mapping)
            rows = {r['ticker']: r for r in result['candidates']}
            for ticker in ['BBB', 'HOT', 'UNKNOWN']:
                self.assertEqual(rows[ticker]['status'], 'DATA_MISSING')
                self.assertEqual(rows[ticker]['setup'], 'NOT_EVALUATED')
                self.assertFalse(rows[ticker]['setup_evaluated'])
                self.assertIsNone(rows[ticker]['score'])
            self.assertEqual(rows['BBB']['valid_sessions126'], 20)
            self.assertEqual(result['price_coverage']['price_evaluated_count'], 1)
            self.assertEqual(result['price_coverage']['price_missing_count'], 3)
            result['warnings'] = []
            result['universe'] = {'count': 4, 'mapped_count': 3, 'mapping_source': 'test', 'mapping_as_of': None}
            page = render_report(result)
            self.assertIn('126세션 가격 평가 1/4', page)
            self.assertIn('평가 불가', page)
            self.assertIn('QuantBot Beta', page)
            self.assertIn('조정 이력은 보증되지 않습니다', page)
            json.dumps(result, allow_nan=False)

    def test_beta_stale_and_invalid_prices_fail_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            lens, beta = self.price_databases(folder)
            with closing(sqlite3.connect(beta)) as con, con:
                con.execute("DELETE FROM ohlcv WHERE ticker='AAA' AND date>?", (self.dates[-4],))
            as_of, tickers, matrices = load_prices(lens, beta)
            row = next(r for r in screen(as_of, tickers, matrices, self.mapping)['candidates'] if r['ticker'] == 'AAA')
            self.assertEqual(row['last_price_date'], self.dates[-4])
            self.assertEqual(row['missing_latest_sessions'], 3)
            self.assertEqual(row['valid_sessions126'], 123)
            self.assertEqual(row['status'], 'DATA_MISSING')
            with closing(sqlite3.connect(beta)) as con, con:
                con.execute("UPDATE ohlcv SET adj_close='broken' WHERE ticker='BBB' AND date=?", (as_of,))
            _, _, matrices = load_prices(lens, beta)
            self.assertTrue(pd.isna(matrices['adj_close'].loc[as_of, 'BBB']))
            row = next(r for r in screen(as_of, tickers, matrices, self.mapping)['candidates'] if r['ticker'] == 'BBB')
            self.assertEqual(row['status'], 'DATA_MISSING')
            self.assertEqual(row['setup'], 'NOT_EVALUATED')

    def test_missing_or_duplicate_beta_never_falls_back(self):
        with tempfile.TemporaryDirectory() as folder:
            lens, beta = self.price_databases(folder)
            missing = Path(folder) / 'missing.db'
            with self.assertRaises(sqlite3.OperationalError):
                load_prices(lens, missing)
            self.assertFalse(missing.exists())
            with closing(sqlite3.connect(beta)) as con, con:
                con.execute('INSERT INTO ohlcv SELECT * FROM ohlcv WHERE ticker=? AND date=?', ('AAA', self.dates[-1]))
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                load_prices(lens, beta)
            # Explicit legacy mode still reads only the complete Market Lens database.
            _, _, matrices = load_prices(lens)
            self.assertEqual(matrices['price_sources']['stock']['provider'], 'Market Lens')
            self.assertEqual(matrices['adj_close']['AAA'].iloc[0], 100)

    def test_beta_bad_schema_and_dates_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            lens, beta = self.price_databases(folder)
            with closing(sqlite3.connect(beta)) as con, con:
                con.execute("UPDATE ohlcv SET date='2025-09-xx' WHERE ticker='AAA' AND date=?", (self.dates[-1],))
            with self.assertRaisesRegex(ValueError, 'Invalid price date'):
                load_prices(lens, beta)
            with closing(sqlite3.connect(beta)) as con, con:
                con.execute('DROP TABLE ohlcv')
                con.execute('CREATE TABLE ohlcv(ticker,date,close,volume)')
            with self.assertRaises(pd.errors.DatabaseError):
                load_prices(lens, beta)

    def test_current_membership_excludes_exited_and_unavailable(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'prices.db'
            con = sqlite3.connect(path)
            con.executescript('''CREATE TABLE sp500_membership(ticker,effective_from,effective_to,membership_as_of,available_at);
                CREATE TABLE daily_prices(ticker,date,adj_close,close,volume);
                INSERT INTO daily_prices VALUES('SPY','2025-09-02',100,100,1000);
                INSERT INTO sp500_membership VALUES('OK','2025-01-01',NULL,'2025-01-01','2025-01-01');
                INSERT INTO sp500_membership VALUES('EXIT','2025-01-01','2025-08-01','2025-01-01','2025-01-01');
                INSERT INTO sp500_membership VALUES('FUTURE','2025-01-01',NULL,'2025-01-01','2099-01-01');''')
            con.commit()
            con.close()
            _, tickers, _ = load_prices(path)
            self.assertEqual(tickers, ['OK'])

    def test_readonly_is_enforced(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'test.db'
            con = sqlite3.connect(path)
            con.execute('CREATE TABLE x(n)')
            con.close()
            with readonly(path) as read:
                self.assertEqual(read.execute('SELECT COUNT(*) FROM x').fetchone()[0], 0)
                with self.assertRaises(sqlite3.OperationalError):
                    read.execute('INSERT INTO x VALUES(1)')

    def test_hybrid_asof_and_weekly_date(self):
        result = self.result()
        result['warnings'] = []
        end = result['as_of']
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'hybrid.json'
            payload = {'as_of': end, 'decision_date': self.dates[-3], 'status': 'READY',
                       'rows': [{'ticker': 'XLK', 'signal': 'IN', 'status': 'READY'}]}
            path.write_text(json.dumps(payload), encoding='utf-8')
            attach_hybrid(result, path)
            self.assertEqual(next(r for r in result['sectors'] if r['ticker'] == 'XLK')['hybrid_signal'], 'IN')
            payload['as_of'] = '2099-01-01'
            path.write_text(json.dumps(payload), encoding='utf-8')
            fresh = self.result()
            fresh['warnings'] = []
            attach_hybrid(fresh, path)
            self.assertTrue(fresh['warnings'])
            self.assertTrue(all(r['hybrid_signal'] is None for r in fresh['sectors']))

    def test_html_untrusted_names_are_escaped(self):
        result = self.result()
        result['candidates'][0]['name'] = '<img src=x onerror=alert(1)>'
        result['warnings'] = ['<script>alert(1)</script>']
        result['universe'] = {'count': 4, 'mapped_count': 3, 'mapping_source': 'test', 'mapping_as_of': None}
        page = render_report(result)
        self.assertNotIn('<img src=x', page)
        self.assertIn('&lt;img src=x', page)
        self.assertNotIn('<script>alert(1)</script>', page)
        self.assertIn('N/A', page)
        json.dumps(result, allow_nan=False)

    def test_pitch_evidence_never_changes_scores_or_claims_ready(self):
        result = self.result()
        result['warnings'] = []
        scores = {r['ticker']: r['score'] for r in result['candidates']}
        mp = {'source': 'SS', 'as_of': result['as_of'], 'status': 'INCOMPLETE', 'held': ['AAA']}
        article = {'id': '1', 'title': '<script>bad</script>', 'url': 'https://example.com/a',
                   'quote': '<img src=x>', 'published_at': '2025-01-01T00:00:00Z',
                   'collected_at': '2025-01-02T00:00:00Z', 'assessed_at': '2025-01-03T00:00:00Z'}
        attach_pitch_evidence(result, mp, {'status': 'READY', 'cutoff': '2025-01-03T00:00:00Z',
                                          'by_ticker': {'BBB': [article]}})
        rows = {r['ticker']: r for r in result['candidates']}
        self.assertEqual(rows['AAA']['mp_status'], 'HELD')
        self.assertEqual(rows['BBB']['mp_status'], 'UNKNOWN')
        self.assertIn('최신 MP 보유 여부', rows['BBB']['pitch_gaps'])
        self.assertIn('실적 성장·가이던스', rows['BBB']['pitch_gaps'])
        self.assertEqual(scores, {r['ticker']: r['score'] for r in result['candidates']})
        self.assertNotIn('held', result['mp'])
        self.assertNotIn('by_ticker', result['news_context'])
        details = pitch_details(rows['BBB'])
        self.assertNotIn('<script>bad', details)
        self.assertIn('&lt;img src=x&gt;', details)
        rows['BBB']['news'][0]['url'] = 'javascript:alert(1)'
        self.assertNotIn('javascript:', pitch_details(rows['BBB']))


if __name__ == '__main__':
    unittest.main()
