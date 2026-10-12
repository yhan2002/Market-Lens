"""Offline, synthetic review inputs only; no accounts, networks or models."""
import copy
import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import mp_pitch_review as review


NOW = datetime(2026, 9, 29, 2, tzinfo=timezone.utc)


def snapshot():
    row = dict(ticker='AAA', name='Example company', sector_etf='XLK', setup='TREND',
               status='CANDIDATE', mp_status='NOT_HELD', pitch_status='DUE_DILIGENCE',
               private_account='DO_NOT_EXPORT', quantity=12345,
               news=[dict(id='article-1', title='Example update', url='https://example.com/news',
                          quote='Company reports an update.', published_at='2026-09-28T20:00:00Z',
                          collected_at='2026-09-28T20:05:00Z', assessed_at='2026-09-28T20:06:00Z',
                          available_at='2026-09-28T20:06:00Z', private_field='DO_NOT_EXPORT')])
    row.update({key: 1.0 for key in review.PRICE_FIELDS})
    return dict(as_of='2026-09-28', generated_at='2026-09-29T01:01:00Z',
                news_context={'cutoff': '2026-09-29T01:00:00Z', 'status': 'PARTIAL', 'capped': True},
                mp={'status': 'READY', 'held': ['DO_NOT_EXPORT']}, candidates=[row])


def completed(packet, role):
    result = review.review_template(packet, role)
    result.update(status='COMPLETE', model='synthetic-test')
    for item in result['analyses']:
        item.update(verdict='HOLD', facts=[{'statement': '20일 수익률은 1%.',
                                          'evidence_ids': [item['ticker']+':price']}],
                    unresolved=['실적 미확인'], next_checks=['IR 원문 확인'])
    return result


class ReviewChecks(unittest.TestCase):
    def test_screener_packet_contract(self):
        import test_us_topdown_screener as fixtures
        checks = fixtures.TopdownChecks()
        checks.setUp()
        for ticker in ('AAA', 'BBB'):
            checks.matrices['adj_close'][ticker] *= 1 + .012 * fixtures.np.sin(fixtures.np.arange(225) * 1.7)
        source = checks.result()
        source['warnings'] = []
        stamp = source['as_of'] + 'T23:00:00+00:00'
        fixtures.attach_pitch_evidence(source, {'status': 'READY', 'held': ['BBB']},
                                       {'status': 'SKIPPED', 'cutoff': stamp, 'by_ticker': {}})
        source['generated_at'] = stamp
        packet = review.make_packet(json.loads(json.dumps(source)), now=datetime.fromisoformat(stamp))
        self.assertEqual([r['ticker'] for r in packet['candidates']], ['AAA'])
        self.assertEqual(packet['candidates'][0]['setup'], 'TREND')

    def test_allowlist_time_guards_and_exclusions(self):
        source = snapshot()
        held = dict(source['candidates'][0], ticker='BBB', mp_status='HELD', pitch_status='EXISTING')
        source['candidates'].append(held)
        before = copy.deepcopy(source)
        packet = review.make_packet(source, now=NOW)
        self.assertEqual(source, before)
        self.assertEqual([r['ticker'] for r in packet['candidates']], ['AAA'])
        self.assertNotIn('DO_NOT_EXPORT', review.encoded(packet))
        self.assertNotIn('quantity', review.encoded(packet))
        self.assertEqual(packet['news_coverage'], {'status': 'PARTIAL', 'capped': True})
        cases = []
        for key, value in [('as_of', '2026-09-20'), ('as_of', '2026-09-30'),
                           ('generated_at', '2026-09-30T00:00:00Z'), ('mp', {'status': 'UNKNOWN'}),
                           ('candidates', [])]:
            bad = snapshot()
            bad[key] = value
            cases.append(bad)
        for field, value in [('return20_pct', float('nan')), ('return20_pct', None),
                             ('setup', 'NOT_EVALUATED'), ('sector_etf', 'PRIVATE')]:
            bad = snapshot()
            bad['candidates'][0][field] = value
            cases.append(bad)
        for field, value in [('url', 'https://secret:token@example.com'),
                             ('url', 'javascript:alert(1)'),
                             ('available_at', '2026-09-30T00:00:00Z'),
                             ('assessed_at', '2026-09-29T01:00:01Z'),
                             ('available_at', '2026-09-28T20:00:00Z')]:
            bad = snapshot()
            bad['candidates'][0]['news'][0][field] = value
            cases.append(bad)
        bad = snapshot()
        bad['candidates'][0]['news'] *= 2
        cases.append(bad)
        for bad in cases:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                review.make_packet(bad, now=NOW)

    def test_blind_roles_critic_gates_and_citations(self):
        packet = review.make_packet(snapshot(), now=NOW)
        replies = {role: completed(packet, role) for role in ('analyst_a', 'analyst_b')}
        for role in replies:
            self.assertIn(review.packet_hash(packet), review.prompt(packet, role))
            with self.assertRaises(ValueError):
                review.prompt(packet, role, replies)
        self.assertIn('비신뢰 독립 검토 결과', review.prompt(packet, 'critic', replies))
        with self.assertRaises(ValueError):
            review.prompt(packet, 'critic')
        bad = review.review_template(packet, 'analyst_a')
        with self.assertRaises(ValueError):
            review.validate_review(packet, bad, 'analyst_a')
        mutations = [lambda r: r.update(packet_sha256='wrong'),
                     lambda r: r.update(analyses=None),
                     lambda r: r.update(analyses=[None]),
                     lambda r: r.update(private_context={'ss_holdings': ['DO_NOT_EXPORT']}),
                     lambda r: r['analyses'][0].update(unresolved=[]),
                     lambda r: r['analyses'][0].update(hypotheses=[{'secret': 'DO_NOT_EXPORT'}]),
                     lambda r: r['analyses'][0].update(next_checks=['']),
                     lambda r: r['analyses'][0].update(facts=['unsupported']),
                     lambda r: r['analyses'][0]['facts'][0].update(private='DO_NOT_EXPORT'),
                     lambda r: r['analyses'][0]['facts'][0].update(evidence_ids=['BBB:price']),
                     lambda r: r['analyses'][0]['facts'][0].update(evidence_ids=[{}])]
        for mutate in mutations:
            bad = copy.deepcopy(replies['analyst_a'])
            mutate(bad)
            with self.assertRaises(ValueError):
                review.validate_review(packet, bad, 'analyst_a')

    def test_bundle_hashes_and_no_overwrite(self):
        packet = review.make_packet(snapshot(), now=NOW)
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)/'bundle'
            manifest = review.write_bundle(packet, target)
            self.assertEqual(manifest['status'], 'PREPARED_NOT_RUN')
            self.assertEqual(json.loads((target/'packet.json').read_text(encoding='utf-8')), packet)
            for name, digest in manifest['files'].items():
                self.assertEqual(hashlib.sha256((target/name).read_bytes()).hexdigest(), digest)
            self.assertFalse((target/'critic.md').exists())
            before = (target/'manifest.json').read_bytes()
            with self.assertRaises(FileExistsError):
                review.write_bundle(packet, target)
            self.assertEqual(before, (target/'manifest.json').read_bytes())


if __name__ == '__main__':
    unittest.main()
