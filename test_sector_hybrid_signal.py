"""Fast local fixtures only: no real fits, market downloads or production writes."""
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from contextlib import closing

import joblib
import numpy as np
import pandas as pd
import sklearn
import sector_hybrid_signal as h


class IdentityScaler:
    def transform(self,x):return np.asarray(x)


class FixedModel:
    def __init__(self,score):self.score=score
    def predict_proba(self,x):return np.tile([1-self.score,self.score],(len(x),1))


def packs():
    return {'IN':{'model':FixedModel(.9),'scaler':IdentityScaler(),'threshold':.8},
            'OUT':{'model':FixedModel(.7),'scaler':IdentityScaler(),'threshold':.8}}


class HybridTests(unittest.TestCase):
    def test_week_selection(self):
        dates=pd.to_datetime(['2026-09-04','2026-09-10','2026-09-14'])
        self.assertEqual(h.completed_week(dates,pd.Timestamp('2026-09-10')),dates[0])
        self.assertEqual(h.completed_week(dates,pd.Timestamp('2026-09-14')),dates[1])
        self.assertIsNone(h.completed_week(dates,pd.Timestamp('2026-09-01')))

    def test_direction_and_missing(self):
        f=pd.DataFrame(0.,index=range(3),columns=h.PRICE_COLUMNS)
        tickers=list(h.SECTOR_ETFS);f['ticker']=tickers[:3];f['ret60']=[-.1,.1,-.1]
        f.loc[2,'ret5']=np.nan
        rows=h.score_rows(f,packs())
        self.assertEqual(rows[0]['signal'],'IN')
        self.assertEqual(rows[1]['model'],'PRICE_EXTRA');self.assertEqual(rows[1]['signal'],'NONE')
        self.assertEqual(rows[2]['status'],'MISSING_FEATURES');self.assertIsNone(rows[2]['signal'])
        self.assertEqual(len(rows),11)
        p=packs();p['OUT']['threshold']=.6
        self.assertEqual(h.score_rows(f,p)[1]['signal'],'OUT')

    def test_persistence_and_integrity(self):
        with tempfile.TemporaryDirectory() as d:
            directory=Path(d);models={}
            for direction,pack in packs().items():
                path=directory/(direction+'.joblib');joblib.dump(pack,path)
                models[direction]={'family':h.ROUTES[direction],'threshold':pack['threshold'],'sha256':h.digest(path),
                    'fit_label_end':'2023-12-29','validation_label_end':'2025-12-31'}
            manifest={'schema':1,'year':2026,'columns':h.COLUMNS,'sklearn':sklearn.__version__,
                'feature_signature':h.feature_signature(),'models':models}
            h.write_json(directory/'manifest.json',manifest)
            _,loaded=h.load_bundle(directory)
            self.assertEqual(loaded['IN']['model'].score,.9)
            models['IN']['sha256']='invalid';h.write_json(directory/'manifest.json',manifest)
            with self.assertRaisesRegex(ValueError,'integrity'):h.load_bundle(directory)
            self.assertFalse(list(directory.glob('*.tmp')))

    def test_snapshot_asof_and_missing_bundle(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);db=root/'fixture.db';bundle=root/'bundle';bundle.mkdir()
            self.assertEqual(h.build_snapshot(db,bundle,'2026-09-18')['status'],'MODEL_NOT_READY')
            (bundle/'manifest.json').touch()
            dates=pd.bdate_range(end='2026-09-18',periods=210)
            rng=np.random.default_rng(9);records=[]
            for ticker in ['SPY',*h.SECTOR_ETFS]:
                values=100*np.exp(np.cumsum(rng.normal(.0004,.01,len(dates))))
                records.extend((str(date.date()),ticker,float(value)) for date,value in zip(dates,values))
            with closing(sqlite3.connect(db)) as con, con:
                con.execute('CREATE TABLE daily_prices(date TEXT,ticker TEXT,adj_close REAL)')
                con.executemany('INSERT INTO daily_prices VALUES(?,?,?)',records)
            with patch.object(h,'load_bundle',return_value=({'year':2026},packs())):
                with patch.object(h,'features',wraps=h.features) as observed_features:
                    before=h.build_snapshot(db,bundle,'2026-09-11')
                    passed_prices=observed_features.call_args.args[0]
                    self.assertEqual(passed_prices.index.max(),pd.Timestamp('2026-09-11'))
                self.assertEqual(before['status'],'READY');self.assertEqual(len(before['rows']),11)
                with closing(sqlite3.connect(db)) as con, con:
                    con.execute("UPDATE daily_prices SET adj_close=adj_close*3 WHERE date>'2026-09-11'")
                after=h.build_snapshot(db,bundle,'2026-09-11')
                self.assertEqual(before['rows'],after['rows'])
                self.assertEqual(h.build_snapshot(db,bundle,'2026-10-10')['status'],'STALE_PRICES')
                with closing(sqlite3.connect(db)) as con, con:
                    con.execute("DELETE FROM daily_prices WHERE ticker='XLK' AND date='2026-09-18'")
                partial=h.build_snapshot(db,bundle,'2026-09-18')
                self.assertEqual(partial['status'],'PARTIAL')
                self.assertIsNone(next(r for r in partial['rows'] if r['ticker']=='XLK')['signal'])
            with patch.object(h,'load_bundle',return_value=({'year':2025},packs())):
                self.assertEqual(h.build_snapshot(db,bundle,'2026-09-18')['status'],'MODEL_YEAR_MISMATCH')


if __name__=='__main__':unittest.main()
