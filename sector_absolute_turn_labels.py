"""Research-only absolute-price path labels for the 11 sector ETFs. No model fit."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from sector_rotation import SECTOR_ETFS

ROOT = Path(__file__).resolve().parent
HORIZON = 20


def path_outcome(path: np.ndarray, direction: int, barrier: float) -> str:
    """Future log moves from decision close, signed by the existing trend.

    Opposite barrier must precede continuation barrier and persist into the tail.
    This is a prospective-window target, NOT an oracle for exact market peaks.
    """
    values = np.asarray(path, dtype=float)
    if direction not in (-1, 1) or not np.isfinite(barrier) or barrier <= 0:
        raise ValueError('invalid direction/barrier')
    if len(values) != HORIZON or not np.isfinite(values).all():
        return 'IMMATURE'
    signed = values * direction
    reversal = np.flatnonzero(signed <= -barrier)
    continuation = np.flatnonzero(signed >= barrier)
    first_reverse = int(reversal[0]) if len(reversal) else HORIZON
    first_continue = int(continuation[0]) if len(continuation) else HORIZON
    persistent = signed[-1] <= -barrier / 2 and (signed[-5:] <= -barrier / 2).sum() >= 3
    if first_reverse < first_continue and persistent:
        return 'TOP_END' if direction == 1 else 'BOTTOM_END'
    if first_continue < first_reverse and signed[-1] >= barrier / 2:
        return 'UP_CONTINUES' if direction == 1 else 'DOWN_CONTINUES'
    return 'UNCERTAIN'


def label_frame(price: pd.Series) -> pd.DataFrame:
    if not isinstance(price.index, pd.DatetimeIndex) or price.index.has_duplicates or not price.index.is_monotonic_increasing:
        raise ValueError('unique ordered daily dates required')
    price = pd.to_numeric(price, errors='raise').astype(float)
    if (price.dropna() <= 0).any() or np.isinf(price.to_numpy()).any():
        raise ValueError('prices must be positive finite or missing')
    log = np.log(price)
    returns = log.diff()
    ret20, ret60 = log.diff(20), log.diff(60)
    state = pd.Series(np.select([(ret20>0)&(ret60>0),(ret20<0)&(ret60<0)], [1,-1], default=0),index=price.index)
    # Freeze the threshold at t; historical volatility excludes the current session.
    barrier = (returns.shift(1).rolling(60,min_periods=60).std()*np.sqrt(HORIZON)).clip(.03,.15)
    rows=[]
    for i,date in enumerate(price.index):
        history = price.iloc[max(0,i-61):i+1]
        if len(history)<62 or history.isna().any() or not np.isfinite(barrier.iloc[i]):
            continue
        direction=int(state.iloc[i])
        future=log.iloc[i+1:i+1+HORIZON]-log.iloc[i]
        target = path_outcome(future.to_numpy(),direction,float(barrier.iloc[i])) if direction else 'NO_CLEAR_TREND'
        mature=len(future)==HORIZON and future.notna().all()
        if not mature:
            target='IMMATURE'
        rows.append(dict(date=date,price=price.iloc[i],trend=direction,barrier_log=float(barrier.iloc[i]),
                         return20_log=ret20.iloc[i],return60_log=ret60.iloc[i],
                         target=target,label_end=price.index[i+HORIZON] if mature else pd.NaT,
                         future_return20=np.expm1(future.iloc[-1]) if mature else np.nan))
    return pd.DataFrame(rows)


def run(database: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True) as c:
        placeholders=','.join('?' for _ in SECTOR_ETFS)
        raw=pd.read_sql_query(f'SELECT date,ticker,adj_close FROM daily_prices WHERE ticker IN ({placeholders})',c,
                              params=list(SECTOR_ETFS),parse_dates=['date'])
    if raw.duplicated(['ticker','date']).any() or set(raw.ticker)!=set(SECTOR_ETFS):
        raise ValueError('invalid 11-ETF coverage or duplicate keys')
    # Keep internal gaps on the shared observed trading calendar; never forward-fill.
    matrix=raw.pivot(index='date',columns='ticker',values='adj_close').sort_index()
    parts=[]
    for ticker in SECTOR_ETFS:
        part=label_frame(matrix[ticker])
        part.insert(0,'ticker',ticker)
        parts.append(part)
    labels=pd.concat(parts,ignore_index=True)
    output.mkdir(parents=True,exist_ok=False)
    labels.to_csv(output/'labels.csv',index=False)
    counts=labels.groupby(['ticker','target']).size().unstack(fill_value=0)
    counts.to_csv(output/'label_counts.csv')
    (output/'manifest.json').write_text(json.dumps({
        'purpose':'LABEL_AUDIT_ONLY_NOT_BACKTEST_OR_VALIDATED_MODEL',
        'universe':list(SECTOR_ETFS),'rows':len(labels),'horizon_sessions':HORIZON,
        'threshold':'prior 60-session log-return std * sqrt(20), clipped to 0.03..0.15 log move, fixed at decision t',
        'trend':'20 and 60-session absolute log returns agree; otherwise no clear trend',
        'target':'first opposing barrier with final-day and 3/last5 followthrough; continuation-first separated',
        'known_limits':['daily closes only','current revised adjusted prices, PIT not verified',
                        'overlapping labels are not independent events','not precise pre-pivot timing',
                        'sideways starts and trends with disagreeing 20/60 returns excluded',
                        'thresholds unvalidated; compare sensitivity before model selection'],
        'training_rule':'label_end < refit_date; targets and future fields NEVER predictors',
        'writes':'research output only; no DB or website mutation'
    },ensure_ascii=False,indent=2),encoding='utf-8')
    print(counts.to_string())


def self_test():
    rising=np.linspace(.002,.08,HORIZON)
    assert path_outcome(rising,1,.03)=='UP_CONTINUES'
    assert path_outcome(-rising,-1,.03)=='DOWN_CONTINUES'
    assert path_outcome(-rising,1,.03)=='TOP_END'
    assert path_outcome(rising,-1,.03)=='BOTTOM_END'
    dip=np.r_[np.linspace(-.01,-.05,7),np.linspace(-.04,.08,13)]
    assert path_outcome(dip,1,.03)=='UNCERTAIN'  # temporary correction is not TOP_END
    assert path_outcome(rising[:5],1,.03)=='IMMATURE'
    dates=pd.bdate_range('2020-01-01',periods=100)
    price=pd.Series(np.exp(np.arange(100)*.001)*100,index=dates)
    before=label_frame(price)
    price.iloc[-1]*=.5
    after=label_frame(price)
    cols=['trend','barrier_log','return20_log','return60_log']
    assert before.iloc[:-1][cols].equals(after.iloc[:-1][cols])
    stable=before.label_end.lt(dates[-1])
    assert before.loc[stable,'target'].equals(after.loc[stable,'target'])
    assert before.tail(HORIZON).target.eq('IMMATURE').all()
    price.iloc[70]=np.nan
    assert not label_frame(price).date.eq(dates[71]).any()
    print('self-test PASS: trend continuation, reversal, recovered dip, maturity, causal inputs, missing data')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database',type=Path,default=ROOT/'regime_router_publish.db')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/absolute_turn_labels_20260921')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test: self_test()
    else: run(args.database,args.output_dir)
