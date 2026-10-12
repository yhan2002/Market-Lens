"""Audit all rebalance offsets without selecting a winner."""
import json
import sqlite3
import numpy as np
import pandas as pd
from research_leading_improvements import ROOT, OUT
from research_leading_followup import portfolio_blocks
from validate_sector_factor_explanation import SECTORS
from validate_nasdaq_ratio import hac


def run():
    with sqlite3.connect((ROOT/'regime_router_publish.db').resolve().as_uri()+'?mode=ro',uri=True) as c:
        ticks=[*SECTORS,'SPY']
        r=pd.read_sql_query("SELECT date,ticker,adj_close FROM daily_prices WHERE ticker IN ("+','.join('?' for _ in ticks)+") AND date<='2026-09-08' ORDER BY date",c,params=ticks,parse_dates=['date'])
    prices=r.pivot(index='date',columns='ticker',values='adj_close').dropna()
    c=pd.read_csv(OUT/'C_predictions.csv',parse_dates=['date'])
    f=pd.read_csv(OUT/'F_predictions.csv',parse_dates=['date'])
    signals={'C_'+k:c.pivot(index='date',columns='ticker',values=k) for k in ['momentum','macro']}
    signals.update({'F_'+k:f.pivot(index='date',columns='ticker',values=k) for k in ['individual_base','individual_options','pooled_base','pooled_options']})
    relative=np.log(prices[list(SECTORS)]).sub(np.log(prices.SPY),axis=0)
    signals['raw60']=relative.diff(60).reindex(signals['C_momentum'].index)
    rows=[]
    for name,signal in signals.items():
        for offset in range(20):
            b=portfolio_blocks(prices,signal,name,1,offset)
            assert (b.signal_date<b.date).all() and (b.exit>b.date).all()
            assert np.allclose(b.net_excess,b.gross_excess-b.cost)
            rows.append({'model':name,'offset':offset,'n':len(b),'net_excess':b.net_excess.mean()})
    offsets=pd.DataFrame(rows)
    offsets.to_csv(OUT/'G_rebalance_offsets.csv',index=False)
    summary=offsets.groupby('model').net_excess.agg(['min','median','max'])
    summary['positive_fraction']=offsets.groupby('model').net_excess.apply(lambda v:float((v>0).mean()))
    summary.to_csv(OUT/'G_offset_summary.csv')
    ic=pd.read_csv(OUT/'F_rank_ic.csv',index_col=0)
    diff=ic.pooled_options-ic.pooled_base
    b,se,p=hac(diff.to_numpy(),np.empty((len(diff),0)),20)
    result={'n':len(diff),'paired_rank_ic_improvement':float(b[0]),'ci_low':float(b[0]-1.96*se[0]),'ci_high':float(b[0]+1.96*se[0]),'p_hac_exploratory':float(p[0])}
    (OUT/'G_rank_test.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(summary.to_string());print(result)


if __name__=='__main__':run()
