"""Read-only audit of existing flow semantics, saved labels and paired rank IC."""
import hashlib
import json
import sqlite3
import numpy as np
import pandas as pd
from research_leading_improvements import ROOT, OUT
from validate_sector_factor_explanation import SECTORS
from validate_nasdaq_ratio import hac


def run():
    source = ROOT/'outputs/bloomberg_fund_flow_3y_staging_20260902/history.csv'
    predictions = ROOT/'outputs/sector_flow_3y_experiment_20260902/predictions.csv'
    flow = pd.read_csv(source,parse_dates=['date'])
    pred = pd.read_csv(predictions,parse_dates=['date','flow_source_date'])
    with sqlite3.connect((ROOT/'regime_router_publish.db').resolve().as_uri()+'?mode=ro',uri=True) as c:
        raw = pd.read_sql_query('SELECT date,ticker,adj_close FROM daily_prices WHERE ticker IN ('+
                               ','.join('?' for _ in SECTORS)+") AND date<='2026-09-08' ORDER BY date",
                               c,params=list(SECTORS),parse_dates=['date'])
    prices = raw.pivot(index='date',columns='ticker',values='adj_close').dropna()
    returns = prices.shift(-20)/prices-1
    excess = returns.sub(returns.mean(axis=1),axis=0)
    computed = np.array([excess.loc[x.date,x.ticker] for x in pred.itertuples()])
    difference = np.abs(computed-pred.target_excess_20d)
    assert np.isfinite(computed).all()
    assert not pred.duplicated(['date','ticker','model']).any()
    coverage = pred.groupby(['date','model']).ticker.nunique()
    assert coverage.eq(11).all()
    assert (pred.flow_source_date < pred.date).all()
    ranks = pred.groupby(['date','model']).apply(
        lambda s:s.prediction.corr(s.target_excess_20d,method='spearman'),include_groups=False).unstack()
    delta = ranks.C2_PRICE_FLOW_RIDGE-ranks.B2_PRICE_RIDGE
    b,se,p = hac(delta.to_numpy(),np.empty((len(delta),0)),4)
    result = {
        'audit':'M_source_and_saved_predictions_only; no refitting',
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'predictions_sha256':hashlib.sha256(predictions.read_bytes()).hexdigest(),
        'source_rows':len(flow),'source_weeks':flow.date.nunique(),
        'source_start':str(flow.date.min()),'source_end':str(flow.date.max()),
        'field_nonmissing':flow.notna().sum().to_dict(),
        'weekly_grid_complete':bool(flow.date.drop_duplicates().sort_values().diff().dropna().eq(pd.Timedelta(days=7)).all()),
        'definition_status':'UNVERIFIED: FUND_FLOW cannot be equated to FUND_FLOW_1_WK from this file',
        'publication_status':'UNVERIFIED: source dates and a one-week delay do not prove publication vintage',
        'saved_oos_weeks':len(ranks),'mean_ic':ranks.mean().to_dict(),
        'paired_flow_increment_ic':float(b[0]),'ci_low':float(b[0]-1.96*se[0]),
        'ci_high':float(b[0]+1.96*se[0]),'p_hac_4weeks':float(p[0]),
        'current_price_label_max_abs_difference':float(difference.max()),
        'label_mismatch_rows_over_1e_8':int((difference>1e-8).sum()),
        'source_age_days_min':int((pred.date-pred.flow_source_date).dt.days.min()),
        'source_age_days_max':int((pred.date-pred.flow_source_date).dt.days.max()),
        'limitations':'Saved output audit does not establish training lineage or tradable after-cost performance. Current prices can differ from original vintages.'}
    (OUT/'M_flow_source_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    ranks.to_csv(OUT/'M_saved_flow_rank_ic.csv')
    print(json.dumps(result,indent=2))


if __name__=='__main__':run()
