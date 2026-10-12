"""N: reproduce old pooled price Ridge, audit implementable long-only variant."""
import json
import argparse
import numpy as np
import pandas as pd
from research_leading_improvements import ROOT, OUT
from research_leading_followup import portfolio_blocks
from sector_flow_experiment import load_flow_history, build_flow_features, build_common_panel, _fit_ridge, BASE_FEATURE_COLUMNS
from sector_shadow_store import _load_price_matrix_read_only
from validate_sector_factor_explanation import SECTORS
from validate_nasdaq_ratio import hac, holm
from sector_rotation import build_rrg_signals
from sector_rotation_forecast import build_sector_feature_frame


def run(long_history=False, rolling=False):
    long_history = long_history or rolling
    prefix = 'P' if rolling else ('O' if long_history else 'N')
    models = ['EXPANDING','ROLLING104'] if rolling else ['B1_RRG','B2_PRICE_RIDGE']
    old = pd.read_csv(ROOT/'outputs/sector_flow_3y_experiment_20260902/predictions.csv',parse_dates=['date'])
    prices = _load_price_matrix_read_only(ROOT/'regime_router_publish.db').loc[:'2026-09-08']
    cutoff = pd.Timestamp('2025-01-31')
    full = build_sector_feature_frame(prices,build_rrg_signals(prices),horizon=20)
    short_prices = prices.loc[:cutoff]
    short = build_sector_feature_frame(short_prices,build_rrg_signals(short_prices),horizon=20)
    cols = ['date','ticker',*BASE_FEATURE_COLUMNS]
    pd.testing.assert_frame_equal(full.loc[full.date<=cutoff,cols].reset_index(drop=True),short[cols].reset_index(drop=True))
    if long_history:
        endpoints = pd.Series(prices.index,index=prices.index).groupby(prices.index.to_period('W-FRI')).max()
        panel = full[full.date.isin(endpoints)].dropna(subset=[*BASE_FEATURE_COLUMNS,'target_excess_20d','label_end','shadow_score']).copy()
        complete = panel.groupby('date').ticker.nunique().loc[lambda s:s==11].index
        panel = panel[panel.date.isin(complete)].copy()
        panel['sector'] = panel.ticker.map(SECTORS)
        panel['flow_source_date'] = pd.NaT  # Existing fit helper's unused provenance column.
        dates = [d for d in sorted(panel.date.unique()) if panel.loc[panel.label_end<d,'date'].nunique()>=(104 if rolling else 26)]
    else:
        flow = load_flow_history(ROOT/'outputs/bloomberg_fund_flow_3y_staging_20260902/history.csv')
        panel,_ = build_common_panel(prices,build_flow_features(flow,prices.index))
        saved = old[old.model.eq('B2_PRICE_RIDGE')]
        dates = sorted(saved.date.unique())
    rows = []
    for day in dates:
        r,n = _fit_ridge(panel,day,BASE_FEATURE_COLUMNS,alpha=10.)
        assert (panel.loc[panel.label_end < day,'label_end'] < day).all()
        rows.append(r.assign(training_weeks=n,model='EXPANDING' if rolling else 'B2_PRICE_RIDGE'))
        if rolling:
            weeks = sorted(panel.loc[panel.label_end<day,'date'].unique())[-104:]
            subset = panel[panel.date.isin(weeks) | panel.date.eq(day)]
            rr,nn = _fit_ridge(subset,day,BASE_FEATURE_COLUMNS,alpha=10.)
            assert nn==104 and len(weeks)==104
            assert subset.loc[subset.date.isin(weeks),'label_end'].max()<day
            rows.append(rr.assign(training_weeks=nn,model='ROLLING104'))
    replay = pd.concat(rows)
    if long_history:
        baseline = panel[panel.date.isin(dates)][['date','ticker','target_excess_20d','shadow_score']].rename(columns={'shadow_score':'prediction'})
        old = replay if rolling else pd.concat([baseline.assign(model='B1_RRG'),replay])
        old.to_csv(OUT/f'{prefix}_predictions.csv',index=False)
        meta = {'oos_weeks':len(dates),'oos_start':str(min(dates)),'oos_end':str(max(dates)),
                'sample_note':'Full price-available11-sector weekly sample; retrospective extension, not independent confirmation.'}
    else:
        matched = replay.merge(saved,on=['date','ticker'],suffixes=('_replay','_saved'),validate='one_to_one')
        difference = (matched.prediction_replay-matched.prediction_saved).abs()
        assert len(matched)==len(saved)
        matched.to_csv(OUT/'N_replay_comparison.csv',index=False)
        meta = {'replayed_rows':len(matched),'replayed_weeks':saved.date.nunique(),
                'max_prediction_difference':difference.max(),'mismatches_above_1e_9':int((difference>1e-9).sum()),
                'training_week_mismatches':int((matched.training_weeks_replay!=matched.training_weeks_saved).sum()),
                'sample_note':'Original flow-aligned2024+ sample; not independent confirmation.'}
    ranks = old[old.model.isin(models)].groupby(['date','model']).apply(lambda s:s.prediction.corr(s.target_excess_20d,method='spearman'),include_groups=False).unstack()
    ranks.to_csv(OUT/f'{prefix}_rank_ic.csv')
    ranks.groupby(ranks.index.year).mean().to_csv(OUT/f'{prefix}_annual_rank_ic.csv')
    meta['mean_rank_ic'] = ranks.mean().to_dict()
    if rolling:
        errors = old.assign(loss=(old.prediction-old.target_excess_20d)**2).groupby(['date','model']).loss.mean().unstack()
        values = {'rank_ic':ranks.ROLLING104-ranks.EXPANDING,'mse':errors.EXPANDING-errors.ROLLING104}
        tests = []
        for name,v in values.items():
            b,se,p = hac(v.to_numpy(),np.empty((len(v),0)),4)
            tests.append({'metric':name,'improvement':b[0],'ci_low':b[0]-1.96*se[0],'ci_high':b[0]+1.96*se[0],'p_hac':p[0]})
        tests = pd.DataFrame(tests); tests['p_holm_2'] = holm(tests.p_hac)
        tests.to_csv(OUT/'P_predictive_tests.csv',index=False)
        meta['mean_mse'] = errors.mean().to_dict()
    prices = prices.dropna()
    blocks = []
    for model in models:
        weekly = old[old.model.eq(model)].pivot(index='date',columns='ticker',values='prediction')
        daily = weekly.reindex(prices.index).ffill(limit=4)
        origin = pd.Series(weekly.index,index=weekly.index).reindex(prices.index).ffill(limit=4)
        daily = daily.dropna()
        for offset in range(20):
            b = portfolio_blocks(prices,daily,model,1,offset)
            b['issued_date'] = b.signal_date.map(origin)
            assert (b.issued_date < b.date).all()
            b['net_vs_equal11'] = [x.gross_return-x.cost-(prices.loc[x.exit,list(SECTORS)]/prices.loc[x.date,list(SECTORS)]-1).mean() for x in b.itertuples()]
            blocks.append(b.assign(offset=offset))
    blocks = pd.concat(blocks)
    assert ((blocks.gross_excess-blocks.cost-blocks.net_excess).abs()<1e-12).all()
    for _,s in blocks.groupby(['model','offset']):
        assert (s.date.iloc[1:].to_numpy() >= s.exit.iloc[:-1].to_numpy()).all()
    for _,s in blocks.groupby('offset'):
        assert s.groupby('model').date.apply(tuple).nunique()==1
    blocks.to_csv(OUT/f'{prefix}_blocks.csv',index=False)
    offsets = blocks.groupby(['model','offset']).agg(n=('net_excess','size'),net_excess=('net_excess','mean'),net_vs_equal11=('net_vs_equal11','mean'))
    offsets.to_csv(OUT/f'{prefix}_offsets.csv')
    summary = offsets.groupby('model')[['net_excess','net_vs_equal11']].agg(['min','median','max'])
    summary.to_csv(OUT/f'{prefix}_offset_summary.csv')
    tests = []
    for offset,s in blocks.groupby('offset'):
        pivot = s.pivot(index='date',columns='model',values='net_excess')
        for name,values in {'candidate_absolute':pivot[models[1]],
                            'candidate_minus_baseline':pivot[models[1]]-pivot[models[0]]}.items():
            b,se,p = hac(values.to_numpy(),np.empty((len(values),0)),1)
            tests.append({'offset':offset,'comparison':name,'n':len(values),'mean':b[0],
                          'ci_low':b[0]-1.96*se[0],'ci_high':b[0]+1.96*se[0],'p_hac':p[0]})
    tests = pd.DataFrame(tests); tests['p_holm_40'] = holm(tests.p_hac)
    tests.to_csv(OUT/f'{prefix}_economic_uncertainty.csv',index=False)
    meta['positive_offsets'] = offsets.groupby('model').net_excess.apply(lambda s:int((s>0).sum())).to_dict()
    (OUT/f'{prefix}_replay_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps(meta,indent=2)); print(summary.to_string())


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--long-history',action='store_true')
    parser.add_argument('--rolling',action='store_true')
    args = parser.parse_args()
    run(args.long_history,args.rolling)
