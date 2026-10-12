"""Fixed exploratory MSCI ETF-proxy increment; download only with --download."""
import argparse
import hashlib
import json
import sqlite3
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from research_leading_improvements import ROOT, OUT, panel, loss_test
from research_leading_followup import portfolio_blocks
from validate_sector_factor_explanation import SECTORS
from validate_nasdaq_ratio import hac, holm

TICKERS = ['MTUM', 'QUAL', 'VLUE', 'USMV']
BASE = ['rel20', 'rel60', 'vol20']
MACRO = ['move5', 'vix5', 'curve']
CACHE = OUT / 'J_msci_proxy_raw.pkl'


def features(prices):
    relative = np.log(prices[TICKERS]).sub(np.log(prices.SPY), axis=0)
    return pd.concat([relative.diff(h).add_suffix(f'_rel{h}') for h in [20, 60]], axis=1)


def residual_features(prices):
    returns = np.log(prices).diff()
    controls = ['SPY', *SECTORS]
    residual = pd.DataFrame(np.nan, index=prices.index, columns=TICKERS)
    for i in range(253, len(prices)):
        train = returns.iloc[i-252:i]
        today = returns.iloc[[i]]
        if train[controls+TICKERS].isna().any().any() or today[controls+TICKERS].isna().any().any():
            continue
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.))
        model.fit(train[controls], train[TICKERS])
        residual.iloc[i] = today[TICKERS].to_numpy()[0]-model.predict(today[controls])[0]
    return pd.concat([residual.rolling(h).sum().add_suffix(f'_resid{h}') for h in [20,60]], axis=1)


def matched_features(prices):
    # One availability mask governs both training and evaluation for every model.
    return features(prices).join(residual_features(prices)).dropna()


def run(download=False, residual=False, classification=False, matched=False, tune=False):
    if tune and (download or residual or classification or matched):
        raise ValueError('--tune must be used alone; it is cache-only')
    if tune:
        from msci_tuning import select, predict_explain, ALPHAS, WEIGHTS, IC_TOLERANCE
        matched = True
    if matched and (download or residual or classification):
        raise ValueError('Matched comparison is a standalone cache-only regression experiment')
    if residual and classification:
        raise ValueError('Residual classification is not part of the fixed experiment')
    prefix = 'X' if tune else ('W' if matched else ('L' if classification else ('K' if residual else 'J')))
    if download:
        import yfinance as yf
        yf.set_tz_cache_location(str(OUT/'yf_cache'))
        raw = yf.download(TICKERS, start='2013-01-01', end='2026-09-09',
                          auto_adjust=False, progress=False, threads=False, timeout=30)
        if raw.empty or not set(TICKERS).issubset(raw['Adj Close'].columns):
            raise ValueError('Incomplete download; cache not written')
        assert raw['Adj Close'].notna().sum().min() > 2500
        raw.to_pickle(CACHE)
    raw = pd.read_pickle(CACHE)
    proxy = raw['Adj Close'][TICKERS]
    assert proxy.index.is_unique and proxy.index.is_monotonic_increasing
    assert (proxy.dropna() > 0).all().all()
    with sqlite3.connect((ROOT/'regime_router_publish.db').resolve().as_uri()+'?mode=ro', uri=True) as c:
        ticks = [*SECTORS, 'SPY']
        r = pd.read_sql_query('SELECT date,ticker,adj_close FROM daily_prices WHERE ticker IN ('+
                             ','.join('?' for _ in ticks)+") AND date<='2026-09-08' ORDER BY date",
                             c, params=ticks, parse_dates=['date'])
    prices = r.pivot(index='date', columns='ticker', values='adj_close').dropna()
    # Keep the full common sector calendar: missing proxy quotes never bridge horizons.
    joined = prices.join(proxy)
    f = matched_features(joined) if matched else (residual_features(joined) if residual else features(joined))
    _, _, macro = panel()
    factorcols = list(f.columns)
    f = f.join(macro[MACRO])
    q = np.log(prices)
    end = pd.Series(q.index, index=q.index).shift(-20)
    rows = []
    for ticker in SECTORS:
        rel = q[ticker]-q.SPY
        frame = f.join(pd.DataFrame({'rel20':rel.diff(20), 'rel60':rel.diff(60),
                                    'vol20':q[ticker].diff().rolling(20).std(),
                                    'y':rel.shift(-20)-rel, 'end':end})).dropna()
        rows.append(frame.assign(ticker=ticker))
    frame = pd.concat(rows).reset_index(names='date')
    shared = frame.groupby('date').ticker.nunique().loc[lambda s:s == 11].index
    frame = frame[frame.date.isin(shared)]
    models = {'base':BASE, 'macro':BASE+MACRO, 'factor':BASE+factorcols,
              'macro_factor':BASE+MACRO+factorcols}
    comparisons = [('base', 'factor'), ('macro', 'macro_factor')]
    if matched:
        rawcols = [c for c in factorcols if '_rel' in c]
        residcols = [c for c in factorcols if '_resid' in c]
        models = {'base':BASE, 'macro':BASE+MACRO,
                  'raw':BASE+rawcols, 'residual':BASE+residcols,
                  'macro_raw':BASE+MACRO+rawcols, 'macro_residual':BASE+MACRO+residcols}
        comparisons = [('base','raw'), ('base','residual'), ('raw','residual'),
                       ('macro','macro_raw'), ('macro','macro_residual'), ('macro_raw','macro_residual')]
    if tune:
        tune_cols = [c for c in factorcols if c.endswith('_resid60')]
        assert len(tune_cols) == 4
        models = {'base':BASE, 'residual':BASE+residcols, 'residual60':BASE+tune_cols,
                  'tuned':BASE+tune_cols}
        comparisons = [('base','tuned'), ('residual','tuned'), ('residual60','tuned')]
        selections, grid, inner_audit, coefficients, contributions = [], [], [], [], []
    saved = []
    splits = []
    for year in range(2021, 2027):
        train = frame[frame.end < pd.Timestamp(year, 1, 1)]
        test = frame[frame.date.dt.year.eq(year)].copy()
        if test.empty or train.groupby('ticker').size().min() < 504:
            continue
        assert train.end.max() < test.date.min()
        if tune:
            selection, candidates, folds = select(train, year, BASE, tune_cols)
            selections.append(dict(selection, outer_year=year))
            grid.extend(candidates)
            inner_audit.extend(folds)
            print(f"{year}: alpha={selection['alpha']}, weight={selection['weight']}, {selection['status']}", flush=True)
        for ticker, sub in train.groupby('ticker'):
            mask = test.ticker.eq(ticker)
            splits.append({'year':year, 'ticker':ticker, 'train_n':len(sub),
                           'train_start':str(sub.date.min()), 'train_end':str(sub.date.max()),
                           'last_label_end':str(sub.end.max()), 'test_n':int(mask.sum()),
                           'train_dates_sha256':hashlib.sha256(sub.date.to_numpy().tobytes()).hexdigest(),
                           'test_dates_sha256':hashlib.sha256(test.loc[mask,'date'].to_numpy().tobytes()).hexdigest()})
            target = (sub.y > 0).astype(int) if classification else sub.y
            if classification:
                assert target.nunique() == 2
                test.loc[mask,'prior'] = target.mean()
            for name, cols in models.items():
                if tune and name == 'tuned':
                    prediction, coef, attribution = predict_explain(
                        sub, test.loc[mask], BASE, tune_cols, selection['alpha'], selection['weight'])
                    test.loc[mask,name] = prediction
                    coefficients.append(coef.assign(year=year,ticker=ticker,
                                                    alpha=selection['alpha'],weight=selection['weight']))
                    contributions.append(attribution.assign(date=test.loc[mask,'date'],ticker=ticker))
                    continue
                estimator = LogisticRegression(C=.1,max_iter=2000) if classification else Ridge(alpha=1.)
                model = make_pipeline(StandardScaler(), estimator)
                model.fit(sub[cols], target)
                test.loc[mask, name] = (model.predict_proba(test.loc[mask, cols])[:,1] if classification
                                       else model.predict(test.loc[mask, cols]))
        saved.append(test)
    pred = pd.concat(saved).sort_values(['date', 'ticker'])
    assert np.isfinite(pred[list(models)].to_numpy()).all()
    if matched:
        pd.DataFrame(splits).to_csv(OUT/f'{prefix}_split_audit.csv', index=False)
    if tune:
        pd.DataFrame(selections).to_csv(OUT/f'{prefix}_selected_settings.csv',index=False)
        pd.DataFrame(grid).to_csv(OUT/f'{prefix}_inner_grid.csv',index=False)
        pd.DataFrame(inner_audit).to_csv(OUT/f'{prefix}_inner_split_audit.csv',index=False)
        pd.concat(coefficients).to_csv(OUT/f'{prefix}_coefficients.csv',index=False)
        pd.concat(contributions).to_csv(OUT/f'{prefix}_contributions.csv',index=False)
    pred[['date', 'ticker', 'y', *models]+(['prior'] if classification else [])].to_csv(OUT/f'{prefix}_predictions.csv', index=False)
    if classification:
        assert pred[list(models)].ge(0).all().all() and pred[list(models)].le(1).all().all()
        metrics = []
        for ticker,sub in pred.groupby('ticker'):
            target = (sub.y > 0).astype(int)
            prior_loss = np.mean((target-sub.prior)**2)
            for name in models:
                loss = np.mean((target-sub[name])**2)
                metrics.append({'ticker':ticker,'model':name,'brier':loss,'prior_brier':prior_loss,
                                'brier_skill_vs_prior':1-loss/prior_loss})
        pd.DataFrame(metrics).to_csv(OUT/f'{prefix}_probability_metrics.csv',index=False)
    stats = []
    for ticker, sub in pred.groupby('ticker'):
        for base, candidate in comparisons:
            target = (sub.y > 0).astype(int) if classification else sub.y
            d = loss_test(target.to_numpy(), sub[base], sub[candidate])
            loss_name = 'brier' if classification else 'mse'
            d[loss_name+'_improvement'] = d.pop('brier_improvement')
            stats.append({'ticker':ticker, 'baseline':base, 'candidate':candidate, 'n':len(sub),
                          'relative_'+loss_name+'_improvement':d[loss_name+'_improvement']/np.mean((target-sub[base])**2), **d})
    stats = pd.DataFrame(stats); stats['p_holm'] = holm(stats.p_hac)
    stats.to_csv(OUT/f'{prefix}_tests.csv', index=False)
    truth = pred.pivot(index='date', columns='ticker', values='y')
    signals = {name:pred.pivot(index='date', columns='ticker', values=name) for name in models}
    ranks = pd.DataFrame({name:s.corrwith(truth, axis=1, method='spearman') for name,s in signals.items()})
    ranks.to_csv(OUT/f'{prefix}_rank_ic.csv')
    ranks.groupby(ranks.index.year).mean().to_csv(OUT/f'{prefix}_annual_rank_ic.csv')
    tests = []
    for base,candidate in comparisons:
        delta = (ranks[candidate]-ranks[base]).dropna()
        b,se,p = hac(delta.to_numpy(), np.empty((len(delta),0)), 20)
        tests.append({'baseline':base, 'candidate':candidate, 'delta_ic':b[0],
                      'ci_low':b[0]-1.96*se[0], 'ci_high':b[0]+1.96*se[0], 'p_hac':p[0]})
    tests = pd.DataFrame(tests); tests['p_holm'] = holm(tests.p_hac)
    tests.to_csv(OUT/f'{prefix}_rank_tests.csv', index=False)
    blocks = []
    for name,s in signals.items():
        for offset in range(20):
            b = portfolio_blocks(prices,s,name,1,offset)
            assert (b.signal_date < b.date).all()
            b['equal11_return'] = [(prices.loc[x.exit,list(SECTORS)]/prices.loc[x.date,list(SECTORS)]-1).mean()
                                   for x in b.itertuples()]
            b['net_vs_equal11'] = b.gross_return-b.cost-b.equal11_return
            blocks.append(b.assign(offset=offset))
    blocks = pd.concat(blocks)
    blocks.to_csv(OUT/f'{prefix}_blocks.csv', index=False)
    offsets = blocks.groupby(['model','offset'])[['net_excess','net_vs_equal11']].mean()
    offsets.to_csv(OUT/f'{prefix}_offsets.csv')
    summary = offsets.groupby('model').agg(['min','median','max'])
    summary.to_csv(OUT/f'{prefix}_offset_summary.csv')
    meta = {'source':'Yahoo Finance adjusted ETF prices; not MSCI index levels',
            'tickers':TICKERS, 'cache_sha256':hashlib.sha256(CACHE.read_bytes()).hexdigest(),
            'proxy_counts':proxy.notna().sum().to_dict(),
            'proxy_first':{t:str(proxy[t].first_valid_index()) for t in TICKERS},
            'proxy_last':{t:str(proxy[t].last_valid_index()) for t in TICKERS},
            'oos_start':str(pred.date.min()), 'oos_end':str(pred.date.max()),
            'oos_sessions':int(pred.date.nunique()), 'features':factorcols,
            'residualized':residual,
            'classification':classification,
            'matched_sample':matched,
            'models':models,
            'comparisons':comparisons,
            'tuning':({'alphas':ALPHAS,'weights':[0.,*WEIGHTS], 'ic_tolerance':IC_TOLERANCE,
                       'inner_years':2, 'factor_features':tune_cols,
                       'selection':'equal-year mean rank IC; near-best prefers less weight then more regularization',
                       'insufficient_history':'base prediction, weight=0',
                       'interpretation':'model prediction attribution, not causal market attribution'} if tune else None),
            'pit_status':'current adjusted histories; original vintages and index methodology history unverified'}
    (OUT/f'{prefix}_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print('RANK IC',ranks.mean().to_dict()); print(tests.to_string(index=False))
    print(stats.to_string(index=False)); print(summary.to_string())


def self_test():
    ix = pd.bdate_range('2020-01-01', periods=100)
    p = pd.DataFrame({t:np.exp(np.arange(100)*.001*(i+1)) for i,t in enumerate(['SPY',*TICKERS])},index=ix)
    f = features(p)
    assert np.isclose(f.MTUM_rel20.iloc[80], .02)
    changed = p.copy(); changed.iloc[81:] *= 3
    pd.testing.assert_frame_equal(f.iloc[:81],features(changed).iloc[:81])
    p.loc[ix[80],'MTUM'] = np.nan
    assert np.isnan(features(p).loc[ix[80],'MTUM_rel20'])
    rng = np.random.default_rng(42)
    prices = pd.DataFrame(np.exp(np.cumsum(rng.normal(0,.01,(330,16)),axis=0)),
                          index=pd.bdate_range('2020-01-01',periods=330),columns=['SPY',*SECTORS,*TICKERS])
    actual = residual_features(prices)
    changed = prices.copy(); changed.iloc[315:] *= 2
    pd.testing.assert_frame_equal(actual.iloc[:315],residual_features(changed).iloc[:315])
    assert actual.iloc[312].notna().all()
    paired = matched_features(prices)
    expected = features(prices).join(actual).dropna()
    pd.testing.assert_frame_equal(paired, expected)
    assert paired.shape[1] == 16 and paired.index.min() == expected.index.min()
    pd.testing.assert_frame_equal(paired.loc[:prices.index[314]],
                                  matched_features(changed).loc[:prices.index[314]])
    print('MSCI feature alignment / future independence / missing-value checks passed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--download',action='store_true')
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--residual',action='store_true')
    parser.add_argument('--classification',action='store_true')
    parser.add_argument('--matched',action='store_true',help='Cache-only raw/residual comparison on identical training and evaluation rows')
    parser.add_argument('--tune',action='store_true',help='Nested annual tuning of residual60 alpha and blend; cache-only')
    args = parser.parse_args()
    if args.self_test:self_test()
    else:run(args.download,args.residual,args.classification,args.matched,args.tune)
