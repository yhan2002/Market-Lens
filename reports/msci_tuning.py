"""Small, predeclared nested tuning grid; no I/O or external requests."""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ALPHAS = (1., 10., 100., 1000.)
WEIGHTS = (.25, .5, 1.)
IC_TOLERANCE = .005


def fit(train, cols, alpha):
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    return model.fit(train[cols], train.y)


def inner_splits(train, year):
    assert (train.end < pd.Timestamp(year, 1, 1)).all()
    for inner_year in (year-2, year-1):
        cutoff = pd.Timestamp(inner_year, 1, 1)
        before = train[train.end < cutoff]
        valid = train[train.date.dt.year.eq(inner_year)]
        counts = before.groupby('ticker').size().reindex(train.ticker.unique(), fill_value=0)
        if counts.min() < 252 or valid.date.nunique() < 126:
            continue
        assert before.end.max() < valid.date.min()
        yield inner_year, before, valid


def rank_ic(frame, prediction):
    result = frame[['date', 'ticker', 'y']].assign(prediction=prediction)
    truth = result.pivot(index='date', columns='ticker', values='y')
    estimate = result.pivot(index='date', columns='ticker', values='prediction')
    values = estimate.corrwith(truth, axis=1, method='spearman')
    if not np.isfinite(values).all():
        raise ValueError('Undefined sector rank correlation; do not silently discard dates')
    return float(values.mean())


def choose(scores):
    best = max(s['mean_ic'] for s in scores)
    near = [s for s in scores if s['mean_ic'] >= best-IC_TOLERANCE]
    return min(near, key=lambda s: (s['weight'], -s['alpha']))


def select(train, year, basecols, factorcols):
    folds = list(inner_splits(train, year))
    if len(folds) < 2:
        return {'alpha':1., 'weight':0., 'folds':len(folds), 'status':'insufficient_inner_history'}, [], []
    configs = [(1., 0.)]+[(a, w) for a in ALPHAS for w in WEIGHTS]
    scores = {c:[] for c in configs}
    audit = []
    cols = basecols+factorcols
    for inner_year, before, valid in folds:
        base = pd.Series(index=valid.index, dtype=float)
        augmented = {a:base.copy() for a in ALPHAS}
        for ticker, sub in before.groupby('ticker'):
            mask = valid.ticker.eq(ticker)
            base.loc[mask] = fit(sub, basecols, 1.).predict(valid.loc[mask, basecols])
            for a in ALPHAS:
                augmented[a].loc[mask] = fit(sub, cols, a).predict(valid.loc[mask, cols])
        assert base.notna().all() and all(p.notna().all() for p in augmented.values())
        for a,w in configs:
            scores[a,w].append(rank_ic(valid, base+w*(augmented[a]-base)))
        audit.append({'outer_year':year, 'inner_year':inner_year,
                      'train_dates':before.date.nunique(), 'valid_dates':valid.date.nunique(),
                      'train_last_label_end':str(before.end.max()),
                      'valid_start':str(valid.date.min()), 'valid_last_label_end':str(valid.end.max())})
    rows = [{'outer_year':year, 'alpha':a, 'weight':w, 'mean_ic':float(np.mean(v)),
             'fold1_ic':v[0], 'fold2_ic':v[1]} for (a,w),v in scores.items()]
    selected = dict(choose(rows), folds=2, status='selected_on_past_inner_folds')
    return selected, rows, audit


def predict_explain(train, test, basecols, factorcols, alpha, weight):
    cols = basecols+factorcols
    base = fit(train, basecols, 1.)
    augmented = fit(train, cols, alpha)
    b = base.predict(test[basecols])
    prediction = b+weight*(augmented.predict(test[cols])-b)
    scaler, ridge = augmented.steps[0][1], augmented.steps[1][1]
    coef = weight*ridge.coef_.copy()
    coef[:len(basecols)] += (1-weight)*base.steps[1][1].coef_
    # Both models see identical training rows, so shared feature scales coincide.
    np.testing.assert_allclose(scaler.scale_[:len(basecols)], base.steps[0][1].scale_)
    intercept = weight*ridge.intercept_+(1-weight)*base.steps[1][1].intercept_
    contribution = pd.DataFrame(scaler.transform(test[cols])*coef, columns=cols, index=test.index)
    contribution['intercept'] = intercept
    np.testing.assert_allclose(contribution.sum(axis=1), prediction, atol=1e-12)
    coefficients = pd.DataFrame({'feature':cols, 'standardized_coefficient':coef,
                                 'training_mean':scaler.mean_, 'training_scale':scaler.scale_,
                                 'raw_slope':coef/scaler.scale_})
    return prediction, coefficients, contribution


def self_test():
    rng = np.random.default_rng(8)
    dates = pd.bdate_range('2017-01-01', '2020-12-31')
    rows = []
    for ticker in ['a','b','c']:
        x = rng.normal(size=len(dates))
        f = rng.normal(size=len(dates))
        rows.append(pd.DataFrame({'date':dates,'end':dates+pd.offsets.BDay(20),
                                  'ticker':ticker,'x':x,'f':f,'y':x+.2*f}))
    frame = pd.concat(rows, ignore_index=True)
    train = frame[frame.end < pd.Timestamp('2021-01-01')]
    first, scores, audit = select(train, 2021, ['x'], ['f'])
    assert first['folds'] == 2 and len(scores) == 13 and len(audit) == 2
    assert all(pd.Timestamp(a['train_last_label_end']) < pd.Timestamp(a['valid_start']) for a in audit)
    test = frame.tail(9).copy()
    p, coef, contributions = predict_explain(train, test, ['x'], ['f'], 10., .25)
    test['y'] = 1e9
    q, _, _ = predict_explain(train, test, ['x'], ['f'], 10., .25)
    np.testing.assert_allclose(p,q)
    np.testing.assert_allclose(contributions.sum(axis=1),p)
    zero, zero_coef, _ = predict_explain(train,test,['x'],['f'],1000.,0.)
    np.testing.assert_allclose(zero,fit(train,['x'],1.).predict(test[['x']]))
    assert zero_coef.loc[zero_coef.feature.eq('f'),'standardized_coefficient'].iloc[0] == 0
    assert choose([{'alpha':1.,'weight':0.,'mean_ic':.1},
                   {'alpha':10.,'weight':.25,'mean_ic':.104}])['weight'] == 0
    assert select(train[train.date >= '2020-01-01'],2021,['x'],['f'])[0]['weight'] == 0
    print('Tuning: purged inner splits, 13 candidates, target isolation, attribution and fallback passed')


if __name__ == '__main__':
    self_test()
