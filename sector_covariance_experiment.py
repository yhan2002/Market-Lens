"""Local research: incremental covariance value for sector turning points."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from it_internals_shadow import SECTOR_ETFS, load_it_price_matrix
from research_drawdown_onset import causal_trailing_percentile
from sector_turning_pooled_shadow import (
    DIRECTIONS, PRICE_FEATURES, _direction_frame, _early_event_metrics,
    _json_ready, _labels_with_calendar_gaps, _price_features, effect_code,
    paired_date_block_auc_delta_ci, paired_date_block_pre_pivot_recall_delta_ci,
    purged_training_mask,
)
from sector_turning_price_multiscale_experiment import (
    add_price_path_features, HORIZON_FEATURES, CURVATURE_FEATURES,
)
from sector_turning_recall_model_compare import estimator
from sector_turning_tradable_label_shadow import qualify_labels

ROOT = Path(__file__).resolve().parent
PRICE = (*PRICE_FEATURES, *HORIZON_FEATURES, *CURVATURE_FEATURES)
VOL = ('own_vol_126', 'own_vol_ratio_20_126', 'market_vol_126', 'market_vol_ratio_20_126')
COV = ('beta_126', 'beta_change_20_126', 'peer_corr_change_20_126',
       'common_component_change', 'risk_share_change', 'residual_return_5', 'residual_return_20')
MODELS = {'PRICE': PRICE, 'PRICE_VOL': (*PRICE, *VOL), 'PRICE_VOL_COV': (*PRICE, *VOL, *COV)}
ALL_FEATURES = MODELS['PRICE_VOL_COV']


def covariance_features(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate matrices through t-1; use today's return only after the close."""
    tickers = list(prices.columns[:-1])
    n = len(tickers)
    returns = prices.pct_change(fill_method=None)
    rows, forecasts = [], []
    w = np.ones(n) / n
    for t in range(127, len(prices)):
        history = returns.iloc[t - 126:t].to_numpy()
        if not np.isfinite(history).all():
            continue
        covs = [LedoitWolf().fit(history[-window:]).covariance_ for window in (20, 126)]
        stats = []
        for cov in covs:
            sd = np.sqrt(np.diag(cov))
            corr = cov / np.outer(sd, sd)
            sector_cov = cov[:n, :n]
            variance = float(w @ sector_cov @ w)
            stats.append({
                'beta': cov[:n, n] / cov[n, n],
                'peer_corr': (corr[:n, :n].sum(axis=1) - 1) / (n - 1),
                'common': np.linalg.eigvalsh(corr[:n, :n])[-1] / n,
                'share': w * (sector_cov @ w) / variance,
                'variance': variance,
            })
        short, long = stats
        # Volatility controls use the same raw return samples, independent of correlation.
        raw_sd = [history[-window:].std(axis=0, ddof=1) for window in (20, 126)]
        current = returns.iloc[t].to_numpy()
        for j, ticker in enumerate(tickers):
            rows.append({
                'date': prices.index[t], 'sector': ticker,
                'own_vol_126': raw_sd[1][j] * np.sqrt(252),
                'own_vol_ratio_20_126': raw_sd[0][j] / raw_sd[1][j],
                'market_vol_126': raw_sd[1][n] * np.sqrt(252),
                'market_vol_ratio_20_126': raw_sd[0][n] / raw_sd[1][n],
                'beta_126': long['beta'][j],
                'beta_change_20_126': short['beta'][j] - long['beta'][j],
                'peer_corr_change_20_126': short['peer_corr'][j] - long['peer_corr'][j],
                'common_component_change': short['common'] - long['common'],
                'risk_share_change': short['share'][j] - long['share'][j],
                'residual_return_1': current[j] - long['beta'][j] * current[n],
                'risk_share': long['share'][j],
            })
        next_returns = returns.iloc[t:t + 20, :n]
        if len(next_returns) == 20 and next_returns.notna().all().all():
            realized = float(next_returns.mean(axis=1).var(ddof=1))
            sample = np.cov(history[:, :n], rowvar=False, ddof=1)
            forecasts.append({
                'date': prices.index[t], 'realized_variance_20': realized,
                'SAMPLE_COV': float(w @ sample @ w),
                'SHRUNK_COV': long['variance'],
                'IGNORE_CORRELATION': float(np.sum(w ** 2 * np.diag(sample))),
            })
    features = pd.DataFrame(rows).sort_values(['sector', 'date'])
    for h in (5, 20):
        features[f'residual_return_{h}'] = features.groupby('sector')['residual_return_1'].transform(
            lambda s: s.rolling(h, min_periods=h).sum())
    return features, pd.DataFrame(forecasts)


def build_frame(prices: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for ticker in prices.columns[:-1]:
        relative = prices[ticker] / prices['SPY']
        part = _price_features(relative).join(_labels_with_calendar_gaps(relative, horizon=10))
        part['date'], part['sector'], part['sector_name'] = part.index, ticker, ticker
        part['relative_price'], part['_position'] = relative, np.arange(len(relative))
        # Execution at the following close, held for 20 sessions; arithmetic excess return.
        part['forward_excess_20'] = (
            prices[ticker].shift(-21) / prices[ticker].shift(-1)
            - prices['SPY'].shift(-21) / prices['SPY'].shift(-1))
        parts.append(part.reset_index(drop=True))
    pooled = add_price_path_features(pd.concat(parts, ignore_index=True))
    return pooled.merge(features, on=['date', 'sector'], validate='one_to_one', how='left')


def walk_forward(pooled: pd.DataFrame, direction: str, end: pd.Timestamp) -> pd.DataFrame:
    qualified, audit = qualify_labels(pooled, direction)
    prefix = 'top' if direction == 'ROTATION_OUT' else 'bottom'
    # Qualification at pivot+20 must never precede the original confirmation window.
    for suffix in ('position', 'date'):
        column = f'{prefix}_label_end_{suffix}'
        earlier = qualified[column].notna() & qualified[column].lt(pooled[column])
        qualified.loc[earlier, column] = pooled.loc[earlier, column]
        if suffix == 'date':
            audit['availability_regressions_prevented'] = int(earlier.sum())
    print(json.dumps({'direction': direction, 'labels': audit}), flush=True)
    frame = _direction_frame(qualified, direction, target_mode='PRE_PIVOT')
    complete = np.isfinite(frame.loc[:, ALL_FEATURES]).all(axis=1)
    dates = pd.DatetimeIndex(frame.loc[frame['date'].between('2021-01-01', end), 'date'].unique()).sort_values()
    levels = tuple(sorted(frame['sector'].unique()))
    parts = []
    for i in range(0, len(dates), 21):
        block = dates[i:i + 21]
        train = frame.loc[purged_training_mask(frame, block[0]) & complete]
        test = frame.loc[frame['at_risk'] & complete & frame['date'].isin(block)]
        if train['date'].nunique() < 252 or train['target'].nunique() < 2 or test.empty:
            continue
        assert train['label_end_date'].max() < block[0]
        train_effects, _ = effect_code(train['sector'], levels)
        test_effects, _ = effect_code(test['sector'], levels)
        for name, columns in MODELS.items():
            scaler = StandardScaler().fit(train.loc[:, columns])
            x = np.column_stack((scaler.transform(train.loc[:, columns]), train_effects))
            y = np.column_stack((scaler.transform(test.loc[:, columns]), test_effects))
            model = estimator('SHALLOW_BOOST').fit(x, train['target'].astype(int))
            out = test[['date', 'sector', 'sector_name', '_position', 'target',
                        'target_event_pivot_position', 'target_event_confirmation_position',
                        'target_event_confirmation_date', 'forward_excess_20']].copy()
            out['model'], out['direction'] = name, direction
            out['refit_date'] = block[0]
            out['training_label_end'] = train['label_end_date'].max()
            out['raw_score'] = model.predict_proba(y)[:, 1]
            parts.append(out)
        if i % 210 == 0:
            print(f'{direction}: {block[0].date()} ({i}/{len(dates)} dates)', flush=True)
    predicted = pd.concat(parts, ignore_index=True).sort_values(['model', 'sector', 'date'])
    predicted['score'] = predicted.groupby(['model', 'sector'])['raw_score'].transform(
        lambda s: causal_trailing_percentile(s, window=504, min_history=60))
    return predicted.dropna(subset=['score', 'target'])


def signal_returns(rows: pd.DataFrame, cutoff: float) -> pd.DataFrame:
    # Fixed 20-session holding period; no overlapping positions in the same sector/direction.
    accepted = []
    for _, group in rows.groupby('sector'):
        previous = -10_000
        for index, row in group.loc[group['score'].ge(cutoff)].sort_values('_position').iterrows():
            if row['_position'] - previous >= 21 and pd.notna(row['forward_excess_20']):
                accepted.append(index)
                previous = row['_position']
    result = rows.loc[accepted].copy()
    result['signed_return'] = result['forward_excess_20'] * np.where(result['direction'].eq('ROTATION_IN'), 1, -1)
    return result


def fixed_alert_budget(rows: pd.DataFrame, fraction: float = .3) -> pd.DataFrame:
    """Same-day, same-universe top scores; no future outcomes determine the cutoff."""
    result = rows.sort_values(['date', 'score', 'sector'], ascending=[True, False, True]).copy()
    rank = result.groupby('date').cumcount()
    count = result.groupby('date')['sector'].transform('size')
    result['score'] = rank.lt(np.ceil(count * fraction)).astype(float)
    return result


def paired_mean_ci(frame: pd.DataFrame, columns: list[str], *, samples: int = 500):
    """40-session moving blocks, retaining cross-sectional observations on the same date."""
    daily = frame.groupby('date')[columns].agg(['sum', 'count'])
    sums = daily.xs('sum', axis=1, level=1).to_numpy()
    counts = daily.xs('count', axis=1, level=1).to_numpy()
    rng = np.random.default_rng(42)
    estimates = []
    n = len(daily)
    for _ in range(samples):
        starts = rng.integers(0, n - 40 + 1, size=int(np.ceil(n / 40)))
        idx = (starts[:, None] + np.arange(40)).ravel()[:n]
        estimates.append(sums[idx].sum(axis=0) / counts[idx].sum(axis=0))
    return {c: {'mean': float(frame[c].mean()), 'ci_low': float(np.quantile(estimates, .025, axis=0)[i]),
                'ci_high': float(np.quantile(estimates, .975, axis=0)[i])} for i, c in enumerate(columns)}


def summarize(predictions: pd.DataFrame, forecasts: pd.DataFrame, prices: pd.DataFrame, output: Path):
    results, comparisons = {}, {}
    evaluated = predictions.loc[predictions['date'].ge('2023-01-01')].copy()
    for direction in DIRECTIONS:
        rows = evaluated.loc[evaluated['direction'].eq(direction)]
        by_model = {}
        for name in MODELS:
            part = rows.loc[rows['model'].eq(name)]
            measures = {'auc': float(roc_auc_score(part['target'], part['score'])), 'n': len(part)}
            for cutoff in (.7, .6):
                event = _early_event_metrics(part, pd.DataFrame(), '', horizon=10, cutoff=cutoff)
                alerts = part['score'].ge(cutoff)
                trades = signal_returns(part, cutoff)
                event['false_alert_day_fraction'] = float(part.loc[alerts, 'target'].eq(0).mean())
                event['nonoverlap_signals'] = len(trades)
                event['mean_signed_excess_20'] = float(trades['signed_return'].mean())
                event['signed_excess_win_rate'] = float(trades['signed_return'].gt(0).mean())
                measures[str(cutoff)] = event
            by_model[name] = measures
        results[direction] = by_model
        candidate = rows.loc[rows['model'].eq('PRICE_VOL_COV')]
        baseline = rows.loc[rows['model'].eq('PRICE_VOL')]
        keys = ['date', 'sector']
        assert candidate[keys].reset_index(drop=True).equals(baseline[keys].reset_index(drop=True))
        comparisons[direction] = {
            'auc': paired_date_block_auc_delta_ci(candidate.set_index(keys)['target'],
                candidate.set_index(keys)['score'], baseline.set_index(keys)['score'], samples=500),
            'recall_0.7': paired_date_block_pre_pivot_recall_delta_ci(candidate, baseline,
                samples=500, cutoff=.7, horizon=10),
            'recall_0.6': paired_date_block_pre_pivot_recall_delta_ci(candidate, baseline,
                samples=500, cutoff=.6, horizon=10),
        }
        budget_candidate, budget_baseline = fixed_alert_budget(candidate), fixed_alert_budget(baseline)
        assert budget_candidate.groupby('date')['score'].sum().equals(
            budget_baseline.groupby('date')['score'].sum())
        comparisons[direction]['same_daily_alert_budget'] = {
            'rule': 'Top ceil(30% of eligible sectors) each date, ties broken by ticker',
            'candidate': _early_event_metrics(budget_candidate, pd.DataFrame(), '', horizon=10, cutoff=.5),
            'baseline': _early_event_metrics(budget_baseline, pd.DataFrame(), '', horizon=10, cutoff=.5),
            'recall_delta_ci': paired_date_block_pre_pivot_recall_delta_ci(
                budget_candidate, budget_baseline, samples=500, cutoff=.5, horizon=10),
        }
        print(f'{direction}: evaluation and paired confidence intervals complete', flush=True)
    # Overlapping 20-session outcomes; 40-session bootstrap blocks retain dependence.
    risk = forecasts.loc[forecasts['date'].ge('2023-01-01')].copy()
    for name in ('SAMPLE_COV', 'SHRUNK_COV', 'IGNORE_CORRELATION'):
        ratio = risk['realized_variance_20'] / risk[name]
        risk[name + '_qlike'] = ratio - np.log(ratio) - 1
    risk['shrink_minus_sample_loss'] = risk['SHRUNK_COV_qlike'] - risk['SAMPLE_COV_qlike']
    risk['cov_minus_ignore_loss'] = risk['SHRUNK_COV_qlike'] - risk['IGNORE_CORRELATION_qlike']
    risk_report = paired_mean_ci(risk, ['SAMPLE_COV_qlike', 'SHRUNK_COV_qlike',
        'IGNORE_CORRELATION_qlike', 'shrink_minus_sample_loss', 'cov_minus_ignore_loss'])
    per_sector = []
    for (direction, model, sector), group in evaluated.groupby(['direction', 'model', 'sector']):
        per_sector.append({'direction': direction, 'model': model, 'sector': sector,
            **_early_event_metrics(group, pd.DataFrame(), '', horizon=10, cutoff=.7)})
    summary = _json_ready({
        'created_at': datetime.now(timezone.utc).isoformat(),
        'status': 'RESEARCH_ONLY',
        'data': {'start': str(prices.index.min().date()), 'end': str(prices.index.max().date()),
                 'sessions': len(prices), 'sectors': list(SECTOR_ETFS),
                 'sha256': hashlib.sha256(prices.to_csv().encode()).hexdigest()},
        'evaluation': {'start': str(evaluated['date'].min().date()), 'end': str(evaluated['date'].max().date()),
                       'label': 'SPY-relative PRE_PIVOT; confirmation within 10 sessions; pivot+20 followthrough >=3%',
                       'models': MODELS, 'refit_sessions':21,
                       'censor': 'last 30 price sessions excluded; max(original confirmation window, pivot+20) strictly before refit',
                       'score': 'causal within-sector percentile, previous 504 eligible observations; minimum 60',
                       'limitations': ['Current revised adjusted prices; not vintage-PIT verified',
                           'Previously reviewed evaluation years, not untouched prospective OOS',
                           'Rising/falling relative to SPY, not absolute price turning points',
                           'Nominal percentile thresholds do not guarantee identical realized alert rates',
                           'Event recall denominator is confirmations represented in common eligible prediction rows',
                           'False alert day fraction is not an independent-event probability',
                           'OUT excess return is avoided relative loss, not realized portfolio profit',
                           'No transaction costs; confidence intervals are pointwise, not multiple-test adjusted']},
        'models':results, 'incremental_covariance_vs_vol':comparisons, 'risk_forecast':risk_report,
    })
    predictions.to_csv(output / 'predictions.csv', index=False)
    risk.to_csv(output / 'risk_forecasts.csv', index=False)
    pd.DataFrame(per_sector).to_csv(output / 'per_sector.csv', index=False)
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    lines = ['# 공분산 추가 추세 전환 백테스트', '', '연구용 · 조정종가 수정 이력 및 진정한 신규 OOS 미검증', '',
             f"평가기간: {summary['evaluation']['start']} ~ {summary['evaluation']['end']}", '',
             '|방향|모델|사전포착률|경보일 비율|경보 중 비포착일|선행 거래일 중앙값|20일 방향조정 초과수익|',
             '|---|---|---:|---:|---:|---:|---:|']
    for direction, models in results.items():
        for name, m in models.items():
            x=m['0.7']
            lines.append(f"|{direction}|{name}|{x['pre_pivot_recall']:.1%}|{x['alert_day_rate']:.1%}|{x['false_alert_day_fraction']:.1%}|{x['median_lead_to_pivot_sessions']}|{x['mean_signed_excess_20']:.2%}|")
    lines += ['', '세부 비교와 95% 신뢰구간은 summary.json, 섹터별 결과는 per_sector.csv에 저장.', '',
              '코드 사양은 사전 고정한 3개이며 결과를 본 뒤 튜닝하지 않음. 운영 모델·DB·웹 변경 없음.']
    (output / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    return summary


def self_test():
    rng = np.random.default_rng(6)
    dates = pd.bdate_range('2020-01-01', periods=175)
    values = np.exp(np.cumsum(rng.normal(0, .01, (175, 4)), axis=0)) * 100
    p = pd.DataFrame(values, index=dates, columns=['A', 'B', 'C', 'SPY'])
    a, _ = covariance_features(p)
    changed = p.copy(); changed.iloc[155:] *= 1.2
    b, _ = covariance_features(changed)
    pd.testing.assert_frame_equal(a.loc[a['date'] < dates[155]], b.loc[b['date'] < dates[155]])
    # Matrix values on day 155 cannot depend on that day's price.
    pd.testing.assert_frame_equal(a.loc[a['date'].eq(dates[155]), list(COV[:5])],
                                  b.loc[b['date'].eq(dates[155]), list(COV[:5])])
    assert np.allclose(a.groupby('date')['risk_share'].sum(), 1)
    assert np.isfinite(a.loc[:, (*VOL, *COV)].dropna()).all().all()
    sample = pd.DataFrame({'date': [dates[0]] * 4, 'sector': ['D', 'C', 'B', 'A'],
                           'score': [.2, .9, .9, .1]})
    budget = fixed_alert_budget(sample)
    assert set(budget.loc[budget['score'].eq(1), 'sector']) == {'B', 'C'}
    print('self-test PASS: t-1 matrix cutoff, future perturbation, finite features, risk shares sum to one')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/sector_covariance_20260914')
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    prices = load_it_price_matrix(ROOT / 'regime_router_publish.db', tickers=(*SECTOR_ETFS, 'SPY'))
    print(f'input: {len(prices)} sessions, {prices.index[0].date()}..{prices.index[-1].date()}', flush=True)
    with threadpool_limits(limits=2):
        features, forecasts = covariance_features(prices)
        pooled = build_frame(prices, features)
        parts = []
        for direction in DIRECTIONS:
            predicted = walk_forward(pooled, direction, prices.index[-31])
            predicted.to_csv(args.output_dir / f'{direction}.csv', index=False)
            parts.append(predicted)
        summary = summarize(pd.concat(parts, ignore_index=True), forecasts, prices, args.output_dir)
    print(json.dumps({'output': str(args.output_dir), 'evaluation': summary['evaluation']['end']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
