"""Research-only reconstruction of public signal outputs; never executes site JS."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from itertools import product
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, export_text

SOURCE = 'https://raw.githubusercontent.com/jangdt/CREAMCHEESE-HOUSE-SITE/816dc4439508a45c25d18438db6697b6aca8b66d/index.html'
OUTPUT = Path(__file__).resolve().parent / 'outputs/creamcheese_reverse_20260914'
STATES = {'GREEN': 0, 'YELLOW': 1, 'RED': 2, 'BLACK': 3}
BASE_FEATURES = ['b20', 'b200', 'nhnl5', 'nhnl20', 'adr', 'crash', 'dd', 'rv', 'rvp',
                 'gap20', 'gap50', 'gap200', 'slope50_5', 'return5', 'market_is_kosdaq']
FEATURES = [*BASE_FEATURES, 'm50u', 'h20', 'relax']


def extract_snapshot(html: bytes) -> dict:
    match = re.search(r'<script id="dtdata" type="application/json">(.*?)</script>',
                      html.decode('utf-8'), re.S)
    if not match:
        raise ValueError('Public dtdata JSON not found')
    data = json.loads(match[1])
    # Keep only public market signals, not stock lists or public service credentials.
    return {'source': SOURCE, 'sha256': hashlib.sha256(html).hexdigest(),
            'retrieved_utc': datetime.now(timezone.utc).isoformat(),
            'as_of': data['as_of'], 'universe': data['universe'],
            'gate': data['regime']['gate']}


def load_frame(snapshot: dict) -> pd.DataFrame:
    parts = []
    for market in ('KOSPI', 'KOSDAQ'):
        part = pd.DataFrame(snapshot['gate'][market]['series'])
        part['market'] = market
        part['date'] = pd.to_datetime(part['d'], format='%Y%m%d', errors='raise')
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True).sort_values(['market', 'date'])
    if frame.duplicated(['market', 'date']).any():
        raise ValueError('Duplicate market-date')
    frame = frame.loc[frame['date'].le(pd.Timestamp(snapshot['as_of']))].copy()
    for column in ('sc', 'lc', 's', 'l', 'fc', 'f', 'base'):
        if not frame[column].isin([*STATES, 'NA']).all():
            raise ValueError(f'Unknown signal state: {column}')
    frame['market_is_kosdaq'] = frame['market'].eq('KOSDAQ').astype(float)
    for _, group in frame.groupby('market'):
        for window in (20, 50, 200):
            ma = group['idx'].rolling(window, min_periods=window).mean()
            frame.loc[group.index, f'gap{window}'] = group['idx'] / ma - 1
            if window == 50:
                frame.loc[group.index, 'slope50_5'] = ma.pct_change(5, fill_method=None)
        frame.loc[group.index, 'return5'] = group['idx'].pct_change(5, fill_method=None)
    if np.isinf(frame[FEATURES].to_numpy(dtype=float)).any():
        raise ValueError('Infinite input feature')
    return frame.reset_index(drop=True)


def stabilize(candidates, *, up: int, yellow: int, red: int, black: int,
              streak_mode: str = 'exact') -> np.ndarray:
    """Causal candidate-to-confirmed-state hypothesis, not the original implementation."""
    state, pending, count = 'NA', None, 0
    result = []
    for candidate in candidates:
        if candidate == 'NA' or state == 'NA':
            state, pending, count = candidate, None, 0
        elif candidate == state:
            pending, count = None, 0
        else:
            improving = STATES[candidate] < STATES[state]
            key = candidate if streak_mode == 'exact' else improving
            count = count + 1 if key == pending else 1
            pending = key
            needed = up if improving else {'YELLOW': yellow, 'RED': red, 'BLACK': black}[candidate]
            if count >= needed:
                state, pending, count = candidate, None, 0
        result.append(state)
    return np.asarray(result, dtype=object)


def stabilize_markets(frame: pd.DataFrame, candidates, config: dict) -> np.ndarray:
    result = np.empty(len(frame), dtype=object)
    candidates = np.asarray(candidates)
    for _, group in frame.groupby('market'):
        result[group.index] = stabilize(candidates[group.index], **config)
    return result


def fuse(short, long) -> np.ndarray:
    # Inferred from training combinations; long risk overrides, short RED alone caps at YELLOW.
    return np.array([l if l in ('RED', 'BLACK') else 'NA' if 'NA' in (s, l)
                     else 'GREEN' if s == l == 'GREEN' else 'YELLOW'
                     for s, l in zip(short, long)], dtype=object)


def metrics(actual, predicted) -> dict:
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    risk, alert = np.isin(actual, ['RED', 'BLACK']), np.isin(predicted, ['RED', 'BLACK'])
    return {'n': len(actual), 'correct': int(np.sum(actual == predicted)),
            'accuracy': float(accuracy_score(actual, predicted)),
            'balanced_accuracy': float(balanced_accuracy_score(actual, predicted)),
            'risk_days': int(risk.sum()), 'missed_risk_days': int((risk & ~alert).sum()),
            'risk_recall': float(alert[risk].mean()) if risk.any() else None,
            'false_risk_days': int((~risk & alert).sum()),
            'class_counts': pd.Series(actual).value_counts().to_dict(),
            'confusion_labels': ['NA', *STATES],
            'confusion_matrix': confusion_matrix(actual, predicted, labels=['NA', *STATES]).tolist()}


def transitions(frame: pd.DataFrame, predicted, mask) -> dict:
    """One-to-one same destination matches within five sessions, never across markets."""
    total = matched = false = 0
    offsets = []
    predicted = np.asarray(predicted)
    for _, group in frame.groupby('market'):
        positions = np.flatnonzero(mask[group.index])
        real = group['f'].to_numpy()
        pred = predicted[group.index]
        actual_changes = [p for p in positions if p > 0 and real[p] != real[p - 1]]
        model_changes = [p for p in positions if p > 0 and pred[p] != pred[p - 1]]
        available = set(model_changes)
        total += len(actual_changes)
        for p in actual_changes:
            choices = [q for q in available if pred[q] == real[p] and abs(q - p) <= 5]
            if choices:
                q = min(choices, key=lambda q: (abs(q - p), q))
                available.remove(q)
                matched += 1
                offsets.append(q - p)
        false += len(available)
    return {'actual_transitions': total, 'matched_within_5_sessions': matched,
            'unmatched_model_transitions': false,
            'median_offset_sessions': float(np.median(offsets)) if offsets else None,
            'median_absolute_offset': float(np.median(np.abs(offsets))) if offsets else None}


def run_experiment(frame: pd.DataFrame, snapshot: dict):
    dates = np.sort(frame['date'].unique())
    split, inner = dates[int(len(dates) * .7) - 1], dates[int(len(dates) * .525) - 1]
    train = frame['date'].le(split).to_numpy()
    test = ~train
    tuning = train & frame['date'].gt(inner).to_numpy()
    inner_train = frame['date'].le(inner).to_numpy()
    report = {'status': 'OUTPUT_REPLICATION_ONLY_NOT_TRADING_BACKTEST',
              'source': snapshot['source'], 'input_sha256': snapshot['sha256'],
              'split': {'train_end': str(pd.Timestamp(split).date()),
                        'test_start': str(pd.Timestamp(dates[int(len(dates)*.7)]).date()),
                        'test_end': str(frame['date'].max().date()),
                        'train_market_days': int(train.sum()), 'test_market_days': int(test.sum())},
              'coverage': {str(k): int(v) for k, v in frame[FEATURES].isna().sum().items()},
              'hysteresis': {}, 'candidate_models': {}, 'test': {}}
    configs = [dict(zip(('up', 'yellow', 'red', 'black', 'streak_mode'), values))
               for values in product((3, 2, 4), (1, 2), (2, 1), (1, 2), ('exact', 'direction'))]
    for candidate, confirmed in (('sc', 's'), ('lc', 'l')):
        scores = []
        for config in configs:
            pred = stabilize_markets(frame, frame[candidate], config)
            scores.append(float(np.mean(pred[train] == frame.loc[train, confirmed])))
        selected = int(np.argmax(scores))
        config = configs[selected]
        frame[f'conditional_{confirmed}'] = stabilize_markets(frame, frame[candidate], config)
        report['hysteresis'][confirmed] = {'config': config, 'train_accuracy': scores[selected],
            'equally_good_configurations': sum(s == scores[selected] for s in scores),
            'test': metrics(frame.loc[test, confirmed], frame.loc[test, f'conditional_{confirmed}'])}
    report['test']['fusion_only_true_states'] = metrics(frame.loc[test, 'f'], fuse(frame['s'], frame['l'])[test])
    report['test']['candidate_fusion_only'] = metrics(frame.loc[test, 'fc'], fuse(frame['sc'], frame['lc'])[test])
    frame['conditional_f'] = fuse(frame['conditional_s'], frame['conditional_l'])
    report['test']['confirmed_from_true_candidates'] = metrics(frame.loc[test, 'f'], frame.loc[test, 'conditional_f'])
    report['test']['confirmed_from_true_candidates']['transitions'] = transitions(frame, frame['conditional_f'], test)
    for variant, columns in (('raw_indicators', BASE_FEATURES), ('with_published_flags', FEATURES)):
        report['candidate_models'][variant] = {}
        for target in ('sc', 'lc'):
            scored = []
            for depth, leaf in product((2, 3, 4, 5, 6), (5, 2)):
                model = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, random_state=42)
                model.fit(frame.loc[inner_train, columns], frame.loc[inner_train, target])
                score = accuracy_score(frame.loc[tuning, target], model.predict(frame.loc[tuning, columns]))
                scored.append((score, -depth, leaf, model))
            score, negdepth, leaf, _ = max(scored, key=lambda row: row[:3])
            model = DecisionTreeClassifier(max_depth=-negdepth, min_samples_leaf=leaf, random_state=42)
            model.fit(frame.loc[train, columns], frame.loc[train, target])
            prediction = model.predict(frame[columns])
            frame[f'{variant}_{target}'] = prediction
            report['candidate_models'][variant][target] = {
                'max_depth': -negdepth, 'min_samples_leaf': leaf, 'inner_validation_accuracy': score,
                'training_classes': list(model.classes_),
                'train': metrics(frame.loc[train, target], prediction[train]),
                'test': metrics(frame.loc[test, target], prediction[test]),
                'rules': export_text(model, feature_names=columns, decimals=4)}
            model_path = OUTPUT / f'{variant}_{target}_rules.txt'
            model_path.write_text(export_text(model, feature_names=columns, decimals=4), encoding='utf-8')
        short = stabilize_markets(frame, frame[f'{variant}_sc'], report['hysteresis']['s']['config'])
        long = stabilize_markets(frame, frame[f'{variant}_lc'], report['hysteresis']['l']['config'])
        frame[f'{variant}_f'] = fuse(short, long)
        report['test'][variant] = metrics(frame.loc[test, 'f'], frame.loc[test, f'{variant}_f'])
        report['test'][variant]['transitions'] = transitions(frame, frame[f'{variant}_f'], test)
        report['test'][variant]['per_market'] = {
            market: metrics(group['f'], group[f'{variant}_f'])
            for market, group in frame.loc[test].groupby('market')}
    persistence = frame.groupby('market')['f'].shift(1).fillna('NA')
    report['test']['yesterday_signal_baseline'] = metrics(frame.loc[test, 'f'], persistence[test])
    report['limitations'] = [
        'Only last 320 days per market; underlying MA200 warmup history not fully exposed',
        'Numerous missing breadth inputs; no price forward-fill or interpolation',
        'Public engineered flags are intermediate outputs, not independently reconstructed inputs',
        'Conditional reconstruction supplies genuine candidate outputs and is not end-to-end prediction',
        'Rules and thresholds may have been revised before this historical series was published',
        'One chronological holdout, no retuning after inspection; inference is not unique',
        'Signal agreement is NOT future-return prediction, early turning-point recall, or investment performance',
    ]
    frame['split'] = np.where(train, 'TRAIN', 'TEST')
    frame.to_csv(OUTPUT / 'replication_rows.csv', index=False)
    (OUTPUT / 'results.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(json.dumps({'split': report['split'], 'hysteresis': report['hysteresis'], 'test': report['test']},
                     ensure_ascii=False, indent=2))


def self_test():
    config = dict(up=3, yellow=1, red=2, black=1)
    values = ['GREEN', 'YELLOW', 'RED', 'RED', 'GREEN', 'GREEN', 'GREEN']
    assert stabilize(values, **config).tolist() == ['GREEN', 'YELLOW', 'YELLOW', 'RED', 'RED', 'RED', 'GREEN']
    assert fuse(['RED', 'GREEN', 'NA'], ['GREEN', 'BLACK', 'GREEN']).tolist() == ['YELLOW', 'BLACK', 'NA']
    changed = values[:4] + ['BLACK'] * 3
    assert np.array_equal(stabilize(values, **config)[:4], stabilize(changed, **config)[:4])
    print('self-test PASS: causal state updates, fusion precedence, future invariance')


def exploratory_rules(frame: pd.DataFrame):
    """Post-hoc interpretation; never substitute these results for the frozen holdout."""
    train = frame['date'].le('2026-04-22').to_numpy()
    grid = []
    for green, red, crash in product((.4, .45, .5, .55), (.55, .6, .65, .7), (.2, .25, .3, .35, .4)):
        pred = np.where(frame['b20'].isna(), 'NA',
               np.where(frame['gap20'].gt(0) & frame['b20'].le(green), 'GREEN',
               np.where(frame['gap20'].lt(0) & (frame['b20'].ge(red) | frame['crash'].ge(crash)), 'RED', 'YELLOW')))
        grid.append((accuracy_score(frame.loc[train, 'sc'], pred[train]), green, red, crash, pred))
    maximum = max(row[0] for row in grid)
    short = [{'green_b20_max': g, 'red_b20_min': r, 'red_crash_min': c,
              'train_accuracy': a, 'reused_test_accuracy': accuracy_score(frame.loc[~train, 'sc'], p[~train]),
              'all_rows_agreement': accuracy_score(frame['sc'], p)}
             for a, g, r, c, p in grid if a == maximum]
    model = DecisionTreeClassifier(max_depth=5, min_samples_leaf=3, random_state=42).fit(frame[FEATURES], frame['lc'])
    rules = export_text(model, feature_names=FEATURES, decimals=4, show_weights=True)
    report = {'status': 'POST_HOC_EXPLORATION_NOT_NEW_HOLDOUT',
              'short_training_tied_rules': short,
              'long_full_sample_fit_accuracy': accuracy_score(frame['lc'], model.predict(frame[FEATURES])),
              'long_rules': rules,
              'warning': 'Holdout results were already viewed. Full-sample tree is descriptive overfit risk, not validated reconstruction.'}
    (OUTPUT / 'exploratory_rules.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'long_rules'}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fetch', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--explore', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / 'public_signal_snapshot.json'
    if args.fetch:
        if path.exists():
            raise FileExistsError('Snapshot exists; refusing to overwrite')
        with urlopen(Request(SOURCE, headers={'User-Agent': 'MarketLens-Research'}), timeout=30) as response:
            html = response.read(8_000_001)
        if len(html) > 8_000_000:
            raise ValueError('Unexpectedly large HTML')
        snapshot = extract_snapshot(html)
        path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding='utf-8')
    snapshot = json.loads(path.read_text(encoding='utf-8'))
    frame = load_frame(snapshot)
    print(frame.groupby('market')['date'].agg(['min', 'max', 'count']).to_string())
    print('columns', frame.columns.tolist())
    for name in ('sc', 'lc', 's', 'l', 'fc', 'f', 'base'):
        print(name, frame.groupby('market')[name].value_counts().to_dict())
    print('missing', frame.isna().sum().to_dict())
    if args.run:
        run_experiment(frame, snapshot)
    if args.explore:
        exploratory_rules(frame)


if __name__ == '__main__':
    main()
