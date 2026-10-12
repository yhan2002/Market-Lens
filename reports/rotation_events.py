"""Fixed leadership-event definitions and label-free alert calibration; no I/O."""
import numpy as np
import pandas as pd

HORIZON = 10
PERSIST = 5
COOLDOWN = 10
EVENT_COLUMNS = ['kind', 'ticker', 'onset_pos', 'confirm_pos', 'onset_date', 'confirm_date']
ALERT_COLUMNS = ['decision_date', 'decision_pos', 'score', 'nominated']


def leadership(prices):
    """Return trailing 20-session SPY-relative log return and its top three sectors."""
    if 'SPY' not in prices or len(prices.columns) < 4:
        raise ValueError('SPY and at least three sectors are required')
    if not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError('Prices need a unique increasing calendar')
    if not prices.columns.is_unique or (prices <= 0).any().any() or np.isinf(prices).any().any():
        raise ValueError('Prices need unique columns and positive finite observed values')
    sectors = sorted(c for c in prices.columns if c != 'SPY')
    relative = np.log(prices[sectors]).sub(np.log(prices.SPY), axis=0).diff(20)
    leaders = relative.rank(axis=1, method='first', ascending=False).le(3)
    leaders.loc[relative.isna().any(axis=1)] = False
    return relative, leaders


def events_from_leaders(leaders):
    """Onset follows five opposite states and starts five consecutive new states."""
    if leaders.isna().any().any():
        raise ValueError('Leadership states must be observed booleans')
    values = leaders.to_numpy(dtype=bool)
    records = []
    for i in range(PERSIST, len(leaders)-PERSIST+1):
        before, after = values[i-PERSIST:i], values[i:i+PERSIST]
        for kind, selected in [('entry', (~before).all(axis=0) & after.all(axis=0)),
                               ('exit', before.all(axis=0) & (~after).all(axis=0))]:
            for j in np.flatnonzero(selected):
                records.append((kind, leaders.columns[j], i, i+PERSIST-1,
                                leaders.index[i], leaders.index[i+PERSIST-1]))
    return pd.DataFrame(records, columns=EVENT_COLUMNS)


def make_labels(leaders, events, kind):
    """Future onset within 1..10 sessions; all labels require 14 future sessions."""
    if kind not in ('entry', 'exit'):
        raise ValueError('kind must be entry or exit')
    labels = pd.DataFrame(0., index=leaders.index, columns=leaders.columns)
    for event in events.loc[events.kind.eq(kind)].itertuples():
        j = labels.columns.get_loc(event.ticker)
        labels.iloc[max(0, int(event.onset_pos)-HORIZON):int(event.onset_pos), j] = 1.
    labels.iloc[max(0, len(labels)-(HORIZON+PERSIST-1)):] = np.nan
    eligible = ~leaders if kind == 'entry' else leaders.copy()
    return labels, eligible


def alerts_from_scores(daily, threshold):
    """One market alert per cooldown, with scalar or date-aligned annual thresholds."""
    if not daily.index.is_unique or not daily.index.is_monotonic_increasing:
        raise ValueError('Daily scores need a unique increasing calendar')
    positions = daily.pos.to_numpy(dtype=float)
    if not np.isfinite(positions).all() or (positions != positions.astype(int)).any() or (np.diff(positions) <= 0).any():
        raise ValueError('pos must contain strictly increasing integer calendar positions')
    thresholds = (threshold.reindex(daily.index).to_numpy(dtype=float)
                  if isinstance(threshold, pd.Series) else np.full(len(daily), threshold, dtype=float))
    records = []
    last = -np.inf
    for date, pos, score, nominee, cutoff in zip(daily.index, positions, daily.score, daily.nominated, thresholds):
        if (not np.isfinite(score) or pd.isna(nominee) or not str(nominee)
                or np.isnan(cutoff) or score < cutoff or pos-last < COOLDOWN):
            continue
        records.append((date, int(pos), float(score), nominee))
        last = pos
    return pd.DataFrame(records, columns=ALERT_COLUMNS)


def calibrate_threshold(daily, target_per_year=6.):
    """Choose count closest to the fixed alert budget, without accessing labels."""
    if not np.isfinite(target_per_year) or target_per_year < 0:
        raise ValueError('Alert budget must be finite and nonnegative')
    observed = daily.loc[np.isfinite(daily.score) & daily.nominated.notna(), 'score']
    if observed.empty:
        return float('inf')
    target = round(len(daily)/252*target_per_year)
    candidates = np.unique(np.r_[observed.quantile(np.linspace(0, 1, 101)), np.inf])[::-1]
    # Descending candidates retain the more conservative threshold on equal error.
    return float(min(candidates, key=lambda cutoff: abs(len(alerts_from_scores(daily, cutoff))-target)))


def match_alerts(alerts, events, kind):
    """Match each alert once to the earliest unmatched future same-kind onset group.

    Caller supplies the evaluation boundaries. Same-day sector events form one
    group; candidate_hit additionally requires the nominated sector in that group.
    nominee_horizon_hit independently checks that sector anywhere in the horizon.
    """
    if kind not in ('entry', 'exit'):
        raise ValueError('kind must be entry or exit')
    groups = {int(pos): set(group.ticker)
              for pos, group in events.loc[events.kind.eq(kind)].groupby('onset_pos', sort=True)}
    result = alerts.sort_values('decision_pos', kind='stable').reset_index(drop=True).copy()
    result['event_pos'] = pd.Series(pd.NA, index=result.index, dtype='Int64')
    result['hit'] = False
    result['candidate_hit'] = False
    result['nominee_horizon_hit'] = False
    result['lead_days'] = np.nan
    matched = set()
    for i, alert in result.iterrows():
        result.loc[i, 'nominee_horizon_hit'] = any(
            1 <= pos-alert.decision_pos <= HORIZON and alert.nominated in tickers
            for pos, tickers in groups.items())
        event_pos = next((pos for pos in groups if pos not in matched
                          and 1 <= pos-alert.decision_pos <= HORIZON), None)
        if event_pos is None:
            continue
        matched.add(event_pos)
        result.loc[i, ['event_pos', 'hit', 'candidate_hit', 'lead_days']] = [
            event_pos, True, alert.nominated in groups[event_pos], event_pos-alert.decision_pos]
    return result


def self_test():
    dates = pd.bdate_range('2020-12-01', periods=60)
    prices = pd.DataFrame(100., index=dates, columns=['D', 'C', 'SPY', 'B', 'A'])
    relative, state = leadership(prices)
    assert relative.columns.tolist() == ['A', 'B', 'C', 'D']
    assert not state.iloc[:20].any().any()
    assert state.iloc[20].tolist() == [True, True, True, False]
    changed = prices.copy()
    changed.iloc[40:, 0] *= 10
    pd.testing.assert_frame_equal(leadership(changed)[1].iloc[:40], state.iloc[:40])

    states = pd.DataFrame(False, index=dates, columns=['A', 'B'])
    states.iloc[10:20] = True
    states.iloc[30:34, 0] = True  # Four sessions alone do not establish a rotation.
    events = events_from_leaders(states)
    assert events.onset_pos.tolist() == [10, 10, 20, 20]
    assert events.confirm_pos.tolist() == [14, 14, 24, 24]
    assert events.onset_date.tolist() == [dates[10], dates[10], dates[20], dates[20]]
    assert events.confirm_date.tolist() == [dates[14], dates[14], dates[24], dates[24]]
    y, eligible = make_labels(states, events, 'entry')
    assert y.iloc[:10].eq(1).all().all() and y.iloc[10:46].eq(0).all().all()
    assert y.iloc[-14:].isna().all().all() and not eligible.iloc[10].any()
    exits, exit_eligible = make_labels(states, events, 'exit')
    assert exits.iloc[10:20].eq(1).all().all() and exit_eligible.iloc[10].all()
    last = states.iloc[:24]
    assert not events_from_leaders(last).kind.eq('exit').any()  # Fifth confirming date missing.
    tail_states = states.copy()
    tail_states.iloc[52:, 0] = True
    tail_labels, _ = make_labels(tail_states, events_from_leaders(tail_states), 'entry')
    assert tail_labels.iloc[45, 0] == 1. and tail_labels.iloc[-14:].isna().all().all()

    daily = pd.DataFrame({'pos':np.arange(60), 'score':1., 'nominated':'A'}, index=dates)
    alerts = alerts_from_scores(daily, .5)
    assert alerts.decision_pos.tolist() == [0, 10, 20, 30, 40, 50]
    thresholds = pd.Series(.5, index=dates)
    thresholds.iloc[20:31] = 2.
    assert alerts_from_scores(daily, thresholds).decision_pos.tolist() == [0, 10, 31, 41, 51]
    # Calibration tie favors no alarms; rows with no nominee cannot emit alerts.
    assert np.isinf(calibrate_threshold(daily, target_per_year=0))
    missing = daily.copy()
    missing.nominated = None
    assert alerts_from_scores(missing, .5).empty
    assert np.isinf(calibrate_threshold(missing))
    repeated = pd.DataFrame([(dates[0], 0, 1., 'A'), (dates[1], 1, 1., 'B'),
                             (dates[10], 10, 1., 'A')], columns=ALERT_COLUMNS)
    matched = match_alerts(repeated, events, 'entry')
    assert matched.hit.tolist() == [True, False, False]
    assert matched.candidate_hit.tolist() == [True, False, False]
    assert matched.lead_days.iloc[0] == 10
    assert not match_alerts(repeated.iloc[[-1]], events, 'entry').hit.any()
    wrong = repeated.iloc[:1].assign(nominated='C')
    wrong_match = match_alerts(wrong, events, 'entry')
    assert wrong_match.hit.iloc[0] and not wrong_match.candidate_hit.iloc[0]
    assert not wrong_match.nominee_horizon_hit.iloc[0]
    ordered_events = pd.DataFrame([
        ('entry', 'B', 2, 6, dates[2], dates[6]),
        ('entry', 'A', 6, 10, dates[6], dates[10])], columns=EVENT_COLUMNS)
    later_nominee = match_alerts(repeated.iloc[:1], ordered_events, 'entry')
    assert later_nominee.hit.iloc[0] and not later_nominee.candidate_hit.iloc[0]
    assert later_nominee.nominee_horizon_hit.iloc[0] and later_nominee.event_pos.iloc[0] == 2
    assert match_alerts(alerts.iloc[:0], events, 'entry').empty
    print('PASS: causal top3, fixed events and confirmation, labels, cooldown, calibration, unique event matching')


if __name__ == '__main__':
    self_test()
