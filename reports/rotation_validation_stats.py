"""Conditional diagnostics for a fixed historical rule, not independent confirmation."""
import numpy as np
import pandas as pd


def _integers(values, name, increasing=False):
    values = np.asarray(values, dtype=float)
    if (values.ndim != 1 or not np.isfinite(values).all()
            or (values != np.floor(values)).any() or (np.abs(values) >= 2**53).any()):
        raise ValueError(f'{name} must be a finite one-dimensional integer array')
    values = values.astype(np.int64)
    if increasing and (np.diff(values) <= 0).any():
        raise ValueError(f'{name} must be unique and increasing')
    return values


def _positive_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)


def random_schedules(positions, years, counts_by_year, draws=2000, seed=20260909, cooldown=10):
    """Uniform feasible schedules conditional on annual counts; outcomes never enter.

    Compressed combinations enforce within-year spacing; rejection enforces spacing
    across year boundaries. Global calendar positions must be consecutive.
    """
    positions = _integers(positions, 'positions', increasing=True)
    years = _integers(years, 'years')
    draws, cooldown = _positive_int(draws, 'draws'), _positive_int(cooldown, 'cooldown')
    if len(positions) != len(years) or (np.diff(years) < 0).any():
        raise ValueError('years must align with positions and be nondecreasing')
    if (np.diff(positions) != 1).any():
        raise ValueError('random schedules require consecutive global calendar positions')
    unique_years = np.unique(years)
    if any(year not in unique_years for year in counts_by_year):
        raise ValueError('counts_by_year contains a year outside the calendar')
    groups, last = [], -np.inf
    for year in unique_years:
        dates, count = positions[years == year], counts_by_year.get(int(year), 0)
        if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count < 0:
            raise ValueError('annual alert counts must be nonnegative integers')
        count = int(count)
        if not count:
            continue
        compressed = len(dates) - (cooldown - 1) * (count - 1)
        first = max(int(dates[0]), last + cooldown)
        last = first + (count - 1) * cooldown
        if compressed < count or last > dates[-1]:
            raise ValueError('annual alert counts are infeasible under the global cooldown')
        groups.append((dates, count, compressed))
    result = np.empty((draws, sum(group[1] for group in groups)), dtype=np.int64)
    if not groups:
        return result
    rng, accepted = np.random.default_rng(seed), 0
    for _ in range(max(10000, draws * 100)):
        schedule = np.concatenate([dates[np.sort(rng.choice(n, k, replace=False))
                                         + (cooldown - 1) * np.arange(k)]
                                   for dates, k, n in groups])
        if (np.diff(schedule) < cooldown).any():
            continue
        result[accepted] = schedule
        accepted += 1
        if accepted == draws:
            return result
    raise RuntimeError('Global cooldown rejection limit reached; no partial schedules returned')


def block_intervals(alert_positions, calendar_positions, outcomes, draws=2000,
                    block_length=63, seed=20260909):
    """Resample nonoverlapping calendar blocks, including blocks without alerts.

    Paired outcomes use identical block draws. Percentile intervals describe this
    fixed historical rule; they do not undo its selection on the same history.
    """
    alerts = _integers(alert_positions, 'alert_positions', increasing=True)
    calendar = _integers(calendar_positions, 'calendar_positions', increasing=True)
    draws, block_length = _positive_int(draws, 'draws'), _positive_int(block_length, 'block_length')
    if not len(calendar) or not np.isin(alerts, calendar).all():
        raise ValueError('Calendar must be nonempty and contain every alert position')
    columns = ['metric', 'estimate', 'ci_low', 'ci_high', 'blocks', 'draws', 'valid_draws', 'block_length']
    values = {name: np.asarray(value, dtype=float) for name, value in outcomes.items()}
    if any(value.shape != (len(alerts),) or not np.isfinite(value).all() for value in values.values()):
        raise ValueError('Every outcome must be a finite vector aligned with alert_positions')
    nblocks = int((calendar[-1] - calendar[0]) // block_length + 1)
    block = (alerts - calendar[0]) // block_length
    counts = np.bincount(block, minlength=nblocks)
    selected = np.random.default_rng(seed).integers(nblocks, size=(draws, nblocks))
    denominator = counts[selected].sum(axis=1)
    valid, records = denominator > 0, []
    for name, value in values.items():
        sums = np.bincount(block, weights=value, minlength=nblocks)
        ratios = sums[selected].sum(axis=1)[valid] / denominator[valid]
        low, high = np.quantile(ratios, [.025, .975]) if len(ratios) else (np.nan, np.nan)
        records.append([name, value.mean() if len(value) else np.nan, low, high,
                        nblocks, draws, int(valid.sum()), block_length])
    return pd.DataFrame(records, columns=columns)


def self_test():
    positions, years = np.arange(100, 160), np.repeat([2023, 2024], 30)
    args = (positions, years, {2023: 2, 2024: 2})
    schedules = random_schedules(*args, draws=40)
    assert schedules.shape == (40, 4) and (np.diff(schedules, axis=1) >= 10).all()
    assert (schedules < 130).sum(axis=1).tolist() == [2] * 40
    assert np.array_equal(schedules, random_schedules(*args, draws=40))
    assert random_schedules(positions, years, {2023: 0}, draws=2).shape == (2, 0)
    assert random_schedules([10, 11], [2023, 2024], {2023: 1}, draws=2).tolist() == [[10], [10]]
    for p, y, counts in [([0, 2], [2023, 2023], {2023: 1}),
                          ([0, 1], [2023, 2024], {2023: 1, 2024: 1}),
                          ([0], [2023], {2023: -1})]:
        try:
            random_schedules(p, y, counts, draws=2)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid schedule inputs accepted')
    ci = block_intervals([100, 101, 140], positions, {'delta': [.25] * 3}, draws=40, block_length=10)
    assert ci.blocks.iloc[0] == 6 and ci.estimate.iloc[0] == ci.ci_low.iloc[0] == ci.ci_high.iloc[0] == .25
    pd.testing.assert_frame_equal(ci, block_intervals([100, 101, 140], positions,
                                                     {'delta': [.25] * 3}, draws=40, block_length=10))
    empty = block_intervals([], positions, {'nominee': []}, draws=2, block_length=10)
    assert empty.valid_draws.iloc[0] == 0 and empty[['estimate', 'ci_low', 'ci_high']].isna().all().all()
    print('PASS: conditional schedules, boundary cooldown, deterministic block intervals and empty blocks')


if __name__ == '__main__':
    self_test()
