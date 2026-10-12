"""Frozen NYSE core-session calendar, verified 2026-09-09 against SOURCE_URL.

Only 2026/2027 scheduled sessions are covered. Unexpected closures require a
reviewed calendar version update; this module does not discover them online.
"""
from zoneinfo import ZoneInfo

import pandas as pd

SOURCE_URL = 'https://www.nyse.com/trade/hours-calendars'
CALENDAR_VERSION = 'nyse-2026-2027-asof-20260909'
HOLIDAYS = (
    '2026-01-01 2026-01-19 2026-02-16 2026-04-03 2026-05-25 '
    '2026-06-19 2026-07-03 2026-09-07 2026-11-26 2026-12-25 '
    '2027-01-01 2027-01-18 2027-02-15 2027-03-26 2027-05-31 '
    '2027-06-18 2027-07-05 2027-09-06 2027-11-25 2027-12-24'
).split()
EARLY_CLOSES = {'2026-11-27', '2026-12-24', '2027-11-26'}


def schedule():
    """Naive session dates with timezone-aware UTC regular open/close timestamps."""
    sessions = pd.bdate_range('2026-01-01', '2027-12-31').difference(
        pd.to_datetime(HOLIDAYS)).rename('session')
    local = ZoneInfo('America/New_York')
    opens = (sessions + pd.Timedelta(hours=9, minutes=30)).tz_localize(local).tz_convert('UTC')
    closes = pd.DatetimeIndex([
        day + pd.Timedelta(hours=13 if day.strftime('%Y-%m-%d') in EARLY_CLOSES else 16)
        for day in sessions]).tz_localize(local).tz_convert('UTC')
    return pd.DataFrame({'open': opens, 'close': closes}, index=sessions)


def _utc(value):
    value = pd.Timestamp(value)
    if pd.isna(value) or value.tzinfo is None:
        raise ValueError('nowUTC must be a finite timezone-aware timestamp')
    return value.tz_convert('UTC')


def latest_completed(nowUTC):
    """Last fully closed session, rejecting times beyond the frozen calendar."""
    now, calendar = _utc(nowUTC), schedule()
    if now < calendar.close.iloc[0] or now > calendar.close.iloc[-1]:
        raise ValueError('nowUTC lies outside the supported completed-session calendar')
    return calendar.index[calendar.close.le(now)][-1]


def observation_window(session, nowUTC):
    """A fresh daily observation may be recorded from its close until next open."""
    session, now, calendar = pd.Timestamp(session), _utc(nowUTC), schedule()
    if pd.isna(session) or session.tzinfo is not None or session != session.normalize():
        raise ValueError('session must be a finite naive normalized session date')
    if session not in calendar.index:
        raise ValueError('session is outside the supported trading calendar')
    position = calendar.index.get_loc(session)
    if position == len(calendar) - 1:
        raise ValueError('Next session open is unknown; update the frozen calendar')
    return bool(calendar.close.iloc[position] <= now < calendar.open.iloc[position + 1])


def self_test():
    calendar = schedule()
    assert calendar.index.name == 'session' and calendar.index.tz is None
    assert calendar.loc['2026-11-27', 'close'] == pd.Timestamp('2026-11-27 18:00Z')
    assert calendar.loc['2026-07-01', 'close'] == pd.Timestamp('2026-07-01 20:00Z')
    assert latest_completed('2026-09-09 10:00Z') == pd.Timestamp('2026-09-08')
    assert latest_completed('2026-09-07 16:00Z') == pd.Timestamp('2026-09-04')
    assert observation_window('2026-09-04', '2026-09-08 13:29:59Z')
    assert not observation_window('2026-09-04', '2026-09-08 13:30Z')
    assert not observation_window('2026-09-08', '2026-09-08 19:59:59Z')
    assert observation_window('2026-09-08', '2026-09-08 20:00Z')
    for operation in [lambda: latest_completed('2026-01-01 00:00Z'),
                      lambda: latest_completed('2027-12-31 21:00:01Z'),
                      lambda: latest_completed('2026-09-09'),
                      lambda: observation_window('2027-12-31', '2027-12-31 21:00Z')]:
        try:
            operation()
        except ValueError:
            pass
        else:
            raise AssertionError('Unsupported calendar input accepted')
    print('PASS: NYSE holiday gap, early close, DST, fresh observation window and calendar bounds')


if __name__ == '__main__':
    self_test()
