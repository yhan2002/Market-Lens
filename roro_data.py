"""Read-only point-in-time input panel for the SHADOW RORO engine."""

from __future__ import annotations

import math
import sqlite3
from contextlib import closing
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from portfolio_risk import portfolio_holdings_date_is_valid
from roro_engine import INPUT_COLUMNS
from roro_market_breadth import (
    CONTRACT_VERSION as BREADTH_OBSERVATION_CONTRACT,
    SOURCE as BREADTH_OBSERVATION_SOURCE,
    load_ndth_series,
)
from roro_price_observations import (
    CONTRACT_VERSION as PRICE_OBSERVATION_CONTRACT,
    SOURCE as PRICE_OBSERVATION_SOURCE,
    load_observation_series,
)


HY_MAX_STALE_CALENDAR_DAYS = 5
PRICE_TICKERS = {
    "nq": "NQ=F",
    "move": "^MOVE",
    "vix": "^VIX",
    "vix3m": "^VIX3M",
}


def _cutoff(value: str | date) -> str:
    try:
        parsed = date.fromisoformat(value) if isinstance(value, str) else value
    except ValueError as error:
        raise ValueError("as_of must be YYYY-MM-DD") from error
    if not isinstance(parsed, date) or parsed.isoformat() != str(value):
        raise ValueError("as_of must be YYYY-MM-DD")
    return parsed.isoformat()


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    if table not in {
        "daily_prices",
        "macro_observations",
        "portfolio_risk_metrics",
        "reviewed_move_closes",
    }:
        return set()
    exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    if not exists:
        return set()
    return {str(row[1]) for row in connection.execute(f'PRAGMA table_info("{table}")')}


def _price_series(
    connection: sqlite3.Connection, ticker: str, as_of: str
) -> pd.Series:
    columns = _columns(connection, "daily_prices")
    if not {"ticker", "date"}.issubset(columns) or not ({"adj_close", "close"} & columns):
        return pd.Series(dtype=float)
    adjusted = '"adj_close"' if "adj_close" in columns else "NULL"
    close = '"close"' if "close" in columns else "NULL"
    rows = connection.execute(
        f"""
        SELECT substr(date, 1, 10), {adjusted}, {close}
        FROM daily_prices
        WHERE ticker=? AND substr(date, 1, 10)<=?
        ORDER BY substr(date, 1, 10)
        """,
        (ticker, as_of),
    ).fetchall()
    values: dict[pd.Timestamp, float] = {}
    for raw_date, raw_adjusted, raw_close in rows:
        timestamp = pd.Timestamp(str(raw_date))
        if timestamp in values:
            raise ValueError(f"duplicate daily price for {ticker} on {timestamp.date()}")
        value = next(
            (
                number
                for raw in (raw_adjusted, raw_close)
                if raw is not None
                and math.isfinite(number := float(raw))
                and number > 0
            ),
            np.nan,
        )
        values[timestamp] = value
    return pd.Series(values, dtype=float).sort_index()


def _reviewed_move_series(connection: sqlite3.Connection, as_of: str) -> pd.Series:
    required = {"date", "close", "source", "imported_at"}
    if not required.issubset(_columns(connection, "reviewed_move_closes")):
        return pd.Series(dtype=float)
    rows = connection.execute(
        """
        SELECT substr(date, 1, 10), close
        FROM reviewed_move_closes
        WHERE substr(date, 1, 10)<=?
        ORDER BY substr(date, 1, 10)
        """,
        (as_of,),
    ).fetchall()
    values: dict[pd.Timestamp, float] = {}
    for raw_date, raw_close in rows:
        timestamp = pd.Timestamp(str(raw_date))
        close = float(raw_close)
        if timestamp in values or not math.isfinite(close) or close <= 0:
            raise ValueError(f"invalid reviewed MOVE close on {timestamp.date()}")
        values[timestamp] = close
    return pd.Series(values, dtype=float).sort_index()


def _hy_oas(
    connection: sqlite3.Connection, calendar: pd.DatetimeIndex, as_of: str
) -> pd.Series:
    required = {"date", "series", "value", "available_at", "source"}
    if not required.issubset(_columns(connection, "macro_observations")):
        return pd.Series(np.nan, index=calendar, dtype=float)
    rows = connection.execute(
        """
        SELECT substr(date, 1, 10), value, substr(available_at, 1, 10), source
        FROM macro_observations
        WHERE upper(series)='HY_OAS'
          AND substr(date, 1, 10)<=?
          AND substr(available_at, 1, 10)<=?
        ORDER BY substr(available_at, 1, 10), substr(date, 1, 10)
        """,
        (as_of, as_of),
    ).fetchall()
    events: list[tuple[int, pd.Timestamp, pd.Timestamp, float]] = []
    for raw_observation, raw_value, raw_available, raw_source in rows:
        try:
            observation = pd.Timestamp(str(raw_observation))
            available = pd.Timestamp(str(raw_available))
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        source = str(raw_source).strip()
        if (
            observation > available
            or not source
            or "PROXY" in source.upper()
            or not math.isfinite(value)
            or value <= 0
        ):
            continue
        effective_position = int(calendar.searchsorted(available, side="right"))
        if effective_position < len(calendar):
            events.append((effective_position, observation, available, value))

    result = pd.Series(np.nan, index=calendar, dtype=float)
    latest_vintage: dict[pd.Timestamp, tuple[pd.Timestamp, float]] = {}
    cursor = 0
    events.sort(key=lambda event: (event[0], event[2], event[1]))
    for position, decision_date in enumerate(calendar):
        while cursor < len(events) and events[cursor][0] <= position:
            _, observation, available, value = events[cursor]
            prior = latest_vintage.get(observation)
            if prior is None or available >= prior[0]:
                latest_vintage[observation] = (available, value)
            cursor += 1
        eligible = [observed for observed in latest_vintage if observed <= decision_date]
        if not eligible:
            continue
        observation = max(eligible)
        if (decision_date - observation).days <= HY_MAX_STALE_CALENDAR_DAYS:
            result.iloc[position] = latest_vintage[observation][1]
    return result


def _conditional_portfolio_correlation(
    connection: sqlite3.Connection, calendar: pd.DatetimeIndex, as_of: str
) -> pd.Series:
    required = {
        "date",
        "holdings_date",
        "average_correlation",
        "coverage",
        "source",
    }
    if not required.issubset(_columns(connection, "portfolio_risk_metrics")):
        return pd.Series(np.nan, index=calendar, dtype=float)
    rows = connection.execute(
        """
        SELECT substr(date, 1, 10), substr(holdings_date, 1, 10),
               average_correlation, coverage, source
        FROM portfolio_risk_metrics
        WHERE substr(date, 1, 10)<=?
        ORDER BY substr(date, 1, 10)
        """,
        (as_of,),
    ).fetchall()
    values: dict[pd.Timestamp, float] = {}
    for raw_date, raw_holdings_date, raw_correlation, raw_coverage, raw_source in rows:
        try:
            metric_date = pd.Timestamp(str(raw_date))
            correlation = float(raw_correlation)
            coverage = float(raw_coverage)
        except (TypeError, ValueError):
            continue
        if (
            metric_date not in calendar
            or not math.isfinite(correlation)
            or not -1 <= correlation <= 1
            or not math.isfinite(coverage)
            or not 0.90 <= coverage <= 1
            or not portfolio_holdings_date_is_valid(
                str(raw_date), str(raw_holdings_date), str(raw_source)
            )
        ):
            continue
        if metric_date in values:
            raise ValueError(f"duplicate portfolio metric on {metric_date.date()}")
        values[metric_date] = correlation
    return pd.Series(values, dtype=float).reindex(calendar)


def load_roro_panel(
    database: str | Path,
    *,
    as_of: str | date,
    allow_conditional_portfolio: bool = False,
    price_observation_database: str | Path | None = None,
    breadth_observation_database: str | Path | None = None,
) -> pd.DataFrame:
    """Load all NQ sessions through ``as_of`` without mutating or filling the DB."""
    cutoff = _cutoff(as_of)
    path = Path(database)
    if not path.is_file():
        raise FileNotFoundError(path)
    uri = path.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        connection.execute("PRAGMA query_only=ON")
        prices = {
            column: _price_series(connection, ticker, cutoff)
            for column, ticker in PRICE_TICKERS.items()
        }
        reviewed_move = _reviewed_move_series(connection, cutoff)
        prices["move"] = reviewed_move.combine_first(prices["move"])
        calendar = pd.DatetimeIndex(prices["nq"].index, name="date")
        panel = pd.DataFrame(index=calendar)
        for column in PRICE_TICKERS:
            panel[column] = prices[column].reindex(calendar)
        panel["hy_oas"] = _hy_oas(connection, calendar, cutoff)
        panel["breadth_200dma"] = np.nan
        panel["portfolio_correlation_20"] = (
            _conditional_portfolio_correlation(connection, calendar, cutoff)
            if allow_conditional_portfolio
            else np.nan
        )

    if price_observation_database is None:
        panel["usdjpy"] = np.nan
        panel["gold"] = np.nan
    else:
        panel["usdjpy"] = load_observation_series(
            price_observation_database,
            ticker="JPY=X",
            nq_sessions=calendar,
        )
        panel["gold"] = load_observation_series(
            price_observation_database,
            ticker="GLD",
            nq_sessions=calendar,
        )

    if breadth_observation_database is not None:
        panel["breadth_200dma"] = load_ndth_series(breadth_observation_database, calendar)

    panel = panel.loc[:, list(INPUT_COLUMNS)]
    lineage = {
        column: {
            "status": "CONDITIONAL_PRICE_NO_VINTAGE",
            "source": f"daily_prices/{ticker}",
            "rule": "exact NQ session date; no forward fill",
        }
        for column, ticker in PRICE_TICKERS.items()
    }
    if not reviewed_move.empty:
        lineage["move"] = {
            "status": "CONDITIONAL_PRICE_NO_VINTAGE_WITH_REVIEWED_OVERRIDE",
            "source": "reviewed_move_closes/INVESTING_REVIEWED_CSV + daily_prices/^MOVE",
            "rule": "reviewed close takes precedence on the exact date; Yahoo remains the fallback; no forward fill",
        }
    lineage["hy_oas"] = {
        "status": "PIT_CAUSAL",
        "source": "macro_observations/HY_OAS",
        "rule": (
            "latest vintage available before the NQ session; usable from the next "
            f"NQ session; stale after {HY_MAX_STALE_CALENDAR_DAYS} calendar days"
        ),
    }
    session_lineage = {}
    for column, ticker in (("usdjpy", "JPY=X"), ("gold", "GLD")):
        available = int(panel[column].notna().sum())
        session_lineage[column] = (
            {
                "status": "CONDITIONAL_PRICE_SESSION_CAUSAL",
                "source": (
                    f"roro_price_observations/{PRICE_OBSERVATION_SOURCE}/{ticker}/close"
                ),
                "contract_version": PRICE_OBSERVATION_CONTRACT,
                "rule": (
                    "first NQ session strictly after max(observation session date, "
                    "available_at date); no fill"
                ),
            }
            if price_observation_database is not None and available
            else {
                "status": "DATA_BLOCKED",
                "source": (
                    "daily_prices/JPY=X" if column == "usdjpy" else None
                ),
                "reason": (
                    "FX session dates are not normalized"
                    if price_observation_database is None and column == "usdjpy"
                    else "no approved Gold proxy is stored"
                    if price_observation_database is None
                    else "no usable session observations are stored"
                ),
            }
        )
    lineage.update(
        {
            **session_lineage,
            "breadth_200dma": (
                {
                    "status": "CONDITIONAL_EXTERNAL_BREADTH",
                    "source": f"roro_market_breadth_observations/{BREADTH_OBSERVATION_SOURCE}/NDTH",
                    "contract_version": BREADTH_OBSERVATION_CONTRACT,
                    "rule": "external NDTH close usable from the next NQ session; no fill; not constituent-level PIT",
                }
                if breadth_observation_database is not None and panel["breadth_200dma"].notna().any()
                else {
                    "status": "DATA_BLOCKED",
                    "source": "breadth_metrics",
                    "reason": "historical point-in-time membership or approved NDTH observations are unavailable",
                }
            ),
            "portfolio_correlation_20": {
                "status": (
                    "CONDITIONAL_SHADOW" if allow_conditional_portfolio else "DATA_BLOCKED"
                ),
                "source": "portfolio_risk_metrics",
                "reason": (
                    "explicit SHADOW opt-in; assumes each metric was known after its date close"
                    if allow_conditional_portfolio
                    else "available_at/vintage is not stored; opt in only for conditional SHADOW"
                ),
            },
        }
    )
    panel.attrs.update(
        {
            "as_of": cutoff,
            "calendar": "NQ=F",
            "database_mode": "mode=ro; query_only=ON",
            "price_observation_database_mode": (
                "mode=ro; query_only=ON"
                if price_observation_database is not None
                else "not configured"
            ),
            "breadth_observation_database_mode": (
                "mode=ro; query_only=ON"
                if breadth_observation_database is not None
                else "not configured"
            ),
            "fill_policy": "exact-date only; no forward/back fill",
            "lineage": lineage,
            "quality": {
                column: {
                    "non_null": int(panel[column].notna().sum()),
                    "rows": len(panel),
                    "status": lineage[column]["status"],
                }
                for column in INPUT_COLUMNS
            },
        }
    )
    return panel
