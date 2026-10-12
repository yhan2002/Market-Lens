"""Causal, aggregate-only portfolio internals for the SHADOW RORO engine."""

from __future__ import annotations

import math
import sqlite3
from contextlib import closing
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from portfolio_risk import average_portfolio_correlation


SHADOW_START = pd.Timestamp("2025-03-12")
WEIGHT_SOURCE_ACTION = "WEIGHT_SOURCE_ACTION"
WEIGHT_SOURCE_MARKET_VALUE = "WEIGHT_SOURCE_MARKET_VALUE_BASE"
WEIGHT_PROXY_ADJ_CLOSE = "WEIGHT_PROXY_ADJ_CLOSE"
DATA_QUALITY = WEIGHT_PROXY_ADJ_CLOSE
MINIMUM_WEIGHT_COVERAGE = 0.85
INCOMPLETE_SNAPSHOT_MARKER = "CARRIED_FORWARD_INCOMPLETE_SNAPSHOT"
SHORT_HISTORY_EXCLUSION_MARKER = "SHORT_HISTORY_SLEEVE_EXCLUDED"
CORRELATION_MIN_OBSERVATIONS = 10
BETA_MIN_OBSERVATIONS = 40

METRIC_COLUMNS = (
    "portfolio_correlation_20",
    "portfolio_correlation_trajectory_1d",
    "portfolio_correlation_trajectory_5d",
    "portfolio_correlation_trajectory_20d",
    "nq_beta_60",
    "concentration_hhi",
    "effective_positions",
    "top3_weight",
    "top5_weight",
    "breadth_above_ma20",
    "breadth_above_ma50",
    "breadth_above_ma200",
    "negative_participation_20",
    "portfolio_return_20",
    "cross_sectional_dispersion_20",
    "portfolio_drawdown_60",
    "holdings_date",
    "price_weight_coverage",
    "history_weight_coverage_20d",
    "history_weight_coverage_60d",
    "data_quality",
)


def _read_only(database: str | Path) -> sqlite3.Connection:
    path = Path(database)
    if not path.is_file():
        raise FileNotFoundError(path)
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    return connection


def _cutoff(value: str | date | pd.Timestamp) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value).normalize()
    except (TypeError, ValueError) as error:
        raise ValueError("as_of must be a calendar date") from error
    if pd.isna(timestamp):
        raise ValueError("as_of must be a calendar date")
    return timestamp


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    if table not in {"daily_holdings", "daily_prices"}:
        return set()
    exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return (
        {str(row[1]) for row in connection.execute(f'PRAGMA table_info("{table}")')}
        if exists
        else set()
    )


def _empty_frame(calendar: pd.DatetimeIndex) -> pd.DataFrame:
    frame = pd.DataFrame(index=calendar, columns=METRIC_COLUMNS)
    frame.index.name = "date"
    for column in METRIC_COLUMNS:
        if column not in {"holdings_date", "data_quality"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _load_nq(
    database: str | Path, cutoff: pd.Timestamp, history_start: pd.Timestamp
) -> pd.Series:
    with closing(_read_only(database)) as connection:
        if not {"ticker", "date", "adj_close"}.issubset(
            _columns(connection, "daily_prices")
        ):
            raise ValueError("benchmark daily_prices requires ticker, date, adj_close")
        rows = connection.execute(
            """
            SELECT substr(date, 1, 10), adj_close
            FROM daily_prices
            WHERE ticker='NQ=F' AND substr(date, 1, 10) BETWEEN ? AND ?
            ORDER BY substr(date, 1, 10)
            """,
            (history_start.date().isoformat(), cutoff.date().isoformat()),
        ).fetchall()
    values: dict[pd.Timestamp, float] = {}
    for raw_date, raw_price in rows:
        timestamp = pd.Timestamp(str(raw_date))
        price = float(raw_price) if raw_price is not None else math.nan
        if timestamp in values:
            raise ValueError(f"duplicate NQ price on {timestamp.date()}")
        values[timestamp] = price if math.isfinite(price) and price > 0 else math.nan
    return pd.Series(values, dtype=float, name="NQ=F").sort_index().dropna()


def _load_holdings(
    database: str | Path, cutoff: pd.Timestamp
) -> tuple[
    dict[pd.Timestamp, dict[str, tuple[float, float, float]]], set[pd.Timestamp]
]:
    with closing(_read_only(database)) as connection:
        required = {"date", "ticker", "quantity"}
        columns = _columns(connection, "daily_holdings")
        if not required.issubset(columns):
            raise ValueError("daily_holdings requires date, ticker, quantity")
        market_value = (
            "SUM(COALESCE(market_value_base, 0))"
            if "market_value_base" in columns
            else "0"
        )
        source_weight = (
            "SUM(COALESCE(action_weight_source, 0))"
            if "action_weight_source" in columns
            else "0"
        )
        rows = connection.execute(
            f"""
            SELECT substr(date, 1, 10), ticker, SUM(quantity),
                   {market_value}, {source_weight}
            FROM daily_holdings
            WHERE substr(date, 1, 10)<=? AND ticker IS NOT NULL AND quantity>0
            GROUP BY substr(date, 1, 10), ticker
            ORDER BY substr(date, 1, 10), ticker
            """,
            (cutoff.date().isoformat(),),
        ).fetchall()
    snapshots: dict[pd.Timestamp, dict[str, tuple[float, float, float]]] = {}
    for raw_date, raw_ticker, raw_quantity, raw_market_value, raw_weight in rows:
        quantity = float(raw_quantity)
        if math.isfinite(quantity) and quantity > 0 and str(raw_ticker).strip():
            snapshots.setdefault(pd.Timestamp(str(raw_date)), {})[
                str(raw_ticker)
            ] = (
                quantity,
                float(raw_market_value or 0),
                float(raw_weight or 0),
            )
    accepted: dict[pd.Timestamp, dict[str, tuple[float, float, float]]] = {}
    rejected: set[pd.Timestamp] = set()
    for snapshot_date, snapshot in sorted(snapshots.items()):
        previous = next(reversed(accepted.values()), None) if accepted else None
        source_total = sum(max(0, values[2]) for values in snapshot.values())
        previous_total = (
            sum(max(0, values[2]) for values in previous.values()) if previous else 0
        )
        incomplete = (
            previous is not None
            and len(previous) >= 4
            and len(snapshot) == 1
            and source_total > 0
            and source_total < 20
            and previous_total >= 40
        )
        if incomplete:
            rejected.add(snapshot_date)
        else:
            accepted[snapshot_date] = snapshot
    return accepted, rejected


def _load_holding_prices(
    database: str | Path,
    tickers: set[str],
    history_start: pd.Timestamp,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    with closing(_read_only(database)) as connection:
        if not {"ticker", "date", "adj_close"}.issubset(
            _columns(connection, "daily_prices")
        ):
            raise ValueError("holding daily_prices requires ticker, date, adj_close")
        placeholders = ",".join("?" for _ in tickers)
        rows = connection.execute(
            f"""
            SELECT substr(date, 1, 10), ticker, adj_close
            FROM daily_prices
            WHERE ticker IN ({placeholders})
              AND substr(date, 1, 10) BETWEEN ? AND ?
            ORDER BY substr(date, 1, 10), ticker
            """,
            [
                *sorted(tickers),
                history_start.date().isoformat(),
                cutoff.date().isoformat(),
            ],
        ).fetchall()
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows, columns=["date", "ticker", "price"])
    frame["date"] = pd.to_datetime(frame["date"])
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce").where(
        lambda values: values.gt(0)
    )
    if frame.duplicated(["date", "ticker"]).any():
        raise ValueError("duplicate holding price rows")
    return frame.pivot(index="date", columns="ticker", values="price").sort_index()


def _daily_portfolio_returns(
    prices: pd.DataFrame, weights: pd.Series, minimum_weight_coverage: float
) -> pd.Series:
    returns = prices.pct_change(fill_method=None)
    coverage = returns.notna().mul(weights, axis=1).sum(axis=1)
    result = returns.mul(weights, axis=1).sum(axis=1, min_count=1).div(coverage)
    return result.where(coverage >= minimum_weight_coverage)


def _weighted_cross_section(
    values: pd.Series, weights: pd.Series, minimum_weight_coverage: float
) -> tuple[pd.Series, pd.Series] | None:
    valid = values.notna() & np.isfinite(values)
    coverage = float(weights[valid].sum())
    if coverage < minimum_weight_coverage:
        return None
    normalized = weights[valid] / coverage
    return values[valid], normalized


def _snapshot_metrics(
    prices: pd.DataFrame,
    holdings: dict[str, tuple[float, float, float]],
    benchmark: pd.Series,
    minimum_weight_coverage: float,
) -> dict[str, object]:
    if not holdings or prices.empty:
        return {"price_weight_coverage": math.nan, "data_quality": None}
    selected = prices.reindex(columns=sorted(holdings))
    latest_known = selected.ffill().iloc[-1]
    quantities = pd.Series({ticker: values[0] for ticker, values in holdings.items()})
    market_values = pd.Series({ticker: values[1] for ticker, values in holdings.items()})
    source_weights = pd.Series({ticker: values[2] for ticker, values in holdings.items()})
    if source_weights.gt(0).all():
        weights = source_weights / source_weights.sum()
        data_quality = WEIGHT_SOURCE_ACTION
    elif market_values.gt(0).all():
        weights = market_values / market_values.sum()
        data_quality = WEIGHT_SOURCE_MARKET_VALUE
    elif latest_known.notna().all() and latest_known.gt(0).all():
        proxy = quantities.mul(latest_known)
        weights = proxy / proxy.sum()
        data_quality = WEIGHT_PROXY_ADJ_CLOSE
    else:
        return {"price_weight_coverage": math.nan, "data_quality": None}
    exact = selected.iloc[-1].notna() & selected.iloc[-1].gt(0)
    coverage = float(weights[exact].sum())
    metrics: dict[str, object] = {
        "price_weight_coverage": coverage,
        "data_quality": data_quality,
    }
    if coverage < minimum_weight_coverage:
        return metrics

    return_observations = (
        selected.pct_change(fill_method=None).tail(20).notna().sum()
    )
    correlation_eligible = return_observations.ge(CORRELATION_MIN_OBSERVATIONS)
    correlation_coverage = float(weights[correlation_eligible].sum())
    try:
        if correlation_eligible.sum() < 2:
            raise ValueError("portfolio correlation requires at least two holdings")
        if correlation_coverage < minimum_weight_coverage:
            raise ValueError("portfolio correlation price coverage is insufficient")
        eligible_weights = weights[correlation_eligible]
        metrics["portfolio_correlation_20"] = average_portfolio_correlation(
            selected.loc[:, correlation_eligible],
            eligible_weights,
            window=20,
            min_observations=CORRELATION_MIN_OBSERVATIONS,
        )
    except ValueError:
        metrics["portfolio_correlation_20"] = math.nan

    ordered = weights.sort_values(ascending=False)
    hhi = float(weights.pow(2).sum())
    metrics.update(
        {
            "concentration_hhi": hhi,
            "effective_positions": 1.0 / hhi,
            "top3_weight": float(ordered.head(3).sum()),
            "top5_weight": float(ordered.head(5).sum()),
        }
    )

    current = selected.iloc[-1]
    moving_average_coverages: dict[int, float] = {}
    for window in (20, 50, 200):
        moving_average = selected.rolling(window, min_periods=window).mean().iloc[-1]
        above = current.gt(moving_average).where(
            current.notna() & moving_average.notna()
        ).astype(float)
        moving_average_coverages[window] = float(weights[above.notna()].sum())
        cross_section = _weighted_cross_section(
            above,
            weights,
            minimum_weight_coverage,
        )
        metrics[f"breadth_above_ma{window}"] = (
            float(np.clip(cross_section[0].mul(cross_section[1]).sum(), 0, 1))
            if cross_section
            else math.nan
        )

    return_20 = selected.pct_change(20, fill_method=None).iloc[-1]
    return_20_coverage = float(weights[return_20.notna()].sum())
    metrics["history_weight_coverage_20d"] = min(
        correlation_coverage,
        moving_average_coverages[20],
        return_20_coverage,
    )
    cross_section = _weighted_cross_section(
        return_20, weights, minimum_weight_coverage
    )
    if cross_section:
        values, normalized = cross_section
        portfolio_return = float(values.mul(normalized).sum())
        metrics.update(
            {
                "portfolio_return_20": portfolio_return,
                "negative_participation_20": float(
                    np.clip(values.lt(0).mul(normalized).sum(), 0, 1)
                ),
                "cross_sectional_dispersion_20": float(
                    np.sqrt(normalized.mul((values - portfolio_return).pow(2)).sum())
                ),
            }
        )
    else:
        metrics.update(
            {
                "portfolio_return_20": math.nan,
                "negative_participation_20": math.nan,
                "cross_sectional_dispersion_20": math.nan,
            }
        )

    daily_returns = selected.pct_change(fill_method=None)
    daily_coverage = daily_returns.notna().mul(weights, axis=1).sum(axis=1).tail(60)
    metrics["history_weight_coverage_60d"] = (
        float(daily_coverage.min()) if len(daily_coverage) == 60 else math.nan
    )
    portfolio_returns = _daily_portfolio_returns(selected, weights, minimum_weight_coverage)
    joined = pd.concat(
        [
            portfolio_returns.rename("portfolio"),
            benchmark.pct_change(fill_method=None).rename("nq"),
        ],
        axis=1,
        join="inner",
    ).tail(60).dropna()
    if len(joined) >= BETA_MIN_OBSERVATIONS and joined["nq"].var(ddof=1) > 0:
        metrics["nq_beta_60"] = float(
            joined["portfolio"].cov(joined["nq"]) / joined["nq"].var(ddof=1)
        )
    else:
        metrics["nq_beta_60"] = math.nan

    drawdown_returns = portfolio_returns.tail(60)
    if len(drawdown_returns) == 60 and drawdown_returns.notna().all():
        wealth = (1 + drawdown_returns).cumprod()
        metrics["portfolio_drawdown_60"] = float(
            wealth.iloc[-1] / wealth.cummax().iloc[-1] - 1
        )
    else:
        metrics["portfolio_drawdown_60"] = math.nan
    history_coverages = (
        metrics["history_weight_coverage_20d"],
        metrics["history_weight_coverage_60d"],
    )
    if any(
        math.isfinite(float(value))
        and minimum_weight_coverage <= float(value) < 1 - 1e-12
        for value in history_coverages
    ):
        metrics["data_quality"] = (
            f"{metrics['data_quality']}|{SHORT_HISTORY_EXCLUSION_MARKER}"
        )
    return metrics


def load_portfolio_internal_series(
    holdings_database: str | Path,
    benchmark_database: str | Path,
    *,
    as_of: str | date | pd.Timestamp,
    minimum_weight_coverage: float = MINIMUM_WEIGHT_COVERAGE,
    include_warmup: bool = False,
) -> pd.DataFrame:
    """Return aggregate portfolio internals on the causal NQ decision calendar.

    Rows from the first recorded holdings snapshot are available as calculation
    warm-up; the public SHADOW start is controlled by the axis/series contract.
    """
    if not 0 < minimum_weight_coverage <= 1:
        raise ValueError("minimum_weight_coverage must be in (0, 1]")
    cutoff = _cutoff(as_of)
    history_start = min(SHADOW_START, cutoff) - pd.Timedelta(days=450)
    nq = _load_nq(benchmark_database, cutoff, history_start)
    calendar = pd.DatetimeIndex(nq.index, name="date")
    if calendar.empty:
        return _empty_frame(calendar)

    snapshots, rejected_snapshots = _load_holdings(holdings_database, cutoff)
    prices = _load_holding_prices(
        holdings_database,
        {ticker for snapshot in snapshots.values() for ticker in snapshot},
        history_start,
        cutoff,
    ).reindex(calendar)
    snapshot_dates = pd.DatetimeIndex(sorted(snapshots))
    rows: list[dict[str, object]] = []
    for decision_date in calendar:
        row: dict[str, object] = {
            column: math.nan
            for column in METRIC_COLUMNS
            if column not in {"holdings_date", "data_quality"}
        }
        row.update({"holdings_date": None, "data_quality": None})
        position = snapshot_dates.searchsorted(decision_date, side="right") - 1
        if position >= 0:
            holdings_date = snapshot_dates[position]
            row["holdings_date"] = holdings_date.date().isoformat()
            row.update(
                _snapshot_metrics(
                    prices.loc[:decision_date],
                    snapshots[holdings_date],
                    nq.loc[:decision_date],
                    minimum_weight_coverage,
                )
            )
            if row["data_quality"] and any(
                holdings_date < rejected_date <= decision_date
                for rejected_date in rejected_snapshots
            ):
                row["data_quality"] = (
                    f"{row['data_quality']}|{INCOMPLETE_SNAPSHOT_MARKER}"
                )
        rows.append(row)

    frame = pd.DataFrame(rows, index=calendar).reindex(columns=METRIC_COLUMNS)
    correlation = frame["portfolio_correlation_20"]
    for lag in (1, 5, 20):
        frame[f"portfolio_correlation_trajectory_{lag}d"] = correlation.diff(lag)
    if not include_warmup:
        frame = frame.loc[frame.index >= SHADOW_START]
    frame.index.name = "date"
    frame.attrs.update(
        {
            "as_of": cutoff.date().isoformat(),
            "benchmark": "NQ=F",
            "start": SHADOW_START.date().isoformat(),
            "includes_warmup": include_warmup,
            "database_mode": "mode=ro; query_only=ON",
            "weight_method": "ACTION_WEIGHT > MARKET_VALUE_BASE > ADJ_CLOSE_PROXY",
            "identity_exposure": "aggregate-only",
        }
    )
    return frame
