from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections.abc import Mapping
from contextlib import closing
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


DAILY_HOLDINGS_SOURCE = "DAILY_HOLDINGS_ADJUSTED_CLOSE"
EVENT_HOLDINGS_SOURCE = "EVENT_HOLDINGS_ADJUSTED_CLOSE"
APPROVED_PORTFOLIO_SOURCES = {DAILY_HOLDINGS_SOURCE, EVENT_HOLDINGS_SOURCE}
CORRELATION_MIN_OBSERVATIONS = 10
PORTFOLIO_METRIC_VERSION = "portfolio-internals-nq-v1"


def _read_only(database: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    return connection


def portfolio_holdings_date_is_valid(
    metric_date: date | str,
    holdings_date: date | str,
    source: str,
) -> bool:
    try:
        current = date.fromisoformat(str(metric_date)[:10])
        observed = date.fromisoformat(str(holdings_date)[:10])
    except ValueError:
        return False
    age = (current - observed).days
    return age >= 0 and (
        source == EVENT_HOLDINGS_SOURCE
        or (source == DAILY_HOLDINGS_SOURCE and age <= 7)
    )


QUALITY_WEIGHTS = {
    "fundamental_momentum": 0.40,
    "earnings_revision": 0.25,
    "relative_strength": 0.20,
    "liquidity": 0.15,
}

PORTFOLIO_METRICS_SCHEMA = """
CREATE TABLE IF NOT EXISTS portfolio_risk_metrics (
    date TEXT PRIMARY KEY,
    holdings_date TEXT NOT NULL,
    average_correlation REAL NOT NULL,
    beta REAL,
    coverage REAL NOT NULL,
    source TEXT NOT NULL,
    benchmark_ticker TEXT,
    metric_version TEXT,
    weight_method TEXT
);
"""


def _ensure_portfolio_metrics_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(PORTFOLIO_METRICS_SCHEMA)
    columns = {row[1] for row in connection.execute("PRAGMA table_info(portfolio_risk_metrics)")}
    for name in ("benchmark_ticker", "metric_version", "weight_method"):
        if name not in columns:
            connection.execute(f"ALTER TABLE portfolio_risk_metrics ADD COLUMN {name} TEXT")


def _normalized_long_weights(
    weights: Mapping[str, float], available: pd.Index
) -> pd.Series:
    series = pd.Series(weights, dtype=float).reindex(available).dropna()
    if series.empty:
        raise ValueError("at least one weighted holding is required")
    if not np.isfinite(series).all() or (series < 0).any() or series.sum() <= 0:
        raise ValueError("weights must be finite, non-negative, and sum above zero")
    return series / series.sum()


def average_portfolio_correlation(
    prices: pd.DataFrame,
    weights: Mapping[str, float],
    *,
    window: int = 20,
    min_observations: int = CORRELATION_MIN_OBSERVATIONS,
) -> float:
    """Return the long-only weight-pair average correlation of holdings."""
    if window < 2 or min_observations < 2 or min_observations > window:
        raise ValueError("require 2 <= min_observations <= window")
    normalized = _normalized_long_weights(weights, prices.columns)
    returns = prices[normalized.index].apply(pd.to_numeric, errors="coerce")
    returns = returns.pct_change(fill_method=None).tail(window)
    if len(normalized) == 1:
        if returns.iloc[:, 0].notna().sum() < min_observations:
            raise ValueError("insufficient return observations")
        return 1.0

    correlation = returns.corr(min_periods=min_observations).to_numpy(dtype=float)
    pair_weights = np.outer(normalized.to_numpy(), normalized.to_numpy())
    upper = np.triu_indices(len(normalized), k=1)
    if not np.isfinite(correlation[upper]).all():
        raise ValueError("insufficient overlapping return observations")
    denominator = float(pair_weights[upper].sum())
    numerator = float((pair_weights[upper] * correlation[upper]).sum())
    if denominator <= 0 or not np.isfinite(numerator):
        raise ValueError("portfolio correlation is unavailable")
    return float(np.clip(numerator / denominator, -1.0, 1.0))


def portfolio_beta(
    prices: pd.DataFrame,
    weights: Mapping[str, float],
    benchmark: pd.Series,
    *,
    window: int = 60,
    min_observations: int = 40,
) -> float:
    """Estimate current weighted portfolio beta against a benchmark."""
    if window < 2 or min_observations < 2 or min_observations > window:
        raise ValueError("require 2 <= min_observations <= window")
    normalized = _normalized_long_weights(weights, prices.columns)
    holdings = prices[normalized.index].apply(pd.to_numeric, errors="coerce")
    joined = holdings.join(
        pd.to_numeric(benchmark, errors="coerce").rename("__benchmark__"),
        how="inner",
    )
    returns = joined.pct_change(fill_method=None).tail(window).dropna()
    if len(returns) < min_observations:
        raise ValueError("insufficient complete return observations")
    portfolio_return = returns[normalized.index].mul(normalized, axis=1).sum(axis=1)
    benchmark_return = returns["__benchmark__"]
    variance = float(benchmark_return.var(ddof=1))
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("benchmark variance is unavailable")
    covariance = float(portfolio_return.cov(benchmark_return))
    return covariance / variance


def portfolio_internal_snapshot(
    prices: pd.DataFrame,
    weights: Mapping[str, float],
    benchmark: pd.Series,
    *,
    as_of: date | str | pd.Timestamp | None = None,
    minimum_weight_coverage: float = 0.90,
) -> dict[str, object]:
    """Aggregate current-holdings risk without exposing position identities."""
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("prices must use a DatetimeIndex")
    if prices.index.has_duplicates or not prices.index.is_monotonic_increasing:
        raise ValueError("prices index must be unique and increasing")
    if not 0 < minimum_weight_coverage <= 1:
        raise ValueError("minimum_weight_coverage must be in (0, 1]")

    cutoff = pd.Timestamp(as_of) if as_of is not None else prices.index.max()
    history = prices.loc[prices.index <= cutoff].apply(pd.to_numeric, errors="coerce")
    market = pd.to_numeric(benchmark.loc[benchmark.index <= cutoff], errors="coerce")
    if history.empty or market.empty:
        raise ValueError("portfolio and benchmark history are required")

    supplied = pd.Series(weights, dtype=float)
    if supplied.empty or not np.isfinite(supplied).all() or (supplied < 0).any():
        raise ValueError("weights must be finite and non-negative")
    total_weight = float(supplied.sum())
    if total_weight <= 0:
        raise ValueError("weights must sum above zero")
    current = history.iloc[-1]
    eligible = [
        ticker
        for ticker in supplied.index
        if ticker in history and pd.notna(current[ticker]) and current[ticker] > 0
    ]
    coverage = float(supplied.reindex(eligible).sum() / total_weight)
    if coverage < minimum_weight_coverage:
        raise ValueError("portfolio price coverage is below the required threshold")
    normalized = supplied.reindex(eligible) / supplied.reindex(eligible).sum()
    selected = history[eligible].where(history[eligible] > 0)

    correlation = average_portfolio_correlation(selected, normalized)
    try:
        beta = portfolio_beta(selected, normalized, market)
    except ValueError:
        beta = math.nan
    hhi = float(np.square(normalized).sum())

    metrics: dict[str, object] = {
        "date": history.index[-1].date().isoformat(),
        "average_correlation_20": correlation,
        "beta_60": beta,
        "concentration_hhi": hhi,
        "effective_positions": 1.0 / hhi,
        "price_weight_coverage": coverage,
    }

    for window in (20, 50, 200):
        moving_average = selected.rolling(window, min_periods=window).mean().iloc[-1]
        valid = current.reindex(eligible).notna() & moving_average.notna()
        valid_weight = float(normalized[valid].sum())
        metrics[f"breadth_above_ma{window}"] = (
            float(
                normalized[valid]
                .div(valid_weight)
                .mul(current.reindex(eligible)[valid].gt(moving_average[valid]))
                .sum()
            )
            if valid_weight >= minimum_weight_coverage
            else math.nan
        )

    return_20 = selected.pct_change(20, fill_method=None).iloc[-1]
    valid_return = return_20.notna()
    return_weight = float(normalized[valid_return].sum())
    if return_weight >= minimum_weight_coverage:
        return_weights = normalized[valid_return] / return_weight
        cross_section = return_20[valid_return]
        portfolio_return = float(return_weights.mul(cross_section).sum())
        metrics["portfolio_return_20"] = portfolio_return
        metrics["negative_participation_20"] = float(
            return_weights.mul(cross_section.lt(0)).sum()
        )
        metrics["cross_sectional_dispersion_20"] = float(
            np.sqrt(return_weights.mul(np.square(cross_section - portfolio_return)).sum())
        )
    else:
        metrics.update(
            {
                "portfolio_return_20": math.nan,
                "negative_participation_20": math.nan,
                "cross_sectional_dispersion_20": math.nan,
            }
        )

    daily = selected.pct_change(fill_method=None).tail(60)
    complete = daily.columns[daily.notna().all(axis=0)]
    drawdown_weight = float(normalized.reindex(complete).sum())
    if len(daily) == 60 and drawdown_weight >= minimum_weight_coverage:
        drawdown_weights = normalized.reindex(complete) / drawdown_weight
        portfolio_returns = daily[complete].mul(drawdown_weights, axis=1).sum(axis=1)
        portfolio_index = (1 + portfolio_returns).cumprod()
        metrics["portfolio_drawdown_60"] = float(
            portfolio_index.iloc[-1] / portfolio_index.cummax().iloc[-1] - 1
        )
    else:
        metrics["portfolio_drawdown_60"] = math.nan
    return metrics


def calculate_portfolio_risk_from_database(
    database: Path,
    as_of: date | str,
    *,
    prices_database: Path | None = None,
    benchmark_database: Path | None = None,
    benchmark: str = "NQ=F",
    minimum_weight_coverage: float = 0.90,
    holdings_source: str = DAILY_HOLDINGS_SOURCE,
) -> dict[str, object]:
    """Calculate only aggregate risk; holding identities never leave this function."""
    cutoff = date.fromisoformat(as_of) if isinstance(as_of, str) else as_of
    if not 0 < minimum_weight_coverage <= 1:
        raise ValueError("minimum_weight_coverage must be in (0, 1]")
    if holdings_source not in APPROVED_PORTFOLIO_SOURCES:
        raise ValueError("holdings source is not approved")
    price_store = prices_database or database
    benchmark_store = benchmark_database or price_store
    with closing(_read_only(database)) as holdings_connection:
        holding_tables = {
            row[0]
            for row in holdings_connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if "daily_holdings" not in holding_tables:
            raise ValueError("daily_holdings is required")
        holding_row = holdings_connection.execute(
            "SELECT MAX(date) FROM daily_holdings WHERE date<=?",
            (cutoff.isoformat(),),
        ).fetchone()
        if not holding_row or not holding_row[0]:
            raise ValueError("no holdings are available at the requested date")
        holdings_date = str(holding_row[0])
        if not portfolio_holdings_date_is_valid(cutoff, holdings_date, holdings_source):
            raise ValueError("latest holdings are stale")
        holding_rows = holdings_connection.execute(
            """
            SELECT ticker, SUM(quantity), SUM(COALESCE(market_value_base, 0))
            FROM daily_holdings
            WHERE date=? AND ticker IS NOT NULL AND quantity>0
            GROUP BY ticker
            """,
            (holdings_date,),
        ).fetchall()

    with closing(_read_only(benchmark_store)) as benchmark_connection:
        benchmark_tables = {
            row[0]
            for row in benchmark_connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if "daily_prices" not in benchmark_tables:
            raise ValueError("benchmark daily_prices is required")
        market_row = benchmark_connection.execute(
            "SELECT MAX(date) FROM daily_prices WHERE ticker=? AND date<=?",
            (benchmark, cutoff.isoformat()),
        ).fetchone()
        if not market_row or not market_row[0]:
            raise ValueError("benchmark prices are unavailable")
        market_date = str(market_row[0])
        start = cutoff - timedelta(days=450)
        benchmark_rows = benchmark_connection.execute(
            """
            SELECT date, ticker, COALESCE(adj_close, close)
            FROM daily_prices
            WHERE ticker=? AND date BETWEEN ? AND ?
            ORDER BY date
            """,
            (benchmark, start.isoformat(), market_date),
        ).fetchall()

    with closing(_read_only(price_store)) as price_connection:
        price_tables = {
            row[0]
            for row in price_connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if "daily_prices" not in price_tables:
            raise ValueError("holding daily_prices is required")
        holdings = {
            str(ticker): (float(quantity), float(value))
            for ticker, quantity, value in holding_rows
        }
        if not holdings:
            raise ValueError("at least one long holding is required")
        tickers = sorted(holdings)
        placeholders = ",".join("?" for _ in tickers)
        holding_price_rows = price_connection.execute(
            f"""
            SELECT date, ticker, COALESCE(adj_close, close)
            FROM daily_prices
            WHERE ticker IN ({placeholders}) AND date BETWEEN ? AND ?
            ORDER BY date, ticker
            """,
            [*tickers, start.isoformat(), market_date],
        ).fetchall()

    prices = pd.DataFrame(
        [*holding_price_rows, *benchmark_rows], columns=["date", "ticker", "price"]
    )
    if prices.empty:
        raise ValueError("holding prices are unavailable")
    prices["date"] = pd.to_datetime(prices["date"])
    pivot = prices.pivot(index="date", columns="ticker", values="price").sort_index()
    weights: dict[str, float] = {}
    for ticker, (quantity, stored_value) in holdings.items():
        if stored_value != 0:
            weights[ticker] = abs(stored_value)
            continue
        latest_price = pivot[ticker].iloc[-1] if ticker in pivot else np.nan
        if ticker.endswith((".HK", ".T")) or not pd.notna(latest_price):
            raise ValueError("portfolio market value is unavailable")
        weights[ticker] = abs(quantity) * float(latest_price)
    eligible = [
        ticker
        for ticker in weights
        if ticker in pivot
        and pivot[ticker].notna().sum() >= CORRELATION_MIN_OBSERVATIONS + 1
        and pd.notna(pivot[ticker].iloc[-1])
    ]
    total_weight = sum(weights.values())
    covered_weight = sum(weights[ticker] for ticker in eligible)
    coverage = covered_weight / total_weight if total_weight > 0 else 0.0
    if coverage < minimum_weight_coverage:
        raise ValueError("portfolio price coverage is below the required threshold")
    covered_weights = {ticker: weights[ticker] for ticker in eligible}
    internal = portfolio_internal_snapshot(
        pivot[eligible],
        covered_weights,
        pivot[benchmark],
        as_of=market_date,
    )
    beta_value = float(internal["beta_60"])
    metric = {
        "date": market_date,
        "holdings_date": holdings_date,
        "average_correlation": internal["average_correlation_20"],
        "beta": beta_value if math.isfinite(beta_value) else None,
        "coverage": coverage,
        "source": holdings_source,
        "benchmark_ticker": benchmark,
        "metric_version": PORTFOLIO_METRIC_VERSION,
        "weight_method": (
            "HOLDINGS_MARKET_VALUE"
            if all(stored_value != 0 for _, stored_value in holdings.values())
            else "WEIGHT_PROXY_ADJ_CLOSE"
        ),
    }
    metric.update(
        {
            key: value
            for key, value in internal.items()
            if key not in {"date", "average_correlation_20", "beta_60", "price_weight_coverage"}
        }
    )
    return metric


def store_portfolio_risk_metric(database: Path, metric: Mapping[str, object]) -> None:
    """Persist the aggregate metric without persisting any position identity."""
    required = {
        "date",
        "holdings_date",
        "average_correlation",
        "beta",
        "coverage",
        "source",
    }
    missing = required - set(metric)
    if missing:
        raise ValueError(f"portfolio metric missing fields: {sorted(missing)}")
    metric_date = date.fromisoformat(str(metric["date"])[:10])
    holdings_date = date.fromisoformat(str(metric["holdings_date"])[:10])
    correlation = float(metric["average_correlation"])
    coverage = float(metric["coverage"])
    beta = metric["beta"]
    if not portfolio_holdings_date_is_valid(
        metric_date, holdings_date, str(metric["source"])
    ):
        raise ValueError("portfolio metric holdings_date is invalid for its source")
    if not math.isfinite(correlation) or not -1.0 <= correlation <= 1.0:
        raise ValueError("average_correlation must be finite and in [-1, 1]")
    if not math.isfinite(coverage) or not 0.90 <= coverage <= 1.0:
        raise ValueError("portfolio metric coverage must be in [0.90, 1.00]")
    if beta is not None and not math.isfinite(float(beta)):
        raise ValueError("portfolio beta must be finite or null")
    if metric["source"] not in APPROVED_PORTFOLIO_SOURCES:
        raise ValueError("portfolio metric source is not an approved exact producer")
    with closing(sqlite3.connect(database)) as connection:
        _ensure_portfolio_metrics_schema(connection)
        connection.execute(
            """
            INSERT INTO portfolio_risk_metrics(
                date, holdings_date, average_correlation, beta, coverage, source,
                benchmark_ticker, metric_version, weight_method
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
                holdings_date=excluded.holdings_date,
                average_correlation=excluded.average_correlation,
                beta=excluded.beta,
                coverage=excluded.coverage,
                source=excluded.source,
                benchmark_ticker=excluded.benchmark_ticker,
                metric_version=excluded.metric_version,
                weight_method=excluded.weight_method
            """,
            tuple(metric[name] for name in (
                "date",
                "holdings_date",
                "average_correlation",
                "beta",
                "coverage",
                "source",
            ))
            + (
                metric.get("benchmark_ticker"),
                metric.get("metric_version"),
                metric.get("weight_method"),
            ),
        )
        connection.commit()


def backfill_portfolio_risk_metrics(
    database: Path,
    start: date | str,
    *,
    end: date | str | None = None,
    prices_database: Path | None = None,
    benchmark_database: Path | None = None,
    target_database: Path | None = None,
    benchmark: str = "NQ=F",
) -> list[dict[str, object]]:
    """Recompute event-holdings aggregates for each benchmark trading day."""
    start_date = date.fromisoformat(str(start)[:10])
    end_date = date.fromisoformat(str(end)[:10]) if end is not None else None
    if end_date is not None and start_date > end_date:
        raise ValueError("backfill start date must not be after end date")
    price_store = prices_database or database
    benchmark_store = benchmark_database or price_store
    query = "SELECT date FROM daily_prices WHERE ticker=? AND date>=?"
    parameters: list[str] = [benchmark, start_date.isoformat()]
    if end_date is not None:
        query += " AND date<=?"
        parameters.append(end_date.isoformat())
    query += " ORDER BY date"
    with closing(_read_only(benchmark_store)) as connection:
        dates = [row[0] for row in connection.execute(query, parameters)]
    if not dates:
        raise ValueError("no benchmark trading dates are available for the backfill range")

    metrics: list[dict[str, object]] = []
    failures: list[tuple[str, str]] = []
    for trading_date in dates:
        try:
            metrics.append(
                calculate_portfolio_risk_from_database(
                    database,
                    trading_date,
                    prices_database=price_store,
                    benchmark_database=benchmark_store,
                    benchmark=benchmark,
                    holdings_source=EVENT_HOLDINGS_SOURCE,
                )
            )
        except (ValueError, sqlite3.DatabaseError) as error:
            failures.append((trading_date, str(error)))
    if failures:
        preview = "; ".join(f"{day}: {message}" for day, message in failures[:5])
        raise ValueError(
            f"portfolio backfill failed for {len(failures)}/{len(dates)} dates; {preview}"
        )

    for metric in metrics:
        store_portfolio_risk_metric(database, metric)
        if target_database is not None and target_database.resolve() != database.resolve():
            store_portfolio_risk_metric(target_database, metric)
    return metrics


def position_quality_score(
    *,
    fundamental_momentum: float,
    earnings_revision: float,
    relative_strength: float,
    liquidity: float,
) -> int:
    """Calculate the PDF 40/25/20/15 position-quality score (0..100)."""
    values = {
        "fundamental_momentum": fundamental_momentum,
        "earnings_revision": earnings_revision,
        "relative_strength": relative_strength,
        "liquidity": liquidity,
    }
    numeric = {name: float(value) for name, value in values.items()}
    if any(not np.isfinite(value) or not 0 <= value <= 100 for value in numeric.values()):
        raise ValueError("position-quality inputs must be finite scores from 0 to 100")
    weighted = sum(numeric[name] * weight for name, weight in QUALITY_WEIGHTS.items())
    return int(np.floor(weighted + 0.5))


def score_position_quality_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Score a manager-supplied table until licensed fundamentals are connected."""
    required = {"ticker", *QUALITY_WEIGHTS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"position-quality table missing columns: {sorted(missing)}")
    if frame["ticker"].isna().any() or frame["ticker"].astype(str).duplicated().any():
        raise ValueError("position-quality tickers must be present and unique")
    output = frame.copy()
    output["quality_score"] = output.apply(
        lambda row: position_quality_score(
            **{name: row[name] for name in QUALITY_WEIGHTS}
        ),
        axis=1,
    )
    output["reduction_priority"] = output["quality_score"].rank(
        method="first", ascending=True
    ).astype(int)
    return output.sort_values(["reduction_priority", "ticker"]).reset_index(drop=True)


def rank_positions_for_reduction(scores: Mapping[str, float]) -> list[str]:
    """List weakest positions first; this function never creates orders."""
    if not scores:
        raise ValueError("at least one position-quality score is required")
    validated = {ticker: float(score) for ticker, score in scores.items()}
    if any(not np.isfinite(score) or not 0 <= score <= 100 for score in validated.values()):
        raise ValueError("position-quality scores must be finite and within 0..100")
    return sorted(validated, key=lambda ticker: (validated[ticker], ticker))


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate portfolio risk diagnostics")
    commands = parser.add_subparsers(dest="command", required=True)
    compute = commands.add_parser("compute", help="store correlation and beta aggregates")
    compute.add_argument("database", type=Path)
    compute.add_argument("--as-of", default=date.today().isoformat())
    compute.add_argument("--prices-database", type=Path)
    backfill = commands.add_parser("backfill", help="store a range of portfolio aggregates")
    backfill.add_argument("database", type=Path)
    backfill.add_argument("--start", required=True)
    backfill.add_argument("--end")
    backfill.add_argument("--prices-database", type=Path)
    backfill.add_argument("--target-database", type=Path)
    backfill.add_argument(
        "--download-prices",
        action="store_true",
        help="backfill two years of Yahoo prices in the private holdings database first",
    )
    score = commands.add_parser("score-quality", help="score a manager-supplied CSV")
    score.add_argument("input", type=Path)
    score.add_argument("output", type=Path)
    args = parser.parse_args()

    if args.command == "compute":
        metric = calculate_portfolio_risk_from_database(
            args.database,
            args.as_of,
            prices_database=args.prices_database,
        )
        store_portfolio_risk_metric(args.database, metric)
        print(json.dumps(metric, ensure_ascii=False, indent=2))
    elif args.command == "backfill":
        price_store = args.prices_database or args.database
        if args.download_prices:
            if price_store.resolve() != args.database.resolve():
                raise ValueError(
                    "--download-prices requires prices to stay in the private holdings database"
                )
            from market_data import download_prices, register_universe

            with closing(sqlite3.connect(args.database)) as connection:
                tickers = tuple(
                    row[0]
                    for row in connection.execute(
                        """
                        WITH prior AS (
                            SELECT MAX(date) AS date FROM daily_holdings WHERE date<?
                        )
                        SELECT DISTINCT ticker FROM daily_holdings
                        WHERE ticker IS NOT NULL AND quantity>0 AND date<=?
                          AND (date>=? OR date=(SELECT date FROM prior))
                        ORDER BY ticker
                        """,
                        (
                            args.start,
                            args.end or date.today().isoformat(),
                            args.start,
                        ),
                    )
                )
            register_universe(
                args.database,
                tuple({*tickers, "NQ=F"}),
                group_name="PORTFOLIO_BACKFILL",
                source="daily_holdings",
            )
            print(
                f"portfolio price backfill: {len(tickers):,} holdings + NQ, "
                "up to 2 years, Yahoo Finance, API cost $0"
            )
            count, rows, downloaded_start = download_prices(
                args.database,
                years=2,
                group_name="PORTFOLIO_BACKFILL",
                backfill=True,
            )
            print(
                f"downloaded/upserted {rows:,} price rows for {count:,} tickers "
                f"since {downloaded_start}"
            )
        metrics = backfill_portfolio_risk_metrics(
            args.database,
            args.start,
            end=args.end,
            prices_database=price_store,
            target_database=args.target_database,
        )
        print(
            f"stored {len(metrics):,} portfolio aggregates: "
            f"{metrics[0]['date']} ~ {metrics[-1]['date']}"
        )
    else:
        scored = score_position_quality_table(pd.read_csv(args.input))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(args.output, index=False, encoding="utf-8-sig")
        print(f"wrote {len(scored)} scored positions to {args.output}")


if __name__ == "__main__":
    main()
