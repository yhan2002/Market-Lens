from __future__ import annotations

import argparse
import sqlite3
from contextlib import closing
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_TICKERS = ("NQ=F", "SPY", "^VIX", "HYG", "IEF", "RSP", "IWM")
MODEL_VERSION = "crash-fixed-rank-nq-v2-shadow"
VULNERABLE_CUTOFF = 0.60
CRASH_WATCH_CUTOFF = 0.80


def causal_trailing_rank(
    values: pd.Series, *, window: int = 756, min_history: int = 60
) -> pd.Series:
    """Rank each value against prior observations only; this is not a probability."""
    if window < 1 or min_history < 1 or min_history > window:
        raise ValueError("require 1 <= min_history <= window")

    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    result = np.full(len(numeric), np.nan, dtype=float)
    history: list[float] = []
    for position, value in enumerate(numeric):
        if not np.isfinite(value):
            continue
        prior = np.asarray(history[-window:], dtype=float)
        if len(prior) >= min_history:
            below = np.count_nonzero(prior < value)
            tied = np.count_nonzero(prior == value)
            result[position] = (below + 0.5 * tied) / len(prior)
        history.append(float(value))
    return pd.Series(result, index=values.index, dtype=float)


def _returns(values: pd.Series, periods: int) -> pd.Series:
    return values.pct_change(periods, fill_method=None)


def _rank_components(
    raw: pd.DataFrame, *, window: int, min_history: int
) -> pd.DataFrame:
    return raw.apply(
        lambda column: causal_trailing_rank(
            column, window=window, min_history=min_history
        )
    )


def _risk_band(score: float) -> str:
    if not np.isfinite(score):
        return "WARMUP"
    if score >= CRASH_WATCH_CUTOFF:
        return "CRASH_WATCH_SHADOW"
    if score >= VULNERABLE_CUTOFF:
        return "VULNERABLE_SHADOW"
    return "NORMAL_SHADOW"


def build_crash_guard_scores(
    prices: pd.DataFrame, *, window: int = 756, min_history: int = 60
) -> pd.DataFrame:
    """Build always-on causal 1-day and 5-day fixed-rank shadow scores."""
    missing = sorted(set(REQUIRED_TICKERS) - set(prices.columns))
    if missing:
        raise ValueError("missing required tickers: " + ", ".join(missing))

    frame = prices.loc[:, REQUIRED_TICKERS].apply(pd.to_numeric, errors="coerce")
    frame = frame.where(frame > 0).sort_index().dropna(how="any")
    if frame.empty:
        raise ValueError("no complete positive price rows")

    vix_change_1d = _returns(frame["^VIX"], 1)
    credit = frame["HYG"] / frame["IEF"]
    rsp_relative = frame["RSP"] / frame["SPY"]
    iwm_relative = frame["IWM"] / frame["SPY"]

    raw_1d = pd.DataFrame(
        {
            "vix_jump_1d": vix_change_1d,
            "vix_acceleration_1d": vix_change_1d.diff(),
            "nq_loss_3d": -_returns(frame["NQ=F"], 3),
            "credit_loss_3d": -_returns(credit, 3),
        },
        index=frame.index,
    )
    raw_5d = pd.DataFrame(
        {
            "vix_level": frame["^VIX"],
            "vix_change_5d": _returns(frame["^VIX"], 5),
            "nq_loss_5d": -_returns(frame["NQ=F"], 5),
            "credit_loss_5d": -_returns(credit, 5),
            "rsp_weakness_5d": -_returns(rsp_relative, 5),
            "iwm_weakness_5d": -_returns(iwm_relative, 5),
        },
        index=frame.index,
    )

    ranks_1d = _rank_components(raw_1d, window=window, min_history=min_history)
    ranks_5d = _rank_components(raw_5d, window=window, min_history=min_history)
    score_1d = ranks_1d.mean(axis=1).where(ranks_1d.notna().all(axis=1))
    score_5d = ranks_5d.mean(axis=1).where(ranks_5d.notna().all(axis=1))
    score = pd.concat((score_1d, score_5d), axis=1).max(axis=1, skipna=False)

    result = pd.DataFrame(
        {"score_1d": score_1d, "score_5d": score_5d, "score": score},
        index=frame.index,
    )
    for name in raw_1d:
        result[f"{name}_raw"] = raw_1d[name]
        result[f"{name}_percentile"] = ranks_1d[name]
    for name in raw_5d:
        result[f"{name}_raw"] = raw_5d[name]
        result[f"{name}_percentile"] = ranks_5d[name]
    result["risk_band"] = result["score"].map(_risk_band)
    return result.rename_axis("date")


def _iso_date(as_of_date: str) -> str:
    try:
        return date.fromisoformat(as_of_date).isoformat()
    except (TypeError, ValueError) as error:
        raise ValueError("date must use YYYY-MM-DD format") from error


def _load_prices(database: Path, as_of_date: str) -> pd.DataFrame:
    placeholders = ",".join("?" for _ in REQUIRED_TICKERS)
    with closing(sqlite3.connect(database)) as connection:
        prices = pd.read_sql_query(
            f"""
            SELECT date, ticker, COALESCE(adj_close, close) AS price
            FROM daily_prices
            WHERE ticker IN ({placeholders}) AND date <= ?
            ORDER BY date, ticker
            """,
            connection,
            params=(*REQUIRED_TICKERS, as_of_date),
            parse_dates=["date"],
        )
    if prices.empty:
        raise ValueError(f"no market prices on or before {as_of_date}")
    matrix = prices.pivot(index="date", columns="ticker", values="price")
    missing = sorted(set(REQUIRED_TICKERS) - set(matrix.columns))
    if missing:
        raise ValueError("missing required tickers in daily_prices: " + ", ".join(missing))
    matrix = matrix.loc[:, REQUIRED_TICKERS].dropna(how="any").sort_index()
    if matrix.empty:
        raise ValueError(f"no complete market prices on or before {as_of_date}")
    return matrix


def get_crash_guard_status(database: Path, as_of_date: str) -> dict[str, object]:
    """Return the latest complete trading-day score on or before ``as_of_date``."""
    target = _iso_date(as_of_date)
    prices = _load_prices(database, target)
    scores = build_crash_guard_scores(prices)
    row = scores.iloc[-1]

    def number(key: str) -> float | None:
        value = row[key]
        return float(value) if np.isfinite(value) else None

    return {
        "date": str(scores.index[-1].date()),
        "model_version": MODEL_VERSION,
        "risk_band": str(row["risk_band"]),
        "score": number("score"),
        "score_1d": number("score_1d"),
        "score_5d": number("score_5d"),
        "score_is_probability": False,
    }


def get_crash_guard_narrative_context(
    database: Path, as_of_date: str
) -> dict[str, object]:
    """Return fact-only component detail for the daily narrative."""
    target = _iso_date(as_of_date)
    scores = build_crash_guard_scores(_load_prices(database, target))
    row = scores.iloc[-1]
    names = (
        "vix_jump_1d", "vix_acceleration_1d", "nq_loss_3d", "credit_loss_3d",
        "vix_level", "vix_change_5d", "nq_loss_5d", "credit_loss_5d",
        "rsp_weakness_5d", "iwm_weakness_5d",
    )

    def number(key: str) -> float | None:
        value = row[key]
        return float(value) if np.isfinite(value) else None

    return {
        "date": str(scores.index[-1].date()),
        "score": number("score"),
        "score_1d": number("score_1d"),
        "score_5d": number("score_5d"),
        "risk_band": str(row["risk_band"]),
        "vulnerable_cutoff": VULNERABLE_CUTOFF,
        "crash_watch_cutoff": CRASH_WATCH_CUTOFF,
        "components": {
            name: {
                "raw": number(f"{name}_raw"),
                "percentile": number(f"{name}_percentile"),
            }
            for name in names
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Always-on shadow crash guard")
    commands = parser.add_subparsers(dest="command", required=True)
    status_date = commands.add_parser("status-date", help="show score on or before a date")
    status_date.add_argument("database", type=Path)
    status_date.add_argument("date")
    args = parser.parse_args()

    status = get_crash_guard_status(args.database, args.date)
    display = lambda value: "N/A" if value is None else f"{value:.3f}"
    print(
        f"{status['date']} | {status['model_version']} | {status['risk_band']} | "
        f"score {display(status['score'])} | 1d {display(status['score_1d'])} | "
        f"5d {display(status['score_5d'])} | rank score, not probability"
    )


if __name__ == "__main__":
    main()
