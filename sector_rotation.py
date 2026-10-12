from __future__ import annotations

import argparse
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

SECTOR_GROUP = "SP500_SECTOR_ETF"
SECTOR_ETFS = {
    "XLB": "Materials",
    "XLC": "Communication Services",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLI": "Industrials",
    "XLK": "Information Technology",
    "XLP": "Consumer Staples",
    "XLRE": "Real Estate",
    "XLU": "Utilities",
    "XLV": "Health Care",
    "XLY": "Consumer Discretionary",
}
DEFAULT_REGIME_MODEL = "hmm3-sticky-nq-core-v1-k25-m252-w1260-r20-s42-n3"


def register_sector_universe(database: Path) -> int:
    from market_data import register_universe

    return register_universe(
        database,
        ("SPY", *SECTOR_ETFS),
        group_name=SECTOR_GROUP,
        source="built-in S&P 500 sector ETF proxies",
    )


def load_price_matrix(database: Path, benchmark: str = "SPY") -> pd.DataFrame:
    tickers = [benchmark, *SECTOR_ETFS]
    placeholders = ",".join("?" for _ in tickers)
    with closing(sqlite3.connect(database)) as connection:
        prices = pd.read_sql_query(
            f"""
            SELECT ticker, date, COALESCE(adj_close, close) AS price
            FROM daily_prices
            WHERE ticker IN ({placeholders})
            ORDER BY date, ticker
            """,
            connection,
            params=tickers,
            parse_dates=["date"],
        )
    if prices.empty:
        raise ValueError("sector prices are missing; register and download the sector group first")
    matrix = prices.pivot(index="date", columns="ticker", values="price").sort_index()
    missing = [ticker for ticker in tickers if ticker not in matrix]
    if missing:
        raise ValueError(f"missing price history: {', '.join(missing)}")
    return matrix


def load_regime_probabilities(database: Path, model_version: str) -> pd.DataFrame:
    with closing(sqlite3.connect(database)) as connection:
        frame = pd.read_sql_query(
            """
            SELECT date, p_bull, p_sideways, p_bear
            FROM regime_model_probabilities
            WHERE model_version=?
            ORDER BY date
            """,
            connection,
            params=(model_version,),
            parse_dates=["date"],
        )
    if frame.empty:
        raise ValueError(f"regime model not found: {model_version}")
    return frame


def classify_quadrants(rs_ratio: pd.Series, rs_momentum: pd.Series) -> pd.Series:
    conditions = [
        (rs_ratio >= 100) & (rs_momentum >= 100),
        (rs_ratio >= 100) & (rs_momentum < 100),
        (rs_ratio < 100) & (rs_momentum < 100),
        (rs_ratio < 100) & (rs_momentum >= 100),
    ]
    return pd.Series(
        np.select(conditions, ["LEADING", "WEAKENING", "LAGGING", "IMPROVING"], default="WARMUP"),
        index=rs_ratio.index,
    )


def _rolling_zscore(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    mean = frame.rolling(window, min_periods=window).mean()
    std = frame.rolling(window, min_periods=window).std(ddof=0).replace(0, np.nan)
    return (frame - mean) / std


def build_rrg_signals(
    prices: pd.DataFrame,
    *,
    benchmark: str = "SPY",
    ratio_window: int = 63,
    momentum_lag: int = 10,
) -> pd.DataFrame:
    """Build a causal, transparent RRG-style proxy; not the proprietary JdK formula."""
    if ratio_window < 20 or momentum_lag < 1:
        raise ValueError("ratio_window must be >=20 and momentum_lag must be >=1")
    if benchmark not in prices:
        raise ValueError(f"benchmark column is missing: {benchmark}")
    sectors = [ticker for ticker in SECTOR_ETFS if ticker in prices]
    if not sectors:
        raise ValueError("no sector ETF columns found")

    clean = prices.sort_index().apply(pd.to_numeric, errors="coerce")
    relative_log_price = np.log(clean[sectors].div(clean[benchmark], axis=0))
    rs_ratio = 100 + _rolling_zscore(relative_log_price, ratio_window)
    ratio_change = rs_ratio.diff(momentum_lag)
    rs_momentum = 100 + _rolling_zscore(ratio_change, ratio_window)
    ratio_rank = rs_ratio.rank(axis=1, pct=True)
    momentum_rank = rs_momentum.rank(axis=1, pct=True)
    score = (ratio_rank + momentum_rank) / 2

    records: list[pd.DataFrame] = []
    for ticker in sectors:
        part = pd.DataFrame(
            {
                "date": clean.index,
                "ticker": ticker,
                "sector": SECTOR_ETFS[ticker],
                "rs_ratio": rs_ratio[ticker].to_numpy(),
                "rs_momentum": rs_momentum[ticker].to_numpy(),
                "shadow_score": score[ticker].to_numpy(),
            }
        )
        part["quadrant"] = classify_quadrants(part["rs_ratio"], part["rs_momentum"])
        records.append(part)
    return pd.concat(records, ignore_index=True).dropna(subset=["shadow_score"])


def _regime_labels(regime: pd.DataFrame) -> pd.DataFrame:
    probability_columns = ["p_bull", "p_sideways", "p_bear"]
    output = regime[["date", *probability_columns]].copy().sort_values("date")
    output["regime"] = (
        output[probability_columns]
        .idxmax(axis=1)
        .str.removeprefix("p_")
        .str.upper()
    )
    return output[["date", "regime"]]


def _mean_ci(values: pd.Series) -> tuple[float | None, float | None]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < 2:
        return None, None
    margin = 1.96 * float(clean.std(ddof=1)) / np.sqrt(len(clean))
    mean = float(clean.mean())
    return mean - margin, mean + margin


def evaluate_shadow(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    regime: pd.DataFrame,
    *,
    benchmark: str = "SPY",
    horizon: int = 20,
    top_k: int = 3,
) -> tuple[dict[str, object], pd.DataFrame]:
    if horizon < 1 or top_k < 1:
        raise ValueError("horizon and top_k must be positive")
    future_returns = prices.shift(-horizon).div(prices).sub(1)
    signal_dates = (
        signals.groupby(signals["date"].dt.to_period("M"))["date"].max().sort_values()
    )
    rows: list[dict[str, object]] = []
    for current_date in signal_dates:
        current = signals.loc[signals["date"] == current_date].dropna(subset=["shadow_score"])
        if len(current) < top_k or current_date not in future_returns.index:
            continue
        available = [ticker for ticker in current["ticker"] if pd.notna(future_returns.at[current_date, ticker])]
        if len(available) < top_k or pd.isna(future_returns.at[current_date, benchmark]):
            continue
        ranked = current[current["ticker"].isin(available)].nlargest(top_k, "shadow_score")
        sector_forward = future_returns.loc[current_date, available]
        ranked_all = current.set_index("ticker").loc[available, "shadow_score"]
        rank_ic = float(ranked_all.rank().corr(sector_forward.rank()))
        rows.append(
            {
                "date": current_date,
                "selected": ",".join(ranked["ticker"]),
                "shadow_return": float(future_returns.loc[current_date, ranked["ticker"]].mean()),
                "equal_sector_return": float(sector_forward.mean()),
                "benchmark_return": float(future_returns.at[current_date, benchmark]),
                "rank_ic": rank_ic,
                "sector_count": len(available),
            }
        )
    observations = pd.DataFrame(rows)
    if observations.empty:
        raise ValueError("no evaluable monthly observations; check price and signal coverage")
    observations = pd.merge_asof(
        observations.sort_values("date"),
        _regime_labels(regime),
        on="date",
        direction="backward",
    )
    observations["excess_vs_equal"] = (
        observations["shadow_return"] - observations["equal_sector_return"]
    )
    observations["excess_vs_benchmark"] = (
        observations["shadow_return"] - observations["benchmark_return"]
    )
    equal_ci = _mean_ci(observations["excess_vs_equal"])
    verdict = "INSUFFICIENT"
    if len(observations) >= 36:
        verdict = "PASS_SHADOW" if equal_ci[0] is not None and equal_ci[0] > 0 else "INCONCLUSIVE"
        if equal_ci[1] is not None and equal_ci[1] <= 0:
            verdict = "FAIL_SHADOW"
    summary: dict[str, object] = {
        "verdict": verdict,
        "method": "transparent RRG-style proxy; not proprietary JdK RS-Ratio/RS-Momentum",
        "rebalance": "monthly, last available signal date",
        "horizon_trading_days": horizon,
        "top_k": top_k,
        "observations": len(observations),
        "start_date": observations["date"].min().date().isoformat(),
        "end_date": observations["date"].max().date().isoformat(),
        "mean_shadow_return": float(observations["shadow_return"].mean()),
        "mean_equal_sector_return": float(observations["equal_sector_return"].mean()),
        "mean_benchmark_return": float(observations["benchmark_return"].mean()),
        "mean_excess_vs_equal": float(observations["excess_vs_equal"].mean()),
        "excess_vs_equal_95ci": list(equal_ci),
        "mean_excess_vs_benchmark": float(observations["excess_vs_benchmark"].mean()),
        "excess_vs_benchmark_95ci": list(_mean_ci(observations["excess_vs_benchmark"])),
        "hit_rate_vs_equal": float((observations["excess_vs_equal"] > 0).mean()),
        "mean_rank_ic": float(observations["rank_ic"].mean()),
        "by_regime": {
            label: {
                "observations": len(group),
                "mean_excess_vs_equal": float(group["excess_vs_equal"].mean()),
                "hit_rate_vs_equal": float((group["excess_vs_equal"] > 0).mean()),
            }
            for label, group in observations.dropna(subset=["regime"]).groupby("regime")
        },
    }
    return summary, observations


def latest_snapshot(
    signals: pd.DataFrame,
    target_date: str | None = None,
    *,
    tail_days: int = 5,
) -> pd.DataFrame:
    if tail_days < 1:
        raise ValueError("tail_days must be positive")
    cutoff = pd.Timestamp(target_date) if target_date else signals["date"].max()
    available_dates = signals.loc[signals["date"] <= cutoff, "date"]
    if available_dates.empty:
        raise ValueError("no signal is available on or before the requested date")
    current_date = available_dates.max()
    history = signals.loc[signals["date"] <= current_date].sort_values(["ticker", "date"]).copy()
    grouped = history.groupby("ticker")
    history["prior_rs_ratio"] = grouped["rs_ratio"].shift(tail_days)
    history["prior_rs_momentum"] = grouped["rs_momentum"].shift(tail_days)
    history["prior_quadrant"] = grouped["quadrant"].shift(tail_days)
    snapshot = history.loc[history["date"] == current_date].copy()
    snapshot["delta_rs_ratio"] = snapshot["rs_ratio"] - snapshot["prior_rs_ratio"]
    snapshot["delta_rs_momentum"] = (
        snapshot["rs_momentum"] - snapshot["prior_rs_momentum"]
    )
    dx, dy = snapshot["delta_rs_ratio"], snapshot["delta_rs_momentum"]
    snapshot["tail_heading"] = np.select(
        [(dx >= 0) & (dy >= 0), (dx >= 0) & (dy < 0), (dx < 0) & (dy < 0), (dx < 0) & (dy >= 0)],
        ["NE_STRENGTHENING", "SE_LOSING_MOMENTUM", "SW_DETERIORATING", "NW_EARLY_RECOVERY"],
        default="WARMUP",
    )
    snapshot["quadrant_transition"] = (
        snapshot["prior_quadrant"].fillna("WARMUP") + "->" + snapshot["quadrant"]
    )
    return snapshot.sort_values(
        ["shadow_score", "rs_ratio"], ascending=False
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="S&P sector RRG-style shadow rotation")
    commands = parser.add_subparsers(dest="command", required=True)

    register = commands.add_parser("register", help="register 11 sector ETF proxies")
    register.add_argument("database", type=Path)

    status = commands.add_parser("status", help="show the latest shadow sector ranking")
    status.add_argument("database", type=Path)
    status.add_argument("--date")
    status.add_argument("--tail-days", type=int, default=5)
    status.add_argument("--model-version", default=DEFAULT_REGIME_MODEL)

    evaluate = commands.add_parser("evaluate", help="evaluate monthly shadow rankings")
    evaluate.add_argument("database", type=Path)
    evaluate.add_argument("output", type=Path)
    evaluate.add_argument("--model-version", default=DEFAULT_REGIME_MODEL)
    evaluate.add_argument("--start-date")
    evaluate.add_argument("--horizon", type=int, default=20)
    evaluate.add_argument("--top-k", type=int, default=3)

    args = parser.parse_args()
    if args.command == "register":
        print(f"registered {register_sector_universe(args.database)} tickers in {SECTOR_GROUP}")
        return

    prices = load_price_matrix(args.database)
    signals = build_rrg_signals(prices)
    if args.command == "status":
        snapshot = latest_snapshot(signals, args.date, tail_days=args.tail_days)
        regime = _regime_labels(load_regime_probabilities(args.database, args.model_version))
        regime = regime.loc[regime["date"] <= snapshot["date"].iloc[0]]
        if regime.empty:
            raise ValueError("no fixed-regime result is available on or before the signal date")
        fixed = regime.iloc[-1]
        print(
            f"fixed_regime={fixed['regime']} as_of={fixed['date'].date().isoformat()} "
            f"model={args.model_version} (attribution only)"
        )
        print(f"RRG tail: {args.tail_days} trading days")
        print(
            snapshot[
                [
                    "date", "ticker", "sector", "quadrant", "quadrant_transition",
                    "tail_heading", "delta_rs_ratio", "delta_rs_momentum",
                    "rs_ratio", "rs_momentum", "shadow_score",
                ]
            ].to_string(index=False)
        )
        return

    regime = load_regime_probabilities(args.database, args.model_version)
    if args.start_date:
        start = pd.Timestamp(args.start_date)
        prices = prices.loc[prices.index >= start]
        signals = signals.loc[signals["date"] >= start]
        regime = regime.loc[regime["date"] >= start]
    summary, observations = evaluate_shadow(
        prices,
        signals,
        regime,
        horizon=args.horizon,
        top_k=args.top_k,
    )
    summary.update(
        {
            "regime_model_version": args.model_version,
            "regime_role": "fixed attribution only; never refit or modified",
            "rrg_proxy_parameters": {"ratio_window": 63, "momentum_lag": 10, "center": 100},
            "confidence_interval": "normal approximation on monthly observations",
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    observations.to_csv(args.output.with_name(args.output.stem + "_observations.csv"), index=False)
    latest_snapshot(signals).to_csv(args.output.with_name(args.output.stem + "_latest.csv"), index=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
