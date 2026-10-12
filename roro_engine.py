"""Causal equal-weight multi-asset RORO score for research shadow use."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


MODEL_VERSION = "roro-ew-v1-w756-m252-u20"
OUTPUT_START = pd.Timestamp("2025-09-02")
TRAILING_WINDOW = 756
MIN_HISTORY = 252
MIN_UNIQUE = 20

INPUT_COLUMNS = (
    "nq",
    "hy_oas",
    "move",
    "vix",
    "vix3m",
    "usdjpy",
    "gold",
    "breadth_200dma",
    "portfolio_correlation_20",
)
BLOCKS = (
    "credit",
    "rates_volatility",
    "equity_volatility",
    "equity_trend",
    "fx_carry",
    "safe_haven",
    "market_internals",
)
MANDATORY_BLOCKS = ("equity_volatility", "equity_trend")
MIN_AVAILABLE_BLOCKS = 5


def causal_midrank_percentile(
    values: pd.Series,
    *,
    window: int = TRAILING_WINDOW,
    min_history: int = MIN_HISTORY,
    min_unique: int = MIN_UNIQUE,
) -> pd.Series:
    """Rank each finite value against finite observations strictly before it."""
    if not 1 <= min_unique <= min_history <= window:
        raise ValueError("require 1 <= min_unique <= min_history <= window")
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    result = np.full(len(numeric), np.nan)
    history: list[float] = []
    for position, value in enumerate(numeric):
        if not np.isfinite(value):
            continue
        trailing = np.asarray(history[-window:], dtype=float)
        if len(trailing) >= min_history and np.unique(trailing).size >= min_unique:
            lower = np.count_nonzero(trailing < value)
            equal = np.count_nonzero(trailing == value)
            result[position] = (0.5 + lower + 0.5 * equal) / (len(trailing) + 1)
        history.append(float(value))
    return pd.Series(result, index=values.index, dtype=float)


def _strict_mean(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    return frame[list(columns)].mean(axis=1, skipna=False)


def _drivers(row: pd.Series, *, positive: bool) -> tuple[str, ...]:
    values = {
        block: float(row[f"{block}_contribution"])
        for block in BLOCKS
        if pd.notna(row[f"{block}_contribution"])
    }
    ordered = sorted(values, key=values.get, reverse=positive)
    return tuple(
        block
        for block in ordered
        if (values[block] > 0 if positive else values[block] < 0)
    )[:3]


def _regime(score: object) -> str:
    if pd.isna(score):
        return "NO_SCORE"
    value = float(score)
    return "GREEN" if value <= 33 else "YELLOW" if value <= 66 else "RED"


def build_roro_scores(
    panel: pd.DataFrame,
    *,
    window: int = TRAILING_WINDOW,
    min_history: int = MIN_HISTORY,
    min_unique: int = MIN_UNIQUE,
    output_start: str | pd.Timestamp = OUTPUT_START,
) -> pd.DataFrame:
    """Build the frozen SHADOW RORO-EW-v1 score on an NQ trading-day panel."""
    requested_start = pd.Timestamp(output_start)
    if requested_start < OUTPUT_START:
        raise ValueError(
            f"output_start cannot precede the frozen backtest start {OUTPUT_START.date()}"
        )
    missing = [column for column in INPUT_COLUMNS if column not in panel]
    if missing:
        raise ValueError(f"missing RORO inputs: {', '.join(missing)}")
    if not isinstance(panel.index, pd.DatetimeIndex):
        raise ValueError("panel must use a DatetimeIndex")
    if panel.index.has_duplicates or not panel.index.is_monotonic_increasing:
        raise ValueError("panel index must be unique and increasing")

    clean = panel.loc[:, list(INPUT_COLUMNS)].apply(pd.to_numeric, errors="coerce").copy()
    for column in ("nq", "hy_oas", "move", "vix", "vix3m", "usdjpy", "gold"):
        clean[column] = clean[column].where(clean[column] > 0)
    clean["breadth_200dma"] = clean["breadth_200dma"].where(
        clean["breadth_200dma"].between(0, 1)
    )
    clean["portfolio_correlation_20"] = clean["portfolio_correlation_20"].where(
        clean["portfolio_correlation_20"].between(-1, 1)
    )

    log_nq = np.log(clean["nq"])
    raw = pd.DataFrame(index=clean.index)
    raw["credit_level"] = np.log(clean["hy_oas"])
    raw["credit_widening_20"] = clean["hy_oas"].diff(20)
    raw["move_level"] = np.log(clean["move"])
    raw["vix_level"] = np.log(clean["vix"])
    raw["vix_curve"] = np.log(clean["vix"] / clean["vix3m"])
    raw["nq_return_20_risk"] = -log_nq.diff(20)
    raw["nq_ma200_risk"] = -np.log(clean["nq"] / clean["nq"].rolling(200).mean())
    raw["nq_drawdown_60_risk"] = -(
        clean["nq"] / clean["nq"].rolling(60).max() - 1
    )
    raw["yen_strength_20"] = -np.log(clean["usdjpy"]).diff(20)
    raw["gold_relative_20"] = np.log(clean["gold"]).diff(20) - log_nq.diff(20)
    raw["breadth_risk"] = 1 - clean["breadth_200dma"]
    raw["portfolio_correlation_risk"] = clean["portfolio_correlation_20"]

    ranks = pd.DataFrame(
        {
            f"{column}_percentile": causal_midrank_percentile(
                raw[column],
                window=window,
                min_history=min_history,
                min_unique=min_unique,
            )
            * 100
            for column in raw
        },
        index=raw.index,
    )
    result = pd.concat([raw, ranks], axis=1)
    result["credit_score"] = _strict_mean(
        ranks, ("credit_level_percentile", "credit_widening_20_percentile")
    )
    result["rates_volatility_score"] = ranks["move_level_percentile"]
    result["equity_volatility_score"] = _strict_mean(
        ranks, ("vix_level_percentile", "vix_curve_percentile")
    )
    result["equity_trend_score"] = _strict_mean(
        ranks,
        (
            "nq_return_20_risk_percentile",
            "nq_ma200_risk_percentile",
            "nq_drawdown_60_risk_percentile",
        ),
    )
    result["fx_carry_score"] = ranks["yen_strength_20_percentile"]
    result["safe_haven_score"] = ranks["gold_relative_20_percentile"]
    result["market_internals_score"] = _strict_mean(
        ranks, ("breadth_risk_percentile", "portfolio_correlation_risk_percentile")
    )

    block_columns = [f"{block}_score" for block in BLOCKS]
    available = result[block_columns].notna()
    result["available_block_count"] = available.sum(axis=1)
    result["completeness"] = 100 * result["available_block_count"] / len(BLOCKS)
    valid = (
        result[[f"{block}_score" for block in MANDATORY_BLOCKS]].notna().all(axis=1)
        & result["available_block_count"].ge(MIN_AVAILABLE_BLOCKS)
    )
    result["score_raw"] = result[block_columns].mean(axis=1).where(valid)
    result["risk_score"] = np.floor(result["score_raw"] + 0.5).astype("Int64")
    result["disagreement"] = result[block_columns].std(axis=1, ddof=0).where(valid)
    result["agreement"] = (100 - 2 * result["disagreement"]).clip(0, 100)
    result["data_status"] = np.select(
        [valid & result["available_block_count"].eq(len(BLOCKS)), valid],
        ["FULL", "AS_AVAILABLE"],
        default="NO_SCORE",
    )
    result["roro_regime"] = result["risk_score"].map(_regime)

    denominator = result["available_block_count"].replace(0, np.nan)
    for block in BLOCKS:
        result[f"{block}_contribution"] = (
            (result[f"{block}_score"] - 50) / denominator
        ).where(valid)
        result[f"{block}_regime"] = result[f"{block}_score"].map(_regime)
        for lag in (1, 5, 20):
            result[f"{block}_trajectory_{lag}d"] = result[f"{block}_score"].diff(lag)
    for lag in (1, 5, 20):
        result[f"trajectory_{lag}d"] = result["risk_score"].astype("Float64").diff(lag)

    result["risk_drivers"] = result.apply(
        lambda row: _drivers(row, positive=True) if pd.notna(row["score_raw"]) else (),
        axis=1,
    )
    result["relief_drivers"] = result.apply(
        lambda row: _drivers(row, positive=False) if pd.notna(row["score_raw"]) else (),
        axis=1,
    )
    result["available_blocks"] = available.apply(
        lambda row: tuple(block for block in BLOCKS if bool(row[f"{block}_score"])),
        axis=1,
    )

    output = result.loc[result.index >= requested_start].copy()
    output.attrs.update(
        {
            "model_version": MODEL_VERSION,
            "spec_status": "FROZEN_SHADOW_NOT_LIVE",
            "output_start": requested_start.date().isoformat(),
            "window": window,
            "min_history": min_history,
            "min_unique": min_unique,
            "block_weights": "equal over available blocks; minimum 5/7",
            "execution": "signal at close t; apply from next tradable bar",
        }
    )
    return output
