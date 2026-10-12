"""Causal NQ stress score that treats ordinary conditions as low risk."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from regime_router import build_features_v2, load_benchmark_from_database
from router_score import MIN_HISTORY, TRAILING_WINDOW, _round_half_up, build_router_scores


ROUTER_SCORE_VERSION = "nq-rule-stress-score-v2-w756-m252"
MIN_UNIQUE_HISTORY = 20


def _history_has_variation(
    series: pd.Series, *, window: int, min_history: int
) -> pd.Series:
    unique = series.shift(1).rolling(window, min_periods=min_history).apply(
        lambda values: float(np.unique(values).size), raw=True
    )
    return unique >= MIN_UNIQUE_HISTORY


def build_router_scores_v2(
    market: pd.DataFrame,
    *,
    window: int = TRAILING_WINDOW,
    min_history: int = MIN_HISTORY,
) -> pd.DataFrame:
    """Score only the portion of each risk block above its trailing median."""
    result = build_router_scores(market, window=window, min_history=min_history)
    features = build_features_v2(market)
    variation_ok = pd.concat(
        [
            _history_has_variation(features[column], window=window, min_history=min_history)
            for column in (
                "momentum_20",
                "momentum_60",
                "ma200_distance",
                "ma200_slope_20",
                "volatility_20",
                "drawdown_60",
            )
        ],
        axis=1,
    ).all(axis=1)
    blocks = (2 * result[["trend_risk", "volatility_risk", "drawdown_risk"]] - 1).clip(
        lower=0,
        upper=1,
    )
    result[["trend_stress", "volatility_stress", "drawdown_stress"]] = blocks.to_numpy()
    complete = blocks.notna().all(axis=1) & variation_ok
    result["score_raw"] = (100 * blocks.mean(axis=1)).where(complete)
    result["risk_score"] = result["score_raw"].map(
        lambda value: _round_half_up(float(value)) if pd.notna(value) else pd.NA
    ).astype("Int64")
    result.attrs.update(
        {
            "model_version": ROUTER_SCORE_VERSION,
            "benchmark": "NQ=F",
            "window": window,
            "min_history": min_history,
        }
    )
    return result


def build_router_scores_from_database(
    database: Path, *, benchmark: str = "NQ=F"
) -> pd.DataFrame:
    result = build_router_scores_v2(load_benchmark_from_database(database, benchmark))
    result.attrs["benchmark"] = benchmark
    return result


def get_router_score(database: Path, as_of: str) -> dict[str, object]:
    scores = build_router_scores_from_database(database)
    target = pd.Timestamp(as_of)
    if target not in scores.index or pd.isna(scores.at[target, "risk_score"]):
        raise ValueError(f"{as_of} has no complete router score")
    row = scores.loc[target]
    return {
        "date": target.date().isoformat(),
        "model_version": ROUTER_SCORE_VERSION,
        "risk_score": int(row["risk_score"]),
        "score_raw": float(row["score_raw"]),
        "trend_stress": float(row["trend_stress"]),
        "volatility_stress": float(row["volatility_stress"]),
        "drawdown_stress": float(row["drawdown_stress"]),
    }
