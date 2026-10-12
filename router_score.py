"""Causal NQ rule-based market risk score used as a research shadow."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from regime_router import build_features_v2, load_benchmark_from_database
from research_drawdown_onset import causal_trailing_percentile


ROUTER_SCORE_VERSION = "nq-rule-score-v1-w756-m252"
TRAILING_WINDOW = 756
MIN_HISTORY = 252


def _round_half_up(value: float) -> int:
    return math.floor(value + 0.5)


def build_router_scores(
    market: pd.DataFrame,
    *,
    window: int = TRAILING_WINDOW,
    min_history: int = MIN_HISTORY,
) -> pd.DataFrame:
    """Build a causal 0-100 score without fitting coefficients to outcomes."""
    features = build_features_v2(market)
    ranks = pd.DataFrame(
        {
            "momentum_20_risk": causal_trailing_percentile(
                -features["momentum_20"], window=window, min_history=min_history
            ),
            "momentum_60_risk": causal_trailing_percentile(
                -features["momentum_60"], window=window, min_history=min_history
            ),
            "ma200_distance_risk": causal_trailing_percentile(
                -features["ma200_distance"], window=window, min_history=min_history
            ),
            "ma200_slope_risk": causal_trailing_percentile(
                -features["ma200_slope_20"], window=window, min_history=min_history
            ),
            "volatility_risk": causal_trailing_percentile(
                features["volatility_20"], window=window, min_history=min_history
            ),
            "drawdown_risk": causal_trailing_percentile(
                -features["drawdown_60"], window=window, min_history=min_history
            ),
        },
        index=features.index,
    )
    ranks["trend_risk"] = ranks[
        [
            "momentum_20_risk",
            "momentum_60_risk",
            "ma200_distance_risk",
            "ma200_slope_risk",
        ]
    ].mean(axis=1)
    complete = ranks[["trend_risk", "volatility_risk", "drawdown_risk"]].notna().all(axis=1)
    ranks["score_raw"] = (
        100 * ranks[["trend_risk", "volatility_risk", "drawdown_risk"]].mean(axis=1)
    ).where(complete)
    ranks["risk_score"] = ranks["score_raw"].map(
        lambda value: _round_half_up(float(value)) if pd.notna(value) else pd.NA
    ).astype("Int64")
    ranks.attrs.update(
        {
            "model_version": ROUTER_SCORE_VERSION,
            "benchmark": "NQ=F",
            "window": window,
            "min_history": min_history,
        }
    )
    return ranks


def build_router_scores_from_database(
    database: Path, *, benchmark: str = "NQ=F"
) -> pd.DataFrame:
    result = build_router_scores(load_benchmark_from_database(database, benchmark))
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
        "trend_risk": float(row["trend_risk"]),
        "volatility_risk": float(row["volatility_risk"]),
        "drawdown_risk": float(row["drawdown_risk"]),
    }
