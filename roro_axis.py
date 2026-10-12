"""Frozen three-axis RORO score for SHADOW research only."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from roro_engine import causal_midrank_percentile


MODEL_VERSION = "roro-3axis-v1-w756-m252-p126-u20"
PORTFOLIO_MIN_HISTORY = 126
OUTPUT_START = pd.Timestamp("2025-09-02")
MACRO_BLOCKS = (
    "credit",
    "rates_volatility",
    "equity_volatility",
    "fx_carry",
    "safe_haven",
)
PORTFOLIO_COMPONENTS = (
    "correlation",
    "beta",
    "concentration",
    "participation",
    "drawdown",
)
PORTFOLIO_INPUTS = (
    "portfolio_correlation_20",
    "nq_beta_60",
    "concentration_hhi",
    "negative_participation_20",
    "portfolio_drawdown_60",
    "breadth_above_ma20",
    "breadth_above_ma50",
    "breadth_above_ma200",
)
LONG_ANCHORS = ((0, 100), (33, 90), (66, 60), (100, 25))
AXIS_COMPONENTS = {
    "macro_risk": (
        "credit",
        "rates_volatility",
        "equity_volatility",
        "fx_carry",
        "safe_haven",
    ),
    "market_trend": ("equity_trend", "market_breadth"),
    "portfolio_internal": (
        "portfolio_correlation",
        "portfolio_beta",
        "portfolio_concentration",
        "portfolio_participation",
        "portfolio_drawdown",
    ),
}
COMPONENT_SCORE_COLUMNS = {
    "credit": "credit_score",
    "rates_volatility": "rates_volatility_score",
    "equity_volatility": "equity_volatility_score",
    "fx_carry": "fx_carry_score",
    "safe_haven": "safe_haven_score",
    "equity_trend": "equity_trend_score",
    "market_breadth": "breadth_risk_percentile",
    "portfolio_correlation": "portfolio_correlation_score",
    "portfolio_beta": "portfolio_beta_score",
    "portfolio_concentration": "portfolio_concentration_score",
    "portfolio_participation": "portfolio_participation_score",
    "portfolio_drawdown": "portfolio_drawdown_score",
}


def _round_half_up(values: pd.Series) -> pd.Series:
    return pd.Series(
        np.floor(pd.to_numeric(values, errors="coerce") + 0.5),
        index=values.index,
        dtype="Int64",
    )


def long_exposure_from_risk_score(score: pd.Series) -> pd.Series:
    """Map a 0--100 risk score to the frozen continuous Long anchors."""
    numeric = pd.to_numeric(score, errors="coerce").where(lambda value: value.between(0, 100))
    xp, fp = zip(*LONG_ANCHORS)
    interpolated = pd.Series(
        np.interp(numeric.fillna(0), xp, fp), index=score.index, dtype=float
    ).where(numeric.notna())
    return _round_half_up(interpolated)


def _check_index(frame: pd.DataFrame, name: str) -> None:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError(f"{name} must use a DatetimeIndex")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError(f"{name} index must be unique and increasing")


def _axis_drivers(row: pd.Series, *, positive: bool) -> tuple[str, ...]:
    contributions = {
        axis: float(row[f"{axis}_contribution"])
        for axis in ("macro_risk", "market_trend", "portfolio_internal")
        if pd.notna(row[f"{axis}_contribution"])
    }
    ordered = sorted(contributions, key=contributions.get, reverse=positive)
    return tuple(
        axis
        for axis in ordered
        if (contributions[axis] > 0 if positive else contributions[axis] < 0)
    )[:3]


def _component_drivers(row: pd.Series, *, positive: bool) -> tuple[str, ...]:
    values = {
        component: float(row[f"{component}_contribution"])
        for component in COMPONENT_SCORE_COLUMNS
        if pd.notna(row[f"{component}_contribution"])
    }
    ordered = sorted(values, key=values.get, reverse=positive)
    return tuple(
        name
        for name in ordered
        if (values[name] > 0 if positive else values[name] < 0)
    )[:3]


def _available_names(row: pd.Series, names: Iterable[str], suffix: str) -> tuple[str, ...]:
    return tuple(name for name in names if pd.notna(row[f"{name}{suffix}"]))


def _portfolio_scores(
    portfolio: pd.DataFrame,
    *,
    window: int,
    min_history: int,
    min_unique: int,
) -> pd.DataFrame:
    missing = [column for column in PORTFOLIO_INPUTS if column not in portfolio]
    if missing:
        raise ValueError(f"missing portfolio inputs: {', '.join(missing)}")
    clean = portfolio.loc[:, PORTFOLIO_INPUTS].apply(pd.to_numeric, errors="coerce")
    clean["portfolio_correlation_20"] = clean["portfolio_correlation_20"].where(
        clean["portfolio_correlation_20"].between(-1, 1)
    )
    clean["concentration_hhi"] = clean["concentration_hhi"].where(
        clean["concentration_hhi"].gt(0) & clean["concentration_hhi"].le(1)
    )
    clean["negative_participation_20"] = clean["negative_participation_20"].where(
        clean["negative_participation_20"].between(0, 1)
    )
    clean["portfolio_drawdown_60"] = clean["portfolio_drawdown_60"].where(
        clean["portfolio_drawdown_60"].between(-1, 0)
    )
    for column in ("breadth_above_ma20", "breadth_above_ma50", "breadth_above_ma200"):
        clean[column] = clean[column].where(clean[column].between(0, 1))

    raw = pd.DataFrame(
        {
            "correlation": clean["portfolio_correlation_20"],
            "beta": clean["nq_beta_60"],
            "concentration": clean["concentration_hhi"],
            "participation": clean["negative_participation_20"],
            "drawdown": -clean["portfolio_drawdown_60"],
            "breadth_ma20": 1 - clean["breadth_above_ma20"],
            "breadth_ma50": 1 - clean["breadth_above_ma50"],
            "breadth_ma200": 1 - clean["breadth_above_ma200"],
        },
        index=portfolio.index,
    )
    scores = pd.DataFrame(index=portfolio.index)
    for column in raw:
        scores[f"portfolio_{column}_score"] = (
            causal_midrank_percentile(
                raw[column],
                window=window,
                min_history=min_history,
                min_unique=min_unique,
            )
            * 100
        )
    scores["portfolio_breadth_score"] = scores["portfolio_breadth_ma20_score"]
    scores["portfolio_participation_score"] = scores[
        ["portfolio_participation_score", "portfolio_breadth_ma20_score"]
    ].mean(axis=1, skipna=False)

    primary = [f"portfolio_{name}_score" for name in PORTFOLIO_COMPONENTS]
    scores["portfolio_available_component_count"] = scores[primary].notna().sum(axis=1)
    valid = (
        scores["portfolio_correlation_score"].notna()
        & scores["portfolio_breadth_score"].notna()
        & scores["portfolio_available_component_count"].ge(4)
    )
    scores["portfolio_internal_score"] = scores[primary].mean(axis=1).where(valid)

    prior_counts = raw.notna().shift(1).rolling(window, min_periods=1).sum().fillna(0)
    mandatory_raw = ("correlation", "participation", "breadth_ma20")
    history_blocked = (
        raw[list(mandatory_raw)].notna().all(axis=1)
        & prior_counts[list(mandatory_raw)].lt(min_history).any(axis=1)
    )
    scores["portfolio_history_blocked"] = history_blocked
    scores["portfolio_data_status"] = np.select(
        [
            valid & scores["portfolio_available_component_count"].eq(5),
            valid,
            ~valid & history_blocked,
        ],
        ["CONDITIONAL_EVENT_PIT", "CONDITIONAL_EVENT_PIT", "DATA_BLOCKED"],
        default="NO_SCORE",
    )
    return scores


def build_roro_axis_scores(
    roro_scores: pd.DataFrame,
    portfolio: pd.DataFrame,
    *,
    window: int = 756,
    min_history: int = 252,
    portfolio_min_history: int = PORTFOLIO_MIN_HISTORY,
    min_unique: int = 20,
    output_start: str | pd.Timestamp = OUTPUT_START,
) -> pd.DataFrame:
    """Combine macro, NQ trend and aggregate portfolio internals causally."""
    requested_start = pd.Timestamp(output_start)
    if requested_start < OUTPUT_START:
        raise ValueError(
            f"output_start cannot precede the frozen backtest start {OUTPUT_START.date()}"
        )
    _check_index(roro_scores, "roro_scores")
    _check_index(portfolio, "portfolio")
    required = [f"{block}_score" for block in MACRO_BLOCKS] + [
        "equity_trend_score",
        "breadth_risk_percentile",
    ]
    missing = [column for column in required if column not in roro_scores]
    if missing:
        raise ValueError(f"missing RORO score inputs: {', '.join(missing)}")

    portfolio_scores = _portfolio_scores(
        portfolio,
        window=window,
        min_history=portfolio_min_history,
        min_unique=min_unique,
    )
    result = roro_scores.loc[:, required].copy()
    result = result.join(portfolio_scores, how="left")

    macro_columns = [f"{block}_score" for block in MACRO_BLOCKS]
    result["macro_available_component_count"] = result[macro_columns].notna().sum(axis=1)
    macro_valid = (
        result["equity_volatility_score"].notna()
        & result["macro_available_component_count"].ge(2)
    )
    result["macro_risk_score"] = result[macro_columns].mean(axis=1).where(macro_valid)
    result["macro_data_status"] = np.select(
        [
            macro_valid & result["macro_available_component_count"].eq(5),
            macro_valid,
        ],
        ["FULL", "AS_AVAILABLE"],
        default="NO_SCORE",
    )

    market_columns = ["equity_trend_score", "breadth_risk_percentile"]
    result["market_available_component_count"] = result[market_columns].notna().sum(axis=1)
    market_valid = result["equity_trend_score"].notna()
    result["market_trend_score"] = result[market_columns].mean(axis=1).where(market_valid)
    result["market_data_status"] = np.select(
        [
            market_valid & result["breadth_risk_percentile"].notna(),
            market_valid,
        ],
        ["FULL", "NQ_ONLY"],
        default="NO_SCORE",
    )

    result["macro_completeness"] = (
        100 * result["macro_available_component_count"] / 5
    ).where(macro_valid, 0)
    result["market_completeness"] = (
        100 * result["market_available_component_count"] / 2
    ).where(market_valid, 0)
    portfolio_valid = result["portfolio_internal_score"].notna()
    result["portfolio_completeness"] = (
        100 * result["portfolio_available_component_count"] / 5
    ).where(portfolio_valid, 0)

    axes = ("macro_risk", "market_trend", "portfolio_internal")
    axis_columns = [f"{axis}_score" for axis in axes]
    result["available_axis_count"] = result[axis_columns].notna().sum(axis=1)
    final_valid = (
        result["market_trend_score"].notna()
        & result["portfolio_internal_score"].notna()
        & result["available_axis_count"].ge(2)
    )
    result["market_risk_score_raw"] = result[axis_columns].mean(axis=1).where(final_valid)
    result["market_risk_score"] = _round_half_up(result["market_risk_score_raw"])
    result["data_completeness"] = result[
        ["macro_completeness", "market_completeness", "portfolio_completeness"]
    ].mean(axis=1)
    result["axis_disagreement"] = result[axis_columns].std(axis=1, ddof=0).where(final_valid)
    result["axis_agreement"] = (100 - 2 * result["axis_disagreement"]).clip(0, 100)
    result["confidence"] = _round_half_up(
        (result["data_completeness"] * result["axis_agreement"] / 100).where(final_valid)
    )

    full = (
        final_valid
        & result["macro_data_status"].eq("FULL")
        & result["market_data_status"].eq("FULL")
        & result["portfolio_data_status"].eq("FULL_STRICT")
    )
    conditional = final_valid & result["portfolio_data_status"].eq(
        "CONDITIONAL_EVENT_PIT"
    )
    blocked = ~final_valid & result["portfolio_data_status"].eq("DATA_BLOCKED")
    result["data_status"] = np.select(
        [full, conditional, final_valid, blocked],
        ["FULL_STRICT", "CONDITIONAL_EVENT_PIT", "AS_AVAILABLE_STRICT", "DATA_BLOCKED"],
        default="NO_SCORE",
    )
    result["risk_regime"] = np.select(
        [
            result["market_risk_score"].le(33).fillna(False),
            result["market_risk_score"].le(66).fillna(False),
        ],
        ["GREEN", "YELLOW"],
        default="RED",
    )
    result.loc[result["market_risk_score"].isna(), "risk_regime"] = "NO_SCORE"
    result["recommended_long"] = long_exposure_from_risk_score(
        result["market_risk_score"]
    )

    denominator = result["available_axis_count"].replace(0, np.nan)
    for axis in axes:
        result[f"{axis}_contribution"] = (
            (result[f"{axis}_score"] - 50) / denominator
        ).where(final_valid)
    result["risk_drivers"] = result.apply(
        lambda row: _axis_drivers(row, positive=True) if pd.notna(row["market_risk_score_raw"]) else (),
        axis=1,
    )
    result["relief_drivers"] = result.apply(
        lambda row: _axis_drivers(row, positive=False) if pd.notna(row["market_risk_score_raw"]) else (),
        axis=1,
    )
    result["available_axes"] = result.apply(
        lambda row: _available_names(row, axes, "_score"), axis=1
    )
    result["axis_completeness"] = result.apply(
        lambda row: {
            "macro_risk": float(row["macro_completeness"]),
            "market_trend": float(row["market_completeness"]),
            "portfolio_internal": float(row["portfolio_completeness"]),
        },
        axis=1,
    )
    for axis, components in AXIS_COMPONENTS.items():
        axis_available = result[f"{axis}_score"].notna()
        component_count = result[
            [COMPONENT_SCORE_COLUMNS[name] for name in components]
        ].notna().sum(axis=1).replace(0, np.nan)
        for component in components:
            score_column = COMPONENT_SCORE_COLUMNS[component]
            result[f"{component}_contribution"] = (
                (result[score_column] - 50)
                / component_count
                / result["available_axis_count"].replace(0, np.nan)
            ).where(final_valid & axis_available)
            for lag in (1, 5, 20):
                result[f"{component}_score_trajectory_{lag}d"] = result[score_column].diff(lag)
    result["available_components"] = result.apply(
        lambda row: tuple(
            name
            for name, column in COMPONENT_SCORE_COLUMNS.items()
            if pd.notna(row[column])
        ),
        axis=1,
    )
    result["component_risk_drivers"] = result.apply(
        lambda row: _component_drivers(row, positive=True)
        if pd.notna(row["market_risk_score_raw"])
        else (),
        axis=1,
    )
    result["component_relief_drivers"] = result.apply(
        lambda row: _component_drivers(row, positive=False)
        if pd.notna(row["market_risk_score_raw"])
        else (),
        axis=1,
    )
    for lag in (1, 5, 20):
        result[f"trajectory_{lag}d"] = result["market_risk_score"].astype("Float64").diff(lag)

    output = result.loc[result.index >= requested_start].copy()
    output.attrs.update(
        {
            "model_version": MODEL_VERSION,
            "spec_status": "FROZEN_SHADOW_NOT_LIVE",
            "output_start": requested_start.date().isoformat(),
            "portfolio_scoring": "5 primary components; strict MA20 + negative participation gate; MA50/200 diagnostic",
            "portfolio_min_history": portfolio_min_history,
            "execution": "signal at close t; apply from next tradable bar",
        }
    )
    return output
