"""Causal turning-phase checklist for the three-axis RORO SHADOW."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


MODEL_VERSION = "roro-phase-v1"
CHECKLIST_MIN_AVAILABLE = 4
CHECKLIST_CONFIRMATION_RATE = 0.60

AXIS_INPUTS = ("market_risk_score", "trajectory_5d")
MARKET_INPUTS = ("nq", "vix", "vix3m", "hy_oas", "breadth_200dma")
PORTFOLIO_INPUTS = ("portfolio_correlation_20", "breadth_above_ma20")

BOTTOM_LABELS = (
    "VIX_CURVE_5D_DOWN",
    "HY_OAS_20D_NON_WIDENING",
    "PORTFOLIO_CORRELATION_5D_DOWN",
    "MARKET_BREADTH_5D_UP",
    "PORTFOLIO_BREADTH_MA20_5D_UP",
    "NQ_ABOVE_MA5",
)
TOP_LABELS = (
    "VIX_CURVE_5D_UP",
    "HY_OAS_20D_WIDENING",
    "PORTFOLIO_CORRELATION_5D_UP",
    "MARKET_BREADTH_5D_DOWN",
    "PORTFOLIO_BREADTH_MA20_5D_DOWN",
    "NQ_BELOW_MA5",
)


def _check_frame(frame: pd.DataFrame, name: str, required: tuple[str, ...]) -> None:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError(f"{name} must use a DatetimeIndex")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError(f"{name} index must be unique and increasing")
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"missing {name} inputs: {', '.join(missing)}")


def _numeric(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    return frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")


def _compare(
    current: pd.Series,
    reference: pd.Series,
    operation: Callable[[pd.Series, pd.Series], pd.Series],
) -> pd.Series:
    available = current.notna() & reference.notna()
    result = pd.Series(pd.NA, index=current.index, dtype="boolean")
    result.loc[available] = operation(current.loc[available], reference.loc[available])
    return result


def _evidence(signals: pd.DataFrame, labels: tuple[str, ...], value: bool) -> pd.Series:
    return pd.Series(
        [
            [label for label, signal in zip(labels, row, strict=True) if signal is value]
            for row in signals.astype(object).itertuples(index=False, name=None)
        ],
        index=signals.index,
        dtype=object,
    )


def _missing_evidence(signals: pd.DataFrame) -> pd.Series:
    names = (
        "VIX_CURVE",
        "HY_OAS",
        "PORTFOLIO_CORRELATION",
        "MARKET_BREADTH",
        "PORTFOLIO_BREADTH_MA20",
        "NQ_MA5",
    )
    return pd.Series(
        [
            [name for name, signal in zip(names, row, strict=True) if pd.isna(signal)]
            for row in signals.astype(object).itertuples(index=False, name=None)
        ],
        index=signals.index,
        dtype=object,
    )


def build_trend_phase(
    axis_scores: pd.DataFrame,
    market_panel: pd.DataFrame,
    portfolio: pd.DataFrame,
) -> pd.DataFrame:
    """Return frozen, causal phase labels on the axis-score decision calendar."""
    _check_frame(axis_scores, "axis_scores", AXIS_INPUTS)
    _check_frame(market_panel, "market_panel", MARKET_INPUTS)
    _check_frame(portfolio, "portfolio", PORTFOLIO_INPUTS)
    missing_market_dates = axis_scores.index.difference(market_panel.index)
    if not missing_market_dates.empty:
        preview = ", ".join(timestamp.date().isoformat() for timestamp in missing_market_dates[:3])
        raise ValueError(f"market_panel is missing axis dates: {preview}")

    index = axis_scores.index
    axis = _numeric(axis_scores, AXIS_INPUTS)
    market = _numeric(market_panel.reindex(index), MARKET_INPUTS)
    internal = _numeric(portfolio.reindex(index), PORTFOLIO_INPUTS)

    score = axis["market_risk_score"].where(axis["market_risk_score"].between(0, 100))
    trajectory_5d = axis["trajectory_5d"]
    nq = market["nq"].where(market["nq"].gt(0))
    vix = market["vix"].where(market["vix"].gt(0))
    vix3m = market["vix3m"].where(market["vix3m"].gt(0))
    hy_oas = market["hy_oas"].where(market["hy_oas"].gt(0))
    market_breadth = market["breadth_200dma"].where(
        market["breadth_200dma"].between(0, 1)
    )
    correlation = internal["portfolio_correlation_20"].where(
        internal["portfolio_correlation_20"].between(-1, 1)
    )
    portfolio_breadth = internal["breadth_above_ma20"].where(
        internal["breadth_above_ma20"].between(0, 1)
    )

    vix_curve = vix / vix3m
    nq_ma5 = nq.rolling(5, min_periods=5).mean()
    nq_ma20 = nq.rolling(20, min_periods=20).mean()
    bottom_signals = pd.DataFrame(
        {
            "vix_curve": _compare(vix_curve, vix_curve.shift(5), lambda a, b: a < b),
            "hy_oas": _compare(hy_oas, hy_oas.shift(20), lambda a, b: a <= b),
            "portfolio_correlation": _compare(
                correlation, correlation.shift(5), lambda a, b: a < b
            ),
            "market_breadth": _compare(
                market_breadth, market_breadth.shift(5), lambda a, b: a > b
            ),
            "portfolio_breadth_ma20": _compare(
                portfolio_breadth, portfolio_breadth.shift(5), lambda a, b: a > b
            ),
            "nq_ma5": _compare(nq, nq_ma5, lambda a, b: a > b),
        },
        index=index,
    )
    top_signals = pd.DataFrame(
        {
            "vix_curve": _compare(vix_curve, vix_curve.shift(5), lambda a, b: a > b),
            "hy_oas": _compare(hy_oas, hy_oas.shift(20), lambda a, b: a > b),
            "portfolio_correlation": _compare(
                correlation, correlation.shift(5), lambda a, b: a > b
            ),
            "market_breadth": _compare(
                market_breadth, market_breadth.shift(5), lambda a, b: a < b
            ),
            "portfolio_breadth_ma20": _compare(
                portfolio_breadth, portfolio_breadth.shift(5), lambda a, b: a < b
            ),
            "nq_ma5": _compare(nq, nq_ma5, lambda a, b: a < b),
        },
        index=index,
    )

    result = pd.DataFrame(index=index)
    for column in bottom_signals:
        result[f"{column}_available"] = bottom_signals[column].notna()
        result[f"bottom_{column}"] = bottom_signals[column]
        result[f"top_{column}"] = top_signals[column]

    available_count = bottom_signals.notna().sum(axis=1)
    required_count = pd.Series(
        np.ceil(CHECKLIST_CONFIRMATION_RATE * available_count).astype(int),
        index=index,
        dtype="Int64",
    )
    eligible = available_count.ge(CHECKLIST_MIN_AVAILABLE)
    bottom_count = bottom_signals.fillna(False).sum(axis=1).astype("Int64")
    top_count = top_signals.fillna(False).sum(axis=1).astype("Int64")
    bottom_pass = eligible & bottom_count.ge(required_count)
    top_pass = eligible & top_count.ge(required_count)

    result["checklist_available_count"] = available_count.astype("Int64")
    result["checklist_required_count"] = required_count
    result["checklist_eligible"] = eligible
    result["bottom_confirmation_count"] = bottom_count
    result["top_confirmation_count"] = top_count
    result["bottom_checklist_pass"] = bottom_pass
    result["top_checklist_pass"] = top_pass
    result["bottom_confirmed_evidence"] = _evidence(bottom_signals, BOTTOM_LABELS, True)
    result["top_confirmed_evidence"] = _evidence(top_signals, TOP_LABELS, True)
    result["bottom_available_evidence"] = _evidence(bottom_signals, BOTTOM_LABELS, True).combine(
        _evidence(bottom_signals, BOTTOM_LABELS, False), lambda a, b: a + b
    )
    result["top_available_evidence"] = _evidence(top_signals, TOP_LABELS, True).combine(
        _evidence(top_signals, TOP_LABELS, False), lambda a, b: a + b
    )
    result["missing_evidence"] = _missing_evidence(bottom_signals)

    rolling_max = score.rolling(20, min_periods=1).max()
    rolling_min = score.rolling(20, min_periods=1).min()
    drop_from_max = rolling_max - score
    rise_from_min = score - rolling_min
    bottom_signal = (
        rolling_max.ge(67)
        & drop_from_max.ge(5)
        & trajectory_5d.le(-5)
        & bottom_pass
    ).fillna(False)
    top_signal = (
        rolling_min.le(33)
        & rise_from_min.ge(5)
        & trajectory_5d.ge(5)
        & top_pass
    ).fillna(False)
    prior_bottom = (
        bottom_signal.shift(1, fill_value=False)
        .astype(int)
        .rolling(20, min_periods=1)
        .max()
        .gt(0)
    )
    recovery_signal = (
        prior_bottom & nq.gt(nq_ma20) & trajectory_5d.le(0)
    ).fillna(False)

    base_available = score.notna() & nq.notna()
    downtrend = (score.ge(67) & nq.lt(nq_ma20)).fillna(False)
    uptrend = (score.le(33) & nq.gt(nq_ma20)).fillna(False)
    risk_building = trajectory_5d.gt(0).fillna(False)
    result["risk_level"] = score
    result["risk_direction_5d"] = trajectory_5d
    result["nq_ma5"] = nq_ma5
    result["nq_ma20"] = nq_ma20
    result["score_20d_max"] = rolling_max
    result["score_20d_min"] = rolling_min
    result["score_drop_from_20d_max"] = drop_from_max
    result["score_rise_from_20d_min"] = rise_from_min
    result["bottoming_candidate_signal"] = bottom_signal
    result["prior_bottoming_candidate_20d"] = prior_bottom
    result["recovery_confirmation_signal"] = recovery_signal
    result["topping_candidate_signal"] = top_signal
    result["trend_phase"] = np.select(
        [
            ~base_available,
            bottom_signal,
            recovery_signal,
            top_signal,
            downtrend,
            uptrend,
            risk_building,
        ],
        [
            "INSUFFICIENT_EVIDENCE",
            "BOTTOMING_CANDIDATE",
            "RECOVERY_CONFIRMING",
            "TOPPING_CANDIDATE",
            "DOWNTREND",
            "UPTREND",
            "RISK_BUILDING",
        ],
        default="TRANSITION",
    )

    def phase_evidence(row: pd.Series) -> list[str]:
        if row["trend_phase"] == "BOTTOMING_CANDIDATE":
            return list(row["bottom_confirmed_evidence"])
        if row["trend_phase"] == "TOPPING_CANDIDATE":
            return list(row["top_confirmed_evidence"])
        if row["trend_phase"] == "RECOVERY_CONFIRMING":
            return ["PRIOR_BOTTOMING_CANDIDATE_20D", "NQ_ABOVE_MA20", "RISK_5D_NOT_RISING"]
        if row["trend_phase"] == "DOWNTREND":
            return ["RISK_SCORE_AT_LEAST_67", "NQ_BELOW_MA20"]
        if row["trend_phase"] == "UPTREND":
            return ["RISK_SCORE_AT_MOST_33", "NQ_ABOVE_MA20"]
        if row["trend_phase"] == "RISK_BUILDING":
            return ["RISK_SCORE_5D_RISING"]
        return []

    result["turn_evidence"] = result.apply(phase_evidence, axis=1)
    result.index.name = axis_scores.index.name
    result.attrs.update(
        {
            "model_version": MODEL_VERSION,
            "spec_status": "FROZEN_SHADOW_NOT_LIVE",
            "calendar": "axis_scores; exact-date reindex; no fill",
            "checklist_min_available": CHECKLIST_MIN_AVAILABLE,
            "checklist_confirmation_rate": CHECKLIST_CONFIRMATION_RATE,
        }
    )
    return result
