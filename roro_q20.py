"""Read-only causal readiness primitives for the frozen q20 research contract."""

from __future__ import annotations

import numpy as np
import pandas as pd


MODEL_VERSION = "roro-q20-logit-v1-frozen8"
HORIZON = 20
DRAWDOWN_THRESHOLD = -0.05
MIN_COMPLETE_LABELS = 252
MIN_INDEPENDENT_EVENTS = 20
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
READY_FOR_WALK_FORWARD_FIT = "READY_FOR_WALK_FORWARD_FIT"
READINESS_METADATA_FIELDS = (
    "model_version",
    "status",
    "probability",
    "fit_allowed",
    "horizon_trading_days",
    "drawdown_threshold",
    "feature_rows",
    "complete_feature_labels",
    "independent_events",
    "minimum_complete_labels",
    "minimum_independent_events",
)

# Deliberately excludes the aggregate score, regime, confidence and Long exposure:
# each is a deterministic transformation of fields already present below.
FEATURE_COLUMNS = (
    "macro_risk_score",
    "market_trend_score",
    "portfolio_internal_score",
    "trajectory_1d",
    "trajectory_5d",
    "trajectory_20d",
    "data_completeness",
    "axis_disagreement",
)
BOUNDED_0_100 = (
    "macro_risk_score",
    "market_trend_score",
    "portfolio_internal_score",
    "data_completeness",
    "axis_disagreement",
)
BOUNDED_TRAJECTORY = ("trajectory_1d", "trajectory_5d", "trajectory_20d")


class InsufficientEvidenceError(RuntimeError):
    """Raised when a caller attempts fitting before the frozen evidence gate."""


def _normalise_daily(
    daily: pd.DataFrame, *, date_column: str, close_column: str
) -> pd.DataFrame:
    required = [date_column, close_column, *FEATURE_COLUMNS]
    missing = [column for column in required if column not in daily]
    if missing:
        raise ValueError("missing q20 columns: " + ", ".join(missing))
    frame = daily.loc[:, required].copy()
    frame[date_column] = pd.to_datetime(frame[date_column], errors="raise")
    if frame[date_column].dt.tz is not None:
        frame[date_column] = frame[date_column].dt.tz_localize(None)
    frame[date_column] = frame[date_column].dt.normalize()
    if frame[date_column].duplicated().any():
        raise ValueError("q20 dates must be unique")
    return frame.sort_values(date_column).reset_index(drop=True)


def exact_feature_matrix(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return the frozen feature vector and a no-fill, exact-completeness mask."""
    missing = [column for column in FEATURE_COLUMNS if column not in frame]
    if missing:
        raise ValueError("missing q20 features: " + ", ".join(missing))
    matrix = frame.loc[:, FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    complete = pd.Series(
        np.isfinite(matrix.to_numpy(dtype=float)).all(axis=1), index=frame.index
    )
    complete &= matrix.loc[:, BOUNDED_0_100].apply(
        lambda column: column.between(0, 100)
    ).all(axis=1)
    complete &= matrix.loc[:, BOUNDED_TRAJECTORY].apply(
        lambda column: column.between(-100, 100)
    ).all(axis=1)
    return matrix, complete.astype(bool)


def forward_path_drawdown(prices: pd.Series, *, horizon: int = HORIZON) -> pd.Series:
    """Compute min(t+1..t+horizon)/price_t-1; incomplete paths remain NaN."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    values = pd.to_numeric(prices, errors="coerce").to_numpy(dtype=float)
    result = np.full(len(values), np.nan)
    for position in range(len(values) - horizon):
        base = values[position]
        future = values[position + 1 : position + horizon + 1]
        if np.isfinite(base) and base > 0 and np.isfinite(future).all() and (future > 0).all():
            result[position] = float(np.min(future / base - 1.0))
    return pd.Series(result, index=prices.index, dtype=float)


def build_q20_research_frame(
    daily: pd.DataFrame,
    *,
    as_of: object | None = None,
    date_column: str = "date",
    close_column: str = "nq_close",
) -> pd.DataFrame:
    """Build exact-feature rows and labels observable no later than ``as_of``."""
    frame = _normalise_daily(
        daily, date_column=date_column, close_column=close_column
    )
    if frame.empty:
        result = pd.DataFrame(
            columns=[
                "date",
                "nq_close",
                *FEATURE_COLUMNS,
                "feature_complete",
                "future_path_drawdown",
                "label_available_date",
                "label_complete",
                "q20_event",
                "training_eligible",
                "nq_position",
            ]
        )
        result.attrs.update(
            model_version=MODEL_VERSION,
            horizon_trading_days=HORIZON,
            drawdown_threshold=DRAWDOWN_THRESHOLD,
            as_of=None,
        )
        return result
    cutoff = pd.Timestamp(as_of) if as_of is not None else frame[date_column].max()
    if cutoff.tzinfo is not None:
        cutoff = cutoff.tz_localize(None)
    frame = frame.loc[frame[date_column].le(cutoff)].reset_index(drop=True)
    matrix, feature_complete = exact_feature_matrix(frame)
    close = pd.to_numeric(frame[close_column], errors="coerce")
    frame.loc[:, FEATURE_COLUMNS] = matrix
    frame["nq_position"] = np.arange(len(frame), dtype=int)
    frame["feature_complete"] = feature_complete
    frame["future_path_drawdown"] = forward_path_drawdown(close, horizon=HORIZON)
    frame["label_available_date"] = frame[date_column].shift(-HORIZON)
    frame["label_complete"] = (
        frame["future_path_drawdown"].notna()
        & frame["label_available_date"].notna()
        & frame["label_available_date"].le(cutoff)
    )
    frame.loc[~frame["label_complete"], "future_path_drawdown"] = np.nan
    event = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    event.loc[frame["label_complete"]] = (
        frame.loc[frame["label_complete"], "future_path_drawdown"]
        <= DRAWDOWN_THRESHOLD
    ).astype(int)
    frame["q20_event"] = event
    frame["training_eligible"] = frame["feature_complete"] & frame["label_complete"]
    result = frame.rename(
        columns={date_column: "date", close_column: "nq_close"}
    ).reset_index(drop=True)
    result.attrs.update(
        model_version=MODEL_VERSION,
        horizon_trading_days=HORIZON,
        drawdown_threshold=DRAWDOWN_THRESHOLD,
        as_of=cutoff.normalize().date().isoformat(),
    )
    return result


def independent_event_count(
    events: pd.Series,
    positions: pd.Series,
    *,
    separation: int = HORIZON,
) -> int:
    """Count event clusters separated by more than ``separation`` NQ rows."""
    if separation < 0:
        raise ValueError("separation cannot be negative")
    event_values = pd.to_numeric(events, errors="coerce").fillna(0).astype(bool)
    event_positions = pd.to_numeric(positions[event_values], errors="raise").astype(int)
    count = 0
    previous_event: int | None = None
    for position in event_positions.sort_values():
        current = int(position)
        if previous_event is None or current - previous_event > separation:
            count += 1
        previous_event = current
    return count


def assess_q20_readiness(frame: pd.DataFrame) -> dict[str, object]:
    """Evaluate only the frozen evidence gate; this function never fits a model."""
    required = {
        "date",
        "nq_close",
        *FEATURE_COLUMNS,
        "feature_complete",
        "future_path_drawdown",
        "label_available_date",
        "label_complete",
        "training_eligible",
        "q20_event",
        "nq_position",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("missing q20 readiness columns: " + ", ".join(missing))
    expected_contract = {
        "model_version": MODEL_VERSION,
        "horizon_trading_days": HORIZON,
        "drawdown_threshold": DRAWDOWN_THRESHOLD,
    }
    if any(frame.attrs.get(key) != value for key, value in expected_contract.items()):
        raise ValueError("q20 frame does not match the frozen model contract")
    cutoff = pd.Timestamp(frame.attrs.get("as_of"))
    if pd.isna(cutoff):
        raise ValueError("q20 frame must record a valid as_of cutoff")
    dates = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("q20 dates must be unique NQ calendar dates in order")
    if dates.gt(cutoff.normalize()).any():
        raise ValueError("q20 frame cannot contain dates after its as_of cutoff")
    positions = pd.to_numeric(frame["nq_position"], errors="raise").to_numpy(dtype=int)
    if not np.array_equal(positions, np.arange(len(frame), dtype=int)):
        raise ValueError("nq_position must be the contiguous NQ decision calendar")

    _, expected_feature_complete = exact_feature_matrix(frame)
    feature_complete = frame["feature_complete"].fillna(False).astype(bool)
    if not feature_complete.equals(expected_feature_complete):
        raise ValueError("feature_complete does not match the frozen feature vector")
    close = pd.to_numeric(frame["nq_close"], errors="coerce")
    expected_drawdown = forward_path_drawdown(close, horizon=HORIZON)
    expected_available_date = dates.shift(-HORIZON)
    drawdown = pd.to_numeric(frame["future_path_drawdown"], errors="coerce")
    if not np.allclose(
        drawdown.to_numpy(dtype=float),
        expected_drawdown.to_numpy(dtype=float),
        equal_nan=True,
    ):
        raise ValueError("future_path_drawdown does not match the frozen NQ path")
    available_date = pd.to_datetime(frame["label_available_date"], errors="coerce")
    if not available_date.equals(expected_available_date):
        raise ValueError("label_available_date does not match the frozen NQ calendar")
    expected_label_complete = (
        expected_drawdown.notna()
        & expected_available_date.notna()
        & expected_available_date.le(cutoff)
    )
    label_complete = frame["label_complete"].fillna(False).astype(bool)
    if not label_complete.equals(expected_label_complete):
        raise ValueError("label_complete does not match the frozen q20 embargo")
    expected_event = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    expected_event.loc[label_complete] = (
        expected_drawdown.loc[label_complete] <= DRAWDOWN_THRESHOLD
    ).astype(int)
    supplied_event = pd.to_numeric(frame["q20_event"], errors="coerce").astype("Int64")
    if not supplied_event.equals(expected_event):
        raise ValueError("q20_event does not match the frozen -5% label")
    eligible = feature_complete & label_complete
    supplied = frame["training_eligible"].fillna(False).astype(bool)
    if not supplied.equals(eligible):
        raise ValueError("training_eligible must equal feature_complete & label_complete")
    usable = frame.loc[eligible]
    complete_labels = int(len(usable))
    episodes = independent_event_count(
        usable["q20_event"], usable["nq_position"], separation=HORIZON
    )
    fit_allowed = (
        complete_labels >= MIN_COMPLETE_LABELS
        and episodes >= MIN_INDEPENDENT_EVENTS
    )
    return {
        "model_version": MODEL_VERSION,
        "status": READY_FOR_WALK_FORWARD_FIT if fit_allowed else INSUFFICIENT_EVIDENCE,
        "probability": None,
        "fit_allowed": fit_allowed,
        "feature_columns": list(FEATURE_COLUMNS),
        "feature_rows": int(frame["feature_complete"].fillna(False).sum()),
        "label_rows": int(frame["label_complete"].fillna(False).sum()),
        "complete_feature_labels": complete_labels,
        "independent_events": episodes,
        "minimum_complete_labels": MIN_COMPLETE_LABELS,
        "minimum_independent_events": MIN_INDEPENDENT_EVENTS,
        "horizon_trading_days": HORIZON,
        "drawdown_threshold": DRAWDOWN_THRESHOLD,
    }


def q20_readiness_metadata(frame: pd.DataFrame) -> dict[str, object]:
    """Return the exact aggregate-only readiness contract used by SHADOW r4."""
    readiness = assess_q20_readiness(frame)
    return {field: readiness[field] for field in READINESS_METADATA_FIELDS}


def validate_q20_readiness_metadata(value: object) -> dict[str, object]:
    """Validate one nested DB/web readiness object without accepting extensions."""
    if not isinstance(value, dict) or set(value) != set(READINESS_METADATA_FIELDS):
        raise ValueError("q20 readiness metadata must match the exact frozen contract")
    if value["model_version"] != MODEL_VERSION:
        raise ValueError("q20 readiness model version mismatch")
    if value["probability"] is not None:
        raise ValueError("q20 probability must remain null before a fitted model exists")
    if value["horizon_trading_days"] != HORIZON:
        raise ValueError("q20 horizon mismatch")
    if value["drawdown_threshold"] != DRAWDOWN_THRESHOLD:
        raise ValueError("q20 drawdown threshold mismatch")
    if value["minimum_complete_labels"] != MIN_COMPLETE_LABELS:
        raise ValueError("q20 minimum complete-label gate mismatch")
    if value["minimum_independent_events"] != MIN_INDEPENDENT_EVENTS:
        raise ValueError("q20 minimum independent-event gate mismatch")
    if type(value["fit_allowed"]) is not bool:
        raise ValueError("q20 fit_allowed must be boolean")
    counts = (
        "feature_rows",
        "complete_feature_labels",
        "independent_events",
    )
    if any(type(value[field]) is not int or value[field] < 0 for field in counts):
        raise ValueError("q20 readiness counts must be non-negative integers")
    if value["complete_feature_labels"] > value["feature_rows"]:
        raise ValueError("q20 complete labels cannot exceed exact feature rows")
    if value["independent_events"] > value["complete_feature_labels"]:
        raise ValueError("q20 independent events cannot exceed complete labels")
    expected_fit_allowed = (
        value["complete_feature_labels"] >= MIN_COMPLETE_LABELS
        and value["independent_events"] >= MIN_INDEPENDENT_EVENTS
    )
    if value["fit_allowed"] != expected_fit_allowed:
        raise ValueError("q20 fit_allowed does not match the frozen evidence gate")
    expected_status = (
        READY_FOR_WALK_FORWARD_FIT
        if expected_fit_allowed
        else INSUFFICIENT_EVIDENCE
    )
    if value["status"] != expected_status:
        raise ValueError("q20 status and fit_allowed disagree")
    return {field: value[field] for field in READINESS_METADATA_FIELDS}


def build_q20_readiness_history(
    daily: pd.DataFrame,
    *,
    decision_dates: object | None = None,
    date_column: str = "date",
    close_column: str = "nq_close",
) -> pd.DataFrame:
    """Build causal readiness metadata independently for each requested NQ date."""
    normalised = _normalise_daily(
        daily, date_column=date_column, close_column=close_column
    )
    if normalised.empty:
        return pd.DataFrame(columns=["q20"], index=pd.DatetimeIndex([], name="date"))

    calendar = pd.DatetimeIndex(normalised[date_column], name="date")
    if decision_dates is None:
        targets = calendar
    else:
        targets = pd.DatetimeIndex(pd.to_datetime(decision_dates, errors="raise"))
        if targets.tz is not None:
            targets = targets.tz_localize(None)
        targets = targets.normalize()
        if targets.has_duplicates or not targets.is_monotonic_increasing:
            raise ValueError("q20 decision dates must be unique and increasing")
        unknown = targets.difference(calendar)
        if not unknown.empty:
            raise ValueError("q20 decision dates must belong to the NQ calendar")
    if targets.empty:
        return pd.DataFrame(columns=["q20"], index=pd.DatetimeIndex([], name="date"))

    rows: list[dict[str, object]] = []
    for cutoff in targets:
        frame = build_q20_research_frame(
            normalised,
            as_of=cutoff,
            date_column=date_column,
            close_column=close_column,
        )
        metadata = validate_q20_readiness_metadata(q20_readiness_metadata(frame))
        rows.append({"date": cutoff, "q20": metadata})
    return pd.DataFrame(rows).set_index("date")


def require_fit_ready(frame: pd.DataFrame) -> dict[str, object]:
    """Reassess a research frame and fail closed before any future fitter."""
    readiness = assess_q20_readiness(frame)
    if readiness.get("status") != READY_FOR_WALK_FORWARD_FIT or not readiness.get(
        "fit_allowed", False
    ):
        raise InsufficientEvidenceError(
            "q20 fitting is blocked until 252 exact complete labels and "
            "20 independent events are available"
        )
    return readiness
