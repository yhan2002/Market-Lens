"""Causal 2026+ validation primitives for precomputed RORO risk scores."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


MIN_OUTPUT_START = pd.Timestamp("2026-01-02")
RORO_TXT_ORIGINAL = "roro_txt_original"
REFERENCE_ONLY = "REFERENCE_ONLY"
SCORE_DIAGNOSTIC_WINDOWS = (63, 126, 252)
RISK_REGIMES = ("GREEN", "YELLOW", "RED")
FROZEN_GATES = {
    "minimum_observations": 1_000,
    "minimum_independent_episodes": 20,
    "auc_ci_low": 0.50,
    "top_quintile_lift": 1.25,
    "quintile_monotonicity": 0.50,
    "paired_auc_noninferiority_margin": -0.02,
}


def _categorical_runs(values: pd.Series) -> list[tuple[object, int]]:
    runs: list[tuple[object, int]] = []
    current: object | None = None
    length = 0
    for value in values:
        if pd.isna(value):
            if length:
                runs.append((current, length))
            current, length = None, 0
        elif length and value == current:
            length += 1
        else:
            if length:
                runs.append((current, length))
            current, length = value, 1
    if length:
        runs.append((current, length))
    return runs


def score_path_diagnostics(scores: Sequence[object]) -> dict[str, object]:
    """Summarise integer-score lock and G/Y/R persistence over frozen windows."""
    source = pd.Series(scores, copy=False)
    numeric = pd.to_numeric(source, errors="coerce")
    if (source.notna() & numeric.isna()).any() or np.isinf(numeric.dropna()).any():
        raise ValueError("scores must be finite numeric values or missing")

    windows: dict[str, object] = {}
    for window in SCORE_DIAGNOSTIC_WINDOWS:
        recent = numeric.tail(window).reset_index(drop=True)
        present = recent.notna()
        integer = pd.Series(pd.NA, index=recent.index, dtype="Int64")
        integer.loc[present] = np.floor(recent.loc[present] + 0.5).astype(int)
        available = integer.dropna().astype(int)
        score_runs = _categorical_runs(integer)
        adjacent = integer.notna() & integer.shift().notna()

        regimes = pd.Series(pd.NA, index=recent.index, dtype="string")
        regimes.loc[present & integer.le(33)] = "GREEN"
        regimes.loc[present & integer.gt(33) & integer.le(66)] = "YELLOW"
        regimes.loc[present & integer.gt(66)] = "RED"
        regime_runs = _categorical_runs(regimes)
        regime_adjacent = regimes.notna() & regimes.shift().notna()
        occupancy = regimes.value_counts(normalize=True)

        windows[str(window)] = {
            "window_rows": int(len(recent)),
            "available_scores": int(len(available)),
            "mode_share": float(available.value_counts(normalize=True).iloc[0])
            if len(available)
            else None,
            "unique_scores": int(available.nunique()),
            "score_changes": int(
                (integer.ne(integer.shift()) & adjacent).fillna(False).sum()
            ),
            "longest_run_days": max((length for _, length in score_runs), default=0),
            "terminal_run_days": score_runs[-1][1]
            if len(integer) and integer.notna().iloc[-1] and score_runs
            else 0,
            "regime_occupancy": {
                regime: float(occupancy.get(regime, 0.0)) if len(available) else None
                for regime in RISK_REGIMES
            },
            "regime_runs": {
                regime: {
                    "median_run_days": float(np.median(lengths)) if lengths else None,
                    "max_run_days": max(lengths, default=0),
                }
                for regime in RISK_REGIMES
                for lengths in [
                    [length for value, length in regime_runs if value == regime]
                ]
            },
            "regime_transitions": int(
                (regimes.ne(regimes.shift()) & regime_adjacent).fillna(False).sum()
            ),
        }
    return {
        "rounding": "half-up to nearest integer",
        "regime_boundaries": "GREEN 0-33; YELLOW 34-66; RED 67-100",
        "windows": windows,
    }


def coverage_status_summary(
    scores: Sequence[object], data_status: Sequence[object] | None = None
) -> dict[str, object]:
    """Report score coverage and observed data-status counts without imputation."""
    source = pd.Series(scores, copy=False).reset_index(drop=True)
    numeric = pd.to_numeric(source, errors="coerce")
    if (source.notna() & numeric.isna()).any() or np.isinf(numeric.dropna()).any():
        raise ValueError("scores must be finite numeric values or missing")
    rows = len(source)
    score_rows = int(numeric.notna().sum())
    result: dict[str, object] = {
        "rows": rows,
        "score_rows": score_rows,
        "no_score_rows": rows - score_rows,
        "coverage_share": score_rows / rows if rows else None,
        "no_score_share": (rows - score_rows) / rows if rows else None,
        "data_status": "AVAILABLE" if data_status is not None else "DATA_BLOCKED",
        "data_status_counts": {},
        "data_status_shares": {},
    }
    if data_status is None:
        result["reason"] = "data_status is unavailable"
        return result
    statuses = pd.Series(data_status, copy=False).reset_index(drop=True)
    if len(statuses) != rows:
        raise ValueError("scores and data_status must have equal length")
    labels = statuses.fillna("MISSING").astype(str)
    counts = labels.value_counts()
    result["data_status_counts"] = {
        str(name): int(value) for name, value in counts.items()
    }
    result["data_status_shares"] = {
        str(name): float(value / rows) if rows else None
        for name, value in counts.items()
    }
    return result


def yearly_auc_summary(
    frame: pd.DataFrame, score_columns: Sequence[str]
) -> dict[str, object]:
    """Compute descriptive calendar-year AUC on an already-common labelled sample."""
    required = ["date", "drawdown_event", *score_columns]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError("missing validation columns: " + ", ".join(missing))
    source = frame.loc[:, list(dict.fromkeys(required))].copy()
    source["date"] = pd.to_datetime(source["date"], errors="raise")
    years: dict[str, object] = {}
    for year, group in source.groupby(source["date"].dt.year, sort=True):
        target = pd.to_numeric(group["drawdown_event"], errors="coerce")
        if target.isna().any() or not target.isin([0, 1]).all():
            raise ValueError("drawdown_event must contain only complete binary labels")
        target_values = target.to_numpy(dtype=int)
        enough_classes = np.unique(target_values).size == 2
        models: dict[str, object] = {}
        for column in score_columns:
            score = pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(score).all():
                raise ValueError("yearly AUC requires finite common-sample scores")
            models[column] = {
                "status": "AVAILABLE"
                if enough_classes
                else "INSUFFICIENT_CLASS_VARIATION",
                "roc_auc": float(roc_auc_score(target_values, score))
                if enough_classes
                else None,
            }
        years[str(int(year))] = {
            "rows": int(len(group)),
            "events": int(target_values.sum()),
            "non_events": int(len(target_values) - target_values.sum()),
            "models": models,
        }
    return {"descriptive_only": True, "years": years}


def red_alert_onset_diagnostics(
    daily: pd.DataFrame,
    *,
    score_column: str = "roro_score",
    date_column: str = "date",
    price_column: str = "nq_close",
    output_start: object = MIN_OUTPUT_START,
    lookback: int = 252,
    horizon: int = 20,
) -> dict[str, object]:
    """Match RED-entry alert clusters to causal NQ drawdown-onset episodes."""
    if lookback < 1 or horizon < 1:
        raise ValueError("lookback and horizon must be positive")
    start = _validate_output_start(output_start)
    frame = _normalise_daily(
        daily, date_column=date_column, required=[price_column, score_column]
    )
    close = pd.to_numeric(frame[price_column], errors="coerce")
    if not np.isfinite(close).all() or (close <= 0).any():
        raise ValueError("NQ prices must be finite and positive")
    raw_score = frame[score_column]
    score = pd.to_numeric(raw_score, errors="coerce")
    observed_scores = score.dropna()
    if (
        (raw_score.notna() & score.isna()).any()
        or np.isinf(observed_scores).any()
        or (observed_scores < 0).any()
        or (observed_scores > 100).any()
        or not np.equal(observed_scores, np.floor(observed_scores)).all()
    ):
        raise ValueError("RED alert scores must be missing or integers from 0 to 100")
    peak = close.rolling(lookback, min_periods=lookback).max()
    drawdown = close / peak - 1.0
    onset = np.zeros(len(frame), dtype=bool)
    armed = False
    for position, value in enumerate(drawdown.to_numpy(dtype=float)):
        if not np.isfinite(value):
            continue
        if value >= -0.03:
            armed = True
        if armed and value <= -0.05:
            onset[position] = True
            armed = False

    in_scope = frame[date_column].ge(start)
    previous = score.shift(1)
    alert = score.ge(67) & (previous.lt(67) | previous.isna()) & in_scope
    alert_positions = np.flatnonzero(alert.to_numpy()).astype(int).tolist()
    onset_positions = np.flatnonzero(onset & in_scope.to_numpy()).astype(int).tolist()
    if not np.isfinite(drawdown.loc[in_scope]).any():
        return {
            "status": "DATA_BLOCKED",
            "reason": f"fewer than {lookback} NQ closes before the evaluation window",
            "lookback_sessions": lookback,
            "alert_horizon_sessions": horizon,
        }

    clusters: list[list[int]] = []
    for position in alert_positions:
        if not clusters or position - clusters[-1][-1] > horizon:
            clusters.append([position])
        else:
            clusters[-1].append(position)

    onset_details: list[dict[str, object]] = []
    matched_onsets: set[int] = set()
    leads: list[int] = []
    for event in onset_positions:
        valid_alerts = [
            alert_position
            for alert_position in alert_positions
            if 1 <= event - alert_position <= horizon
        ]
        lead = event - min(valid_alerts) if valid_alerts else None
        if lead is not None:
            matched_onsets.add(event)
            leads.append(int(lead))
        onset_details.append(
            {
                "onset_date": frame.at[event, date_column].date().isoformat(),
                "matched": bool(valid_alerts),
                "first_valid_alert_date": frame.at[
                    min(valid_alerts), date_column
                ].date().isoformat()
                if valid_alerts
                else None,
                "lead_sessions": int(lead) if lead is not None else None,
            }
        )

    cluster_details: list[dict[str, object]] = []
    matched_clusters = false_clusters = pending_clusters = 0
    last_observation = len(frame) - 1
    for cluster in clusters:
        matching_onsets = [
            event
            for event in onset_positions
            if any(1 <= event - alert_position <= horizon for alert_position in cluster)
        ]
        if matching_onsets:
            status = "MATCHED"
            matched_clusters += 1
        elif last_observation - cluster[-1] >= horizon:
            status = "FALSE"
            false_clusters += 1
        else:
            status = "PENDING"
            pending_clusters += 1
        cluster_details.append(
            {
                "first_alert_date": frame.at[cluster[0], date_column].date().isoformat(),
                "last_alert_date": frame.at[cluster[-1], date_column].date().isoformat(),
                "alert_count": len(cluster),
                "status": status,
                "matched_onset_dates": [
                    frame.at[event, date_column].date().isoformat()
                    for event in matching_onsets
                ],
            }
        )
    evaluated_clusters = matched_clusters + false_clusters
    return {
        "status": "AVAILABLE",
        "lookback_sessions": lookback,
        "rearm_drawdown": -0.03,
        "onset_drawdown": -0.05,
        "red_entry_threshold": 67,
        "alert_horizon_sessions": horizon,
        "onsets": len(onset_positions),
        "matched_onsets": len(matched_onsets),
        "onset_recall": len(matched_onsets) / len(onset_positions)
        if onset_positions
        else None,
        "alert_count": len(alert_positions),
        "alert_clusters": len(clusters),
        "matched_clusters": matched_clusters,
        "false_alert_clusters": false_clusters,
        "pending_clusters": pending_clusters,
        "cluster_precision": matched_clusters / evaluated_clusters
        if evaluated_clusters
        else None,
        "median_lead_sessions": float(np.median(leads)) if leads else None,
        "onset_details": onset_details,
        "cluster_details": cluster_details,
    }


def reentry_relapse_diagnostics(
    daily: pd.DataFrame,
    *,
    policy_column: str = "policy",
    date_column: str = "date",
    price_column: str = "nq_close",
    output_start: object = MIN_OUTPUT_START,
    horizon: int = 10,
) -> dict[str, object]:
    """Evaluate actual NQ close-path relapse after nested REENTRY_DIRECT decisions."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    start = _validate_output_start(output_start)
    if policy_column not in daily:
        return {
            "status": "DATA_BLOCKED",
            "reason": "nested policy.phase is unavailable",
            "horizon_sessions": horizon,
        }
    frame = _normalise_daily(
        daily, date_column=date_column, required=[price_column, policy_column]
    )
    close = pd.to_numeric(frame[price_column], errors="coerce")
    if not np.isfinite(close).all() or (close <= 0).any():
        raise ValueError("NQ prices must be finite and positive")
    policies = frame[policy_column]
    if not policies.notna().any():
        return {
            "status": "DATA_BLOCKED",
            "reason": "nested policy.phase is unavailable",
            "horizon_sessions": horizon,
        }
    phases: list[object] = []
    for value in policies:
        if value is None or value is pd.NA or (
            isinstance(value, (float, np.floating)) and np.isnan(value)
        ):
            phases.append(None)
        elif not isinstance(value, dict) or not isinstance(value.get("phase"), str):
            raise ValueError("policy observations must contain a string phase")
        else:
            phases.append(value["phase"])

    decisions = [
        position
        for position, phase in enumerate(phases)
        if phase == "REENTRY_DIRECT" and frame.at[position, date_column] >= start
    ]
    details: list[dict[str, object]] = []
    failures = pending = 0
    for position in decisions:
        decision_date = frame.at[position, date_column].date().isoformat()
        if position + horizon >= len(frame):
            pending += 1
            details.append(
                {
                    "decision_date": decision_date,
                    "status": "PENDING",
                    "minimum_forward_return": None,
                }
            )
            continue
        path = close.iloc[position + 1 : position + horizon + 1]
        minimum = float((path / close.iloc[position] - 1.0).min())
        failed = minimum <= -0.03
        failures += int(failed)
        details.append(
            {
                "decision_date": decision_date,
                "status": "FAIL" if failed else "PASS",
                "minimum_forward_return": minimum,
            }
        )
    evaluated = len(decisions) - pending
    return {
        "status": "AVAILABLE",
        "phase": "REENTRY_DIRECT",
        "horizon_sessions": horizon,
        "failure_drawdown": -0.03,
        "events": len(decisions),
        "evaluated_events": evaluated,
        "failures": failures,
        "failure_rate": failures / evaluated if evaluated else None,
        "tail_pending_events": pending,
        "event_details": details,
    }


def _performance_metrics(returns: pd.Series) -> dict[str, float | None]:
    values = pd.to_numeric(returns, errors="coerce").to_numpy(dtype=float)
    if not len(values):
        return {"cagr": None, "sharpe": None, "max_drawdown": None}
    if not np.isfinite(values).all() or (values <= -1).any():
        raise ValueError("returns must be finite and greater than -100%")
    wealth = np.concatenate(([1.0], np.cumprod(1.0 + values)))
    if not np.isfinite(wealth).all():
        raise ValueError("wealth path must be finite")
    drawdown = wealth / np.maximum.accumulate(wealth) - 1.0
    standard_deviation = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    metrics = {
        "cagr": float(wealth[-1] ** (252 / len(values)) - 1.0),
        "sharpe": float(np.mean(values) / standard_deviation * math.sqrt(252))
        if standard_deviation > 0
        else 0.0,
        "max_drawdown": float(drawdown.min()),
    }
    if not all(np.isfinite(value) for value in metrics.values()):
        raise ValueError("performance metrics must be finite")
    return metrics


def _timestamp(value: object) -> pd.Timestamp:
    result = pd.Timestamp(value)
    return result.tz_localize(None) if result.tzinfo is not None else result


def _validate_output_start(value: object) -> pd.Timestamp:
    start = _timestamp(value)
    if start < MIN_OUTPUT_START:
        raise ValueError(f"output_start must be on or after {MIN_OUTPUT_START.date()}")
    return start


def _normalise_daily(
    daily: pd.DataFrame, *, date_column: str, required: Iterable[str]
) -> pd.DataFrame:
    columns = [date_column, *required]
    missing = [column for column in columns if column not in daily]
    if missing:
        raise ValueError("missing required columns: " + ", ".join(missing))
    result = daily.loc[:, list(dict.fromkeys(columns))].copy()
    result[date_column] = pd.to_datetime(result[date_column], errors="raise")
    if result[date_column].dt.tz is not None:
        result[date_column] = result[date_column].dt.tz_convert(None)
    if result[date_column].duplicated().any():
        raise ValueError("daily dates must be unique")
    return result.sort_values(date_column).reset_index(drop=True)


def forward_path_drawdown(prices: Sequence[float], horizon: int = 20) -> np.ndarray:
    """Return min(t+1..t+horizon)/price_t-1; incomplete tails stay NaN."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    values = np.asarray(prices, dtype=float)
    result = np.full(len(values), np.nan)
    for position in range(len(values) - horizon):
        base = values[position]
        future = values[position + 1 : position + horizon + 1]
        if np.isfinite(base) and base > 0 and np.isfinite(future).all():
            result[position] = float(np.min(future / base - 1.0))
    return result


def build_causal_validation_frame(
    daily: pd.DataFrame,
    *,
    score_columns: Sequence[str],
    date_column: str = "date",
    price_column: str = "nq_close",
    output_start: object = MIN_OUTPUT_START,
    as_of: object | None = None,
    horizon: int = 20,
    drawdown_threshold: float = -0.05,
) -> pd.DataFrame:
    """Build labels whose full forward path is observable by ``as_of``."""
    start = _validate_output_start(output_start)
    score_columns = list(dict.fromkeys(score_columns))
    frame = _normalise_daily(
        daily,
        date_column=date_column,
        required=[price_column, *score_columns],
    )
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "nq_close",
                *score_columns,
                "future_path_drawdown",
                "label_available_date",
                "label_complete",
                "drawdown_event",
                "_nq_position",
            ]
        )

    cutoff = _timestamp(as_of) if as_of is not None else frame[date_column].max()
    close = pd.to_numeric(frame[price_column], errors="coerce")
    if (close.dropna() <= 0).any():
        raise ValueError("NQ prices must be positive")
    frame["_nq_position"] = np.arange(len(frame))
    frame["future_path_drawdown"] = forward_path_drawdown(close, horizon)
    frame["label_available_date"] = frame[date_column].shift(-horizon)
    complete = (
        frame["future_path_drawdown"].notna()
        & frame["label_available_date"].notna()
        & (frame["label_available_date"] <= cutoff)
    )
    frame["label_complete"] = complete
    frame.loc[~complete, "future_path_drawdown"] = np.nan
    event = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    event.loc[complete] = (
        frame.loc[complete, "future_path_drawdown"] <= drawdown_threshold
    ).astype(int)
    frame["drawdown_event"] = event
    frame = frame.loc[(frame[date_column] >= start) & (frame[date_column] <= cutoff)].copy()
    rename = {date_column: "date", price_column: "nq_close"}
    return frame.rename(columns=rename).reset_index(drop=True)


def common_labelled_frame(
    frame: pd.DataFrame, score_columns: Sequence[str]
) -> pd.DataFrame:
    """Return the exact complete-label intersection for every ranked score."""
    columns = list(dict.fromkeys(score_columns))
    missing = [column for column in ["label_complete", *columns] if column not in frame]
    if missing:
        raise ValueError("missing validation columns: " + ", ".join(missing))
    mask = frame["label_complete"].fillna(False)
    if columns:
        numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
        mask &= numeric.notna().all(axis=1) & np.isfinite(numeric).all(axis=1)
    result = frame.loc[mask].copy()
    if columns:
        result.loc[:, columns] = numeric.loc[mask, columns]
    return result.reset_index(drop=True)


def _independent_episode_count(
    target: Sequence[int], positions: Sequence[int], *, separation: int
) -> int:
    event_positions = np.asarray(positions)[np.asarray(target, dtype=bool)]
    count = 0
    last_event: int | None = None
    for position in event_positions:
        position = int(position)
        if last_event is None or position - last_event > separation:
            count += 1
        last_event = position
    return count


def _moving_block_indices(
    size: int, *, block_days: int, rng: np.random.Generator
) -> np.ndarray:
    if size <= block_days:
        return np.arange(size)
    starts = rng.integers(
        0, size - block_days + 1, size=int(np.ceil(size / block_days))
    )
    return np.concatenate(
        [np.arange(start, start + block_days) for start in starts]
    )[:size]


def _ranking_metrics(
    frame: pd.DataFrame,
    score_column: str,
    *,
    horizon: int,
    bootstrap_samples: int,
    block_days: int,
    random_state: int,
) -> dict[str, object]:
    target = frame["drawdown_event"].to_numpy(dtype=int)
    score = pd.to_numeric(frame[score_column], errors="coerce").to_numpy(dtype=float)
    metrics: dict[str, object] = {
        "n": int(len(frame)),
        "event_rate": float(target.mean()) if len(target) else None,
        "independent_episodes": _independent_episode_count(
            target,
            frame.get("_nq_position", pd.Series(range(len(frame)))).to_numpy(),
            separation=horizon,
        )
        if len(target)
        else 0,
        "roc_auc": None,
        "auc_ci_low": None,
        "auc_ci_high": None,
        "average_precision": None,
        "top_quintile_event_rate": None,
        "top_quintile_lift": None,
        "quintile_monotonicity": None,
    }
    if len(target) == 0 or np.unique(target).size < 2:
        return metrics

    metrics["roc_auc"] = float(roc_auc_score(target, score))
    metrics["average_precision"] = float(average_precision_score(target, score))
    base_rate = float(target.mean())
    cutoff = float(np.quantile(score, 0.8))
    top_rate = float(target[score >= cutoff].mean())
    metrics["top_quintile_event_rate"] = top_rate
    metrics["top_quintile_lift"] = top_rate / base_rate

    buckets = pd.qcut(pd.Series(score), 5, labels=False, duplicates="drop")
    bucket_rates = pd.DataFrame({"bucket": buckets, "event": target}).groupby(
        "bucket", observed=True
    )["event"].mean()
    if len(bucket_rates) >= 2:
        metrics["quintile_monotonicity"] = float(
            pd.Series(range(len(bucket_rates))).corr(
                bucket_rates.reset_index(drop=True), method="spearman"
            )
        )

    rng = np.random.default_rng(random_state)
    aucs: list[float] = []
    for _ in range(bootstrap_samples):
        sampled = _moving_block_indices(len(target), block_days=block_days, rng=rng)
        if np.unique(target[sampled]).size == 2:
            aucs.append(float(roc_auc_score(target[sampled], score[sampled])))
    if aucs:
        metrics["auc_ci_low"], metrics["auc_ci_high"] = map(
            float, np.quantile(aucs, [0.025, 0.975])
        )
    return metrics


def _paired_auc(
    frame: pd.DataFrame,
    candidate_column: str,
    comparator_column: str,
    *,
    bootstrap_samples: int,
    block_days: int,
    random_state: int,
) -> dict[str, object]:
    target = frame["drawdown_event"].to_numpy(dtype=int)
    candidate = frame[candidate_column].to_numpy(dtype=float)
    comparator = frame[comparator_column].to_numpy(dtype=float)
    result: dict[str, object] = {
        "n": int(len(frame)),
        "candidate_auc": None,
        "comparator_auc": None,
        "auc_delta": None,
        "auc_delta_ci_low": None,
        "auc_delta_ci_high": None,
    }
    if len(target) == 0 or np.unique(target).size < 2:
        return result
    result["candidate_auc"] = float(roc_auc_score(target, candidate))
    result["comparator_auc"] = float(roc_auc_score(target, comparator))
    result["auc_delta"] = float(result["candidate_auc"] - result["comparator_auc"])
    rng = np.random.default_rng(random_state)
    deltas: list[float] = []
    for _ in range(bootstrap_samples):
        sampled = _moving_block_indices(len(target), block_days=block_days, rng=rng)
        if np.unique(target[sampled]).size == 2:
            deltas.append(
                float(
                    roc_auc_score(target[sampled], candidate[sampled])
                    - roc_auc_score(target[sampled], comparator[sampled])
                )
            )
    if deltas:
        result["auc_delta_ci_low"], result["auc_delta_ci_high"] = map(
            float, np.quantile(deltas, [0.025, 0.975])
        )
    return result


def evaluate_exposure_policy(
    daily: pd.DataFrame,
    *,
    long_column: str = "recommended_long_pct",
    date_column: str = "date",
    price_column: str = "nq_close",
    output_start: object = MIN_OUTPUT_START,
    initial_long_pct: float = 100.0,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Apply each close-of-day Long signal to the next NQ bar at a fixed 10bp."""
    start = _validate_output_start(output_start)
    frame = _normalise_daily(
        daily, date_column=date_column, required=[price_column, long_column]
    )
    close = pd.to_numeric(frame[price_column], errors="coerce")
    long_pct = pd.to_numeric(frame[long_column], errors="coerce")
    if ((long_pct.dropna() < 0) | (long_pct.dropna() > 100)).any():
        raise ValueError("Long percentage must be between 0 and 100")
    eligible_signal = long_pct.where(frame[date_column] >= start)
    frame["signal_date"] = frame[date_column].where(eligible_signal.notna()).shift(1)
    frame["signal_long_pct"] = eligible_signal
    frame["applied_long_pct"] = eligible_signal.shift(1)
    frame["nq_return"] = close.pct_change(fill_method=None)
    valid = frame.loc[
        (frame[date_column] >= start)
        & frame["applied_long_pct"].notna()
        & frame["nq_return"].notna()
    ].copy()
    valid["applied_exposure"] = valid["applied_long_pct"] / 100.0
    valid["turnover"] = valid["applied_exposure"].diff().abs()
    if not valid.empty:
        valid.iloc[0, valid.columns.get_loc("turnover")] = abs(
            valid.iloc[0]["applied_exposure"] - initial_long_pct / 100.0
        )
    valid["gross_return"] = valid["applied_exposure"] * valid["nq_return"]
    valid["transaction_cost_10bp"] = valid["turnover"] * 10.0 / 10_000.0
    valid["net_10bp_return"] = valid["gross_return"] - valid["transaction_cost_10bp"]
    net_metrics = _performance_metrics(valid["net_10bp_return"])
    benchmark_metrics = _performance_metrics(valid["nq_return"])
    valid["gross_wealth"] = (1.0 + valid["gross_return"]).cumprod()
    valid["net_10bp_wealth"] = (1.0 + valid["net_10bp_return"]).cumprod()
    benchmark_mdd = (
        abs(benchmark_metrics["max_drawdown"])
        if benchmark_metrics["max_drawdown"] is not None
        else None
    )

    def total_return(column: str) -> float | None:
        return float(valid[column].iloc[-1] - 1.0) if not valid.empty else None

    summary = {
        "execution_lag_bars": 1,
        "transaction_cost_bps": 10,
        "initial_wealth": 1.0,
        "rows": int(len(valid)),
        "start": valid[date_column].min().date().isoformat() if not valid.empty else None,
        "end": valid[date_column].max().date().isoformat() if not valid.empty else None,
        "gross_total_return": total_return("gross_wealth"),
        "net_10bp_total_return": total_return("net_10bp_wealth"),
        "net_10bp_cagr": net_metrics["cagr"],
        "net_10bp_sharpe": net_metrics["sharpe"],
        "net_10bp_max_drawdown": net_metrics["max_drawdown"],
        "nq_benchmark_cagr": benchmark_metrics["cagr"],
        "nq_benchmark_sharpe": benchmark_metrics["sharpe"],
        "nq_benchmark_max_drawdown": benchmark_metrics["max_drawdown"],
        "annual_return_drag": (
            float(benchmark_metrics["cagr"] - net_metrics["cagr"])
            if benchmark_metrics["cagr"] is not None
            and net_metrics["cagr"] is not None
            else None
        ),
        "relative_mdd_improvement": (
            float(
                (benchmark_mdd - abs(net_metrics["max_drawdown"]))
                / benchmark_mdd
            )
            if benchmark_mdd
            and net_metrics["max_drawdown"] is not None
            else None
        ),
        "total_turnover": float(valid["turnover"].sum()) if not valid.empty else 0.0,
        "total_cost_rate_10bp": float(valid["transaction_cost_10bp"].sum())
        if not valid.empty
        else 0.0,
    }
    return valid.rename(columns={date_column: "date", price_column: "nq_close"}), summary


def validate_roro(
    daily: pd.DataFrame,
    *,
    candidate_column: str = "roro_score",
    benchmark_columns: Sequence[str] = (),
    reference_only_columns: Sequence[str] = (RORO_TXT_ORIGINAL,),
    long_column: str | None = None,
    output_start: object = MIN_OUTPUT_START,
    as_of: object | None = None,
    horizon: int = 20,
    drawdown_threshold: float = -0.05,
    bootstrap_samples: int = 500,
    block_days: int = 40,
    random_state: int = 42,
) -> dict[str, object]:
    """Validate one RORO candidate and arbitrary risk-score comparators."""
    references = set(reference_only_columns)
    all_benchmarks = list(dict.fromkeys(benchmark_columns))
    ranked_benchmarks = [name for name in all_benchmarks if name not in references]
    ranked_columns = [candidate_column, *ranked_benchmarks]
    score_columns = list(dict.fromkeys([*ranked_columns, *references]))
    score_columns = [column for column in score_columns if column in daily.columns]
    if candidate_column not in score_columns:
        score_columns.insert(0, candidate_column)
    frame = build_causal_validation_frame(
        daily,
        score_columns=score_columns,
        output_start=output_start,
        as_of=as_of,
        horizon=horizon,
        drawdown_threshold=drawdown_threshold,
    )
    common = common_labelled_frame(frame, ranked_columns)
    candidate_score_rows = int(frame[candidate_column].notna().sum())
    target = common["drawdown_event"].to_numpy(dtype=int) if not common.empty else np.array([])
    positions = (
        common["_nq_position"].to_numpy(dtype=int) if not common.empty else np.array([])
    )
    episodes = _independent_episode_count(target, positions, separation=horizon)

    models = {
        column: {
            "role": "CANDIDATE" if column == candidate_column else "COMPARATOR",
            **_ranking_metrics(
                common,
                column,
                horizon=horizon,
                bootstrap_samples=bootstrap_samples,
                block_days=block_days,
                random_state=random_state,
            ),
        }
        for column in ranked_columns
    }
    for column in sorted(references.intersection(daily.columns)):
        models[column] = {
            "role": REFERENCE_ONLY,
            "metrics": None,
            "reason": "Original RORO.txt is non-causal reference material and is excluded from paired ranking.",
        }

    paired = {
        comparator: _paired_auc(
            common,
            candidate_column,
            comparator,
            bootstrap_samples=bootstrap_samples,
            block_days=block_days,
            random_state=random_state,
        )
        for comparator in ranked_benchmarks
    }
    candidate = models[candidate_column]
    gates = {
        "minimum_observations": len(common)
        >= FROZEN_GATES["minimum_observations"],
        "minimum_independent_episodes": episodes
        >= FROZEN_GATES["minimum_independent_episodes"],
        "both_event_classes": np.unique(target).size == 2,
        "auc_ci_above_random": candidate["auc_ci_low"] is not None
        and candidate["auc_ci_low"] > FROZEN_GATES["auc_ci_low"],
        "top_quintile_lift": candidate["top_quintile_lift"] is not None
        and candidate["top_quintile_lift"]
        >= FROZEN_GATES["top_quintile_lift"],
        "quintile_monotonicity": candidate["quintile_monotonicity"] is not None
        and candidate["quintile_monotonicity"]
        >= FROZEN_GATES["quintile_monotonicity"],
        "ranked_comparator_present": bool(ranked_benchmarks),
        "paired_auc_noninferiority": bool(paired)
        and all(
            item["auc_delta_ci_low"] is not None
            and item["auc_delta_ci_low"]
            >= FROZEN_GATES["paired_auc_noninferiority_margin"]
            for item in paired.values()
        ),
    }
    if candidate_score_rows == 0:
        status = "DATA_BLOCKED"
    elif (
        len(common) < FROZEN_GATES["minimum_observations"]
        or episodes < FROZEN_GATES["minimum_independent_episodes"]
    ):
        status = "INSUFFICIENT_EVIDENCE"
    else:
        status = "HISTORICAL_PASS" if all(gates.values()) else "HISTORICAL_FAIL"

    complete = frame["label_complete"].fillna(False)
    report: dict[str, object] = {
        "model": candidate_column,
        "status": status,
        "overall_pass": status == "HISTORICAL_PASS" and all(gates.values()),
        "definition": {
            "benchmark": "NQ=F",
            "output_start": _validate_output_start(output_start).date().isoformat(),
            "horizon_trading_days": horizon,
            "drawdown_event_threshold": drawdown_threshold,
            "label_path": f"t+1..t+{horizon}",
            "label_embargo_bars": horizon,
            "common_date_rule": "exact intersection of candidate and ranked comparator scores",
        },
        "coverage": {
            "score_rows": candidate_score_rows,
            "output_rows": int(len(frame)),
            "complete_label_rows": int(complete.sum()),
            "incomplete_label_rows": int((~complete).sum()),
            "common_labelled_rows": int(len(common)),
            "common_start": common["date"].min().date().isoformat()
            if not common.empty
            else None,
            "common_end": common["date"].max().date().isoformat()
            if not common.empty
            else None,
            "independent_episodes": episodes,
        },
        "models": models,
        "paired_comparisons": paired,
        "yearly_auc": yearly_auc_summary(common, ranked_columns),
        "reference_only": sorted(references.intersection(daily.columns)),
        "score_diagnostics": score_path_diagnostics(frame[candidate_column]),
        "gates": gates,
        "frozen_gate_limits": FROZEN_GATES.copy(),
    }
    if long_column is not None:
        policy_daily = daily
        if as_of is not None:
            if "date" not in daily:
                raise ValueError("missing validation columns: date")
            policy_daily = daily.loc[
                pd.to_datetime(daily["date"], errors="raise") <= _timestamp(as_of)
            ].copy()
        _, report["policy_10bp"] = evaluate_exposure_policy(
            policy_daily, long_column=long_column, output_start=output_start
        )
    return report


def markdown_report(report: dict[str, object]) -> str:
    """Render a compact audit report without changing any database or website."""
    coverage = report["coverage"]
    rows = [
        "# RORO 2026 Causal OOS Validation",
        "",
        f"- Status: **{report['status']}**",
        f"- Model: `{report['model']}`",
        f"- Exact common labelled rows: {coverage['common_labelled_rows']:,}",
        f"- Independent {report['definition']['horizon_trading_days']}-day episodes: "
        f"{coverage['independent_episodes']}",
        "- Original RORO.txt: `REFERENCE_ONLY` (excluded from paired ranking)",
        "",
        "| Score | Role | AUC | AP | Top-quintile lift |",
        "|---|---|---:|---:|---:|",
    ]
    for name, metric in report["models"].items():
        if metric["role"] == REFERENCE_ONLY:
            rows.append(f"| {name} | {REFERENCE_ONLY} | — | — | — |")
            continue
        value = lambda key: "—" if metric[key] is None else f"{metric[key]:.3f}"
        rows.append(
            f"| {name} | {metric['role']} | {value('roc_auc')} | "
            f"{value('average_precision')} | {value('top_quintile_lift')} |"
        )
    diagnostics = report.get("score_diagnostics", {}).get("windows", {})
    if diagnostics:
        rows.extend(
            [
                "",
                "## Integer score path diagnostics",
                "",
                "| Window | Available | Mode share | Unique | Changes | Longest run | Terminal run |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for window, metric in diagnostics.items():
            mode = "—" if metric["mode_share"] is None else f"{metric['mode_share']:.1%}"
            rows.append(
                f"| {window} | {metric['available_scores']} | {mode} | "
                f"{metric['unique_scores']} | {metric['score_changes']} | "
                f"{metric['longest_run_days']} | {metric['terminal_run_days']} |"
            )
        rows.extend(
            [
                "",
                "| Window | Regime | Occupancy | Median run | Max run | Transitions |",
                "|---:|---|---:|---:|---:|---:|",
            ]
        )
        for window, metric in diagnostics.items():
            for regime in RISK_REGIMES:
                occupancy = metric["regime_occupancy"][regime]
                run = metric["regime_runs"][regime]
                occupancy_text = "—" if occupancy is None else f"{occupancy:.1%}"
                median_text = (
                    "—"
                    if run["median_run_days"] is None
                    else f"{run['median_run_days']:.1f}"
                )
                rows.append(
                    f"| {window} | {regime} | {occupancy_text} | {median_text} | "
                    f"{run['max_run_days']} | {metric['regime_transitions']} |"
                )
    yearly = report.get("yearly_auc", {}).get("years", {})
    if yearly:
        rows.extend(
            [
                "",
                "## Descriptive yearly AUC",
                "",
                "| Year | Score | Rows | Events | Non-events | AUC | Status |",
                "|---:|---|---:|---:|---:|---:|---|",
            ]
        )
        for year, year_summary in yearly.items():
            for name, metric in year_summary["models"].items():
                auc = "—" if metric["roc_auc"] is None else f"{metric['roc_auc']:.3f}"
                rows.append(
                    f"| {year} | {name} | {year_summary['rows']} | "
                    f"{year_summary['events']} | {year_summary['non_events']} | "
                    f"{auc} | {metric['status']} |"
                )
    status_coverage = report.get("coverage_status")
    if status_coverage:
        coverage_text = (
            "—"
            if status_coverage["coverage_share"] is None
            else f"{status_coverage['coverage_share']:.1%}"
        )
        no_score_text = (
            "—"
            if status_coverage["no_score_share"] is None
            else f"{status_coverage['no_score_share']:.1%}"
        )
        rows.extend(
            [
                "",
                "## Score and data-status coverage",
                "",
                f"- Score coverage: {status_coverage['score_rows']} / "
                f"{status_coverage['rows']} ({coverage_text})",
                f"- NO_SCORE: {status_coverage['no_score_rows']} ({no_score_text})",
                f"- data_status availability: **{status_coverage['data_status']}**",
                "",
                "| data_status | Rows | Share |",
                "|---|---:|---:|",
            ]
        )
        for name, count in status_coverage["data_status_counts"].items():
            rows.append(
                f"| {name} | {count} | "
                f"{status_coverage['data_status_shares'][name]:.1%} |"
            )
    episode = report.get("red_alert_onset")
    if episode:
        rows.extend(["", "## RED alert and NQ drawdown onset", ""])
        if episode["status"] == "DATA_BLOCKED":
            rows.append(f"- **DATA_BLOCKED** — {episode['reason']}")
        else:
            rows.extend(
                [
                    f"- Onsets matched: {episode['matched_onsets']} / {episode['onsets']}",
                    f"- Alert clusters: {episode['alert_clusters']} "
                    f"(matched {episode['matched_clusters']}, false "
                    f"{episode['false_alert_clusters']}, pending {episode['pending_clusters']})",
                    f"- Median lead: "
                    f"{episode['median_lead_sessions'] if episode['median_lead_sessions'] is not None else '—'} sessions",
                ]
            )
    relapse = report.get("reentry_relapse_10d")
    if relapse:
        rows.extend(["", "## REENTRY_DIRECT 10-session relapse", ""])
        if relapse["status"] == "DATA_BLOCKED":
            rows.append(f"- **DATA_BLOCKED** — {relapse['reason']}")
        else:
            failure_rate = (
                "—"
                if relapse["failure_rate"] is None
                else f"{relapse['failure_rate']:.1%}"
            )
            rows.extend(
                [
                    f"- Evaluated: {relapse['evaluated_events']} / {relapse['events']}",
                    f"- Failures: {relapse['failures']} ({failure_rate})",
                    f"- Tail pending: {relapse['tail_pending_events']}",
                ]
            )
    rows.extend(["", "## Frozen gates", "", "| Gate | Result |", "|---|---|"])
    rows.extend(
        f"| {name} | {'PASS' if passed else 'FAIL'} |"
        for name, passed in report["gates"].items()
    )
    return "\n".join(rows) + "\n"
