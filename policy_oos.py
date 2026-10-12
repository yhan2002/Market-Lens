"""Sequential historical replay of the production NQ risk policy."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import platform
import sqlite3
import sys
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from risk_policy import (
    POLICY_VERSION,
    RULE_SCORE_POLICY_VERSION,
    RULE_STRESS_SCORE_POLICY_VERSION,
    RULE_STRESS_SCORE_V3_POLICY_VERSION,
)
from site_publisher import HMM_MODEL_VERSION, replay_policy_history


BENCHMARK = "NQ=F"
SCOPE = "SEQUENTIAL_POLICY_AS_AVAILABLE"
MIN_OBSERVATIONS = 1_000
TRADING_DAYS = 252
COST_SCENARIOS_BPS = (0, 10, 25)
V3_COST_SCENARIOS_BPS = (0, 10)
INITIAL_EXPOSURE = 1.0
CASH_SERIES = "CASH_3M"
CASH_DAY_COUNT = 365
NQ_PROXY_WARNING = (
    "NQ=F is a continuous futures price proxy; this evaluation excludes roll, "
    "collateral yield, financing, margin, tax, and implementation slippage."
)
V3_NQ_PROXY_WARNING = (
    "NQ=F is a continuous futures price proxy; this evaluation excludes roll, "
    "futures financing, margin, tax, and implementation slippage. The uninvested "
    "sleeve is modeled separately with point-in-time DGS3MO."
)
CASH_PROXY_WARNING = (
    "Residual cash uses the latest point-in-time DGS3MO yield available at the "
    "start of each holding period with simple ACT/365 accrual; it is a cash proxy, "
    "not a realized fund return."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_only_connection(database: Path) -> sqlite3.Connection:
    database = database.resolve()
    if not database.is_file():
        raise ValueError(f"database does not exist: {database.name}")
    return sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)


def load_inputs(
    database: Path, hmm_model_version: str = HMM_MODEL_VERSION
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the frozen benchmark/model pair without permitting SQLite writes."""
    with closing(_read_only_connection(database)) as connection:
        prices = pd.read_sql_query(
            """
            SELECT substr(date, 1, 10) AS date, COALESCE(adj_close, close) AS close
            FROM daily_prices
            WHERE ticker=?
            ORDER BY substr(date, 1, 10)
            """,
            connection,
            params=(BENCHMARK,),
        )
        probabilities = pd.read_sql_query(
            """
            SELECT model_version, substr(date, 1, 10) AS date,
                   p_bull, p_sideways, p_bear
            FROM regime_model_probabilities
            WHERE model_version=?
            ORDER BY substr(date, 1, 10)
            """,
            connection,
            params=(hmm_model_version,),
        )
    return prices, probabilities


def load_cash_rates(database: Path) -> pd.DataFrame:
    """Load all stored 3-month Treasury vintages without permitting writes."""
    with closing(_read_only_connection(database)) as connection:
        try:
            return pd.read_sql_query(
                """
                SELECT substr(date, 1, 10) AS observation_date,
                       value AS annual_yield_pct,
                       available_at,
                       source
                FROM macro_observations
                WHERE upper(series)=?
                ORDER BY available_at, substr(date, 1, 10)
                """,
                connection,
                params=(CASH_SERIES,),
            )
        except sqlite3.DatabaseError:
            return pd.DataFrame(
                columns=["observation_date", "annual_yield_pct", "available_at", "source"]
            )


def align_cash_returns(
    price_dates: pd.Series, cash_rates: pd.DataFrame
) -> pd.DataFrame:
    """Align the latest rate available at each holding-period start, ACT/365."""
    required = {"observation_date", "annual_yield_pct", "available_at"}
    if required - set(cash_rates):
        raise ValueError(f"cash-rate columns missing: {sorted(required - set(cash_rates))}")
    dates = pd.to_datetime(price_dates, errors="coerce")
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("price dates for cash alignment must be valid, unique, and sorted")

    rates = cash_rates.copy()
    rates["observation_date"] = pd.to_datetime(rates["observation_date"], errors="coerce")
    rates["available_at"] = pd.to_datetime(rates["available_at"], errors="coerce", utc=True)
    rates["annual_yield_pct"] = pd.to_numeric(
        rates["annual_yield_pct"], errors="coerce"
    )
    if "source" not in rates:
        rates["source"] = ""
    if rates.empty or rates[["observation_date", "available_at", "annual_yield_pct"]].isna().any().any():
        raise ValueError(
            "complete CASH_3M history is required; backfill it with "
            "macro_data.py backfill-cash-rate-alfred before v3 evaluation"
        )
    values = rates["annual_yield_pct"].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < -20).any() or (values > 100).any():
        raise ValueError("CASH_3M annual yields must be finite percentage points")
    rates["available_date"] = rates["available_at"].dt.tz_convert(None).dt.normalize()
    if (rates["available_date"] < rates["observation_date"]).any():
        raise ValueError("cash-rate availability must not precede its observation")
    rates = rates.sort_values(["available_at", "observation_date"])

    starts = dates.shift(1)
    annual_yields: list[float] = []
    observation_dates: list[pd.Timestamp | pd.NaT] = []
    availability_dates: list[pd.Timestamp | pd.NaT] = []
    sources: list[str | None] = []
    event_index = 0
    current_observation = pd.NaT
    current_availability = pd.NaT
    current_yield = math.nan
    current_source: str | None = None
    events = list(rates.itertuples(index=False))
    for start in starts:
        if pd.isna(start):
            annual_yields.append(math.nan)
            observation_dates.append(pd.NaT)
            availability_dates.append(pd.NaT)
            sources.append(None)
            continue
        while event_index < len(events) and events[event_index].available_date <= start:
            event = events[event_index]
            if pd.isna(current_observation) or event.observation_date >= current_observation:
                current_observation = event.observation_date
                current_availability = event.available_at
                current_yield = float(event.annual_yield_pct)
                current_source = str(event.source)
            event_index += 1
        annual_yields.append(current_yield)
        observation_dates.append(current_observation)
        availability_dates.append(current_availability)
        sources.append(current_source)

    result = pd.DataFrame(
        {
            "date": dates,
            "cash_observation_date": observation_dates,
            "cash_available_at": availability_dates,
            "cash_source": sources,
            "cash_rate_annual_pct": annual_yields,
        }
    )
    calendar_days = dates.diff().dt.days
    result["cash_day_count"] = calendar_days
    result["cash_return"] = (
        result["cash_rate_annual_pct"] / 100 * calendar_days / CASH_DAY_COUNT
    )
    missing = starts.notna() & result["cash_return"].isna()
    if missing.any():
        first_missing = result.loc[missing, "date"].iloc[0].date().isoformat()
        raise ValueError(
            f"CASH_3M point-in-time coverage is missing for holding period ending {first_missing}"
        )
    return result


def _validate_inputs(
    prices: pd.DataFrame,
    probabilities: pd.DataFrame,
    hmm_model_version: str = HMM_MODEL_VERSION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    price_required = {"date", "close"}
    probability_required = {
        "model_version",
        "date",
        "p_bull",
        "p_sideways",
        "p_bear",
    }
    if price_required - set(prices):
        raise ValueError(f"price columns missing: {sorted(price_required - set(prices))}")
    if probability_required - set(probabilities):
        raise ValueError(
            f"probability columns missing: {sorted(probability_required - set(probabilities))}"
        )
    if prices.empty or probabilities.empty:
        raise ValueError("fixed NQ price or HMM probability history is empty")

    prices = prices.copy()
    probabilities = probabilities.copy()
    for name, frame in (("price", prices), ("probability", probabilities)):
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        if frame["date"].isna().any():
            raise ValueError(f"{name} dates must be valid")
        if frame["date"].duplicated().any():
            raise ValueError(f"duplicate {name} date")

    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
    if (~np.isfinite(prices["close"])).any() or (prices["close"] <= 0).any():
        raise ValueError("NQ closes must be finite and positive")

    if not probabilities["model_version"].eq(hmm_model_version).all():
        raise ValueError("HMM model-version mismatch")
    columns = ["p_bull", "p_sideways", "p_bear"]
    probabilities[columns] = probabilities[columns].apply(
        pd.to_numeric, errors="coerce"
    )
    values = probabilities[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("HMM probabilities must be finite and in [0, 1]")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("HMM probabilities must sum to 1")
    return prices.sort_values("date"), probabilities.sort_values("date")


def decisions_frame(
    records: list[dict[str, Any]], *, policy_version: str = POLICY_VERSION
) -> pd.DataFrame:
    """Flatten production replay records without recomputing any decision."""
    rows: list[dict[str, Any]] = []
    for record in records:
        regime = record.get("regime")
        policy = record.get("risk_policy")
        if not isinstance(regime, dict) or not isinstance(policy, dict):
            raise ValueError("each replay record needs regime and risk_policy mappings")
        state = policy.get("policy_state")
        held_exposure = state.get("exposure") if isinstance(state, dict) else None
        conditions = policy.get("reentry", {}).get("conditions", [])
        rows.append(
            {
                "date": record.get("as_of"),
                "policy_version": policy.get("model_version"),
                "decision_status": policy.get("decision_status"),
                "p_bull": regime.get("p_bull"),
                "p_sideways": regime.get("p_sideways"),
                "p_bear": regime.get("p_bear"),
                "raw_recommended_exposure": policy.get("raw_recommended_exposure"),
                "previous_recommended_exposure": policy.get(
                    "previous_recommended_exposure"
                ),
                "recommended_exposure": policy.get("recommended_exposure"),
                "held_exposure": held_exposure,
                "episode_started_as_of": (
                    state.get("episode_started_as_of")
                    if isinstance(state, dict)
                    else None
                ),
                "primary_driver_code": policy.get("primary_driver_code"),
                "recovery_phase": policy.get("recovery_phase"),
                "fast_status": policy.get("fast_overlay", {}).get("status"),
                "fast_trigger_count": policy.get("fast_overlay", {}).get(
                    "trigger_count"
                ),
                "fast_complete": policy.get("fast_overlay", {}).get("complete"),
                "fast_available_count": policy.get("fast_available_count"),
                "reentry_status": policy.get("reentry", {}).get("status"),
                "reentry_exact_count": sum(
                    isinstance(item, dict) and item.get("status") != "UNAVAILABLE"
                    for item in conditions
                ),
                "missing_channels": "|".join(policy.get("missing_channels", [])),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("sequential policy replay is empty")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if frame["date"].isna().any() or frame["date"].duplicated().any():
        raise ValueError("policy decision dates must be valid and unique")
    if not frame["policy_version"].eq(policy_version).all():
        raise ValueError("policy replay version mismatch")
    return frame.sort_values("date").reset_index(drop=True)


def build_evaluation_frame(
    prices: pd.DataFrame,
    probabilities: pd.DataFrame,
    decisions: pd.DataFrame,
    hmm_model_version: str = HMM_MODEL_VERSION,
    *,
    cash_rates: pd.DataFrame | None = None,
    cost_scenarios_bps: tuple[int, ...] = COST_SCENARIOS_BPS,
) -> pd.DataFrame:
    """Apply each NQ-calendar close decision to the next available NQ bar."""
    prices, probabilities = _validate_inputs(
        prices, probabilities, hmm_model_version
    )
    required = {
        "date",
        "decision_status",
        "p_bull",
        "p_sideways",
        "p_bear",
        "raw_recommended_exposure",
        "previous_recommended_exposure",
        "recommended_exposure",
        "held_exposure",
        "episode_started_as_of",
        "primary_driver_code",
        "recovery_phase",
        "fast_status",
        "fast_trigger_count",
        "fast_complete",
        "fast_available_count",
        "reentry_status",
        "reentry_exact_count",
        "missing_channels",
    }
    if required - set(decisions):
        raise ValueError(f"decision columns missing: {sorted(required - set(decisions))}")
    decisions = decisions.copy()
    decisions["date"] = pd.to_datetime(decisions["date"], errors="coerce")
    if decisions["date"].isna().any() or decisions["date"].duplicated().any():
        raise ValueError("policy decision dates must be valid and unique")
    decisions = decisions.sort_values("date")

    if not cost_scenarios_bps or cost_scenarios_bps[0] != 0 or any(
        bps < 0 for bps in cost_scenarios_bps
    ):
        raise ValueError("cost scenarios must start with zero and be non-negative")
    cash = (
        align_cash_returns(prices["date"], cash_rates)
        if cash_rates is not None
        else pd.DataFrame(
            {
                "date": prices["date"],
                "cash_observation_date": pd.NaT,
                "cash_available_at": pd.NaT,
                "cash_source": "ZERO_RETURN_LEGACY",
                "cash_rate_annual_pct": 0.0,
                "cash_day_count": prices["date"].diff().dt.days,
                "cash_return": 0.0,
            }
        )
    )
    frame = (
        prices.merge(cash, on="date", how="left", validate="one_to_one")
        .merge(decisions, on="date", how="left", validate="one_to_one", sort=True)
    )
    ok = frame["decision_status"].eq("OK")
    frame["decision_exposure"] = frame["recommended_exposure"].where(ok)
    frame["effective_exposure"] = frame["decision_exposure"].ffill()
    frame["effective_signal_date"] = frame["date"].where(ok).ffill()
    for column in ("p_bull", "p_sideways", "p_bear"):
        frame[f"effective_{column}"] = frame[column].where(ok).ffill()

    frame["benchmark_return"] = frame["close"].pct_change(fill_method=None)
    frame["signal_date"] = frame["effective_signal_date"].shift(1)
    for column in ("p_bull", "p_sideways", "p_bear"):
        frame[f"signal_{column}"] = frame[f"effective_{column}"].shift(1)
    frame["signal_target_exposure"] = frame["effective_exposure"].shift(1)
    frame["applied_exposure"] = frame["signal_target_exposure"]
    frame = frame.dropna(
        subset=["benchmark_return", "applied_exposure", "cash_return"]
    ).copy()
    if len(frame) < MIN_OBSERVATIONS:
        raise ValueError(
            f"usable observations must be at least {MIN_OBSERVATIONS}; got {len(frame)}"
        )
    if not np.isfinite(
        frame[["benchmark_return", "applied_exposure", "cash_return"]].to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError("non-finite aligned return or exposure")
    if not frame["applied_exposure"].between(0, 1).all():
        raise ValueError("applied exposure must be in [0, 1]")

    frame["turnover"] = frame["applied_exposure"].diff().abs()
    frame.iloc[0, frame.columns.get_loc("turnover")] = abs(
        float(frame["applied_exposure"].iloc[0]) - INITIAL_EXPOSURE
    )
    frame["cash_weight"] = 1 - frame["applied_exposure"]
    frame["cash_contribution"] = frame["cash_weight"] * frame["cash_return"]
    frame["gross_return"] = (
        frame["applied_exposure"] * frame["benchmark_return"]
        + frame["cash_contribution"]
    )
    for bps in cost_scenarios_bps[1:]:
        cost_rate = frame["turnover"] * bps / 10_000
        frame[f"cost_{bps}bps_rate"] = cost_rate
        frame[f"net_{bps}bps_return"] = frame["gross_return"] - cost_rate
    output_columns = [
        "date",
        "signal_date",
        "close",
        "benchmark_return",
        "cash_observation_date",
        "cash_available_at",
        "cash_source",
        "cash_rate_annual_pct",
        "cash_day_count",
        "cash_return",
        "signal_p_bull",
        "signal_p_sideways",
        "signal_p_bear",
        "signal_target_exposure",
        "decision_status",
        "raw_recommended_exposure",
        "previous_recommended_exposure",
        "recommended_exposure",
        "episode_started_as_of",
        "primary_driver_code",
        "recovery_phase",
        "fast_status",
        "fast_trigger_count",
        "fast_complete",
        "fast_available_count",
        "reentry_status",
        "reentry_exact_count",
        "missing_channels",
        "applied_exposure",
        "cash_weight",
        "cash_contribution",
        "turnover",
        "gross_return",
    ]
    for bps in cost_scenarios_bps[1:]:
        output_columns.extend([f"cost_{bps}bps_rate", f"net_{bps}bps_return"])
    return frame[output_columns].reset_index(drop=True)


def performance_metrics(returns: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(returns, errors="coerce").to_numpy(dtype=float)
    if not len(values) or not np.isfinite(values).all() or (values <= -1).any():
        raise ValueError("returns must be finite, non-empty, and greater than -100%")
    wealth = np.concatenate(([1.0], np.cumprod(1.0 + values)))
    drawdown = wealth / np.maximum.accumulate(wealth) - 1.0
    standard_deviation = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    sharpe = (
        float(np.mean(values) / standard_deviation * math.sqrt(TRADING_DAYS))
        if standard_deviation > 0
        else 0.0
    )
    return {
        "total_return": float(wealth[-1] - 1.0),
        "cagr": float(wealth[-1] ** (TRADING_DAYS / len(values)) - 1.0),
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
    }


def _scenario_metrics(
    returns: pd.Series,
    benchmark: dict[str, float],
    *,
    bps: int,
    turnover: pd.Series,
) -> dict[str, float | int | None]:
    metrics: dict[str, float | int | None] = performance_metrics(returns)
    benchmark_mdd = abs(benchmark["max_drawdown"])
    metrics.update(
        {
            "transaction_cost_bps": bps,
            "total_turnover": float(turnover.sum()),
            "average_daily_turnover": float(turnover.mean()),
            "total_cost_rate": float((turnover * bps / 10_000).sum()),
            "relative_mdd_improvement": (
                float(
                    (benchmark_mdd - abs(float(metrics["max_drawdown"])))
                    / benchmark_mdd
                )
                if benchmark_mdd
                else None
            ),
            "annual_return_drag": float(benchmark["cagr"] - float(metrics["cagr"])),
            "sharpe_degradation": float(
                benchmark["sharpe"] - float(metrics["sharpe"])
            ),
        }
    )
    return metrics


def coverage_summary(decisions: pd.DataFrame) -> dict[str, Any]:
    total = len(decisions)
    missing_counts: dict[str, int] = {}
    for value in decisions["missing_channels"].fillna(""):
        for channel in str(value).split("|"):
            if channel:
                missing_counts[channel] = missing_counts.get(channel, 0) + 1
    return {
        "decision_days": total,
        "ok_days": int(decisions["decision_status"].eq("OK").sum()),
        "no_decision_days": int(decisions["decision_status"].eq("NO_DECISION").sum()),
        "fast_complete_days": int(decisions["fast_complete"].eq(True).sum()),
        "fast_minimum_recovery_coverage_days": int(
            decisions["fast_available_count"].ge(5).sum()
        ),
        "reentry_four_exact_days": int(decisions["reentry_exact_count"].eq(4).sum()),
        "missing_channel_days": dict(sorted(missing_counts.items())),
    }


def rerisk_failure_summary(
    decisions: pd.DataFrame,
    prices: pd.DataFrame,
) -> dict[str, Any]:
    """Evaluate the first recovery step of each episode over the next 10 NQ bars."""
    ordered_prices = prices.copy()
    ordered_prices["date"] = pd.to_datetime(ordered_prices["date"], errors="coerce")
    ordered_prices["close"] = pd.to_numeric(ordered_prices["close"], errors="coerce")
    ordered_prices = ordered_prices.sort_values("date")
    events = (
        decisions.loc[
            decisions["recovery_phase"].eq("RECOVERY_STEP")
            & decisions["episode_started_as_of"].notna()
        ]
        .sort_values("date")
        .drop_duplicates("episode_started_as_of", keep="first")
    )
    evaluated = 0
    failures = 0
    pending = 0
    details: list[dict[str, Any]] = []
    for event in events.itertuples(index=False):
        event_date = pd.Timestamp(event.date)
        base = ordered_prices.loc[ordered_prices["date"].le(event_date)].tail(1)
        future = ordered_prices.loc[ordered_prices["date"].gt(event_date)].head(10)
        if base.empty or len(future) < 10:
            pending += 1
            continue
        evaluated += 1
        minimum_return = float(future["close"].min() / float(base.iloc[0]["close"]) - 1)
        end_date = pd.Timestamp(future.iloc[-1]["date"])
        future_regimes = decisions.loc[
            decisions["date"].gt(event_date) & decisions["date"].le(end_date)
        ]
        bear_recurrence = bool(
            (
                future_regimes["p_bear"].gt(future_regimes["p_bull"])
                & future_regimes["p_bear"].gt(future_regimes["p_sideways"])
            ).any()
        )
        failed = minimum_return <= -0.03 or bear_recurrence
        failures += int(failed)
        details.append(
            {
                "episode_started_as_of": str(event.episode_started_as_of),
                "reentry_as_of": event_date.date().isoformat(),
                "minimum_10bar_return": minimum_return,
                "bear_recurrence": bear_recurrence,
                "failed": failed,
            }
        )
    failure_rate = failures / evaluated if evaluated else None
    return {
        "status": (
            "PASS" if failure_rate is not None and failure_rate <= 0.25 else "FAIL"
        ) if evaluated >= 20 else "NOT_EVALUATED",
        "events": len(events),
        "evaluated_events": evaluated,
        "pending_events": pending,
        "failures": failures,
        "failure_rate": failure_rate,
        "minimum_events": 20,
        "maximum_failure_rate": 0.25,
        "definition": "first recovery step; next 10 NQ bars min return <= -3% or BEAR argmax recurrence",
        "details": details,
    }


def evaluate(
    database: Path,
    *,
    progress: Callable[[str], None] | None = None,
    hmm_model_version: str = HMM_MODEL_VERSION,
    use_router_score: bool = False,
    use_router_score_v2: bool = False,
    use_router_score_v3: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if sum((use_router_score, use_router_score_v2, use_router_score_v3)) > 1:
        raise ValueError("choose only one router score version")
    notify = progress or (lambda message: None)
    database = database.resolve()
    notify("1/4  NQ 가격과 고정 HMM 확률을 확인합니다.")
    prices, probabilities = load_inputs(database, hmm_model_version)
    validated_prices, validated_probabilities = _validate_inputs(
        prices, probabilities, hmm_model_version
    )
    replay_end = validated_probabilities["date"].max().date().isoformat()
    policy_version = (
        RULE_STRESS_SCORE_V3_POLICY_VERSION
        if use_router_score_v3
        else RULE_STRESS_SCORE_POLICY_VERSION
        if use_router_score_v2
        else RULE_SCORE_POLICY_VERSION
        if use_router_score
        else POLICY_VERSION
    )
    score_by_date = None
    score_model_version = None
    cash_rates = None
    if use_router_score or use_router_score_v2 or use_router_score_v3:
        if use_router_score_v2 or use_router_score_v3:
            from router_score_v2 import (
                ROUTER_SCORE_VERSION,
                build_router_scores_from_database,
            )

            score_model_version = ROUTER_SCORE_VERSION
        else:
            from router_score import ROUTER_SCORE_VERSION, build_router_scores_from_database

            score_model_version = ROUTER_SCORE_VERSION

        scores = build_router_scores_from_database(database)["risk_score"].dropna()
        score_by_date = {
            date.date().isoformat(): int(score) for date, score in scores.items()
        }
    if use_router_score_v3:
        cash_rates = load_cash_rates(database)
        align_cash_returns(validated_prices["date"], cash_rates)
    notify("2/4  동일 비중 규칙을 과거 거래일 순서대로 재생합니다.")
    records = replay_policy_history(
        database,
        end=replay_end,
        progress=lambda done, total, date: notify(
            f"      {done:,}/{total:,}일 완료 · {date}"
        ),
        model_version=hmm_model_version,
        score_by_date=score_by_date,
        policy_version=policy_version,
    )
    notify("3/4  NQ 기준 성과와 거래비용 시나리오를 계산합니다.")
    decisions = decisions_frame(records, policy_version=policy_version)
    cost_scenarios_bps = (
        V3_COST_SCENARIOS_BPS if use_router_score_v3 else COST_SCENARIOS_BPS
    )
    frame = build_evaluation_frame(
        prices,
        probabilities,
        decisions,
        hmm_model_version,
        cash_rates=cash_rates,
        cost_scenarios_bps=cost_scenarios_bps,
    )
    coverage = coverage_summary(decisions)
    rerisk_gate = rerisk_failure_summary(decisions, prices)
    benchmark = performance_metrics(frame["benchmark_return"])
    scenarios: dict[str, dict[str, float | int | None]] = {}
    for bps in cost_scenarios_bps:
        returns = frame["gross_return"] if bps == 0 else frame[f"net_{bps}bps_return"]
        scenarios["gross" if bps == 0 else f"cost_{bps}bps"] = _scenario_metrics(
            returns, benchmark, bps=bps, turnover=frame["turnover"]
        )

    scenario_gates = {
        name: {
            "relative_mdd_improvement": float(metrics["relative_mdd_improvement"] or -math.inf)
            >= 0.10,
            "annual_return_drag": float(metrics["annual_return_drag"]) <= 0.02,
            "sharpe_degradation": float(metrics["sharpe_degradation"]) <= 0.10,
        }
        for name, metrics in scenarios.items()
        if name != "gross"
    }
    metric_gate_pass = len(frame) >= MIN_OBSERVATIONS and all(
        all(gates.values()) for gates in scenario_gates.values()
    )
    full_exact = (
        coverage["fast_complete_days"] == coverage["decision_days"]
        and coverage["reentry_four_exact_days"] == coverage["decision_days"]
    )
    historical_approval = (
        metric_gate_pass and full_exact and rerisk_gate["status"] == "PASS"
    )
    governance_gates = {
        "minimum_observations": len(frame) >= MIN_OBSERVATIONS,
        "next_day_execution": True,
        **{
            f"{scenario}_{gate}": passed
            for scenario, gates in scenario_gates.items()
            for gate, passed in gates.items()
        },
        "full_pdf_policy_evaluated": full_exact,
        "rerisk_failure_evaluated": rerisk_gate["status"] != "NOT_EVALUATED",
    }

    source = Path(__file__).resolve()
    policy_source = source.with_name("risk_policy.py")
    router_spec = source.with_name(
        "MODEL_SPEC_ROUTER_SCORE_v3.md"
        if use_router_score_v3
        else "MODEL_SPEC_ROUTER_SCORE_v2.md"
        if use_router_score_v2
        else "MODEL_SPEC_ROUTER_SCORE_v1.md"
        if use_router_score
        else "MODEL_SPEC_ROUTER_STATE_v2.md"
    )
    report: dict[str, Any] = {
        "role": "POLICY",
        "model_id": (
            f"{policy_version}--nq-core-v1-sequential-policy"
            if hmm_model_version == HMM_MODEL_VERSION
            else f"{policy_version}--{hmm_model_version}--sequential-policy"
        ),
        "model_version": policy_version,
        "model": (
            "NQ rule-based stress score v2 with v3 exposure map and HMM double-check"
            if use_router_score_v3
            else "NQ rule-based stress score with HMM double-check sequential policy"
            if use_router_score_v2
            else "NQ rule-based score with HMM double-check sequential policy"
            if use_router_score
            else "NQ Core v1 sequential risk policy"
            if hmm_model_version == HMM_MODEL_VERSION
            else "NQ HMM candidate sequential risk policy"
        ),
        "status": "HISTORICAL_PASS" if historical_approval else "HISTORICAL_FAIL",
        "design": {
            "scope": SCOPE,
            "engine": "site_publisher.replay_policy_history -> risk_policy.evaluate_risk_policy",
            "benchmark": BENCHMARK,
            "hmm_model_version": hmm_model_version,
            "policy_version": policy_version,
            "execution": "signal at t close; exposure applies to t+1 return",
            "decision_calendar": "NQ dates with the frozen HMM version",
            "return_calendar": "all NQ price bars; non-decision bars hold prior exposure",
            "initial_exposure": INITIAL_EXPOSURE,
            "cash_return": (
                {
                    "series": CASH_SERIES,
                    "source_series": "DGS3MO",
                    "alignment": "latest vintage available at holding-period start",
                    "accrual": "simple ACT/365",
                }
                if use_router_score_v3
                else 0.0
            ),
            "leverage_allowed": False,
            "transaction_cost_bps": list(cost_scenarios_bps),
        },
        "gates": governance_gates,
        "scope": SCOPE,
        "evaluation_status": (
            "AS_AVAILABLE_HISTORICAL_PASS"
            if metric_gate_pass
            else "AS_AVAILABLE_HISTORICAL_FAIL"
        ),
        "benchmark": BENCHMARK,
        "hmm_model_version": hmm_model_version,
        "policy_version": policy_version,
        "score_model_version": score_model_version,
        "observations": len(frame),
        "start_date": frame["date"].min().date().isoformat(),
        "end_date": frame["date"].max().date().isoformat(),
        "execution": "signal at t close; exposure applies to t+1 return",
        "cash_return": (
            "PIT CASH_3M (DGS3MO), latest available vintage, simple ACT/365"
            if use_router_score_v3
            else 0.0
        ),
        "leverage_allowed": False,
        "excluded_components": [],
        "coverage": coverage,
        "historical_metric_gate": {
            "status": "PASS" if metric_gate_pass else "FAIL",
            "minimum_observations": len(frame) >= MIN_OBSERVATIONS,
            "next_day_execution": True,
            "scenario_gates": scenario_gates,
        },
        "full_pdf_gate": {
            "status": (
                "PASS" if metric_gate_pass else "FAIL"
            ) if full_exact else "NOT_EVALUATED",
            "reason": (
                "All exact Fast and re-entry channels are present."
                if full_exact
                else "One or more exact Fast/re-entry channels are unavailable in history."
            ),
        },
        "rerisk_failure_gate": rerisk_gate,
        "benchmark_metrics": benchmark,
        "scenarios": scenarios,
        "warnings": (
            [V3_NQ_PROXY_WARNING, CASH_PROXY_WARNING]
            if use_router_score_v3
            else [NQ_PROXY_WARNING]
        ),
        "provenance": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "argv": sys.argv.copy(),
            "data_cutoff": frame["date"].max().date().isoformat(),
            "database": {
                "name": database.name,
                "size_bytes": database.stat().st_size,
                "sha256": sha256_file(database),
            },
            "source_sha256": {
                source.name: sha256_file(source),
                policy_source.name: sha256_file(policy_source),
                "site_publisher.py": sha256_file(source.with_name("site_publisher.py")),
                "risk_signals.py": sha256_file(source.with_name("risk_signals.py")),
                "MODEL_SPEC_v1.md": sha256_file(source.with_name("MODEL_SPEC_v1.md")),
                router_spec.name: sha256_file(router_spec),
                **(
                    {
                        "router_score_v2.py": sha256_file(
                            source.with_name("router_score_v2.py")
                        ),
                        "router_score.py": sha256_file(source.with_name("router_score.py")),
                    }
                    if use_router_score_v2 or use_router_score_v3
                    else {"router_score.py": sha256_file(source.with_name("router_score.py"))}
                    if use_router_score
                    else {}
                ),
                **(
                    {"macro_data.py": sha256_file(source.with_name("macro_data.py"))}
                    if use_router_score_v3
                    else {}
                ),
            },
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
        },
    }
    notify("4/4  검증 결과를 정리합니다.")
    return report, frame


def _svg_path(
    values: pd.Series,
    *,
    left: float,
    top: float,
    width: float,
    height: float,
    minimum: float,
    maximum: float,
) -> str:
    numbers = values.to_numpy(dtype=float)
    if len(numbers) > 720:
        indexes = np.unique(np.linspace(0, len(numbers) - 1, 720).astype(int))
        numbers = numbers[indexes]
    span = maximum - minimum or 1.0
    x_span = max(len(numbers) - 1, 1)
    points = [
        (
            left + width * index / x_span,
            top + height * (maximum - value) / span,
        )
        for index, value in enumerate(numbers)
        if math.isfinite(value)
    ]
    if not points:
        return ""
    return "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in points)


def _write_presentation_html(
    report: dict[str, Any], output: pd.DataFrame, path: Path
) -> None:
    scenario_name = "cost_10bps"
    scenario = report["scenarios"][scenario_name]
    benchmark = report["benchmark_metrics"]
    policy_wealth = np.log(output[f"{scenario_name}_wealth"].clip(lower=1e-12))
    benchmark_wealth = np.log(output["benchmark_wealth"].clip(lower=1e-12))
    wealth_min = float(min(policy_wealth.min(), benchmark_wealth.min()))
    wealth_max = float(max(policy_wealth.max(), benchmark_wealth.max()))
    drawdown_min = float(
        min(output[f"{scenario_name}_drawdown"].min(), output["benchmark_drawdown"].min(), -0.01)
    )
    policy_wealth_path = _svg_path(
        policy_wealth, left=58, top=36, width=1060, height=250,
        minimum=wealth_min, maximum=wealth_max,
    )
    benchmark_wealth_path = _svg_path(
        benchmark_wealth, left=58, top=36, width=1060, height=250,
        minimum=wealth_min, maximum=wealth_max,
    )
    policy_drawdown_path = _svg_path(
        output[f"{scenario_name}_drawdown"], left=58, top=352, width=1060, height=180,
        minimum=drawdown_min, maximum=0,
    )
    benchmark_drawdown_path = _svg_path(
        output["benchmark_drawdown"], left=58, top=352, width=1060, height=180,
        minimum=drawdown_min, maximum=0,
    )
    full_exact = report["full_pdf_gate"]["status"] == "PASS"
    badge = "전체 입력 검증 완료" if full_exact else "가용 데이터 기준 참고 결과"
    badge_class = "pass" if full_exact else "limited"
    coverage = report["coverage"]
    warning = html.escape(report["warnings"][0])
    cash_note = (
        " 미투자 비중에는 각 거래일 당시 이용 가능했던 3개월 미국채 금리를 "
        "일수에 맞춰 반영했습니다."
        if report["cash_return"] != 0.0
        else ""
    )
    page = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>동일 규칙 과거 성과</title><style>
:root{{--bg:#07130f;--panel:#0b1c16;--line:#244238;--text:#f3f8f5;--muted:#82a99a;--mint:#63e6ad;--gold:#e7b955;--soft:#b8c8c1}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font-family:Arial,'Malgun Gothic',sans-serif}}
main{{width:1200px;margin:0 auto;padding:42px 50px 34px}} .eyebrow{{color:var(--mint);font:12px monospace;letter-spacing:2px}}
h1{{font-size:38px;margin:10px 0 8px}} .sub{{color:var(--soft);font-size:17px;line-height:1.55;max-width:940px}}
.top{{display:flex;justify-content:space-between;gap:28px;align-items:flex-start}} .badge{{padding:10px 14px;border:1px solid;font:12px monospace;white-space:nowrap}}
.pass{{color:var(--mint);border-color:var(--mint)}} .limited{{color:var(--gold);border-color:var(--gold)}}
.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:28px 0 18px}} .card{{background:var(--panel);border:1px solid var(--line);padding:17px}}
.label{{color:var(--muted);font:11px monospace;letter-spacing:1px}} .value{{font-size:27px;font-weight:700;margin-top:7px}} .detail{{font-size:12px;color:var(--muted);margin-top:5px}}
.chart{{background:var(--panel);border:1px solid var(--line);padding:14px 14px 8px}} svg{{display:block;width:100%;height:auto}}
.legend{{display:flex;gap:22px;color:var(--soft);font-size:13px;margin:0 0 8px 44px}} .dot{{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:7px}}
.method{{display:grid;grid-template-columns:1.3fr 1fr;gap:12px;margin-top:14px}} .note{{border:1px solid var(--line);padding:15px;color:var(--soft);font-size:13px;line-height:1.55}}
.note b{{color:var(--text)}} footer{{display:flex;justify-content:space-between;color:var(--muted);font:11px monospace;margin-top:16px}}
</style></head><body><main>
<div class="top"><div><div class="eyebrow">Market Lens · RISK MANAGEMENT</div><h1>동일 규칙 과거 성과</h1>
<div class="sub">현재 운용 화면과 같은 비중 조절 규칙을 각 과거 시점에서 순서대로 적용했습니다. 신호는 다음 거래일부터 반영했으며, 대표 결과에는 거래비용 10bp를 포함했습니다.</div></div>
<div class="badge {badge_class}">{badge}</div></div>
<section class="cards">
<div class="card"><div class="label">연평균 수익률</div><div class="value">{scenario['cagr']:.1%}</div><div class="detail">NQ 기준 {benchmark['cagr']:.1%}</div></div>
<div class="card"><div class="label">최대 낙폭</div><div class="value">{scenario['max_drawdown']:.1%}</div><div class="detail">NQ 기준 {benchmark['max_drawdown']:.1%}</div></div>
<div class="card"><div class="label">위험조정 성과</div><div class="value">{scenario['sharpe']:.2f}</div><div class="detail">NQ 기준 {benchmark['sharpe']:.2f}</div></div>
<div class="card"><div class="label">분석 기간</div><div class="value" style="font-size:20px">{report['start_date']} — {report['end_date']}</div><div class="detail">{report['observations']:,} 거래일</div></div>
</section>
<section class="chart"><div class="legend"><span><i class="dot" style="background:var(--mint)"></i>비중 조절 적용</span><span><i class="dot" style="background:#82948d"></i>NQ 기준</span></div>
<svg viewBox="0 0 1176 560" role="img" aria-label="누적 성과와 최대 낙폭 비교">
<text x="58" y="20" fill="#82a99a" font-size="12">누적 가치 · 시작 1 (로그 축)</text>
<line x1="58" y1="286" x2="1118" y2="286" stroke="#244238"/><path d="{benchmark_wealth_path}" fill="none" stroke="#82948d" stroke-width="2"/><path d="{policy_wealth_path}" fill="none" stroke="#63e6ad" stroke-width="3"/>
<text x="58" y="336" fill="#82a99a" font-size="12">고점 대비 하락률</text>
<line x1="58" y1="352" x2="1118" y2="352" stroke="#244238"/><path d="{benchmark_drawdown_path}" fill="none" stroke="#82948d" stroke-width="2"/><path d="{policy_drawdown_path}" fill="none" stroke="#e7b955" stroke-width="2.5"/>
<text x="58" y="554" fill="#82a99a" font-size="11">{report['start_date']}</text><text x="1118" y="554" text-anchor="end" fill="#82a99a" font-size="11">{report['end_date']}</text>
</svg></section>
<section class="method"><div class="note"><b>비교 원칙</b><br>과거 어느 날에도 그 이후의 정보는 사용하지 않았습니다. 비중 축소와 재확대 모두 실제 화면과 동일한 순차 규칙을 사용하며, 포지션 변경은 다음 NQ 거래일부터 계산합니다.{cash_note}</div>
<div class="note"><b>데이터 확인</b><br>정상 판단 {coverage['ok_days']:,}/{coverage['decision_days']:,}일 · 8개 위험지표 완비 {coverage['fast_complete_days']:,}일 · 4개 재진입지표 완비 {coverage['reentry_four_exact_days']:,}일</div></section>
<footer><span>{html.escape(report['policy_version'])}</span><span>과거 성과는 미래 수익을 보장하지 않습니다. {warning}</span></footer>
</main></body></html>"""
    path.write_text(page, encoding="utf-8")


def write_outputs(
    report: dict[str, Any], frame: pd.DataFrame, output_dir: Path
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output_dir / "policy_oos.json",
        "markdown": output_dir / "policy_oos.md",
        "csv": output_dir / "policy_oos_daily.csv",
        "presentation": output_dir / "policy_oos_presentation.html",
    }
    paths["json"].write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    scenario_lines: list[str] = []
    for name, metrics in report["scenarios"].items():
        mdd_improvement = metrics["relative_mdd_improvement"]
        mdd_improvement_text = (
            "N/A" if mdd_improvement is None else f"{mdd_improvement:.2%}"
        )
        scenario_lines.append(
            f"| {name} | {metrics['total_return']:.2%} | {metrics['cagr']:.2%} | "
            f"{metrics['sharpe']:.3f} | {metrics['max_drawdown']:.2%} | "
            f"{mdd_improvement_text} | "
            f"{metrics['annual_return_drag']:.2%} | "
            f"{metrics['sharpe_degradation']:.3f} | {metrics['total_turnover']:.3f} | "
            f"{metrics['total_cost_rate']:.3%} |"
        )
    markdown = [
        "# Sequential Policy Historical Replay",
        "",
        f"- Scope: `{report['scope']}`",
        f"- Status: `{report['evaluation_status']}`",
        f"- Historical metric gate: `{report['historical_metric_gate']['status']}`",
        f"- Full PDF gate: `{report['full_pdf_gate']['status']}`",
        f"- Re-entry failure gate: `{report['rerisk_failure_gate']['status']}`",
        f"- Period: {report['start_date']} to {report['end_date']} ({report['observations']:,} observations)",
        f"- HMM: `{report['hmm_model_version']}`",
        f"- Policy: `{report['policy_version']}`",
        f"- Cash return: `{report['cash_return']}`",
        "- Engine: the same sequential policy transition used by live/backfill",
        f"- Decision coverage: {report['coverage']['ok_days']:,}/{report['coverage']['decision_days']:,} OK days",
        f"- Fast 8/8 exact: {report['coverage']['fast_complete_days']:,} days",
        f"- Re-entry 4/4 exact: {report['coverage']['reentry_four_exact_days']:,} days",
        "",
        "| Scenario | Total Return | CAGR | Sharpe | MDD | Relative MDD Improvement | Annual Return Drag | Sharpe Degradation | Turnover | Cost Rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        *scenario_lines,
        "",
        f"> Warning: {NQ_PROXY_WARNING}",
        "",
    ]
    paths["markdown"].write_text("\n".join(markdown), encoding="utf-8")

    output = frame.copy()
    output["benchmark_wealth"] = (1 + output["benchmark_return"]).cumprod()
    output["benchmark_drawdown"] = (
        output["benchmark_wealth"] / output["benchmark_wealth"].cummax() - 1
    )
    for name, metrics in report["scenarios"].items():
        returns = (
            output["gross_return"]
            if name == "gross"
            else output[f"net_{metrics['transaction_cost_bps']}bps_return"]
        )
        output[f"{name}_wealth"] = (1 + returns).cumprod()
        output[f"{name}_drawdown"] = (
            output[f"{name}_wealth"] / output[f"{name}_wealth"].cummax() - 1
        )
    output.to_csv(paths["csv"], index=False, encoding="utf-8")
    _write_presentation_html(report, output, paths["presentation"])
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay the frozen sequential NQ risk policy historically."
    )
    parser.add_argument("database", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--hmm-model-version", default=HMM_MODEL_VERSION)
    score_group = parser.add_mutually_exclusive_group()
    score_group.add_argument("--router-score", action="store_true")
    score_group.add_argument("--router-score-v2", action="store_true")
    score_group.add_argument("--router-score-v3", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.check_only:
        prices, probabilities = _validate_inputs(
            *load_inputs(args.database, args.hmm_model_version),
            hmm_model_version=args.hmm_model_version,
        )
        score_text = ""
        if args.router_score or args.router_score_v2 or args.router_score_v3:
            if args.router_score_v2 or args.router_score_v3:
                from router_score_v2 import build_router_scores_from_database
            else:
                from router_score import build_router_scores_from_database

            scores = build_router_scores_from_database(args.database)["risk_score"].dropna()
            probability_dates = set(probabilities["date"])
            score_dates = set(scores.index)
            missing = probability_dates - score_dates
            if missing:
                raise ValueError(f"rule score is missing on {len(missing)} HMM decision dates")
            score_text = f" rule_scores={len(scores):,}"
        if args.router_score_v3:
            cash = align_cash_returns(prices["date"], load_cash_rates(args.database))
            score_text += f" cash_rates={cash['cash_rate_annual_pct'].notna().sum():,}"
        print(
            f"preflight=OK nq_prices={len(prices):,} "
            f"hmm_probabilities={len(probabilities):,}{score_text}",
            flush=True,
        )
        return
    report, frame = evaluate(
        args.database,
        progress=lambda message: print(message, flush=True),
        hmm_model_version=args.hmm_model_version,
        use_router_score=args.router_score,
        use_router_score_v2=args.router_score_v2,
        use_router_score_v3=args.router_score_v3,
    )
    paths = write_outputs(report, frame, args.output_dir)
    print(
        f"scope={report['scope']} status={report['evaluation_status']} "
        f"full_pdf_gate={report['full_pdf_gate']['status']} observations={report['observations']}"
    )
    for kind, path in paths.items():
        print(f"{kind}={path.resolve()}")


if __name__ == "__main__":
    main()
