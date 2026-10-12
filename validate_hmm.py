from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sqlite3
import subprocess
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score


POSTERIOR_HEALTH_LIMITS = {
    63: {
        "saturation_share_max": 0.80,
        "mean_entropy_min": 0.05,
        "low_entropy_share_max": 0.85,
        "unique_scores_min": 8,
        "score_changes_min": 12,
        "score_changes_max": 42,
        "mode_share_max": 0.50,
        "state_switches_max": 12,
    },
    126: {
        "saturation_share_max": 0.75,
        "mean_entropy_min": 0.08,
        "low_entropy_share_max": 0.80,
        "unique_scores_min": 12,
        "score_changes_min": 24,
        "score_changes_max": 84,
        "mode_share_max": 0.45,
        "state_switches_max": 25,
    },
    252: {
        "saturation_share_max": 0.70,
        "mean_entropy_min": 0.10,
        "low_entropy_share_max": 0.75,
        "unique_scores_min": 16,
        "score_changes_min": 48,
        "score_changes_max": 168,
        "mode_share_max": 0.40,
        "state_switches_max": 50,
    },
}


def _future_drawdown(prices: np.ndarray, horizon: int) -> np.ndarray:
    result = np.full(len(prices), np.nan)
    for position in range(len(prices) - horizon):
        result[position] = np.min(prices[position + 1 : position + horizon + 1] / prices[position] - 1)
    return result


def _independent_episode_count(target: np.ndarray, *, separation: int = 1) -> int:
    if separation < 1:
        raise ValueError("episode separation must be positive")
    positions = np.flatnonzero(np.asarray(target, dtype=bool))
    count = 0
    last_event: int | None = None
    for position in positions:
        if last_event is None or int(position) - last_event > separation:
            count += 1
        last_event = int(position)
    return count


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_metadata(path: Path) -> dict[str, object]:
    try:
        root = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        sha = subprocess.run(
            ["git", "-C", root, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", root, "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"available": False, "sha": None, "dirty": None}
    return {"available": True, "sha": sha, "dirty": dirty}


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


def load_validation_frame(
    database: Path,
    *,
    benchmark: str = "NQ=F",
    horizon: int = 20,
    drawdown_threshold: float = -0.05,
    model_version: str | None = None,
    start_date: str | None = None,
) -> tuple[pd.DataFrame, int]:
    database = database.resolve()
    if not database.is_file():
        raise ValueError(f"database does not exist: {database.name}")
    with closing(sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)) as connection:
        prices = pd.read_sql_query(
            "SELECT date, adj_close FROM daily_prices WHERE ticker=? ORDER BY date",
            connection,
            params=(benchmark,),
            parse_dates=["date"],
        )
        if model_version is None:
            probabilities = pd.read_sql_query(
                "SELECT date, p_bull, p_sideways, p_bear FROM regime_probabilities ORDER BY date",
                connection,
                parse_dates=["date"],
            )
        else:
            probabilities = pd.read_sql_query(
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
    if prices.empty or probabilities.empty:
        raise ValueError("benchmark prices and regime probabilities are required")

    prices["return"] = prices["adj_close"].pct_change()
    ma200 = prices["adj_close"].rolling(200).mean()
    prices["ma200_risk"] = -(prices["adj_close"] / ma200 - 1)
    prices["ma200_slope_risk"] = -(ma200 / ma200.shift(20) - 1)
    prices["vol20_risk"] = prices["return"].rolling(20).std() * np.sqrt(252)
    prices["forward_return"] = prices["adj_close"].shift(-horizon) / prices["adj_close"] - 1
    prices["future_drawdown"] = _future_drawdown(prices["adj_close"].to_numpy(), horizon)

    raw = probabilities.merge(prices, on="date", how="left", validate="one_to_one")
    if start_date is not None:
        raw = raw.loc[raw["date"] >= pd.Timestamp(start_date)].copy()
    probability_sum = raw[["p_bull", "p_sideways", "p_bear"]].sum(axis=1)
    invalid_probabilities = int(
        raw[["p_bull", "p_sideways", "p_bear"]].isna().any(axis=1).sum()
        + (~np.isclose(probability_sum, 1.0, atol=1e-8)).sum()
    )
    raw["state"] = (
        raw[["p_bull", "p_sideways", "p_bear"]]
        .idxmax(axis=1)
        .str.removeprefix("p_")
        .str.upper()
    )
    raw["drawdown_event"] = (raw["future_drawdown"] <= drawdown_threshold).astype(int)
    usable = raw.dropna(
        subset=[
            "adj_close",
            "forward_return",
            "future_drawdown",
            "ma200_risk",
            "ma200_slope_risk",
            "vol20_risk",
        ]
    ).reset_index(drop=True)
    return usable, invalid_probabilities


def load_probability_history(
    database: Path,
    *,
    model_version: str | None = None,
    start_date: str | None = None,
) -> tuple[pd.DataFrame, int]:
    """Load posterior history through its true latest date, independent of outcomes."""
    database = database.resolve()
    with closing(sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)) as connection:
        if model_version is None:
            probabilities = pd.read_sql_query(
                "SELECT date, p_bull, p_sideways, p_bear FROM regime_probabilities ORDER BY date",
                connection,
                parse_dates=["date"],
            )
        else:
            probabilities = pd.read_sql_query(
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
    if start_date is not None:
        probabilities = probabilities.loc[
            probabilities["date"] >= pd.Timestamp(start_date)
        ].copy()
    values = probabilities[["p_bull", "p_sideways", "p_bear"]]
    probability_sum = values.sum(axis=1)
    invalid = int(
        values.isna().any(axis=1).sum()
        + (~np.isclose(probability_sum, 1.0, atol=1e-8)).sum()
        + ((values < 0) | (values > 1)).any(axis=1).sum()
    )
    return probabilities.reset_index(drop=True), invalid


def posterior_health(probabilities: pd.DataFrame) -> dict[str, object]:
    """Measure recent posterior certainty and displayed-score diversity."""
    columns = ["p_bull", "p_sideways", "p_bear"]
    values = probabilities[columns].to_numpy(dtype=float)
    if len(values) < max(POSTERIOR_HEALTH_LIMITS):
        raise ValueError("posterior health requires at least 252 observations")
    safe_values = np.clip(values, np.finfo(float).tiny, 1.0)
    entropy = -np.sum(values * np.log(safe_values), axis=1) / np.log(3)
    scores = np.floor(10 * values[:, 0] + 50 * values[:, 1] + 80 * values[:, 2] + 0.5).astype(int)
    states = np.argmax(values, axis=1)
    windows: dict[str, dict[str, object]] = {}
    for window, limits in POSTERIOR_HEALTH_LIMITS.items():
        recent_values = values[-window:]
        recent_entropy = entropy[-window:]
        recent_scores = scores[-window:]
        recent_states = states[-window:]
        counts = pd.Series(recent_scores).value_counts(normalize=True)
        metrics = {
            "saturation_share": float((recent_values.max(axis=1) >= 0.99).mean()),
            "mean_normalized_entropy": float(recent_entropy.mean()),
            "low_entropy_share": float((recent_entropy <= 0.10).mean()),
            "unique_integer_scores": int(len(np.unique(recent_scores))),
            "integer_score_changes": int(np.count_nonzero(np.diff(recent_scores))),
            "integer_score_mode_share": float(counts.iloc[0]),
            "hard_state_switches": int(np.count_nonzero(np.diff(recent_states))),
        }
        checks = {
            "saturation": metrics["saturation_share"] <= limits["saturation_share_max"],
            "mean_entropy": metrics["mean_normalized_entropy"] >= limits["mean_entropy_min"],
            "low_entropy_share": metrics["low_entropy_share"] <= limits["low_entropy_share_max"],
            "score_diversity": metrics["unique_integer_scores"] >= limits["unique_scores_min"],
            "score_changes": limits["score_changes_min"] <= metrics["integer_score_changes"] <= limits["score_changes_max"],
            "score_mode_share": metrics["integer_score_mode_share"] <= limits["mode_share_max"],
            "state_churn": metrics["hard_state_switches"] <= limits["state_switches_max"],
        }
        windows[str(window)] = {**metrics, "checks": checks, "pass": all(checks.values())}
    return {"windows": windows, "pass": all(item["pass"] for item in windows.values())}


def score_metrics(
    frame: pd.DataFrame,
    score_column: str,
    *,
    bootstrap_samples: int = 500,
    block_days: int = 40,
    random_state: int = 42,
    episode_separation: int = 1,
) -> dict[str, float | int]:
    observed = frame[["date", "drawdown_event", score_column]].dropna()
    target = observed["drawdown_event"].to_numpy()
    score = observed[score_column].to_numpy()
    if len(np.unique(target)) < 2:
        raise ValueError("validation target needs both event and non-event observations")

    auc = float(roc_auc_score(target, score))
    average_precision = float(average_precision_score(target, score))
    base_rate = float(target.mean())
    top_cutoff = float(np.quantile(score, 0.8))
    top_rate = float(target[score >= top_cutoff].mean())
    top_lift = top_rate / base_rate if base_rate else np.nan

    buckets = pd.qcut(observed[score_column], 5, labels=False, duplicates="drop")
    bucket_rates = observed.assign(bucket=buckets).groupby("bucket")["drawdown_event"].mean()
    monotonicity = float(
        pd.Series(range(len(bucket_rates))).corr(bucket_rates.reset_index(drop=True), method="spearman")
    )

    rng = np.random.default_rng(random_state)
    bootstrap_auc: list[float] = []
    for _ in range(bootstrap_samples):
        sampled = _moving_block_indices(len(observed), block_days=block_days, rng=rng)
        if np.unique(target[sampled]).size == 2:
            bootstrap_auc.append(
                float(roc_auc_score(target[sampled], score[sampled]))
            )
    auc_low, auc_high = np.quantile(bootstrap_auc, [0.025, 0.975])
    return {
        "n": int(len(observed)),
        "event_rate": base_rate,
        "independent_drawdown_episodes": _independent_episode_count(
            target, separation=episode_separation
        ),
        "episode_separation": episode_separation,
        "roc_auc": auc,
        "auc_ci_low": float(auc_low),
        "auc_ci_high": float(auc_high),
        "average_precision": average_precision,
        "top_quintile_event_rate": top_rate,
        "top_quintile_lift": float(top_lift),
        "bucket_monotonicity": monotonicity,
    }


def paired_auc_comparison(
    frame: pd.DataFrame,
    candidate_column: str,
    champion_column: str,
    *,
    bootstrap_samples: int = 500,
    block_days: int = 40,
    random_state: int = 42,
) -> dict[str, float | int | str]:
    observed = frame[
        ["date", "drawdown_event", candidate_column, champion_column]
    ].dropna()
    target = observed["drawdown_event"]
    if target.nunique() < 2:
        raise ValueError("paired validation target needs both event and non-event observations")

    candidate_auc = float(roc_auc_score(target, observed[candidate_column]))
    champion_auc = float(roc_auc_score(target, observed[champion_column]))
    rng = np.random.default_rng(random_state)
    deltas: list[float] = []
    for _ in range(bootstrap_samples):
        sampled = _moving_block_indices(len(observed), block_days=block_days, rng=rng)
        if target.iloc[sampled].nunique() == 2:
            deltas.append(
                float(
                    roc_auc_score(target.iloc[sampled], observed[candidate_column].iloc[sampled])
                    - roc_auc_score(target.iloc[sampled], observed[champion_column].iloc[sampled])
                )
            )
    if not deltas:
        raise ValueError("paired bootstrap produced no valid samples")
    delta_low, delta_high = np.quantile(deltas, [0.025, 0.975])
    return {
        "n": int(len(observed)),
        "start": observed["date"].min().date().isoformat(),
        "end": observed["date"].max().date().isoformat(),
        "candidate_auc": candidate_auc,
        "champion_auc": champion_auc,
        "auc_delta": candidate_auc - champion_auc,
        "auc_delta_ci_low": float(delta_low),
        "auc_delta_ci_high": float(delta_high),
    }


def _state_statistics(frame: pd.DataFrame) -> list[dict[str, object]]:
    summary = (
        frame.groupby("state")
        .agg(
            n=("state", "size"),
            forward_return_mean=("forward_return", "mean"),
            forward_return_median=("forward_return", "median"),
            future_drawdown_mean=("future_drawdown", "mean"),
            drawdown_event_rate=("drawdown_event", "mean"),
        )
        .reset_index()
    )
    return summary.to_dict(orient="records")


def _persistence(frame: pd.DataFrame) -> dict[str, float | int]:
    run_id = frame["state"].ne(frame["state"].shift()).cumsum()
    lengths = frame.groupby(run_id).size()
    return {
        "transitions": int(max(len(lengths) - 1, 0)),
        "median_run_days": float(lengths.median()),
        "mean_run_days": float(lengths.mean()),
    }


def validate_hmm(
    database: Path,
    *,
    benchmark: str = "NQ=F",
    horizon: int = 20,
    drawdown_threshold: float = -0.05,
    bootstrap_samples: int = 500,
    model_version: str | None = None,
    start_date: str | None = None,
    minimum_sample: int = 1000,
    compare_with_v1: bool = False,
    compare_model_version: str | None = None,
    block_days: int = 40,
) -> dict[str, object]:
    if compare_with_v1 and compare_model_version:
        raise ValueError("choose either legacy v1 or an explicit comparator model version")
    frame, invalid_probabilities = load_validation_frame(
        database,
        benchmark=benchmark,
        horizon=horizon,
        drawdown_threshold=drawdown_threshold,
        model_version=model_version,
        start_date=start_date,
    )
    models = {
        "HMM_P_BEAR": score_metrics(
            frame, "p_bear", bootstrap_samples=bootstrap_samples, block_days=block_days,
            episode_separation=horizon,
        ),
        "MA200_DISTANCE": score_metrics(
            frame, "ma200_risk", bootstrap_samples=bootstrap_samples, block_days=block_days,
            episode_separation=horizon,
        ),
        "MA200_SLOPE_20": score_metrics(
            frame,
            "ma200_slope_risk",
            bootstrap_samples=bootstrap_samples,
            block_days=block_days,
            episode_separation=horizon,
        ),
        "VOLATILITY_20": score_metrics(
            frame, "vol20_risk", bootstrap_samples=bootstrap_samples, block_days=block_days,
            episode_separation=horizon,
        ),
    }
    paired_comparison = None
    if compare_with_v1 or compare_model_version:
        if model_version is None:
            raise ValueError("candidate comparison requires --model-version")
        champion, champion_invalid = load_validation_frame(
            database,
            benchmark=benchmark,
            horizon=horizon,
            drawdown_threshold=drawdown_threshold,
            model_version=compare_model_version,
            start_date=start_date,
        )
        candidate_scores = frame[["date", "drawdown_event", "p_bear"]].rename(
            columns={"drawdown_event": "candidate_target", "p_bear": "candidate_p_bear"}
        )
        champion_scores = champion[["date", "drawdown_event", "p_bear"]].rename(
            columns={"drawdown_event": "champion_target", "p_bear": "champion_p_bear"}
        )
        shared = candidate_scores.merge(
            champion_scores, on="date", how="inner", validate="one_to_one"
        )
        if shared.empty or not shared["candidate_target"].equals(shared["champion_target"]):
            raise ValueError("candidate and v1 targets do not align on shared dates")
        shared["drawdown_event"] = shared["candidate_target"]
        paired_comparison = paired_auc_comparison(
            shared,
            "candidate_p_bear",
            "champion_p_bear",
            bootstrap_samples=bootstrap_samples,
            block_days=block_days,
        )
        paired_comparison["champion_invalid_probabilities"] = champion_invalid
        paired_comparison["comparator_model_version"] = (
            compare_model_version or "legacy-regime_probabilities"
        )
    yearly = []
    eligible_years: list[int] = []
    excluded_years: list[int] = []
    for year, group in frame.groupby(frame["date"].dt.year):
        if len(group) >= 60:
            eligible_years.append(int(year))
        if len(group) >= 60 and group["drawdown_event"].nunique() == 2:
            yearly.append(
                {
                    "year": int(year),
                    "n": int(len(group)),
                    "event_rate": float(group["drawdown_event"].mean()),
                    "hmm_auc": float(roc_auc_score(group["drawdown_event"], group["p_bear"])),
                }
            )
        elif len(group) >= 60:
            excluded_years.append(int(year))
    persistence = _persistence(frame)
    hmm = models["HMM_P_BEAR"]
    positive_year_ratio = (
        sum(row["hmm_auc"] > 0.5 for row in yearly) / len(eligible_years)
        if eligible_years
        else 0.0
    )
    probability_history, history_invalid = load_probability_history(
        database, model_version=model_version, start_date=start_date
    )
    health = posterior_health(probability_history)
    state_occupancy = {
        state: float((frame["state"] == state).mean())
        for state in ("BULL", "SIDEWAYS", "BEAR")
    }
    gates = {
        "probability_integrity": invalid_probabilities == 0 and history_invalid == 0,
        "sample_size": int(hmm["n"]) >= minimum_sample,
        "independent_episodes": int(hmm["independent_drawdown_episodes"]) >= 20,
        "auc_ci_above_random": float(hmm["auc_ci_low"]) > 0.5,
        "beats_simple_baseline": float(hmm["roc_auc"])
        >= max(
            float(metric["roc_auc"])
            for name, metric in models.items()
            if name != "HMM_P_BEAR"
        )
        + 0.02,
        "top_quintile_lift": float(hmm["top_quintile_lift"]) >= 1.25,
        "bucket_monotonicity": float(hmm["bucket_monotonicity"]) >= 0.5,
        "regime_persistence": float(persistence["median_run_days"]) >= 5,
        "state_occupancy": all(0.05 <= value <= 0.80 for value in state_occupancy.values()),
        "posterior_saturation": health["pass"],
        "yearly_consistency": positive_year_ratio >= 0.6,
    }
    if paired_comparison is not None:
        gates["paired_auc_improvement_vs_comparator"] = (
            float(paired_comparison["auc_delta_ci_low"]) > 0
        )
    resolved_model_version = model_version or "hmm3-v1"
    overall_pass = all(gates.values())
    return {
        "role": "CURRENT_REGIME",
        "model_id": resolved_model_version,
        "model_version": resolved_model_version,
        "model": "Three-state Gaussian HMM current-regime posterior",
        "status": "HISTORICAL_PASS" if overall_pass else "HISTORICAL_FAIL",
        "definition": {
            "benchmark": benchmark,
            "model_version": resolved_model_version,
            "validation_start": start_date,
            "minimum_sample": minimum_sample,
            "bootstrap_block_days": block_days,
            "horizon_trading_days": horizon,
            "drawdown_event_threshold": drawdown_threshold,
            "note": "p_bear is a current-state posterior, evaluated here only as a future-risk ranking score.",
        },
        "coverage": {
            "rows": len(frame),
            "start": frame["date"].min().date().isoformat(),
            "end": frame["date"].max().date().isoformat(),
            "invalid_probabilities": invalid_probabilities,
            "posterior_end": probability_history["date"].max().date().isoformat(),
        },
        "models": models,
        "state_statistics": _state_statistics(frame),
        "persistence": persistence,
        "posterior_diagnostics": {
            "hard_state_occupancy": state_occupancy,
            "health": health,
        },
        "yearly": yearly,
        "yearly_total_years": len(eligible_years),
        "yearly_excluded_years": excluded_years,
        "positive_year_ratio": positive_year_ratio,
        "paired_comparison": paired_comparison,
        "gates": gates,
        "overall_pass": overall_pass,
    }


def _percent(value: object) -> str:
    return f"{float(value):.2%}"


def markdown_report(report: dict[str, object]) -> str:
    definition = report["definition"]
    coverage = report["coverage"]
    lines = [
        "# HMM Validation Report",
        "",
        f"- Coverage: {coverage['start']} to {coverage['end']} ({coverage['rows']:,} rows)",
        f"- Model: `{definition['model_version']}`",
        f"- Target: next {definition['horizon_trading_days']} trading-day drawdown <= {_percent(definition['drawdown_event_threshold'])}",
        f"- Bootstrap: {definition['bootstrap_block_days']}-trading-day moving blocks",
        f"- Independent drawdown episodes ({definition['horizon_trading_days']}-day separation): {report['models']['HMM_P_BEAR']['independent_drawdown_episodes']}",
        f"- Overall gate: **{'PASS' if report['overall_pass'] else 'FAIL'}**",
        "- Important: `p_bear` is not a calibrated probability of a future decline.",
        "",
        "## Validation Gates",
        "",
        "| Gate | Result |",
        "|---|---|",
    ]
    lines.extend(f"| {name} | {'PASS' if passed else 'FAIL'} |" for name, passed in report["gates"].items())
    lines.extend(["", "## Risk Score Comparison", "", "| Score | ROC-AUC | 95% CI | Top-quintile lift | Monotonicity |", "|---|---:|---:|---:|---:|"])
    for name, metric in report["models"].items():
        lines.append(
            f"| {name} | {metric['roc_auc']:.3f} | {metric['auc_ci_low']:.3f}–{metric['auc_ci_high']:.3f} | "
            f"{metric['top_quintile_lift']:.2f}x | {metric['bucket_monotonicity']:.3f} |"
        )
    if report.get("paired_comparison") is not None:
        paired = report["paired_comparison"]
        lines.extend(
            [
                "",
                "## Paired Comparison",
                "",
                f"- Comparator: `{paired['comparator_model_version']}`",
                f"- Shared dates: {paired['start']} to {paired['end']} ({paired['n']:,} rows)",
                f"- Candidate AUC: {paired['candidate_auc']:.3f}",
                f"- v1 AUC: {paired['champion_auc']:.3f}",
                f"- Paired ΔAUC: {paired['auc_delta']:+.3f} "
                f"(95% CI {paired['auc_delta_ci_low']:+.3f}–{paired['auc_delta_ci_high']:+.3f})",
            ]
        )
    lines.extend(["", "## State Statistics", "", "| State | N | Forward return | Future drawdown | Drawdown event |", "|---|---:|---:|---:|---:|"])
    for row in report["state_statistics"]:
        lines.append(
            f"| {row['state']} | {row['n']:,} | {_percent(row['forward_return_mean'])} | "
            f"{_percent(row['future_drawdown_mean'])} | {_percent(row['drawdown_event_rate'])} |"
        )
    lines.extend([
        "",
        "## Stability",
        "",
        f"- State transitions: {report['persistence']['transitions']}",
        f"- Median state run: {report['persistence']['median_run_days']:.1f} trading days",
        f"- Years with HMM AUC > 0.50: {report['positive_year_ratio']:.1%}",
        f"- Posterior history through: {report['coverage']['posterior_end']}",
        f"- Recent posterior health: {'PASS' if report['posterior_diagnostics']['health']['pass'] else 'FAIL'}",
        "",
        "| Year | N | Event rate | HMM AUC |",
        "|---:|---:|---:|---:|",
    ])
    for row in report["yearly"]:
        lines.append(
            f"| {row['year']} | {row['n']:,} | {_percent(row['event_rate'])} | {row['hmm_auc']:.3f} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "The HMM is eligible for portfolio routing only when every validation gate passes and it beats simple baselines out of sample.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate HMM as a future drawdown risk score")
    parser.add_argument("database", type=Path)
    parser.add_argument("--benchmark", default="NQ=F")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--drawdown-threshold", type=float, default=-0.05)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--model-version")
    parser.add_argument("--start-date")
    parser.add_argument("--minimum-sample", type=int, default=1000)
    parser.add_argument("--compare-with-v1", action="store_true")
    parser.add_argument("--compare-model-version")
    parser.add_argument("--block-days", type=int, default=40)
    parser.add_argument("--spec", type=Path, default=Path("MODEL_SPEC_v1.md"))
    parser.add_argument("--json", type=Path, default=Path("hmm_validation_report.json"))
    parser.add_argument("--markdown", type=Path, default=Path("hmm_validation_report.md"))
    args = parser.parse_args()

    report = validate_hmm(
        args.database,
        benchmark=args.benchmark,
        horizon=args.horizon,
        drawdown_threshold=args.drawdown_threshold,
        bootstrap_samples=args.bootstrap_samples,
        model_version=args.model_version,
        start_date=args.start_date,
        minimum_sample=args.minimum_sample,
        compare_with_v1=args.compare_with_v1,
        compare_model_version=args.compare_model_version,
        block_days=args.block_days,
    )
    source_path = Path(__file__).resolve()
    spec_path = args.spec.resolve()
    if not spec_path.is_file():
        raise FileNotFoundError(f"model specification not found: {spec_path}")
    database_path = args.database.resolve()
    report["provenance"] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_cutoff": report["coverage"]["end"],
        "model_spec": spec_path.name,
        "database": {
            "name": database_path.name,
            "size_bytes": database_path.stat().st_size,
            "sha256": _sha256_file(database_path),
        },
        "source_sha256": {
            path.name: _sha256_file(path)
            for path in (source_path, spec_path)
            if path.exists()
        },
        "git": _git_metadata(source_path.parent),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    args.markdown.write_text(markdown_report(report), encoding="utf-8")
    print(f"overall={'PASS' if report['overall_pass'] else 'FAIL'}")
    for name, passed in report["gates"].items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
