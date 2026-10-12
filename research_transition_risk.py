from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sqlite3
import subprocess
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


Q20_MODEL_VERSION = "q20-nq-core-v1-logit7-c1-m756-w1260-r20"
Q20_REQUIRED_HMM_VERSION = "hmm3-sticky-nq-core-v1-k25-m252-w1260-r20-s42-n3"
Q20_FEATURES = (
    "p_bear",
    "p_sideways",
    "p_bear_change_5",
    "momentum_20",
    "volatility_20",
    "drawdown_60",
    "ma200_slope_20",
)
Q20_HORIZON = 20
Q20_DRAWDOWN_THRESHOLD = -0.05
Q20_MIN_TRAIN = 756
Q20_TRAIN_WINDOW = 1260
Q20_REFIT_EVERY = 20
Q20_RANDOM_STATE = 42
Q20_BOOTSTRAP_SAMPLES = 500
Q20_BLOCK_LENGTH = 40
Q20_VALIDATION_PROTOCOL = "q20-oos-v1-b500-block40-decluster20-rawvol"


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


@dataclass(frozen=True)
class Candidate:
    features: tuple[str, ...]
    factory: Callable[[], object]


def fixed_q20_candidate(
    features: tuple[str, ...] = Q20_FEATURES,
) -> Candidate:
    return Candidate(
        features,
        lambda: LogisticRegression(
            C=1.0,
            class_weight=None,
            max_iter=1_000,
            random_state=Q20_RANDOM_STATE,
        ),
    )


def _future_drawdown(prices: np.ndarray, horizon: int) -> np.ndarray:
    result = np.full(len(prices), np.nan)
    for position in range(len(prices) - horizon):
        result[position] = np.min(
            prices[position + 1 : position + horizon + 1] / prices[position] - 1
        )
    return result


def load_frame(
    database: Path,
    *,
    benchmark: str = "NQ=F",
    model_version: str | None = None,
    horizon: int = 20,
    drawdown_threshold: float = -0.05,
) -> pd.DataFrame:
    uri = database.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        prices = pd.read_sql_query(
            "SELECT date, adj_close FROM daily_prices WHERE ticker=? ORDER BY date",
            connection,
            params=(benchmark,),
            parse_dates=["date"],
        )
        if model_version is None:
            query = "SELECT date, p_bull, p_sideways, p_bear FROM regime_probabilities ORDER BY date"
            probabilities = pd.read_sql_query(query, connection, parse_dates=["date"])
        else:
            query = """
                SELECT date, p_bull, p_sideways, p_bear
                FROM regime_model_probabilities
                WHERE model_version=? ORDER BY date
            """
            probabilities = pd.read_sql_query(
                query, connection, params=(model_version,), parse_dates=["date"]
            )
            champion = pd.read_sql_query(
                "SELECT date, p_bear AS v1_p_bear FROM regime_probabilities ORDER BY date",
                connection,
                parse_dates=["date"],
            )
    if prices.empty or probabilities.empty:
        raise ValueError("benchmark prices and regime probabilities are required")

    close = prices["adj_close"]
    daily_return = close.pct_change()
    ma200 = close.rolling(200).mean()
    prices = prices.assign(
        return_5=close.pct_change(5),
        momentum_20=close.pct_change(20),
        momentum_60=close.pct_change(60),
        volatility_20=daily_return.rolling(20).std() * np.sqrt(252),
        drawdown_60=close / close.rolling(60).max() - 1,
        ma200_distance=close / ma200 - 1,
        ma200_slope_20=ma200 / ma200.shift(20) - 1,
        future_drawdown=_future_drawdown(close.to_numpy(), horizon),
    )
    frame = probabilities.merge(prices, on="date", how="inner", validate="one_to_one")
    if model_version is None:
        frame["v1_p_bear"] = frame["p_bear"]
    else:
        frame = frame.merge(champion, on="date", how="left", validate="one_to_one")
    probability_sum = frame[["p_bull", "p_sideways", "p_bear"]].sum(axis=1)
    if frame[["p_bull", "p_sideways", "p_bear"]].isna().any(axis=None) or not np.allclose(
        probability_sum, 1.0, atol=1e-8
    ):
        raise ValueError("invalid regime probabilities")
    probability_array = frame[["p_bull", "p_sideways", "p_bear"]].clip(1e-12, 1)
    frame["regime_entropy"] = -(probability_array * np.log(probability_array)).sum(axis=1)
    frame["p_bear_change_5"] = frame["p_bear"].diff(5)
    frame["bear_minus_bull"] = frame["p_bear"] - frame["p_bull"]
    frame["ma200_risk"] = -frame["ma200_distance"]
    frame["drawdown_event"] = np.where(
        frame["future_drawdown"].notna(),
        (frame["future_drawdown"] <= drawdown_threshold).astype(float),
        np.nan,
    )
    return frame.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)


def candidates(random_state: int = 42) -> dict[str, Candidate]:
    hmm = ("p_bear", "p_sideways", "p_bear_change_5", "bear_minus_bull", "regime_entropy")
    price = (
        "return_5",
        "momentum_20",
        "momentum_60",
        "volatility_20",
        "drawdown_60",
        "ma200_distance",
        "ma200_slope_20",
    )
    logistic = lambda: LogisticRegression(
        C=0.3, class_weight="balanced", max_iter=1_000, random_state=random_state
    )
    result = {
        "logit_hmm": Candidate(hmm, logistic),
        "logit_price": Candidate(price, logistic),
        "logit_all": Candidate(hmm + price, logistic),
        "hist_all": Candidate(
            hmm + price,
            lambda: HistGradientBoostingClassifier(
                max_iter=100,
                max_leaf_nodes=7,
                min_samples_leaf=30,
                l2_regularization=1.0,
                class_weight="balanced",
                random_state=random_state,
            ),
        ),
    }
    if len(result) > 6:
        raise RuntimeError("research candidate cap exceeded")
    return result


def _eligible_training_frame(
    frame: pd.DataFrame,
    *,
    position: int,
    horizon: int,
    feature_columns: tuple[str, ...],
    train_window: int | None,
) -> pd.DataFrame:
    # At t, the newest usable label belongs to t-horizon; its forward window ends at t.
    train = frame.iloc[: max(0, position - horizon + 1)]
    if train_window:
        train = train.iloc[-train_window:]
    return train.dropna(subset=[*feature_columns, "drawdown_event"])


def walk_forward_predict(
    frame: pd.DataFrame,
    candidate: Candidate,
    *,
    start_date: str,
    end_date: str | None,
    horizon: int,
    min_train: int,
    train_window: int | None,
    refit_every: int,
) -> pd.Series:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) if end_date else frame["date"].max()
    positions = frame.index[(frame["date"] >= start) & (frame["date"] <= end)].tolist()
    predictions = pd.Series(np.nan, index=frame.index, dtype=float)
    fitted = None
    last_refit_position: int | None = None
    for position in positions:
        if fitted is None or position - (last_refit_position or 0) >= refit_every:
            train = _eligible_training_frame(
                frame,
                position=position,
                horizon=horizon,
                feature_columns=candidate.features,
                train_window=train_window,
            )
            if len(train) < min_train or train["drawdown_event"].nunique() < 2:
                continue
            fitted = make_pipeline(StandardScaler(), candidate.factory())
            fitted.fit(train[list(candidate.features)], train["drawdown_event"].astype(int))
            last_refit_position = position
        row = frame.loc[[position], list(candidate.features)]
        if not row.isna().any(axis=None):
            predictions.loc[position] = float(fitted.predict_proba(row)[0, 1])
    return predictions


def independent_episode_count(target: np.ndarray, *, separation: int = 1) -> int:
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


def _moving_block_indices(
    size: int, *, block_length: int, rng: np.random.Generator
) -> np.ndarray:
    if size <= block_length:
        raise ValueError("block length must be smaller than the validation sample")
    starts = rng.integers(0, size - block_length + 1, size=int(np.ceil(size / block_length)))
    return np.concatenate([np.arange(start, start + block_length) for start in starts])[:size]


def score(
    observed: pd.DataFrame,
    score_column: str,
    *,
    bootstrap_samples: int,
    block_length: int,
    random_state: int,
    episode_separation: int = 1,
) -> dict[str, object]:
    data = observed[["date", "drawdown_event", score_column]].dropna().reset_index(drop=True)
    target = data["drawdown_event"].astype(int).to_numpy()
    values = data[score_column].to_numpy()
    if len(data) == 0 or np.unique(target).size < 2:
        return {"n": int(len(data)), "error": "both target classes are required"}
    auc = float(roc_auc_score(target, values))
    cutoff = float(np.quantile(values, 0.8))
    top_rate = float(target[values >= cutoff].mean())
    base_rate = float(target.mean())
    all_years = sorted(str(int(year)) for year in data["date"].dt.year.unique())
    yearly_auc = {
        str(int(year)): float(roc_auc_score(group["drawdown_event"], group[score_column]))
        for year, group in data.groupby(data["date"].dt.year)
        if group["drawdown_event"].nunique() == 2
    }
    rng = np.random.default_rng(random_state)
    bootstrapped = []
    for _ in range(bootstrap_samples):
        sampled = _moving_block_indices(len(data), block_length=block_length, rng=rng)
        if np.unique(target[sampled]).size == 2:
            bootstrapped.append(float(roc_auc_score(target[sampled], values[sampled])))
    low, high = np.quantile(bootstrapped, [0.025, 0.975]) if bootstrapped else (np.nan, np.nan)
    return {
        "n": int(len(data)),
        "start": data["date"].min().date().isoformat(),
        "end": data["date"].max().date().isoformat(),
        "independent_drawdown_episodes": independent_episode_count(
            target, separation=episode_separation
        ),
        "episode_separation": episode_separation,
        "event_rate": float(target.mean()),
        "roc_auc": auc,
        "auc_ci_low": float(low),
        "auc_ci_high": float(high),
        "average_precision": float(average_precision_score(target, values)),
        "top_quintile_lift": top_rate / base_rate if base_rate else np.nan,
        "yearly_auc": yearly_auc,
        "yearly_auc_total_years": len(all_years),
        "yearly_auc_excluded_years": [year for year in all_years if year not in yearly_auc],
    }


def paired_delta(
    observed: pd.DataFrame,
    candidate_column: str,
    baseline_column: str,
    *,
    bootstrap_samples: int,
    block_length: int,
    random_state: int,
) -> dict[str, float | int]:
    data = observed[["drawdown_event", candidate_column, baseline_column]].dropna().reset_index(drop=True)
    target = data["drawdown_event"].astype(int).to_numpy()
    candidate_score = data[candidate_column].to_numpy()
    baseline_score = data[baseline_column].to_numpy()
    delta = float(roc_auc_score(target, candidate_score) - roc_auc_score(target, baseline_score))
    rng = np.random.default_rng(random_state)
    deltas = []
    for _ in range(bootstrap_samples):
        sampled = _moving_block_indices(len(data), block_length=block_length, rng=rng)
        if np.unique(target[sampled]).size == 2:
            deltas.append(
                float(
                    roc_auc_score(target[sampled], candidate_score[sampled])
                    - roc_auc_score(target[sampled], baseline_score[sampled])
                )
            )
    low, high = np.quantile(deltas, [0.025, 0.975]) if deltas else (np.nan, np.nan)
    return {"n": int(len(data)), "delta_auc": delta, "ci_low": float(low), "ci_high": float(high)}


def q20_probability_metrics(
    observed: pd.DataFrame, score_column: str = "q20_raw"
) -> dict[str, float]:
    data = observed[["drawdown_event", score_column]].dropna()
    target = data["drawdown_event"].astype(int).to_numpy()
    probability = data[score_column].astype(float).to_numpy()
    if len(data) == 0 or np.unique(target).size < 2:
        raise ValueError("q20 probability metrics require both target classes")
    if np.any(~np.isfinite(probability)) or np.any((probability < 0) | (probability > 1)):
        raise ValueError("q20 probabilities must be finite values in [0, 1]")

    base_rate = float(target.mean())
    brier = float(np.mean((probability - target) ** 2))
    base_brier = float(np.mean((base_rate - target) ** 2))
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    calibration = LogisticRegression(C=1e6, max_iter=1_000, random_state=Q20_RANDOM_STATE)
    calibration.fit(logit, target)

    bin_index = np.minimum((probability * 10).astype(int), 9)
    ece = 0.0
    for index in range(10):
        selected = bin_index == index
        if selected.any():
            ece += float(selected.mean()) * abs(
                float(probability[selected].mean()) - float(target[selected].mean())
            )
    return {
        "brier_score": brier,
        "base_brier_score": base_brier,
        "brier_skill": 1 - brier / base_brier if base_brier else float("nan"),
        "calibration_slope": float(calibration.coef_[0, 0]),
        "ece_10": float(ece),
        "probability_min": float(probability.min()),
        "probability_max": float(probability.max()),
    }


def build_fixed_q20_predictions(
    frame: pd.DataFrame,
    *,
    min_train: int = Q20_MIN_TRAIN,
    train_window: int | None = Q20_TRAIN_WINDOW,
    refit_every: int = Q20_REFIT_EVERY,
) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("q20 frame is empty")
    result = frame.copy()
    start_date = pd.Timestamp(result["date"].min()).date().isoformat()
    result["q20_raw"] = walk_forward_predict(
        result,
        fixed_q20_candidate(),
        start_date=start_date,
        end_date=None,
        horizon=Q20_HORIZON,
        min_train=min_train,
        train_window=train_window,
        refit_every=refit_every,
    )
    result["volatility_20_baseline"] = result["volatility_20"]
    return result


def fixed_q20_gates(
    metrics: dict[str, object], comparison: dict[str, float | int]
) -> dict[str, bool]:
    yearly_auc = metrics.get("yearly_auc", {})
    yearly_values = list(yearly_auc.values()) if isinstance(yearly_auc, dict) else []
    total_years = int(metrics.get("yearly_auc_total_years", len(yearly_values)))
    positive_year_ratio = (
        sum(float(value) > 0.50 for value in yearly_values) / total_years
        if total_years > 0
        else 0.0
    )
    metrics["positive_year_ratio"] = positive_year_ratio
    return {
        "usable_observations": int(metrics.get("n", 0)) >= 1_000,
        "independent_episodes": int(metrics.get("independent_drawdown_episodes", 0)) >= 20,
        "auc_ci_above_random": float(metrics.get("auc_ci_low", -np.inf)) > 0.50,
        "paired_volatility_delta_ci": float(comparison.get("ci_low", -np.inf)) >= 0.0,
        "top_quintile_lift": float(metrics.get("top_quintile_lift", -np.inf)) >= 1.25,
        "brier_skill": float(metrics.get("brier_skill", -np.inf)) > 0.0,
        "calibration_slope": 0.75
        <= float(metrics.get("calibration_slope", -np.inf))
        <= 1.25,
        "ece_10": float(metrics.get("ece_10", np.inf)) <= 0.10,
        "positive_year_ratio": positive_year_ratio >= 0.60,
    }


def run_fixed_q20(
    database: Path,
    *,
    model_version: str,
    benchmark: str = "NQ=F",
    bootstrap_samples: int = Q20_BOOTSTRAP_SAMPLES,
    block_length: int = Q20_BLOCK_LENGTH,
) -> tuple[dict[str, object], pd.DataFrame]:
    if not model_version:
        raise ValueError("model_version is required for fixed q20")
    if model_version != Q20_REQUIRED_HMM_VERSION:
        raise ValueError(
            "fixed q20 requires its frozen NQ core v1 HMM; create a new q20 model version for other posteriors"
        )
    if benchmark != "NQ=F":
        raise ValueError("fixed q20 benchmark must be NQ=F")
    if bootstrap_samples != Q20_BOOTSTRAP_SAMPLES or block_length != Q20_BLOCK_LENGTH:
        raise ValueError("fixed q20 validation protocol requires 500 bootstrap samples and 40-day blocks")
    frame = load_frame(
        database,
        benchmark=benchmark,
        model_version=model_version,
        horizon=Q20_HORIZON,
        drawdown_threshold=Q20_DRAWDOWN_THRESHOLD,
    )
    predictions = build_fixed_q20_predictions(frame)
    observed = predictions.dropna(
        subset=["drawdown_event", "q20_raw", "volatility_20_baseline"]
    )
    metrics = score(
        observed,
        "q20_raw",
        bootstrap_samples=bootstrap_samples,
        block_length=block_length,
        random_state=Q20_RANDOM_STATE,
        episode_separation=Q20_HORIZON,
    )
    if "error" in metrics:
        raise ValueError(str(metrics["error"]))
    metrics.update(q20_probability_metrics(observed))
    baseline_metrics = score(
        observed,
        "volatility_20_baseline",
        bootstrap_samples=bootstrap_samples,
        block_length=block_length,
        random_state=Q20_RANDOM_STATE,
        episode_separation=Q20_HORIZON,
    )
    comparison = paired_delta(
        observed,
        "q20_raw",
        "volatility_20_baseline",
        bootstrap_samples=bootstrap_samples,
        block_length=block_length,
        random_state=Q20_RANDOM_STATE,
    )
    gates = fixed_q20_gates(metrics, comparison)
    latest = predictions.dropna(subset=["q20_raw"]).iloc[-1]
    source_path = Path(__file__).resolve()
    spec_path = source_path.with_name("MODEL_SPEC_v1.md")
    database_path = database.resolve()
    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_cutoff": pd.Timestamp(frame["date"].max()).date().isoformat(),
        "validation_cutoff": pd.Timestamp(observed["date"].max()).date().isoformat(),
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
    report: dict[str, object] = {
        "role": "Q20",
        "model_version": Q20_MODEL_VERSION,
        "status": "HISTORICAL_PASS" if all(gates.values()) else "HISTORICAL_FAIL",
        "deployment_status": "SHADOW_ELIGIBLE" if all(gates.values()) else "RESEARCH_ONLY",
        "design": {
            "benchmark": benchmark,
            "regime_model_version": model_version,
            "target": "t+1..t+20 maximum drawdown <= -5%",
            "horizon": Q20_HORIZON,
            "drawdown_threshold": Q20_DRAWDOWN_THRESHOLD,
            "label_embargo_observations": Q20_HORIZON,
            "episode_declustering_observations": Q20_HORIZON,
            "features": list(Q20_FEATURES),
            "model": "StandardScaler + LogisticRegression",
            "C": 1.0,
            "class_weight": None,
            "max_iter": 1_000,
            "random_state": Q20_RANDOM_STATE,
            "minimum_train": Q20_MIN_TRAIN,
            "train_window": Q20_TRAIN_WINDOW,
            "refit_every": Q20_REFIT_EVERY,
            "database_mode": "read_only",
            "bootstrap": f"moving_block_{block_length}_trading_days",
            "bootstrap_samples": bootstrap_samples,
            "validation_protocol_version": Q20_VALIDATION_PROTOCOL,
        },
        "latest": {
            "date": pd.Timestamp(latest["date"]).date().isoformat(),
            "q20_raw": float(latest["q20_raw"]),
            "outcome_complete": bool(pd.notna(latest["drawdown_event"])),
        },
        "metrics": metrics,
        "volatility_20_baseline_metrics": baseline_metrics,
        "paired_delta_vs_volatility_20": comparison,
        "gates": gates,
        "provenance": provenance,
    }
    columns = [
        "date",
        "adj_close",
        "future_drawdown",
        "drawdown_event",
        "q20_raw",
        "volatility_20_baseline",
    ]
    output = predictions[columns].copy()
    output.insert(1, "benchmark", benchmark)
    output.insert(2, "hmm_model_version", model_version)
    output.insert(3, "q20_model_version", Q20_MODEL_VERSION)
    output["target_complete"] = output["drawdown_event"].notna()
    return report, output


def fixed_q20_markdown(report: dict[str, object]) -> str:
    metrics = report["metrics"]
    baseline = report["volatility_20_baseline_metrics"]
    comparison = report["paired_delta_vs_volatility_20"]
    gates = report["gates"]
    lines = [
        "# NQ Fixed q20 Historical OOS",
        "",
        f"- Status: **{report['status']}**",
        f"- Model: `{report['model_version']}`",
        f"- Latest raw q20: {report['latest']['q20_raw']:.2%} ({report['latest']['date']})",
        f"- OOS observations: {metrics['n']}",
        f"- ROC AUC: {metrics['roc_auc']:.3f} [{metrics['auc_ci_low']:.3f}, {metrics['auc_ci_high']:.3f}]",
        f"- Volatility-only ROC AUC: {baseline['roc_auc']:.3f} [{baseline['auc_ci_low']:.3f}, {baseline['auc_ci_high']:.3f}]",
        f"- Top-quintile lift: {metrics['top_quintile_lift']:.2f}x",
        f"- Brier skill: {metrics['brier_skill']:.3f}",
        f"- Calibration slope / ECE10: {metrics['calibration_slope']:.3f} / {metrics['ece_10']:.3f}",
        f"- Delta AUC vs volatility-only: {comparison['delta_auc']:.3f} [{comparison['ci_low']:.3f}, {comparison['ci_high']:.3f}]",
        "",
        "## Frozen gates",
        "",
        *[f"- {'PASS' if passed else 'FAIL'} — `{name}`" for name, passed in gates.items()],
        "",
        "Historical gate를 통과해도 24시간 cooling-off와 126거래일 shadow 전에는 대시보드 확률 또는 주문 신호로 사용할 수 없습니다.",
        "",
    ]
    return "\n".join(lines)


def candidate_gates(
    metrics: dict[str, object],
    rolling_blocks: dict[str, dict[str, object]],
    comparison: dict[str, float | int],
) -> dict[str, bool]:
    selection_blocks = [
        result
        for name, result in rolling_blocks.items()
        if name.endswith("_selection") and "roc_auc" in result
    ]
    return {
        "sample_size": int(metrics.get("n", 0)) >= 1_000,
        "independent_episodes": int(metrics.get("independent_drawdown_episodes", 0)) >= 20,
        "auc": float(metrics.get("roc_auc", 0)) >= 0.60,
        "auc_ci_above_random": float(metrics.get("auc_ci_low", 0)) > 0.50,
        "top_quintile_lift": float(metrics.get("top_quintile_lift", 0)) >= 1.50,
        "rolling_blocks": bool(selection_blocks)
        and all(float(result["roc_auc"]) > 0.50 for result in selection_blocks),
        "baseline_margin": float(comparison.get("delta_auc", -np.inf)) >= 0.02,
        "paired_ci_above_zero": float(comparison.get("ci_low", -np.inf)) > 0,
    }


def run_research(
    database: Path,
    *,
    model_version: str | None = None,
    benchmark: str = "NQ=F",
    horizon: int = 20,
    drawdown_threshold: float = -0.05,
    train_window: int | None = 1260,
    min_train: int = 252,
    refit_every: int = 20,
    bootstrap_samples: int = 500,
    block_length: int = 40,
    random_state: int = 42,
    candidate_names: tuple[str, ...] | None = None,
) -> dict[str, object]:
    frame = load_frame(
        database,
        benchmark=benchmark,
        model_version=model_version,
        horizon=horizon,
        drawdown_threshold=drawdown_threshold,
    )
    all_candidates = candidates(random_state)
    selected_candidates = {
        name: all_candidates[name] for name in (candidate_names or tuple(all_candidates))
    }
    for name, candidate in selected_candidates.items():
        frame[name] = walk_forward_predict(
            frame,
            candidate,
            start_date="2018-07-01",
            end_date="2023-12-31",
            horizon=horizon,
            min_train=min_train,
            train_window=train_window,
            refit_every=refit_every,
        )

    common = frame.dropna(subset=["drawdown_event", *selected_candidates])
    selection = common.loc[common["date"].between("2020-01-01", "2023-12-31")]
    if selection.empty:
        raise ValueError("no common 2020-2023 development predictions")
    blocks = {
        "2018H2-2019_diagnostic": ("2018-07-01", "2019-12-31"),
        "2020-2021_selection": ("2020-01-01", "2021-12-31"),
        "2022-2023_selection": ("2022-01-01", "2023-12-31"),
    }
    development: dict[str, object] = {}
    for name in selected_candidates:
        development[name] = {
            "selection_2020_2023": score(
                selection,
                name,
                bootstrap_samples=bootstrap_samples,
                block_length=block_length,
                random_state=random_state,
            ),
            "rolling_origin_blocks": {
                block: score(
                    common.loc[common["date"].between(start, end)],
                    name,
                    bootstrap_samples=bootstrap_samples,
                    block_length=block_length,
                    random_state=random_state,
                )
                for block, (start, end) in blocks.items()
            },
        }
    baseline_names = ("p_bear", "ma200_risk", "v1_p_bear")
    development_baselines = {
        name: score(
            selection,
            name,
            bootstrap_samples=bootstrap_samples,
            block_length=block_length,
            random_state=random_state,
        )
        for name in baseline_names
    }
    strongest_baseline = max(
        development_baselines,
        key=lambda name: development_baselines[name].get("roc_auc", -np.inf),
    )
    for name, result in development.items():
        comparison = paired_delta(
            selection,
            name,
            strongest_baseline,
            bootstrap_samples=bootstrap_samples,
            block_length=block_length,
            random_state=random_state,
        )
        result["paired_delta_vs_strongest_baseline"] = comparison
        result["gates"] = candidate_gates(
            result["selection_2020_2023"], result["rolling_origin_blocks"], comparison
        )

    eligible = [
        name for name, result in development.items() if all(result["gates"].values())
    ]
    winner = (
        max(
            eligible,
            key=lambda name: development[name]["paired_delta_vs_strongest_baseline"]["ci_low"],
        )
        if eligible
        else None
    )

    confirmation = None
    if winner is not None:
        # This period has been seen in earlier research, so it is confirmation, not a holdout.
        frame["winner_score"] = walk_forward_predict(
            frame,
            selected_candidates[winner],
            start_date="2024-01-01",
            end_date=None,
            horizon=horizon,
            min_train=min_train,
            train_window=train_window,
            refit_every=refit_every,
        )
        confirmation_frame = frame.loc[frame["date"] >= "2024-01-01"].dropna(
            subset=["drawdown_event", "winner_score", *baseline_names]
        )
        confirmation = {
            "candidate": winner,
            "metrics": score(
                confirmation_frame,
                "winner_score",
                bootstrap_samples=bootstrap_samples,
                block_length=block_length,
                random_state=random_state,
            ),
            "baselines": {
                column: score(
                    confirmation_frame,
                    column,
                    bootstrap_samples=bootstrap_samples,
                    block_length=block_length,
                    random_state=random_state,
                )
                for column in baseline_names
            },
            "paired_delta_vs_baseline": {
                column: paired_delta(
                    confirmation_frame,
                    "winner_score",
                    column,
                    bootstrap_samples=bootstrap_samples,
                    block_length=block_length,
                    random_state=random_state,
                )
                for column in baseline_names
            },
        }
    return {
        "design": {
            "benchmark": benchmark,
            "model_version": model_version or "v1",
            "horizon": horizon,
            "drawdown_threshold": drawdown_threshold,
            "label_embargo_observations": horizon,
            "candidate_selection_period": "2020-01-01/2023-12-31",
            "winner_only_seen_confirmation_period": "2024-01-01/latest_complete_target",
            "train_window": train_window or "expanding",
            "refit_every": refit_every,
            "bootstrap": f"moving_block_{block_length}_trading_days",
            "candidates": {
                name: list(candidate.features)
                for name, candidate in selected_candidates.items()
            },
        },
        "common_development_dates": {
            "n": int(len(common)),
            "start": common["date"].min().date().isoformat(),
            "end": common["date"].max().date().isoformat(),
        },
        "development_candidates": development,
        "development_baselines": development_baselines,
        "strongest_development_baseline": strongest_baseline,
        "status": "SUCCESS" if winner is not None else "FUTILITY",
        "selected_candidate": winner,
        "seen_confirmation": confirmation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage-free transition-risk research")
    parser.add_argument("database", type=Path)
    parser.add_argument("--model-version")
    parser.add_argument("--benchmark")
    parser.add_argument("--fixed-q20", action="store_true")
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--drawdown-threshold", type=float)
    parser.add_argument("--train-window", type=int, help="0 means expanding")
    parser.add_argument("--min-train", type=int)
    parser.add_argument("--refit-every", type=int)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--block-length", type=int, default=40)
    parser.add_argument("--random-state", type=int)
    parser.add_argument("--output", type=Path, default=Path("transition_risk_research.json"))
    parser.add_argument("--predictions-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()
    if args.fixed_q20:
        if not args.model_version:
            parser.error("--model-version is required with --fixed-q20")
        frozen_overrides = {
            name: getattr(args, name)
            for name in (
                "horizon",
                "drawdown_threshold",
                "train_window",
                "min_train",
                "refit_every",
                "random_state",
            )
            if getattr(args, name) is not None
        }
        if frozen_overrides:
            parser.error(
                "--fixed-q20 uses frozen model settings; remove: "
                + ", ".join("--" + name.replace("_", "-") for name in frozen_overrides)
            )
        if (
            args.bootstrap_samples != Q20_BOOTSTRAP_SAMPLES
            or args.block_length != Q20_BLOCK_LENGTH
        ):
            parser.error("--fixed-q20 requires --bootstrap-samples 500 and --block-length 40")
        report, predictions = run_fixed_q20(
            args.database,
            model_version=args.model_version,
            benchmark=args.benchmark or "NQ=F",
            bootstrap_samples=args.bootstrap_samples,
            block_length=args.block_length,
        )
        predictions_output = args.predictions_output or args.output.with_name(
            f"{args.output.stem}_predictions.csv"
        )
        markdown_output = args.markdown_output or args.output.with_suffix(".md")
        for path in (args.output, predictions_output, markdown_output):
            path.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        predictions.to_csv(predictions_output, index=False, date_format="%Y-%m-%d")
        markdown_output.write_text(fixed_q20_markdown(report), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        print(f"saved report: {args.output.resolve()}")
        print(f"saved predictions: {predictions_output.resolve()}")
        print(f"saved markdown: {markdown_output.resolve()}")
        return
    report = run_research(
        args.database,
        model_version=args.model_version,
        benchmark=args.benchmark or "NQ=F",
        horizon=args.horizon if args.horizon is not None else 20,
        drawdown_threshold=(
            args.drawdown_threshold if args.drawdown_threshold is not None else -0.05
        ),
        train_window=(
            None if args.train_window == 0 else args.train_window
        ) if args.train_window is not None else 1260,
        min_train=args.min_train if args.min_train is not None else 252,
        refit_every=args.refit_every if args.refit_every is not None else 20,
        bootstrap_samples=args.bootstrap_samples,
        block_length=args.block_length,
        random_state=args.random_state if args.random_state is not None else 42,
    )
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
