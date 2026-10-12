from __future__ import annotations

import argparse
import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DIRECT_BENCHMARK = "NQ=F"
REQUIRED_TICKERS = (DIRECT_BENCHMARK, "SPY", "^VIX", "HYG", "IEF", "RSP", "IWM", "UUP")
HORIZON = 20
MAX_CANDIDATES = 3
MAX_RESEARCH_TRIALS = 4
SHADOW_MODEL_VERSION = "onset-fixed-core-nq-v2-shadow"
SHADOW_ALERT_CUTOFF = 0.72
MACRO_FEATURES = (
    "nq_return_5",
    "nq_return_20",
    "vol20",
    "ma200_distance",
    "drawdown_60",
    "log_vix",
    "vix_z252",
    "vix_change_5",
)
CROSS_FEATURES = MACRO_FEATURES + (
    "hyg_ief_20",
    "rsp_spy_20",
    "iwm_spy_20",
    "uup_momentum_20",
)


@dataclass(frozen=True)
class Candidate:
    name: str
    features: tuple[str, ...]
    factory: Callable[[], object]


def candidate_specs(random_state: int = 42) -> tuple[Candidate, ...]:
    def logit() -> object:
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced", max_iter=2_000, random_state=random_state
            ),
        )

    candidates = (
        Candidate("logit_macro", MACRO_FEATURES, logit),
        Candidate("logit_cross", CROSS_FEATURES, logit),
        Candidate(
            "shallow_hist_cross",
            CROSS_FEATURES,
            lambda: HistGradientBoostingClassifier(
                max_depth=2,
                max_iter=100,
                learning_rate=0.05,
                l2_regularization=1.0,
                class_weight="balanced",
                random_state=random_state,
            ),
        ),
    )
    if len(candidates) > MAX_CANDIDATES:
        raise RuntimeError(f"candidate cap exceeded: {len(candidates)} > {MAX_CANDIDATES}")
    return candidates


def load_prices(database: Path) -> pd.DataFrame:
    placeholders = ",".join("?" for _ in REQUIRED_TICKERS)
    with closing(sqlite3.connect(database)) as connection:
        prices = pd.read_sql_query(
            f"SELECT date, ticker, adj_close FROM daily_prices "
            f"WHERE ticker IN ({placeholders}) ORDER BY date, ticker",
            connection,
            params=REQUIRED_TICKERS,
            parse_dates=["date"],
        )
    matrix = prices.pivot(index="date", columns="ticker", values="adj_close")
    missing = sorted(set(REQUIRED_TICKERS) - set(matrix.columns))
    if missing:
        raise ValueError("missing required tickers in daily_prices: " + ", ".join(missing))
    return matrix.loc[:, REQUIRED_TICKERS].sort_index()


def drawdown_onset_labels(
    benchmark: pd.Series,
    *,
    horizon: int = HORIZON,
    at_risk_threshold: float = -0.03,
    onset_threshold: float = -0.05,
) -> pd.DataFrame:
    high = benchmark.astype(float).rolling(252, min_periods=252).max()
    drawdown = benchmark / high - 1
    at_risk = drawdown >= at_risk_threshold
    onset = np.zeros(len(benchmark), dtype=bool)
    armed = False
    for position, value in enumerate(drawdown.to_numpy()):
        if not np.isfinite(value):
            continue
        if value >= at_risk_threshold:
            armed = True
        if armed and value <= onset_threshold:
            onset[position] = True
            armed = False

    target = np.full(len(benchmark), np.nan)
    label_end = np.full(len(benchmark), np.nan)
    for position in range(len(benchmark) - horizon):
        if at_risk.iloc[position]:
            target[position] = float(onset[position + 1 : position + horizon + 1].any())
            label_end[position] = position + horizon
    return pd.DataFrame(
        {
            "drawdown_252": drawdown.to_numpy(),
            "at_risk": at_risk.to_numpy(),
            "onset": onset,
            "target": target,
            "label_end_position": label_end,
        },
        index=benchmark.index,
    )


def build_frame(prices: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(REQUIRED_TICKERS) - set(prices.columns))
    if missing:
        raise ValueError("missing required price columns: " + ", ".join(missing))
    prices = prices.sort_index()
    prices = prices.loc[prices[DIRECT_BENCHMARK].notna()]
    nq = prices[DIRECT_BENCHMARK].astype(float)
    spy = prices["SPY"].astype(float)
    vix_log = np.log(prices["^VIX"].astype(float))
    vix_mean = vix_log.rolling(252).mean()
    vix_std = vix_log.rolling(252).std().replace(0, np.nan)
    frame = pd.DataFrame(index=prices.index)
    frame["nq_return_5"] = nq.pct_change(5)
    frame["nq_return_20"] = nq.pct_change(20)
    frame["vol20"] = nq.pct_change().rolling(20).std() * np.sqrt(252)
    frame["ma200_distance"] = nq / nq.rolling(200).mean() - 1
    frame["drawdown_60"] = nq / nq.rolling(60).max() - 1
    frame["log_vix"] = vix_log
    frame["vix_z252"] = (vix_log - vix_mean) / vix_std
    frame["vix_change_5"] = vix_log.diff(5)
    frame["hyg_ief_20"] = (prices["HYG"] / prices["IEF"]).pct_change(20)
    frame["rsp_spy_20"] = (prices["RSP"] / spy).pct_change(20)
    frame["iwm_spy_20"] = (prices["IWM"] / spy).pct_change(20)
    frame["uup_momentum_20"] = prices["UUP"].pct_change(20)
    frame = frame.join(drawdown_onset_labels(nq))
    frame.insert(0, "date", frame.index)
    frame["_position"] = np.arange(len(frame))
    frame["baseline_vix"] = frame["log_vix"]
    frame["baseline_vol"] = frame["vol20"]
    frame["baseline_neg_ma200"] = -frame["ma200_distance"]
    return frame.reset_index(drop=True)


def purged_train_mask(
    frame: pd.DataFrame, prediction_position: int, *, horizon: int = HORIZON
) -> pd.Series:
    expected_end = frame["_position"] + horizon
    recorded_end = frame.get("label_end_position", expected_end)
    return (
        frame["at_risk"].astype(bool)
        & frame["target"].notna()
        & (recorded_end <= prediction_position)
        & (expected_end <= prediction_position)
    )


def walk_forward_scores(
    frame: pd.DataFrame,
    candidate: Candidate,
    prediction_mask: pd.Series,
    *,
    horizon: int = HORIZON,
    refit_every: int = 20,
    min_train: int = 252,
) -> pd.Series:
    rows = frame.index[
        prediction_mask
        & frame["at_risk"].astype(bool)
        & frame[list(candidate.features)].notna().all(axis=1)
    ].to_numpy()
    output = pd.Series(np.nan, index=frame.index, dtype=float)
    for start in range(0, len(rows), refit_every):
        block = rows[start : start + refit_every]
        position = int(frame.loc[block[0], "_position"])
        train_mask = purged_train_mask(frame, position, horizon=horizon)
        train_mask &= frame[list(candidate.features)].notna().all(axis=1)
        train = frame.loc[train_mask]
        if len(train) < min_train or train["target"].nunique() < 2:
            continue
        model = candidate.factory()
        model.fit(train[list(candidate.features)], train["target"].astype(int))
        output.loc[block] = model.predict_proba(
            frame.loc[block, list(candidate.features)]
        )[:, 1]
    return output


def causal_trailing_percentile(
    scores: pd.Series, *, window: int = 756, min_history: int = 60
) -> pd.Series:
    result = pd.Series(np.nan, index=scores.index, dtype=float)
    finite_positions: list[int] = []
    values = scores.to_numpy(dtype=float)
    for position, value in enumerate(values):
        if not np.isfinite(value):
            continue
        history_positions = finite_positions[-window:]
        if len(history_positions) >= min_history:
            history = values[history_positions]
            result.iloc[position] = float(np.mean(history <= value))
        finite_positions.append(position)
    return result


def fixed_core_rank_components(
    frame: pd.DataFrame, *, window: int = 756, min_history: int = 60
) -> pd.DataFrame:
    at_risk = frame["at_risk"].astype(bool)
    return pd.DataFrame(
        {
            "vix_level": causal_trailing_percentile(
                frame["vix_z252"].where(at_risk), window=window, min_history=min_history
            ),
            "vix_acceleration": causal_trailing_percentile(
                frame["vix_change_5"].where(at_risk), window=window, min_history=min_history
            ),
            "credit_risk": causal_trailing_percentile(
                (-frame["hyg_ief_20"]).where(at_risk),
                window=window,
                min_history=min_history,
            ),
        }
    )


def fixed_core_rank_score(
    frame: pd.DataFrame, *, window: int = 756, min_history: int = 60
) -> pd.Series:
    """Causal, sign-fixed ensemble; no coefficients are estimated from outcomes."""
    at_risk = frame["at_risk"].astype(bool)
    components = fixed_core_rank_components(
        frame, window=window, min_history=min_history
    )
    return components.mean(axis=1).where(components.notna().all(axis=1) & at_risk)


def event_metrics(
    evaluation: pd.DataFrame,
    onset_positions: Sequence[int],
    scores: pd.Series,
    cutoff: float,
    *,
    horizon: int = HORIZON,
    cluster_gap: int = HORIZON,
) -> dict[str, float | int]:
    valid = evaluation.loc[scores.reindex(evaluation.index).notna()].copy()
    valid["score"] = scores.reindex(valid.index)
    positions = set(valid["_position"].astype(int))
    eligible = [
        int(event)
        for event in onset_positions
        if any(event - horizon <= position < event for position in positions)
    ]
    alerts = valid.loc[valid["score"] >= cutoff, "_position"].astype(int).tolist()
    clusters: list[list[int]] = []
    for position in alerts:
        if not clusters or position - clusters[-1][-1] > cluster_gap:
            clusters.append([position])
        else:
            clusters[-1].append(position)
    leads = [
        event - min(alert for alert in alerts if 1 <= event - alert <= horizon)
        for event in eligible
        if any(1 <= event - alert <= horizon for alert in alerts)
    ]
    matched_onsets = len(leads)
    matched_clusters = 0
    for cluster in clusters:
        if any(
            1 <= event - alert <= horizon
            for event in eligible
            for alert in cluster
        ):
            matched_clusters += 1
    if len(valid) > 1 and "date" in valid:
        years = max((valid["date"].max() - valid["date"].min()).days / 365.25, 1 / 252)
    else:
        years = max(len(valid) / 252, 1 / 252)
    return {
        "eligible_onsets": len(eligible),
        "matched_onsets": matched_onsets,
        "onset_recall": matched_onsets / len(eligible) if eligible else 0.0,
        "alert_clusters": len(clusters),
        "matched_clusters": matched_clusters,
        "false_alert_clusters": len(clusters) - matched_clusters,
        "cluster_precision": matched_clusters / len(clusters) if clusters else 0.0,
        "false_alert_clusters_per_year": (len(clusters) - matched_clusters) / years,
        "alert_day_rate": len(alerts) / len(valid) if len(valid) else 0.0,
        "median_lead_days": float(np.median(leads)) if leads else float("nan"),
    }


def episode_diagnostics(
    frame: pd.DataFrame,
    onset_positions: Sequence[int],
    scores: pd.Series,
    cutoff: float,
    *,
    horizon: int = HORIZON,
    severity_horizon: int = 60,
) -> list[dict[str, object]]:
    aligned = scores.reindex(frame.index)
    details: list[dict[str, object]] = []
    last_position = int(frame["_position"].max())
    for event in onset_positions:
        event_rows = frame.loc[frame["_position"] == int(event)]
        if event_rows.empty:
            continue
        warning = frame.loc[
            frame["_position"].between(int(event) - horizon, int(event) - 1)
            & aligned.notna()
        ]
        warning_scores = aligned.loc[warning.index]
        alert_rows = warning.loc[warning_scores >= cutoff]
        severity = frame.loc[
            frame["_position"].between(int(event), int(event) + severity_horizon),
            "drawdown_252",
        ]
        hit = not alert_rows.empty
        first_alert = alert_rows.iloc[0] if hit else None
        details.append(
            {
                "onset_date": str(event_rows.iloc[0]["date"].date()),
                "eligible": not warning.empty,
                "prior_20d_max_score": (
                    float(warning_scores.max()) if not warning_scores.empty else None
                ),
                "hit": hit,
                "first_alert_date": str(first_alert["date"].date()) if hit else None,
                "lead_days": int(event - first_alert["_position"]) if hit else None,
                "min_drawdown_next_60d": (
                    float(severity.min()) if not severity.empty else None
                ),
                "severity_window_complete": int(event) + severity_horizon <= last_position,
            }
        )
    return details


def severity_summary(details: Sequence[dict[str, object]]) -> dict[str, object]:
    complete = [
        item
        for item in details
        if item["eligible"]
        and item["severity_window_complete"]
        and item["min_drawdown_next_60d"] is not None
    ]
    hit_values = [float(item["min_drawdown_next_60d"]) for item in complete if item["hit"]]
    miss_values = [float(item["min_drawdown_next_60d"]) for item in complete if not item["hit"]]
    severe = [item for item in complete if float(item["min_drawdown_next_60d"]) <= -0.10]
    severe_hits = sum(bool(item["hit"]) for item in severe)
    return {
        "complete_eligible_episodes": len(complete),
        "hit_median_min_drawdown_60d": float(np.median(hit_values)) if hit_values else None,
        "miss_median_min_drawdown_60d": float(np.median(miss_values)) if miss_values else None,
        "severe_10pct_episodes": len(severe),
        "severe_10pct_hits": severe_hits,
        "severe_10pct_recall": severe_hits / len(severe) if severe else None,
    }


def moving_block_auc_ci(
    target: np.ndarray,
    scores: np.ndarray,
    *,
    block_size: int = 40,
    samples: int = 500,
    random_state: int = 42,
) -> tuple[float, float]:
    if len(target) < 2 or np.unique(target).size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(random_state)
    block = min(block_size, len(target))
    estimates: list[float] = []
    for _ in range(samples):
        selected: list[int] = []
        while len(selected) < len(target):
            start = int(rng.integers(0, len(target) - block + 1))
            selected.extend(range(start, start + block))
        index = np.asarray(selected[: len(target)])
        if np.unique(target[index]).size == 2:
            estimates.append(float(roc_auc_score(target[index], scores[index])))
    if not estimates:
        return float("nan"), float("nan")
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def paired_auc_delta_ci(
    target: np.ndarray,
    candidate_scores: np.ndarray,
    baseline_scores: np.ndarray,
    *,
    block_size: int = 40,
    samples: int = 500,
    random_state: int = 42,
) -> dict[str, float]:
    if len(target) < 2 or np.unique(target).size < 2:
        return {"delta_auc": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    delta = float(
        roc_auc_score(target, candidate_scores)
        - roc_auc_score(target, baseline_scores)
    )
    rng = np.random.default_rng(random_state)
    block = min(block_size, len(target))
    estimates: list[float] = []
    for _ in range(samples):
        selected: list[int] = []
        while len(selected) < len(target):
            start = int(rng.integers(0, len(target) - block + 1))
            selected.extend(range(start, start + block))
        index = np.asarray(selected[: len(target)])
        if np.unique(target[index]).size == 2:
            estimates.append(
                float(
                    roc_auc_score(target[index], candidate_scores[index])
                    - roc_auc_score(target[index], baseline_scores[index])
                )
            )
    low, high = np.quantile(estimates, [0.025, 0.975]) if estimates else (np.nan, np.nan)
    return {"delta_auc": delta, "ci_low": float(low), "ci_high": float(high)}


def evaluate_score(
    frame: pd.DataFrame,
    mask: pd.Series,
    scores: pd.Series,
    onset_positions: Sequence[int],
    *,
    bootstrap_samples: int = 500,
    fixed_cutoff: float | None = None,
) -> dict[str, object]:
    sample = frame.loc[mask & scores.notna()].copy()
    sample_scores = scores.reindex(sample.index)
    target = sample["target"].astype(int).to_numpy()
    values = sample_scores.to_numpy()
    auc = float(roc_auc_score(target, values)) if np.unique(target).size == 2 else float("nan")
    ap = float(average_precision_score(target, values)) if target.sum() else float("nan")
    low, high = moving_block_auc_ci(target, values, samples=bootstrap_samples)
    grid = (
        (fixed_cutoff,)
        if fixed_cutoff is not None
        else tuple(round(float(value), 2) for value in np.arange(0.50, 0.981, 0.01))
    )
    choices = []
    for cutoff in grid:
        metrics = event_metrics(sample, onset_positions, sample_scores, float(cutoff))
        feasible = (
            metrics["false_alert_clusters_per_year"] <= 2
            and metrics["alert_day_rate"] <= 0.20
        )
        choices.append((feasible, float(cutoff), metrics))
    feasible = [choice for choice in choices if choice[0]]
    pool = feasible or choices
    _, cutoff, events = max(
        pool,
        key=lambda item: (
            item[2]["onset_recall"],
            item[2]["cluster_precision"],
            -item[2]["false_alert_clusters_per_year"],
            item[1],
        ),
    )
    return {
        "n": len(sample),
        "event_rate": float(target.mean()) if len(target) else 0.0,
        "auc": auc,
        "average_precision": ap,
        "auc_ci_low": low,
        "auc_ci_high": high,
        "alert_cutoff": cutoff,
        "threshold_feasible": bool(feasible),
        "event_metrics": events,
    }


def _split_mask(frame: pd.DataFrame, start: str, end: str) -> pd.Series:
    end_rows = frame.index[frame["date"] <= pd.Timestamp(end)]
    if len(end_rows) <= HORIZON:
        return pd.Series(False, index=frame.index)
    end_position = int(frame.loc[end_rows[-1], "_position"])
    return (
        (frame["date"] >= pd.Timestamp(start))
        & (frame["date"] <= pd.Timestamp(end))
        & (frame["label_end_position"] <= end_position)
        & frame["at_risk"].astype(bool)
        & frame["target"].notna()
    )


def run_research(
    database: Path,
    *,
    bootstrap_samples: int = 500,
) -> dict[str, object]:
    frame = build_frame(load_prices(database))
    dev_mask = _split_mask(frame, "2008-01-01", "2019-12-31")
    prediction_mask = (frame["date"] <= pd.Timestamp("2019-12-31")) & frame["at_risk"]
    dev_onsets = frame.loc[
        frame["onset"] & frame["date"].between("2008-01-01", "2019-12-31"),
        "_position",
    ].astype(int).tolist()
    results: dict[str, dict[str, object]] = {}
    raw_scores: dict[str, pd.Series] = {}
    for candidate in candidate_specs():
        raw = walk_forward_scores(frame, candidate, prediction_mask)
        raw_scores[candidate.name] = raw
        score = causal_trailing_percentile(raw)
        results[candidate.name] = evaluate_score(
            frame, dev_mask, score, dev_onsets, bootstrap_samples=bootstrap_samples
        )

    baseline_scores = {
        name: causal_trailing_percentile(frame[name].where(frame["at_risk"]))
        for name in ("baseline_vix", "baseline_vol", "baseline_neg_ma200")
    }
    baselines = {
        name: evaluate_score(
            frame, dev_mask, score, dev_onsets, bootstrap_samples=bootstrap_samples
        )
        for name, score in baseline_scores.items()
    }
    winner = max(
        results,
        key=lambda name: (
            results[name]["threshold_feasible"],
            results[name]["event_metrics"]["onset_recall"],
            results[name]["auc"],
        ),
    )
    winner_result = results[winner]
    strongest_baseline = max(float(item["auc"]) for item in baselines.values())
    winner_score = causal_trailing_percentile(raw_scores[winner])
    block_aucs = []
    for start, end in (("2008-01-01", "2013-12-31"), ("2014-01-01", "2019-12-31")):
        subset = frame.loc[_split_mask(frame, start, end) & winner_score.notna()]
        block_aucs.append(
            float(roc_auc_score(subset["target"], winner_score.loc[subset.index]))
            if len(subset) and subset["target"].nunique() == 2
            else float("nan")
        )
    events = winner_result["event_metrics"]
    gates = {
        "dev_onsets_at_least_15": events["eligible_onsets"] >= 15,
        "auc_at_least_0_60": winner_result["auc"] >= 0.60,
        "auc_ci_low_above_0_50": winner_result["auc_ci_low"] > 0.50,
        "recall_at_least_0_60": events["onset_recall"] >= 0.60,
        "false_clusters_per_year_at_most_2": events["false_alert_clusters_per_year"] <= 2,
        "auc_delta_vs_best_baseline_at_least_0_02": winner_result["auc"] - strongest_baseline >= 0.02,
        "both_dev_blocks_auc_above_0_50": all(np.isfinite(block_aucs)) and min(block_aucs) > 0.50,
    }
    passed = all(gates.values())
    confirmation: dict[str, object] = {"status": "SKIPPED_GATE_FAILURE"}
    confirmation_gates: dict[str, bool] | None = None
    if passed:
        candidate = next(item for item in candidate_specs() if item.name == winner)
        confirm_mask = _split_mask(frame, "2020-01-01", "2099-12-31")
        confirm_raw = walk_forward_scores(frame, candidate, confirm_mask)
        combined_raw = raw_scores[winner].combine_first(confirm_raw)
        confirm_score = causal_trailing_percentile(combined_raw)
        confirm_onsets = frame.loc[
            frame["onset"] & (frame["date"] >= pd.Timestamp("2020-01-01")), "_position"
        ].astype(int).tolist()
        confirmation = evaluate_score(
            frame,
            confirm_mask,
            confirm_score,
            confirm_onsets,
            bootstrap_samples=bootstrap_samples,
            fixed_cutoff=float(winner_result["alert_cutoff"]),
        )
        confirmation["status"] = "RUN_WINNER_ONLY"
    return {
        "protocol": {
            "target": "NQ dd252 >= -3%, onset is first NQ dd252 <= -5%; re-arm only after recovery to -3%",
            "label": "at-risk t has onset in t+1..t+20",
            "purge": "training label end position must be <= prediction position",
            "normalization": "current score percentile versus prior trailing 756 finite scores only",
            "development": "2008-01-01..2019-12-31; final horizon purged",
            "confirmation": "2020+ winner only, and only after every hard gate passes",
            "selection": "among cutoff grid meeting false alerts <=2/year and alert days <=20%, maximize onset recall",
        },
        "data": {"start": str(frame["date"].min().date()), "end": str(frame["date"].max().date())},
        "development": results,
        "baselines": baselines,
        "winner": winner,
        "winner_block_aucs": block_aucs,
        "gates": gates,
        "gates_passed": passed,
        "confirmation": confirmation,
    }


def run_fixed_core_research(
    database: Path,
    *,
    bootstrap_samples: int = 500,
) -> dict[str, object]:
    """Fourth and final development trial: a pre-signed rank ensemble."""
    frame = build_frame(load_prices(database))
    dev_mask = _split_mask(frame, "2008-01-01", "2019-12-31")
    dev_onsets = frame.loc[
        frame["onset"] & frame["date"].between("2008-01-01", "2019-12-31"),
        "_position",
    ].astype(int).tolist()
    candidate_score = fixed_core_rank_score(frame)
    baseline_scores = {
        name: causal_trailing_percentile(frame[name].where(frame["at_risk"]))
        for name in ("baseline_vix", "baseline_vol", "baseline_neg_ma200")
    }
    common_mask = dev_mask & candidate_score.notna()
    for score in baseline_scores.values():
        common_mask &= score.notna()

    candidate_result = evaluate_score(
        frame,
        common_mask,
        candidate_score,
        dev_onsets,
        bootstrap_samples=bootstrap_samples,
    )
    development_details = episode_diagnostics(
        frame,
        dev_onsets,
        candidate_score,
        float(candidate_result["alert_cutoff"]),
    )
    candidate_result["episode_diagnostics"] = development_details
    candidate_result["severity_summary"] = severity_summary(development_details)
    baselines = {
        name: evaluate_score(
            frame,
            common_mask,
            score,
            dev_onsets,
            bootstrap_samples=bootstrap_samples,
        )
        for name, score in baseline_scores.items()
    }
    strongest_baseline = max(
        baselines,
        key=lambda name: (
            baselines[name]["event_metrics"]["onset_recall"],
            baselines[name]["auc"],
        ),
    )
    comparison_rows = frame.loc[common_mask]
    paired = paired_auc_delta_ci(
        comparison_rows["target"].astype(int).to_numpy(),
        candidate_score.loc[comparison_rows.index].to_numpy(),
        baseline_scores[strongest_baseline].loc[comparison_rows.index].to_numpy(),
        samples=bootstrap_samples,
    )
    block_aucs: dict[str, float] = {}
    for name, start, end in (
        ("2008_2013", "2008-01-01", "2013-12-31"),
        ("2014_2019", "2014-01-01", "2019-12-31"),
    ):
        subset = frame.loc[
            common_mask & frame["date"].between(pd.Timestamp(start), pd.Timestamp(end))
        ]
        block_aucs[name] = (
            float(roc_auc_score(subset["target"], candidate_score.loc[subset.index]))
            if len(subset) and subset["target"].nunique() == 2
            else float("nan")
        )

    events = candidate_result["event_metrics"]
    baseline_events = baselines[strongest_baseline]["event_metrics"]
    event_rate = float(candidate_result["event_rate"])
    ap_lift = (
        float(candidate_result["average_precision"]) / event_rate
        if event_rate
        else float("nan")
    )
    gates = {
        "dev_onsets_at_least_15": events["eligible_onsets"] >= 15,
        "auc_at_least_0_60": candidate_result["auc"] >= 0.60,
        "auc_ci_low_above_0_50": candidate_result["auc_ci_low"] > 0.50,
        "average_precision_lift_at_least_1_25": ap_lift >= 1.25,
        "recall_at_least_0_60": events["onset_recall"] >= 0.60,
        "false_clusters_per_year_at_most_2": events["false_alert_clusters_per_year"] <= 2,
        "alert_days_at_most_0_20": events["alert_day_rate"] <= 0.20,
        "median_lead_at_least_5": np.isfinite(events["median_lead_days"])
        and events["median_lead_days"] >= 5,
        "additional_onsets_vs_best_baseline_at_least_2": (
            events["matched_onsets"] - baseline_events["matched_onsets"] >= 2
        ),
        "both_dev_blocks_auc_above_0_52": all(np.isfinite(list(block_aucs.values())))
        and min(block_aucs.values()) > 0.52,
    }
    passed = all(gates.values())
    confirmation: dict[str, object] = {"status": "SKIPPED_GATE_FAILURE"}
    confirmation_gates: dict[str, bool] | None = None
    if passed:
        confirm_mask = _split_mask(frame, "2020-01-01", "2099-12-31") & candidate_score.notna()
        for score in baseline_scores.values():
            confirm_mask &= score.notna()
        confirm_onsets = frame.loc[
            frame["onset"] & (frame["date"] >= pd.Timestamp("2020-01-01")), "_position"
        ].astype(int).tolist()
        confirmation = {
            "status": "RUN_FIXED_WINNER_ONLY",
            "candidate": evaluate_score(
                frame,
                confirm_mask,
                candidate_score,
                confirm_onsets,
                bootstrap_samples=bootstrap_samples,
                fixed_cutoff=float(candidate_result["alert_cutoff"]),
            ),
            "baselines": {
                name: evaluate_score(
                    frame,
                    confirm_mask,
                    score,
                    confirm_onsets,
                    bootstrap_samples=bootstrap_samples,
                    fixed_cutoff=float(baselines[name]["alert_cutoff"]),
                )
                for name, score in baseline_scores.items()
            },
        }
        confirmation_details = episode_diagnostics(
            frame,
            confirm_onsets,
            candidate_score,
            float(candidate_result["alert_cutoff"]),
        )
        confirmation["candidate"]["episode_diagnostics"] = confirmation_details
        confirmation["candidate"]["severity_summary"] = severity_summary(
            confirmation_details
        )
        confirmed = confirmation["candidate"]
        confirmed_events = confirmed["event_metrics"]
        confirmation_gates = {
            "auc_at_least_0_60": confirmed["auc"] >= 0.60,
            "auc_ci_low_above_0_50": confirmed["auc_ci_low"] > 0.50,
            "recall_at_least_0_60": confirmed_events["onset_recall"] >= 0.60,
            "false_clusters_per_year_at_most_2": (
                confirmed_events["false_alert_clusters_per_year"] <= 2
            ),
            "alert_days_at_most_0_20": confirmed_events["alert_day_rate"] <= 0.20,
            "median_lead_at_least_5": np.isfinite(confirmed_events["median_lead_days"])
            and confirmed_events["median_lead_days"] >= 5,
        }
    common_rows = frame.loc[common_mask]
    status = "DEVELOPMENT_GATE_FAILURE"
    if passed and confirmation_gates and all(confirmation_gates.values()):
        status = "SEEN_CONFIRMATION_ACTION_POLICY_PASS"
    elif passed and confirmation_gates:
        ranking_gates = (
            confirmation_gates["auc_at_least_0_60"]
            and confirmation_gates["auc_ci_low_above_0_50"]
        )
        status = (
            "VALID_RISK_RANKING_ACTION_THRESHOLD_FAILED"
            if ranking_gates
            else "CONFIRMATION_FAILURE"
        )
    return {
        "protocol": {
            "trial": "4_of_4_fixed_core_rank",
            "prior_trials": ["logit_macro", "logit_cross", "shallow_hist_cross"],
            "candidate": "mean of causal ranks: VIX z252, 5d VIX change, negative HYG/IEF 20d return",
            "target": "NQ dd252 >= -3%, onset is first NQ dd252 <= -5%; re-arm only after recovery to -3%",
            "label": "at-risk t has onset in t+1..t+20; action is no earlier than t+1",
            "purge": "training/evaluation label window must be complete",
            "development": "2008-01-01..2019-12-31; common candidate/baseline dates only",
            "confirmation": "2020+ seen confirmation only after every hard gate passes",
            "alert_budget": "cutoff searched over 0.50..0.98 on development with <=2 false clusters/year and <=20% alert days; frozen thereafter",
        },
        "data": {
            "start": str(frame["date"].min().date()),
            "end": str(frame["date"].max().date()),
            "raw_development_onsets": len(dev_onsets),
            "common_development_rows": int(common_mask.sum()),
            "common_development_start": str(common_rows["date"].min().date()),
            "common_development_end": str(common_rows["date"].max().date()),
        },
        "candidate": candidate_result,
        "baselines": baselines,
        "strongest_action_baseline": strongest_baseline,
        "paired_auc_delta_vs_strongest_action_baseline": paired,
        "average_precision_lift": ap_lift,
        "block_aucs": block_aucs,
        "gates": gates,
        "gates_passed": passed,
        "confirmation": confirmation,
        "confirmation_gates": confirmation_gates,
        "status": status,
    }


def write_shadow_scores(
    database: Path,
    *,
    model_version: str = SHADOW_MODEL_VERSION,
    alert_cutoff: float = SHADOW_ALERT_CUTOFF,
) -> int:
    frame = build_frame(load_prices(database))
    components = fixed_core_rank_components(frame)
    scores = fixed_core_rank_score(frame)
    generated_at = datetime.now(timezone.utc).isoformat()
    records: list[tuple[object, ...]] = []
    for index, row in frame.iterrows():
        if not np.isfinite(row["drawdown_252"]):
            continue
        score = float(scores.loc[index]) if np.isfinite(scores.loc[index]) else None
        at_risk = bool(row["at_risk"])
        band = (
            "DORMANT"
            if not at_risk
            else "WARMUP"
            if score is None
            else "ALERT_SHADOW"
            if score >= alert_cutoff
            else "NORMAL_SHADOW"
        )
        values = [
            float(components.loc[index, column])
            if np.isfinite(components.loc[index, column])
            else None
            for column in ("vix_level", "vix_acceleration", "credit_risk")
        ]
        records.append(
            (
                str(row["date"].date()),
                model_version,
                score,
                int(at_risk),
                *values,
                band,
                generated_at,
            )
        )
    with closing(sqlite3.connect(database)) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS drawdown_hazard_scores (
                date TEXT NOT NULL,
                model_version TEXT NOT NULL,
                score REAL,
                at_risk INTEGER NOT NULL CHECK (at_risk IN (0, 1)),
                vix_level_rank REAL,
                vix_acceleration_rank REAL,
                credit_risk_rank REAL,
                risk_band TEXT NOT NULL,
                generated_at TEXT NOT NULL,
                PRIMARY KEY (date, model_version)
            )
            """
        )
        connection.executemany(
            """
            INSERT INTO drawdown_hazard_scores(
                date, model_version, score, at_risk, vix_level_rank,
                vix_acceleration_rank, credit_risk_rank, risk_band, generated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(date, model_version) DO UPDATE SET
                score=excluded.score,
                at_risk=excluded.at_risk,
                vix_level_rank=excluded.vix_level_rank,
                vix_acceleration_rank=excluded.vix_acceleration_rank,
                credit_risk_rank=excluded.credit_risk_rank,
                risk_band=excluded.risk_band,
                generated_at=excluded.generated_at
            """,
            records,
        )
        connection.commit()
    return len(records)


def get_shadow_hazard_status(
    database: Path,
    as_of_date: str,
    *,
    model_version: str = SHADOW_MODEL_VERSION,
) -> dict[str, object]:
    with closing(sqlite3.connect(database)) as connection:
        row = connection.execute(
            """
            SELECT date, model_version, score, at_risk, vix_level_rank,
                   vix_acceleration_rank, credit_risk_rank, risk_band
            FROM drawdown_hazard_scores
            WHERE model_version=? AND date<=?
            ORDER BY date DESC LIMIT 1
            """,
            (model_version, str(pd.Timestamp(as_of_date).date())),
        ).fetchone()
    if row is None:
        raise ValueError(f"no shadow hazard score on or before {as_of_date}")
    keys = (
        "date",
        "model_version",
        "score",
        "at_risk",
        "vix_level_rank",
        "vix_acceleration_rank",
        "credit_risk_rank",
        "risk_band",
    )
    result = dict(zip(keys, row))
    result["at_risk"] = bool(result["at_risk"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Research NQ drawdown-onset hazard")
    parser.add_argument("database", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--mode", choices=("learned", "fixed-core"), default="learned")
    parser.add_argument("--write-shadow", action="store_true")
    args = parser.parse_args()
    report = (
        run_fixed_core_research(args.database, bootstrap_samples=args.bootstrap_samples)
        if args.mode == "fixed-core"
        else run_research(args.database, bootstrap_samples=args.bootstrap_samples)
    )
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8"
    )
    print(f"mode={args.mode} gates_passed={report['gates_passed']}")
    print(f"report={args.output}")
    if args.write_shadow:
        cutoff = (
            float(report["candidate"]["alert_cutoff"])
            if args.mode == "fixed-core"
            else SHADOW_ALERT_CUTOFF
        )
        count = write_shadow_scores(args.database, alert_cutoff=cutoff)
        print(f"shadow_scores_upserted={count} model={SHADOW_MODEL_VERSION}")


if __name__ == "__main__":
    main()
