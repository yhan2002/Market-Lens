from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sector_rotation import (
    DEFAULT_REGIME_MODEL,
    SECTOR_ETFS,
    build_rrg_signals,
    load_price_matrix,
    load_regime_probabilities,
)


BASE_FEATURE_COLUMNS = [
    "rrg_ratio",
    "rrg_momentum",
    "rrg_ratio_delta_5",
    "rrg_momentum_delta_5",
    "rrg_ratio_delta_20",
    "rrg_momentum_delta_20",
    "relative_return_20",
    "relative_return_60",
    "relative_return_120",
    "relative_volatility_20",
    "relative_drawdown_60",
]
FEATURE_COLUMNS = [
    *BASE_FEATURE_COLUMNS,
    "p_sideways",
    "p_bear",
]


def build_sector_feature_frame(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    benchmark: str = "SPY",
    horizon: int = 20,
) -> pd.DataFrame:
    """Build causal sector predictors and a future sector-excess target."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    sectors = [ticker for ticker in SECTOR_ETFS if ticker in prices]
    if benchmark not in prices or not sectors:
        raise ValueError("benchmark and sector price columns are required")

    clean = prices.sort_index().apply(pd.to_numeric, errors="coerce")
    forward_returns = clean.shift(-horizon).div(clean).sub(1)
    equal_sector_forward = forward_returns[sectors].mean(axis=1)
    future_dates = pd.Series(clean.index, index=clean.index).shift(-horizon)
    benchmark_return = clean[benchmark].pct_change()
    benchmark_volatility = benchmark_return.rolling(20).std()
    benchmark_drawdown = clean[benchmark].div(clean[benchmark].rolling(60).max()).sub(1)

    signal_features = signals.sort_values(["ticker", "date"]).copy()
    grouped = signal_features.groupby("ticker")
    signal_features["rrg_ratio"] = signal_features["rs_ratio"] - 100
    signal_features["rrg_momentum"] = signal_features["rs_momentum"] - 100
    for lag in (5, 20):
        signal_features[f"rrg_ratio_delta_{lag}"] = grouped["rs_ratio"].diff(lag)
        signal_features[f"rrg_momentum_delta_{lag}"] = grouped["rs_momentum"].diff(lag)

    rows: list[pd.DataFrame] = []
    for ticker in sectors:
        sector_return = clean[ticker].pct_change()
        part = pd.DataFrame(
            {
                "date": clean.index,
                "ticker": ticker,
                "relative_return_20": clean[ticker].pct_change(20)
                - clean[benchmark].pct_change(20),
                "relative_return_60": clean[ticker].pct_change(60)
                - clean[benchmark].pct_change(60),
                "relative_return_120": clean[ticker].pct_change(120)
                - clean[benchmark].pct_change(120),
                "relative_volatility_20": sector_return.rolling(20).std()
                - benchmark_volatility,
                "relative_drawdown_60": clean[ticker].div(clean[ticker].rolling(60).max()).sub(1)
                - benchmark_drawdown,
                "future_sector_return": forward_returns[ticker],
                "future_equal_sector_return": equal_sector_forward,
                "target_excess_20d": forward_returns[ticker] - equal_sector_forward,
                "label_end": future_dates,
            }
        )
        rows.append(part)
    return pd.concat(rows, ignore_index=True).merge(
        signal_features[
            [
                "date",
                "ticker",
                "shadow_score",
                "rrg_ratio",
                "rrg_momentum",
                "rrg_ratio_delta_5",
                "rrg_momentum_delta_5",
                "rrg_ratio_delta_20",
                "rrg_momentum_delta_20",
            ]
        ],
        on=["date", "ticker"],
        how="inner",
    )


def build_forecast_frame(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    regime: pd.DataFrame,
    *,
    benchmark: str = "SPY",
    horizon: int = 20,
) -> pd.DataFrame:
    """Build the sector feature frame with backward-looking HMM context."""
    frame = build_sector_feature_frame(
        prices,
        signals,
        benchmark=benchmark,
        horizon=horizon,
    )
    regime_context = regime[["date", "p_sideways", "p_bear"]].sort_values("date")
    return pd.merge_asof(
        frame.sort_values("date"),
        regime_context,
        on="date",
        direction="backward",
    ).sort_values(["date", "ticker"])


def _weekly_rows(frame: pd.DataFrame) -> pd.DataFrame:
    week_end_dates = frame.groupby(frame["date"].dt.to_period("W-FRI"))["date"].max()
    return frame.loc[frame["date"].isin(week_end_dates)].copy()


def fit_leading_forecast(
    frame: pd.DataFrame,
    *,
    as_of: str | pd.Timestamp | None = None,
    min_train_weeks: int = 156,
    alpha: float = 10.0,
    top_k: int = 3,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Fit only labels known before as_of, then rank the current sector cross-section."""
    if min_train_weeks < 52 or alpha <= 0 or top_k < 1:
        raise ValueError("min_train_weeks>=52, alpha>0 and top_k>=1 are required")
    selected_features = list(feature_columns)
    if not selected_features:
        raise ValueError("feature_columns must not be empty")
    cutoff = pd.Timestamp(as_of) if as_of is not None else frame["date"].max()
    eligible = frame.loc[frame["date"] <= cutoff].copy()
    if eligible.empty:
        raise ValueError("no forecast features are available on or before as_of")
    if as_of is None:
        weekly = _weekly_rows(eligible)
        completed = weekly["date"].dt.to_period("W-FRI").dt.end_time.dt.normalize() <= cutoff.normalize()
        weekly = weekly.loc[completed]
        if weekly.empty:
            raise ValueError("no completed weekly signal is available")
        current_date = weekly["date"].max()
    else:
        current_date = eligible["date"].max()
    current = eligible.loc[eligible["date"] == current_date].dropna(
        subset=selected_features
    )
    if len(current) < top_k:
        raise ValueError("not enough sectors have complete current features")

    weekly = _weekly_rows(eligible)
    training = weekly.loc[
        weekly["label_end"].notna()
        & (weekly["label_end"] < current_date)
        & weekly["target_excess_20d"].notna()
    ].dropna(subset=selected_features)
    train_weeks = training["date"].nunique()
    if train_weeks < min_train_weeks:
        raise ValueError(
            f"need {min_train_weeks} known training weeks; found {train_weeks}"
        )

    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(training[selected_features], training["target_excess_20d"])
    forecast = current[["date", "ticker", "shadow_score"]].copy()
    forecast.insert(2, "sector", forecast["ticker"].map(SECTOR_ETFS))
    forecast["predicted_excess_20d"] = model.predict(current[selected_features])
    forecast["forecast_score"] = forecast["predicted_excess_20d"].rank(pct=True)
    forecast["forecast_rank"] = (
        forecast["predicted_excess_20d"].rank(ascending=False, method="min").astype(int)
    )
    positive = forecast["predicted_excess_20d"] > 0
    forecast["signal"] = np.select(
        [positive & (forecast["forecast_rank"] <= top_k), positive],
        ["MODEL_TOP3", "POSITIVE_WATCH"],
        default="NEGATIVE",
    )
    forecast = forecast.sort_values("forecast_rank")
    coefficients = model.named_steps["ridge"].coef_
    has_hmm_context = {"p_sideways", "p_bear"}.issubset(selected_features)
    metadata: dict[str, object] = {
        "as_of": current_date.date().isoformat(),
        "target": "next-20-trading-day sector return minus equal-sector return",
        "training_rows": len(training),
        "training_weeks": train_weeks,
        "training_start": training["date"].min().date().isoformat(),
        "training_end": training["date"].max().date().isoformat(),
        "latest_training_label_end": training["label_end"].max().date().isoformat(),
        "alpha": alpha,
        "top_k": top_k,
        "context_type": "HMM" if has_hmm_context else "NONE",
        "fixed_regime_context": (
            {
                "p_bull": float(
                    1 - current["p_sideways"].iloc[0] - current["p_bear"].iloc[0]
                ),
                "p_sideways": float(current["p_sideways"].iloc[0]),
                "p_bear": float(current["p_bear"].iloc[0]),
            }
            if has_hmm_context
            else None
        ),
        "standardized_coefficients": {
            feature: float(coefficient)
            for feature, coefficient in zip(selected_features, coefficients, strict=True)
        },
    }
    return forecast, metadata


def _block_bootstrap_mean_ci(
    values: pd.Series,
    *,
    block_weeks: int = 4,
    samples: int = 2000,
    random_state: int = 42,
) -> tuple[float | None, float | None]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < block_weeks or block_weeks < 1 or samples < 100:
        return None, None
    data = clean.to_numpy()
    rng = np.random.default_rng(random_state)
    starts = np.arange(len(data) - block_weeks + 1)
    means = []
    blocks_needed = int(np.ceil(len(data) / block_weeks))
    for _ in range(samples):
        selected = rng.choice(starts, size=blocks_needed, replace=True)
        sample = np.concatenate([data[start : start + block_weeks] for start in selected])
        means.append(float(sample[: len(data)].mean()))
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def _safe_rank_correlation(scores: pd.Series, targets: pd.Series) -> float:
    ranked = pd.concat(
        [
            pd.to_numeric(scores, errors="coerce").rank(),
            pd.to_numeric(targets, errors="coerce").rank(),
        ],
        axis=1,
    ).dropna()
    if len(ranked) < 2 or (ranked.nunique() < 2).any():
        return float("nan")
    return float(ranked.iloc[:, 0].corr(ranked.iloc[:, 1]))


def _top_k_tie_weights(
    frame: pd.DataFrame, score_column: str, *, top_k: int
) -> pd.Series:
    if top_k < 1 or len(frame) < top_k:
        raise ValueError("top_k must be positive and no larger than the cross-section")
    tickers = frame["ticker"].astype(str)
    if tickers.duplicated().any():
        raise ValueError("ticker must be unique within the cross-section")
    scores = pd.Series(
        pd.to_numeric(frame[score_column], errors="coerce").to_numpy(),
        index=tickers,
        dtype=float,
    )
    if not np.isfinite(scores).all():
        raise ValueError(f"{score_column} must be finite")
    cutoff = float(scores.nlargest(top_k).iloc[-1])
    above = scores > cutoff
    tied = scores == cutoff
    remaining_slots = top_k - int(above.sum())
    weights = pd.Series(0.0, index=scores.index)
    weights.loc[above] = 1 / top_k
    weights.loc[tied] = remaining_slots / (top_k * int(tied.sum()))
    return weights


def _portfolio_turnover(current: pd.Series, previous: pd.Series | None) -> float:
    if previous is None:
        return 1.0
    tickers = current.index.union(previous.index)
    return float(
        current.reindex(tickers, fill_value=0).sub(
            previous.reindex(tickers, fill_value=0)
        ).abs().sum()
        / 2
    )


def _format_weights(weights: pd.Series) -> str:
    return ",".join(
        f"{ticker}:{weight:.6f}"
        for ticker, weight in weights[weights > 0].sort_index().items()
    )


def walk_forward_validate(
    frame: pd.DataFrame,
    *,
    min_train_weeks: int = 156,
    alpha: float = 10.0,
    top_k: int = 3,
    cost_bps: float = 10.0,
    block_weeks: int = 4,
    bootstrap_samples: int = 2000,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Evaluate weekly forecasts with an embargo and overlapping-label block CI."""
    if cost_bps < 0 or block_weeks < 1 or bootstrap_samples < 100:
        raise ValueError("cost_bps>=0, block_weeks>=1 and bootstrap_samples>=100 are required")
    weekly = _weekly_rows(frame)
    dates = weekly.loc[weekly["target_excess_20d"].notna(), "date"].drop_duplicates()
    rows: list[dict[str, object]] = []
    previous_model: set[str] = set()
    previous_rrg: pd.Series | None = None
    for current_date in dates:
        try:
            forecast, _ = fit_leading_forecast(
                frame,
                as_of=current_date,
                min_train_weeks=min_train_weeks,
                alpha=alpha,
                top_k=top_k,
                feature_columns=feature_columns,
            )
        except ValueError:
            continue
        actual = weekly.loc[weekly["date"] == current_date].dropna(
            subset=["target_excess_20d"]
        )
        scored = forecast.merge(
            actual[["ticker", "target_excess_20d", "shadow_score"]],
            on="ticker",
            suffixes=("", "_actual"),
        )
        if len(scored) < top_k:
            continue
        model_selected = set(scored.nsmallest(top_k, "forecast_rank")["ticker"])
        rrg_weights = _top_k_tie_weights(
            scored, "shadow_score_actual", top_k=top_k
        )
        rrg_selected = set(rrg_weights[rrg_weights > 0].index)
        model_turnover = 1.0 if not previous_model else 1 - len(model_selected & previous_model) / top_k
        rrg_turnover = _portfolio_turnover(rrg_weights, previous_rrg)
        model_gross = float(
            scored.loc[scored["ticker"].isin(model_selected), "target_excess_20d"].mean()
        )
        rrg_targets = scored.set_index("ticker")["target_excess_20d"]
        rrg_gross = float(
            rrg_targets.reindex(rrg_weights.index).mul(rrg_weights).sum()
        )
        rows.append(
            {
                "date": current_date,
                "model_selected": ",".join(sorted(model_selected)),
                "rrg_selected": ",".join(sorted(rrg_selected)),
                "rrg_selection_weights": _format_weights(rrg_weights),
                "rrg_boundary_tie": len(rrg_selected) > top_k,
                "model_gross_excess": model_gross,
                "rrg_gross_excess": rrg_gross,
                "model_turnover": model_turnover,
                "rrg_turnover": rrg_turnover,
                "model_net_excess": model_gross - model_turnover * cost_bps / 10_000,
                "rrg_net_excess": rrg_gross - rrg_turnover * cost_bps / 10_000,
                "model_rank_ic": _safe_rank_correlation(
                    scored["predicted_excess_20d"], scored["target_excess_20d"]
                ),
                "rrg_rank_ic": _safe_rank_correlation(
                    scored["shadow_score_actual"], scored["target_excess_20d"]
                ),
            }
        )
        previous_model, previous_rrg = model_selected, rrg_weights
    observations = pd.DataFrame(rows)
    if observations.empty:
        raise ValueError("no walk-forward observations; reduce min_train_weeks or check coverage")
    observations["net_delta_vs_rrg"] = (
        observations["model_net_excess"] - observations["rrg_net_excess"]
    )
    model_ci = _block_bootstrap_mean_ci(
        observations["model_net_excess"],
        block_weeks=block_weeks,
        samples=bootstrap_samples,
    )
    delta_ci = _block_bootstrap_mean_ci(
        observations["net_delta_vs_rrg"],
        block_weeks=block_weeks,
        samples=bootstrap_samples,
    )
    verdict = "INSUFFICIENT"
    if len(observations) >= 104:
        verdict = "INCONCLUSIVE"
        if model_ci[0] is not None and delta_ci[0] is not None and model_ci[0] > 0 and delta_ci[0] > 0:
            verdict = "PASS_SHADOW"
        elif model_ci[1] is not None and model_ci[1] <= 0:
            verdict = "FAIL_SHADOW"
    summary: dict[str, object] = {
        "verdict": verdict,
        "target": "next-20-trading-day sector return minus equal-sector return",
        "model": f"StandardScaler + Ridge(alpha={alpha:g})",
        "features": list(feature_columns),
        "minimum_training_weeks": min_train_weeks,
        "signal_frequency": "weekly; last trading session of W-FRI week",
        "bootstrap_block_weeks": block_weeks,
        "bootstrap_samples": bootstrap_samples,
        "top_k": top_k,
        "cost_bps_per_one_way_turnover": cost_bps,
        "observations": len(observations),
        "start_date": observations["date"].min().date().isoformat(),
        "end_date": observations["date"].max().date().isoformat(),
        "mean_model_net_excess": float(observations["model_net_excess"].mean()),
        "model_net_excess_95ci": list(model_ci),
        "mean_rrg_net_excess": float(observations["rrg_net_excess"].mean()),
        "mean_net_delta_vs_rrg": float(observations["net_delta_vs_rrg"].mean()),
        "net_delta_vs_rrg_95ci": list(delta_ci),
        "mean_model_rank_ic": float(observations["model_rank_ic"].mean()),
        "mean_rrg_rank_ic": float(observations["rrg_rank_ic"].mean()),
        "model_rank_ic_observations": int(observations["model_rank_ic"].notna().sum()),
        "rrg_rank_ic_observations": int(observations["rrg_rank_ic"].notna().sum()),
        "rrg_boundary_tie_observations": int(observations["rrg_boundary_tie"].sum()),
        "rrg_tie_policy": "equal weight across all names tied at the top-k boundary",
    }
    return summary, observations


def load_frame(database: Path, model_version: str) -> pd.DataFrame:
    prices = load_price_matrix(database)
    signals = build_rrg_signals(prices)
    regime = load_regime_probabilities(database, model_version)
    return build_forecast_frame(prices, signals, regime, horizon=20)


def main() -> None:
    parser = argparse.ArgumentParser(description="Leading S&P sector rotation shadow forecast")
    commands = parser.add_subparsers(dest="command", required=True)

    status = commands.add_parser("status", help="forecast the next sector leaders")
    status.add_argument("database", type=Path)
    status.add_argument("--date")
    status.add_argument("--model-version", default=DEFAULT_REGIME_MODEL)
    status.add_argument("--min-train-weeks", type=int, default=156)
    status.add_argument("--alpha", type=float, default=10.0)
    status.add_argument("--top-k", type=int, default=3)

    validate = commands.add_parser("validate", help="run strict weekly walk-forward validation")
    validate.add_argument("database", type=Path)
    validate.add_argument("output", type=Path)
    validate.add_argument("--model-version", default=DEFAULT_REGIME_MODEL)
    validate.add_argument("--min-train-weeks", type=int, default=156)
    validate.add_argument("--alpha", type=float, default=10.0)
    validate.add_argument("--top-k", type=int, default=3)
    validate.add_argument("--cost-bps", type=float, default=10.0)
    validate.add_argument("--block-weeks", type=int, default=4)
    validate.add_argument("--bootstrap-samples", type=int, default=2000)

    args = parser.parse_args()
    frame = load_frame(args.database, args.model_version)
    if args.command == "status":
        forecast, metadata = fit_leading_forecast(
            frame,
            as_of=args.date,
            min_train_weeks=args.min_train_weeks,
            alpha=args.alpha,
            top_k=args.top_k,
        )
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        print(forecast.to_string(index=False))
        return

    summary, observations = walk_forward_validate(
        frame,
        min_train_weeks=args.min_train_weeks,
        alpha=args.alpha,
        top_k=args.top_k,
        cost_bps=args.cost_bps,
        block_weeks=args.block_weeks,
        bootstrap_samples=args.bootstrap_samples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    observations.to_csv(args.output.with_name(args.output.stem + "_observations.csv"), index=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
