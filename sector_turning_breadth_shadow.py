from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sector_breadth_features import MODEL_FEATURES as BREADTH_FEATURES
from sector_turning_pooled_shadow import (
    PRICE_FEATURES,
    _direction_frame,
    _model_metrics,
    _paired_uplifts,
    build_pooled_daily_frame,
    causal_trailing_percentile,
    effect_code,
    purged_training_mask,
)


GROUPS = ("2010", "4010", "4510")
MODELS = ("POOLED_PRICE", "POOLED_BREADTH")
ALL_FEATURES = (*PRICE_FEATURES, *BREADTH_FEATURES)
CUTOFF = 0.90


def merge_inputs(package_dir: Path, feature_csv: Path, horizon: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    pooled, price_audit, _ = build_pooled_daily_frame(package_dir, None, horizon=horizon)
    pooled = pooled.loc[pooled["sector"].astype(str).isin(GROUPS)].copy()
    breadth = pd.read_csv(feature_csv, parse_dates=["date", "usable_from_session"])
    if breadth.duplicated(["date", "sector"]).any():
        raise ValueError("duplicate breadth date/sector rows")
    if set(breadth["sector"].astype(str).unique()) != set(GROUPS):
        raise ValueError("breadth groups do not match the pre-registered cohort")
    breadth["sector"] = breadth["sector"].astype(str)
    pooled["sector"] = pooled["sector"].astype(str)
    keep = ["date", "sector", "usable_from_session", *BREADTH_FEATURES]
    pooled = pooled.merge(breadth[keep], on=["date", "sector"], how="inner", validate="one_to_one")
    if pooled.loc[:, ALL_FEATURES].replace([np.inf, -np.inf], np.nan).isna().any().any():
        pooled = pooled.dropna(subset=ALL_FEATURES).copy()
    pooled = pooled.sort_values(["date", "sector"]).reset_index(drop=True)
    if pooled.empty:
        raise ValueError("no complete paired price/breadth rows")
    return pooled, {
        "price_input": price_audit,
        "breadth_first_date": str(pooled["date"].min().date()),
        "breadth_last_date": str(pooled["date"].max().date()),
        "paired_rows": int(len(pooled)),
        "paired_dates": int(pooled["date"].nunique()),
        "groups": list(GROUPS),
        "pit_status": "PIT_PARTIAL",
        "feature_timing": "EOD_T_OBSERVED_ACTION_NO_EARLIER_THAN_T_PLUS_1",
    }


def _design(frame: pd.DataFrame, scaled: np.ndarray, model: str, levels: tuple[str, ...]) -> pd.DataFrame:
    continuous = pd.DataFrame(scaled, index=frame.index, columns=ALL_FEATURES)
    effects, _ = effect_code(frame["sector"], levels)
    features = PRICE_FEATURES if model == "POOLED_PRICE" else ALL_FEATURES
    return continuous.loc[:, features].join(effects)


def walk_forward(
    pooled: pd.DataFrame,
    direction: str,
    *,
    min_train_dates: int,
    refit_every: int,
    c_value: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = _direction_frame(pooled, direction, target_mode="PRE_PIVOT")
    complete = frame.loc[:, ALL_FEATURES].notna().all(axis=1)
    scorable = frame["at_risk"].astype(bool) & complete
    levels = tuple(sorted(frame["sector"].unique()))
    dates = pd.DatetimeIndex(frame.loc[scorable, "date"].drop_duplicates().sort_values())
    predictions: list[pd.DataFrame] = []
    fitted_blocks = skipped_history = skipped_class = 0
    output_columns = [
        "date", "_position", "sector", "sector_name", "state", "pivot_position",
        "pivot_date", "confirmation_position", "confirmation_date", "barrier",
        "cooldown_remaining", "event", "at_risk", "target", "label_end_position",
        "label_end_date", "target_event_pivot_position", "target_event_pivot_date",
        "target_event_confirmation_position", "target_event_confirmation_date",
        "macro_asof_date", "relative_price",
    ]
    for start in range(0, len(dates), refit_every):
        block_dates = dates[start : start + refit_every]
        prediction_date = pd.Timestamp(block_dates[0])
        train = frame.loc[purged_training_mask(frame, prediction_date) & complete]
        if train["date"].nunique() < min_train_dates:
            skipped_history += 1
            continue
        if train["target"].nunique() < 2:
            skipped_class += 1
            continue
        test = frame.loc[scorable & frame["date"].isin(block_dates)]
        if test.empty:
            continue
        scaler = StandardScaler().fit(train.loc[:, ALL_FEATURES])
        train_scaled = scaler.transform(train.loc[:, ALL_FEATURES])
        test_scaled = scaler.transform(test.loc[:, ALL_FEATURES])
        target = train["target"].astype(int)
        for model_name in MODELS:
            model = LogisticRegression(
                C=c_value, class_weight="balanced", max_iter=2_000, random_state=42
            ).fit(_design(train, train_scaled, model_name, levels), target)
            rows = test[output_columns].copy()
            rows.insert(0, "direction", direction)
            rows.insert(1, "model", model_name)
            rows.insert(2, "target_mode", "PRE_PIVOT")
            rows["refit_date"] = prediction_date
            rows["training_start_date"] = train["date"].min()
            rows["training_end_date"] = train["date"].max()
            rows["training_date_count"] = train["date"].nunique()
            rows["score"] = model.predict_proba(
                _design(test, test_scaled, model_name, levels)
            )[:, 1]
            predictions.append(rows)
        fitted_blocks += 1
    output = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    if not output.empty:
        output = output.sort_values(["model", "sector", "date"]).reset_index(drop=True)
        output["raw_score"] = output["score"]
        output["score"] = output.groupby(["model", "sector"], sort=False)["raw_score"].transform(
            lambda values: causal_trailing_percentile(values, window=252, min_history=60)
        )
        output = output.dropna(subset=["score"]).reset_index(drop=True)
    return output, {
        "status": "OK" if not output.empty else "INSUFFICIENT_HISTORY",
        "fitted_blocks": fitted_blocks,
        "skipped_history_blocks": skipped_history,
        "skipped_one_class_blocks": skipped_class,
        "scorable_dates": int(len(dates)),
    }


def economic_rows(
    predictions: pd.DataFrame, pooled: pd.DataFrame, horizon: int, *, cutoff: float = CUTOFF
) -> pd.DataFrame:
    prices = {
        sector: group.set_index("_position")["relative_price"].sort_index()
        for sector, group in pooled.groupby("sector")
    }
    records: list[dict[str, Any]] = []
    alerts = predictions.loc[predictions["score"] >= cutoff].sort_values(
        ["direction", "model", "sector", "_position"]
    )
    for (direction, model, sector), group in alerts.groupby(["direction", "model", "sector"]):
        last_position = -10**9
        price = prices[str(sector)]
        for row in group.to_dict("records"):
            position = int(row["_position"])
            same_cluster = position - last_position <= horizon
            last_position = position
            if same_cluster:
                continue
            entry_position, exit_position = position + 1, position + 1 + 20
            if entry_position not in price.index or exit_position not in price.index:
                continue
            entry = float(price.loc[entry_position])
            exit_value = float(price.loc[exit_position])
            window = price.reindex(range(entry_position, exit_position + 1)).dropna()
            if len(window) != 21 or not np.isfinite([entry, exit_value]).all():
                continue
            raw_return = exit_value / entry - 1
            signed = -raw_return if direction == "ROTATION_OUT" else raw_return
            favorable = 1 - float(window.min()) / entry if direction == "ROTATION_OUT" else float(window.max()) / entry - 1
            records.append(
                {
                    "direction": direction,
                    "model": model,
                    "sector": sector,
                    "signal_date": row["date"],
                    "entry_position": entry_position,
                    "signed_relative_return_20d": signed,
                    "favorable_excursion_20d": favorable,
                    "favorable_3pct": int(favorable >= 0.03),
                }
            )
    return pd.DataFrame.from_records(records)


def block_mean_ci(
    rows: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    column: str,
    *,
    samples: int,
    block: int = 40,
) -> tuple[float, float]:
    if rows.empty or len(calendar) < 2 * block:
        return np.nan, np.nan
    grouped = {date: group[column] for date, group in rows.groupby("signal_date")}
    rng = np.random.default_rng(42)
    estimates = []
    for _ in range(samples):
        sampled: list[pd.Timestamp] = []
        while len(sampled) < len(calendar):
            start = int(rng.integers(0, len(calendar) - block + 1))
            sampled.extend(calendar[start : start + block].tolist())
        values = [grouped[date] for date in sampled[: len(calendar)] if date in grouped]
        if values:
            estimates.append(float(pd.concat(values).mean()))
    if len(estimates) < int(samples * 0.8):
        return np.nan, np.nan
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def summarize_economics(rows: pd.DataFrame, calendar: pd.DatetimeIndex, samples: int) -> list[dict[str, Any]]:
    output = []
    for (direction, model), group in rows.groupby(["direction", "model"]):
        for period, mask in {
            "ALL_OOS": pd.Series(True, index=group.index),
            "2025": pd.to_datetime(group["signal_date"]).dt.year.eq(2025),
            "2026": pd.to_datetime(group["signal_date"]).dt.year.eq(2026),
        }.items():
            sample = group.loc[mask]
            period_calendar = calendar if period == "ALL_OOS" else calendar[calendar.year == int(period)]
            low, high = block_mean_ci(
                sample, period_calendar, "signed_relative_return_20d", samples=samples
            )
            output.append(
                {
                    "direction": direction,
                    "model": model,
                    "period": period,
                    "clusters": int(len(sample)),
                    "mean_signed_relative_return_20d": float(sample["signed_relative_return_20d"].mean()) if len(sample) else None,
                    "ci_low": low if np.isfinite(low) else None,
                    "ci_high": high if np.isfinite(high) else None,
                    "favorable_3pct_rate": float(sample["favorable_3pct"].mean()) if len(sample) else None,
                }
            )
    return output


def self_test() -> None:
    dates = pd.bdate_range("2026-01-01", periods=50)
    pooled = pd.DataFrame(
        {
            "date": dates,
            "_position": range(50),
            "sector": "2010",
            "relative_price": np.linspace(1.0, 0.90, 50),
        }
    )
    predictions = pd.DataFrame(
        {
            "direction": ["ROTATION_OUT"] * 3,
            "model": ["POOLED_PRICE"] * 3,
            "sector": ["2010"] * 3,
            "date": [dates[2], dates[4], dates[20]],
            "_position": [2, 4, 20],
            "score": [0.95, 0.96, 0.97],
        }
    )
    rows = economic_rows(predictions, pooled, horizon=10)
    assert len(rows) == 2
    assert (rows["signed_relative_return_20d"] > 0).all()
    chained = predictions.iloc[[0]].copy()
    chained = pd.concat(
        [chained.assign(date=dates[position], _position=position) for position in range(2, 21, 2)],
        ignore_index=True,
    )
    assert len(economic_rows(chained, pooled, horizon=10)) == 1
    print("self-test: PASS")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Research-only GICS breadth PRE_PIVOT comparison")
    result.add_argument("--gics-package-dir", type=Path)
    result.add_argument("--breadth-features", type=Path)
    result.add_argument("--output-dir", type=Path)
    result.add_argument("--horizon", type=int, default=10)
    result.add_argument("--min-train-dates", type=int, default=252)
    result.add_argument("--refit-every", type=int, default=21)
    result.add_argument("--c-value", type=float, default=0.1)
    result.add_argument("--bootstrap-samples", type=int, default=1_000)
    result.add_argument("--self-test", action="store_true")
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    pooled, audit = merge_inputs(args.gics_package_dir.resolve(), args.breadth_features.resolve(), args.horizon)
    prediction_parts = []
    directions: dict[str, Any] = {}
    prediction_calendars: list[pd.DatetimeIndex] = []
    for direction in ("ROTATION_OUT", "ROTATION_IN"):
        predictions, walk = walk_forward(
            pooled,
            direction,
            min_train_dates=args.min_train_dates,
            refit_every=args.refit_every,
            c_value=args.c_value,
        )
        prediction_parts.append(predictions)
        evaluation_calendar = pd.DatetimeIndex(
            predictions["date"].drop_duplicates().sort_values()
        )
        prediction_calendars.append(evaluation_calendar)
        evaluated = predictions.loc[predictions["target"].notna()].copy()
        model_metrics = {
            model: _model_metrics(
                evaluated.loc[evaluated["model"].eq(model)], pooled, direction,
                horizon=args.horizon, cutoff=CUTOFF,
            )
            for model in MODELS
        }
        uplifts = _paired_uplifts(
            evaluated,
            bootstrap_samples=args.bootstrap_samples,
            calendar_dates=evaluation_calendar,
            horizon=args.horizon,
            cutoff=CUTOFF,
            models=MODELS,
        )
        directions[direction] = {"walk_forward": walk, "models": model_metrics, "uplifts": uplifts}
    predictions = pd.concat(prediction_parts, ignore_index=True)
    economics = economic_rows(predictions, pooled, args.horizon)
    prediction_calendar = pd.DatetimeIndex(
        sorted(set().union(*(calendar.tolist() for calendar in prediction_calendars)))
    )
    economic_summary = summarize_economics(
        economics, prediction_calendar, args.bootstrap_samples
    )
    summary = {
        "verdict": "RESEARCH_SHADOW_ONLY",
        "pit_status": "PIT_PARTIAL",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "spec": {
            "groups": list(GROUPS),
            "models": list(MODELS),
            "price_features": list(PRICE_FEATURES),
            "breadth_features": list(BREADTH_FEATURES),
            "horizon": args.horizon,
            "min_train_dates": args.min_train_dates,
            "refit_every": args.refit_every,
            "c_value": args.c_value,
            "cutoff": CUTOFF,
            "execution": "signal after EOD t; economic entry at next-session close",
        },
        "audit": audit,
        "directions": directions,
        "economics": economic_summary,
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    economics.to_csv(output_dir / "economic_clusters.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return summary


def main() -> None:
    args = parser().parse_args()
    if args.self_test:
        self_test()
        return
    if not all((args.gics_package_dir, args.breadth_features, args.output_dir)):
        raise ValueError("--gics-package-dir, --breadth-features, and --output-dir are required")
    if args.bootstrap_samples < 100:
        raise ValueError("--bootstrap-samples must be at least 100")
    summary = run(args)
    print(json.dumps({"verdict": summary["verdict"], "pit_status": summary["pit_status"]}))


if __name__ == "__main__":
    main()
