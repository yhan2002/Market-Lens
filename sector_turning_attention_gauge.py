"""Research-only IN/OUT gauges with causal attention-adjusted alert thresholds."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research_drawdown_onset import causal_trailing_percentile
from roro_shadow import _canonical_json, _canonical_provenance_encoding
from sector_turning_pooled_shadow import (
    DIRECTIONS,
    _early_event_metrics,
    _json_ready,
    _wilson_interval,
    paired_date_block_pre_pivot_recall_delta_ci,
)


SCHEMA_VERSION = 1
SERIES_VERSION = "sector-turning-attention-gauge-v0-pit-partial"
SOURCE_MODEL = "BETA_BREADTH_SHALLOW_BOOST"
THRESHOLD_MODEL = "attention-proxy-v0-causal252-m60-b90-a6-c4"
PIT_STATUS = "PIT_PARTIAL/CURRENT_REVISED_HISTORY_ONLY"
FEATURE_TIMING = "EOD_T_INPUT; ACTION_NO_EARLIER_THAN_T_PLUS_1"
BASE_THRESHOLD = 0.90
PRIORITY_THRESHOLD = 0.95
THRESHOLD_FLOOR = 0.85
THRESHOLD_CEILING = 0.95
PERCENTILE_WINDOW = 252
PERCENTILE_MIN_HISTORY = 60
HORIZON = 10
ATTENTION_COLUMNS = (
    "attention_move",
    "attention_volatility",
    "attention_acceleration",
)
CONFIRMATION_INPUTS = (
    "beta_relative_acceleration_5",
    "breadth_net_rel_spx_accel_5d",
    "breadth_pct200_rel_spx_change_5d",
)


def _required(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {', '.join(sorted(missing))}")


def _prepare_key(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise").dt.normalize()
    result["sector"] = result["sector"].astype("string").str.strip()
    if result[["date", "sector"]].isna().any().any():
        raise ValueError(f"{name} has missing date/sector keys")
    if result.duplicated(["date", "sector"]).any():
        raise ValueError(f"{name} has duplicate date/sector rows")
    return result.sort_values(["sector", "date"]).reset_index(drop=True)


def _sector_percentile(frame: pd.DataFrame, values: pd.Series) -> pd.Series:
    material = pd.DataFrame(
        {"sector": frame["sector"].to_numpy(), "value": values.to_numpy()},
        index=frame.index,
    )
    return material.groupby("sector", sort=False, group_keys=False)["value"].transform(
        lambda series: causal_trailing_percentile(
            series, window=PERCENTILE_WINDOW, min_history=PERCENTILE_MIN_HISTORY
        )
    )


def build_attention_features(
    beta_history: pd.DataFrame, breadth_features: pd.DataFrame
) -> pd.DataFrame:
    """Build market-observable proxies; every percentile excludes the current row."""
    beta_columns = {
        "date",
        "sector",
        "beta_adjusted_return_1d",
        "beta_relative_volatility_20",
        "beta_relative_acceleration_5",
    }
    breadth_columns = {
        "date",
        "sector",
        "breadth_net_rel_spx_accel_5d",
        "breadth_pct200_rel_spx_change_5d",
    }
    _required(beta_history, beta_columns, "beta history")
    _required(breadth_features, breadth_columns, "breadth features")
    beta = _prepare_key(beta_history.loc[:, sorted(beta_columns)], "beta history")
    breadth = _prepare_key(
        breadth_features.loc[:, sorted(breadth_columns)], "breadth features"
    )
    frame = beta.merge(breadth, on=["date", "sector"], how="inner", validate="one_to_one")
    if frame.empty:
        raise ValueError("no paired beta/breadth attention rows")
    frame = frame.sort_values(["sector", "date"]).reset_index(drop=True)
    frame["attention_move"] = _sector_percentile(
        frame, frame["beta_adjusted_return_1d"].abs()
    )
    frame["attention_volatility"] = _sector_percentile(
        frame, frame["beta_relative_volatility_20"]
    )
    frame["attention_acceleration"] = _sector_percentile(
        frame, frame["beta_relative_acceleration_5"].abs()
    )
    frame["attention_score"] = frame.loc[:, ATTENTION_COLUMNS].mean(axis=1, skipna=False)

    for direction, sign in (("in", 1.0), ("out", -1.0)):
        components = []
        for column in CONFIRMATION_INPUTS:
            output = f"{direction}_{column}_confirmation"
            frame[output] = _sector_percentile(frame, sign * frame[column])
            components.append(output)
        frame[f"{direction}_confirmation_score"] = frame.loc[:, components].mean(
            axis=1, skipna=False
        )
    return frame


def adaptive_threshold(attention: pd.Series, confirmation: pd.Series) -> pd.Series:
    """Raise the line for noisy attention and lower it only with confirming evidence."""
    threshold = BASE_THRESHOLD + 0.06 * (attention - 0.5) - 0.04 * (confirmation - 0.5)
    return threshold.clip(THRESHOLD_FLOOR, THRESHOLD_CEILING)


def alert_level(score: float, threshold: float) -> str:
    if score >= max(PRIORITY_THRESHOLD, threshold):
        return "PRIORITY"
    if score >= threshold:
        return "WARNING"
    if score >= max(0.0, threshold - 0.10):
        return "WATCH"
    return "NORMAL"


def build_gauge_history(
    predictions: pd.DataFrame, attention_features: pd.DataFrame
) -> pd.DataFrame:
    required_predictions = {
        "direction",
        "model",
        "date",
        "sector",
        "sector_name",
        "target",
        "target_event_pivot_position",
        "target_event_confirmation_position",
        "target_event_confirmation_date",
        "score",
    }
    _required(predictions, required_predictions, "predictions")
    rows = predictions.loc[predictions["model"].eq(SOURCE_MODEL)].copy()
    if rows.empty:
        raise ValueError(f"predictions do not contain {SOURCE_MODEL}")
    if set(rows["direction"].unique()) != set(DIRECTIONS):
        raise ValueError("predictions must contain both IN and OUT directions")
    rows["date"] = pd.to_datetime(rows["date"], errors="raise").dt.normalize()
    rows["target_event_confirmation_date"] = pd.to_datetime(
        rows["target_event_confirmation_date"], errors="coerce"
    ).dt.normalize()
    rows["sector"] = rows["sector"].astype("string").str.strip()
    if rows.duplicated(["direction", "date", "sector"]).any():
        raise ValueError("duplicate prediction direction/date/sector rows")
    features = attention_features.copy()
    keep = [
        "date",
        "sector",
        "attention_score",
        "in_confirmation_score",
        "out_confirmation_score",
    ]
    rows = rows.merge(features.loc[:, keep], on=["date", "sector"], how="left", validate="many_to_one")
    rows["confirmation_score"] = np.where(
        rows["direction"].eq("ROTATION_IN"),
        rows["in_confirmation_score"],
        rows["out_confirmation_score"],
    )
    complete = rows[["attention_score", "confirmation_score"]].notna().all(axis=1)
    rows["dynamic_threshold"] = BASE_THRESHOLD
    rows.loc[complete, "dynamic_threshold"] = adaptive_threshold(
        rows.loc[complete, "attention_score"], rows.loc[complete, "confirmation_score"]
    )
    rows["noise_score"] = rows["attention_score"] * (1 - rows["confirmation_score"])
    rows["proxy_status"] = np.where(complete, "DYNAMIC", "BASE_FALLBACK")
    rows["alert_margin"] = rows["score"] - rows["dynamic_threshold"]
    rows["adjusted_score"] = rows["alert_margin"] + BASE_THRESHOLD
    rows["level"] = [
        alert_level(float(score), float(threshold))
        for score, threshold in zip(rows["score"], rows["dynamic_threshold"], strict=True)
    ]
    return rows.sort_values(["date", "sector", "direction"]).reset_index(drop=True)


def _optional_percent(value: Any) -> float | None:
    return None if value is None or not np.isfinite(float(value)) else round(float(value) * 100, 4)


def _reading(row: pd.Series | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "score": round(float(row["score"]) * 100, 4),
        "threshold": round(float(row["dynamic_threshold"]) * 100, 4),
        "alert_margin": round(float(row["alert_margin"]) * 100, 4),
        "level": str(row["level"]),
        "confirmation_score": _optional_percent(row["confirmation_score"]),
        "noise_score": _optional_percent(row["noise_score"]),
        "proxy_status": str(row["proxy_status"]),
    }


def build_snapshot(gauge_history: pd.DataFrame) -> dict[str, Any]:
    if gauge_history.empty:
        raise ValueError("no gauge history")
    as_of = pd.Timestamp(gauge_history["date"].max())
    latest = gauge_history.loc[gauge_history["date"].eq(as_of)]
    rows: list[dict[str, Any]] = []
    for (sector, sector_name), group in latest.groupby(["sector", "sector_name"], sort=True):
        by_direction = {str(row.direction): row for row in group.itertuples(index=False)}
        attention = group["attention_score"].dropna()
        attention_value = float(attention.iloc[0]) if len(attention) else None
        if attention_value is None:
            band = "UNAVAILABLE"
        elif attention_value < 1 / 3:
            band = "LOW"
        elif attention_value >= 2 / 3:
            band = "HIGH"
        else:
            band = "NORMAL"
        rows.append(
            {
                "sector": str(sector),
                "sector_name": str(sector_name),
                "attention_score": _optional_percent(attention_value),
                "attention_band": band,
                "rotation_out": _reading(
                    pd.Series(by_direction["ROTATION_OUT"]._asdict())
                    if "ROTATION_OUT" in by_direction
                    else None
                ),
                "rotation_in": _reading(
                    pd.Series(by_direction["ROTATION_IN"]._asdict())
                    if "ROTATION_IN" in by_direction
                    else None
                ),
            }
        )
    if not rows:
        raise ValueError("latest date has no gauges")
    payload = {
        "status": "RESEARCH_SHADOW_ONLY",
        "pit_status": PIT_STATUS,
        "source_model": SOURCE_MODEL,
        "threshold_model": THRESHOLD_MODEL,
        "base_threshold": BASE_THRESHOLD * 100,
        "priority_threshold": PRIORITY_THRESHOLD * 100,
        "threshold_floor": THRESHOLD_FLOOR * 100,
        "threshold_ceiling": THRESHOLD_CEILING * 100,
        "feature_timing": FEATURE_TIMING,
        "attention_proxy": "beta-residual move/volatility/acceleration; no news or search data",
        "rows": rows,
    }
    model_versions = {
        "turning_model": SOURCE_MODEL,
        "attention_threshold": THRESHOLD_MODEL,
    }
    material = {
        "series_version": SERIES_VERSION,
        "date": str(as_of.date()),
        "input_cutoff": str(as_of.date()),
        "model_versions": model_versions,
        "payload": payload,
    }
    provenance = hashlib.sha256(
        _canonical_provenance_encoding(material).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": SCHEMA_VERSION,
        "series_version": SERIES_VERSION,
        "as_of": str(as_of.date()),
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "input_cutoff": str(as_of.date()),
        "provenance_hash": provenance,
        "model_versions": model_versions,
        "payload": payload,
    }


def evaluate(gauge_history: pd.DataFrame, bootstrap_samples: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for direction in DIRECTIONS:
        rows = gauge_history.loc[
            gauge_history["direction"].eq(direction) & gauge_history["target"].notna()
        ].copy()
        baseline = rows.copy()
        dynamic = rows.copy()
        dynamic["score"] = dynamic["adjusted_score"]
        baseline_metric = _early_event_metrics(
            baseline, pd.DataFrame(), direction, horizon=HORIZON, cutoff=BASE_THRESHOLD
        )
        dynamic_metric = _early_event_metrics(
            dynamic, pd.DataFrame(), direction, horizon=HORIZON, cutoff=BASE_THRESHOLD
        )
        for metric in (baseline_metric, dynamic_metric):
            low, high = _wilson_interval(
                metric["pre_pivot_hits"], metric["eligible_confirmations"]
            )
            metric["pre_pivot_recall_ci_low"] = low
            metric["pre_pivot_recall_ci_high"] = high
        calendar = pd.DatetimeIndex(rows["date"].drop_duplicates().sort_values())
        result[direction] = {
            "fixed_threshold": baseline_metric,
            "attention_adjusted_threshold": dynamic_metric,
            "paired_recall_uplift": paired_date_block_pre_pivot_recall_delta_ci(
                dynamic,
                baseline,
                samples=bootstrap_samples,
                block_dates=40,
                calendar_dates=calendar,
                horizon=HORIZON,
                cutoff=BASE_THRESHOLD,
            ),
        }
    return result


def upload(snapshot: dict[str, Any], config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    headers = {
        "Authorization": f"Bearer {config['upload_token']}",
        "Content-Type": "application/json",
    }
    if config.get("sites_bypass_token"):
        headers["OAI-Sites-Authorization"] = f"Bearer {config['sites_bypass_token']}"
    request = urllib.request.Request(
        str(config["site_url"]).rstrip("/") + "/api/sector-turning",
        data=_canonical_json(snapshot).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Sector Turning upload failed HTTP {error.code}: {detail}") from error


def self_test() -> None:
    dates = pd.bdate_range("2025-01-01", periods=90)
    beta = pd.DataFrame(
        {
            "date": dates,
            "sector": "1010",
            "beta_adjusted_return_1d": np.linspace(-0.01, 0.02, len(dates)),
            "beta_relative_volatility_20": np.linspace(0.01, 0.03, len(dates)),
            "beta_relative_acceleration_5": np.linspace(-0.02, 0.02, len(dates)),
        }
    )
    breadth = pd.DataFrame(
        {
            "date": dates,
            "sector": "1010",
            "breadth_net_rel_spx_accel_5d": np.linspace(-0.5, 0.5, len(dates)),
            "breadth_pct200_rel_spx_change_5d": np.linspace(-0.2, 0.2, len(dates)),
        }
    )
    original = build_attention_features(beta, breadth)
    changed = beta.copy()
    changed.loc[changed.index[-1], "beta_adjusted_return_1d"] = 100
    revised = build_attention_features(changed, breadth)
    assert original.iloc[:-1].equals(revised.iloc[:-1])
    thresholds = adaptive_threshold(
        pd.Series([1.0, 0.0]), pd.Series([0.0, 1.0])
    )
    assert np.allclose(thresholds, [THRESHOLD_CEILING, THRESHOLD_FLOOR])
    assert alert_level(0.96, 0.92) == "PRIORITY"
    assert alert_level(0.91, 0.90) == "WARNING"
    assert alert_level(0.85, 0.90) == "WATCH"
    assert alert_level(0.70, 0.90) == "NORMAL"
    print("self-test: PASS")


def run(args: argparse.Namespace) -> dict[str, Any]:
    predictions = pd.read_csv(args.predictions, low_memory=False)
    beta = pd.read_csv(args.beta_history, dtype={"sector": "string"}, low_memory=False)
    breadth = pd.read_csv(
        args.breadth_features, dtype={"sector": "string"}, low_memory=False
    )
    features = build_attention_features(beta, breadth)
    gauges = build_gauge_history(predictions, features)
    snapshot = build_snapshot(gauges)
    evaluation = evaluate(gauges, args.bootstrap_samples)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    gauges.to_csv(output / "gauge_history.csv", index=False)
    (output / "latest_snapshot.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    summary = {
        "verdict": "RESEARCH_SHADOW_ONLY",
        "pit_status": PIT_STATUS,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "spec": {
            "series_version": SERIES_VERSION,
            "source_model": SOURCE_MODEL,
            "threshold_model": THRESHOLD_MODEL,
            "base_threshold": BASE_THRESHOLD,
            "priority_threshold": PRIORITY_THRESHOLD,
            "threshold_bounds": [THRESHOLD_FLOOR, THRESHOLD_CEILING],
            "percentile_window": PERCENTILE_WINDOW,
            "percentile_min_history": PERCENTILE_MIN_HISTORY,
            "feature_timing": FEATURE_TIMING,
            "news_or_search_inputs": False,
            "coefficients": "predeclared research hypothesis; not selected on labels",
        },
        "coverage": {
            "first_date": str(pd.Timestamp(gauges["date"].min()).date()),
            "last_date": str(pd.Timestamp(gauges["date"].max()).date()),
            "dates": int(gauges["date"].nunique()),
            "sectors": int(gauges["sector"].nunique()),
            "rows": int(len(gauges)),
            "dynamic_rows": int(gauges["proxy_status"].eq("DYNAMIC").sum()),
        },
        "evaluation": evaluation,
    }
    clean_summary = _json_ready(summary)
    (output / "summary.json").write_text(
        json.dumps(clean_summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    uploaded = upload(snapshot, args.config.resolve()) if args.upload else None
    return {"output_dir": str(output), "as_of": snapshot["as_of"], "upload": uploaded}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--predictions", type=Path)
    result.add_argument("--beta-history", type=Path)
    result.add_argument("--breadth-features", type=Path)
    result.add_argument("--output-dir", type=Path)
    result.add_argument("--bootstrap-samples", type=int, default=500)
    result.add_argument("--upload", action="store_true")
    result.add_argument("--config", type=Path, default=Path("site_upload_config.json"))
    result.add_argument("--self-test", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.self_test:
        self_test()
        return
    if not all((args.predictions, args.beta_history, args.breadth_features, args.output_dir)):
        raise ValueError(
            "--predictions, --beta-history, --breadth-features, and --output-dir are required"
        )
    if args.bootstrap_samples < 0:
        raise ValueError("--bootstrap-samples must be non-negative")
    if args.upload and not args.config.is_file():
        raise FileNotFoundError(args.config)
    print(json.dumps(run(args), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
