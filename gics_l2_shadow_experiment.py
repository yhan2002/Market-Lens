from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sector_rotation import classify_quadrants
from sector_rotation_forecast import (
    BASE_FEATURE_COLUMNS,
    _block_bootstrap_mean_ci,
    _format_weights,
    _portfolio_turnover,
    _safe_rank_correlation,
    _top_k_tie_weights,
    walk_forward_validate,
)


IT3 = ("4510", "4520", "4530")
DEFAULTS = {
    "full25": {"min_train_weeks": 52, "top_k": 5},
    "core24": {"min_train_weeks": 156, "top_k": 5},
    "it3": {"min_train_weeks": 156, "top_k": 1},
}
REQUIRED_MASTER = {
    "industry_group_code",
    "industry_group_name",
    "parent_sector_code",
    "parent_sector_name",
    "group_security",
    "group_bbgid",
    "provider",
    "universe",
    "currency",
    "return_type",
    "pit_status",
}
REQUIRED_GROUP = {
    "date",
    "industry_group_code",
    "industry_group_name",
    "parent_sector_code",
    "security",
    "bbgid",
    "total_return_level",
    "currency",
    "bloomberg_field",
    "source",
}
REQUIRED_BENCHMARK = {
    "date",
    "benchmark_name",
    "security",
    "bbgid",
    "total_return_level",
    "currency",
    "bloomberg_field",
    "source",
}


def _require_columns(frame: pd.DataFrame, required: set[str], filename: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{filename}: missing columns: {', '.join(missing)}")


def _clean_code(series: pd.Series, filename: str, column: str) -> pd.Series:
    cleaned = series.astype("string").str.strip()
    if cleaned.isna().any() or cleaned.eq("").any():
        raise ValueError(f"{filename}: blank {column}")
    return cleaned


def _clean_date(series: pd.Series, filename: str) -> pd.Series:
    parsed = pd.to_datetime(series, format="%Y-%m-%d", errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"{filename}: invalid date")
    return parsed


def _clean_positive(series: pd.Series, filename: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().any() or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(f"{filename}: total_return_level must be finite and positive")
    return values.astype(float)


def _require_single_value(
    frame: pd.DataFrame, column: str, expected: str, filename: str
) -> None:
    values = set(frame[column].astype("string").str.strip().dropna())
    if values != {expected}:
        raise ValueError(f"{filename}: {column} must be {expected!r}; found {sorted(values)}")


def load_package(package_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = {
        "master": package_dir / "industry_group_master.csv",
        "groups": package_dir / "industry_group_total_return.csv",
        "benchmark": package_dir / "benchmark_total_return.csv",
    }
    missing = [path.name for path in paths.values() if not path.is_file()]
    if missing:
        raise ValueError(f"missing input files: {', '.join(missing)}")

    master = pd.read_csv(paths["master"], dtype="string")
    groups = pd.read_csv(
        paths["groups"],
        dtype={"industry_group_code": "string", "parent_sector_code": "string"},
    )
    benchmark = pd.read_csv(paths["benchmark"])
    _require_columns(master, REQUIRED_MASTER, paths["master"].name)
    _require_columns(groups, REQUIRED_GROUP, paths["groups"].name)
    _require_columns(benchmark, REQUIRED_BENCHMARK, paths["benchmark"].name)

    for column in ("industry_group_code", "parent_sector_code"):
        master[column] = _clean_code(master[column], paths["master"].name, column)
        groups[column] = _clean_code(groups[column], paths["groups"].name, column)
    for column in ("industry_group_name", "parent_sector_name"):
        master[column] = _clean_code(master[column], paths["master"].name, column)
    groups["industry_group_name"] = _clean_code(
        groups["industry_group_name"], paths["groups"].name, "industry_group_name"
    )
    groups["date"] = _clean_date(groups["date"], paths["groups"].name)
    benchmark["date"] = _clean_date(benchmark["date"], paths["benchmark"].name)
    groups["total_return_level"] = _clean_positive(
        groups["total_return_level"], paths["groups"].name
    )
    benchmark["total_return_level"] = _clean_positive(
        benchmark["total_return_level"], paths["benchmark"].name
    )

    if master["industry_group_code"].duplicated().any():
        raise ValueError("industry_group_master.csv: duplicate industry_group_code")
    if master["industry_group_code"].nunique() != 25:
        raise ValueError("industry_group_master.csv: exactly 25 group codes are required")
    if groups.duplicated(["industry_group_code", "date"]).any():
        raise ValueError("industry_group_total_return.csv: duplicate code/date key")
    if benchmark["date"].duplicated().any():
        raise ValueError("benchmark_total_return.csv: duplicate date key")
    master_codes = set(master["industry_group_code"])
    if set(groups["industry_group_code"]) != master_codes:
        raise ValueError("industry_group_total_return.csv: group codes do not match master")
    if "6020" not in master_codes or not set(IT3).issubset(master_codes):
        raise ValueError("master must contain 6020 and IT groups 4510/4520/4530")

    for frame, filename in (
        (groups, paths["groups"].name),
        (benchmark, paths["benchmark"].name),
    ):
        _require_single_value(frame, "currency", "USD", filename)
        _require_single_value(
            frame,
            "bloomberg_field",
            "TOT_RETURN_INDEX_GROSS_DVDS",
            filename,
        )
        _require_single_value(frame, "source", "Bloomberg Desktop API", filename)
    _require_single_value(benchmark, "security", "SPX Index", paths["benchmark"].name)
    _require_single_value(benchmark, "benchmark_name", "S&P 500", paths["benchmark"].name)
    _require_single_value(master, "provider", "Standard & Poor's", paths["master"].name)
    _require_single_value(master, "universe", "S&P 500", paths["master"].name)
    _require_single_value(master, "currency", "USD", paths["master"].name)
    _require_single_value(
        master, "return_type", "GROSS_TOTAL_RETURN", paths["master"].name
    )
    _require_single_value(master, "pit_status", "PIT_NOT_VERIFIED", paths["master"].name)

    mapping = master.set_index("industry_group_code")
    checked = groups.join(
        mapping[
            [
                "industry_group_name",
                "parent_sector_code",
                "group_security",
                "group_bbgid",
            ]
        ],
        on="industry_group_code",
        rsuffix="_master",
    )
    comparisons = {
        "industry_group_name": "industry_group_name_master",
        "parent_sector_code": "parent_sector_code_master",
        "security": "group_security",
        "bbgid": "group_bbgid",
    }
    for actual, expected in comparisons.items():
        if not checked[actual].astype("string").eq(checked[expected].astype("string")).all():
            raise ValueError(f"industry_group_total_return.csv: inconsistent {actual}")
    return (
        master.sort_values("industry_group_code").reset_index(drop=True),
        groups.sort_values(["date", "industry_group_code"]).reset_index(drop=True),
        benchmark.sort_values("date").reset_index(drop=True),
    )


def cohort_codes(master: pd.DataFrame, cohort: str) -> list[str]:
    all_codes = master["industry_group_code"].tolist()
    if cohort == "full25":
        codes = all_codes
    elif cohort == "core24":
        codes = [code for code in all_codes if code != "6020"]
    elif cohort == "it3":
        codes = list(IT3)
    else:
        raise ValueError(f"unknown cohort: {cohort}")
    expected = {"full25": 25, "core24": 24, "it3": 3}[cohort]
    if len(codes) != expected:
        raise ValueError(f"{cohort}: expected {expected} group codes")
    return codes


def build_cohort_prices(
    groups: pd.DataFrame, benchmark: pd.DataFrame, codes: list[str]
) -> pd.DataFrame:
    benchmark_series = benchmark.set_index("date")["total_return_level"].rename("SPX")
    group_matrix = groups.loc[groups["industry_group_code"].isin(codes)].pivot(
        index="date", columns="industry_group_code", values="total_return_level"
    )
    # Benchmark sessions are canonical; missing group observations stay missing.
    return benchmark_series.to_frame().join(group_matrix.reindex(columns=codes), how="left")


def _rolling_zscore(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    mean = frame.rolling(window, min_periods=window).mean()
    std = frame.rolling(window, min_periods=window).std(ddof=0).replace(0, np.nan)
    return (frame - mean) / std


def build_rrg_signals(
    prices: pd.DataFrame,
    codes: list[str],
    *,
    ratio_window: int = 63,
    momentum_lag: int = 10,
) -> pd.DataFrame:
    if ratio_window < 20 or momentum_lag < 1:
        raise ValueError("ratio_window>=20 and momentum_lag>=1 are required")
    relative_log_price = np.log(prices[codes].div(prices["SPX"], axis=0))
    rs_ratio = 100 + _rolling_zscore(relative_log_price, ratio_window)
    rs_momentum = 100 + _rolling_zscore(rs_ratio.diff(momentum_lag), ratio_window)
    score = (rs_ratio.rank(axis=1, pct=True) + rs_momentum.rank(axis=1, pct=True)) / 2
    score = score.where(rs_ratio.notna().all(axis=1) & rs_momentum.notna().all(axis=1))
    rows = []
    for code in codes:
        part = pd.DataFrame(
            {
                "date": prices.index,
                "ticker": code,
                "rs_ratio": rs_ratio[code],
                "rs_momentum": rs_momentum[code],
                "shadow_score": score[code],
            }
        )
        part["quadrant"] = classify_quadrants(part["rs_ratio"], part["rs_momentum"])
        rows.append(part)
    return pd.concat(rows, ignore_index=True).dropna(subset=["shadow_score"])


def build_feature_frame(
    prices: pd.DataFrame, codes: list[str], *, horizon: int = 20
) -> pd.DataFrame:
    signals = build_rrg_signals(prices, codes)
    signal_features = signals.sort_values(["ticker", "date"]).copy()
    grouped = signal_features.groupby("ticker")
    signal_features["rrg_ratio"] = signal_features["rs_ratio"] - 100
    signal_features["rrg_momentum"] = signal_features["rs_momentum"] - 100
    for lag in (5, 20):
        signal_features[f"rrg_ratio_delta_{lag}"] = grouped["rs_ratio"].diff(lag)
        signal_features[f"rrg_momentum_delta_{lag}"] = grouped["rs_momentum"].diff(lag)

    forward = prices.shift(-horizon).div(prices).sub(1)
    equal_forward = forward[codes].mean(axis=1, skipna=False)
    label_end = pd.Series(prices.index, index=prices.index).shift(-horizon)
    benchmark_return = prices["SPX"].pct_change(fill_method=None)
    benchmark_volatility = benchmark_return.rolling(20).std()
    benchmark_drawdown = prices["SPX"].div(prices["SPX"].rolling(60).max()).sub(1)
    rows = []
    for code in codes:
        group_return = prices[code].pct_change(fill_method=None)
        rows.append(
            pd.DataFrame(
                {
                    "date": prices.index,
                    "ticker": code,
                    "relative_return_20": prices[code].pct_change(20, fill_method=None)
                    - prices["SPX"].pct_change(20, fill_method=None),
                    "relative_return_60": prices[code].pct_change(60, fill_method=None)
                    - prices["SPX"].pct_change(60, fill_method=None),
                    "relative_return_120": prices[code].pct_change(120, fill_method=None)
                    - prices["SPX"].pct_change(120, fill_method=None),
                    "relative_volatility_20": group_return.rolling(20).std()
                    - benchmark_volatility,
                    "relative_drawdown_60": prices[code]
                    .div(prices[code].rolling(60).max())
                    .sub(1)
                    - benchmark_drawdown,
                    "target_excess_20d": forward[code] - equal_forward,
                    "label_end": label_end,
                }
            )
        )
    return pd.concat(rows, ignore_index=True).merge(
        signal_features[
            ["date", "ticker", "shadow_score", *BASE_FEATURE_COLUMNS[:6]]
        ],
        on=["date", "ticker"],
        how="inner",
    )


def weekly_complete_rows(frame: pd.DataFrame, codes: list[str]) -> pd.DataFrame:
    feature_columns = ["shadow_score", *BASE_FEATURE_COLUMNS]
    valid = frame.loc[frame["ticker"].isin(codes)].copy()
    feature_complete = valid.groupby("date").filter(
        lambda group: len(group) == len(codes)
        and group["ticker"].nunique() == len(codes)
        and group[feature_columns].notna().all().all()
    )
    if feature_complete.empty:
        return feature_complete
    week_ends = feature_complete.groupby(
        feature_complete["date"].dt.to_period("W-FRI")
    )["date"].max()
    weekly = feature_complete.loc[feature_complete["date"].isin(week_ends)]
    # Fix the signal date before inspecting future labels; otherwise a missing
    # Friday label could move the historical decision backward to Thursday.
    labelled = weekly.groupby("date").filter(
        lambda group: group[["target_excess_20d", "label_end"]].notna().all().all()
    )
    return labelled.sort_values(["date", "ticker"])


def evaluate_rrg(
    weekly: pd.DataFrame, *, top_k: int, bootstrap_samples: int = 2000
) -> tuple[dict[str, object], pd.DataFrame]:
    rows: list[dict[str, object]] = []
    previous: pd.Series | None = None
    for date, current in weekly.groupby("date", sort=True):
        weights = _top_k_tie_weights(current, "shadow_score", top_k=top_k)
        selected = set(weights[weights > 0].index)
        targets = current.set_index("ticker")["target_excess_20d"]
        excess = float(targets.reindex(weights.index).mul(weights).sum())
        if abs(excess) < 1e-15:
            excess = 0.0
        active = current["shadow_score"].nunique() > 1
        rows.append(
            {
                "date": date,
                "selected": ",".join(sorted(selected)),
                "selection_weights": _format_weights(weights),
                "boundary_tie": len(selected) > top_k,
                "active": active,
                "top_k_excess": excess,
                "hit": bool(excess > 0) if active else None,
                "rank_ic": _safe_rank_correlation(
                    current["shadow_score"], current["target_excess_20d"]
                ),
                "turnover": _portfolio_turnover(weights, previous),
            }
        )
        previous = weights
    observations = pd.DataFrame(rows)
    if observations.empty:
        raise ValueError("no complete labelled weekly RRG observations")
    ci = _block_bootstrap_mean_ci(
        observations["top_k_excess"], samples=bootstrap_samples
    )
    return {
        "status": "OK",
        "observations": len(observations),
        "mean_top_k_excess": float(observations["top_k_excess"].mean()),
        "top_k_excess_95ci": list(ci),
        "hit_rate": float(observations["hit"].mean()),
        "active_observations": int(observations["active"].sum()),
        "no_edge_observations": int((~observations["active"]).sum()),
        "boundary_tie_observations": int(observations["boundary_tie"].sum()),
        "mean_rank_ic": float(observations["rank_ic"].mean()),
        "rank_ic_observations": int(observations["rank_ic"].notna().sum()),
        "mean_turnover": float(observations["turnover"].mean()),
        "tie_policy": "equal weight across all names tied at the top-k boundary",
    }, observations


def run_cohort(
    master: pd.DataFrame,
    groups: pd.DataFrame,
    benchmark: pd.DataFrame,
    cohort: str,
    *,
    bootstrap_samples: int = 2000,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    codes = cohort_codes(master, cohort)
    prices = build_cohort_prices(groups, benchmark, codes)
    weekly = weekly_complete_rows(build_feature_frame(prices, codes), codes)
    settings = DEFAULTS[cohort]
    rrg_summary, rrg_observations = evaluate_rrg(
        weekly, top_k=settings["top_k"], bootstrap_samples=bootstrap_samples
    )
    try:
        ridge_summary, ridge_observations = walk_forward_validate(
            weekly,
            min_train_weeks=settings["min_train_weeks"],
            top_k=settings["top_k"],
            bootstrap_samples=bootstrap_samples,
            feature_columns=BASE_FEATURE_COLUMNS,
        )
        ridge_summary["target"] = (
            "next-20-trading-day industry-group return minus same-date "
            "equal-weight cohort return"
        )
        ridge_summary["status"] = "OK"
    except ValueError as exc:
        ridge_summary = {
            "status": "INSUFFICIENT",
            "reason": str(exc),
            "minimum_training_weeks": settings["min_train_weeks"],
            "observations": 0,
        }
        ridge_observations = pd.DataFrame()
    summary = {
        "group_count": len(codes),
        "groups": codes,
        "canonical_benchmark_dates": len(prices),
        "exact_level_complete_dates": int(prices.notna().all(axis=1).sum()),
        "labelled_weekly_dates": int(weekly["date"].nunique()),
        "top_k": settings["top_k"],
        "minimum_training_weeks": settings["min_train_weeks"],
        "rrg_rule": rrg_summary,
        "ridge_oos": ridge_summary,
    }
    return summary, rrg_observations, ridge_observations


def main() -> None:
    parser = argparse.ArgumentParser(description="Research-only GICS L2 shadow experiment")
    parser.add_argument("package_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--cohort", choices=[*DEFAULTS, "all"], default="all")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be >=100")
    package_dir = args.package_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir == package_dir or output_dir.is_relative_to(package_dir):
        raise ValueError("output_dir must be outside package_dir")
    master, groups, benchmark = load_package(package_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = list(DEFAULTS) if args.cohort == "all" else [args.cohort]
    summary: dict[str, object] = {
        "research_only": True,
        "pit_status": "PIT_PARTIAL",
        "production_promotion": "PROHIBITED",
        "taxonomy_history": "NOT PIT VERIFIED; current hierarchy mapped backward",
        "parent_sector_series": "EXCLUDED; noncanonical parent candidates found in semantic audit",
        "benchmark": "SPX gross total return",
        "rrg_parameters": {"ratio_window": 63, "momentum_lag": 10},
        "target": "next-20-trading-day group return minus same-date equal-weight cohort return",
        "signal_frequency": "weekly; last complete benchmark session of W-FRI week",
        "missing_data_policy": "no interpolation or forward fill; exact-date complete cohort only",
        "cohorts": {},
    }
    for cohort in selected:
        try:
            result, rrg_rows, ridge_rows = run_cohort(
                master,
                groups,
                benchmark,
                cohort,
                bootstrap_samples=args.bootstrap_samples,
            )
        except ValueError as exc:
            result = {"status": "INSUFFICIENT", "reason": str(exc)}
            rrg_rows, ridge_rows = pd.DataFrame(), pd.DataFrame()
        summary["cohorts"][cohort] = result
        rrg_rows.to_csv(output_dir / f"{cohort}_rrg_observations.csv", index=False)
        ridge_rows.to_csv(output_dir / f"{cohort}_ridge_observations.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
