"""Validate and export three research-only sector decision scores.

The underlying transition model is the already-tested price multiscale model.
This module only adapts it to the 11-sector ETF universe and derives two
dashboard-friendly views from the same signal so the screen cannot contradict
itself.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

from sklearn.metrics import average_precision_score, roc_auc_score

from sector_rotation import SECTOR_ETFS, load_price_matrix
from sector_turning_pooled_shadow import (
    DIRECTIONS,
    _early_event_metrics,
    _labels_with_calendar_gaps,
    _price_features,
    _wilson_interval,
)
from sector_turning_price_multiscale_experiment import (
    BASELINE,
    add_price_path_features,
    walk_forward,
)
from sector_turning_tradable_label_shadow import qualify_labels


ROOT = Path(__file__).resolve().parent
MODEL = "PRICE_MULTISCALE"
SCORE_SCHEMA_VERSION = 1


def build_sector_frame(database: Path, *, horizon: int = 10) -> tuple[pd.DataFrame, dict[str, Any]]:
    prices = load_price_matrix(database)
    calendar = prices["SPY"].dropna().index
    parts: list[pd.DataFrame] = []
    for ticker, name in SECTOR_ETFS.items():
        relative = prices[ticker].div(prices["SPY"]).reindex(calendar)
        labels = _labels_with_calendar_gaps(relative, horizon=horizon)
        part = _price_features(relative).join(labels)
        part.insert(0, "date", part.index)
        part.insert(1, "sector", ticker)
        part.insert(2, "sector_name", name)
        part.insert(3, "relative_price", relative)
        part.insert(4, "_position", np.arange(len(part)))
        part["macro_asof_date"] = pd.NaT
        parts.append(part.reset_index(drop=True))
    pooled = add_price_path_features(pd.concat(parts, ignore_index=True))
    pooled["forward_relative_return_20"] = pooled.groupby("sector", sort=False)[
        "relative_price"
    ].transform(lambda values: values.shift(-20).div(values).sub(1))
    pooled["stretch_score"] = pooled.groupby("date", sort=False)["relative_return_20"].rank(
        pct=True, method="average"
    ).add(
        pooled.groupby("date", sort=False)["relative_return_60"].rank(
            pct=True, method="average"
        )
    ).div(2)
    audit = {
        "database": str(database.resolve()),
        "first_date": str(pooled["date"].min().date()),
        "last_date": str(pooled["date"].max().date()),
        "sessions": int(pooled["date"].nunique()),
        "sectors": int(pooled["sector"].nunique()),
        "benchmark": "SPY",
        "raw_fill_or_interpolation": False,
    }
    return pooled.sort_values(["date", "sector"]).reset_index(drop=True), audit


def attach_scores(predictions: pd.DataFrame, pooled: pd.DataFrame) -> pd.DataFrame:
    context = pooled[
        [
            "date", "sector", "_position", "state", "relative_return_20",
            "relative_return_60", "forward_relative_return_20", "stretch_score",
        ]
    ]
    rows = predictions.loc[predictions["model"].eq(MODEL)].merge(
        context, on=["date", "sector", "_position"], how="left", validate="one_to_one"
    )
    rows["transition_score"] = rows["score"].mul(100)
    rows["trend_persistence_score"] = 100 - rows["transition_score"]
    rows["chase_risk_score"] = np.where(
        rows["state"].eq("UP_ACTIVE"),
        np.sqrt(rows["score"].clip(0, 1) * rows["stretch_score"].clip(0, 1)) * 100,
        np.nan,
    )
    rows["signed_forward_return_20"] = rows["forward_relative_return_20"] * np.where(
        rows["state"].eq("UP_ACTIVE"), 1, -1
    )
    rows["chase_reversal_target"] = np.where(
        rows["state"].eq("UP_ACTIVE") & rows["forward_relative_return_20"].notna(),
        rows["forward_relative_return_20"].lt(0).astype(float),
        np.nan,
    )
    return rows.sort_values(["date", "sector"]).reset_index(drop=True)


def transition_metrics(rows: pd.DataFrame, direction: str) -> dict[str, Any]:
    sample = rows.loc[rows["direction"].eq(direction) & rows["target"].notna()].copy()
    if sample.empty or sample["target"].nunique() < 2:
        raise ValueError(
            f"{direction} evaluation needs both target classes; widen --start and "
            "--evaluation-start (the score itself needs 60 prior predictions)"
        )
    target = sample["target"].astype(int)
    event = _early_event_metrics(sample, pd.DataFrame(), "", horizon=10, cutoff=0.70)
    low, high = _wilson_interval(event["pre_pivot_hits"], event["eligible_confirmations"])
    return {
        "rows": int(len(sample)),
        "positives": int(target.sum()),
        "roc_auc": float(roc_auc_score(target, sample["score"])),
        "pr_auc": float(average_precision_score(target, sample["score"])),
        "alert_budget": 0.30,
        "pre_pivot_recall": event["pre_pivot_recall"],
        "pre_pivot_recall_ci_low": low,
        "pre_pivot_recall_ci_high": high,
        "median_lead_to_pivot_sessions": event["median_lead_to_pivot_sessions"],
        "false_alert_clusters_per_sector_year": event["false_alert_clusters_per_sector_year"],
    }


def score_bucket_metrics(rows: pd.DataFrame) -> dict[str, Any]:
    mature = rows.loc[rows["forward_relative_return_20"].notna()].copy()
    persistent = mature["trend_persistence_score"].ge(80)
    chase = mature["chase_risk_score"].ge(80)
    chase_rows = mature.loc[mature["chase_risk_score"].notna()]
    return {
        "persistence_80_plus": {
            "rows": int(persistent.sum()),
            "mean_signed_forward_relative_return_20d": float(
                mature.loc[persistent, "signed_forward_return_20"].mean()
            ),
            "other_mean_signed_forward_relative_return_20d": float(
                mature.loc[~persistent, "signed_forward_return_20"].mean()
            ),
        },
        "chase_risk_80_plus": {
            "rows": int(chase.sum()),
            "mean_forward_relative_return_20d": float(
                mature.loc[chase, "forward_relative_return_20"].mean()
            ),
            "reversal_rate": float(mature.loc[chase, "chase_reversal_target"].mean()),
            "other_reversal_rate": float(
                chase_rows.loc[chase_rows["chase_risk_score"].lt(80), "chase_reversal_target"].mean()
            ),
        },
    }


def latest_snapshot(rows: pd.DataFrame, pooled: pd.DataFrame) -> dict[str, Any]:
    latest_date = pooled["date"].max()
    context = pooled.loc[
        pooled["date"].eq(latest_date),
        ["sector", "sector_name", "state", "relative_return_20", "relative_return_60"],
    ]
    latest = context.merge(
        rows.loc[
            rows["date"].eq(latest_date),
            [
                "sector", "direction", "transition_score", "trend_persistence_score",
                "chase_risk_score",
            ],
        ],
        on="sector",
        how="left",
        validate="one_to_one",
    )
    records = []
    for row in latest.sort_values("transition_score", ascending=False, na_position="last").itertuples():
        records.append(
            {
                "ticker": row.sector,
                "sector": SECTOR_ETFS[row.sector],
                "state": row.state,
                "transition_direction": row.direction if pd.notna(row.direction) else None,
                "transition_score": (
                    round(float(row.transition_score), 1)
                    if np.isfinite(row.transition_score) else None
                ),
                "trend_persistence_score": (
                    round(float(row.trend_persistence_score), 1)
                    if np.isfinite(row.trend_persistence_score) else None
                ),
                "chase_risk_score": (
                    round(float(row.chase_risk_score), 1)
                    if np.isfinite(row.chase_risk_score) else None
                ),
                "relative_return_20": round(float(row.relative_return_20), 6),
                "relative_return_60": round(float(row.relative_return_60), 6),
                "status": "SCORED" if pd.notna(row.direction) else "N/A_COOLDOWN_OR_NOT_SCORABLE",
            }
        )
    return {
        "schema_version": SCORE_SCHEMA_VERSION,
        "as_of": str(latest_date.date()),
        "score_semantics": "causal trailing percentile; not calibrated probability",
        "thresholds": {"watch": 60, "high": 80},
        "sectors": records,
    }


def report(summary: dict[str, Any]) -> str:
    lines = [
        "# Sector decision scores — long-history validation", "",
        "세 지표는 하나의 가격 멀티스케일 전환 모델을 공유합니다. 점수는 확률이 아니라 해당 섹터의 과거 대비 백분위입니다.", "",
        "| 전환 방향 | AUC | 30% 경보예산 사전 포착률 | 95% 구간 | 중앙 선행일 |",
        "|---|---:|---:|---:|---:|",
    ]
    for direction, item in summary["transition_validation"].items():
        lines.append(
            f"| {direction} | {item['roc_auc']:.3f} | {item['pre_pivot_recall']:.1%} | "
            f"[{item['pre_pivot_recall_ci_low']:.1%}, {item['pre_pivot_recall_ci_high']:.1%}] | "
            f"{item['median_lead_to_pivot_sessions'] or 0:.1f}일 |"
        )
    bucket = summary["score_bucket_validation"]
    persist = bucket["persistence_80_plus"]
    chase = bucket["chase_risk_80_plus"]
    lines.extend(
        [
            "", "## 신규 항목 정의", "",
            "- 전환 가능성: 현재 상승·하락 추세가 향후 10거래일 안에 끝날 위험의 과거 대비 점수.",
            "- 추세 지속: 전환 가능성의 보완값. 두 값을 별도 모델로 만들어 충돌시키지 않음.",
            "- 추격 위험: 상승 추세 섹터에서 전환 가능성과 20·60일 상대강도 과열도를 함께 반영. 하락 추세에는 N/A.",
            "", "## 경제성 점검", "",
            f"- 추세 지속 80점 이상: 향후 20일 추세 방향 상대수익률 {persist['mean_signed_forward_relative_return_20d']:+.2%} "
            f"(그 외 {persist['other_mean_signed_forward_relative_return_20d']:+.2%}).",
            f"- 추격 위험 80점 이상: 향후 20일 상대수익률 {chase['mean_forward_relative_return_20d']:+.2%}, "
            f"역전 비율 {chase['reversal_rate']:.1%} (그 외 {chase['other_reversal_rate']:.1%}).",
            "", "운영 DB와 웹에는 반영하지 않은 연구 결과입니다.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    pooled, audit = build_sector_frame(args.database.resolve(), horizon=10)
    predictions = []
    label_audit = {}
    for direction in DIRECTIONS:
        qualified, direction_audit = qualify_labels(pooled, direction)
        scored, _ = walk_forward(
            qualified,
            direction,
            start=pd.Timestamp(args.start),
            end=pd.Timestamp(args.end),
            min_train_dates=args.min_train_dates,
            refit_every=args.refit_every,
        )
        predictions.append(scored)
        label_audit[direction] = direction_audit
    scored = attach_scores(pd.concat(predictions, ignore_index=True), pooled)
    evaluation = scored.loc[scored["date"].ge(pd.Timestamp(args.evaluation_start))].copy()
    summary = {
        "verdict": "RESEARCH_ONLY",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "score_schema_version": SCORE_SCHEMA_VERSION,
        "input_audit": audit,
        "label_audit": label_audit,
        "evaluation_start": args.evaluation_start,
        "transition_validation": {
            direction: transition_metrics(evaluation, direction) for direction in DIRECTIONS
        },
        "score_bucket_validation": score_bucket_metrics(evaluation),
        "operational_changes": False,
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    scored.to_csv(output / "predictions.csv", index=False)
    (output / "latest_snapshot.json").write_text(
        json.dumps(latest_snapshot(scored, pooled), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    (output / "REPORT.md").write_text(report(summary), encoding="utf-8")
    return summary


def self_test() -> None:
    rows = pd.DataFrame(
        {
            "direction": ["ROTATION_OUT", "ROTATION_IN"],
            "model": [MODEL, MODEL],
            "date": pd.to_datetime(["2026-01-02", "2026-01-02"]),
            "sector": ["XLE", "XLK"],
            "_position": [1, 1],
            "score": [0.81, 0.25],
            "target": [1.0, 0.0],
        }
    )
    pooled = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-02", "2026-01-02"]),
            "sector": ["XLE", "XLK"],
            "_position": [1, 1],
            "state": ["UP_ACTIVE", "DOWN_ACTIVE"],
            "relative_return_20": [0.10, -0.05],
            "relative_return_60": [0.20, -0.10],
            "forward_relative_return_20": [-0.03, 0.02],
            "stretch_score": [1.0, 0.1],
        }
    )
    scored = attach_scores(rows, pooled)
    assert scored.loc[scored["sector"].eq("XLE"), "transition_score"].iat[0] == 81
    assert scored.loc[scored["sector"].eq("XLE"), "trend_persistence_score"].iat[0] == 19
    assert scored.loc[scored["sector"].eq("XLE"), "chase_risk_score"].iat[0] == 90
    assert scored.loc[scored["sector"].eq("XLK"), "chase_risk_score"].isna().all()
    print("self-test: PASS")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--database", type=Path, default=ROOT / "regime_router_publish.db")
    result.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "sector_turning_dashboard_scores_20260918",
    )
    result.add_argument("--start", default="2010-01-04")
    result.add_argument("--end", default="2099-12-31")
    result.add_argument("--evaluation-start", default="2016-01-04")
    result.add_argument("--min-train-dates", type=int, default=756)
    result.add_argument("--refit-every", type=int, default=63)
    result.add_argument("--self-test", action="store_true")
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    if arguments.self_test:
        self_test()
    else:
        result = run(arguments)
        print(json.dumps({"output": str(arguments.output_dir.resolve()), "verdict": result["verdict"]}))
