"""Historical ranking validation for the causal NQ rule-based risk score."""

from __future__ import annotations

import argparse
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from regime_router import LIVE_HMM_MODEL_VERSION, load_benchmark_from_database
from router_score import ROUTER_SCORE_VERSION, build_router_scores
from validate_hmm import _future_drawdown, paired_auc_comparison, score_metrics


def build_validation_frame(
    database: Path,
    *,
    horizon: int = 20,
    drawdown_threshold: float = -0.05,
    score_builder: Callable[[pd.DataFrame], pd.DataFrame] = build_router_scores,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    market = load_benchmark_from_database(database, "NQ=F")
    market["date"] = pd.to_datetime(market["date"])
    market = market.sort_values("date").drop_duplicates("date", keep="last")
    close = pd.to_numeric(market["close"], errors="coerce")
    market["future_drawdown"] = _future_drawdown(close.to_numpy(), horizon)
    market["drawdown_event"] = (
        market["future_drawdown"] <= drawdown_threshold
    ).astype(int)
    market["volatility_20"] = close.pct_change().rolling(20).std() * np.sqrt(252)

    scores = score_builder(market[["date", "close"]]).reset_index()
    with closing(sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)) as connection:
        hmm = pd.read_sql_query(
            """
            SELECT date, p_bull, p_sideways, p_bear
            FROM regime_model_probabilities
            WHERE model_version=?
            ORDER BY date
            """,
            connection,
            params=(LIVE_HMM_MODEL_VERSION,),
            parse_dates=["date"],
        )
    hmm["hmm_risk_score"] = (
        10 * hmm["p_bull"] + 50 * hmm["p_sideways"] + 80 * hmm["p_bear"]
    )
    full = (
        market.merge(
            scores[["date", "risk_score", "score_raw"]],
            on="date",
            how="left",
            validate="one_to_one",
        )
        .merge(hmm[["date", "hmm_risk_score"]], on="date", how="left", validate="one_to_one")
        .sort_values("date")
    )
    labelled = full.dropna(
        subset=[
            "future_drawdown",
            "risk_score",
            "hmm_risk_score",
            "volatility_20",
        ]
    ).reset_index(drop=True)
    return labelled, full


def validate_router_score(
    database: Path,
    *,
    horizon: int = 20,
    drawdown_threshold: float = -0.05,
    bootstrap_samples: int = 500,
    score_builder: Callable[[pd.DataFrame], pd.DataFrame] = build_router_scores,
    model_version: str = ROUTER_SCORE_VERSION,
) -> dict[str, object]:
    frame, full = build_validation_frame(
        database,
        horizon=horizon,
        drawdown_threshold=drawdown_threshold,
        score_builder=score_builder,
    )
    models = {
        "RULE_SCORE": score_metrics(
            frame, "risk_score", bootstrap_samples=bootstrap_samples, episode_separation=horizon
        ),
        "HMM_DISPLAY_SCORE": score_metrics(
            frame, "hmm_risk_score", bootstrap_samples=bootstrap_samples, episode_separation=horizon
        ),
        "VOLATILITY_20": score_metrics(
            frame, "volatility_20", bootstrap_samples=bootstrap_samples, episode_separation=horizon
        ),
    }
    paired = paired_auc_comparison(
        frame,
        "risk_score",
        "hmm_risk_score",
        bootstrap_samples=bootstrap_samples,
    )
    paired_vs_volatility = paired_auc_comparison(
        frame,
        "risk_score",
        "volatility_20",
        bootstrap_samples=bootstrap_samples,
    )
    latest_scores = full["risk_score"].dropna().astype(int)
    recent = latest_scores.tail(63)
    recent_diagnostics = {
        "end": full.loc[latest_scores.index[-1], "date"].date().isoformat(),
        "unique_scores": int(recent.nunique()),
        "score_changes": int(recent.diff().ne(0).sum() - 1),
        "mode_share": float(recent.value_counts(normalize=True).iloc[0]),
        "latest_score": int(recent.iloc[-1]),
    }
    rule = models["RULE_SCORE"]
    gates = {
        "minimum_observations": int(rule["n"]) >= 1_000,
        "auc_ci_above_random": float(rule["auc_ci_low"]) > 0.50,
        "top_quintile_lift": float(rule["top_quintile_lift"]) >= 1.25,
        "beats_hmm_display_score": float(paired["auc_delta"]) >= 0,
        "paired_noninferiority": float(paired["auc_delta_ci_low"]) >= -0.02,
        "simple_volatility_noninferiority": float(
            paired_vs_volatility["auc_delta"]
        )
        >= -0.02,
        "recent_score_diversity": recent_diagnostics["unique_scores"] >= 8,
        "recent_score_movement": recent_diagnostics["score_changes"] >= 12,
        "recent_mode_share": recent_diagnostics["mode_share"] <= 0.50,
    }
    return {
        "role": "RULE_BASED_REGIME_SEVERITY",
        "model_version": model_version,
        "status": "SIGNAL_PASS" if all(gates.values()) else "SIGNAL_FAIL",
        "definition": {
            "benchmark": "NQ=F",
            "horizon": horizon,
            "drawdown_threshold": drawdown_threshold,
            "note": "The score is a current market-risk severity rank, not a calibrated future-loss probability.",
        },
        "coverage": {
            "rows": len(frame),
            "start": frame["date"].min().date().isoformat(),
            "labelled_end": frame["date"].max().date().isoformat(),
            "score_end": recent_diagnostics["end"],
        },
        "models": models,
        "paired_vs_hmm": paired,
        "paired_vs_volatility": paired_vs_volatility,
        "recent_63": recent_diagnostics,
        "gates": gates,
        "overall_pass": all(gates.values()),
    }


def markdown_report(report: dict[str, object]) -> str:
    rows = [
        "# NQ Rule-Based Risk Score Validation",
        "",
        f"- Model: `{report['model_version']}`",
        f"- Status: **{report['status']}**",
        f"- Coverage: {report['coverage']['start']} to {report['coverage']['labelled_end']} ({report['coverage']['rows']:,} rows)",
        f"- Latest score: {report['recent_63']['latest_score']} ({report['coverage']['score_end']})",
        "- The score is not a calibrated probability of a future decline.",
        "",
        "| Score | ROC-AUC | 95% CI | Top-quintile lift |",
        "|---|---:|---:|---:|",
    ]
    for name, metric in report["models"].items():
        rows.append(
            f"| {name} | {metric['roc_auc']:.3f} | {metric['auc_ci_low']:.3f}-{metric['auc_ci_high']:.3f} | {metric['top_quintile_lift']:.2f}x |"
        )
    rows.extend(["", "## Gates", "", "| Gate | Result |", "|---|---|"])
    rows.extend(
        f"| {name} | {'PASS' if passed else 'FAIL'} |"
        for name, passed in report["gates"].items()
    )
    return "\n".join(rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the causal NQ rule score")
    parser.add_argument("database", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/router_score_v1"))
    parser.add_argument("--router-score-v2", action="store_true")
    args = parser.parse_args()
    score_builder = build_router_scores
    model_version = ROUTER_SCORE_VERSION
    if args.router_score_v2:
        from router_score_v2 import (
            ROUTER_SCORE_VERSION as ROUTER_SCORE_VERSION_V2,
            build_router_scores_v2,
        )

        score_builder = build_router_scores_v2
        model_version = ROUTER_SCORE_VERSION_V2
    report = validate_router_score(
        args.database,
        bootstrap_samples=args.bootstrap_samples,
        score_builder=score_builder,
        model_version=model_version,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "validation.md").write_text(
        markdown_report(report), encoding="utf-8"
    )
    print(f"status={report['status']} latest_score={report['recent_63']['latest_score']}")
    for name, passed in report["gates"].items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
