"""Fixed, causal rejection filters on cached GICS alerts; no fitting or publishing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sector_turning_layer_weight_experiment import _json_ready
from sector_turning_reference_policy_experiment import confirmed_alerts, policy_metrics

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "outputs/sector_turning_price_context_confirmation_20260904/signals.csv"


def filters(rows: pd.DataFrame) -> dict[str, pd.Series]:
    base = rows["price_watch_alert"].astype(str).str.lower().map({"true": True, "false": False})
    if base.isna().any():
        raise ValueError("price_watch_alert must contain explicit True/False values")
    base = base.astype(bool)
    # Fixed before evaluation; reject only, never move or create an alert.
    persistence = confirmed_alerts(rows, rows["price_score"], 0.90, 2).astype(bool)
    breadth = rows["breadth_context_score"].ge(0.60)
    macro = rows["macro_context_score"].ge(0.60)
    return {
        "BASELINE": base,
        "PERSISTENCE_2_OF_3": base & persistence,
        "BREADTH_CONFIRM": base & breadth,
        "PERSISTENCE_AND_BREADTH": base & persistence & breadth,
        "PERSISTENCE_BREADTH_MACRO": base & persistence & breadth & macro,
    }


def run(source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    rows = pd.read_csv(source, dtype={"sector": "string"})
    for col in ("date", "target_event_confirmation_date"):
        rows[col] = pd.to_datetime(rows[col], errors="raise")
    if rows.duplicated(["direction", "sector", "date"]).any():
        raise ValueError("duplicate direction/sector/date")
    scores = ["price_score", "breadth_context_score", "macro_context_score"]
    numeric = rows[scores].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy()).all() or not numeric.ge(0).all().all() or not numeric.le(1).all().all():
        raise ValueError("scores must be finite percentiles in [0,1]")
    records = []
    for direction, group in rows.groupby("direction"):
        policies = filters(group)
        for year in sorted(group["date"].dt.year.unique()):
            sample = group.loc[group["date"].dt.year.eq(year)]
            baseline = policy_metrics(sample, policies["BASELINE"].loc[sample.index])
            for name, alerts in policies.items():
                metrics = policy_metrics(sample, alerts.loc[sample.index])
                count = metrics["alert_clusters"]
                recall = metrics["pre_pivot_recall"]
                retention = recall / baseline["pre_pivot_recall"] if baseline["pre_pivot_recall"] else None
                records.append({
                    "direction": direction, "year": int(year), "policy": name,
                    "alert_clusters": count,
                    "false_discovery_fraction": 1 - metrics["alert_cluster_precision"] if count else None,
                    "false_clusters_per_sector_year": metrics["false_alert_clusters_per_sector_year"],
                    "pre_pivot_recall": recall, "recall_retention": retention,
                    "median_lead_sessions": metrics["median_lead_to_pivot_sessions"],
                    "screen_sample_and_recall_ok": count >= 20 and retention is not None and retention >= 0.75,
                })
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(records).to_csv(output / "comparison.csv", index=False)
    (output / "summary.json").write_text(json.dumps(_json_ready({
        "status": "DESCRIPTIVE_RESEARCH_ONLY_NO_WINNER_SELECTED",
        "source": str(source.resolve()), "source_rows": len(rows),
        "scope": "Cached GICS turning alerts, NOT live 11-ETF rank-entry alerts",
        "warnings": [
            "Reviewed historical sample; not untouched out-of-sample evidence.",
            "PIT_PARTIAL. No probability calibration or significance claim.",
            "False discovery fraction is unmatched alert clusters / all alert clusters, not FPR.",
            "Year slices use existing event matching and may censor boundary events.",
            "No-alert policy has undefined error fraction, not zero error.",
            "20 clusters and 75% recall retention are screening guards, not promotion criteria.",
            "No model, DB, web, or deployment changes. No downloads or paid API calls.",
        ],
    }), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def self_test() -> None:
    rows = pd.DataFrame({"sector": ["A"] * 5, "_position": [1, 2, 3, 5, 6],
                         "price_score": [.95] * 5, "price_watch_alert": [True] * 5,
                         "breadth_context_score": [.7, .7, .4, .7, .7],
                         "macro_context_score": [.7] * 5})
    result = filters(rows)
    assert result["PERSISTENCE_AND_BREADTH"].tolist() == [False, True, False, False, True]
    assert all((~value | result["BASELINE"]).all() for value in result.values())
    rows.loc[4, "price_score"] = .1
    assert filters(rows)["PERSISTENCE_AND_BREADTH"].iloc[:4].equals(result["PERSISTENCE_AND_BREADTH"].iloc[:4])
    print("self-test: PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/sector_noise_audit_20260921")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        run(args.source, args.output_dir)
        print(f"Research output: {args.output_dir.resolve()}")
