from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from regime_router import (
    INDEPENDENT_A2C_HMM_FEATURE_COLUMNS,
    build_independent_a2c_hmm_features,
    build_independent_a3a_hmm_features,
    load_benchmark_from_database,
    load_vix_curve_market_from_database,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def effective_rank(frame) -> float:
    eigenvalues = np.linalg.eigvalsh(frame.corr().to_numpy())
    weights = eigenvalues[eigenvalues > 0] / eigenvalues.sum()
    return float(np.exp(-np.sum(weights * np.log(weights))))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run label-free A3-A input checks")
    parser.add_argument("database", type=Path)
    parser.add_argument("--benchmark", default="NQ=F")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    benchmark_market = load_benchmark_from_database(args.database, args.benchmark)
    curve_market = load_vix_curve_market_from_database(args.database, args.benchmark)
    a2c = build_independent_a2c_hmm_features(benchmark_market)
    a3a = build_independent_a3a_hmm_features(curve_market)
    full_corr = a3a.corr()["vix_curve"].drop("vix_curve").abs()
    recent_corr = a3a.tail(1260).corr()["vix_curve"].drop("vix_curve").abs()
    retention = len(a3a) / len(a2c)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "database": str(args.database),
        "database_sha256": file_sha256(args.database),
        "benchmark": args.benchmark,
        "raw_exact_positive_rows": len(curve_market),
        "raw_start": curve_market["date"].min().strftime("%Y-%m-%d"),
        "raw_end": curve_market["date"].max().strftime("%Y-%m-%d"),
        "duplicate_dates": int(curve_market["date"].duplicated().sum()),
        "a2c_rows": len(a2c),
        "a3a_rows": len(a3a),
        "usable_row_retention": retention,
        "a3a_start": a3a.index.min().strftime("%Y-%m-%d"),
        "a3a_end": a3a.index.max().strftime("%Y-%m-%d"),
        "feature_columns": list(a3a.columns),
        "full_vix_curve_absolute_correlations": full_corr.to_dict(),
        "recent_1260_vix_curve_absolute_correlations": recent_corr.to_dict(),
        "max_full_absolute_correlation": float(full_corr.max()),
        "max_recent_1260_absolute_correlation": float(recent_corr.max()),
        "a2c_effective_rank": effective_rank(a2c),
        "a3a_effective_rank": effective_rank(a3a),
        "recent_1260_a2c_effective_rank": effective_rank(a3a.tail(1260).iloc[:, :3]),
        "recent_1260_a3a_effective_rank": effective_rank(a3a.tail(1260)),
        "checks": {
            "usable_row_retention_at_least_99pct": bool(retention >= 0.99),
            "no_duplicate_dates": bool(not curve_market["date"].duplicated().any()),
            "feature_columns_exact": bool(
                list(a3a.columns)
                == [*INDEPENDENT_A2C_HMM_FEATURE_COLUMNS, "vix_curve"]
            ),
            "max_full_absolute_correlation_below_0_70": bool(full_corr.max() < 0.70),
            "max_recent_absolute_correlation_below_0_70": bool(
                recent_corr.max() < 0.70
            ),
        },
    }
    report["preflight_pass"] = all(report["checks"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["preflight_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
