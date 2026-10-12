from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from regime_router import (
    _FlooredFullGaussianHMM,
    _four_state_diagnostics,
    _initial_transition_matrix,
    build_independent_a2c_hmm_features,
    load_benchmark_from_database,
)


MODEL_VERSION = "hmm4-sticky-nq-independent-a2g-logvol-v1-full-ef10-k25-preflight"


def run_preflight(database: Path) -> dict[str, object]:
    features = build_independent_a2c_hmm_features(
        load_benchmark_from_database(database, "NQ=F")
    )
    positions = (252, 2500, 4812)
    rows = []
    for position in positions:
        train = features.iloc[max(0, position - 1260) : position]
        scaled = StandardScaler().fit_transform(train)
        prior = np.ones((4, 4))
        np.fill_diagonal(prior, 25)
        seeds = []
        for seed in (42, 43, 44):
            model = _FlooredFullGaussianHMM(
                n_components=4,
                covariance_type="full",
                n_iter=200,
                min_covar=0.10,
                random_state=seed,
                transmat_prior=prior,
                init_params="mc",
            )
            model.covariance_floor = 0.10
            model.startprob_ = np.full(4, 0.25)
            model.transmat_ = _initial_transition_matrix(4)
            model.fit(scaled)
            history = list(model.monitor_.history)
            stable = bool(model.monitor_.converged) and (
                len(history) < 2 or history[-1] >= history[-2] - 1e-3
            )
            diagnostics = _four_state_diagnostics(model, scaled, train.columns)
            seeds.append(
                {
                    "seed": seed,
                    "stable": stable,
                    "admissible": stable and bool(diagnostics["admissible"]),
                    "occupancy": [float(value) for value in diagnostics["occupancy"]],
                    "minimum_centroid_gap": float(diagnostics["minimum_centroid_gap"]),
                    "train_log_likelihood": float(model.score(scaled)),
                }
            )
        rows.append(
            {
                "refit_date": features.index[position].date().isoformat(),
                "train_start": train.index.min().date().isoformat(),
                "train_end": train.index.max().date().isoformat(),
                "train_rows": len(train),
                "seeds": seeds,
                "admissible_seed_count": sum(item["admissible"] for item in seeds),
            }
        )
    return {
        "model_version": MODEL_VERSION,
        "status": "PREFLIGHT_PASS" if all(row["admissible_seed_count"] for row in rows) else "PREFLIGHT_FAIL",
        "database": database.name,
        "gates": {"latent_occupancy_min": 0.05, "latent_occupancy_max": 0.80, "minimum_centroid_gap": 0.25},
        "windows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the isolated four-state preflight")
    parser.add_argument("database", type=Path)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    result = run_preflight(args.database)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
