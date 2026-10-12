"""Compare price, price+macro, and price+macro+options on one frozen OOS panel."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from sector_turning_layer_weight_experiment import _json_ready
from sector_turning_leading_joint_weight_experiment import (
    choose,
    false_delta_ci,
    frozen,
    scan,
    stack,
    window,
)
from sector_turning_model_policy_scan import forward_returns
from sector_turning_pooled_shadow import (
    paired_date_block_auc_delta_ci,
    paired_date_block_pre_pivot_recall_delta_ci,
)


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "outputs" / "sector_turning_sector_macro_long_core_20260903" / "predictions.csv"
OPTION = ROOT / "outputs" / "sector_turning_option_full_20260904" / "predictions.csv"
OUTPUT = ROOT / "outputs" / "sector_turning_layer_progression_full24_20260904_v2"
MODELS = {
    "PRICE_ONLY": ("price_score",),
    "PRICE_MACRO": ("price_score", "macro_score"),
    "PRICE_MACRO_OPTION": ("price_score", "macro_score", "option_score"),
}


def has_incremental_option_value(directions: dict[str, Any]) -> bool:
    for item in directions.values():
        comparison = item["incremental_comparisons"]["PRICE_MACRO_OPTION_VS_PRICE_MACRO"]
        recall_low = comparison["recall_delta"]["early_capture_ci_low"]
        false_high = comparison["false_alarm_delta"]["ci_high"]
        auc_low = comparison["auc_delta"]["ci_low"]
        if (
            auc_low > 0
            or (recall_low is not None and recall_low > 0 and false_high <= 0)
            or (false_high < 0 and recall_low is not None and recall_low >= 0)
        ):
            return True
    return False


def score_auc(rows: pd.DataFrame, score: pd.Series) -> float:
    sample = window(rows, "evaluation_2023_2026")
    values = score.loc[sample.index]
    return float(roc_auc_score(sample["target"].astype(int), values))


def load_progression_panel(base_csv: Path, option_csv: Path) -> pd.DataFrame:
    dates = ["date", "target_event_confirmation_date"]
    base = pd.read_csv(base_csv, parse_dates=dates, dtype={"sector": "string"})
    base["sector"] = base["sector"].str.zfill(4)
    keys = ["direction", "date", "sector"]
    base_columns = [
        *keys, "_position", "sector_name", "target", "target_event_pivot_position",
        "target_event_confirmation_position", "target_event_confirmation_date",
    ]
    panel = base.loc[base["model"].eq("PRICE_SHALLOW"), [*base_columns, "score"]].rename(
        columns={"score": "price_score"}
    )
    macro = base.loc[
        base["model"].eq("SECTOR_MACRO_NETWORK_SHALLOW"), [*keys, "score"]
    ].rename(columns={"score": "macro_score"})
    panel = panel.merge(macro, on=keys, validate="one_to_one")
    option = pd.read_csv(option_csv, parse_dates=["date"], dtype={"sector": "string"})
    option["sector"] = option["sector"].str.zfill(4)
    option = option.loc[option["model"].eq("OPTION_LOGIT"), [*keys, "score"]].rename(
        columns={"score": "option_score"}
    )
    panel = panel.merge(option, on=keys, validate="one_to_one")
    panel = panel.merge(
        forward_returns(), on=["date", "sector", "_position"], how="left", validate="many_to_one"
    )
    panel["signed_forward_return_20"] = panel["forward_relative_return_20"] * np.where(
        panel["direction"].eq("ROTATION_IN"), 1, -1
    )
    panel = panel.dropna(subset=["price_score", "macro_score", "option_score", "target"])
    if panel.duplicated(keys).any():
        raise ValueError("progression panel is duplicated")
    return panel.sort_values(["direction", "sector", "_position"]).reset_index(drop=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    panel = load_progression_panel(args.base_csv.resolve(), args.option_csv.resolve())
    directions: dict[str, Any] = {}
    sweep_parts, signal_parts = [], []
    for direction, rows in panel.groupby("direction", sort=True):
        rows = rows.copy()
        chosen_alerts, chosen_scores, results = {}, {}, {}
        for model, columns in MODELS.items():
            sweep, cache = scan(rows, columns)
            chosen, passed = choose(sweep)
            weights = tuple(float(chosen[column]) for column in columns)
            score = stack(rows, columns, weights)
            alerts = cache[chosen["_key"]]
            chosen_alerts[model], chosen_scores[model] = alerts, score
            metrics = frozen(rows, alerts)
            metrics["auc"] = score_auc(rows, score)
            results[model] = {
                "selection_constraints_passed": passed,
                "policy": {
                    "weights": dict(zip(columns, weights, strict=True)),
                    "cutoff": float(chosen["cutoff"]),
                    "cooldown_sessions": int(chosen["cooldown_sessions"]),
                },
                "evaluation_2023_2026": metrics,
            }
            clean = sweep.drop(columns="_key")
            clean.insert(0, "direction", direction)
            clean.insert(1, "model", model)
            sweep_parts.append(clean)

        comparisons = {}
        for candidate, baseline in (("PRICE_MACRO", "PRICE_ONLY"), ("PRICE_MACRO_OPTION", "PRICE_MACRO")):
            evaluation = window(rows, "evaluation_2023_2026")
            candidate_frame = evaluation.assign(
                score=chosen_alerts[candidate].loc[evaluation.index].astype(float)
            )
            baseline_frame = evaluation.assign(
                score=chosen_alerts[baseline].loc[evaluation.index].astype(float)
            )
            paired = pd.DataFrame({
                "target": evaluation["target"].astype(int).to_numpy(),
                "candidate": chosen_scores[candidate].loc[evaluation.index].to_numpy(),
                "baseline": chosen_scores[baseline].loc[evaluation.index].to_numpy(),
            }, index=pd.MultiIndex.from_arrays(
                [evaluation["date"], evaluation["sector"]], names=["date", "sector"]
            ))
            comparisons[f"{candidate}_VS_{baseline}"] = {
                "false_alarm_delta": false_delta_ci(
                    evaluation, chosen_alerts[candidate], chosen_alerts[baseline]
                ),
                "recall_delta": paired_date_block_pre_pivot_recall_delta_ci(
                    candidate_frame, baseline_frame, samples=2_000, block_dates=40,
                    calendar_dates=evaluation["date"].drop_duplicates(), horizon=10, cutoff=0.5,
                ),
                "auc_delta": paired_date_block_auc_delta_ci(
                    paired["target"], paired["candidate"], paired["baseline"],
                    samples=2_000, block_dates=40,
                    calendar_dates=evaluation["date"].drop_duplicates(),
                ),
            }
        output = rows[["direction", "date", "sector", "sector_name", "target"]].copy()
        for model in MODELS:
            output[f"{model.lower()}_score"] = chosen_scores[model]
            output[f"{model.lower()}_alert"] = chosen_alerts[model]
        signal_parts.append(output)
        directions[direction] = {"models": results, "incremental_comparisons": comparisons}

    option_useful = has_incremental_option_value(directions)
    summary = _json_ready({
        "verdict": "RESEARCH_REFERENCE_CANDIDATE" if option_useful else "RESEARCH_ONLY",
        "pit_status": "PIT_NOT_VERIFIED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection": "2021 and 2022 separately; frozen evaluation 2023-2026",
        "panel": {
            "rows": len(panel), "groups": panel["sector"].nunique(),
            "first_date": str(panel["date"].min().date()),
            "last_date": str(panel["date"].max().date()),
        },
        "directions": directions,
        "operational_changes": False,
    })
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    pd.concat(sweep_parts, ignore_index=True).to_csv(output_dir / "policy_sweep.csv", index=False)
    pd.concat(signal_parts, ignore_index=True).to_csv(output_dir / "signals.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    return summary


def self_test() -> None:
    assert tuple(MODELS) == ("PRICE_ONLY", "PRICE_MACRO", "PRICE_MACRO_OPTION")
    assert MODELS["PRICE_MACRO_OPTION"][-1] == "option_score"
    comparison = {
        "incremental_comparisons": {"PRICE_MACRO_OPTION_VS_PRICE_MACRO": {
            "recall_delta": {"early_capture_ci_low": 0.1},
            "false_alarm_delta": {"ci_high": 0.1},
            "auc_delta": {"ci_low": 0.0},
        }}
    }
    assert not has_incremental_option_value({"IN": comparison})
    comparison["incremental_comparisons"]["PRICE_MACRO_OPTION_VS_PRICE_MACRO"]["false_alarm_delta"]["ci_high"] = 0.0
    assert has_incremental_option_value({"IN": comparison})
    print("self-test: PASS")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--base-csv", type=Path, default=BASE)
    result.add_argument("--option-csv", type=Path, default=OPTION)
    result.add_argument("--output-dir", type=Path, default=OUTPUT)
    result.add_argument("--self-test", action="store_true")
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    if arguments.self_test:
        self_test()
    else:
        result = run(arguments)
        print(json.dumps({"output": str(arguments.output_dir.resolve()), "verdict": result["verdict"]}, ensure_ascii=False))
