"""Tune the four historical Market Lens sector layers without touching live paths."""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from research_drawdown_onset import causal_trailing_percentile
from sector_rotation import SECTOR_ETFS, build_rrg_signals, load_price_matrix
from sector_turning_pooled_shadow import (
    _early_event_metrics,
    _json_ready,
    paired_date_block_auc_delta_ci,
    paired_date_block_pre_pivot_recall_delta_ci,
)


ROOT = Path(__file__).resolve().parent
MACRO_PREDICTIONS = ROOT / "outputs" / "sector_turning_sector_macro_long_core_20260903" / "predictions.csv"
BREADTH_PREDICTIONS = ROOT / "outputs" / "sector_turning_beta_breadth_shadow_20260903" / "predictions.csv"
MASTER = ROOT / "outputs" / "bloomberg_gics_breadth_full_request_20260903_130258" / "industry_group_master.csv"
DATABASE = ROOT / "regime_router_publish.db"
OUTPUT = ROOT / "outputs" / "sector_turning_layer_weights_20260903_v2"
COMPONENTS = ("rrg_score", "price_score", "breadth_score", "macro_score")
TRAIN_END = pd.Timestamp("2022-12-30")
VALID_START = pd.Timestamp("2023-01-03")
VALID_END = pd.Timestamp("2024-12-31")
TEST_START = pd.Timestamp("2025-01-02")
CUTOFF = 0.60


def weight_grid(step: float = 0.2) -> list[tuple[float, ...]]:
    units = round(1 / step)
    return [
        tuple(value / units for value in values)
        for values in product(range(units + 1), repeat=4)
        if sum(values) == units
    ]


def exhaustion_percentile(values: pd.Series) -> pd.Series:
    ranked = causal_trailing_percentile(values.where(values.gt(0)), window=504, min_history=60)
    return ranked.where(values.gt(0), 0.0).where(values.notna())


def rrg_context() -> pd.DataFrame:
    signals = build_rrg_signals(load_price_matrix(DATABASE)).sort_values(["ticker", "date"])
    signals["momentum_delta_5"] = signals.groupby("ticker")["rs_momentum"].diff(5)
    rows = []
    for direction in ("ROTATION_OUT", "ROTATION_IN"):
        ratio = signals["rs_ratio"] - 100
        momentum = signals["momentum_delta_5"]
        if direction == "ROTATION_OUT":
            raw = np.sqrt(ratio.clip(lower=0) * (-momentum).clip(lower=0))
        else:
            raw = np.sqrt((-ratio).clip(lower=0) * momentum.clip(lower=0))
        part = signals[["date", "ticker"]].copy()
        part["direction"] = direction
        part["rrg_score"] = raw
        part["rrg_score"] = part.groupby("ticker", group_keys=False)["rrg_score"].transform(
            exhaustion_percentile
        )
        rows.append(part)
    return pd.concat(rows, ignore_index=True).dropna(subset=["rrg_score"])


def load_panel() -> pd.DataFrame:
    dates = ["date", "target_event_confirmation_date"]
    macro = pd.read_csv(MACRO_PREDICTIONS, parse_dates=dates, dtype={"sector": "string"})
    macro["sector"] = macro["sector"].str.zfill(4)
    keys = ["direction", "date", "sector"]
    base_columns = [
        *keys, "_position", "sector_name", "target", "target_event_pivot_position",
        "target_event_confirmation_position", "target_event_confirmation_date",
    ]
    panel = macro.loc[macro["model"].eq("PRICE_SHALLOW"), [*base_columns, "score"]].rename(
        columns={"score": "price_score"}
    )
    macro_score = macro.loc[
        macro["model"].eq("SECTOR_MACRO_NETWORK_SHALLOW"), [*keys, "score"]
    ].rename(columns={"score": "macro_score"})
    panel = panel.merge(macro_score, on=keys, validate="one_to_one")

    breadth = pd.read_csv(BREADTH_PREDICTIONS, parse_dates=["date"], dtype={"sector": "string"})
    breadth["sector"] = breadth["sector"].str.zfill(4)
    breadth = breadth.loc[
        breadth["model"].eq("POOLED_BETA_BREADTH"), [*keys, "score"]
    ].rename(columns={"score": "breadth_score"})
    panel = panel.merge(breadth, on=keys, validate="one_to_one")

    master = pd.read_csv(MASTER, dtype="string")
    ticker_for_parent = {name: ticker for ticker, name in SECTOR_ETFS.items()}
    master["sector"] = master["industry_group_code"].str.zfill(4)
    master["parent_ticker"] = master["parent_sector_name"].map(ticker_for_parent)
    panel = panel.merge(
        master[["sector", "parent_ticker"]], on="sector", validate="many_to_one"
    )
    panel = panel.merge(
        rrg_context(), left_on=["direction", "date", "parent_ticker"],
        right_on=["direction", "date", "ticker"], validate="many_to_one",
    ).drop(columns="ticker")
    panel = panel.dropna(subset=[*COMPONENTS, "target"]).sort_values(
        ["direction", "sector", "date"]
    ).reset_index(drop=True)
    if panel.duplicated(keys).any():
        raise ValueError("duplicate direction/date/sector rows")
    return panel


def score_stack(rows: pd.DataFrame, weights: tuple[float, ...]) -> pd.Series:
    raw = sum(rows[column] * weight for column, weight in zip(COMPONENTS, weights, strict=True))
    ordered = rows.assign(_raw=raw).sort_values(["sector", "date"])
    calibrated = ordered.groupby("sector", group_keys=False)["_raw"].transform(
        lambda values: causal_trailing_percentile(values, window=504, min_history=60)
    )
    return calibrated.reindex(rows.index)


def metrics(rows: pd.DataFrame, score: pd.Series) -> dict:
    evaluated = rows.assign(score=score).dropna(subset=["score"])
    event = _early_event_metrics(evaluated, pd.DataFrame(), "", horizon=10, cutoff=CUTOFF)
    return {
        "auc": float(roc_auc_score(evaluated["target"].astype(int), evaluated["score"])),
        **event,
    }


def period(rows: pd.DataFrame, name: str) -> pd.DataFrame:
    if name == "train":
        return rows.loc[rows["date"].le(TRAIN_END)]
    if name == "validation":
        return rows.loc[rows["date"].between(VALID_START, VALID_END)]
    if name == "test":
        return rows.loc[rows["date"].ge(TEST_START)]
    raise ValueError(name)


def tune_direction(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    grid_rows = []
    scored = {}
    for weights in weight_grid():
        score = score_stack(rows, weights)
        scored[weights] = score
        record = dict(zip(COMPONENTS, weights, strict=True))
        for name in ("train", "validation"):
            subset = period(rows, name)
            result = metrics(subset, score.loc[subset.index])
            record.update({
                f"{name}_recall": result["pre_pivot_recall"],
                f"{name}_auc": result["auc"],
                f"{name}_alert_rate": result["alert_day_rate"],
            })
        grid_rows.append(record)
    grid = pd.DataFrame(grid_rows)
    top_train = grid.sort_values(["train_recall", "train_auc"], ascending=False).head(5)
    chosen = top_train.sort_values(["validation_recall", "validation_auc"], ascending=False).iloc[0]
    weights = tuple(float(chosen[name]) for name in COMPONENTS)
    final_score = scored[weights]

    component_results = {}
    for column in COMPONENTS:
        component_results[column] = {
            name: metrics(period(rows, name), rows.loc[period(rows, name).index, column])
            for name in ("train", "validation", "test")
        }
    stack_results = {
        name: metrics(period(rows, name), final_score.loc[period(rows, name).index])
        for name in ("train", "validation", "test")
    }
    reference = max(
        COMPONENTS,
        key=lambda column: (
            component_results[column]["validation"]["pre_pivot_recall"],
            component_results[column]["validation"]["auc"],
        ),
    )
    test = period(rows, "test").assign(score=final_score.loc[period(rows, "test").index])
    baseline = period(rows, "test").assign(score=rows.loc[period(rows, "test").index, reference])
    paired = pd.DataFrame({
        "target": test["target"].to_numpy(),
        "candidate": test["score"].to_numpy(),
        "baseline": baseline["score"].to_numpy(),
    }, index=pd.MultiIndex.from_arrays([test["date"], test["sector"]], names=["date", "sector"]))
    auc_ci = paired_date_block_auc_delta_ci(
        paired["target"], paired["candidate"], paired["baseline"], samples=400,
        block_dates=40, calendar_dates=test["date"].drop_duplicates(),
    )
    recall_ci = paired_date_block_pre_pivot_recall_delta_ci(
        test, baseline, samples=400, block_dates=40,
        calendar_dates=test["date"].drop_duplicates(), horizon=10, cutoff=CUTOFF,
    )
    output = rows.copy()
    output["tuned_score"] = final_score
    return grid, output, {
        "weights": dict(zip(COMPONENTS, weights, strict=True)),
        "reference_selected_on_validation": reference,
        "components": component_results,
        "tuned_stack": stack_results,
        "test_auc_delta_vs_reference": auc_ci,
        "test_recall_delta_vs_reference": recall_ci,
    }


def self_test() -> None:
    grid = weight_grid()
    assert len(grid) == 56 and all(abs(sum(weights) - 1) < 1e-12 for weights in grid)
    sample = pd.Series([1.0, 2.0, 3.0, 99.0])
    before = causal_trailing_percentile(sample, window=10, min_history=2)
    changed = sample.copy(); changed.iloc[-1] = -99
    after = causal_trailing_percentile(changed, window=10, min_history=2)
    assert before.iloc[:-1].equals(after.iloc[:-1])
    assert exhaustion_percentile(pd.Series([0.0, 0.0, 1.0])).iloc[:2].eq(0).all()
    print("self-test: PASS")


def main() -> None:
    panel = load_panel()
    grid_parts, score_parts, directions = [], [], {}
    for direction, rows in panel.groupby("direction", sort=True):
        grid, scores, result = tune_direction(rows.copy())
        grid.insert(0, "direction", direction)
        grid_parts.append(grid)
        score_parts.append(scores)
        directions[direction] = result
    summary = {
        "verdict": "RESEARCH_SHADOW_ONLY",
        "pit_status": "PIT_PARTIAL/CURRENT_REVISED_HISTORY_ONLY",
        "spec": {
            "layers": list(COMPONENTS), "grid_step": 0.2, "cutoff": CUTOFF,
            "selection": "top-5 train recall, then validation recall; AUC tie-break",
            "train": f"through {TRAIN_END.date()}",
            "validation": f"{VALID_START.date()}..{VALID_END.date()}",
            "test": f"{TEST_START.date()}..latest matured label",
            "rrg_layer": "11 parent-sector ETF RRG exhaustion propagated to GICS groups",
        },
        "panel": {
            "rows": len(panel), "groups": panel["sector"].nunique(),
            "first_date": str(panel["date"].min().date()),
            "last_matured_date": str(panel["date"].max().date()),
        },
        "directions": directions,
    }
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    pd.concat(grid_parts, ignore_index=True).to_csv(OUTPUT / "weight_grid.csv", index=False)
    pd.concat(score_parts, ignore_index=True).to_csv(OUTPUT / "scores.csv", index=False)
    clean = _json_ready(summary)
    (OUTPUT / "summary.json").write_text(
        json.dumps(clean, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({"output": str(OUTPUT), "verdict": clean["verdict"]}, ensure_ascii=False))


if __name__ == "__main__":
    import sys

    self_test() if "--self-test" in sys.argv else main()
