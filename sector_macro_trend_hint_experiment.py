"""Causal SHADOW test of macro trend exhaustion before sector turns."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
PACKAGE = ROOT / "outputs" / "gics_l2_staging_20260902_145743" / "bloomberg_gics_l2_20260902_145743"
MACRO_CSV = ROOT / "outputs" / "bloomberg_turning_macro_staging_20260902_212424" / "bloomberg_turning_macro_20260902_212424" / "macro_history.csv"
SECTOR_SOURCE = ROOT / "outputs" / "sector_turning_recall_breadth_boost_20260903" / "predictions.csv"
DATABASE = ROOT / "regime_router_publish.db"
OUTPUT_DIR = ROOT / "outputs" / "sector_macro_trend_hint_20260903"
TRAIN_START = pd.Timestamp("2018-01-02")
TRAIN_END = pd.Timestamp("2022-12-30")
TEST_START = pd.Timestamp("2023-01-03")
ALERT_CUTOFF = 0.80
BOOTSTRAP_SAMPLES = 400


def auc_score(target: pd.Series, score: pd.Series) -> float:
    pair = pd.DataFrame({"target": target, "score": score}).dropna()
    positives = int(pair["target"].eq(1).sum())
    negatives = int(pair["target"].eq(0).sum())
    if not positives or not negatives:
        return float("nan")
    ranks = pair["score"].rank(method="average")
    return float((ranks[pair["target"].eq(1)].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def empirical_percentile(reference: pd.Series, values: pd.Series) -> pd.Series:
    reference = np.sort(pd.to_numeric(reference, errors="coerce").dropna().to_numpy(dtype=float))
    output = pd.Series(np.nan, index=values.index, dtype=float)
    valid = pd.to_numeric(values, errors="coerce").notna()
    if not len(reference):
        return output
    observed = pd.to_numeric(values.loc[valid], errors="coerce").to_numpy(dtype=float)
    left = np.searchsorted(reference, observed, side="left")
    right = np.searchsorted(reference, observed, side="right")
    output.loc[valid] = (left + right) / (2 * len(reference))
    return output


def exhaustion_score(long_impact: pd.Series, recent_change: pd.Series, direction: str) -> pd.Series:
    acceleration = recent_change - long_impact / 4
    if direction == "ROTATION_OUT":
        left, right = long_impact.clip(lower=0), (-acceleration).clip(lower=0)
    elif direction == "ROTATION_IN":
        left, right = (-long_impact).clip(lower=0), acceleration.clip(lower=0)
    else:
        raise ValueError(direction)
    return np.sqrt(left * right)


def block_bootstrap_delta(
    rows: pd.DataFrame, candidate: str, baseline: str, *, samples: int, seed: int
) -> dict[str, float | int | None]:
    point = auc_score(rows["target"], rows[candidate]) - auc_score(rows["target"], rows[baseline])
    dates = pd.DatetimeIndex(rows["date"].drop_duplicates().sort_values())
    groups = [rows.index[rows["date"].eq(date)].to_numpy() for date in dates]
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(samples):
        chosen: list[np.ndarray] = []
        while len(chosen) < len(groups):
            start = int(rng.integers(0, max(1, len(groups) - 39)))
            chosen.extend(groups[start : start + 40])
        sample = rows.loc[np.concatenate(chosen[: len(groups)])]
        delta = auc_score(sample["target"], sample[candidate]) - auc_score(sample["target"], sample[baseline])
        if np.isfinite(delta):
            deltas.append(delta)
    if not deltas:
        return {"delta_auc": point, "ci_low": None, "ci_high": None, "samples": 0}
    low, high = np.quantile(deltas, [0.025, 0.975])
    return {"delta_auc": point, "ci_low": float(low), "ci_high": float(high), "samples": len(deltas)}


def prepare_frame() -> tuple[pd.DataFrame, tuple[str, ...], dict]:
    from sector_turning_pooled_shadow import build_pooled_daily_frame
    from sector_turning_sector_macro_boost import DRIVERS, attach_sector_features, build_factor_features
    from turning_macro_shadow_experiment import merge_macro_next_session

    pooled, audit, _ = build_pooled_daily_frame(PACKAGE, None, horizon=10)
    selected = pd.read_csv(SECTOR_SOURCE, usecols=["sector"])["sector"].astype(str).str.zfill(4).unique()
    pooled = pooled.loc[pooled["sector"].astype(str).isin(selected)].copy()
    factors = build_factor_features(MACRO_CSV, DATABASE)
    enriched = attach_sector_features(pooled, factors)
    shock_20 = pd.DataFrame(index=factors.index)
    for driver in DRIVERS:
        shock_20[f"{driver}_shock_20"] = factors[f"{driver}_shock_1"].rolling(20, min_periods=20).sum()
    aligned = merge_macro_next_session(
        pd.DatetimeIndex(enriched["date"].drop_duplicates().sort_values()),
        shock_20,
        shock_20.columns,
        asof_column="trend_asof_date",
    ).reset_index()
    enriched = enriched.merge(aligned, on="date", how="left", validate="many_to_one")
    for driver in DRIVERS:
        beta = enriched[f"{driver}_beta_126"]
        enriched[f"{driver}_impact_20"] = beta * enriched[f"{driver}_shock_20"]
        enriched[f"{driver}_recent_impact_5"] = beta * enriched[f"{driver}_shock_5"]
    audit.update(
        {
            "selected_groups": int(len(selected)),
            "factor_first_date": str(factors.dropna().index.min().date()),
            "factor_last_date": str(factors.dropna().index.max().date()),
            "factor_same_day_use": False,
        }
    )
    return enriched, tuple(DRIVERS), audit


def strategy_rows(frame: pd.DataFrame, drivers: tuple[str, ...], direction: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sector_turning_pooled_shadow import _direction_frame
    from sector_turning_tradable_label_shadow import qualify_labels

    qualified, label_audit = qualify_labels(frame, direction)
    rows = _direction_frame(qualified, direction, target_mode="PRE_PIVOT")
    rows = rows.loc[rows["at_risk"] & rows["target"].notna()].copy()
    rows["price_raw"] = exhaustion_score(rows["relative_return_20"], rows["relative_acceleration_5"], direction)
    for driver in drivers:
        rows[f"{driver}_raw"] = exhaustion_score(
            rows[f"{driver}_impact_20"], rows[f"{driver}_recent_impact_5"], direction
        )
    return rows, pd.DataFrame([label_audit])


def evaluate_direction(rows: pd.DataFrame, drivers: tuple[str, ...], direction: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    from sector_turning_pooled_shadow import _early_event_metrics, _wilson_interval

    pair_records: list[dict] = []
    calibrated_parts: list[pd.DataFrame] = []
    for sector, group in rows.groupby("sector", sort=True):
        train = group.loc[group["date"].between(TRAIN_START, TRAIN_END)].copy()
        test = group.loc[group["date"].ge(TEST_START)].copy()
        if train.empty or test.empty:
            continue
        driver_rows = []
        for driver in drivers:
            train_auc = auc_score(train["target"], train[f"{driver}_raw"])
            test_auc = auc_score(test["target"], test[f"{driver}_raw"])
            driver_rows.append((driver, train_auc, test_auc))
        selected_driver = max(driver_rows, key=lambda item: (-np.inf if not np.isfinite(item[1]) else item[1]))[0]
        for driver, train_auc, test_auc in driver_rows:
            pair_records.append(
                {
                    "direction": direction,
                    "sector": str(sector),
                    "sector_name": str(group["sector_name"].iloc[0]),
                    "driver": driver,
                    "train_auc": train_auc,
                    "test_auc": test_auc,
                    "selected_on_train": driver == selected_driver,
                    "stable_auc_55": bool(train_auc >= 0.55 and test_auc >= 0.55),
                    "train_rows": int(len(train)),
                    "test_rows": int(len(test)),
                    "train_positives": int(train["target"].sum()),
                    "test_positives": int(test["target"].sum()),
                }
            )

        selected = test[
            [
                "date", "_position", "sector", "sector_name", "target",
                "target_event_pivot_position", "target_event_confirmation_position",
                "target_event_confirmation_date",
            ]
        ].copy()
        price_train = empirical_percentile(train["price_raw"], train["price_raw"])
        macro_train = empirical_percentile(train[f"{selected_driver}_raw"], train[f"{selected_driver}_raw"])
        selected["price_score"] = empirical_percentile(train["price_raw"], test["price_raw"])
        selected["macro_score"] = empirical_percentile(
            train[f"{selected_driver}_raw"], test[f"{selected_driver}_raw"]
        )
        combined_train = (price_train + macro_train) / 2
        selected["combined_score"] = empirical_percentile(
            combined_train, (selected["price_score"] + selected["macro_score"]) / 2
        )
        selected["selected_driver"] = selected_driver
        calibrated_parts.append(selected)

    pairs = pd.DataFrame(pair_records)
    calibrated = pd.concat(calibrated_parts, ignore_index=True).dropna(
        subset=["price_score", "macro_score", "combined_score"]
    )
    strategies = {}
    for name, column in (
        ("PRICE_EXHAUSTION", "price_score"),
        ("MACRO_TREND_HINT", "macro_score"),
        ("PRICE_PLUS_MACRO", "combined_score"),
    ):
        scored = calibrated.copy()
        scored["score"] = scored[column]
        event = _early_event_metrics(scored, pd.DataFrame(), direction, horizon=10, cutoff=ALERT_CUTOFF)
        low, high = _wilson_interval(event["pre_pivot_hits"], event["eligible_confirmations"])
        strategies[name] = {
            "auc": auc_score(scored["target"], scored["score"]),
            "rows": int(len(scored)),
            "positives": int(scored["target"].sum()),
            "pre_pivot_recall_ci_low": low,
            "pre_pivot_recall_ci_high": high,
            **event,
        }
    summary = {
        "direction": direction,
        "test_first_date": str(calibrated["date"].min().date()),
        "test_last_matured_date": str(calibrated["date"].max().date()),
        "selected_driver_counts": calibrated[["sector", "selected_driver"]]
        .drop_duplicates()["selected_driver"]
        .value_counts()
        .to_dict(),
        "stable_pairs_55": int(pairs["stable_auc_55"].sum()),
        "selected_sector_test_auc_median": float(
            pairs.loc[pairs["selected_on_train"], "test_auc"].median()
        ),
        "selected_sectors_above_50": int(
            pairs.loc[pairs["selected_on_train"], "test_auc"].gt(0.50).sum()
        ),
        "selected_sectors_above_55": int(
            pairs.loc[pairs["selected_on_train"], "test_auc"].ge(0.55).sum()
        ),
        "strategies": strategies,
        "auc_delta_vs_price": {
            "MACRO_TREND_HINT": block_bootstrap_delta(
                calibrated, "macro_score", "price_score", samples=BOOTSTRAP_SAMPLES, seed=17
            ),
            "PRICE_PLUS_MACRO": block_bootstrap_delta(
                calibrated, "combined_score", "price_score", samples=BOOTSTRAP_SAMPLES, seed=23
            ),
        },
    }
    return pairs, calibrated, summary


def render_matrix(output_dir: Path) -> None:
    from PIL import Image, ImageDraw
    from reports.build_macro_sector_correlation_matrix import font, color_for

    pairs = pd.read_csv(output_dir / "pair_metrics.csv")
    pairs["sector"] = pairs["sector"].astype(str).str.zfill(4)
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    drivers = summary["spec"]["drivers"]
    driver_labels = {
        "curve_2s10s": "2s10s\n기울기", "real_rate_10y": "10Y\n실질금리",
        "breakeven_10y": "10Y\n기대인플레", "wti_spot": "WTI\n유가",
        "brent_wti_spread": "Brent-WTI\n스프레드", "yen_strength": "엔 강세",
        "vix": "VIX", "move": "MOVE",
    }
    sectors = pairs[["sector", "sector_name"]].drop_duplicates().sort_values("sector")
    width, height = 2820, 1540
    image = Image.new("RGB", (width, height), (250, 250, 247))
    draw = ImageDraw.Draw(image)
    ink, muted, grid = (28, 35, 38), (96, 105, 105), (215, 219, 216)
    draw.text((48, 35), "Macro trend exhaustion → Sector turn", font=font(40, True), fill=ink)
    draw.text(
        (48, 91),
        "과거 20일 매크로 영향이 강했지만 최근 5일 영향이 반대로 감속할 때 · 검증구간 AUC (0.50=무작위)",
        font=font(22), fill=muted,
    )
    cell_w, cell_h, left_x, gap, top_y = 132, 48, 520, 92, 330
    right_x = left_x + cell_w * len(drivers) + gap
    panels = (("ROTATION_OUT", "상승 종료 · ROTATION OUT"), ("ROTATION_IN", "하락 종료 · ROTATION IN"))
    limit = 0.20
    for panel_x, (direction, title) in zip((left_x, right_x), panels, strict=True):
        draw.text((panel_x, 170), title, font=font(27, True), fill=ink)
        selected = pairs.loc[pairs["direction"].eq(direction)].set_index(["sector", "driver"])
        for col, driver in enumerate(drivers):
            x = panel_x + col * cell_w
            label = driver_labels[driver]
            box = draw.multiline_textbbox((0, 0), label, font=font(19, True), spacing=3, align="center")
            draw.multiline_text((x + (cell_w - box[2] + box[0]) / 2, 240), label, font=font(19, True), fill=ink, spacing=3, align="center")
        for row, sector_row in enumerate(sectors.itertuples(index=False)):
            sector = str(sector_row.sector).zfill(4)
            y = top_y + row * cell_h
            if panel_x == left_x:
                draw.text((48, y + 11), f"{sector}  {sector_row.sector_name}", font=font(18), fill=ink)
            for col, driver in enumerate(drivers):
                record = selected.loc[(sector, driver), :]
                value = float(record["test_auc"])
                x = panel_x + col * cell_w
                fill = color_for(value - 0.5, limit)
                outline = ink if bool(record["stable_auc_55"]) else grid
                draw.rectangle((x, y, x + cell_w, y + cell_h), fill=fill, outline=outline, width=3 if bool(record["stable_auc_55"]) else 1)
                chosen = bool(record["selected_on_train"])
                text = f"{value:.2f}"
                box = draw.textbbox((0, 0), text, font=font(17, chosen))
                draw.text((x + (cell_w - box[2] + box[0]) / 2, y + 12), text, font=font(17, chosen), fill=ink)

    out = summary["directions"]["ROTATION_OUT"]
    inn = summary["directions"]["ROTATION_IN"]
    draw.text((520, 1380), "파랑 < 0.50  ·  흰색 = 0.50  ·  빨강 > 0.50", font=font(18), fill=muted)
    draw.text((520, 1415), "굵은 숫자: 2018~2022 학습구간에서 선택 · 검은 테두리: 학습/검증 모두 AUC≥0.55", font=font(18), fill=muted)
    draw.text(
        (1380, 1372),
        f"OUT  선택 매크로 AUC {out['strategies']['MACRO_TREND_HINT']['auc']:.3f} · 결합 {out['strategies']['PRICE_PLUS_MACRO']['auc']:.3f}",
        font=font(19, True), fill=ink,
    )
    draw.text(
        (1380, 1410),
        f"IN     선택 매크로 AUC {inn['strategies']['MACRO_TREND_HINT']['auc']:.3f} · 결합 {inn['strategies']['PRICE_PLUS_MACRO']['auc']:.3f}",
        font=font(19, True), fill=ink,
    )
    draw.text((48, 1490), "RESEARCH SHADOW ONLY · PIT_PARTIAL / CURRENT_REVISED_HISTORY_ONLY", font=font(16, True), fill=muted)
    image.save(output_dir / "macro_trend_hint_auc_matrix.png", optimize=True)


def self_test() -> None:
    target = pd.Series([0, 0, 1, 1])
    assert abs(auc_score(target, pd.Series([0.1, 0.2, 0.8, 0.9])) - 1) < 1e-12
    out = exhaustion_score(pd.Series([1.0]), pd.Series([0.0]), "ROTATION_OUT")
    inn = exhaustion_score(pd.Series([-1.0]), pd.Series([0.0]), "ROTATION_IN")
    assert out.iloc[0] > 0 and inn.iloc[0] > 0
    print("self-test: PASS")


def run() -> dict:
    frame, drivers, audit = prepare_frame()
    all_pairs, all_rows, directions = [], [], {}
    for direction in ("ROTATION_OUT", "ROTATION_IN"):
        rows, label_audit = strategy_rows(frame, drivers, direction)
        pairs, calibrated, result = evaluate_direction(rows, drivers, direction)
        result["label_audit"] = label_audit.iloc[0].to_dict()
        all_pairs.append(pairs)
        all_rows.append(calibrated)
        directions[direction] = result
    summary = {
        "verdict": "RESEARCH_SHADOW_ONLY",
        "pit_status": "PIT_PARTIAL/CURRENT_REVISED_HISTORY_ONLY",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "spec": {
            "train": f"{TRAIN_START.date()}..{TRAIN_END.date()}",
            "test": f"{TEST_START.date()}..latest matured label",
            "drivers": list(drivers),
            "macro_rule": "positive 20-session beta impact with negative 5-vs-20 acceleration for OUT; symmetric inverse for IN",
            "sector_beta": "126-session rolling beta shifted one session",
            "macro_timing": "strictly prior source date; no fill/interpolation",
            "label": "directional-change pivot followed by >=3% relative follow-through over 20 sessions",
            "bootstrap": f"{BOOTSTRAP_SAMPLES} resamples of 40-date blocks",
        },
        "input_audit": audit,
        "directions": directions,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(all_pairs, ignore_index=True).to_csv(OUTPUT_DIR / "pair_metrics.csv", index=False)
    pd.concat(all_rows, ignore_index=True).to_csv(OUTPUT_DIR / "test_scores.csv", index=False)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.render_only:
        render_matrix(OUTPUT_DIR)
    else:
        print(json.dumps(run(), ensure_ascii=False))


if __name__ == "__main__":
    main()
