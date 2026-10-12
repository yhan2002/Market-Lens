"""Build research-only macro/sector correlation matrices from existing data."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
MACRO_CSV = (
    ROOT
    / "outputs"
    / "bloomberg_turning_macro_staging_20260902_212424"
    / "bloomberg_turning_macro_20260902_212424"
    / "macro_history.csv"
)
GICS_DIR = (
    ROOT
    / "outputs"
    / "gics_l2_staging_20260902_145743"
    / "bloomberg_gics_l2_20260902_145743"
)
SECTOR_SOURCE = ROOT / "outputs" / "sector_turning_recall_breadth_boost_20260903" / "predictions.csv"
DATABASE = ROOT / "regime_router_publish.db"
OUTPUT_DIR = ROOT / "output" / "analysis" / "macro_sector_correlation_20260903"

DRIVERS = (
    "curve_2s10s",
    "real_rate_10y",
    "breakeven_10y",
    "wti_spot",
    "brent_wti_spread",
    "yen_strength",
    "vix",
    "move",
)
DRIVER_LABELS = {
    "curve_2s10s": "2s10s\n기울기",
    "real_rate_10y": "10Y\n실질금리",
    "breakeven_10y": "10Y\n기대인플레",
    "wti_spot": "WTI\n유가",
    "brent_wti_spread": "Brent-WTI\n스프레드",
    "yen_strength": "엔 강세",
    "vix": "VIX",
    "move": "MOVE",
}


def load_factor_levels() -> pd.DataFrame:
    macro = pd.read_csv(MACRO_CSV, parse_dates=["date"])
    expected = {"UST_2Y", "UST_10Y", "UST_REAL_10Y", "BREAKEVEN_10Y", "WTI_SPOT", "BRENT_SPOT"}
    if set(macro["series"]) != expected or macro.duplicated(["series", "date"]).any():
        raise ValueError("unexpected or duplicated Bloomberg macro history")
    wide = macro.pivot(index="date", columns="series", values="value").sort_index()

    uri = DATABASE.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.execute("PRAGMA query_only=ON")
        market = pd.read_sql_query(
            """
            SELECT date, ticker, COALESCE(adj_close, close) AS value
            FROM daily_prices
            WHERE ticker IN ('JPY=X', '^VIX', '^MOVE')
            ORDER BY date, ticker
            """,
            connection,
            parse_dates=["date"],
        ).pivot(index="date", columns="ticker", values="value")

    levels = pd.DataFrame(
        {
            "curve_2s10s": wide["UST_10Y"] - wide["UST_2Y"],
            "real_rate_10y": wide["UST_REAL_10Y"],
            "breakeven_10y": wide["BREAKEVEN_10Y"],
            "wti_spot": wide["WTI_SPOT"],
            "brent_wti_spread": wide["BRENT_SPOT"] - wide["WTI_SPOT"],
        }
    ).join(market.rename(columns={"JPY=X": "usdjpy", "^VIX": "vix", "^MOVE": "move"}), how="inner")
    levels["yen_strength"] = -np.log(levels.pop("usdjpy"))
    levels["vix"] = np.log(levels["vix"])
    levels["move"] = np.log(levels["move"])
    return levels.loc[:, DRIVERS].replace([np.inf, -np.inf], np.nan).dropna().sort_index()


def align_next_session(calendar: pd.DatetimeIndex, shocks: pd.DataFrame) -> pd.DataFrame:
    left = pd.DataFrame({"date": calendar, "_order": np.arange(len(calendar))})
    right = shocks.reset_index(names="macro_asof_date")
    result = pd.merge_asof(
        left.sort_values("date"),
        right.sort_values("macro_asof_date"),
        left_on="date",
        right_on="macro_asof_date",
        direction="backward",
        allow_exact_matches=False,
        tolerance=pd.Timedelta(days=7),
    )
    return result.sort_values("_order").drop(columns="_order").set_index("date")


def spearman_pair(x: pd.Series, y: pd.Series) -> tuple[float, int]:
    pair = pd.concat([x, y], axis=1).dropna()
    if len(pair) < 3:
        return np.nan, len(pair)
    return float(pair.iloc[:, 0].rank().corr(pair.iloc[:, 1].rank())), len(pair)


def build_matrices() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    sectors = pd.read_csv(SECTOR_SOURCE, usecols=["sector", "sector_name"])
    sectors["sector"] = sectors["sector"].astype(str).str.zfill(4)
    sectors = sectors.drop_duplicates().sort_values("sector")

    prices = pd.read_csv(GICS_DIR / "industry_group_total_return.csv", parse_dates=["date"])
    prices["sector"] = prices["industry_group_code"].astype(str).str.zfill(4)
    prices = prices.loc[prices["sector"].isin(sectors["sector"])]
    sector_levels = prices.pivot(index="date", columns="sector", values="total_return_level").sort_index()
    benchmark = (
        pd.read_csv(GICS_DIR / "benchmark_total_return.csv", parse_dates=["date"])
        .set_index("date")["total_return_level"]
        .sort_index()
    )
    calendar = pd.DatetimeIndex(benchmark.index.drop_duplicates().sort_values())
    relative = sector_levels.reindex(calendar).div(benchmark.reindex(calendar), axis=0)
    relative_5 = relative.pct_change(5, fill_method=None)
    forward_20 = relative.shift(-20).div(relative).sub(1)

    shocks = load_factor_levels().diff(5).dropna()
    aligned = align_next_session(calendar, shocks)
    weekly_dates = pd.Series(calendar, index=calendar).groupby(calendar.to_period("W-FRI")).max()
    weekly_dates = pd.DatetimeIndex(weekly_dates)
    aligned = aligned.reindex(weekly_dates)
    relative_5 = relative_5.reindex(weekly_dates)
    forward_20 = forward_20.reindex(weekly_dates)

    concurrent = pd.DataFrame(index=sectors["sector"], columns=DRIVERS, dtype=float)
    leading = concurrent.copy()
    counts = concurrent.copy()
    for sector in sectors["sector"]:
        for driver in DRIVERS:
            concurrent.loc[sector, driver], _ = spearman_pair(aligned[driver], relative_5[sector])
            leading.loc[sector, driver], counts.loc[sector, driver] = spearman_pair(aligned[driver], forward_20[sector])

    label_map = sectors.set_index("sector")["sector_name"].to_dict()
    concurrent.index = [f"{code}  {label_map[code]}" for code in concurrent.index]
    leading.index = concurrent.index
    counts.index = concurrent.index
    flat = (
        leading.stack(future_stack=True)
        .rename("correlation")
        .rename_axis(["sector", "driver"])
        .reset_index()
    )
    flat["absolute_correlation"] = flat["correlation"].abs()
    top = flat.nlargest(12, "absolute_correlation").to_dict("records")
    valid_dates = aligned.dropna().index
    metadata = {
        "status": "RESEARCH_SHADOW_ONLY",
        "pit_status": "CURRENT_REVISED_HISTORY_ONLY / PIT_PARTIAL",
        "method": "Spearman correlation on weekly observations; macro 5-session shock is available from the next market session",
        "targets": {
            "concurrent": "same-date trailing 5-session GICS group excess total return vs S&P 500",
            "leading": "next 20-session GICS group excess total return vs S&P 500",
        },
        "first_week": str(valid_dates.min().date()),
        "last_week": str(valid_dates.max().date()),
        "sector_count": int(len(sectors)),
        "driver_count": len(DRIVERS),
        "leading_sample_min": int(np.nanmin(counts.to_numpy())),
        "leading_sample_max": int(np.nanmax(counts.to_numpy())),
        "top_leading_correlations": top,
    }
    return concurrent, leading, counts, metadata


def color_for(value: float, limit: float) -> tuple[int, int, int]:
    base = np.array((246, 246, 241), dtype=float)
    target = np.array((180, 55, 48) if value >= 0 else (34, 101, 152), dtype=float)
    weight = min(abs(value) / limit, 1.0)
    return tuple(np.round(base * (1 - weight) + target * weight).astype(int))


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "malgunbd.ttf" if bold else "malgun.ttf"
    return ImageFont.truetype(str(Path("C:/Windows/Fonts") / name), size)


def render(concurrent: pd.DataFrame, leading: pd.DataFrame, metadata: dict, output: Path) -> None:
    width, height = 2820, 1530
    image = Image.new("RGB", (width, height), (250, 250, 247))
    draw = ImageDraw.Draw(image)
    ink, muted, grid = (28, 35, 38), (96, 105, 105), (215, 219, 216)
    draw.text((48, 35), "Macro × Sector 상관관계", font=font(42, True), fill=ink)
    draw.text(
        (48, 92),
        "최근 5거래일 매크로 변화와 GICS 산업그룹의 S&P 500 대비 총수익 · 주간 관측 · Spearman 상관계수",
        font=font(23),
        fill=muted,
    )

    label_width, cell_w, cell_h = 520, 132, 48
    left_x, gap, top_y = 520, 92, 330
    right_x = left_x + cell_w * len(DRIVERS) + gap
    panel_titles = (
        "동행 관계 · 같은 5거래일 상대수익",
        "선행 관계 · 이후 20거래일 상대수익",
    )
    matrices = (concurrent, leading)
    limit = max(0.15, float(np.nanmax(np.abs(np.concatenate([concurrent.to_numpy(), leading.to_numpy()])))))

    for panel_x, title, matrix in zip((left_x, right_x), panel_titles, matrices, strict=True):
        draw.text((panel_x, 170), title, font=font(27, True), fill=ink)
        for col, driver in enumerate(DRIVERS):
            x = panel_x + col * cell_w
            label = DRIVER_LABELS[driver]
            box = draw.multiline_textbbox((0, 0), label, font=font(19, True), spacing=3, align="center")
            draw.multiline_text(
                (x + (cell_w - (box[2] - box[0])) / 2, 240),
                label,
                font=font(19, True),
                fill=ink,
                spacing=3,
                align="center",
            )
        for row, (sector, values) in enumerate(matrix.iterrows()):
            y = top_y + row * cell_h
            if panel_x == left_x:
                draw.text((48, y + 11), sector, font=font(18), fill=ink)
            for col, value in enumerate(values):
                x = panel_x + col * cell_w
                fill = color_for(float(value), limit)
                draw.rectangle((x, y, x + cell_w, y + cell_h), fill=fill, outline=grid, width=1)
                text_fill = (255, 255, 255) if max(fill) - min(fill) > 45 and sum(fill) < 460 else ink
                label = f"{float(value):+.2f}"
                box = draw.textbbox((0, 0), label, font=font(17, abs(value) >= 0.15))
                draw.text(
                    (x + (cell_w - (box[2] - box[0])) / 2, y + 12),
                    label,
                    font=font(17, abs(value) >= 0.15),
                    fill=text_fill,
                )

    bar_x, bar_y, bar_w, bar_h = 520, 1380, 720, 22
    for offset in range(bar_w):
        value = -limit + 2 * limit * offset / (bar_w - 1)
        draw.line((bar_x + offset, bar_y, bar_x + offset, bar_y + bar_h), fill=color_for(value, limit))
    draw.text((bar_x, bar_y + 31), f"-{limit:.2f}  반대 방향", font=font(17), fill=muted)
    draw.text((bar_x + bar_w - 116, bar_y + 31), f"같은 방향  +{limit:.2f}", font=font(17), fill=muted, anchor="ra")
    draw.text(
        (1360, 1367),
        f"기간 {metadata['first_week']}~{metadata['last_week']} · 선행 표본 {metadata['leading_sample_min']}~{metadata['leading_sample_max']}주",
        font=font(18),
        fill=muted,
    )
    draw.text(
        (1360, 1410),
        "매크로 종가는 동일일에 쓰지 않고 다음 시장 거래일부터 사용 · 보간 없음 · 현재 수정 이력 기반(PIT_PARTIAL)",
        font=font(18),
        fill=muted,
    )
    draw.text((48, 1480), "RESEARCH SHADOW ONLY", font=font(16, True), fill=muted)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, optimize=True)


def self_test() -> None:
    x = pd.Series([1.0, 2.0, 3.0, 4.0])
    correlation, count = spearman_pair(x, x[::-1].reset_index(drop=True))
    assert count == 4 and abs(correlation + 1) < 1e-12
    assert color_for(-0.2, 0.4) != color_for(0.2, 0.4)
    print("self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    concurrent, leading, counts, metadata = build_matrices()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    concurrent.to_csv(OUTPUT_DIR / "concurrent_5d_spearman.csv", encoding="utf-8-sig")
    leading.to_csv(OUTPUT_DIR / "forward_20d_spearman.csv", encoding="utf-8-sig")
    counts.to_csv(OUTPUT_DIR / "forward_20d_sample_counts.csv", encoding="utf-8-sig")
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    render(concurrent, leading, metadata, OUTPUT_DIR / "macro_sector_correlation_matrix.png")
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
