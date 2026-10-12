"""Compare current and attention-adjusted sector turning alerts by sector."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from build_sector_turning_alert_backtest_pdf import (
    GRID,
    GREEN,
    INK,
    MID_GREY,
    PAPER,
    RED,
    TEAL,
    actual_events,
    draw_footer,
    draw_text,
    draw_triangle,
    register_fonts,
)
GAUGE_PATH = (
    ROOT / "outputs" / "sector_turning_attention_compare_20260903" / "gauge_history.csv"
)
GICS_DIR = (
    ROOT
    / "outputs"
    / "gics_l2_staging_20260902_145743"
    / "bloomberg_gics_l2_20260902_145743"
)
DEFAULT_OUTPUT = (
    ROOT / "output" / "pdf" / "sector_turning_attention_compare_20260629_20260831.pdf"
)
START = pd.Timestamp("2026-06-29")
CURRENT_CUTOFF = 0.70
ADAPTIVE_CUTOFF = 0.90
HORIZON = 10
BLUE = colors.HexColor("#1479B8")
AMBER = colors.HexColor("#D29A2E")
LIGHT_GREY = colors.HexColor("#EEF1EF")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    gauges = pd.read_csv(
        GAUGE_PATH,
        parse_dates=["date", "target_event_confirmation_date"],
        low_memory=False,
    )
    gauges["sector"] = gauges["sector"].astype(str).str.zfill(4)
    end = pd.Timestamp(gauges["date"].max())
    gauges = gauges.loc[gauges["date"].between(START, end)].copy()
    if gauges.empty or gauges["proxy_status"].ne("DYNAMIC").any():
        raise ValueError("comparison window requires complete dynamic attention inputs")

    prices = pd.read_csv(
        GICS_DIR / "industry_group_total_return.csv", parse_dates=["date"]
    )
    prices["sector"] = prices["industry_group_code"].astype(str).str.zfill(4)
    prices = prices.loc[
        prices["date"].between(START, end),
        ["date", "sector", "industry_group_name", "total_return_level"],
    ].copy()
    prices["total_return_level"] = pd.to_numeric(
        prices["total_return_level"], errors="raise"
    )
    calendar = (
        pd.read_csv(GICS_DIR / "benchmark_total_return.csv", parse_dates=["date"])["date"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )
    missing_prices = set(gauges["sector"].unique()) - set(prices["sector"].unique())
    if missing_prices:
        raise ValueError(f"missing comparison price paths: {sorted(missing_prices)}")
    return gauges, prices, calendar


def signal_mask(rows: pd.DataFrame, strategy: str) -> pd.Series:
    if strategy == "CURRENT_70":
        return rows["score"].ge(CURRENT_CUTOFF)
    if strategy == "ADAPTIVE":
        return rows["score"].ge(rows["dynamic_threshold"])
    raise ValueError(strategy)


def alert_clusters(rows: pd.DataFrame, strategy: str) -> list[tuple[int, int]]:
    positions = sorted(rows.loc[signal_mask(rows, strategy), "_position"].astype(int).unique())
    groups: list[list[int]] = []
    for position in positions:
        if not groups or position - groups[-1][-1] > HORIZON:
            groups.append([position])
        else:
            groups[-1].append(position)
    return [(group[0], group[-1]) for group in groups]


def direction_metrics(rows: pd.DataFrame, strategy: str) -> dict[str, int]:
    events = (
        rows.loc[rows["target_event_confirmation_position"].notna()]
        .groupby(["sector", "target_event_confirmation_position"], sort=True)
        .agg(pivot=("target_event_pivot_position", "first"))
        .reset_index()
    )
    clusters = matched = hits = 0
    for sector, sector_rows in rows.groupby("sector", sort=True):
        selected = sorted(
            sector_rows.loc[signal_mask(sector_rows, strategy), "_position"]
            .astype(int)
            .unique()
        )
        sector_clusters: list[list[int]] = []
        for position in selected:
            if not sector_clusters or position - sector_clusters[-1][-1] > HORIZON:
                sector_clusters.append([position])
            else:
                sector_clusters[-1].append(position)
        clusters += len(sector_clusters)
        available = events.index[events["sector"].eq(sector)].tolist()
        available.sort(key=lambda index: events.loc[index, "target_event_confirmation_position"])
        for positions in sector_clusters:
            match = next(
                (
                    index
                    for index in available
                    if any(
                        1 <= int(events.loc[index, "target_event_confirmation_position"]) - position <= HORIZON
                        for position in positions
                    )
                ),
                None,
            )
            if match is None:
                continue
            available.remove(match)
            matched += 1
            confirmation = int(events.loc[match, "target_event_confirmation_position"])
            pivot = int(events.loc[match, "pivot"])
            if any(1 <= confirmation - position <= HORIZON and position < pivot for position in positions):
                hits += 1
    return {"eligible": len(events), "hits": hits, "clusters": clusters, "matched_clusters": matched}


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    z = 1.959963984540054
    estimate = successes / total
    denominator = 1 + z * z / total
    centre = (estimate + z * z / (2 * total)) / denominator
    radius = z / denominator * math.sqrt(estimate * (1 - estimate) / total + z * z / (4 * total * total))
    return max(0.0, centre - radius), min(1.0, centre + radius)


def strategy_metrics(rows: pd.DataFrame, strategy: str) -> dict[str, Any]:
    totals = {
        "eligible": 0,
        "hits": 0,
        "event_hits": 0,
        "clusters": 0,
        "matched_clusters": 0,
        "signal_days": 0,
        "scored_days": 0,
    }
    for direction in ("ROTATION_OUT", "ROTATION_IN"):
        frame = rows.loc[rows["direction"].eq(direction) & rows["target"].notna()].copy()
        metric = direction_metrics(frame, strategy)
        totals["eligible"] += metric["eligible"]
        totals["hits"] += metric["hits"]
        totals["event_hits"] += metric["matched_clusters"]
        totals["clusters"] += metric["clusters"]
        totals["matched_clusters"] += metric["matched_clusters"]
        totals["signal_days"] += int(signal_mask(frame, strategy).sum())
        totals["scored_days"] += len(frame)
    totals["recall"] = totals["hits"] / totals["eligible"] if totals["eligible"] else 0.0
    totals["event_recall"] = (
        totals["event_hits"] / totals["eligible"] if totals["eligible"] else 0.0
    )
    totals["signal_day_rate"] = (
        totals["signal_days"] / totals["scored_days"] if totals["scored_days"] else 0.0
    )
    totals["recall_ci"] = wilson_interval(totals["hits"], totals["eligible"])
    return totals


def date_x(date: pd.Timestamp, start: pd.Timestamp, end: pd.Timestamp, x: float, width: float) -> float:
    span = max((end - start).days, 1)
    return x + ((pd.Timestamp(date) - start).days / span) * width


def position_date(calendar: pd.Series, position: int) -> pd.Timestamp:
    if position < 0 or position >= len(calendar):
        raise ValueError(f"calendar position out of range: {position}")
    return pd.Timestamp(calendar.iloc[position])


def draw_metric_card(
    pdf: canvas.Canvas, x: float, y: float, width: float, title: str, metric: dict[str, Any], accent
) -> None:
    pdf.setFillColor(colors.HexColor("#F4F7F5"))
    pdf.roundRect(x, y, width, 102, 7, fill=1, stroke=0)
    pdf.setFillColor(accent)
    pdf.roundRect(x, y, 7, 102, 7, fill=1, stroke=0)
    draw_text(pdf, title, x + 20, y + 77, font="Malgun-Bold", size=12)
    draw_text(
        pdf,
        f"사전포착 {metric['hits']}/{metric['eligible']}  ({metric['recall']:.1%})",
        x + 20,
        y + 47,
        font="Malgun-Bold",
        size=16,
        color=accent,
    )
    low, high = metric["recall_ci"]
    draw_text(
        pdf,
        f"95% CI {low:.1%}~{high:.1%}  |  신호일 {metric['signal_day_rate']:.1%}",
        x + 20,
        y + 18,
        size=8,
        color=MID_GREY,
    )


def draw_cover(
    pdf: canvas.Canvas,
    sectors: pd.DataFrame,
    current: dict[str, Any],
    adaptive: dict[str, Any],
    end: pd.Timestamp,
    pages: int,
) -> None:
    width, height = landscape(A4)
    pdf.setFillColor(TEAL)
    pdf.rect(0, height - 98, width, 98, fill=1, stroke=0)
    draw_text(
        pdf,
        "섹터 추세 전환 경보 비교",
        42,
        height - 50,
        font="Malgun-Bold",
        size=24,
        color=PAPER,
    )
    draw_text(
        pdf,
        "기존 고정 경고선과 관심도 조정 경고선을 동일 기간·동일 점수로 비교",
        43,
        height - 76,
        size=10.5,
        color=colors.HexColor("#DCEBE6"),
    )
    improved = adaptive["hits"] > current["hits"]
    verdict = "포착 성과 개선" if improved else "포착 성과 개선 없음"
    badge = GREEN if improved else RED
    pdf.setFillColor(badge)
    pdf.roundRect(width - 200, height - 69, 158, 31, 5, fill=1, stroke=0)
    draw_text(pdf, verdict, width - 184, height - 59, font="Malgun-Bold", size=10, color=PAPER)

    draw_metric_card(pdf, 42, 370, 360, "현재 전략 · 고정 70점", current, BLUE)
    draw_metric_card(pdf, 430, 370, 360, "신규 전략 · 동적 85~95점", adaptive, AMBER)

    draw_text(pdf, "판정", 42, 338, font="Malgun-Bold", size=13)
    messages = [
        f"사전포착은 {current['hits']}건에서 {adaptive['hits']}건으로 감소해, 포착률 우선 기준에서는 개선되지 않았습니다.",
        f"신호일 비중은 {current['signal_day_rate']:.1%}에서 {adaptive['signal_day_rate']:.1%}로 감소해 선별성은 높아졌습니다.",
        f"표본은 확정 전환 {current['eligible']}건뿐이며 신뢰구간이 넓어, 신규 경고선으로 교체할 근거는 부족합니다.",
        "따라서 현재 전략은 유지하고, 관심도 임계치는 SHADOW 보조정보로만 관찰하는 것이 적절합니다.",
    ]
    y = 316
    for message in messages:
        draw_text(pdf, "• " + message, 48, y, size=9.3)
        y -= 20

    draw_text(pdf, "분석 범위와 읽는 법", 42, 218, font="Malgun-Bold", size=13)
    scope = [
        f"기간 {START.date()}~{end.date()} · {len(sectors)}개 GICS Industry Group · 10거래일 확인창",
        "성과는 수익률이 아니라 실제 고점·저점 이전 사전포착률입니다.",
        "각 섹터 페이지: 파란선=총수익지수, 삼각형=실제 전환점, 빨강/초록 막대=OUT/IN 신호일",
        "회색 영역은 아직 향후 10거래일 결과가 끝나지 않은 성과판정 대기 구간입니다.",
        "PIT_PARTIAL/CURRENT_REVISED_HISTORY_ONLY · RESEARCH SHADOW ONLY",
    ]
    y = 196
    for line in scope:
        draw_text(pdf, "• " + line, 48, y, size=9)
        y -= 18
    draw_footer(pdf, 1, pages)


def draw_track(
    pdf: canvas.Canvas,
    rows: pd.DataFrame,
    calendar: pd.Series,
    sector: str,
    strategy: str,
    label: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    x: float,
    width: float,
    y: float,
) -> None:
    draw_text(pdf, label, x, y + 20, font="Malgun-Bold", size=8)
    for direction, offset, color in (
        ("ROTATION_OUT", 4, RED),
        ("ROTATION_IN", -9, GREEN),
    ):
        pdf.setStrokeColor(GRID)
        pdf.line(x, y + offset + 3, x + width, y + offset + 3)
        draw_text(pdf, "OUT" if direction.endswith("OUT") else "IN", x - 29, y + offset, size=6.5, color=color)
        frame = rows.loc[rows["sector"].eq(sector) & rows["direction"].eq(direction)]
        for first, last in alert_clusters(frame, strategy):
            first_date = position_date(calendar, first)
            last_date = position_date(calendar, last)
            if last_date < start or first_date > end:
                continue
            left = date_x(max(first_date, start), start, end, x, width)
            right = date_x(min(last_date, end), start, end, x, width)
            pdf.setFillColor(color)
            pdf.rect(left, y + offset, max(3, right - left), 7, fill=1, stroke=0)


def draw_sector_page(
    pdf: canvas.Canvas,
    page: int,
    pages: int,
    sector: str,
    name: str,
    price_rows: pd.DataFrame,
    gauges: pd.DataFrame,
    events: pd.DataFrame,
    calendar: pd.Series,
    matured_end: pd.Timestamp,
    end: pd.Timestamp,
) -> None:
    width, height = landscape(A4)
    pdf.setFillColor(TEAL)
    pdf.rect(0, height - 58, width, 58, fill=1, stroke=0)
    draw_text(pdf, f"{sector}  {name}", 42, height - 37, font="Malgun-Bold", size=18, color=PAPER)
    sector_rows = gauges.loc[gauges["sector"].eq(sector)]
    current = strategy_metrics(sector_rows, "CURRENT_70")
    adaptive = strategy_metrics(sector_rows, "ADAPTIVE")
    delta = adaptive["hits"] - current["hits"]
    delta_text = f"포착 {delta:+d}건" if current["eligible"] else "확정 전환 없음"
    draw_text(
        pdf,
        f"현재 {current['hits']}/{current['eligible']} · 신규 {adaptive['hits']}/{adaptive['eligible']} · {delta_text} · 신호일 {current['signal_days']}→{adaptive['signal_days']}",
        58,
        height - 77,
        size=8.5,
        color=MID_GREY,
    )

    chart_x, chart_y, chart_width, chart_height = 58, 240, width - 100, 250
    series = price_rows.sort_values("date").drop_duplicates("date")
    series = series.loc[series["total_return_level"].notna()].copy()
    if series.empty:
        raise ValueError(f"no price data for {sector}")
    series["normalized"] = series["total_return_level"] / series["total_return_level"].iloc[0] * 100
    values = series["normalized"].to_numpy(dtype=float)
    low, high = float(values.min()), float(values.max())
    padding = max((high - low) * 0.12, 1.5)
    y_low, y_high = low - padding, high + padding

    def y_map(value: float) -> float:
        return chart_y + (value - y_low) / (y_high - y_low) * chart_height

    pdf.setFillColor(colors.HexColor("#FAFBFA"))
    pdf.setStrokeColor(GRID)
    pdf.rect(chart_x, chart_y, chart_width, chart_height, fill=1, stroke=1)
    pending_start = min(max(matured_end + pd.Timedelta(days=1), START), end)
    pending_x = date_x(pending_start, START, end, chart_x, chart_width)
    pdf.setFillColor(LIGHT_GREY)
    pdf.rect(pending_x, chart_y, chart_x + chart_width - pending_x, chart_height, fill=1, stroke=0)
    for tick in np.linspace(y_low, y_high, 4):
        y_tick = y_map(float(tick))
        pdf.setStrokeColor(GRID)
        pdf.line(chart_x, y_tick, chart_x + chart_width, y_tick)
        pdf.setFillColor(MID_GREY)
        pdf.setFont("Malgun", 7)
        pdf.drawRightString(chart_x - 7, y_tick - 2, f"{tick:.0f}")

    path = pdf.beginPath()
    for index, row in enumerate(series.itertuples(index=False)):
        px = date_x(row.date, START, end, chart_x, chart_width)
        py = y_map(float(row.normalized))
        path.moveTo(px, py) if index == 0 else path.lineTo(px, py)
    pdf.setStrokeColor(BLUE)
    pdf.setLineWidth(1.8)
    pdf.drawPath(path, fill=0, stroke=1)

    lookup = series.set_index("date")["normalized"]
    for event in events.loc[events["sector"].eq(sector)].itertuples(index=False):
        if event.pivot_date not in lookup.index or not START <= event.pivot_date <= end:
            continue
        px = date_x(event.pivot_date, START, end, chart_x, chart_width)
        py = y_map(float(lookup.loc[event.pivot_date]))
        is_bottom = event.direction == "ROTATION_IN"
        draw_triangle(pdf, px, py, 5, GREEN if is_bottom else RED, is_bottom)

    draw_text(pdf, "총수익지수 · 2026-06-29=100", chart_x, 501, size=7.5, color=MID_GREY)
    draw_text(pdf, "성과판정 대기", pending_x + 4, 478, size=6.5, color=MID_GREY)
    draw_track(pdf, gauges, calendar, sector, "CURRENT_70", "현재 전략 · 고정 70점", START, end, chart_x, chart_width, 168)
    draw_track(pdf, gauges, calendar, sector, "ADAPTIVE", "신규 전략 · 동적 85~95점", START, end, chart_x, chart_width, 108)

    ticks = pd.date_range(START, end, periods=6)
    for tick in ticks:
        tx = date_x(tick, START, end, chart_x, chart_width)
        pdf.setStrokeColor(GRID)
        pdf.line(tx, 71, tx, 77)
        pdf.setFillColor(MID_GREY)
        pdf.setFont("Malgun", 7)
        pdf.drawCentredString(tx, 58, tick.strftime("%m-%d"))
    draw_footer(pdf, page, pages)


def self_test() -> None:
    frame = pd.DataFrame(
        {"score": [0.69, 0.71, 0.89, 0.91], "dynamic_threshold": [0.68, 0.72, 0.88, 0.92]}
    )
    assert signal_mask(frame, "CURRENT_70").tolist() == [False, True, True, True]
    assert signal_mask(frame, "ADAPTIVE").tolist() == [True, False, True, False]
    events = pd.DataFrame(
        {
            "sector": ["1010", "1010", "1010"],
            "_position": [1, 3, 20],
            "score": [0.8, 0.9, 0.8],
            "dynamic_threshold": [0.9, 0.9, 0.9],
            "target_event_confirmation_position": [5.0, 5.0, np.nan],
            "target_event_pivot_position": [3.0, 3.0, np.nan],
        }
    )
    assert direction_metrics(events, "CURRENT_70") == {
        "eligible": 1,
        "hits": 1,
        "clusters": 2,
        "matched_clusters": 1,
    }


def build(output: Path) -> dict[str, Any]:
    register_fonts()
    gauges, prices, calendar = load_inputs()
    end = pd.Timestamp(gauges["date"].max())
    matured = gauges.loc[gauges["target"].notna(), "date"]
    if matured.empty:
        raise ValueError("no matured labels in comparison window")
    matured_end = pd.Timestamp(matured.max())
    events = actual_events(gauges, calendar)
    sectors = gauges[["sector", "sector_name"]].drop_duplicates().sort_values("sector")
    current = strategy_metrics(gauges, "CURRENT_70")
    adaptive = strategy_metrics(gauges, "ADAPTIVE")
    pages = 1 + len(sectors)
    output.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(output), pagesize=landscape(A4), pageCompression=1)
    pdf.setTitle("섹터 추세 전환 경보 비교")
    pdf.setAuthor("Market Lens Asset Management · Research Shadow")
    draw_cover(pdf, sectors, current, adaptive, end, pages)
    pdf.showPage()
    for page, row in enumerate(sectors.itertuples(index=False), start=2):
        draw_sector_page(
            pdf,
            page,
            pages,
            row.sector,
            row.sector_name,
            prices.loc[prices["sector"].eq(row.sector)],
            gauges,
            events,
            calendar,
            matured_end,
            end,
        )
        pdf.showPage()
    pdf.save()
    reader = PdfReader(str(output))
    if len(reader.pages) != pages or output.stat().st_size < 80_000:
        raise RuntimeError("generated PDF failed page-count or size validation")
    return {
        "output": str(output),
        "period": [str(START.date()), str(end.date())],
        "matured_through": str(matured_end.date()),
        "sectors": len(sectors),
        "current": current,
        "adaptive": adaptive,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("self-test: PASS")
        return
    print(json.dumps(build(args.output.resolve()), ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
