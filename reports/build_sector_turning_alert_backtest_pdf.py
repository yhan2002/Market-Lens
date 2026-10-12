"""Render existing sector turning backtest alerts over Bloomberg total-return paths.

This is a visualization-only report. It does not fit a model or rerun a backtest.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "outputs" / "sector_turning_recall_breadth_boost_20260903"
GICS_DIR = (
    ROOT
    / "outputs"
    / "gics_l2_staging_20260902_145743"
    / "bloomberg_gics_l2_20260902_145743"
)
DEFAULT_OUTPUT = ROOT / "output" / "pdf" / "sector_turning_alert_backtest_20260903.pdf"

PLOT_START = pd.Timestamp("2023-09-01")
PLOT_END = pd.Timestamp("2026-08-31")
ALERT_CUTOFF = 0.70
HORIZON = 10

CAPTURE_MODEL = "BETA_BREADTH_SHALLOW_BOOST"
FALSE_FREQUENCY_MODEL = "BREADTH_SHALLOW_BOOST"
MODEL_LABELS = {
    CAPTURE_MODEL: "전환 포착형",
    FALSE_FREQUENCY_MODEL: "신호 선별형",
}

INK = colors.HexColor("#172522")
TEAL = colors.HexColor("#0C5A4B")
BLUE = colors.HexColor("#1479B8")
RED = colors.HexColor("#D4483B")
GREEN = colors.HexColor("#1E8A5A")
MID_GREY = colors.HexColor("#718079")
LIGHT_GREY = colors.HexColor("#E8ECE9")
GRID = colors.HexColor("#D7DEDA")
PAPER = colors.white


@dataclass(frozen=True)
class AlertCluster:
    start_position: int
    end_position: int


def register_fonts() -> None:
    pdfmetrics.registerFont(TTFont("Malgun", r"C:\Windows\Fonts\malgun.ttf"))
    pdfmetrics.registerFont(TTFont("Malgun-Bold", r"C:\Windows\Fonts\malgunbd.ttf"))


def cluster_alerts(rows: pd.DataFrame) -> list[AlertCluster]:
    """Group nearby alert days for display without judging their outcomes."""
    positions = sorted(
        rows.loc[rows["score"] >= ALERT_CUTOFF, "_position"]
        .astype(int)
        .unique()
        .tolist()
    )
    groups: list[list[int]] = []
    for position in positions:
        if not groups or position - groups[-1][-1] > HORIZON:
            groups.append([position])
        else:
            groups[-1].append(position)

    return [AlertCluster(group[0], group[-1]) for group in groups]


def position_date(calendar: pd.Series, position: int) -> pd.Timestamp:
    if position < 0 or position >= len(calendar):
        raise ValueError(f"calendar position out of range: {position}")
    return pd.Timestamp(calendar.iloc[position])


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict]:
    predictions = pd.read_csv(
        RESULT_DIR / "predictions.csv",
        parse_dates=["date", "target_event_confirmation_date"],
    )
    predictions["sector"] = predictions["sector"].astype(str).str.zfill(4)
    prices = pd.read_csv(
        GICS_DIR / "industry_group_total_return.csv", parse_dates=["date"]
    )
    prices["sector"] = prices["industry_group_code"].astype(str).str.zfill(4)
    prices = prices.loc[
        prices["date"].between(PLOT_START, PLOT_END),
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
    summary = json.loads((RESULT_DIR / "summary.json").read_text(encoding="utf-8"))
    return predictions, prices, calendar, summary


def model_summary(summary: dict, model: str) -> dict[str, float]:
    rows = [
        summary["directions"][direction]["models"][model]["budgets"]["30pct"]
        for direction in ("ROTATION_OUT", "ROTATION_IN")
    ]
    return {
        "recall": float(np.mean([row["pre_pivot_recall"] for row in rows])),
        "ci_low": float(np.mean([row["pre_pivot_recall_ci_low"] for row in rows])),
        "alert_days": float(np.mean([row["alert_day_rate"] for row in rows])),
    }


def actual_events(rows: pd.DataFrame, calendar: pd.Series) -> pd.DataFrame:
    events = (
        rows.loc[
            rows["target"].notna()
            & rows["target_event_confirmation_position"].notna(),
            [
                "direction",
                "sector",
                "target_event_pivot_position",
                "target_event_confirmation_position",
            ],
        ]
        .drop_duplicates()
        .copy()
    )
    events["pivot_date"] = events["target_event_pivot_position"].map(
        lambda value: position_date(calendar, int(value))
    )
    events["confirmation_date"] = events["target_event_confirmation_position"].map(
        lambda value: position_date(calendar, int(value))
    )
    return events.sort_values(["sector", "pivot_date", "direction"])


def draw_text(
    pdf: canvas.Canvas,
    text: str,
    x: float,
    y: float,
    *,
    font: str = "Malgun",
    size: float = 9,
    color=INK,
) -> None:
    pdf.setFont(font, size)
    pdf.setFillColor(color)
    pdf.drawString(x, y, text)


def draw_wrapped(
    pdf: canvas.Canvas,
    text: str,
    x: float,
    y: float,
    width: float,
    *,
    font: str = "Malgun",
    size: float = 9,
    leading: float = 13,
    color=INK,
) -> float:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if pdfmetrics.stringWidth(candidate, font, size) <= width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    for line in lines:
        draw_text(pdf, line, x, y, font=font, size=size, color=color)
        y -= leading
    return y


def draw_footer(pdf: canvas.Canvas, page: int, pages: int) -> None:
    width, _ = landscape(A4)
    pdf.setStrokeColor(GRID)
    pdf.line(36, 34, width - 36, 34)
    draw_text(
        pdf,
        "RESEARCH SHADOW ONLY · PIT_PARTIAL / CURRENT_REVISED_HISTORY_ONLY",
        40,
        18,
        size=7,
        color=MID_GREY,
    )
    label = f"{page} / {pages}"
    pdf.setFont("Malgun", 7)
    pdf.setFillColor(MID_GREY)
    pdf.drawRightString(width - 40, 18, label)


def draw_card(
    pdf: canvas.Canvas,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    metric: dict[str, float],
    accent,
) -> None:
    pdf.setFillColor(colors.HexColor("#F4F7F5"))
    pdf.roundRect(x, y, width, height, 7, fill=1, stroke=0)
    pdf.setFillColor(accent)
    pdf.roundRect(x, y, 7, height, 7, fill=1, stroke=0)
    draw_text(pdf, title, x + 22, y + height - 25, font="Malgun-Bold", size=13)
    draw_text(
        pdf,
        f"사전포착률  {metric['recall']:.1%}",
        x + 22,
        y + height - 54,
        font="Malgun-Bold",
        size=18,
        color=accent,
    )
    draw_text(
        pdf,
        f"95% 하한 {metric['ci_low']:.1%}  |  경보일 {metric['alert_days']:.1%}",
        x + 22,
        y + 18,
        size=8.5,
        color=MID_GREY,
    )


def draw_legend_sample(
    pdf: canvas.Canvas, x: float, y: float, label: str, color
) -> None:
    pdf.setFillColor(color)
    pdf.setStrokeColor(color)
    pdf.rect(x, y - 3, 22, 7, fill=1, stroke=0)
    draw_text(pdf, label, x + 30, y - 2, size=8.5)


def draw_cover(pdf: canvas.Canvas, summary: dict, sectors: pd.DataFrame, pages: int) -> None:
    width, height = landscape(A4)
    pdf.setFillColor(TEAL)
    pdf.rect(0, height - 95, width, 95, fill=1, stroke=0)
    draw_text(
        pdf,
        "섹터 추세 전환 경보 백테스트 시각화",
        42,
        height - 50,
        font="Malgun-Bold",
        size=24,
        color=PAPER,
    )
    draw_text(
        pdf,
        "GICS Industry Group 총수익지수와 실제 경보 시점을 한 화면에서 비교",
        43,
        height - 76,
        size=11,
        color=colors.HexColor("#DCEBE6"),
    )
    pdf.setFillColor(colors.HexColor("#E9B949"))
    pdf.roundRect(width - 170, height - 67, 128, 28, 5, fill=1, stroke=0)
    draw_text(
        pdf,
        "목표 모드: 중단 유지",
        width - 157,
        height - 58,
        font="Malgun-Bold",
        size=9.5,
        color=INK,
    )

    capture = model_summary(summary, CAPTURE_MODEL)
    false_min = model_summary(summary, FALSE_FREQUENCY_MODEL)
    draw_card(pdf, 42, 374, 360, 105, "전환 포착형", capture, BLUE)
    draw_card(pdf, 430, 374, 360, 105, "신호 선별형", false_min, TEAL)

    draw_text(pdf, "분석 범위", 42, 340, font="Malgun-Bold", size=13)
    scope_lines = [
        "모델 설정 기간 2023-09-01~2026-08-31 · 실제 OOS 점수 2024-12-30부터",
        "21개 GICS Industry Group · 30% 명목 경보예산(score ≥ 0.70) · 10거래일 확인창",
        "가격선: Bloomberg Gross Total Return Index, 2023-09-01=100",
        "새 모델 학습·재백테스트·운영 반영 없이 기존 predictions.csv만 시각화",
    ]
    y = 318
    for line in scope_lines:
        draw_text(pdf, "• " + line, 48, y, size=9.5)
        y -= 19

    draw_text(pdf, "그래프 읽는 법", 42, 225, font="Malgun-Bold", size=13)
    draw_legend_sample(pdf, 48, 198, "빨강 막대: OUT 신호가 켜진 기간", RED)
    draw_legend_sample(pdf, 310, 198, "초록 막대: IN 신호가 켜진 기간", GREEN)
    draw_text(pdf, "▼ 빨강: 실제 고점 종료", 48, 170, size=9, color=RED)
    draw_text(pdf, "▲ 초록: 실제 저점 종료", 220, 170, size=9, color=GREEN)
    draw_text(pdf, "회색 구간: 학습/점수 준비 또는 라벨 미성숙", 395, 170, size=9, color=MID_GREY)

    draw_text(pdf, "포함 업종", 42, 132, font="Malgun-Bold", size=12)
    sector_rows = list(sectors.itertuples(index=False))
    columns = 3
    per_column = int(np.ceil(len(sector_rows) / columns))
    for index, row in enumerate(sector_rows):
        column = index // per_column
        line = index % per_column
        draw_text(
            pdf,
            f"{row.sector}  {row.sector_name}",
            48 + column * 255,
            110 - line * 11,
            size=7.3,
            color=MID_GREY,
        )
    draw_footer(pdf, 1, pages)


def date_x(date: pd.Timestamp, x: float, width: float) -> float:
    span = (PLOT_END - PLOT_START).days
    return x + ((pd.Timestamp(date) - PLOT_START).days / span) * width


def draw_triangle(pdf: canvas.Canvas, x: float, y: float, size: float, color, up: bool) -> None:
    path = pdf.beginPath()
    if up:
        path.moveTo(x, y + size)
        path.lineTo(x - size, y - size)
        path.lineTo(x + size, y - size)
    else:
        path.moveTo(x, y - size)
        path.lineTo(x - size, y + size)
        path.lineTo(x + size, y + size)
    path.close()
    pdf.setFillColor(color)
    pdf.setStrokeColor(PAPER)
    pdf.setLineWidth(0.7)
    pdf.drawPath(path, fill=1, stroke=1)


def draw_cluster(
    pdf: canvas.Canvas,
    cluster: AlertCluster,
    calendar: pd.Series,
    x: float,
    width: float,
    y: float,
    direction: str,
) -> None:
    start = date_x(position_date(calendar, cluster.start_position), x, width)
    end = date_x(position_date(calendar, cluster.end_position), x, width)
    bar_width = max(2.5, end - start)
    color = RED if direction == "ROTATION_OUT" else GREEN
    pdf.setFillColor(color)
    pdf.setStrokeColor(color)
    pdf.rect(start, y, bar_width, 7, fill=1, stroke=0)


def draw_signal_track(
    pdf: canvas.Canvas,
    predictions: pd.DataFrame,
    calendar: pd.Series,
    sector: str,
    model: str,
    x: float,
    width: float,
    baseline_y: float,
) -> None:
    draw_text(pdf, MODEL_LABELS[model], x, baseline_y + 17, font="Malgun-Bold", size=8.5)
    pdf.setStrokeColor(GRID)
    pdf.setLineWidth(0.6)
    pdf.line(x, baseline_y, x + width, baseline_y)
    draw_text(pdf, "OUT", x - 31, baseline_y + 3, size=6.5, color=RED)
    draw_text(pdf, "IN", x - 24, baseline_y - 10, size=6.5, color=GREEN)
    for direction, offset in (("ROTATION_OUT", 2), ("ROTATION_IN", -10)):
        rows = predictions.loc[
            predictions["model"].eq(model)
            & predictions["direction"].eq(direction)
            & predictions["sector"].eq(sector)
        ]
        for cluster in cluster_alerts(rows):
            draw_cluster(
                pdf, cluster, calendar, x, width, baseline_y + offset, direction
            )


def draw_sector_page(
    pdf: canvas.Canvas,
    page: int,
    pages: int,
    sector: str,
    name: str,
    price_rows: pd.DataFrame,
    predictions: pd.DataFrame,
    events: pd.DataFrame,
    calendar: pd.Series,
    oos_start: pd.Timestamp,
    matured_end: pd.Timestamp,
) -> None:
    page_width, page_height = landscape(A4)
    pdf.setFillColor(TEAL)
    pdf.rect(0, page_height - 58, page_width, 58, fill=1, stroke=0)
    draw_text(
        pdf,
        f"{sector}  {name}",
        42,
        page_height - 37,
        font="Malgun-Bold",
        size=18,
        color=PAPER,
    )
    draw_text(
        pdf,
        "총수익지수 변화와 30% 경보예산의 경보군집",
        page_width - 325,
        page_height - 35,
        size=9,
        color=colors.HexColor("#DCEBE6"),
    )

    chart_x, chart_y, chart_height = 58, 227, 280
    chart_width = page_width - chart_x - 42
    series = price_rows.sort_values("date").drop_duplicates("date")
    series = series.loc[series["total_return_level"].notna()].copy()
    if series.empty:
        raise ValueError(f"no price data for {sector}")
    series["normalized"] = series["total_return_level"] / series["total_return_level"].iloc[0] * 100
    values = series["normalized"].to_numpy(dtype=float)
    low, high = float(values.min()), float(values.max())
    padding = max((high - low) * 0.10, 2.0)
    y_low, y_high = low - padding, high + padding

    def y_map(value: float) -> float:
        return chart_y + (value - y_low) / (y_high - y_low) * chart_height

    pdf.setFillColor(colors.HexColor("#FAFBFA"))
    pdf.setStrokeColor(GRID)
    pdf.rect(chart_x, chart_y, chart_width, chart_height, fill=1, stroke=1)
    pdf.setFillColor(LIGHT_GREY)
    pdf.rect(
        chart_x,
        chart_y,
        max(0, date_x(oos_start, chart_x, chart_width) - chart_x),
        chart_height,
        fill=1,
        stroke=0,
    )
    pending_x = date_x(matured_end + pd.Timedelta(days=1), chart_x, chart_width)
    pdf.setFillColor(colors.HexColor("#F1F2F1"))
    pdf.rect(
        pending_x,
        chart_y,
        chart_x + chart_width - pending_x,
        chart_height,
        fill=1,
        stroke=0,
    )
    for tick in np.linspace(y_low, y_high, 5):
        y_tick = y_map(float(tick))
        pdf.setStrokeColor(GRID)
        pdf.setLineWidth(0.5)
        pdf.line(chart_x, y_tick, chart_x + chart_width, y_tick)
        pdf.setFont("Malgun", 7)
        pdf.setFillColor(MID_GREY)
        pdf.drawRightString(chart_x - 7, y_tick - 2, f"{tick:.0f}")

    path = pdf.beginPath()
    first = True
    for row in series.itertuples(index=False):
        px = date_x(row.date, chart_x, chart_width)
        py = y_map(float(row.normalized))
        if first:
            path.moveTo(px, py)
            first = False
        else:
            path.lineTo(px, py)
    pdf.setStrokeColor(BLUE)
    pdf.setLineWidth(1.6)
    pdf.drawPath(path, fill=0, stroke=1)

    sector_events = events.loc[events["sector"].eq(sector)]
    price_lookup = series.set_index("date")["normalized"]
    for event in sector_events.itertuples(index=False):
        if event.pivot_date not in price_lookup.index:
            continue
        px = date_x(event.pivot_date, chart_x, chart_width)
        py = y_map(float(price_lookup.loc[event.pivot_date]))
        is_bottom = event.direction == "ROTATION_IN"
        draw_triangle(pdf, px, py, 5, GREEN if is_bottom else RED, is_bottom)

    for boundary, label in ((oos_start, "OOS 점수 시작"), (matured_end, "최종 성숙 라벨")):
        bx = date_x(boundary, chart_x, chart_width)
        pdf.setStrokeColor(MID_GREY)
        pdf.setDash(2, 2)
        pdf.line(bx, chart_y, bx, chart_y + chart_height)
        pdf.setDash()
        draw_text(pdf, label, bx + 3, chart_y + chart_height - 12, size=6.5, color=MID_GREY)

    draw_text(pdf, "Gross Total Return Index · 2023-09-01=100", chart_x, 514, size=7.5, color=MID_GREY)
    draw_signal_track(
        pdf, predictions, calendar, sector, CAPTURE_MODEL, chart_x, chart_width, 176
    )
    draw_signal_track(
        pdf,
        predictions,
        calendar,
        sector,
        FALSE_FREQUENCY_MODEL,
        chart_x,
        chart_width,
        129,
    )

    ticks = pd.to_datetime(
        [
            "2023-09-01",
            "2024-03-01",
            "2024-09-01",
            "2025-03-01",
            "2025-09-01",
            "2026-03-01",
            "2026-08-31",
        ]
    )
    for tick in ticks:
        tx = date_x(tick, chart_x, chart_width)
        pdf.setStrokeColor(GRID)
        pdf.line(tx, chart_y, tx, chart_y - 6)
        pdf.setFont("Malgun", 7.5)
        pdf.setFillColor(MID_GREY)
        pdf.drawCentredString(tx, chart_y - 17, tick.strftime("%Y-%m"))
        pdf.line(tx, 109, tx, 115)
        pdf.setFont("Malgun", 7)
        pdf.setFillColor(MID_GREY)
        pdf.drawCentredString(tx, 96, tick.strftime("%Y-%m"))

    draw_footer(pdf, page, pages)


def self_test() -> None:
    frame = pd.DataFrame(
        {
            "_position": [1, 2, 9, 10, 30],
            "score": [0.8, 0.75, 0.2, 0.9, 0.95],
            "target": [0.0, 1.0, 1.0, 1.0, 0.0],
            "target_event_pivot_position": [np.nan, 8, 8, 8, np.nan],
            "target_event_confirmation_position": [np.nan, 12, 12, 12, np.nan],
        }
    )
    clusters = cluster_alerts(frame)
    assert clusters == [
        AlertCluster(1, 10),
        AlertCluster(30, 30),
    ]


def build(output: Path) -> Path:
    register_fonts()
    predictions, prices, calendar, summary = load_inputs()
    selected = predictions.loc[
        predictions["model"].isin([CAPTURE_MODEL, FALSE_FREQUENCY_MODEL])
    ].copy()
    sectors = (
        selected[["sector", "sector_name"]]
        .drop_duplicates()
        .sort_values("sector")
        .reset_index(drop=True)
    )
    events = actual_events(
        selected.loc[selected["model"].eq(CAPTURE_MODEL)], calendar
    )
    oos_start = pd.Timestamp(selected["date"].min())
    matured_end = pd.Timestamp(selected.loc[selected["target"].notna(), "date"].max())
    pages = 1 + len(sectors)
    output.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(output), pagesize=landscape(A4), pageCompression=1)
    pdf.setTitle("섹터 추세 전환 경보 백테스트 시각화")
    pdf.setAuthor("Market Lens Asset Management · Research Shadow")
    draw_cover(pdf, summary, sectors, pages)
    pdf.showPage()
    for offset, row in enumerate(sectors.itertuples(index=False), start=2):
        sector_prices = prices.loc[prices["sector"].eq(row.sector)]
        draw_sector_page(
            pdf,
            offset,
            pages,
            row.sector,
            row.sector_name,
            sector_prices,
            selected,
            events,
            calendar,
            oos_start,
            matured_end,
        )
        pdf.showPage()
    pdf.save()
    reader = PdfReader(str(output))
    if len(reader.pages) != pages or output.stat().st_size < 100_000:
        raise RuntimeError("generated PDF failed page-count or size validation")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("self-test passed")
        return
    print(build(args.output.resolve()))


if __name__ == "__main__":
    main()
