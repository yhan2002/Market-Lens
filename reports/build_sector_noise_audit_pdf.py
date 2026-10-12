"""Reuse the previous sector PDF style; show actual signal dates without outcome labels."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from pypdf import PdfReader
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfgen import canvas

from build_sector_turning_alert_backtest_pdf import (
    GICS_DIR, BLUE, GREEN, RED, TEAL, GRID, MID_GREY, PAPER,
    draw_text, draw_triangle, draw_footer, register_fonts,
)
from sector_turning_noise_audit import SOURCE, filters

OUTPUT = ROOT / "output/pdf/sector_noise_filter_comparison_20260921.pdf"
LABELS = {
    "BASELINE": "기존 경보",
    "PERSISTENCE_2_OF_3": "3일 중 2일 유지",
    "BREADTH_CONFIRM": "Breadth 확인",
    "PERSISTENCE_AND_BREADTH": "신호 유지 + Breadth",
    "PERSISTENCE_BREADTH_MACRO": "신호 유지 + Breadth + 매크로",
}


def build():
    rows = pd.read_csv(SOURCE, dtype={"sector": "string"}, parse_dates=["date"])
    for _, group in rows.groupby("direction"):
        for key, mask in filters(group).items():
            rows.loc[group.index, key] = mask
    prices = pd.read_csv(GICS_DIR / "industry_group_total_return.csv", parse_dates=["date"])
    prices["sector"] = prices["industry_group_code"].astype(str).str.zfill(4)
    if prices.duplicated(["sector", "date"]).any():
        raise ValueError("duplicate prices")
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    windows = [(rows.date.min(), pd.Timestamp("2022-12-31")),
               (pd.Timestamp("2023-01-01"), rows.date.max())]
    sectors = sorted(rows.sector.unique())
    register_fonts()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(OUTPUT), pagesize=landscape(A4))
    pdf.setTitle("섹터별 노이즈 필터 비교 - IN / OUT 신호")
    width, height = landscape(A4)
    page = 0
    marks = 0
    for sector in sectors:
        full_price = prices.loc[prices.sector.eq(sector)].sort_values("date")
        if full_price.empty:
            raise ValueError(f"missing sector {sector}")
        name = str(full_price.industry_group_name.iloc[0])
        for start, end in windows:
            page += 1
            price = full_price.loc[full_price.date.between(start, end)].copy()
            if price.empty:
                raise ValueError(f"missing price window {sector}")
            price["level"] = price.total_return_level / price.total_return_level.iloc[0] * 100
            lookup = price.set_index("date")["level"]
            sample = rows.loc[rows.sector.eq(sector) & rows.date.between(start, end)]
            low, high = float(price.level.min()), float(price.level.max())
            padding = max((high-low)*.12, 2)
            low, high = low-padding, high+padding
            pdf.setFillColor(TEAL)
            pdf.rect(0, height-56, width, 56, fill=1, stroke=0)
            draw_text(pdf, f"{sector}  {name}", 38, height-35, font="Malgun-Bold", size=17, color=PAPER)
            draw_text(pdf, f"{start:%Y-%m-%d} ~ {end:%Y-%m-%d}  |  초록 ▲ IN / 빨강 ▼ OUT", 40, height-74, size=9)
            draw_text(pdf, "가격: 업종 총수익지수, 구간 첫 관측일=100 | 신호: SPY 대비 상대추세 연구 | 정답·오답 구분 없음", 40, height-89, size=8, color=MID_GREY)
            x, w = 62, width-102
            def dx(date):
                return x+(pd.Timestamp(date)-start).days/(end-start).days*w
            ticks = [start, *pd.date_range(start, end, freq="YS").tolist(), end]
            ticks = sorted(set(ticks))
            # Avoid overlapping first/last date labels with nearby annual ticks.
            ticks = [t for t in ticks if t in (start,end) or ((t-start).days>100 and (end-t).days>100)]
            for i, (key, label) in enumerate(LABELS.items()):
                y, h = height-180-i*83,  fifty_height
                draw_text(pdf, label, x, y+h+7, font="Malgun-Bold", size=9)
                pdf.setLineWidth(.35)
                for value in (low, (low+high)/2, high):
                    yy=y+(value-low)/(high-low)*h
                    pdf.setStrokeColor(GRID)
                    pdf.line(x,yy,x+w,yy)
                    draw_text(pdf, f"{value:.0f}", 32, yy-2, size=6.5, color=MID_GREY)
                path=pdf.beginPath()
                for n, p in enumerate(price.itertuples()):
                    px, py=dx(p.date), y+(p.level-low)/(high-low)*h
                    (path.moveTo if n==0 else path.lineTo)(px,py)
                pdf.setStrokeColor(BLUE)
                pdf.setLineWidth(.8)
                pdf.drawPath(path)
                selected=sample.loc[sample[key].eq(True)]
                if not selected.date.isin(lookup.index).all():
                    raise ValueError(f"signal date without price: {sector}")
                for signal in selected.itertuples():
                    up=signal.direction=="ROTATION_IN"
                    draw_triangle(pdf, dx(signal.date), y+(lookup.loc[signal.date]-low)/(high-low)*h,
                                  3, GREEN if up else RED, up)
                    marks += 1
                for tick in ticks:
                    pdf.setFont("Malgun",6.5)
                    pdf.setFillColor(MID_GREY)
                    pdf.drawCentredString(dx(tick),y-12,tick.strftime("%Y-%m"))
            draw_footer(pdf,page,len(sectors)*len(windows))
            pdf.showPage()
    pdf.save()
    assert len(PdfReader(OUTPUT).pages)==page
    assert marks>0
    print(f"pages={page}; signal markers={marks}; output={OUTPUT}")


fifty_height = 49
if __name__ == "__main__":
    build()
