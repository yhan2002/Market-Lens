"""Read-only price review: retrospective pivots versus causal confirmations, no fitting."""
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from pypdf import PdfReader
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfgen import canvas
from sector_rotation import SECTOR_ETFS
from sector_turning_pooled_shadow import directional_change_labels
from build_sector_turning_alert_backtest_pdf import (
    register_fonts, draw_text, draw_triangle, TEAL, PAPER, BLUE, GREEN, RED, GRID, MID_GREY,
)

SETTINGS = [('민감', 4, .04), ('중간', 6, .06), ('큰 움직임', 8, .08)]
PDF = ROOT / 'output/pdf/variable_duration_trend_labels_20260921.pdf'
OUT = ROOT / 'outputs/variable_duration_label_review_20260921'


def events(series, multiplier, minimum):
    available = series.notna()
    pieces = []
    for segment_id, part in series[available].groupby((~available).cumsum()[available]):
        if len(part) < 63:
            continue
        # horizon only controls unused future target columns, not event duration.
        labels = directional_change_labels(
            part, horizon=1, cooldown_sessions=0, volatility_window=60,
            volatility_multiplier=multiplier, barrier_horizon=1,
            minimum_barrier=minimum, maximum_barrier=.30,
        )
        selected = labels.loc[labels.event.ne(''), [
            'event', 'pivot_date', 'confirmation_date', 'barrier',
            'pivot_position', 'confirmation_position',
        ]].copy()
        selected['delay_sessions'] = selected.confirmation_position - selected.pivot_position
        selected['pivot_spacing_sessions'] = selected.pivot_position.diff()
        selected['segment'] = segment_id
        pieces.append(selected)
    columns = ['event', 'pivot_date', 'confirmation_date', 'barrier', 'delay_sessions', 'pivot_spacing_sessions', 'segment']
    return pd.concat(pieces)[columns] if pieces else pd.DataFrame(columns=columns)


def self_check():
    idx = pd.bdate_range('2020-01-01', periods=600)
    price = pd.Series(np.exp(np.linspace(0, .3, 600) + .17*np.sin(np.arange(600)/24)), index=idx)
    whole = events(price, 4, .04)
    assert len(whole) > 5
    pd.testing.assert_frame_equal(events(price.iloc[:400], 4, .04), whole.loc[whole.confirmation_date <= idx[399]])
    assert (whole.pivot_date < whole.confirmation_date).all()
    assert whole.event.ne(whole.event.shift()).all()
    broken = price.copy(); broken.iloc[300] = np.nan
    separated = events(broken, 4, .04)
    assert not ((separated.pivot_date < idx[300]) & (separated.confirmation_date > idx[300])).any()
    print('PASS: prefix invariance, alternating events, confirmation delay, gap reset')


def build():
    self_check()
    if PDF.exists() or OUT.exists():
        raise FileExistsError('Review output already exists; preserve it.')
    with sqlite3.connect((ROOT/'regime_router_publish.db').as_uri()+'?mode=ro', uri=True) as con:
        raw = pd.read_sql_query('SELECT date,ticker,adj_close FROM daily_prices', con, parse_dates=['date'])
    raw = raw[raw.ticker.isin([*SECTOR_ETFS, 'SPY'])]
    assert not raw.duplicated(['date', 'ticker']).any()
    prices = raw.pivot(index='date', columns='ticker', values='adj_close').sort_index()
    prices = prices.loc[prices.SPY.notna()]
    observed = prices.stack().dropna()
    assert np.isfinite(observed).all() and observed.gt(0).all()
    all_events = []
    for ticker in SECTOR_ETFS:
        for name, multiplier, minimum in SETTINGS:
            e = events(prices[ticker], multiplier, minimum)
            e['ticker'], e['sensitivity'] = ticker, name
            all_events.append(e)
    labels = pd.concat(all_events, ignore_index=True)
    OUT.mkdir(parents=True); PDF.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(OUT/'confirmed_events.csv', index=False, encoding='utf-8-sig')
    summary = labels.groupby(['ticker', 'sensitivity']).agg(
        events=('event', 'size'), median_confirmation_delay=('delay_sessions', 'median'),
        median_pivot_spacing=('pivot_spacing_sessions', 'median'))
    summary.to_csv(OUT/'summary.csv', encoding='utf-8-sig')
    register_fonts(); w, h = landscape(A4)
    pdf = canvas.Canvas(str(PDF), pagesize=landscape(A4))
    pdf.setTitle('Variable-duration sector trend labels - NOT predictions')

    def footer(page):
        draw_text(pdf, '정답 후보 검수 · 예측/성과 평가 아님 · 현재 조정가격 이력 · 운영 변경 없음', 40, 22, size=8, color=MID_GREY)
        draw_text(pdf, f'{page} / 23', w-75, 22, size=8, color=MID_GREY)

    draw_text(pdf, '기간을 고정하지 않은 섹터 추세 구간', 40, h-65, font='Malgun-Bold', size=24, color=TEAL)
    draw_text(pdf, '먼저 정답의 모양을 확인합니다. 학습·예측 결과가 아닙니다.', 40, h-100, size=14, color=RED)
    notes = [
        '11개 GICS Level 1 섹터 ETF | 각 섹터: 2023년 이후 + 2025년 이후 확대',
        f'가격 최신일: {prices.index[-1]:%Y-%m-%d} | 계산은 보유한 전체 이력에서 시작',
        '빨강 ▼ = 사후 고점 / 초록 ▲ = 사후 저점 / 같은 색 빈 원 = 전환 확인일',
        '삼각형 날짜에는 정답을 몰랐습니다. 반대 방향 움직임이 충분히 커진 원 날짜에 확인됩니다.',
        '추세 길이에는 20일·3개월·6개월 등 제한이 없습니다. 마지막 미확정 고점/저점도 만들지 않습니다.',
        '확인 기준: 직전 60거래일 일간 로그수익률 변동성 × 배수, 고점/저점 갱신 시 기준 고정',
        '민감: 4배·최소 4% / 중간: 6배·최소 6% / 큰 움직임: 8배·최소 8% (로그변화 기준)',
        '세 기준 모두 최대 로그변화 30%. 60일은 변동성 계산 기간이며 추세 유지 조건이 아닙니다.',
        '세 가지는 비교용 후보이지 최적값이 아닙니다. 큰 기준은 작은 조정을 줄이지만 확인도 늦습니다.',
        '절대 조정가격 기준이며 SPY 대비 순환매와는 다릅니다. 결측값은 채우지 않고 구간을 재시작합니다.',
        '검수 질문: 상승 중 작은 눌림을 과도하게 종료로 나누는가? 중요한 하락/회복을 놓치는가?',
        '선택 이후에만 과거 정보로 예측하는 실험을 설계합니다. 지금은 모델 학습·백테스트·웹 변경 없음.',
    ]
    for i, text in enumerate(notes): draw_text(pdf, text, 42, h-145-i*28, size=10)
    footer(1); pdf.showPage(); page = 2
    for ticker, name in SECTOR_ETFS.items():
        for start in [pd.Timestamp('2023-01-01'), pd.Timestamp('2025-01-01')]:
            view = prices[ticker].loc[start:].copy(); view = view/view.dropna().iloc[0]*100
            start, end = view.index[0], view.index[-1]
            low, high = view.min(), view.max(); pad = max((high-low)*.12, 1); low -= pad; high += pad
            pdf.setFillColor(TEAL); pdf.rect(0, h-55, w, 55, fill=1, stroke=0)
            draw_text(pdf, f'{ticker}  {name}', 38, h-35, font='Malgun-Bold', size=18, color=PAPER)
            draw_text(pdf, '▼ 사후 고점   ▲ 사후 저점   ○ 전환 확인일 | 삼각형은 당시 알 수 없었던 위치', 40, h-77, size=10)
            draw_text(pdf, f'{start:%Y-%m-%d} ~ {end:%Y-%m-%d} · 시작가격 = 100 · 세 패널 가격선 동일', 40, h-95, size=9, color=MID_GREY)
            for i, (label, mult, minimum) in enumerate(SETTINGS):
                x, pw, ph, y = 64, w-108, 95, h-230-i*140
                dx = lambda d: x+(d-start).days/(end-start).days*pw
                dy = lambda v: y+(v-low)/(high-low)*ph
                e = labels[labels.ticker.eq(ticker) & labels.sensitivity.eq(label)]
                visible = e[e.confirmation_date.between(start, end)]
                delay = f'{visible.delay_sessions.median():.0f}일' if len(visible) else 'N/A'
                draw_text(pdf, f'{label} · {mult}배 / 최소 {minimum:.0%} | 구간 내 확인 {len(visible)}회 · 고저점→확인 중앙값 {delay}', x, y+ph+10, font='Malgun-Bold', size=10)
                for v in np.linspace(low, high, 3):
                    pdf.setStrokeColor(GRID); pdf.setLineWidth(.4); pdf.line(x, dy(v), x+pw, dy(v))
                    draw_text(pdf, f'{v:.0f}', 32, dy(v)-2, size=7, color=MID_GREY)
                path = pdf.beginPath(); connected = False
                for date, value in view.items():
                    if pd.isna(value): connected = False; continue
                    if connected: path.lineTo(dx(date), dy(value))
                    else: path.moveTo(dx(date), dy(value))
                    connected = True
                pdf.setStrokeColor(BLUE); pdf.setLineWidth(.8); pdf.drawPath(path)
                for r in e.itertuples():
                    up = r.event == 'BOTTOM_END'; color = GREEN if up else RED
                    if start <= r.pivot_date <= end:
                        draw_triangle(pdf, dx(r.pivot_date), dy(view.loc[r.pivot_date]), 3.5, color, up)
                    if start <= r.confirmation_date <= end:
                        pdf.setStrokeColor(color); pdf.setLineWidth(1)
                        pdf.circle(dx(r.confirmation_date), dy(view.loc[r.confirmation_date]), 3, fill=0, stroke=1)
                ticks = [start, *[t for t in pd.date_range(start, end, freq='6MS') if (t-start).days>65 and (end-t).days>65], end]
                for t in ticks:
                    pdf.setFont('Malgun', 7); pdf.setFillColor(MID_GREY); pdf.drawCentredString(dx(t), y-15, t.strftime('%Y-%m'))
            footer(page); pdf.showPage(); page += 1
    pdf.save(); assert len(PdfReader(PDF).pages) == 23
    print(f'23 pages; {len(labels)} confirmed events across all stored history; {PDF}')


if __name__ == '__main__':
    self_check() if '--self-check' in sys.argv else build()
