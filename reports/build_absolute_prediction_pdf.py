"""Plot saved walk-forward predictions, never retrospective target labels."""
from pathlib import Path
import sqlite3
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
import numpy as np
from pypdf import PdfReader
from reportlab.lib.pagesizes import A4,landscape
from reportlab.pdfgen import canvas
from sector_rotation import SECTOR_ETFS
from build_sector_turning_alert_backtest_pdf import (
    register_fonts,draw_text,draw_triangle,draw_footer,TEAL,PAPER,BLUE,GREEN,RED,GRID,MID_GREY,
)

SOURCE=ROOT/'outputs/absolute_turn_prediction_20260921/predictions.csv'
PDF=ROOT/'output/pdf/absolute_turn_predictions_20260921.pdf'
POLICIES={'RRG':'기존 RRG 경보 (동일 평가 대상일)',
          'ABS_LOGIT':'절대가격 예측 모델', 'ABS_RRG_LOGIT':'절대가격 + RRG 예측 모델'}


def build():
    if PDF.exists():raise FileExistsError(PDF)
    # Target, label_end and future-return fields are intentionally NOT read for plotting.
    columns=['date','ticker','trend','refit_date','latest_train_label_end',*POLICIES]
    data=pd.read_csv(SOURCE,usecols=columns,parse_dates=['date','refit_date','latest_train_label_end'])
    assert set(data.ticker)==set(SECTOR_ETFS)
    assert not data.duplicated(['date','ticker']).any()
    assert (data.latest_train_label_end<data.refit_date).all()
    assert (data.refit_date<=data.date).all()
    assert data.trend.isin([-1,1]).all()
    for policy in POLICIES:
        assert data[policy].isin([True,False]).all()
    with sqlite3.connect((ROOT/'regime_router_publish.db').as_uri()+'?mode=ro',uri=True) as c:
        raw=pd.read_sql_query('SELECT date,ticker,adj_close FROM daily_prices',c,parse_dates=['date'])
    raw=raw.loc[raw.ticker.isin(SECTOR_ETFS)]
    assert not raw.duplicated(['date','ticker']).any()
    prices=raw.pivot(index='date',columns='ticker',values='adj_close').sort_index()
    register_fonts()
    pdf=canvas.Canvas(str(PDF),pagesize=landscape(A4))
    pdf.setTitle('11-sector walk-forward predictions: RRG versus absolute-price models')
    w,h=landscape(A4)
    draw_text(pdf,'11개 섹터 실제 예측 경보 비교',40,h-60,font='Malgun-Bold',size=25,color=TEAL)
    draw_text(pdf,'사후 정답 표시가 아니라 저장된 시간순 예측 결과입니다.',42,h-96,font='Malgun-Bold',size=14)
    lines=[
        f'예측 기간: {data.date.min():%Y-%m-%d} ~ {data.date.max():%Y-%m-%d} | 주간 판단 / 분기별 재학습',
        '비교: 기존 RRG / 절대가격 로지스틱 / 절대가격 + RRG 로지스틱',
        '초록 ▲ IN, 빨강 ▼ OUT. 각 판단일 종가 위치에 표시하며 실제 체결 지점은 아닙니다.',
        '미래 결과 라벨·실제 고저점·정답/오답 표시는 그래프에 넣지 않았습니다.',
        '각 섹터 2페이지: 전체 기간과 2025년 이후 확대. 각 그래프에 날짜축을 표시했습니다.',
        '세 모델 모두 같은 평가 대상일만 비교합니다. RRG의 전체 운영 이력을 뜻하지 않습니다.',
        '평가 대상은 20일·60일 가격 방향이 일치하고 학습/입력 자료가 충분한 주간 관측입니다.',
        '표시가 없으면 경보가 없었다는 뜻입니다. 평가 대상이 아니었던 날짜도 포함됩니다.',
        '예측 모델의 경보 문턱은 0.5로 고정. 점수는 보정된 확률이 아닙니다.',
        '현재 조정가격으로 복원한 과거 실험으로, 당시 실제 운용 기록이나 완전한 PIT 검증은 아닙니다.',
        '실험 결과: 새 모델은 특히 OUT 경보를 거의 내지 못했습니다. 성능 개선으로 해석하지 않습니다.',
        '이번 작업은 저장된 결과의 시각화만 수행했습니다. 재학습·웹·운영 DB 변경 없음.',
    ]
    for i,line in enumerate(lines):draw_text(pdf,line,44,h-143-i*25,size=10)
    draw_footer(pdf,1,23);pdf.showPage()
    page=1;plotted=0;expected=0
    for ticker,name in SECTOR_ETFS.items():
        for recent in (False,True):
            page+=1
            start=max(data.date.min(),pd.Timestamp('2025-01-01')) if recent else data.date.min()
            end=data.date.max()
            series=prices.loc[start:end,ticker].dropna()
            if series.empty:raise ValueError(f'no prices {ticker}')
            start=series.index[0];series=series/series.iloc[0]*100
            low,high=series.min(),series.max();pad=max((high-low)*.1,1);low-=pad;high+=pad
            sample=data.loc[data.ticker.eq(ticker)&data.date.between(start,end)]
            pdf.setFillColor(TEAL);pdf.rect(0,h-55,w,55,fill=1,stroke=0)
            draw_text(pdf,f'{ticker}  {name}',38,h-35,font='Malgun-Bold',size=18,color=PAPER)
            scope='최근 구간 확대' if recent else '전체 예측 기간'
            draw_text(pdf,f'{scope} | {start:%Y-%m-%d} ~ {end:%Y-%m-%d} | ETF 조정가격, 시작=100',40,h-76,size=10)
            draw_text(pdf,'초록 ▲ IN / 빨강 ▼ OUT | 미래 정답 표시 없음 | 같은 날짜·같은 가격축 비교',40,h-93,size=9,color=MID_GREY)
            x,pw=64,w-108
            def dx(date):return x+(date-start).days/(end-start).days*pw
            ticks=[start,*pd.date_range(start,end,freq='6MS' if recent else 'YS'),end]
            ticks=sorted(set(t for t in ticks if t in (start,end) or ((t-start).days>65 and (end-t).days>65)))
            for i,(policy,label) in enumerate(POLICIES.items()):
                y,ph=h-235-i*132,99
                chosen=sample.loc[sample[policy]]
                expected+=len(chosen)
                incoming=int(chosen.trend.eq(-1).sum());outgoing=int(chosen.trend.eq(1).sum())
                draw_text(pdf,f'{label}  |  IN {incoming} / OUT {outgoing}',x,y+ph+9,font='Malgun-Bold',size=10)
                for value in np.linspace(low,high,3):
                    yy=y+(value-low)/(high-low)*ph
                    pdf.setStrokeColor(GRID);pdf.setLineWidth(.4);pdf.line(x,yy,x+pw,yy)
                    draw_text(pdf,f'{value:.0f}',32,yy-2,size=7,color=MID_GREY)
                path=pdf.beginPath()
                for n,(date,value) in enumerate(series.items()):
                    (path.moveTo if n==0 else path.lineTo)(dx(date),y+(value-low)/(high-low)*ph)
                pdf.setStrokeColor(BLUE);pdf.setLineWidth(1);pdf.drawPath(path)
                for r in chosen.itertuples():
                    if r.date not in series.index:raise ValueError('signal without price')
                    up=r.trend==-1
                    draw_triangle(pdf,dx(r.date),y+(series.loc[r.date]-low)/(high-low)*ph,3.5,GREEN if up else RED,up)
                    plotted+=1
                if chosen.empty:draw_text(pdf,'이 구간 경보 없음',x+pw-125,y+ph-14,size=8,color=MID_GREY)
                for tick in ticks:
                    pdf.setFont('Malgun',7);pdf.setFillColor(MID_GREY);pdf.drawCentredString(dx(tick),y-15,tick.strftime('%Y-%m'))
            draw_text(pdf,f'공통 평가 표본: {len(sample)}개 주간 섹터 관측. 표시되지 않은 날짜를 정상 판정으로 간주하지 않습니다.',42,56,size=8,color=MID_GREY)
            draw_footer(pdf,page,23);pdf.showPage()
    pdf.save()
    assert len(PdfReader(PDF).pages)==23 and plotted==expected and plotted>0
    print(f'PASS: pages={page}; plotted markers including repeated zoom views={plotted}; source rows={len(data)}')
    print(PDF)


if __name__=='__main__':build()
