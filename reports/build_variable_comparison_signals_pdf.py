"""Plot saved validation-threshold alerts only. No fitting, ranking, or signal thinning."""
import json
import sqlite3
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from pypdf import PdfReader
from reportlab.lib.pagesizes import A4,landscape
from reportlab.pdfgen import canvas
from sector_rotation import SECTOR_ETFS
from build_sector_turning_alert_backtest_pdf import (
    register_fonts,draw_text,draw_triangle,TEAL,PAPER,BLUE,GREEN,RED,GRID,MID_GREY)

SOURCE=ROOT/'outputs/variable_barrier_comparison_20260921'
PDF=ROOT/'output/pdf/variable_barrier_comparison_signals_20260921.pdf'
MODELS={'PRICE_BOOST':'부스팅 | 가격 기준', 'SPREAD_BOOST':'부스팅 | 가격 + 스프레드',
        'PRICE_EXTRA':'Extra Trees | 가격 기준', 'SPREAD_EXTRA':'Extra Trees | 가격 + 스프레드'}


def build(price_only=False, hybrid=False):
    PDF=ROOT/('output/pdf/sector_hybrid_in_boost_out_extra_20260921.pdf' if hybrid else
              'output/pdf/variable_barrier_price_models_signals_20260921.pdf' if price_only else
              'output/pdf/variable_barrier_comparison_signals_20260921.pdf')
    models={k:v for k,v in MODELS.items() if not (price_only or hybrid) or k.startswith('PRICE_')}
    if PDF.exists():raise FileExistsError(PDF)
    data=pd.read_csv(SOURCE/'predictions.csv',parse_dates=['date'])
    data=data.loc[data.model.isin(models)].copy()
    assert set(data.model)==set(models) and set(data.ticker)==set(SECTOR_ETFS)
    assert not data.duplicated(['model','direction','ticker','date']).any()
    assert data.alert.isin([True,False]).all()
    assert (data.alert==(data.score>=data.threshold)).all()
    if hybrid:
        data=data.loc[(data.model.eq('PRICE_BOOST')&data.direction.eq('IN'))|
                      (data.model.eq('PRICE_EXTRA')&data.direction.eq('OUT'))].copy()
        assert not data.duplicated(['ticker','date']).any()
        models={'HYBRID':'초록 ▲ 부스팅 IN / 빨강 ▼ Extra Trees OUT'}
        data.loc[data.alert,['date','ticker','direction','model','score','threshold']].to_csv(
            SOURCE/'hybrid_in_boost_out_extra_signals.csv',index=False)
    with sqlite3.connect((ROOT/'regime_router_publish.db').as_uri()+'?mode=ro',uri=True) as c:
        raw=pd.read_sql_query('SELECT date,ticker,adj_close FROM daily_prices',c,parse_dates=['date'])
    raw=raw.loc[raw.ticker.isin([*SECTOR_ETFS,'SPY'])]
    assert not raw.duplicated(['date','ticker']).any()
    prices=raw.pivot(index='date',columns='ticker',values='adj_close').sort_index()
    prices=prices.loc[prices.SPY.notna()]
    saved=pd.read_csv(SOURCE/'research_frame.csv',parse_dates=['date'])
    for ticker,g in saved.groupby('ticker'):
        assert np.allclose(prices[ticker].reindex(g.date),g.observed_price,rtol=1e-8,atol=1e-8)
    register_fonts();w,h=landscape(A4)
    pdf=canvas.Canvas(str(PDF),pagesize=landscape(A4));pdf.setTitle('11-sector comparison - saved IN OUT alerts')
    total=23
    def footer(page):
        draw_text(pdf,'저장된 예측 시점만 표시 · 연구 결과 · 현재 수정 이력 / PIT 미검증 · 웹 변경 없음',40,22,size=8,color=MID_GREY)
        draw_text(pdf,f'{page} / {total}',w-75,22,size=8,color=MID_GREY)
    draw_text(pdf,'섹터 전환 신호 | 중간점검' if hybrid else '가격 기준: 부스팅 vs Extra Trees' if price_only else '섹터 전환·지속 비교학습: 경보 위치',40,h-65,font='Malgun-Bold',size=24,color=TEAL)
    notes=[
        f'11개 섹터 ETF | 저장된 예측기간 {data.date.min():%Y-%m-%d} ~ {data.date.max():%Y-%m-%d}',
        '섹터마다 전체 평가기간 1장 + 2025년 이후 확대 1장. 모든 패널의 날짜축·가격축은 같습니다.',
        '초록 ▲ IN / 빨강 ▼ OUT. 경보가 발생한 날짜의 조정종가 위치에 표시합니다.',
        '정답·오답·사후 고저점은 표시하지 않았습니다. 연속 경보도 숨기거나 합치지 않았습니다.',
        ('IN은 가격 기준 부스팅, OUT은 가격 기준 Extra Trees. 같은 가격선에 두 신호를 합쳤습니다.' if hybrid else
         '위: 부스팅 가격 기준 / 아래: Extra Trees 가격 기준. 스프레드 추가 모델은 제외했습니다.' if price_only else
         '위 두 패널: 부스팅의 가격 기준 vs 스프레드 추가 / 아래 두 패널: Extra Trees의 동일 비교'),
        '경보 기준은 이전 2년의 결과 확정 검증 점수 90백분위. 평가 연도에는 고정 적용했습니다.',
        '사후 상위 10%를 골라 그린 그래프가 아닙니다. 평가기간별 경보 횟수는 달라질 수 있습니다.',
        '주간 관측 중 공통 입력이 완비된 날짜만 평가했습니다. 빈 구간은 정상 판정만을 뜻하지 않습니다.',
        '신호 시점은 실제 체결가가 아닙니다. 실험의 진입 대용값은 다음 거래일 종가였습니다.',
        'IN·OUT은 큰 반대 방향 움직임 예측이며 정확한 고저점이나 보유 포지션 변경 지시가 아닙니다.',
        ('기존 결과를 본 뒤 선택한 중간점검용 조합입니다. 신규 학습·독립 검증·매매 수익 검증은 아닙니다.' if hybrid else
         '이미 반복 검토한 기간이며 완전히 새로운 독립 검증은 아닙니다. 새 학습·문턱 조정은 없습니다.'),
        '읽는 순서: XLK 12~13쪽 → XLE 6~7쪽. 상승 도중 OUT과 반복 경보 위치를 먼저 비교하세요.',
    ]
    for i,line in enumerate(notes):draw_text(pdf,line,42,h-115-i*29,size=10)
    footer(1);pdf.showPage();page=2;drawn=expected=0
    for ticker,name in SECTOR_ETFS.items():
        for recent in (False,True):
            start=max(data.date.min(),pd.Timestamp('2025-01-01')) if recent else data.date.min()
            end=data.date.max();series=prices.loc[start:end,ticker].copy()
            start=series.index[0];series=series/series.dropna().iloc[0]*100
            low,high=series.min(),series.max();pad=max((high-low)*.12,1);low-=pad;high+=pad
            pdf.setFillColor(TEAL);pdf.rect(0,h-55,w,55,fill=1,stroke=0)
            draw_text(pdf,f'{ticker}  {name}',38,h-35,font='Malgun-Bold',size=18,color=PAPER)
            draw_text(pdf,f'{start:%Y-%m-%d} ~ {end:%Y-%m-%d} | 조정가격 시작=100 | 초록 ▲ IN / 빨강 ▼ OUT',40,h-77,size=10)
            draw_text(pdf,'모든 경보 시점 표시 · 사후 선별/합치기 없음 · 실제 경보 기준은 전년도까지의 검증으로 결정',40,h-94,size=8,color=MID_GREY)
            sample=data.loc[data.ticker.eq(ticker)&data.date.between(start,end)]
            ticks=[start,*[t for t in pd.date_range(start,end,freq='6MS') if (t-start).days>65 and (end-t).days>65],end]
            for i,(model,label) in enumerate(models.items()):
                x,pw=64,w-108
                ph,y=(310,125) if hybrid else (155,h-290-i*210) if price_only else (76,h-210-i*105)
                dx=lambda d:x+(d-start).days/(end-start).days*pw
                dy=lambda v:y+(v-low)/(high-low)*ph
                chosen=sample.loc[sample.alert] if hybrid else sample.loc[sample.model.eq(model)&sample.alert]
                incoming=int(chosen.direction.eq('IN').sum());expected+=len(chosen)
                draw_text(pdf,f'{label} | IN {incoming} / OUT {len(chosen)-incoming}',x,y+ph+8,font='Malgun-Bold',size=9)
                for v in np.linspace(low,high,3):
                    pdf.setStrokeColor(GRID);pdf.setLineWidth(.4);pdf.line(x,dy(v),x+pw,dy(v))
                    draw_text(pdf,f'{v:.0f}',32,dy(v)-2,size=7,color=MID_GREY)
                path=pdf.beginPath();connected=False
                for date,value in series.items():
                    if pd.isna(value):connected=False;continue
                    if connected:path.lineTo(dx(date),dy(value))
                    else:path.moveTo(dx(date),dy(value))
                    connected=True
                pdf.setStrokeColor(BLUE);pdf.setLineWidth(.8);pdf.drawPath(path)
                for r in chosen.itertuples():
                    assert np.isfinite(series.loc[r.date])
                    up=r.direction=='IN';draw_triangle(pdf,dx(r.date),dy(series.loc[r.date]),3.2,GREEN if up else RED,up);drawn+=1
                for tick in ticks:
                    pdf.setFont('Malgun',7);pdf.setFillColor(MID_GREY);pdf.drawCentredString(dx(tick),y-14,tick.strftime('%Y-%m'))
            footer(page);pdf.showPage();page+=1
    pdf.save()
    assert drawn==expected and page-1==total and len(PdfReader(PDF).pages)==total
    audit=dict(pages=total,markers_including_repeated_zoom=drawn,unique_alert_rows=int(data.alert.sum()),
               source=str(SOURCE),models=list(models),retraining=False,threshold='saved validation threshold',checks='PASS: duplicates, thresholds, price vintage, markers, pages')
    if hybrid:audit['direction_models']={'IN':'PRICE_BOOST','OUT':'PRICE_EXTRA'}
    (SOURCE/('hybrid_visualization_audit.json' if hybrid else 'price_visualization_audit.json' if price_only else 'visualization_audit.json')).write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit));print(PDF)


if __name__=='__main__':build(price_only='--price-only' in sys.argv,hybrid='--hybrid' in sys.argv)
