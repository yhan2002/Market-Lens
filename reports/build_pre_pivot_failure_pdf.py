"""Read saved scores only: descriptive failure audit and unfiltered signal plots."""
from pathlib import Path
import json
import sqlite3
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from pypdf import PdfReader
from reportlab.lib.pagesizes import A4,landscape
from reportlab.pdfgen import canvas
from sector_rotation import SECTOR_ETFS
from build_sector_turning_alert_backtest_pdf import (
    register_fonts,draw_text,draw_triangle,draw_footer,TEAL,PAPER,BLUE,GREEN,RED,GRID,MID_GREY)

SOURCE=ROOT/'outputs/pre_pivot_divergence_20260921_corrected'
DEST=ROOT/'outputs/pre_pivot_failure_audit_20260921'
PDF=ROOT/'output/pdf/pre_pivot_divergence_signals_20260921.pdf'
MODELS={'PRICE_TREE':'가격 기준', 'PRICE_BOTH_TREE':'가격 + 거래량·옵션',
        'PRICE_BOTH_DIVERGENCE_TREE':'가격 + 거래량·옵션 + 반응 차이 (이번 버전)'}
KINDS=['LATE','CONTINUATION','ROUND_TRIP','OPPOSITE_UNMATCHED','INCOMPLETE']
LABELS={'LATE':'전환점 당일·이후 신호','CONTINUATION':'기존 방향 유지 (20일 종가 기준)',
        'ROUND_TRIP':'3% 이상 반대 움직임 후 되돌림','OPPOSITE_UNMATCHED':'방향은 맞지만 전환 라벨 불일치',
        'INCOMPLETE':'미래 가격 불충분'}


def classify(target,lead,path,direction):
    if target==1:return 'MATCH'
    if pd.notna(lead) and lead<=0:return 'LATE'
    if len(path)!=20 or not np.isfinite(path).all():return 'INCOMPLETE'
    signed=path if direction=='ROTATION_OUT' else -path
    if signed[-1]>=0:
        return 'ROUND_TRIP' if signed.min()<=-.03 else 'CONTINUATION'
    return 'OPPOSITE_UNMATCHED'


def self_test():
    assert classify(1,np.nan,[], 'ROTATION_OUT')=='MATCH'
    assert classify(0,-1,[], 'ROTATION_OUT')=='LATE'
    assert classify(0,np.nan,np.linspace(.001,.1,20),'ROTATION_OUT')=='CONTINUATION'
    assert classify(0,np.nan,np.r_[[-.04],np.linspace(-.02,.01,19)],'ROTATION_OUT')=='ROUND_TRIP'
    assert classify(0,np.nan,np.linspace(-.001,-.1,20),'ROTATION_OUT')=='OPPOSITE_UNMATCHED'
    assert classify(0,np.nan,-np.linspace(.001,.1,20),'ROTATION_IN')=='CONTINUATION'


def build():
    self_test()
    if (PDF.exists() or DEST.exists()) and '--replace-generated' not in sys.argv:raise FileExistsError('Output already exists')
    data=pd.read_csv(SOURCE/'predictions.csv',parse_dates=['date'])
    assert not data.duplicated(['direction','sector','date']).any()
    assert set(data.sector)==set(SECTOR_ETFS)
    with sqlite3.connect((ROOT/'regime_router_publish.db').as_uri()+'?mode=ro',uri=True) as con:
        names=['SPY',*SECTOR_ETFS]
        raw=pd.read_sql_query('SELECT date,ticker,adj_close FROM daily_prices WHERE ticker IN ('+
                             ','.join('?' for _ in names)+')',con,params=names,parse_dates=['date'])
    assert not raw.duplicated(['date','ticker']).any()
    prices=raw.pivot(index='date',columns='ticker',values='adj_close').sort_index()
    prices=prices.loc[prices.SPY.notna()]
    details=[]
    for r in data.itertuples():
        pos=prices.index.get_loc(r.date);price=prices.at[r.date,r.sector]
        assert np.isfinite(price) and price>0
        path=prices[r.sector].iloc[pos+1:pos+21].to_numpy()/price-1
        kind=classify(r.target,r.lead,path,r.direction)
        for model in MODELS:
            score=getattr(r,model)
            assert np.isfinite(score) and 0<=score<=1
            if score>=.5:
                details.append(dict(model=model,sector=r.sector,date=r.date,direction=r.direction,
                    score=score,target=r.target,category=kind,lead=r.lead,
                    return20=path[-1] if len(path)==20 else np.nan,
                    min20=float(np.nanmin(path)),max20=float(np.nanmax(path))))
    detail=pd.DataFrame(details)
    counts=detail.groupby(['model','direction','category']).size()
    latest=detail.loc[detail.model.eq('PRICE_BOTH_DIVERGENCE_TREE')]
    print('LATEST_COUNTS',latest.groupby(['direction','category']).size().to_json())
    print('XLK_COUNTS',latest.loc[latest.sector.eq('XLK')].groupby(['direction','category']).size().to_json())
    sector_summary=latest.assign(wrong=latest.target.eq(0)).groupby(['direction','sector']).agg(
        alerts=('wrong','size'),mismatches=('wrong','sum'))
    sector_summary['mismatch_share']=sector_summary.mismatches/sector_summary.alerts
    print(sector_summary.to_string())
    DEST.mkdir(parents=True,exist_ok=True)
    detail.to_csv(DEST/'alert_path_audit.csv',index=False)
    counts.rename('count').to_csv(DEST/'counts.csv')
    sector_summary.to_csv(DEST/'sector_summary.csv')

    register_fonts();pdf=canvas.Canvas(str(PDF),pagesize=landscape(A4))
    pdf.setTitle('11-sector pre-pivot signals and failure audit')
    w,h=landscape(A4);total=24
    draw_text(pdf,'11개 섹터 추세전환: 어디서 실패했나',40,h-65,font='Malgun-Bold',size=25,color=TEAL)
    notes=[
        '저장된 예측 점수만 시각화했습니다. 새 학습·문턱 튜닝·운영 변경은 없습니다.',
        f'기간: {data.date.min():%Y-%m-%d} ~ {data.date.max():%Y-%m-%d} | 일별 평가 대상 관측',
        '각 섹터: 전체 기간 1페이지 + 2025년 이후 확대 1페이지. 모든 그래프에 날짜축 표시.',
        '초록 ▲ IN / 빨강 ▼ OUT. 점수 0.5 이상인 모든 날짜를 종가 위치에 표시합니다.',
        '연속 경보도 생략·합치지 않습니다. 삼각형이 겹쳐 보이는 구간은 경보가 반복된 구간입니다.',
        '점수는 보정된 실제 확률이 아니며 신호 위치는 실제 체결 지점이 아닙니다.',
        '차트에 정답·오답·사후 고저점 표시는 넣지 않았습니다.',
        '신호가 없는 날에는 평가 대상이 아니거나 자료가 없었던 날짜도 포함됩니다.',
        '이번 버전과 거래량·옵션 버전은 저장된 예측값이 같으므로 두 패널이 동일합니다.',
        '학습은 2023년 이전, 이후 모델 고정. 반복 열람한 OOS이며 완전한 PIT 검증은 아닙니다.',
        '다음 페이지 실패 분해는 사후 평가일 뿐, 예측 입력이나 투자 수익 평가가 아닙니다.',
    ]
    for i,line in enumerate(notes):draw_text(pdf,line,44,h-115-i*29,size=11)
    draw_footer(pdf,1,total);pdf.showPage()
    draw_text(pdf,'이번 버전의 경보를 사후 분해한 결과',40,h-55,font='Malgun-Bold',size=23,color=TEAL)
    for i,direction in enumerate(('ROTATION_IN','ROTATION_OUT')):
        x=44+i*395;group=latest.loc[latest.direction.eq(direction)];wrong=group.loc[group.target.eq(0)]
        title='IN: 하락 종료 예측' if i==0 else 'OUT: 상승 종료 예측'
        draw_text(pdf,title,x,h-100,font='Malgun-Bold',size=15)
        draw_text(pdf,f'경보일 {len(group):,} / 라벨 불일치 {len(wrong):,} ({len(wrong)/len(group):.1%})',x,h-130,size=11)
        for j,kind in enumerate(KINDS):
            n=int(wrong.category.eq(kind).sum())
            draw_text(pdf,f'{LABELS[kind]}: {n:,} ({n/len(wrong):.1%})',x,h-167-j*32,size=10)
    foot=[
        '분모는 경보 날짜 수이며 독립 사건 수가 아닙니다. 같은 전환 근처의 반복 신호가 여러 번 집계됩니다.',
        '분류 순서: 전환점 당일·이후 → 20일 종가 방향과 3% 반대 움직임 여부. 서로 중복하지 않습니다.',
        '기존 방향 유지도 경로 전체의 단조로운 상승·하락을 뜻하지 않습니다. 3%는 고정한 설명용 기준입니다.',
        '라벨은 전환점 이전 + 20일 내 확정을 요구합니다. 늦은 경보는 방향이 맞아도 사전 포착에는 실패입니다.',
        '따라서 라벨 불일치를 전부 매매 손실로 해석하면 안 됩니다. 수수료·체결·보유전략은 평가하지 않았습니다.',
    ]
    for i,line in enumerate(foot):draw_text(pdf,line,44,170-i*23,size=9,color=MID_GREY)
    draw_footer(pdf,2,total);pdf.showPage()
    page=2;marks=0;expected=0
    for ticker,name in SECTOR_ETFS.items():
        for recent in (False,True):
            page+=1;start=max(data.date.min(),pd.Timestamp('2025-01-01')) if recent else data.date.min()
            end=data.date.max();series=prices.loc[start:end,ticker].dropna();start=series.index[0]
            series=series/series.iloc[0]*100
            low,high=series.min(),series.max();pad=max((high-low)*.1,1);low-=pad;high+=pad
            sample=data.loc[data.sector.eq(ticker)&data.date.between(start,end)]
            pdf.setFillColor(TEAL);pdf.rect(0,h-55,w,55,fill=1,stroke=0)
            draw_text(pdf,f'{ticker}  {name}',38,h-35,font='Malgun-Bold',size=18,color=PAPER)
            scope='2025년 이후 확대' if recent else '전체 평가 기간'
            draw_text(pdf,f'{scope} | {start:%Y-%m-%d} ~ {end:%Y-%m-%d} | ETF 조정가격, 시작=100',40,h-76,size=10)
            draw_text(pdf,'초록 ▲ IN / 빨강 ▼ OUT | 경보일 모두 표시 | 같은 날짜·같은 가격축 비교',40,h-93,size=9,color=MID_GREY)
            x,pw=64,w-108
            def dx(date):return x+(date-start).days/(end-start).days*pw
            ticks=[start,*pd.date_range(start,end,freq='6MS'),end]
            ticks=sorted(set(t for t in ticks if t in (start,end) or ((t-start).days>65 and (end-t).days>65)))
            for i,(model,label) in enumerate(MODELS.items()):
                y,ph=h-235-i*132,99;chosen=sample.loc[sample[model]>=.5];expected+=len(chosen)
                incoming=int(chosen.direction.eq('ROTATION_IN').sum());outgoing=len(chosen)-incoming
                draw_text(pdf,f'{label} | IN {incoming} / OUT {outgoing}',x,y+ph+9,font='Malgun-Bold',size=10)
                for value in np.linspace(low,high,3):
                    yy=y+(value-low)/(high-low)*ph;pdf.setStrokeColor(GRID);pdf.setLineWidth(.4);pdf.line(x,yy,x+pw,yy)
                    draw_text(pdf,f'{value:.0f}',32,yy-2,size=7,color=MID_GREY)
                path=pdf.beginPath()
                for n,(date,value) in enumerate(series.items()):
                    (path.moveTo if n==0 else path.lineTo)(dx(date),y+(value-low)/(high-low)*ph)
                pdf.setStrokeColor(BLUE);pdf.setLineWidth(.8);pdf.drawPath(path)
                for r in chosen.itertuples():
                    assert r.date in series.index
                    up=r.direction=='ROTATION_IN'
                    draw_triangle(pdf,dx(r.date),y+(series.loc[r.date]-low)/(high-low)*ph,2.5,GREEN if up else RED,up);marks+=1
                for tick in ticks:
                    pdf.setFont('Malgun',7);pdf.setFillColor(MID_GREY);pdf.drawCentredString(dx(tick),y-15,tick.strftime('%Y-%m'))
            draw_text(pdf,'경보 문턱 0.5 고정. 미래 정답으로 신호를 이동·제거하지 않았습니다. 하단 두 패널의 동일함은 실제 결과입니다.',42,51,size=8,color=MID_GREY)
            draw_footer(pdf,page,total);pdf.showPage()
    pdf.save()
    assert marks==expected and page==total and len(PdfReader(PDF).pages)==total
    (DEST/'audit.json').write_text(json.dumps(dict(source=str(SOURCE),pages=page,plotted_markers=marks,
        threshold=.5,path_horizon=20,round_trip_threshold=.03,retraining=False),indent=2),encoding='utf-8')
    print('PASS',PDF,'pages',page,'markers',marks)


if __name__=='__main__':build()
