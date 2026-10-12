"""Visual audit of retrospective absolute-price labels; never present them as predictions."""
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
    register_fonts,draw_text,draw_triangle,draw_footer,TEAL,PAPER,BLUE,GREEN,RED,GRID,MID_GREY,
)

OUT=ROOT/'outputs/absolute_turn_labels_20260921'
PDF=ROOT/'output/pdf/absolute_turn_label_audit_20260921.pdf'


def build():
    if PDF.exists(): raise FileExistsError(PDF)
    labels=pd.read_csv(OUT/'labels.csv',parse_dates=['date','label_end'])
    with sqlite3.connect((ROOT/'regime_router_publish.db').as_uri()+'?mode=ro',uri=True) as c:
        raw=pd.read_sql_query('SELECT date,ticker,adj_close FROM daily_prices',c,parse_dates=['date'])
    raw=raw.loc[raw.ticker.isin(SECTOR_ETFS)]
    assert not raw.duplicated(['ticker','date']).any()
    prices=raw.pivot(index='date',columns='ticker',values='adj_close').sort_index()
    audit=[]
    for ticker in SECTOR_ETFS:
        group=labels.loc[labels.ticker.eq(ticker)].sort_values('date')
        log=np.log(prices[ticker])
        recovery40=prices[ticker].shift(-40)/prices[ticker]-1
        upper20=pd.concat([log.shift(-k)-log for k in range(1,21)],axis=1).max(axis=1)
        lower20=pd.concat([log.shift(-k)-log for k in range(1,21)],axis=1).min(axis=1)
        top=group.loc[group.target.eq('TOP_END')]
        bottom=group.loc[group.target.eq('BOTTOM_END')]
        assert (top.future_return20<0).all() and (bottom.future_return20>0).all()
        assert np.all(lower20.reindex(top.date).to_numpy() <= -top.barrier_log.to_numpy()+1e-10)
        assert np.all(upper20.reindex(bottom.date).to_numpy() >= bottom.barrier_log.to_numpy()-1e-10)
        assert group.tail(20).target.eq('IMMATURE').all()
        recovery=recovery40.reindex(top.date).dropna()
        audit.append(dict(ticker=ticker,start=str(group.date.min().date()),end=str(group.date.max().date()),
                          rows=len(group),top_label_days=len(top),bottom_label_days=len(bottom),
                          top_mature40=len(recovery),top_recovered_at40=int(recovery.ge(0).sum()),
                          recovered_fraction_at40=float(recovery.ge(0).mean()) if len(recovery) else None))
    audit=pd.DataFrame(audit)
    audit.to_csv(OUT/'label_audit.csv',index=False)
    register_fonts()
    pdf=canvas.Canvas(str(PDF),pagesize=landscape(A4))
    pdf.setTitle('11 ETF absolute trend label audit - retrospective, not predictions')
    w,h=landscape(A4)
    draw_text(pdf,'11개 ETF 절대가격 추세 라벨 검수',40,h-60,font='Malgun-Bold',size=25,color=TEAL)
    draw_text(pdf,'미래를 보고 만든 정답 후보입니다. 예측 신호·백테스트 성과가 아닙니다.',40,h-95,font='Malgun-Bold',size=14,color=RED)
    lines=[
        '대상: Market Lens의 기존 11개 GICS Level 1 섹터 ETF',
        f'전체 {len(labels):,}개 날짜-섹터 표본 | 가격 및 라벨 끝: {labels.date.max():%Y-%m-%d}',
        '라벨 가용 시작: 대부분 2016-11-01, XLC 2018-09-14. 과거 결측값은 채우지 않았습니다.',
        '기존 20일·60일 가격 방향이 같을 때만 상승 또는 하락 추세로 분류합니다.',
        '변동성 기준 반대 방향 장벽을 먼저 넘고, 20일 뒤에도 움직임 일부가 남아야 종료 후보입니다.',
        '장벽 최소 약 3% 로그변화; 마지막 날은 절반 이상 유지. 작은 조정도 후보에 포함될 수 있습니다.',
        '빨강 ▼: 상승 종료 후보 / 초록 ▲: 하락 종료 후보. 연속 같은 라벨 구간의 첫날만 표시합니다.',
        '표시 날짜는 분류 대상일이며 실제 고점·저점이나 당시 알 수 있었던 경보일이 아닙니다.',
        '최종 20거래일은 미성숙으로 제외. 가격선은 조정가격이며 각 그래프 시작을 100으로 환산합니다.',
        '하단 회복 진단은 40일 뒤 가격이 분류일 가격 이상인 비율입니다. 중간 회복 여부는 아닙니다.',
        '표본은 서로 겹칩니다. 날짜 수를 독립 전환 횟수나 적중률로 해석하면 안 됩니다.',
        '현재 단계: 라벨 검수만 수행. 모델 학습·웹·운영 DB 변경 없음.',
    ]
    for i,line in enumerate(lines): draw_text(pdf,line,44,h-140-i*25,size=10)
    draw_footer(pdf,1,12);pdf.showPage()
    for page,(ticker,name) in enumerate(SECTOR_ETFS.items(),start=2):
        group=labels.loc[labels.ticker.eq(ticker)].sort_values('date')
        series=prices[ticker]
        # Collapse only consecutive calendar-session runs for display; retain all data in CSV.
        pos=pd.Series(np.arange(len(prices)),index=prices.index)
        positions=pos.reindex(group.date).to_numpy()
        starts=group.target.ne(group.target.shift()) | pd.Series(np.r_[True,np.diff(positions)!=1],index=group.index)
        markers=group.loc[starts & group.target.isin(['TOP_END','BOTTOM_END'])]
        pdf.setFillColor(TEAL);pdf.rect(0,h-55,w,55,fill=1,stroke=0)
        draw_text(pdf,f'{ticker}  {name}',38,h-35,font='Malgun-Bold',size=19,color=PAPER)
        draw_text(pdf,'사후 정답 후보 - 예측 신호 아님 | 빨강 ▼ 상승 종료 / 초록 ▲ 하락 종료',40,h-77,size=10,color=RED)
        for start,y,ph,title in [(group.date.min(),310,165,'전체 라벨 가용 기간'),(pd.Timestamp('2025-01-01'),105,145,'최근 구간 확대')]:
            end=series.last_valid_index()
            view=series.loc[start:end].dropna()
            start=view.index[0]
            view=view/view.iloc[0]*100
            low,high=view.min(),view.max(); pad=max((high-low)*.1,1);low-=pad;high+=pad
            x,pw=65,w-108
            def dx(date):return x+(date-start).days/(end-start).days*pw
            draw_text(pdf,f'{title} | {start:%Y-%m-%d} = 100',x,y+ph+10,font='Malgun-Bold',size=10)
            for v in np.linspace(low,high,4):
                yy=y+(v-low)/(high-low)*ph
                pdf.setStrokeColor(GRID);pdf.setLineWidth(.4);pdf.line(x,yy,x+pw,yy)
                draw_text(pdf,f'{v:.0f}',32,yy-2,size=7,color=MID_GREY)
            path=pdf.beginPath()
            previous=None
            for date,value in view.items():
                point=(dx(date),y+(value-low)/(high-low)*ph)
                if previous is None or pos.loc[date]-pos.loc[previous]>1:path.moveTo(*point)
                else:path.lineTo(*point)
                previous=date
            pdf.setStrokeColor(BLUE);pdf.setLineWidth(1);pdf.drawPath(path)
            for r in markers.loc[markers.date.between(start,end)].itertuples():
                up=r.target=='BOTTOM_END'
                draw_triangle(pdf,dx(r.date),y+(view.loc[r.date]-low)/(high-low)*ph,3.5,GREEN if up else RED,up)
            ticks=[start,*pd.date_range(start,end,freq='2YS' if title.startswith('전체') else '6MS'),end]
            ticks=sorted(set(t for t in ticks if t in (start,end) or ((t-start).days>70 and (end-t).days>70)))
            for t in ticks:
                pdf.setFont('Malgun',7);pdf.setFillColor(MID_GREY);pdf.drawCentredString(dx(t),y-15,t.strftime('%Y-%m'))
        a=audit.loc[audit.ticker.eq(ticker)].iloc[0]
        draw_text(pdf,f"상승 종료 후보 {a.top_label_days}일 중 40일 결과가 있는 {a.top_mature40}일: {a.top_recovered_at40}일({a.recovered_fraction_at40:.1%})은 가격 회복",42,65,size=9)
        draw_text(pdf,'주의: 회복한 사례는 단기 조정일 수 있습니다. 장기 추세 종료 라벨로 채택하기 전 추가 검토가 필요합니다.',42,49,size=8,color=MID_GREY)
        draw_footer(pdf,page,12);pdf.showPage()
    pdf.save()
    assert len(PdfReader(PDF).pages)==12
    total=int(audit.top_mature40.sum());recovered=int(audit.top_recovered_at40.sum())
    summary={'rows':len(labels),'sectors':len(audit),'pages':12,'checks':'PASS: 20D final sign, barrier hit, terminal maturity',
             'top_days_mature40':total,'top_days_recovered_at40':recovered,'recovered_fraction_at40':recovered/total,
             'status':'LABEL_REVIEW_REQUIRED_NOT_READY_FOR_TRAINING','warning':'retrospective overlapping sample days, not model signals or independent events'}
    (OUT/'visual_audit_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary));print(PDF)


if __name__=='__main__': build()
