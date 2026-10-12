"""Audit the saved 11-ETF RRG table and plot fixed rejection filters. Read-only DB."""
from pathlib import Path
import json
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from pypdf import PdfReader
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfgen import canvas
from sector_rotation import SECTOR_ETFS, build_rrg_signals
from build_sector_turning_alert_backtest_pdf import (
    register_fonts, draw_text, draw_triangle, draw_footer, TEAL, PAPER,
    BLUE, GREEN, RED, GRID, MID_GREY,
)

SOURCE = ROOT / 'outputs/refresh_audit_20260910/sector_history_corrected.json'
OUT = ROOT / 'outputs/lv1_rrg_noise_20260921'
PDF = ROOT / 'output/pdf/lv1_rrg_noise_comparison_20260921.pdf'
POLICIES = {'BASE': '기존 RRG IN / OUT', 'PERSIST': '3거래일 방향 유지',
            'MARGIN': 'RRG 경계 이탈 폭 0.25 이상', 'ALPHA': '20D Alpha 방향 동의',
            'FIT': 'Market Fit 방향 동의', 'COMBINED': '방향 유지 + 이탈 폭 + Alpha'}


def event(transition):
    before, after = transition.split('->')
    strong, weak = {'LEADING', 'IMPROVING'}, {'LAGGING', 'WEAKENING'}
    return 1 if before in weak and after in strong else -1 if before in strong and after in weak else 0


def load():
    h = pd.DataFrame(json.loads(SOURCE.read_text(encoding='utf-8'))['rows'])
    h['date'] = pd.to_datetime(h.decision_date)
    h = h.rename(columns={'sector': 'ticker'})
    assert set(h.ticker) == set(SECTOR_ETFS)
    assert not h.duplicated(['date', 'ticker']).any()
    assert h.groupby('date').ticker.nunique().eq(11).all()
    db = ROOT / 'regime_router_publish.db'
    with sqlite3.connect(db.as_uri()+'?mode=ro', uri=True) as c:
        p = pd.read_sql_query("SELECT date,ticker,adj_close FROM daily_prices WHERE ticker IN ('SPY','XLB','XLC','XLE','XLF','XLI','XLK','XLP','XLRE','XLU','XLV','XLY')", c, parse_dates=['date'])
    if p.duplicated(['date','ticker']).any():
        raise ValueError('duplicate adjusted price keys')
    prices = p.pivot(index='date',columns='ticker',values='adj_close').sort_index()
    if prices.loc[h.date.min():].isna().any().any() or (prices.dropna()<=0).any().any():
        raise ValueError('missing/nonpositive adjusted prices; no filling permitted')
    s = build_rrg_signals(prices).sort_values(['ticker','date'])
    g = s.groupby('ticker',sort=False)
    s['replay_transition'] = g.quadrant.shift(5).fillna('WARMUP')+'->'+s.quadrant
    s['strong'] = s.rs_momentum.ge(100)
    s['strong3'] = s.groupby('ticker').strong.transform(lambda x:x.rolling(3,min_periods=3).sum())
    s['flip5'] = s.groupby('ticker').strong.transform(lambda x: pd.concat([x.shift(-k).ne(x) for k in range(1,6)],axis=1).any(axis=1))
    s['flip_mature'] = s.groupby('ticker').date.shift(-5).notna()
    h = h.merge(s[['date','ticker','rs_momentum','strong3','replay_transition','flip5','flip_mature']],on=['date','ticker'],validate='one_to_one')
    mismatches = int(h.quadrant_transition.ne(h.replay_transition).sum())
    if mismatches:
        raise ValueError(f'{mismatches} saved/recomputed RRG transitions differ: cannot mix vintages')
    h['direction'] = h.quadrant_transition.map(event)
    h['BASE'] = h.direction.ne(0)
    h['PERSIST'] = h.BASE & np.where(h.direction.eq(1),h.strong3.eq(3),h.strong3.eq(0))
    h['MARGIN'] = h.BASE & ((h.rs_momentum-100)*h.direction).ge(.25)
    h['ALPHA'] = h.BASE & (h.predicted_excess_20d*h.direction).gt(0)
    h['fit_available'] = h.roro_context.isin(['FIT','RORO_MISMATCH'])
    h['FIT'] = h.BASE & np.where(h.direction.eq(1),h.roro_context.eq('FIT'),h.roro_context.eq('RORO_MISMATCH'))
    h['COMBINED'] = h.PERSIST & h.MARGIN & h.ALPHA
    assert all((~h[key] | h.BASE).all() for key in POLICIES)
    forward = prices.shift(-21).div(prices.shift(-1))-1
    for ticker in SECTOR_ETFS:
        idx = h.index[h.ticker.eq(ticker)]
        dates = h.loc[idx,'date']
        h.loc[idx,'forward_absolute'] = forward[ticker].reindex(dates).to_numpy()
        h.loc[idx,'forward_spy_excess'] = (forward[ticker]-forward.SPY).reindex(dates).to_numpy()
        h.loc[idx,'forward_equal_excess'] = (forward[ticker]-forward[list(SECTOR_ETFS)].mean(axis=1)).reindex(dates).to_numpy()
    h['signed_excess'] = h.forward_spy_excess*h.direction
    return h, prices, mismatches


def error_delta_ci(base, policy):
    # Paired 8-week moving-block bootstrap; preserve cross-sector dependence.
    if base.empty:
        return [None,None]
    q = base.assign(bad=base.signed_excess.le(0).astype(int))
    q['date'] = q.date.dt.to_period('W-FRI').dt.end_time.dt.normalize()
    dates = pd.date_range(q.date.min(),q.date.max(),freq='W-FRI')
    if len(dates)<8:
        return [None,None]
    a = q.groupby('date').agg(n=('bad','size'),bad=('bad','sum')).reindex(dates,fill_value=0)
    b = q.loc[q[policy]].groupby('date').agg(n=('bad','size'),bad=('bad','sum')).reindex(dates,fill_value=0)
    rng=np.random.default_rng(42)
    deltas=[]
    for _ in range(1000):
        starts=rng.integers(0,len(dates)-7,size=int(np.ceil(len(dates)/8)))
        ix=np.concatenate([np.arange(k,k+8) for k in starts])[:len(dates)]
        aa,bb=a.iloc[ix].sum(),b.iloc[ix].sum()
        if bb['n'] and aa['n']:
            deltas.append(float(bb.bad/bb['n']-aa.bad/aa['n']))
    return np.quantile(deltas,[.025,.975]).tolist() if len(deltas)>=900 else [None,None]


def evaluate(h):
    records=[]
    for direction in (1,-1):
        for period in ('ALL','2025','2026'):
            universe=h.loc[h.direction.eq(direction)&h.signed_excess.notna()]
            if period!='ALL':
                universe=universe.loc[universe.date.dt.year.eq(int(period))]
            for policy in POLICIES:
                base=universe.loc[universe.fit_available] if policy=='FIT' else universe
                selected=base.loc[base[policy]]
                hits=int(base.signed_excess.gt(0).sum())
                low,high=error_delta_ci(base,policy) if period=='ALL' else (None,None)
                records.append(dict(direction='IN' if direction==1 else 'OUT',period=period,policy=policy,
                    n=len(selected),baseline_n=len(base),
                    error_rate=float(selected.signed_excess.le(0).mean()) if len(selected) else None,
                    baseline_error=float(base.signed_excess.le(0).mean()) if len(base) else None,
                    error_delta_ci_low=low,error_delta_ci_high=high,
                    true_signal_retention=float(selected.signed_excess.gt(0).sum()/hits) if hits else None,
                    mean_signed_spy_excess=float(selected.signed_excess.mean()) if len(selected) else None,
                    mean_signed_equal_excess=float((selected.forward_equal_excess*selected.direction).mean()) if len(selected) else None,
                    absolute_direction_hit=float((selected.forward_absolute*selected.direction).gt(0).mean()) if len(selected) else None,
                    flip_within_5_rate=float(selected.loc[selected.flip_mature,'flip5'].mean()) if len(selected) else None))
    return pd.DataFrame(records)


def render(h,prices):
    register_fonts()
    pdf=canvas.Canvas(str(PDF),pagesize=landscape(A4))
    pdf.setTitle('Market Lens 11-sector RRG noise-filter comparison')
    width,height=landscape(A4)
    start,end=h.date.min(),prices.index.max()
    page=0
    for ticker,name in SECTOR_ETFS.items():
        for relative in (False,True):
            page+=1
            price=prices.loc[start:end,ticker]
            if relative:
                price=price/prices.loc[start:end,'SPY']
            price=price/price.iloc[0]*100
            low,high=price.min(),price.max()
            pad=max((high-low)*.1,1)
            low,high=low-pad,high+pad
            pdf.setFillColor(TEAL);pdf.rect(0,height-52,width,52,fill=1,stroke=0)
            draw_text(pdf,f'{ticker}  {name}',38,height-33,font='Malgun-Bold',size=17,color=PAPER)
            mode='SPY 대비 상대강도' if relative else 'ETF 조정가격'
            draw_text(pdf,f'{mode} | {start:%Y-%m-%d}=100 | 초록 ▲ IN / 빨강 ▼ OUT',40,height-70,size=9)
            draw_text(pdf,f'저장된 주간 표: {start:%Y-%m-%d} ~ {h.date.max():%Y-%m-%d} | 가격: {end:%Y-%m-%d} | 성패·고저점 표시는 없음',40,height-84,size=8,color=MID_GREY)
            x,w=62,width-104
            def dx(date): return x+(pd.Timestamp(date)-start).days/(end-start).days*w
            ticks=[start,*pd.date_range(start,end,freq='6MS').tolist(),end]
            ticks=sorted(set(t for t in ticks if t in (start,end) or ((t-start).days>60 and (end-t).days>60)))
            for i,(policy,label) in enumerate(POLICIES.items()):
                y,ph=height-154-i*73,39
                draw_text(pdf,label,x,y+ph+6,font='Malgun-Bold',size=8.5)
                for v in (low,(low+high)/2,high):
                    yy=y+(v-low)/(high-low)*ph
                    pdf.setStrokeColor(GRID);pdf.setLineWidth(.35);pdf.line(x,yy,x+w,yy)
                    draw_text(pdf,f'{v:.0f}',33,yy-2,size=6,color=MID_GREY)
                path=pdf.beginPath()
                for n,(date,value) in enumerate(price.items()):
                    (path.moveTo if n==0 else path.lineTo)(dx(date),y+(value-low)/(high-low)*ph)
                pdf.setStrokeColor(BLUE);pdf.setLineWidth(.8);pdf.drawPath(path)
                signals=h.loc[h.ticker.eq(ticker)&h[policy]]
                for r in signals.itertuples():
                    draw_triangle(pdf,dx(r.date),y+(price.loc[r.date]-low)/(high-low)*ph,3,GREEN if r.direction==1 else RED,r.direction==1)
                if policy=='FIT':
                    available=h.loc[h.ticker.eq(ticker)&h.fit_available,'date']
                    if len(available):
                        draw_text(pdf,f'Fit 자료 시작: {available.min():%Y-%m-%d}; 이전은 N/A',x+w-225,y+ph+6,size=6.5,color=MID_GREY)
                for t in ticks:
                    pdf.setFont('Malgun',6.5);pdf.setFillColor(MID_GREY);pdf.drawCentredString(dx(t),y-11,t.strftime('%Y-%m'))
            draw_footer(pdf,page,22);pdf.showPage()
    pdf.save()
    assert len(PdfReader(PDF).pages)==22


def self_test():
    assert event('LAGGING->IMPROVING')==1
    assert event('LEADING->LAGGING')==-1
    assert event('WEAKENING->LAGGING')==0
    assert event('IMPROVING->LEADING')==0
    # Matches the UI: momentum side, not ratio side, determines IN/OUT.
    assert event('WEAKENING->LEADING')==1
    print('self-test PASS')


if __name__=='__main__':
    self_test()
    if '--self-test' not in sys.argv:
        if OUT.exists() or PDF.exists(): raise FileExistsError('output exists')
        h,prices,mismatches=load()
        print(f'Validated 11 ETFs, {len(h)} rows, replay mismatches={mismatches}',flush=True)
        results=evaluate(h)
        OUT.mkdir(parents=True)
        h.to_csv(OUT/'signals.csv',index=False)
        results.to_csv(OUT/'comparison.csv',index=False)
        meta={'source':str(SOURCE),'first':str(h.date.min().date()),'last':str(h.date.max().date()),
              'sectors':list(SECTOR_ETFS),'replay_mismatches':mismatches,
              'outcome':'signed sector minus SPY simple return, close t+1 to close t+21; <=0 is wrong direction, NOT pivot detection accuracy',
              'filters':'fixed beforehand; no sweep, no winner selection; FIT evaluated only where exact historical context exists',
              'limitations':['Previously reviewed history, not fresh OOS','Current adjusted prices are not historical vintages',
                            '20D Alpha targets equal-sector excess, not SPY excess; both evaluated',
                            'Error CI: paired 8-week block bootstrap, 1000 samples, not multiple-test adjusted',
                            'No costs or trade execution model; OUT is not a short recommendation',
                            'Recent immature 20-day outcomes excluded, signals retained in charts'],
              'web_db_changed':False}
        (OUT/'summary.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
        render(h,prices)
        print(results.loc[results.period.eq('ALL')].to_string(index=False))
        print(PDF)
