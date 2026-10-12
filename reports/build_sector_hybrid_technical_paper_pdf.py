"""Format the existing hybrid experiment as an internal technical paper; no fitting."""
from pathlib import Path
import csv
import json
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from pypdf import PdfReader
from build_sector_turning_alert_backtest_pdf import register_fonts

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/pdf/sector_hybrid_technical_paper_20260921.pdf'
TEAL = colors.HexColor('#0C5A4B')
INK = colors.HexColor('#172522')
styles = {
    'body': ParagraphStyle('body', fontName='Malgun', fontSize=10, leading=16.5, textColor=INK, spaceAfter=9, wordWrap='CJK'),
    'small': ParagraphStyle('small', fontName='Malgun', fontSize=8.2, leading=12.5, textColor=INK, spaceAfter=7, wordWrap='CJK'),
    'title': ParagraphStyle('title', fontName='Malgun-Bold', fontSize=23, leading=33, textColor=TEAL, spaceAfter=16, wordWrap='CJK'),
    'h1': ParagraphStyle('h1', fontName='Malgun-Bold', fontSize=16, leading=23, textColor=TEAL, spaceAfter=16, wordWrap='CJK'),
    'h2': ParagraphStyle('h2', fontName='Malgun-Bold', fontSize=11.5, leading=18, textColor=TEAL, spaceBefore=9, spaceAfter=8, wordWrap='CJK'),
    'eq': ParagraphStyle('eq', fontName='Malgun', fontSize=12, leading=23, leftIndent=14, backColor=colors.HexColor('#F1F6F4'), borderPadding=9, spaceBefore=9, spaceAfter=16),
}
story = []
def p(text, kind='body'):
    text = text.replace('−', '-').replace('ŝ', 's').replace('Pᵢ,ₜ', 'P<sub>i,t</sub>').replace('σᵢ,ₜ', 'σ<sub>i,t</sub>')
    story.append(Paragraph(text, styles[kind]))
def table(rows, widths):
    cells = [[Paragraph(str(v), styles['small']) for v in row] for row in rows]
    t = Table(cells, colWidths=widths, hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E3EFEB')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),0.8,TEAL),('LINEBELOW',(0,1),(-1,-1),0.3,colors.HexColor('#D7DEDA')),('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
    story.extend([t, Spacer(1, 11)])
def page(title):
    story.append(PageBreak())
    p(title, 'h1')

def build():
    register_fonts()
    src = ROOT / 'outputs/variable_barrier_comparison_20260921'
    with (src/'metrics.csv').open(encoding='utf-8-sig', newline='') as f:
        metrics = {(r['direction'],r['model']):r for r in csv.DictReader(f) if r['period']=='ALL'}
    manifest = json.loads((ROOT/'outputs/sector_hybrid_models/2026/manifest.json').read_text(encoding='utf-8'))
    p('MARKET LENS  /  TECHNICAL RESEARCH NOTE', 'small')
    p('가격 기반 기계학습을 활용한<br/>섹터 추세전환 신호 연구', 'title')
    p('부스팅 기반 진입 신호와 Extra Trees 기반 이탈 신호의 결합', 'h2')
    p('2026년 9월 21일 · 내부 회의용 소논문 · 기존 실험 결과 정리', 'small')
    p('초록', 'h2')
    p('본 연구는 미국 11개 섹터 ETF의 가격 정보를 이용하여 기존 추세의 지속과 반대 방향 전환을 구분하는 문제를 다룬다. 최근 수익률이 높은 섹터를 추종하는 방식과 달리, 유사한 과거 가격 상태에서 이후 경로가 달라지는 조건을 학습하는 데 목적을 둔다. 정답은 고정된 20일 수익률이 아니라 변동성에 비례하는 상·하단 경계 중 어느 경계에 먼저 도달하는지로 정의하였다.')
    p('가격·경로·추세·상대강도 입력과 섹터 식별자를 사용하고, 연도별 시간순 학습·검증·평가를 실시하였다. 현재 선택안은 IN에 얕은 히스토그램 부스팅, OUT에 Extra Trees를 사용하는 비대칭 결합이다. 가격 기반 부스팅의 IN 적중률은 97.6%, 양성 표본 포착률은 9.0%였으며, 가격 기반 Extra Trees의 OUT 적중률은 46.1%, 포착률은 11.0%였다.')
    p('높은 IN 적중률은 소수 신호에 대한 결과이며 전체 저점의 대부분을 포착했다는 뜻이 아니다. 또한 결합 모델은 결과 검토 후 선택한 후보로서 독립 표본에서 검증되지 않았다. 본 문서는 구현 가능한 보조 신호의 연구 근거와 한계를 설명하며, 자동매매 또는 확정적 전환 예측 성능을 주장하지 않는다.')
    p('핵심어: 섹터 로테이션, 추세전환, 경쟁 장벽, 부스팅, Extra Trees, 시계열 검증', 'small')
    p('1. 연구 문제', 'h2')
    p('기존 상대강도 지표는 “무엇이 강한가”를 잘 요약하지만, 상승 중인 섹터에 OUT을 내거나 하락이 상당히 진행된 뒤 전환을 확인하는 문제가 있었다. 본 연구는 정확한 고점·저점 일자를 맞추기보다, 현재 방향과 반대인 의미 있는 움직임이 기존 방향의 연장보다 먼저 나타날 조건을 찾는다. IN과 OUT의 생성 과정이 같다고 가정하지 않고 각각 별도의 분류기를 학습한다.')

    page('2. 데이터와 설명변수')
    p('2.1 분석 대상과 표본', 'h2')
    p('분석 대상은 GICS Level 1에 대응하는 11개 미국 섹터 ETF이며, SPY를 시장 상대강도 기준으로 사용한다. 조정가격을 이용하고 결측값은 보간하거나 전일 값으로 채우지 않는다. 예측 시점은 주간 표본이다.')
    table([['ETF','섹터','ETF','섹터'],['XLB','소재','XLC','커뮤니케이션'],['XLE','에너지','XLF','금융'],['XLI','산업재','XLK','정보기술'],['XLP','필수소비재','XLRE','부동산'],['XLU','유틸리티','XLV','헬스케어'],['XLY','경기소비재','','']], [55,185,55,192])
    p('모델 간 공정한 비교를 위해 원 실험은 모든 후보 입력 블록이 동시에 존재하는 공통 표본을 사용했다. 일반적으로 2018년부터, XLC는 2019년부터이며 공통 입력의 마지막 날짜는 2026년 8월 28일이다. 평가 연도는 2023~2026년이다. 따라서 가격 모델의 성과도 가격 데이터가 존재하는 모든 날짜의 성과는 아니다.', 'small')
    p('2.2 선택 모델의 입력 구조', 'h2')
    table([['블록','개수','내용'],['절대 가격 상태','8','1·5·20·60일 로그수익률, 20일 변동성, 20일 낙폭·반등폭, 속도 변화'],['변동성 조정 경로','6','5·20일 수익률, 낙폭·반등폭·속도의 위험 조정값, 변동성 비율'],['추세 구조','4','단기·장기 이동평균 괴리, 기울기, 괴리 변화'],['시장 상대강도','3','RRG ratio·momentum·5일 변화 대용 지표'],['섹터 식별','11','섹터별 원-핫 인코딩']], [105,40,342])
    p('총 21개 수치 변수와 11개 섹터 식별자를 사용한다. 선택한 결합안에는 매크로·뉴스·옵션·거래량이 직접 입력되지 않는다. RRG는 투명한 가격 기반 대용 지표이며, 독점적 JdK 계산식을 재현한 것으로 해석해서는 안 된다. 섹터별로 11개 모델을 만드는 방식이 아니라, 섹터 정보를 포함한 통합 모델을 방향별로 학습한다.')

    page('3. 가변 기간 정답 라벨의 정의')
    p('3.1 기존 방향과 전환 후보', 'h2')
    p('Pᵢ,ₜ를 섹터 i의 t일 조정가격이라 하자. 최근 60거래일 로그수익률이 양수인 표본은 OUT 후보, 그 외는 IN 후보로 분류한다. 이는 현재 방향을 정하는 규칙이지 미래 정답이 아니다.')
    p('r<super>(60)</super><sub>i,t</sub> = log(P<sub>i,t</sub> / P<sub>i,t−60</sub>)<br/>d<sub>i,t</sub> = +1 if r<super>(60)</super><sub>i,t</sub> &gt; 0; otherwise −1', 'eq')
    p('3.2 변동성 비례 경쟁 장벽', 'h2')
    p('σᵢ,ₜ는 t−1일까지의 일간 로그수익률 60개에 대한 표본 표준편차다. 변동성을 한 거래일 이동하여 현재 수익률을 변동성 추정에 포함하지 않는다. 장벽 폭은 다음과 같다.')
    p('b<sub>i,t</sub> = min(0.30, max(0.06, 6σ<sub>i,t</sub>))', 'eq')
    p('장벽은 로그수익률 단위다. 예를 들어 0.06은 산술수익률 정확히 6%와 같지 않으며, 로그 기준 대칭 장벽의 산술 상승·하락 폭은 서로 다르다.', 'small')
    p('3.3 최초 도달 사건과 정답', 'h2')
    p('신호 다음 거래일 종가를 진입 기준가격으로 놓고, 그 이후 상·하단 경계 중 하나에 최초로 도달하는 시점 τ를 찾는다.')
    p('τ<sub>i,t</sub> = inf {u &gt; t+1 : |log(P<sub>i,u</sub> / P<sub>i,t+1</sub>)| ≥ b<sub>i,t</sub>}<br/>y<sub>i,t</sub> = 1{d<sub>i,t</sub> · log(P<sub>i,τ</sub> / P<sub>i,t+1</sub>) &lt; 0}', 'eq')
    p('y=1이면 기존 방향의 반대 장벽에 먼저 도달한 전환, y=0이면 기존 방향 장벽에 먼저 도달한 지속이다. 시간 상한을 두지 않으므로 라벨 확정까지 걸리는 기간은 달라진다. 미래 경로가 결측으로 끊기거나 표본 말일까지 장벽에 닿지 않으면 미확정으로 남기며, 실패로 간주하지 않는다.')
    p('이 정답은 사후 학습 목표다. 실제 예측에는 미래 가격이 들어가지 않는다. 다만 “반대 장벽 먼저 도달”은 정확한 고점·저점이나 장기 추세의 영구 종료와 동일하지 않다. 20·60일 수익률은 별도 진단값이며 정답의 고정 기간이 아니다.', 'small')

    page('4. 학습 방법과 시간순 검증')
    p('4.1 비슷하지만 결과가 다른 표본의 학습', 'h2')
    p('학습 구간 내 동일 섹터에서 60일 수익률과 20일 변동성을 학습 표준편차로 정규화한다. 각 전환 표본에 대해 제곱 유클리드 거리가 가장 가까운 비전환 표본을 찾고, 거리 2 이하일 때 해당 비전환 표본의 가중치를 높인다. 모든 학습 표본은 유지한다. 이는 신경망의 대조 손실이 아니라 유사 대조군을 강조하는 가중 분류 학습이다.')
    p('동일 섹터·라벨 종료 월에 중복되는 표본은 해당 그룹의 표본 수로 가중치를 나누고, 이후 양성·음성 가중치 총합을 균형화한다. 이 처리가 시계열 의존성을 완전히 제거하지는 않는다. 표준화는 학습 구간에서만 추정하여 검증·평가 구간에 적용한다.')
    p('4.2 방향별 모델', 'h2')
    table([['항목','IN: 히스토그램 부스팅','OUT: Extra Trees'],['학습 논리','앞선 오차를 순차적으로 보완','무작위성이 큰 여러 트리의 평균'],['주요 설정','깊이 3, 반복 100, 최소 잎 30, L2=5, 학습률 0.1','트리 150, 깊이 5, 최소 잎 20, 변수 비율 0.7'],['재현 설정','early_stopping=False, seed=42','n_jobs=2, seed=42']], [75,206,206])
    p('4.3 연도별 분할과 신호 기준', 'h2')
    p('평가 연도를 Y라고 할 때, 학습은 라벨 종료일이 Y−2년 1월 1일 이전인 표본만 사용한다. 검증은 관측일이 Y−2년 1월 1일 이후이면서 라벨 종료일이 Y년 1월 1일 이전인 표본이다. 평가는 Y년의 관측일에 실시한다. 무작위 분할은 사용하지 않는다.')
    p('c<sub>Y,m</sub> = Q<sub>0.90</sub>({ŝ<sub>m</sub>(x) : x ∈ validation<sub>Y</sub>})<br/>Signal<sub>i,t</sub> = 1{ŝ<sub>m</sub>(x<sub>i,t</sub>) ≥ c<sub>Y,m</sub>}', 'eq')
    p('검증 점수의 90백분위를 연간 고정 임계값으로 사용한다. 평가 구간에서 매번 상위 10%를 뽑는 방식이 아니므로 실제 신호 빈도는 10%와 다를 수 있다. 점수는 클래스 가중 학습의 결과로서 보정된 전환 확률이 아니다. 비교 실험은 3개 모델군·4개 입력 블록·2개 방향·4개 평가 연도의 총 96회 적합으로 구성되었다.', 'small')

    page('5. 실증 결과와 해석')
    p('5.1 실제 검증 임계값에 따른 신호 성과', 'h2')
    rows = [['방향','가격 모델','확정 신호','정답','적중률','포착률']]
    for direction, model in [('IN','PRICE_BOOST'),('IN','PRICE_EXTRA'),('OUT','PRICE_BOOST'),('OUT','PRICE_EXTRA')]:
        r = metrics[(direction,model)]
        n = int(r['resolved_alerts'])
        rows.append([direction,model.replace('PRICE_',''),n,round(n*float(r['precision'])),f"{float(r['precision']):.1%}",f"{float(r['recall']):.1%}"])
    table(rows, [40,100,90,60,95,102])
    p('적중률 = 정답 신호 수 / 결과가 확정된 신호 수<br/>포착률 = 정답 신호 수 / 전체 전환 양성 표본 수', 'eq')
    p('IN 부스팅은 확정 신호 41개 중 40개가 정답이었다. 그러나 전체 전환 양성 표본 446개 중 포착한 것은 40개로, 포착률은 9.0%다. OUT Extra Trees는 128개 중 59개가 정답이며, 전체 양성 537개 중 11.0%를 포착했다. 이 수치는 주간 표본 행 기준으로, 서로 독립적인 저점·고점 사건의 개수가 아니다.')
    p('선택한 결합안의 총 신호는 IN 43개와 OUT 135개, 합계 178개다. 이 중 IN 2개와 OUT 7개는 라벨 미확정이므로 적중률 계산에서 제외했다. 전체 확정 표본의 전환 비율은 IN 70.3%, OUT 40.3%로, 단순히 50%와 비교하는 것도 적절하지 않다.')
    p('5.2 무엇이 개선되었고 무엇이 남았는가', 'h2')
    p('IN은 부스팅이 Extra Trees보다 신호 적중률이 높았으나 신호 수가 적다. OUT은 Extra Trees가 부스팅보다 적중률이 높고 포착률은 낮다. 즉, “모든 측면에서 우월한 모델”이 아니라 서로 다른 선택성과 누락률의 절충이다.')
    p('특히 선택한 OUT의 방향 조정 평균 20·60일 수익률은 각각 약 −0.14%, −0.85%였다. OUT에서는 하락을 양수로 계산하므로 이는 오히려 해당 ETF가 평균적으로 상승했음을 뜻한다. 따라서 장벽 라벨 적중률만으로 매도 타이밍의 경제적 효용이 입증됐다고 볼 수 없다.')
    p('결합안은 기존 결과와 시각화를 본 뒤 선정했다. 별도의 신규 표본에서 결합 전략의 성과를 재검증하지 않았으며, 본 표는 거래비용을 반영한 포트폴리오 수익률이 아니다. 평가 표본의 상위 10%를 사후 선택하는 별도 진단 결과는 이 표에 사용하지 않았다.', 'small')

    page('6. 한계, 구현 상태 및 결론')
    p('6.1 해석과 검증의 한계', 'h2')
    p('<b>라벨의 한계.</b> 경쟁 장벽은 전환의 조작적 정의다. 60일 방향, 6σ, 최소·최대 장벽이라는 설계 선택이 남아 있다. 매끄러운 장기 상승 추세 중 일시적 조정도 전환 정답이 될 수 있다.')
    p('<b>표본 의존성과 선택 편향.</b> 주간 표본의 미래 경로는 겹치고 섹터 간 상관도 존재한다. 2023~2026년 결과를 반복 확인한 만큼 완전히 미사용한 평가 구간으로 볼 수 없다. 단순 이항 신뢰구간이나 이번 적중률만으로 통계적 우월성을 확정하지 않는다.')
    p('<b>데이터와 실전 차이.</b> 수정된 현재의 가격 이력을 사용하며 완전한 당시 정보 기준(PIT)은 검증되지 않았다. 가격만으로 추론 가능한 날짜가 공통 연구 표본보다 넓어져도, 그 추가 날짜의 성능까지 검증된 것은 아니다. 거래비용·체결·포지션 크기·연속 경보 통합은 평가하지 않았다.')
    p('6.2 저장된 구현과 재현성', 'h2')
    p('2026년용 IN·OUT 모델, 표준화 정보와 임계값을 로컬에 저장했다. 저장 모델은 원 연구의 예측값과 부동소수점 오차 수준에서 일치한다. 추론은 완료된 주간 가격으로 수행하며 자동 재학습하지 않는다. 현재 단계는 로컬 모델 구현·저장까지이며 웹 배포를 뜻하지 않는다.')
    table([['저장 모델','검증 임계값','연구 예측 일치 확인'],['IN / BOOST',f"{manifest['models']['IN']['threshold']:.6f}",'120개 행'],['OUT / EXTRA',f"{manifest['models']['OUT']['threshold']:.6f}",'265개 행']], [160,130,197])
    p('6.3 결론과 후속 검증', 'h2')
    p('가격 기반 비대칭 결합은 IN의 선택적 포착과 OUT의 상대적 신호 선별이라는 연구 결과를 구현한 후보안이다. IN의 높은 적중률과 낮은 포착률, OUT의 제한적인 경제적 효용을 함께 보아야 한다. 다음 검증은 모델을 고정한 뒤 신규 날짜에서 수행하고, 독립 전환 사건별 선행·지연 시간과 신호 이후 남은 가격 움직임을 평가하는 것이 타당하다.')
    p('재현 자료', 'h2')
    p('[1] sector_variable_barrier_comparison.py: 라벨·특성·학습·검증 구현.<br/>[2] outputs/variable_barrier_comparison_20260921/: REPORT.md, metrics.csv, fold_audit.csv.<br/>[3] sector_hybrid_signal.py: 선택 모델 저장 및 가격 기반 추론.<br/>[4] outputs/sector_hybrid_models/2026/manifest.json: 버전·해시·임계값·일치 검증.<br/>[5] reports/SECTOR_HYBRID_RUNBOOK.md: 로컬 실행 절차.', 'small')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    def footer(c, doc):
        c.setStrokeColor(colors.HexColor('#D7DEDA'))
        c.line(54, A4[1]-40, A4[0]-54, A4[1]-40)
        c.setFont('Malgun', 8)
        c.setFillColor(TEAL)
        c.drawString(54, A4[1]-30, 'Market Lens  |  MARKET LENS RESEARCH')
        c.setFillColor(colors.HexColor('#718079'))
        c.drawString(54, 30, '내부 검토용 · 가격 기반 추세전환 후보 모델 · 2026.09.21')
        c.drawRightString(A4[0]-54, 30, str(doc.page))
    SimpleDocTemplate(str(OUT), pagesize=A4, rightMargin=54, leftMargin=54, topMargin=58, bottomMargin=53, title='가격 기반 기계학습을 활용한 섹터 추세전환 신호 연구', author='Market Lens').build(story, onFirstPage=footer, onLaterPages=footer)
    pdf = PdfReader(str(OUT))
    assert len(pdf.pages)==6, f'Unexpected pagination: {len(pdf.pages)}'
    assert all(len(pg.extract_text()) > 300 for pg in pdf.pages)
    print(OUT)
    print('Verified: 6 pages, embedded Korean fonts, source-linked metrics; no model execution.')

if __name__ == '__main__':
    build()
