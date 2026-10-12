# Market Lens Sector RORO Context v1

Status: **FROZEN RESEARCH SHADOW — NOT LIVE**  
Frozen date: 2026-08-13  
Model version: `sector-roro-context-v1-spy-rel-ridge-a10-m156-w20`  
Signal start: 2026-01-07  
Relative-strength benchmark: `SPY`  
Market-risk context: `RORO-3AXIS-v1` based on `NQ=F`

이 모델은 기존 Sector Rotation을 웹 운영모델에 덮어쓰지 않는다. RRG 이동경로,
20거래일 상대 alpha, NQ 중심 RORO와의 적합도를 서로 다른 열로 유지하고 마지막
상태표시에서만 결합한다. HMM 확률은 입력하지 않는다.

## 1. 20D 상대 alpha

예측 대상은 각 섹터 ETF의 `t+1..t+20` 수익률에서 같은 날 11개 섹터 ETF의
동일가중 평균수익률을 뺀 값이다. 신호는 주간 마지막 거래일 종가 후 계산한다.

입력은 아래 11개로 고정한다.

1. RRG ratio와 momentum
2. RRG ratio 5·20일 변화
3. RRG momentum 5·20일 변화
4. SPY 대비 20·60·120일 상대수익률
5. SPY 대비 20일 상대변동성
6. SPY 대비 60일 상대낙폭

모델은 train-only `StandardScaler + Ridge(alpha=10)`이고 최소 156개의 완료된
훈련 주를 요구한다. 훈련행은 `label_end < decision_date`인 것만 사용한다. 미래
라벨, HMM state, RORO 점수는 alpha 회귀학습에 넣지 않는다.

## 2. RRG 이동방향

RRG는 proprietary JdK 값이 아닌 기존의 transparent proxy를 사용한다.

- supportive: `NE_STRENGTHENING` 또는 `NW_EARLY_RECOVERY`
- non-supportive: `SE_LOSING_MOMENTUM`, `SW_DETERIORATING`, `WARMUP`
- 현재 quadrant, 5일 delta, heading, quadrant transition을 모두 보존한다.

RRG와 alpha 방향이 다르면 어느 쪽도 덮어쓰지 않고 `DIVERGENCE`로 표시한다.

## 3. RORO context fit

각 결정일의 RORO는 **같은 날짜 exact row만** 사용한다. 과거 값을 forward-fill하지
않으며 없으면 최종 상태를 `DATA_BLOCKED`로 둔다.

섹터별 60거래일 SPY beta를 해당 날짜 이전 가격만으로 계산한다. 최소 40개의
수익률 관측이 필요하다. 같은 날짜 섹터 횡단면에서 beta의 mid-rank percentile을
`beta_rank`(0~1)로 둔다.

```text
risk_appetite = 1 - market_risk_score / 100
roro_fit = risk_appetite * beta_rank
           + (1 - risk_appetite) * (1 - beta_rank)
```

따라서 Risk-On에서는 고beta, Risk-Off에서는 저beta 섹터의 적합도가 높다. 위험점수
50에서는 모든 beta의 적합도가 0.5가 되며, 이 값을 예측확률이라고 부르지 않는다.

## 4. 상태표

- `SELECT`: predicted alpha > 0, alpha rank <= 3, RRG supportive,
  `roro_fit >= 0.5`를 모두 만족
- `AVOID`: predicted alpha <= 0이고 RRG도 non-supportive
- `WATCH`: 나머지 모든 조합
- `DIVERGENCE`: alpha 양(+) 여부와 RRG supportive 여부가 서로 다름
- `RORO_MISMATCH`: `roro_fit < 0.5`

SELECT 종목 수를 억지로 3개로 채우지 않는다. `sector_status`는 매매주문이 아니라
연구용 우선순위다.

## 5. 검증과 중단조건

- 평가 시작: 2026-01-07
- 완료 label만 사용, weekly walk-forward, 20일 embargo
- 비용: SELECT 집합 one-way turnover에 10bp
- 보고: rank IC, Top3-Bottom3, SELECT hit rate, SELECT 평균 상대수익,
  RRG-alpha 일치율, turnover, RORO 구간별 결과
- 최소 증거: 52개 평가 주와 30개 SELECT 관측
- PASS: 위 최소표본, 평균 rank IC > 0, SELECT hit rate > 50%, SELECT 순수
  상대수익의 4주 moving-block 95% CI 하단 > 0을 모두 만족

최소표본 전에는 `INSUFFICIENT_EVIDENCE`다. 결과를 본 뒤 같은 v1의 alpha, training
기간, beta window, status 경계를 바꾸지 않는다. 변경은 새 model version으로만 한다.
운영 DB, 기존 Sector Rotation, 웹, RORO 라이브 series는 사용자 승인 전 변경하지
않는다.

결합 단계별 결과는 성과를 본 뒤 선택하지 않도록 아래 네 가지로 고정해 함께
보고한다. `alpha-only`는 predicted alpha가 양수이고 횡단면 rank가 3위 이내,
`alpha+RRG`는 여기에 supportive RRG를 추가, `alpha+RORO`는 alpha-only에
`roro_fit >= 0.5`를 추가, `full`은 세 조건을 모두 적용한 현재 SELECT다. 각 단계는
강제로 3개를 채우지 않으며 같은 주·같은 20일 상대수익, 동일가중, 현금을 포함한
one-way turnover와 10bp 비용으로 observations·hit rate·평균 gross/net excess를
보고한다. 이 표는 원인 귀속용 ablation이며 v1 조합 선택이나 gate 변경에 사용하지
않는다.

RORO 구간별 표는 `full` SELECT만 대상으로 GREEN/YELLOW/RED 각각의 주 수,
SELECT 관측 수, 적중률과 평균 20일 상대수익을 보고한다. 표본이 없으면 `null`로
명시하고 조용히 다른 구간과 합치지 않는다.
