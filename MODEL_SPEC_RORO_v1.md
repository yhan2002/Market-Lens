# Market Lens Multi-Asset RORO Specification v1

Status: **FROZEN SHADOW DESIGN — NOT LIVE**  
Draft date: 2026-08-13  
Role: 새로운 `CURRENT_MARKET_RISK` 연구 계열의 사전 설계문서  
Primary market and Risk Management benchmark: `NQ=F`; sector-relative/comparison benchmark only: `SPY`
Web / 3-axis SHADOW output start: **2026-01-07**  
Reference-only RORO-EW calculation start: **2026-01-02**

이 문서는 `RORO.txt`를 운영모델로 복사하지 않고, 그 경제적 아이디어를
인과적·재현 가능한 다중자산 위험지도와 섹터 로테이션 시스템으로 다시 정의한다.
사용자 승인 전 운영 DB, 라이브 웹, 기존 A2-D/core 모델, 현재 권고비중을 변경하지
않는다.

이 문서의 산식·결측 규칙·2026 평가구간은 역사 OOS 결과를 보기 전에 동결했다.
이후 변경은 같은 `v1`을 덮어쓰지 않고 새 model version으로만 수행한다.

## 0. 최종 목적과 구현 우선순위

`RORO.txt`는 **외부 매크로 위험을 보는 benchmark**이지 복제 대상이 아니다. 최종
상단 Risk Management 엔진은 다음 세 축을 서로 분리해 보여준다.

1. `macro_risk`: HY OAS, MOVE, VIX/VIX3M, USDJPY, Gold로 외부 위험환경을 측정한다.
2. `market_trend`: NQ 추세·낙폭과 PIT market breadth로 시장 상승·하락 추세의 힘과
   약화를 측정한다.
3. `portfolio_internals`: 실제 보유종목의 상관, beta, 집중도, 상승참여율, 낙폭,
   수익률 dispersion으로 포트폴리오 내부 동조화와 추세의 질을 측정한다.

`market_risk_score`는 위 세 축을 결합한 최종 위험점수다. `macro_risk`만을 최종
점수라고 부르지 않으며, HMM은 별도 비교·확인 신호일 뿐 최종 점수의 필수 입력이
아니다. 웹 상단 Risk Management 완성이 1차 목표다. Sector Rotation은 이 구조와
검증계약이 안정된 뒤 각 섹터 ETF 내부/상대 정보를 같은 방식으로 적용하는 2차
목표다.

상단 Risk Management의 beta·추세·낙폭·q20·정책 성과 benchmark는 모두 `NQ=F`다.
기존 `portfolio_risk_metrics.beta`에 저장된 SPY beta는 `LEGACY_SPY_BETA`로 간주하고
신규 NQ 기반 SHADOW 결과와 섞지 않는다. `SPY`는 Sector Rotation의 11개 ETF 상대강도
기준과 비교용 buy-and-hold baseline에만 남긴다.

## 1. 서로 섞지 않는 출력

다음 출력은 별도 필드와 별도 검증 결과를 가진다.

1. `macro_risk_score`: 외부 매크로 스트레스 강도다.
2. `market_trend_score`: NQ·시장폭 기반 추세 위험 강도다.
3. `portfolio_internal_score`: 실제 보유종목 내부 위험 강도다.
4. `market_risk_score`: 세 축의 최종 종합점수. 0은 Risk-On, 100은 Risk-Off다.
5. `risk_regime`: 최종 점수의 설명용 Green/Yellow/Red 구간이다.
6. `trend_phase`: 상승·하락 추세의 진행/약화/전환 후보 상태다.
7. `turn_evidence`: 바닥·천장 후보를 뒷받침하거나 반박하는 관측항목 목록이다.
8. `fast_stress_overlay`: 1~수일 급변을 표시하는 별도 안전장치다.
9. `q20`: 향후 20거래일 NQ 최대낙폭이 -5% 이하일 확률이다.
10. `recommended_long`: 점수와 Fast Overlay를 정책 함수에 넣은 참고 비중이다.
11. `sector_rrg`, `sector_alpha20`, `sector_status`: 2차 Sector Rotation 출력이다.

`market_risk_score`나 HMM state posterior를 `q20`이라고 부르지 않는다. RRG와 예측 alpha가
다르면 어느 한쪽을 덮어쓰지 않고 `DIVERGENCE`로 표시한다.

## 2. RORO.txt 기술 감사

감사한 원본은 `RORO.txt` SHA-256
`7140ce128f1b9707b6304470ef9a19b0b63cbe106a67e3122ada415c9e03e7ce`로 고정한다.
해시가 다르면 같은 benchmark로 간주하지 않는다.

원형에서 재사용할 것은 다중자산 관점, 연속점수, state와 continuous score의 분리,
시각화 아이디어다. 아래 구현은 운영 후보에서 제거한다.

| 원형 위치 | 문제 | 영향 | v1 처리 |
|---|---|---|---|
| 6, 110~160 | 설명은 일별 변화지만 실제 HMM 입력은 수준의 10일 z-score이며 계산한 `*_chg`는 미사용 | 문서와 산식 불일치 | 블록별 원자료와 방향을 명시적으로 고정 |
| 100~103 | 여섯 시계열을 전 기간 inner join | 결측이 많은 자산이 전체 표본을 선택 | 날짜별 availability와 completeness를 따로 저장 |
| 163~165 | 전체표본 StandardScaler | 과거 점수에 미래 분포 유입 | 결정일 이전 trailing history만 사용 |
| 173~198 | 전체표본 HMM, 50 seed 중 동일표본 likelihood 최상, 전체 시계열 `predict/predict_proba` | in-sample 선택 및 양방향 smoothing | HMM은 비교용만 허용; 필요 시 walk-forward filtered probability만 사용 |
| 204~224 | 전체표본 state mean으로 상태 이름 결정 | 미래를 사용해 과거 state 의미를 재정의 | 훈련창 안에서만 의미 부여하거나 HMM 자체를 제외 |
| 133~136 | credit·VIX 수준의 20일 상관 | 비정상 수준 상관과 동일 stress의 중복 계산 | v1 기본 블록에서 제외; 별도 진단항목만 검토 |
| 138~140 | 252일 peak에 `min_periods=1` | 초기 표본의 정의가 이후와 다름 | 완전한 lookback 전에는 NO_SCORE |
| 278~334 | 전체표본 scaler·PCA·PC1 방향 후 expanding percentile | PCA loading과 방향에 미래정보 유입 | rolling PCA는 3차 후보로만 별도 walk-forward 적합 |
| 283~301 | 금 상승을 항상 Risk-Off로 고정 | 인플레이션·달러·실질금리 국면을 오판 | 금의 NQ 대비 상대수익으로 안전자산 수요를 측정 |
| 339~342, 449 | 30/70 blend와 ±25 경계 | 근거·OOS 검증 없는 정책 임계값 | 실행 전 가중치·경계를 동결하고 동일 OOS로 검증 |
| 247~268 | 알려진 위기구간만 사후 확인 | 정상기 오탐과 통계적 불확실성을 평가하지 못함 | 전체 OOS AUC·PR-AUC·lift·연도별 일관성·lead/lag 평가 |
| 32 | 모든 warning 숨김 | 수렴·결측·수치오류 은폐 | 오류와 데이터 품질을 명시적으로 기록 |
| 전체 | observation date만 있고 `available_at` 없음 | 역사 시점에 몰랐던 값 사용 가능 | 모든 입력에 source·observation date·available_at 저장 |
| 전체 | SPX 중심, 같은 날 적용 여부 불명확 | 성장주 포트폴리오와 실행시점 불일치 | NQ 중심, t 종가 후 산출, t+1부터 적용 |

따라서 원형의 역사 결과는 성능 benchmark가 아니라 **비인과적 참고선**으로만
재현한다. 신규 후보와 우열을 말할 때는 반드시 별도의 causal 복제본도 함께 둔다.

## 3. 공통 시점 계약

- 결정일 `t`의 신호는 해당 시장의 종가가 확정된 뒤 계산한다.
- 포트폴리오 수익과 비중 적용은 다음 거래 가능한 bar인 `t+1`부터 시작한다.
- 가격 이외 데이터는 `observation_date <= t`와 `available_at <= decision_at`을 모두
  만족해야 한다.
- 동일 날짜의 수정 vintage가 여러 개면 `t`에 실제 사용 가능했던 마지막 vintage만
  사용한다.
- 과거 결측을 미래 값으로 backfill하지 않으며 서로 다른 시장 달력은 제한 없는
  forward-fill을 하지 않는다.
- 각 채널에 허용된 최대 stale 일수를 별도로 고정한다. 이를 넘으면 `UNAVAILABLE`이다.
- 변환·scale·PCA·학습기의 모든 통계량은 해당 결정일 이전 자료만 이용한다.

## 4. 인과적 percentile 변환

1차 후보는 모델 적합 없이 trailing empirical percentile만 사용한다.

- history: 결정일을 제외한 직전 756개 유효 관측치
- minimum history: 252개
- minimum unique values: 20개
- 동일값 처리: mid-rank
- history가 부족하거나 변동이 없으면 `NO_SCORE`

위험할수록 큰 방향으로 정렬한 원자료 `x_t`의 percentile은 다음과 같다.

`P_t(x) = 100 * (0.5 + count(x_s < x_t) + 0.5 * count(x_s = x_t)) / (n + 1)`

여기서 `s`는 오직 trailing history에 속한다. 이 정의는 극단값에 강하고 미래의
평균·표준편차를 사용하지 않는다. 결과는 계산 중 실수로 유지하고 공개 시에만
round-half-up으로 자연수화한다.

위 252개 기준은 Macro와 Market 블록에 적용한다. 실제 보유내역은 원본이
2025-03-12부터 시작해 2026-01-02 직전에 상관 148일, NQ beta 147일, 집중도 148일,
20일 breadth 128일, 음의 참여율 146일만 존재한다. 이보다 오래된 포지션을 합성하지
않고 2026-01-02부터 점수를 내기 위해 **Portfolio 구성점수에만 minimum history
126개**를 적용한다. trailing window 756개와 minimum unique 20개, 현재일 제외,
mid-rank 규칙은 동일하다. 이는 미래 수익을 보지 않고 원천 데이터 시작일만 확인해
동결한 별도 품질 계약이다.

## 5. 동일가중 독립 블록 후보 `RORO-EW-v1`

### 5.1 블록 산식

각 하위 항목을 먼저 `P_t()`로 바꾼 뒤 블록 안에서 동일가중한다.

| 블록 | 위험 방향 원자료 | 블록 점수 |
|---|---|---|
| Credit | `log(HY OAS)`, `HY OAS_t - HY OAS_t-20` | 두 percentile 평균 |
| Rates Volatility | `log(MOVE)` | percentile |
| Equity Volatility | `log(VIX)`, `log(VIX / VIX3M)` | 두 percentile 평균 |
| Equity Trend | `-log(NQ_t/NQ_t-20)`, `-log(NQ_t/MA200_t)`, `-(NQ_t/rolling_max_60-1)` | 세 percentile 평균 |
| FX / Carry | `-log(JPY=X_t/JPY=X_t-20)` | `close` percentile; 엔화 강세를 stress로 측정 |
| Safe Haven | `log(GLD_t/GLD_t-20) - log(NQ_t/NQ_t-20)` | `close`; 금 자체 상승이 아니라 NQ 대비 상대수요 percentile |
| Market Internals | `1-breadth_200dma`, `portfolio_correlation_20` | 두 percentile 평균 |

블록 내부의 필수 하위 항목 하나라도 없으면 그 블록은 `UNAVAILABLE`이다. 결측값을
50점으로 바꾸지 않는다.

### 5.2 종합점수와 결측

가용 블록 집합을 `A_t`라고 할 때:

`RORO_t = round_half_up(mean(B_j,t for j in A_t))`

다음 조건을 모두 만족할 때만 점수를 낸다.

- Equity Trend와 Equity Volatility가 모두 가용
- 7개 중 최소 5개 블록 가용
- 가용 블록의 총 가중치가 5/7 이상

그 외에는 `NO_SCORE`다. 가용 블록만으로 평균을 낸 날은 `FULL`이 아니라
`AS_AVAILABLE`로 표시하고, 동일 표본 성능 비교에서는 완전한 공통 가용일을 별도로
보고한다.

### 5.3 설명 필드

- `trajectory_1d/5d/20d = RORO_t - RORO_t-k`
- 블록 기여도:
  `contribution_j = (B_j,t - 50) / number_of_available_blocks`
- 위험 driver: 양(+)의 기여도가 큰 상위 3개
- 완화 driver: 음(-)의 기여도가 작은 상위 3개
- completeness: `100 * available_blocks / 7`
- disagreement: 가용 블록 점수의 population standard deviation
- agreement: `100 * (1 - min(1, 2 * std(B_j/100)))`
- data status: `FULL / AS_AVAILABLE / NO_SCORE`

여기서 `k`는 NQ 의사결정 거래일 행 수이고 `trajectory`는 공개 자연수 RORO 점수의
차이다. 2026년 첫 행의 변화값을 초기화하기 위해 2025년 말 점수를 메모리에서만
계산할 수 있지만, 해당 과거 점수 행은 저장·보고하지 않는다. 가용 블록 구성이
달라진 날에는 변화값과 함께 구성 변경 여부를 별도로 표시한다.

블록과 종합 상태의 설명 경계는 `0~33 GREEN`, `34~66 YELLOW`,
`67~100 RED`다. 각각 Risk-On, Neutral, Risk-Off를 뜻하며 v1에서 동결한다.

### 5.4 실제 포트폴리오 내부위험

기존 7블록의 `portfolio_correlation_20` 한 항목만으로는 사용자의 목적을 충분히
반영하지 못한다. 실제 보유종목과 해당 시점의 비중을 이용해 다음 raw metric을
별도 산출한다. 종목명·수량·개별 비중은 SHADOW private DB 밖으로 내보내지 않고
aggregate만 저장한다.

| 항목 | 산식/의미 | 점수 방향 |
|---|---|---|
| Weighted correlation 20D | 보유비중으로 가중한 종목쌍 20일 평균 상관 | 높을수록 위험 |
| Correlation trajectory | 위 상관의 5·20일 변화 | 상승은 동조화 악화, 하락은 완화 |
| Beta 60D | 보유비중 포트폴리오 수익률의 NQ beta | 1 초과분이 클수록 위험 |
| HHI / effective positions | `HHI=sum(w_i^2)`, `N_eff=1/HHI` | HHI가 높고 N_eff가 낮을수록 위험 |
| Portfolio breadth | 비중가중 종목 중 20·50·200DMA 상회 비율 | 낮을수록 위험 |
| Negative participation 20D | 20일 수익률이 음수인 비중 합 | 높을수록 위험 |
| Portfolio drawdown 60D | 비중가중 portfolio index의 60일 고점 대비 낙폭 | 낙폭이 클수록 위험 |
| Cross-sectional dispersion | 종목별 20일 수익률의 비중가중 표준편차 | 방향점수 대신 내부 순환/불일치 진단 |

`portfolio_internal_score`는 다음 5개 구성점수를 인과적 percentile로 만든 뒤
가용 항목 기준 동일가중한다.

1. correlation: `portfolio_correlation_20`
2. beta: `nq_beta_60`
3. concentration: `concentration_hhi`
4. participation: `1-breadth_above_ma20`과 `negative_participation_20`의
   **strict 평균**
5. drawdown: `-portfolio_drawdown_60`

correlation과 participation은 필수이며 5개 중 최소 4개 구성점수가 가용해야 한다.
participation의 두 하위 항목 중 하나라도 없으면 participation 전체를 `UNAVAILABLE`로
둔다. `breadth_above_ma50/200`은 느린 추세 진단과 checklist에는 표시하지만, 현재
보유내역 이력으로 252개 prior 관측을 만들 수 없어 v1 score에는 넣지 않는다. 이는
OOS 결과가 아니라 label-free 데이터 가용성 확인 후 동결한 선택이다. dispersion은
위험의 단조 방향이 불명확하므로 score에 넣지 않고 confidence와 driver 설명에만 쓴다.

보유내역은 포지션 변경일에만 갱신되고 다음 변경 전까지 유지되는 event snapshot으로
취급한다. 역사적으로 파일을 실제 관측한 시각이 없는 행은 `CONDITIONAL_EVENT_PIT`로
표시하고 strict PIT 결과와 섞지 않는다. 앞으로 수집하는 snapshot에는 파일 fingerprint,
`holdings_date`, `available_at`, 계산 model version을 함께 보존한다.

가격기반 구성점수는 현재 equity sleeve 가중치의 85% 이상에 필요한 이력이 있을 때만
계산한다. 신규상장 등으로 이력이 짧은 15% 이내 sleeve는 점수에서 제외하고 실제
coverage를 함께 저장한다. 직전 정상 snapshot이 4종목·주식비중 40% 이상인데 새
snapshot이 1종목·주식비중 20% 미만이면 불완전 추출로 판정해 직전 정상 snapshot을
causal carry-forward하고 `CARRIED_FORWARD_INCOMPLETE_SNAPSHOT`을 표시한다. 미래
snapshot의 회복 여부는 이 판정에 사용하지 않는다.

### 5.5 추세 전환을 보는 출력

상승·하락의 끝을 하나의 확률로 단정하지 않는다. 다음 세 정보를 분리한다.

- `risk_level`: 현재 `market_risk_score`
- `risk_direction`: 1·5·20일 score trajectory와 가속/감속
- `trend_phase`: `UPTREND / TOPPING_CANDIDATE / RISK_BUILDING / DOWNTREND /
  BOTTOMING_CANDIDATE / RECOVERY_CONFIRMING`

바닥 후보 근거는 높은 위험수준 이후 VIX curve 정상화, HY widening 중단,
portfolio correlation 하락, portfolio/market breadth 개선, NQ 단기추세 회복을 각각
독립 checklist로 보여준다. 천장 후보는 낮은 위험수준 이후 이 조건들의 반대 방향을
사용한다. 지수 가격과 breadth가 엇갈리면 divergence를 별도 근거로 표시한다.
관측 근거가 부족하면 phase를 만들지 않고 `INSUFFICIENT_EVIDENCE`로 둔다. phase는
미래 낙폭확률 `q20`과 분리하여 검증한다.

### 5.6 웹 상단용 3축 합성 `RORO-3AXIS-v1`

`RORO-EW-v1`의 7블록은 RORO 원형을 인과적으로 고친 비교선으로 보존한다. 웹 상단의
최종 엔진은 사용자 목적에 맞게 외부환경·시장추세·실제 포트폴리오를 다음처럼
분리한다.

1. `macro_risk_score`: Credit, Rates Volatility, Equity Volatility, FX/Carry,
   Safe Haven의 가용 블록 동일가중 평균. Equity Volatility가 필수이고 최소 2개
   블록이 있어야 한다. 5개 미만이면 `AS_AVAILABLE`이며 결측을 50으로 대체하지 않는다.
2. `market_trend_score`: Equity Trend와 PIT breadth risk의 가용 항목 동일가중 평균.
   Equity Trend가 필수다. breadth가 없으면 `NQ_ONLY`로 표시한다.
3. `portfolio_internal_score`: 5.4의 계약을 사용한다. correlation과 portfolio
   breadth가 필수이며 correlation·beta·concentration·participation·drawdown 중 최소
   4개가 가용해야 한다.

최종점수는 가용 축의 동일가중 평균이다. `market_trend_score`와
`portfolio_internal_score`는 필수이며 최소 2개 축이 있어야 한다. Macro가 빠진 날에도
SHADOW 점수는 낼 수 있지만 `AS_AVAILABLE`로 표시하고 confidence를 낮춘다.

- `axis_completeness`: Macro 5개, Market 2개, Portfolio 5개 하위항목의 가용률
- `data_completeness`: 세 축 completeness의 단순평균; 없는 축은 0. Macro는 5개
  블록, Market은 2개 항목, Portfolio는 위 5개 구성점수 기준
- `axis_disagreement`: 가용 축 점수의 population standard deviation
- `axis_agreement = max(0, 100 - 2 * axis_disagreement)`
- `confidence = round_half_up(data_completeness * axis_agreement / 100)`
- 축 기여도: `(axis_score - 50) / available_axis_count`
- `trajectory_1d/5d/20d`: 공개 자연수 최종점수의 NQ 거래일 차이

데이터 상태는 `FULL_STRICT / AS_AVAILABLE_STRICT / CONDITIONAL_EVENT_PIT /
NO_SCORE` 네 값으로 고정한다. 과거 holdings 파일의 실제 관측시각이 없으므로 이를
사용한 2026 결과는 점수가 완전해도 `CONDITIONAL_EVENT_PIT`이며 strict 결과와 섞지
않는다.

이 confidence는 미래 낙폭확률이 아니라 **데이터 완전성과 축 간 합의도**다. Macro가
2/5만 있는 현재 무료데이터 결과를 `FULL` 또는 고신뢰도로 표현하지 않는다.

## 6. Green/Yellow/Red와 Long 비중

요청된 목표범위 Green 90~100%, Yellow 60~80%, Red 25~40%는 설명용 기준으로
보존한다. 실제 참고비중은 경계에서 갑자기 뛰지 않도록 아래 anchor 사이를 선형
보간하고 round-half-up으로 1%p 단위 자연수화한다.

`(위험점수, Long %) = (0,100), (33,90), (66,60), (100,25)`

따라서 색상은 현재 위험구간을 설명하고, 비중은 전체 점수에 연속적으로 반응한다.
Yellow 초입·Red 초입의 일부 값은 설명 범위 사이를 통과할 수 있으며 이는 불연속
점프를 피하기 위한 v1의 사전 동결된 선택이다.

어느 안이든 감축은 즉시, 증액은 위험점수 비상승과 Fast Overlay 해제를 확인한다.
기존의 `정확히 같은 목표비중 3일` 조건은 사용하지 않는다. Fast와 재진입의 효과는
score-only / overlay-only / combined ablation으로 분리 검증한다. 거래비용은 10bp만
평가하고 자동 주문에는 연결하지 않는다.

### 6.1 RORO policy/state v1 (SHADOW)

`roro-policy-v1-score-fast3of8-confirm3-shadow`는 위 Long anchor와
`risk_signals.build_fast_signal_snapshot`의 정확한 8개 condition 계약만 받는 별도
순수 state-fold다. HMM posterior나 state는 입력하지 않는다.

- score anchor는 위 선형보간을 round-half-up한 자연수 %다.
- 유효한 Fast condition이 3/8 이상이면 당일 즉시 `-20%p`, 최저 25%를 적용한다.
- score/base 악화와 Fast 감축은 당일 즉시 반영한다.
- 증액은 Fast trigger가 해제되고 5/8 이상 관측되며, 위험점수가 직전 의사결정일보다
  비상승인 상태가 연속 3 의사결정일 확인될 때 현재 raw target으로 바로 이동한다.
- 같은 raw target이 3일 유지될 필요는 없고 재진입 ladder도 사용하지 않는다.
- Fast 관측이 5/8 미만이면 증액만 보류한다. score 또는 Fast에 따른 감축은 허용한다.
- 기존 `daily_prices / JPY=X`의 UTC 날짜는 NQ 세션과 어긋나므로 승인된 별도 세션
  관측이 실제로 없으면 `jpy_3d_drop`을 `UNAVAILABLE`로 처리한다. 일요일 값을
  금요일로 되돌리거나 관측되지 않은 값을 채우지 않는다.
- 과거 holdings snapshot에는 실제 관측시각 `available_at`이 없으므로 Fast Overlay의
  `portfolio_correlation`도 `UNAVAILABLE`로 처리한다. 동일 aggregate는 포트폴리오
  내부위험축에서만 `CONDITIONAL_EVENT_PIT`로 사용할 수 있으며, Fast의 exact 조건으로
  승격하지 않는다.
- 상태는 `state_version`, 정확한 `policy_version`, ablation mode, 직전 날짜·점수·비중,
  episode active/inactive, 시작일과 clear streak를 보존한다. 버전·mode가 다른 상태는
  이어 붙이지 않으며 같은 날짜나 미래 상태도 거부한다.
- `score_only`, `overlay_only`, `combined`는 동일 reducer와 동일 replay API로 산출한다.
  overlay-only는 100% base에서 Fast만, score-only는 Fast를 의사결정에서 제외한다.

## 7. q20

`q20`은 RORO 점수와 별도 모델이다.

- target: `t+1 ... t+20` NQ 경로의 최대낙폭 `<= -5%`
- label embargo: 20거래일
- 최초 후보: 사전 동결된 RORO 블록 점수만 입력하는 단순 logistic model
- train-only scaler, rolling/expanding walk-forward, 예측일보다 늦게 완성된 label 제외
- 출력: raw probability, calibration 상태, 데이터 기준일, model version
- RORO나 HMM posterior를 q20로 재명명하지 않는다.
- 최소 학습자료: 완료된 20일 label 252개와 서로 20거래일 넘게 떨어진 독립
  -5% 하락 episode 20개. 둘 중 하나라도 부족하면 확률은 `null`, 상태는
  `INSUFFICIENT_EVIDENCE`다.
- 위 최소자료 전에는 임계값을 낮춰 숫자를 만들거나 웹에 미보정 확률을 표시하지
  않는다.

## 8. 검증계획

### 8.1 사전 동결

실행 전에 다음을 spec과 manifest에 기록한다.

- 입력 DB 복제본 경로·SHA-256·cutoff
- RORO.txt hash와 재현 방식
- feature 산식·방향·stale 규칙·history window
- 평가 시작·종료일, NQ 기준, horizon 20, drawdown -5%
- 비교모델 version, moving-block 길이 40일, bootstrap 500회, seed 42
- 비용 10bp, t+1 적용, 정책 anchor

RORO-EW 참고 블록은 2026-01-02부터 계산하되, 3축 최종점수의 평가·표시·SHADOW
signal row는 portfolio participation의 126개 prior 관측이 처음 완성되는
**2026-01-07부터** 생성한다. 첫 signal을 다음 NQ 거래일에 적용하므로 첫 성과
귀속일은 2026-01-08이다. 그 이전 자료는
252/756 유효관측치 percentile과 train-only 변환을 만들기 위한 warm-up으로만 읽고,
이 기간의 성과·점수는 보고하지 않는다. 과거 warm-up이 부족한 블록은 미래 자료로
채우지 않고 `NO_SCORE`로 둔다.

2026년 시작 표본은 현재 약 7개월에 불과하므로 최초 결과는 운용·데이터 계약을
점검하는 제한된 OOS 증거다. 통계적 우월성이나 승격 근거는 사전 동결한 forward
SHADOW가 추가된 뒤 판단한다.

### 8.2 동일 표본 비교

1. RORO.txt 원형 in-sample 결과 — 참고선
2. 원형을 causal rolling 방식으로 고친 결과
3. 기존 Rule Score
4. 기존 LIVE HMM
5. A2-D
6. NQ 20일 변동성
7. 단순 NQ 추세·변동성 결합
8. `RORO-EW-v1`

모든 순위지표는 동일 날짜 intersection에서도 다시 계산한다. coverage가 다른 결과를
그대로 우열 비교하지 않는다.

비교점수 산식도 첫 통합 실행 전에 다음처럼 고정한다.

- 기존 Rule Score: `nq-rule-stress-score-v2-w756-m252`의 공개 정수점수
- HMM core와 A2-D: 인과적 filtered posterior의 `100 * p_bear`; HMM의 현재상태
  확률이지 q20이 아니다.
- NQ 20일 변동성: `std(NQ 일수익률, 20) * sqrt(252)` 원값을 순위 검증에 사용한다.
  같은 Long anchor 비교에만 이 값을 직전 756개·최소 252개 causal percentile로
  바꾼다.
- 단순 NQ 추세·변동성: `-NQ 20일 수익률`과 위 20일 변동성을 각각 직전
  756개·최소 252개 causal percentile로 바꾼 뒤 동일가중한다. 이 기준선은 첫
  통합 실행 전에 성능 gate가 아닌 descriptive comparator로 동결한다.
- RORO-EW는 5/7 블록 계약을 만족하는 날짜가 없으면 `DATA_BLOCKED`로 보고하고
  임의 결측대체로 공통표본을 만들지 않는다.
- RORO.txt 원형은 full-sample scaling/PCA/HMM 때문에 `REFERENCE_ONLY`이며 paired
  ranking에서 제외한다.

10bp Long 비교는 후보와 모든 ranked comparator가 동시에 존재하는 정확한 공통
신호일만 사용한다. 모든 점수에 `(0,100),(33,90),(66,60),(100,25)` anchor를 동일하게
적용하고 `t` 종가 신호를 `t+1` NQ bar에 반영한다. 모델별 가용일을 따로 사용한
성과는 동일표본 비교로 부르지 않는다.

### 8.3 Market Risk 지표

- ROC-AUC, PR-AUC
- moving-block bootstrap 95% CI와 paired delta
- 위험점수 quintile별 미래 낙폭·event rate 단조성
- top-quintile lift
- 독립 위기 episode별 lead/lag와 오탐
- 1·5·20일 score 변화, 최빈점수 비중, terminal lock
- 상태 점유율, 지속기간, 전이 빈도, 연도별 AUC
- 권고비중 turnover, CAGR, Sharpe, MDD, annual return drag
- 재진입 후 10거래일 내 실제 -3% 하락 재발률
- coverage·freshness·NO_SCORE 비율

episode 지표는 첫 r6 통합 실행 전에 다음처럼 고정한다. NQ의 직전 252거래일
고점 대비 낙폭이 `-3%` 이상으로 회복되면 다시 대기 상태가 되고, 이후 처음
`-5%` 이하로 내려간 날을 위기 onset으로 본다. RORO 정수점수가 전일 67 미만
(또는 결측)에서 67 이상으로 들어간 날을 RED alert로 정의한다. 인접한 alert 사이의
간격이 20 NQ 거래일 이하면 같은 cluster로 연쇄 결합한다. cluster 안 alert 중 하나라도
onset 직전 1~20거래일에 있으면 적중이며, lead는 그 유효 1~20일 창 안에서 가장 이른
alert부터 onset까지의 NQ 거래일 수다. cluster의 마지막 alert 뒤 20거래일이 완성된
시점에도 매칭 onset이 없을 때만 오탐이고, 그 전까지는 `PENDING`이다.

재진입 재발은 `policy.phase == REENTRY_DIRECT`인 의사결정일의 NQ 종가를 기준으로
한다. 다음 1~10거래일 중 종가 경로가 기준 종가 대비 `-3%` 이하로 내려가면 실패로
세며, 당일은 포함하지 않는다. 미래 10거래일이 아직 완성되지 않은 tail 사건은
평가 대상에서 제외하고 별도로 보고한다. 이 정의에는 HMM 상태 판정을 넣지 않는다.

포화·고착만 개선되고 미래 낙폭 구분력이 단순 기준보다 나빠지면 기각한다. 첫 OOS
결과를 본 뒤 같은 구간에 맞춰 가중치·경계·window를 조정하지 않는다.

## 9. Sector Rotation 연결

- 시장 context는 NQ 중심 RORO를 사용한다.
- 상대강도 benchmark는 SPY, universe는 11개 GICS sector ETF다.
- RRG proxy는 현재 위치, 5일 delta, heading, quadrant transition만 설명한다.
- 20D alpha는 별도 causal weekly walk-forward 예측으로 유지한다.
- 최종 상태는 RRG, alpha rank/sign, RORO 적합도를 사전 동결된 표로 연결한다.
- RRG와 alpha 방향이 다르면 `DIVERGENCE`를 표시한다.
- 검증: rank IC, Top3-Bottom3, SELECT hit rate, RORO 구간별 안정성, turnover,
  RRG 이동과 미래 상대수익 관계, alpha 방향 일치율.

기존 `sector_rotation.py`의 RRG-style proxy와 5일 trajectory,
`sector_rotation_forecast.py`의 known-label weekly Ridge/10bp 검증은 재사용 후보지만,
기존 HMM probability feature는 RORO context로 교체한 새 model version에서만 검증한다.

## 10. 데이터 카탈로그 계약

정식 `RORO_DATA_CATALOG.csv`에는 최소 다음 열을 둔다.

`block, signal, local_security, bloomberg_security, table, field, unit, timezone,`
`observation_date_rule, available_at_rule, source, coverage_start, coverage_end, rows,`
`pit_status, freshness_rule, transform, risk_direction, status, action`

현재 가용성 매트릭스는 이 문서의 체크포인트 보고에 먼저 기록하고, 보안·PIT 상태와
security 명칭을 사용자 검토한 뒤 CSV로 동결한다.

### 10.1 2026-08-13 운영 DB 읽기 전용 가용성 감사

아래 수치는 `regime_router_publish.db`를 SQLite `mode=ro`와 `query_only=ON`으로
조회한 결과다. 이 감사 중 다운로드·upsert·모델 실행은 하지 않았다.

| 블록/자산 | 로컬 식별자 | 행·기간 | 현재 판단 |
|---|---|---|---|
| NQ | `daily_prices / NQ=F` | 5,035행, 2006-08-10~2026-08-12 | 다음 bar 기준 가격연구 가능. `available_at`과 roll/collateral 수익은 없어 strict PIT·실물선물 성과로 해석 금지 |
| SPX 비교 | `daily_prices / SPY` | 5,035행, 2006-08-07~2026-08-12 | ETF total-return proxy. 현금 SPX가 아니며 strict PIT 이력 없음 |
| HY OAS | `macro_observations / HY_OAS` | 973행·790개 관측일, 2023-08-08~2026-08-11 | 진짜 vintage 구조 사용. ALFRED causal 구간은 2025-12-01부터 183일; 이전 FRED 역사는 2026-08에 일괄 가용되어 과거 OOS 사용 불가 |
| MOVE | `daily_prices / ^MOVE` | 4,926행, 2006-08-10~2026-08-12 | 2026 NQ 표본 133/150. 2026-07-20~08-11의 17개 NQ 거래일이 결측이고 08-12에 재개. exact-date와 staleness gate 필요 |
| VIX | `daily_prices / ^VIX` | 5,036행, 2006-08-07~2026-08-12 | 가격연구 가능, strict PIT 이력 없음 |
| VIX3M | `daily_prices / ^VIX3M` | 5,032행, 2006-08-10~2026-08-12 | 자체 기간의 SPY 날짜와 일치. Cboe/Yahoo provider가 행별 저장되지 않아 provenance 보강 필요 |
| USDJPY | `daily_prices / JPY=X` | 5,190행, 2006-08-09~2026-08-13 | SPY 대비 610일 누락·767일 추가, 금요일 누락/일요일 추가가 반복. UTC 날짜 변환을 세션일로 정규화하기 전 모델 입력 금지 |
| Gold | 승인 `roro_price_observations / GLD / close` | 0행 | proxy와 세션 계약은 동결했지만 관측은 아직 적재하지 않아 `DATA_BLOCKED` |
| Breadth | `breadth_metrics` | 1행, 2026-08-12 | 현재 구성 503개 기반 `PUBLIC_CURRENT`; 역사 PIT가 아니며 1일 변화도 NULL. Historical OOS 사용 금지 |
| Portfolio corr | `portfolio_risk_metrics` | 153행, 2026-01-02~2026-08-12 | 가격 coverage 93.2~100%. `holdings_date`는 있으나 `available_at`/vintage가 없어 조건부 분석만 가능 |
| Sector ETFs | XLB/XLE/XLF/XLI/XLK/XLP/XLRE/XLU/XLV/XLY | 각 2,518행, 2016-08-05~2026-08-12 | SPY 날짜와 일치, 다음 bar OOS 가능; strict price vintage는 없음 |
| XLC | XLC | 2,048행, 2018-06-19~2026-08-12 | 상장·가용 시작이 늦으므로 11개 공통표본은 2018-06-19 이후 |

가격 테이블은 `(ticker,date)`를 덮어쓰는 구조이고 `fetched_at`은 ingest 시각일 뿐
당시 공개 vintage가 아니다. 따라서 본 연구의 가격 OOS는 **재현 가능한 계산
인과성**은 확보할 수 있지만 **vendor revision까지 포함한 strict PIT**라고 부르지
않는다. Macro만 `(series,date,available_at)` vintage key를 갖는다.

### 10.2 데이터 선결 작업 우선순위

2026-01-07~2026-08-12의 웹 SHADOW 구간은 NQ 거래일 150일이다. 현재 exact-date
가용성은 NQ·VIX·VIX3M·HY OAS가 150/150, MOVE가 133/150이다. USDJPY·Gold·역사
시장 breadth는 데이터 품질 계약에 따라 0/150으로 차단한다. 포트폴리오 상관·NQ
beta·집중도·20D/50D breadth는 150/150이며, 200DMA breadth는 2026-04-21부터
79/150이다. 포트폴리오 집계는 과거 파일 관측시각이 없어 모두
`CONDITIONAL_EVENT_PIT`로 표시한다.

포트폴리오 품질 이력에는 short-history sleeve 제외 14일, 불완전 1종목 snapshot의
causal carry-forward 3일, 조정종가 proxy weight 사용 2일이 포함된다. 가격 비중
coverage는 96.0~100.0%, score-relevant history coverage는 89.46~100.0%다.

1. USDJPY timestamp를 New York/FX session observation date로 정규화하고 원 timestamp를
   보존한다.
2. 동결한 Gold proxy `GLD / close`를 세션 관측 DB에 적재한다.
3. HY OAS ALFRED vintage coverage를 연구 시작일까지 확대한다.
4. breadth는 역사 PIT membership 확보 전 `DATA_BLOCKED`로 유지한다.
5. holdings snapshot에 실제 파일 관측시각을 `available_at`으로 저장한다.
6. `daily_prices`에 source/provider와 immutable ingest vintage를 추가할지는 별도 migration
   사양으로 검토한다. 첫 체크포인트에서는 스키마를 변경하지 않는다.

### 10.3 기존 모듈 재사용 경계

| 영역 | 재사용 | 그대로 재사용하지 않을 것 |
|---|---|---|
| 가격 | `market_data.register_universe`, `download_prices`를 ingestion job으로만 사용; runtime reader는 `regime_router.load_benchmark_from_database`, `sector_rotation.load_price_matrix` | signal 계산 중 다운로드·DB 초기화 호출, active-universe synthetic breadth |
| Macro | `macro_data.series_as_of`, `latest_as_of`, `risk_signals.load_hy_oas_history` | live FRED fetch 시각을 과거 vintage로 간주 |
| Breadth | `risk_signals.load_breadth_history`와 EXACT/PUBLIC_LIVE provenance·coverage 계약 | `compute_breadth_metrics`를 runtime에서 실행하거나 current membership을 역사 PIT로 승격 |
| Portfolio | `portfolio_risk.average_portfolio_correlation`과 저장된 aggregate reader | runtime 전체 backfill, `available_at` 없는 holdings를 strict PIT라고 표현 |
| Fast stress | `risk_signals.build_fast_signal_snapshot`의 8개 condition·quality·fail-closed 구조 | Fast 3-of-8을 RORO 본체 점수로 혼합 |
| Slow score | `router_score_v2`의 shifted trailing percentile·variation gate 패턴 | 기존 3개 NQ 블록 점수를 새 7블록 RORO와 동일 모델이라고 명명 |
| 정책 | `risk_policy`의 순수 state-fold·즉시 감축 패턴 | HMM posterior/version과 결합된 exposure/re-entry 함수를 새 family에 억지로 주입 |
| Market 검증 | `validate_router_score`와 `validate_hmm`의 future drawdown, score metric, paired block bootstrap; `policy_oos`의 t+1·10bp·성과 함수 | HMM-specific evaluator 전체와 private 함수에 장기 결합 |
| Sector | `sector_rotation.build_rrg_signals`, `latest_snapshot`; `sector_rotation_forecast`의 weekly known-label embargo, Ridge, rank IC·turnover·bootstrap 패턴 | 기존 HMM `p_sideways/p_bear` feature를 RORO 이름 아래 그대로 사용 |
| 게시 | immutable `series_version`, FULL_REPLAY→STRICT_PRIOR, parent hash 패턴 | 기존 schema v3와 active series에 새 필드를 직접 추가 |

최소 결합경로는 `RORO slow blocks + 기존 Fast Overlay -> 별도 RORO policy/state`,
`RRG + 20D sector alpha -> 별도 sector rank`, 마지막에만
`market exposure × sector preference`로 연결하는 방식이다.

### 10.4 FX·Gold 세션 관측 계약

- 계약 버전은 `yahoo-session-v1-jpyx-gld-close`, source는 `YAHOO_CHART_V1`이다.
- 허용 security는 `JPY=X`, `GLD` 두 개뿐이며 모델 가격은 모두 `close`로 동결한다.
- 기존 `daily_prices` 스키마와 API는 변경하지 않는다. 별도 append-only DB의
  `roro_price_observations` 테이블에
  `contract_version, source, ticker, raw_timestamp_utc, source_timezone,`
  `observation_session_date, available_at, close, adj_close, fetched_at`을 저장한다.
- `source_timezone`은 빈 값이 아닌 IANA timezone이어야 하고, 세 timestamp는 timezone이
  명시된 UTC여야 한다. 자연키는
  `(contract_version, source, ticker, raw_timestamp_utc, available_at)`이다. 완전히 같은
  행의 재저장만 idempotent로 허용하며 같은 키의 값 변경과 UPDATE·DELETE는 중단한다.
- 관측값은 각 NQ 거래일의 뉴욕 17:00(정규장 마감 후 1시간) cutoff 이후 첫
  의사결정일에만 배치한다. Sunday FX timestamp를 Friday로 되돌리지 않고,
  빈 NQ일을 채우지 않는다. 휴일 때문에 여러 관측이 같은 NQ일로 모이면
  `available_at`이 가장 늦은 관측을 사용한다.
- r6 `shadow-r6-q20-policy-fxgold`는 별도 DB 경로를 필수로 받고, 동결
  contract/source의 관측 행이 하나도 없으면 생성을 fail-closed한다. DB 자체는
  유효하지만 특정 결정일에 사용 가능한 관측이 없으면 값을 채우지 않고
  해당 블록을 `DATA_BLOCKED`로 남긴다. 실제 관측이 있을 때만
  `CONDITIONAL_PRICE_SESSION_CAUSAL`로 표시한다.
- r6 provenance의 `model_versions.price_observations`는
  `yahoo-session-v1-jpyx-gld-close`로 동결하고 게시 envelope는 schema 4를 사용한다.
  r3~r5 행과 provenance 계약은 변경하지 않고 명시적 조회를 허용한다.
- Yahoo 역사자료는 vendor vintage가 아니므로 strict PIT라고 하지 않는다. 코드와
  focused test만 준비하며 다운로드·전체 r6 생성·승격은 별도 승인 전
  수행하지 않는다.

## 11. 개발 순서와 안전 경계

1. 동결 명세와 데이터 매트릭스 일치 확인
2. 별도 복제 DB와 `outputs/roro_ew_v1/` 준비; RORO-EW row는 2026-01-02,
   3축 최종 row는 2026-01-07부터 저장
3. 블록 산식·결측·causality focused tests
4. 작은 smoke test
5. 실행 범위·예상시간을 보고하고 전체 walk-forward 승인
6. 동일 표본 비교 및 기각/유지 결정
7. q20, 정책, Sector 순으로 별도 후보 진행

운영 DB, 라이브 웹, 현재 정책 series, A2-D/core rollback은 명시 승인 전 변경하지
않는다. 현재 문서는 SHADOW 설계로 동결됐으며, 장시간 walk-forward 전에는 입력 DB,
출력 경로, 예상시간과 비용을 먼저 보고하고 실행 승인을 받는다. 역사 OOS 결과를 본
뒤 v1 산식·경계·결측 규칙을 바꾸지 않으며 변경이 필요하면 새 model version으로
분리한다.
