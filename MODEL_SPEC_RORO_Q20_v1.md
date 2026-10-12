# RORO q20 SHADOW Model Spec v1

## 1. 목적과 경계

`q20`은 의사결정일 `t` 이후 20 NQ 거래일 안에 NQ가 `t` 종가 대비 5% 이상
하락할 조건부 확률이다. RORO 점수, HMM posterior, confidence를 q20로 재명명하지
않는다. 이 문서는 입력·label·학습 개시 조건만 사전 동결한다. SHADOW r4 DB와 웹에는
각 기준일 당시 관측 가능했던 readiness 집계만 저장할 수 있다. 확률을 만들거나
readiness를 실제 배분에 쓰지 않는다.

- model version: `roro-q20-logit-v1-frozen8`
- benchmark/calendar: `NQ=F` 거래일
- target horizon: `t+1 ... t+20`
- event: `min(close[t+1:t+20] / close[t] - 1) <= -0.05`
- label available date: `t+20` NQ 거래일 종가 이후
- missing price 또는 20일 경로 미완성: label 결측

## 2. 동결 입력 8개

모든 값은 같은 날짜의 immutable RORO SHADOW payload에서 exact-date로 읽는다.
forward fill, 평균 대체, 50점 대체는 금지한다.

| 입력 | 경제적 근거 |
|---|---|
| `macro_risk_score` | 신용·금리변동성·주식변동성·FX/carry·안전자산의 거시 스트레스 |
| `market_trend_score` | NQ 가격 추세와 가용한 PIT 시장폭의 약화 정도 |
| `portfolio_internal_score` | 실제 보유 포트폴리오의 상관·beta·집중·참여·낙폭 위험 |
| `trajectory_1d` | 하루 충격과 급변 방향 |
| `trajectory_5d` | 주간 위험 가속·완화 |
| `trajectory_20d` | 월간 위험 국면의 누적 변화 |
| `data_completeness` | 그날 점수가 의존한 정보의 가용 범위 |
| `axis_disagreement` | 거시·시장·포트폴리오 위험 간 불일치 |

종합 `market_risk_score`는 세 축의 평균이고, regime·Long·confidence도 위 입력의
결정적 변환이므로 중복 입력에서 제외한다. 개별 하위 블록과 HMM도 v1에 추가하지
않는다. 모든 8개가 유한하고 점수·완전성·불일치는 0~100, trajectory는 -100~100
범위일 때만 `feature_complete=true`다.

## 3. 인과적 연구 frame

1. 날짜는 중복 없는 오름차순 NQ 거래일이어야 한다.
2. `t` feature는 `t` SHADOW payload만 사용한다.
3. label은 `t+20` 이후에만 학습 가능하다.
4. 의사결정 기준일 `as_of` 뒤의 feature와 가격은 `as_of` 이전 결과를 바꾸면 안 된다.
5. `feature_complete`와 `label_complete`가 모두 참인 행만 학습 후보가 된다.
6. 입력은 calendar date당 NQ 의사결정 행 하나만 허용한다. 같은 날짜의 intraday
   복수행을 여러 거래일로 세지 않는다.
7. builder의 horizon과 event threshold는 각각 20거래일과 -5%로 고정하며 호출자가
   덮어쓸 수 없다. readiness와 fitter guard는 저장된 flag를 신뢰하지 않고 feature,
   embargo, event label의 일관성을 다시 계산한다.

## 4. 학습 개시 gate

다음을 동시에 만족하기 전에는 fitter 호출을 금지한다.

- exact complete feature-label 252개 이상
- 서로 20 NQ 거래일을 넘게 떨어진 독립 -5% event cluster 20개 이상

미달 시 출력은 `status=INSUFFICIENT_EVIDENCE`, `probability=null`,
`fit_allowed=false`다. gate를 낮추거나 결측을 채워 확률 숫자를 만들지 않는다.
gate 통과는 연구 적합 허용일 뿐 승격 또는 성능 통과를 뜻하지 않는다.

SHADOW r4의 `q20` nested object는 다음 11개 필드만 허용한다.

`model_version`, `status`, `probability`, `fit_allowed`,
`horizon_trading_days`, `drawdown_threshold`, `feature_rows`,
`complete_feature_labels`, `independent_events`, `minimum_complete_labels`,
`minimum_independent_events`.

각 날짜의 count는 그 날짜까지 존재한 exact feature와
`label_available_date <= as_of`인 label만 사용한다. 최신 count를 과거 payload에
복사하거나 미래 날짜를 이용해 과거 count를 다시 쓰는 것은 금지한다.

## 5. 향후 walk-forward logistic 계약

gate가 통과된 뒤 별도 버전에서만 구현한다.

1. 각 예측일 `t`의 train은 `label_available_date <= t`인 과거 행만 쓴다.
2. expanding walk-forward를 사용하고 20 NQ 거래일마다 재적합한다.
3. `StandardScaler`는 매 refit의 train에만 fit하고 그 파라미터로 해당 미래 구간을
   transform한다. 전체표본 scaling/PCA/HMM/smoothing은 금지한다.
4. Logistic은 L2, `C=1`, `class_weight=None`, `solver=lbfgs`, deterministic 설정을
   사용한다. calibration은 사전 정의된 별도 train 구간 없이는 하지 않는다.
5. 동일표본·동일 label·10bp 기준으로 기존 RORO/Rule/HMM/단순 추세·변동성
   benchmark와 비교하고, OOS 결과를 본 뒤 feature·gate·threshold를 바꾸지 않는다.

## 6. 구현·안전 확인

`roro_q20.py`는 DataFrame을 받아 DataFrame/dict를 반환하는 read-only primitive만
제공한다. DB 연결, 다운로드, 적합 코드는 포함하지 않는다. `roro_shadow.py`와
`roro_site_publisher.py`는 위 11개 readiness 필드만 별도 r4 immutable series로
전달한다. r3 row는 수정하지 않는다. focused test는 `t+1..t+20` label, as-of 이후
불변성, 날짜별 count, exact 결측 차단, evidence gate와 fail-closed fitter guard를
검증한다.
