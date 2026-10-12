# Cross-Asset HMM A3-A SHADOW Specification

Status: **RESEARCH SHADOW — NOT APPROVED FOR LIVE USE**  
Model version: `hmm3-sticky-nq-crossasset-a3a-vixcurve-v1-diag-mc10-k25-m252-w1260-r20-s42-n3`  
Benchmark: `NQ=F`

## 단일 변경 가설

A2-C의 입력은 모두 NQ 가격에서 파생된다. A3-A는 옵션시장이 가격화한 단기 위험과
3개월 위험의 차이를 별도 정보축으로 보기 위해 아래 변수 하나만 추가한다.

`vix_curve = log(VIX / VIX3M)`

양수일수록 단기 공포가 큰 backwardation이므로 위험 방향 가중치는 `+1`이다.
NQ·VIX·VIX3M의 같은 날짜 종가만 inner join한다. Forward-fill, clipping, smoothing,
temperature scaling은 사용하지 않는다. EOD 값은 다음 거래일 판단에만 사용한다.

## 실행 전 동결 진단

- 원시 NQ/VIX/VIX3M 양수 공통일: 5,030일, 2006-08-10~2026-08-12
- A2-C 대비 usable-row 유지율: 99% 이상
- 전체 4,811일에서 VIX curve와 기존 세 변수의 최대 절대상관: 0.564
- 마지막 1,260일에서 VIX curve와 기존 세 변수의 최대 절대상관: 0.460
- 마지막 1,260일 effective rank: A2-C 세 변수 2.570, VIX curve 추가 후 3.269

현재 `daily_prices`는 관측일 기준으로 사용하며 strict historical-vintage PIT 저장소는
아니다. 이 한계는 결과 보고서에 유지하며, 해당 후보는 SHADOW 밖으로 승격하지 않는다.

## 입력 및 고정 설정

1. `trend_acceleration_20_60`
2. `slow_trend_ma200_slope_20`
3. `log_absolute_volatility_20`
4. `vix_curve`

각 변수는 매 refit의 학습구간 안에서만 `StandardScaler`로 표준화한다. 상태 3개,
diagonal covariance, 실제 분산 하한 0.10, sticky prior 25, 초기 전이 0.96/0.02,
minimum train 252, rolling train 1,260, refit 20일, seeds 42·43·44를 고정한다.

## 검증 및 기각 조건

LIVE comparator를 명시하고 현재 `validate_hmm.py`의 모든 gate와 bootstrap 설정을
그대로 사용한다. 하나라도 실패하면 `HISTORICAL_FAIL`로 기각한다. 특히 AUC는 동일
표본의 최고 단순 기준보다 0.02 이상 높고, LIVE 대비 paired AUC 95% CI lower가 0보다
커야 한다. 63·126·252일 posterior health도 모두 PASS해야 한다.

PASS여도 24시간 clean 재실행과 126거래일 forward shadow 전에는 승격하지 않는다.

## 승인 후 재현 명령

```powershell
New-Item -ItemType Directory -Force .\outputs\shadow_hmm_a3a
Copy-Item .\regime_router_publish.db .\outputs\shadow_hmm_a3a\a3a_input_20260813.db

.\.venv\Scripts\python.exe .\research_a3a_preflight.py `
  .\outputs\shadow_hmm_a3a\a3a_input_20260813.db --benchmark NQ=F `
  --output .\outputs\shadow_hmm_a3a\preflight.json

.\.venv\Scripts\python.exe .\regime_router.py fit-db-v2 .\outputs\shadow_hmm_a3a\a3a_input_20260813.db `
  --feature-set independent-a3a `
  --model-version hmm3-sticky-nq-crossasset-a3a-vixcurve-v1-diag-mc10-k25-m252-w1260-r20-s42-n3 `
  --benchmark NQ=F --covariance-type diag --covariance-floor 0.10 `
  --sticky-strength 25 --min-train 252 --train-window 1260 --refit-every 20 `
  --random-state 42 --n-init 3

.\.venv\Scripts\python.exe .\validate_hmm.py .\outputs\shadow_hmm_a3a\a3a_input_20260813.db `
  --benchmark NQ=F `
  --model-version hmm3-sticky-nq-crossasset-a3a-vixcurve-v1-diag-mc10-k25-m252-w1260-r20-s42-n3 `
  --compare-model-version hmm3-sticky-nq-core-v1-k25-m252-w1260-r20-s42-n3 `
  --spec .\MODEL_SPEC_HMM_A3A.md `
  --json .\outputs\shadow_hmm_a3a\hmm_validation.json `
  --markdown .\outputs\shadow_hmm_a3a\hmm_validation.md
```
