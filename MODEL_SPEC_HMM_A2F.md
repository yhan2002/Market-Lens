# Independent HMM A2-F SHADOW Specification

Status: **RESEARCH SHADOW — NOT APPROVED FOR LIVE USE**  
Model version: `hmm3-sticky-nq-independent-a2f-logvol-v1-full-ef10-k25-m252-w1260-r20-s42-n3`  
Benchmark: `NQ=F`

## 단일 변경 가설

A2-C 마지막 학습창에서 장기 추세와 로그 절대변동성의 상관은 -0.626이다.
Diagonal covariance는 이 상관을 모델링하지 못한다. A2-F는 A2-C의 모든 입력과
설정을 고정하고 state-specific covariance만 diagonal에서 full로 바꾼다. 각 상태
공분산의 최소 고유값 0.10을 매 M-step 뒤 실제로 강제해 특이행렬과 과도하게 좁은
방향을 막는다.

기존 tied 후보는 모든 상태에 같은 covariance를 강제해 AUC 0.501, 252일 포화
100%였다. A2-F의 full covariance는 상태별 상관구조를 독립적으로 허용하므로 tied와
다른 가설이다.

## 고정 항목

- 입력: `trend_acceleration_20_60`, `slow_trend_ma200_slope_20`,
  `log_absolute_volatility_20`
- 학습구간 내부 StandardScaler
- 상태 3개, state-specific full covariance, eigenvalue floor 0.10
- sticky prior diagonal 25/off-diagonal 1, 초기 transition 0.96/0.02
- minimum train 252, rolling train 1,260, refit 20일
- seeds 42·43·44 및 기존 likelihood 선택·상태명 부여 유지
- 현재 `validate_hmm.py`와 LIVE comparator 유지

## 기각 조건

확률 무결성, occupancy, persistence, posterior health, OOS AUC, 단순 기준, yearly,
paired gate를 모두 통과해야 한다. 포화만 개선되고 AUC가 A2-C보다 나빠지면 즉시
기각한다.

## 재현 명령

```powershell
New-Item -ItemType Directory -Force .\outputs\shadow_hmm_a2f
Copy-Item .\regime_router_publish.db .\outputs\shadow_hmm_a2f\a2f_input_20260813.db

.\.venv\Scripts\python.exe .\regime_router.py fit-db-v2 .\outputs\shadow_hmm_a2f\a2f_input_20260813.db `
  --feature-set independent-a2c `
  --model-version hmm3-sticky-nq-independent-a2f-logvol-v1-full-ef10-k25-m252-w1260-r20-s42-n3 `
  --benchmark NQ=F --covariance-type full --covariance-floor 0.10 `
  --sticky-strength 25 --min-train 252 --train-window 1260 --refit-every 20 `
  --random-state 42 --n-init 3

.\.venv\Scripts\python.exe .\validate_hmm.py .\outputs\shadow_hmm_a2f\a2f_input_20260813.db `
  --benchmark NQ=F `
  --model-version hmm3-sticky-nq-independent-a2f-logvol-v1-full-ef10-k25-m252-w1260-r20-s42-n3 `
  --compare-model-version hmm3-sticky-nq-core-v1-k25-m252-w1260-r20-s42-n3 `
  --spec .\MODEL_SPEC_HMM_A2F.md `
  --json .\outputs\shadow_hmm_a2f\hmm_validation.json `
  --markdown .\outputs\shadow_hmm_a2f\hmm_validation.md
```
