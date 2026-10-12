# Independent HMM A2-E SHADOW Specification

Status: **RESEARCH SHADOW — NOT APPROVED FOR LIVE USE**  
Model version: `hmm3-nq-independent-a2e-logvol-v1-diag-mc10-k1-m252-w1260-r20-s42-n3`  
Benchmark: `NQ=F`

## 단일 변경 가설

A2-C의 최신 학습창에서 세 seed는 상태 순열을 제외하면 같은 해에 수렴했지만 fitted
transition diagonal은 약 0.967~0.994였다. 초기화·seed가 아닌 sticky prior의 영향을
분리하기 위해 transition prior의 diagonal만 25에서 1로 바꿔 전체 prior를 균등하게
한다. 입력, emission, covariance floor 0.10과 초기 transition 0.96/0.02는 A2-C와
동일하다.

## 고정 항목

- 입력: `trend_acceleration_20_60`, `slow_trend_ma200_slope_20`,
  `log_absolute_volatility_20`
- 학습구간 안에서만 StandardScaler 적합
- 상태 3개, diagonal covariance, 실제 분산 하한 0.10
- transition prior diagonal/off-diagonal 모두 1
- 초기 transition diagonal 0.96, off-diagonal 0.02
- minimum train 252, rolling train 1,260, refit 20일
- seeds 42·43·44, 기존 likelihood 선택과 상태명 부여 유지
- LIVE comparator 및 현재 `validate_hmm.py` gate 유지

## 기각 조건

모든 기존 gate를 동시에 통과해야 한다. 상태 전환이 늘어 persistence 또는 churn
gate를 위반하거나, 포화 개선과 함께 AUC가 A2-C보다 낮아지면 즉시 기각한다.

## 재현 명령

```powershell
New-Item -ItemType Directory -Force .\outputs\shadow_hmm_a2e
Copy-Item .\regime_router_publish.db .\outputs\shadow_hmm_a2e\a2e_input_20260813.db

.\.venv\Scripts\python.exe .\regime_router.py fit-db-v2 .\outputs\shadow_hmm_a2e\a2e_input_20260813.db `
  --feature-set independent-a2c `
  --model-version hmm3-nq-independent-a2e-logvol-v1-diag-mc10-k1-m252-w1260-r20-s42-n3 `
  --benchmark NQ=F --covariance-type diag --covariance-floor 0.10 `
  --sticky-strength 1 --min-train 252 --train-window 1260 --refit-every 20 `
  --random-state 42 --n-init 3

.\.venv\Scripts\python.exe .\validate_hmm.py .\outputs\shadow_hmm_a2e\a2e_input_20260813.db `
  --benchmark NQ=F `
  --model-version hmm3-nq-independent-a2e-logvol-v1-diag-mc10-k1-m252-w1260-r20-s42-n3 `
  --compare-model-version hmm3-sticky-nq-core-v1-k25-m252-w1260-r20-s42-n3 `
  --spec .\MODEL_SPEC_HMM_A2E.md `
  --json .\outputs\shadow_hmm_a2e\hmm_validation.json `
  --markdown .\outputs\shadow_hmm_a2e\hmm_validation.md
```
