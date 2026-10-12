# Independent HMM A2-D SHADOW Specification

Status: **RESEARCH SHADOW — NOT APPROVED FOR LIVE USE**  
Model version: `hmm3-sticky-nq-independent-a2d-logvol-v1-diag-mc25-k25-m252-w1260-r20-s42-n3`  
Benchmark: `NQ=F`

## 단일 변경 가설

A2-C는 OOS AUC와 63일 posterior health를 개선했지만 126·252일에는 emission
분리가 지나치게 날카로웠다. A2-D는 A2-C의 입력과 모든 설정을 고정하고 diagonal
covariance의 실제 사후 분산 하한만 `0.10`에서 `0.25`로 높인다. 표준화 공간의
최소 상태별 표준편차가 약 0.316에서 0.500으로 바뀐다. 이는 사후 확률 보정이
아니라 각 refit 학습 중 emission 분산에 적용되는 구조적 제약이다.

## 고정 항목

- 입력: `trend_acceleration_20_60`, `slow_trend_ma200_slope_20`,
  `log_absolute_volatility_20`
- 학습구간 안에서만 StandardScaler 적합
- 상태 3개, diagonal covariance, sticky prior 25, 초기 전이 0.96/0.02
- minimum train 252, rolling train 1,260, refit 20일
- seeds 42·43·44 중 안정 수렴 likelihood 최상
- LIVE comparator와 `validate_hmm.py`의 모든 gate·bootstrap 설정 유지
- temperature scaling, clipping, smoothing 금지

## 기각 조건

현재 검증 gate 중 하나라도 실패하면 `HISTORICAL_FAIL`이다. 특히 포화율만
개선되고 OOS AUC 또는 미래 낙폭 구분력이 A2-C보다 악화되면 즉시 기각한다.
PASS여도 24시간 clean 재실행과 126거래일 forward shadow 전에는 승격하지 않는다.

## 재현 명령

```powershell
New-Item -ItemType Directory -Force .\outputs\shadow_hmm_a2d
Copy-Item .\regime_router_publish.db .\outputs\shadow_hmm_a2d\a2d_input_20260813.db

.\.venv\Scripts\python.exe .\regime_router.py fit-db-v2 .\outputs\shadow_hmm_a2d\a2d_input_20260813.db `
  --feature-set independent-a2c `
  --model-version hmm3-sticky-nq-independent-a2d-logvol-v1-diag-mc25-k25-m252-w1260-r20-s42-n3 `
  --benchmark NQ=F --covariance-type diag --covariance-floor 0.25 `
  --sticky-strength 25 --min-train 252 --train-window 1260 --refit-every 20 `
  --random-state 42 --n-init 3

.\.venv\Scripts\python.exe .\validate_hmm.py .\outputs\shadow_hmm_a2d\a2d_input_20260813.db `
  --benchmark NQ=F `
  --model-version hmm3-sticky-nq-independent-a2d-logvol-v1-diag-mc25-k25-m252-w1260-r20-s42-n3 `
  --compare-model-version hmm3-sticky-nq-core-v1-k25-m252-w1260-r20-s42-n3 `
  --spec .\MODEL_SPEC_HMM_A2D.md `
  --json .\outputs\shadow_hmm_a2d\hmm_validation.json `
  --markdown .\outputs\shadow_hmm_a2d\hmm_validation.md
```
