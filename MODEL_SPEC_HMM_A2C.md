# Independent HMM A2-C SHADOW Specification

Status: **RESEARCH SHADOW — NOT APPROVED FOR LIVE USE**  
Model version: `hmm3-sticky-nq-independent-a2c-logvol-v1-diag-mc10-k25-m252-w1260-r20-s42-n3`  
Benchmark: `NQ=F`

## 단일 변경 가설

A2-B의 절대 20일 변동성은 미래 낙폭 정보와 관계없이 229개 refit 학습창에서
지속적으로 우측 꼬리가 길었다. Gaussian emission에 더 적합한 척도로 바꾸기 위해
`absolute_volatility_20`만 `log(absolute_volatility_20)`으로 변환한다. 다른 입력,
HMM 설정, 상태명 부여, 검증 기준은 모두 A2-B와 같다.

## 결과를 보지 않은 사전진단

| 229개 학습창 중앙값 | 절대 변동성 | 로그 절대 변동성 |
|---|---:|---:|
| 왜도 | 2.174 | 0.324 |
| 초과첨도 | 5.995 | 0.103 |
| 최대 표준화 거리 | 4.751σ | 3.049σ |
| `|z| > 3` 비중 | 1.75% | 0.32% |

마지막 1,260일 학습창의 최대 변수 간 절대상관은 로그 변환 전 0.629, 변환 후
0.626으로 거의 같다. 이 후보는 변수 중복이 아니라 marginal distribution과
극단값 영향만 검사한다.

## 입력 및 고정 설정

1. `trend_acceleration_20_60`
2. `slow_trend_ma200_slope_20`
3. `log_absolute_volatility_20 = log(연율화 20일 변동성)`

각 변수는 매 refit의 학습구간 안에서만 `StandardScaler`로 표준화한다. 상태 3개,
diagonal covariance, 실제 분산 하한 0.10, sticky prior 25, 초기 전이 0.96/0.02,
minimum train 252, rolling train 1,260, refit 20일, seeds 42·43·44를 고정한다.
Clipping, temperature scaling, smoothing은 사용하지 않는다.

## 검증 및 기각 조건

LIVE comparator를 명시하고 현재 `validate_hmm.py`의 모든 gate와 bootstrap 설정을
그대로 사용한다. 하나라도 실패하면 `HISTORICAL_FAIL`로 기각한다. PASS여도 24시간
clean 재실행과 126거래일 forward shadow 전에는 승격하지 않는다.

## 승인 후 재현 명령

```powershell
New-Item -ItemType Directory -Force .\outputs\shadow_hmm_a2c
Copy-Item .\regime_router_publish.db .\outputs\shadow_hmm_a2c\a2c_input_20260813.db

.\.venv\Scripts\python.exe .\regime_router.py fit-db-v2 .\outputs\shadow_hmm_a2c\a2c_input_20260813.db `
  --feature-set independent-a2c `
  --model-version hmm3-sticky-nq-independent-a2c-logvol-v1-diag-mc10-k25-m252-w1260-r20-s42-n3 `
  --benchmark NQ=F --covariance-type diag --covariance-floor 0.10 `
  --sticky-strength 25 --min-train 252 --train-window 1260 --refit-every 20 `
  --random-state 42 --n-init 3

.\.venv\Scripts\python.exe .\validate_hmm.py .\outputs\shadow_hmm_a2c\a2c_input_20260813.db `
  --benchmark NQ=F `
  --model-version hmm3-sticky-nq-independent-a2c-logvol-v1-diag-mc10-k25-m252-w1260-r20-s42-n3 `
  --compare-model-version hmm3-sticky-nq-core-v1-k25-m252-w1260-r20-s42-n3 `
  --spec .\MODEL_SPEC_HMM_A2C.md `
  --json .\outputs\shadow_hmm_a2c\hmm_validation.json `
  --markdown .\outputs\shadow_hmm_a2c\hmm_validation.md
```
