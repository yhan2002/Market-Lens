# Independent HMM A2 SHADOW Specification

Status: **RESEARCH SHADOW — NOT APPROVED FOR LIVE USE**  
Model version: `hmm3-sticky-nq-independent-a2-v1-diag-mc10-k25-m252-w1260-r20-s42-n3`  
Benchmark: `NQ=F`

## 목적

기존 HMM의 여섯 입력이 같은 가격 추세와 낙폭을 반복 측정해 상태확률이 100%에
가깝게 쏠리는 문제를 줄인다. A2는 역할이 다른 세 입력만 사용하며 라이브 모델과
웹 권고비중을 변경하지 않는다.

## 입력 변수

모든 산식은 해당 날짜까지의 종가만 사용한다.

1. `trend_acceleration_20_60`
   - `20일 수익률 - 60일 수익률의 복리 20일 환산값`
   - `(P_t / P_{t-20} - 1) - ((P_t / P_{t-60})^(1/3) - 1)`
   - 단기 추세가 중기 추세보다 약해질수록 위험하다.
2. `slow_trend_ma200_slope_20`
   - `MA200_t / MA200_t-20 - 1`
   - 장기 추세가 하락할수록 위험하다.
3. `volatility_shock_20_120`
   - `log(연율화 20일 변동성 / 연율화 120일 변동성)`
   - 단기 변동성이 장기 수준보다 높아질수록 위험하다.

가격 낙폭과 MA200 이격도는 별도 Drawdown Onset·Fast Overlay와 중복되므로 A2
HMM에서 제외한다. 각 변수는 학습구간 안에서만 StandardScaler로 표준화한다.

## HMM 설정

- 상태 3개, Gaussian emission, diagonal covariance
- 실제 사후 분산 하한 `0.10`
- sticky transition prior: diagonal 25, off-diagonal 1
- 초기 transition: diagonal 0.96, off-diagonal 0.02
- minimum train 252, rolling train 1,260, refit 20거래일
- seeds 42·43·44 중 수렴한 학습 likelihood 최상 모델
- state label은 세 입력의 동일가중 위험방향 평균으로 Risk-on/Transition/Risk-off
- posterior clipping과 temperature scaling 금지

## 검증 및 승격 제한

현재 `validate_hmm.py`의 최신 63·126·252일 posterior health gate를 모두 통과해야
한다. 확률 포화, entropy, 자연수 점수 다양성, 상태 전환 과다를 함께 검사하고,
라이브 NQ core v1과 공통 날짜 paired OOS 비교를 수행한다. Historical PASS만으로
승격하지 않으며 24시간 clean 재실행과 126거래일 forward shadow가 추가로 필요하다.

## 재현 명령

아래 작업은 수 분 이상 걸릴 수 있으므로 승인 후 실행한다. 운영 DB 대신 복제한
SHADOW DB를 사용한다.

```powershell
New-Item -ItemType Directory -Force .\outputs\shadow_hmm_a2
Copy-Item .\regime_router_publish.db .\outputs\shadow_hmm_a2\a2_input.db

.\.venv\Scripts\python.exe .\regime_router.py fit-db-v2 .\outputs\shadow_hmm_a2\a2_input.db `
  --feature-set independent-a2 `
  --model-version hmm3-sticky-nq-independent-a2-v1-diag-mc10-k25-m252-w1260-r20-s42-n3 `
  --benchmark NQ=F --covariance-type diag --covariance-floor 0.10 `
  --sticky-strength 25 --min-train 252 --train-window 1260 --refit-every 20 `
  --random-state 42 --n-init 3

.\.venv\Scripts\python.exe .\validate_hmm.py .\outputs\shadow_hmm_a2\a2_input.db `
  --benchmark NQ=F `
  --model-version hmm3-sticky-nq-independent-a2-v1-diag-mc10-k25-m252-w1260-r20-s42-n3 `
  --compare-model-version hmm3-sticky-nq-core-v1-k25-m252-w1260-r20-s42-n3 `
  --spec .\MODEL_SPEC_HMM_A2.md `
  --json .\outputs\shadow_hmm_a2\hmm_validation.json `
  --markdown .\outputs\shadow_hmm_a2\hmm_validation.md
```
