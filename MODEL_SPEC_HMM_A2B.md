# Independent HMM A2-B SHADOW Specification

Status: **RESEARCH SHADOW — NOT APPROVED FOR LIVE USE**  
Model version: `hmm3-sticky-nq-independent-a2b-v1-diag-mc10-k25-m252-w1260-r20-s42-n3`  
Benchmark: `NQ=F`

## 단일 변경 가설

A2의 `log(20일 변동성 / 120일 변동성)`은 변동성의 절대 수준을 제거하면서 미래
낙폭 구분력이 약해졌을 수 있다. A2-B는 다른 설정을 모두 고정하고 이 입력 하나만
`연율화 20일 절대 변동성`으로 교체한다. A2와 LIVE 모델, 웹, 권고비중은 변경하지
않는다.

## 입력 변수

모든 산식은 해당 날짜까지의 `NQ=F` 종가만 사용한다.

1. `trend_acceleration_20_60`
   - `(P_t / P_{t-20} - 1) - ((P_t / P_{t-60})^(1/3) - 1)`
2. `slow_trend_ma200_slope_20`
   - `MA200_t / MA200_t-20 - 1`
3. `absolute_volatility_20`
   - `NQ 일간수익률의 20일 표준편차 × sqrt(252)`

각 변수는 매 refit의 학습구간 안에서만 `StandardScaler`로 표준화한다. 상태명 부여
시 A2와 동일하게 세 입력의 위험방향 가중치 절댓값을 모두 1로 유지한다.

## 결과를 보지 않은 입력 사전진단

2026-08-13 복제 DB의 마지막 refit 학습창(2021-08-04~2026-08-06, 1,260행)에서
확인했다. A2-B는 A2와 전체 4,816개 날짜 및 앞의 두 입력이 정확히 같다.

| 구분 | 최대 절대상관 | 상관행렬 유효 rank | 세 입력의 lag-1 자기상관 |
|---|---:|---:|---|
| A2 | 0.167 | 2.954 | 0.936 / 1.000 / 0.983 |
| A2-B | 0.629 | 2.572 | 0.936 / 1.000 / 0.986 |

A2-B의 최대상관은 장기 추세와 절대 변동성 사이의 `-0.629`다. 따라서 절대
변동성이 미래 낙폭 구분력을 회복할 가능성은 있지만, 확률 포화를 줄인다고 사전에
가정하지 않는다. 높은 자기상관과 sticky transition의 결합은 별도 구조적 원인으로
남아 있으며, A2-B 전체 OOS 검증 결과를 보기 전에는 다음 설정을 변경하지 않는다.

## 고정 HMM 설정

- 상태 3개, Gaussian emission, diagonal covariance
- 실제 사후 분산 하한 `0.10`
- sticky transition prior: diagonal 25, off-diagonal 1
- 초기 transition: diagonal 0.96, off-diagonal 0.02
- minimum train 252, rolling train 1,260, refit 20거래일
- seeds 42·43·44 중 수렴한 학습 likelihood 최상 모델
- posterior clipping, temperature scaling, 양방향 smoothing 금지

## 검증 및 기각 조건

`validate_hmm.py`의 기준과 bootstrap 설정을 변경하지 않는다. LIVE NQ core v1을
명시적 comparator로 넣고 63·126·252일 posterior health, 모든 historical gate,
paired AUC gate를 한 번에 평가한다. 하나라도 실패하면 A2-B는 `HISTORICAL_FAIL`로
기각하며 같은 OOS 결과에 맞춰 재조정하지 않는다. PASS여도 승격하지 않고 24시간
clean 재실행과 126거래일 forward shadow가 추가로 필요하다.

## 승인 후 재현 명령

```powershell
New-Item -ItemType Directory -Force .\outputs\shadow_hmm_a2b
Copy-Item .\regime_router_publish.db .\outputs\shadow_hmm_a2b\a2b_input_20260813.db

.\.venv\Scripts\python.exe .\regime_router.py fit-db-v2 .\outputs\shadow_hmm_a2b\a2b_input_20260813.db `
  --feature-set independent-a2b `
  --model-version hmm3-sticky-nq-independent-a2b-v1-diag-mc10-k25-m252-w1260-r20-s42-n3 `
  --benchmark NQ=F --covariance-type diag --covariance-floor 0.10 `
  --sticky-strength 25 --min-train 252 --train-window 1260 --refit-every 20 `
  --random-state 42 --n-init 3

.\.venv\Scripts\python.exe .\validate_hmm.py .\outputs\shadow_hmm_a2b\a2b_input_20260813.db `
  --benchmark NQ=F `
  --model-version hmm3-sticky-nq-independent-a2b-v1-diag-mc10-k25-m252-w1260-r20-s42-n3 `
  --compare-model-version hmm3-sticky-nq-core-v1-k25-m252-w1260-r20-s42-n3 `
  --spec .\MODEL_SPEC_HMM_A2B.md `
  --json .\outputs\shadow_hmm_a2b\hmm_validation.json `
  --markdown .\outputs\shadow_hmm_a2b\hmm_validation.md
```
