# Market Lens Risk Management Model Spec v1.0

Status: **FROZEN FOR RESEARCH**  
Effective date: 2026-08-10  
Scope: Bloomberg 연결 전 로컬 가격·HMM·정책·검증 사양

이 문서는 `risk 관리모델.pdf`의 정성적 지침을 재현 가능한 수치 규칙으로
고정한다. 이 문서의 feature, target, threshold, exposure map을 바꾸면 반드시 새
`model_version`을 사용하고 Historical OOS와 Shadow 검증을 다시 시작한다.

## 1. 서로 다른 세 가지 출력

1. **Current Regime HMM posterior**
   - `p_risk_on`, `p_transition`, `p_risk_off`
   - `P(state_t | information available at t)`이며 미래 하락확률이 아니다.
2. **Regime Severity Score**
   - `round_half_up(100 * (p_risk_off + 0.5 * p_transition))`
   - 현재 국면의 위험 강도를 나타내는 0~100 정책지수다.
3. **Forward Liquidation Risk 20D (`q20`)**
   - 향후 20거래일 NDX 최대낙폭이 현재 종가 대비 `-5%` 이하일 확률이다.
   - Historical OOS 확률검증을 모두 통과하기 전에는 `q20_raw`와
     `PROVISIONAL`로만 부르고 대시보드의 확률로 사용하지 않는다.

HMM posterior나 Regime Severity를 `향후 하락확률`이라고 표시하지 않는다.

## 2. PDF8 원자료와 변환

모든 원자료는 `available_at <= decision_at` 조건을 만족해야 하고, 위험할수록 큰
값이 되도록 방향을 통일한다.

| 이름 | 고정 산식 |
|---|---|
| `nq_trend20` | `-log(NDX_t / NDX_t-20)` |
| `nq_rv20` | 20일 로그수익률 표준편차 `* sqrt(252)` |
| `vix_curve` | `log(VIX / VIX3M)` |
| `move_level` | `log(MOVE)` |
| `hy_oas_level` | `log(HY OAS in bp)` |
| `yen_carry5` | `-log(USDJPY_t / USDJPY_t-5)` |
| `breadth_risk` | `1 - S&P500 PIT 구성종목의 200DMA 상회비율` |
| `correlation20` | 해당 날짜 보유종목 20일 수익률의 평균 pairwise correlation |

Slow HMM은 상관된 위험 신호를 독립증거처럼 중복 계산하지 않도록 네 블록만
사용한다.

- `TREND = z(nq_trend20)`
- `STRESS = mean(z(nq_rv20), z(vix_curve), z(move_level), z(hy_oas_level))`
- `CARRY = z(yen_carry5)`
- `INTERNALS = mean(z(breadth_risk), z(correlation20))`

전처리는 각 학습구간에서만 1·99 percentile winsorization을 하고, 각 원자료를
`median / (1.4826 * MAD)`로 robust z-score화한 뒤 `[-4, 4]`로 제한한다. 네 블록을
만든 뒤에는 그 학습구간에서만 블록별 StandardScaler를 적합한다. MAD가 0인
원자료 또는 표준편차가 0인 블록이 있으면 해당 fit을 무효로 한다. 미래 전체
표본으로 median, MAD, clipping 경계 또는 scaler를 계산하지 않는다.

## 3. PDF8 HMM 후보

- model role: Current Regime only
- state 수: 3
- emission: Gaussian
- covariance: `diag`
- rolling train window: 1,260 complete trading days
- minimum train: 756 complete trading days
- refit: 20 trading days
- sticky transition prior: diagonal 25, off-diagonal 1
- initial transition matrix: diagonal 0.96, off-diagonal 0.02
- seeds: 42, 43, 44
- maximum EM iterations: 500; tolerance: `1e-4`
- diagonal variance floor: `0.10`
- state label: 네 risk block state mean 평균의 오름차순으로
  `RISK_ON / TRANSITION / RISK_OFF`

각 refit 학습구간에서 아래 gate를 먼저 계산한다. 다음 중 하나면 신규 fit은
무효이며 직전 정상 fit과 그 전처리기를 유지한다. 최초 정상 fit 전에는 posterior를
출력하지 않는다.

- 확률 NaN, 범위 오류 또는 합계 오류
- 한 state posterior occupancy가 5% 미만 또는 80% 초과
- 인접 state risk centroid 차이가 0.25 표준편차 미만
- 최근 252일 `max posterior >= 0.99` 비율이 80% 초과

포화를 숨기기 위한 posterior clipping·temperature scaling은 하지 않는다.

## 4. q20 고정 후보

- benchmark: `NQ=F` (현재 로컬 연구); Bloomberg 전환 시 별도 version
- target: `t+1 ... t+20`의 최대낙폭 `<= -5%`
- label embargo: 20 trading days
- model: `StandardScaler + LogisticRegression`
- `C=1.0`, `class_weight=None`, `max_iter=1000`, `random_state=42`
- minimum train: 756 observations
- rolling train window: 1,260 observations
- refit: 20 observations
- feature search와 hyperparameter grid search 금지

고정 feature는 다음 7개다.

1. `p_bear`
2. `p_sideways`
3. `p_bear_change_5`
4. `momentum_20`
5. `volatility_20`
6. `drawdown_60`
7. `ma200_slope_20`

마지막 20거래일은 q20 예측을 만들 수 있지만 outcome이 아직 완성되지 않았으므로
검증 통계에서 제외한다.

## 5. 점수와 기본 Long 비중

검증된 q20이 없을 때는 Regime Severity를 정책 score로 사용하되
`PROVISIONAL REGIME SEVERITY`라고 표시한다. 검증된 q20이 생긴 뒤에는
`Risk Score = round_half_up(100 * q20)`으로 교체한다.

점수에서 Long 비중으로의 함수는 아래 anchor를 선형보간하고 자연수 %로
round-half-up 한다.

| Score | Long exposure |
|---:|---:|
| 0 | 100% |
| 20 | 90% |
| 40 | 75% |
| 60 | 60% |
| 80 | 40% |
| 100 | 25% |

최종 범위는 25~100%다. 색상은 `GREEN >=90%`, `YELLOW 60~89%`,
`RED <60%`로 고정한다.

## 6. Fast Overlay

같은 미국 거래일 기준 다음 8개 중 확인된 trigger가 3개 이상이면 활성화한다.

1. VIX 1일 수익률 `>= +20%`
2. `VIX / VIX3M > 1.00`
3. NDX/NQ 1일 수익률 `<= -2.5%`
4. MOVE 1일 수익률 `>= +10%`
5. HY OAS 1일 변화 `>= +20bp`
6. USDJPY 3일 수익률 `<= -4%`
7. S&P500 200DMA breadth 1일 변화 `<= -10%p`
8. portfolio correlation20 `>= 0.70`

활성 시 `fast_target = max(25%, slow_target - 20%p)`다. 같은 episode에서 매일
누적 차감하지 않는다. 확인된 trigger가 3개면 다른 채널이 누락돼도 감축하며,
trigger가 3개 미만이고 데이터가 불완전하면 비중 증액을 금지한다.

## 7. Re-entry와 Bottom Fishing

감축은 즉시 반영한다. 증액은 다음을 모두 만족할 때만 허용한다.

- Fast Overlay 비활성
- 최근 20거래일 `p_risk_off` 고점이 40% 이상
- 현재 `p_risk_off`가 최근 고점보다 최소 5%p 낮음
- 현재 `p_risk_off`가 직전 거래일보다 낮음
- 다음 4개 exact 조건 중 3개 확인
  1. `VIX / VIX3M <= 1.00`
  2. 최신 HY OAS가 직전 관측치보다 높지 않음
  3. NQ 종가가 5DMA 이상
  4. NQ가 신규 20일 저점일 때 52주 신저가 수 감소와 상승종목 수 증가

증액 ladder는 `[25, 40, 60, 80, 100]%`다. 한 번에 현재보다 큰 다음 rung까지만
이동하고 raw target을 초과하지 않는다. 같은 거래일 재실행은 추가 증액하지 않는다.
Bottom Fishing은 주문이 아니라 `ALLOWED` advisory flag다.

## 8. 데이터·실패 안전장치

- proxy를 exact로 승격하지 않는다.
- 공개 현재 구성종목으로 산출한 breadth는 최초 관측일 이후 운용 Fast/Re-entry에만
  `PUBLIC_LIVE`로 사용하며, exact PIT breadth나 PDF 8변수 HMM 학습·OOS로 승격하지 않는다.
- `PUBLIC_LIVE` breadth는 전일가·200DMA·252일 저점 계산 가능 종목이 현재 구성종목의
  98% 이상일 때 실제 coverage를 함께 저장한다. exact PIT breadth는 100%를 유지한다.
- 핵심 입력 누락, 잘못된 확률, model-version mismatch는 `NO_DECISION`이다.
- 데이터나 모델이 불완전할 때 비중 증액은 금지한다.
- Fast 3개 확인에 따른 위험 감축은 다른 채널이 누락돼도 허용한다.
- 모든 결과는 advisory-only이며 자동 주문을 만들지 않는다.
- 공개 snapshot에는 보유종목·수량·계좌·내부 경로를 넣지 않는다.
- 포지션 변경 시에만 갱신되는 event holdings는 다음 변경 전까지 carry-forward한다.
  시장가격과 portfolio correlation은 매 거래일 다시 계산하며, 가격 coverage 90%
  미만 또는 미래 holdings date는 사용할 수 없다.

## 9. 검증과 승격 gate

### q20 probability

- usable observations >= 1,000
- independent drawdown episodes >= 20
- block-bootstrap AUC 95% CI lower > 0.50
- 단순 `volatility_20` 대비 paired delta AUC CI lower >= 0
- top-quintile lift >= 1.25
- Brier skill > 0
- calibration slope 0.75~1.25
- 10-bin ECE <= 0.10
- yearly AUC > 0.50인 연도 비율 >= 60%

### 정책 OOS

- frozen OOS >= 1,000 trading days
- next-day execution
- 거래비용 10bp 기본·25bp stress
- benchmark 대비 Max Drawdown 상대 개선 >= 10%
- 연환산 수익률 drag <= 2%p
- Sharpe 감소 <= 0.10
- 잘못된 re-risk 후 10일 내 -3% 또는 risk-off 재발 비율 <= 25%

모든 Historical gate를 통과해야만 126거래일 Shadow를 시작한다. Shadow에는 실제
감축과 재진입 cycle이 각각 최소 1회 필요하며, 없으면 최대 252거래일까지 연장한다.

## 10. 변경관리와 rollback

절차는 `RESEARCH -> CANDIDATE -> COOLING_OFF -> SHADOW -> CHAMPION`으로 고정한다.

- candidate 생성 후 최소 24시간 동안 승격 금지
- 같은 날 코드 변경·검증·승격 금지
- material change마다 새 model/policy version 사용
- Git SHA, dirty 상태, data cutoff, source, feature, target, hyperparameter, seed,
  gate 결과와 승격·기각 이유를 기록
- 직전 정상 champion 두 개를 rollback 대상으로 유지
- invalid output, refit failure, version mismatch 시 최근 정상 champion으로 복귀
- Daily data check, monthly outcome review, quarterly gate report, annual full validation

## 11. 현재 기준 상태

- NQ core v1: `PROVISIONAL` current-regime display only; Historical OOS gate `FAIL`
  (`outputs/hmm_nq_core_v1_validation.md`)
- NQ tied4 v2: Historical OOS gate `FAIL / REJECTED`
  (`outputs/hmm_nq_tied4_v2_validation.md`)
- q20 fixed logistic: Historical OOS `HISTORICAL_FAIL / RESEARCH_ONLY`
  (`outputs/q20_validation.md`)
- NQ core v1 slow policy: MDD 방어는 확인됐지만 return-drag gate 실패로
  `SLOW_HISTORICAL_FAIL`; Fast/Re-entry를 포함한 full PDF gate는 `NOT_EVALUATED`
  (`outputs/policy_oos.md`)
- PDF8 block4 HMM: exact PIT breadth·portfolio correlation 전 `DATA_BLOCKED`
  (`PDF8_SHADOW_STATUS.md`)
- Position Quality: Top-down 정책 승격 후 별도 phase로 연기
