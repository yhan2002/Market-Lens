# Market Lens Sequential Risk Policy Candidate v2

Status: **PROVISIONAL / NOT APPROVED**  
Effective date: 2026-08-11  
Policy version: `risk-policy-v5.0-sequential-replay-candidate`

## 목적

라이브, 과거 백필, 성과평가가 동일한 날짜순 상태 전이를 사용한다. 과거 날짜를
각각 독립 계산하거나, 다른 정책 버전의 직전 비중을 새 정책의 시작값으로 쓰지
않는다.

## 유지하는 기본 정책

- HMM argmax 상태별 Risk Score: `BULL=10`, `SIDEWAYS=50`, `BEAR=80`
- `MODEL_SPEC_v1.md`의 exposure anchor 선형보간과 자연수 % round-half-up
- 결과 기본 비중: `BULL=95%`, `SIDEWAYS=68%`, `BEAR=40%`
- Fast 8개 중 3개 이상이면 기본 비중에서 한 번만 `-20%p`, 최저 25%
- 위험 감축은 당일 즉시 반영하며 자동 주문과 연결하지 않는다.

## 단일 상태 전이

`risk_policy.evaluate_risk_policy()`가 하루의 유일한 정책 전이다. 다음 상태는
snapshot에 그대로 저장하고 다음 거래일에 전달한다.

| 필드 | 의미 |
|---|---|
| `state_version` | 상태 직렬화 버전 |
| `policy_version` | 정책 코드·임계값 버전 |
| `regime_model_version` | HMM 계열 버전 |
| `last_as_of` | 마지막 처리 결정일 |
| `exposure` | 마지막 권고 비중 |
| `raw_target` | 마지막 당일 원시 목표 비중 |
| `recovery_clear_streak` | 정상 복귀 확인 연속일 |
| `episode_peak_p_bear` | 감축 episode 중 최고 Risk-Off posterior |
| `episode_started_as_of` | 감축 episode 시작일 |

정책·HMM 버전이 하나라도 다르거나 미래 날짜의 상태는 재사용하지 않는다. 같은
날짜 재실행은 추가 증액하지 않지만 새 위험에 따른 감축은 즉시 반영한다.
`NO_DECISION`은 마지막 정상 상태를 삭제하거나 cold-start하지 않는다.

## 재진입 규칙

과거 사양의 3-of-4 조건은 위기 저점 부근에서 한 단계 일찍 들어가는
`Bottom Fishing accelerator`로 유지한다. 이 조건을 놓쳤다는 이유만으로 정상
상승 국면에서 비중을 영구 동결하지 않는다.

### 조기 재진입

- Fast trigger 3개 미만
- episode 중 `p_bear` 고점 40% 이상
- 현재 `p_bear`가 episode 고점보다 5%p 이상 낮고 전일보다 하락
- 기존 exact 4개 조건 중 3개 확인

### 정상 복귀

- Fast trigger 3개 미만
- Fast 8개 중 exact 관측 채널 5개 이상
- 같은 raw target이 3거래일 연속 유지

두 경로 중 하나를 만족하면 `[25, 40, 60, 80, 100]%`의 다음 한 단계까지만
증액하고 raw target을 넘지 않는다. raw target이 95%이면 마지막 단계는 95%로
제한한다. 95%에 도달하면 episode를 종료한다.

이 분리는 Fast/Transition 감축이 실제로 발생했지만 `p_bear >= 40%`가 없었던
경우 존재하지 않는 Risk-Off 고점을 무기한 기다리는 결함을 제거한다. exact 관측
채널이 5개 미만이면 정상 복귀는 보류하며 결측을 CLEAR로 간주하지 않는다.

## 이력과 달력

- 정책 series: `policy_version + HMM model_version + chain revision`
- 최초 상태는 같은 series의 가장 이른 SPY/HMM 공통 결정일부터 순차 재생한다.
- 임의의 백필 시작일에서 raw 비중으로 다시 시작하지 않는다.
- 의사결정 달력은 SPY 거래일 중 해당 HMM posterior가 존재하는 날이다.
- 결정일 `t` 종가의 비중은 다음 이용 가능한 NQ price bar의 수익률부터 적용한다.
- NQ bar가 있으나 새 SPY 결정이 없는 날은 직전 비중을 유지한다.
- 다른 series 또는 legacy v1/v2 날짜 이력은 성과평가 입력으로 사용하지 않는다.

## Historical 보고 구분

`SEQUENTIAL_POLICY_AS_AVAILABLE`은 해당 날짜에 실제로 저장된 exact/UNAVAILABLE
상태를 그대로 재생한 결과다. Fast 8/8과 Re-entry 4/4가 모든 결정일에 exact가
아니면 `FULL_PDF_POLICY` 검증으로 표시하지 않는다.

- 보고서에는 채널별 사용 가능 일수와 결측 일수를 함께 기록한다.
- 결측일을 삭제하거나 미래 값으로 채우지 않는다.
- 초기 overlay exposure는 100%로 두고 첫 turnover는 첫 적용 비중과 100%의
  차이로 계산한다.
- 거래비용은 10bp 기본, 25bp stress이며 signal `t`는 return `t+1`에만 적용한다.

## 재진입 실패 gate

하나의 감축 episode에서 첫 `RECOVERY_STEP`만 재진입 사건으로 센다. 사건 다음
10개 NQ bar 중 누적수익의 최저값이 -3% 이하이거나, 같은 기간의 SPY 결정일 중
HMM argmax가 다시 `BEAR`이면 실패다. 독립 사건 20개 미만이면
`NOT_EVALUATED`; 20개 이상에서 실패율 25% 이하만 통과한다.

Historical metric, full exact-data coverage, 재진입 실패 gate, 24시간 cooling-off,
126거래일 shadow를 모두 통과하기 전에는 CHAMPION으로 승격하지 않는다.
