# Market Lens Router-State Policy Candidate v1

Status: **PROVISIONAL / NOT APPROVED**  
Effective date: 2026-08-11  
Model version: `risk-policy-v4.0-router-state-candidate`

## 사양 우선순위

이 문서는 `MODEL_SPEC_v1.md` §1의 `Regime Severity Score` 산식과 §5의
`Regime Severity` 기반 정책 score를 v4에 한해 대체한다. §5의 exposure anchor와
색상 구간, §6 Fast Overlay, §7 Re-entry, 데이터 안전장치와 승격 절차는 그대로
유지한다.

## 목적

HMM posterior는 현재 잠재상태의 소속 확신도이며 미래 하락확률이 아니다. 이
후보는 원래 Regime Router처럼 posterior를 `BULL / SIDEWAYS / BEAR` 상태 판정에만
사용하고, posterior 크기를 Risk Score와 Long 비중에 직접 연결하지 않는다.

## 상태별 기본 정책

상태는 `get_regime_status()`와 동일한 argmax 규칙으로 선택한다.

| HMM state | Risk Score | 기본 Long 비중 |
|---|---:|---:|
| BULL | 10 | 95% |
| SIDEWAYS | 50 | 68% |
| BEAR | 80 | 40% |

Long 비중은 `MODEL_SPEC_v1.md`의 기존 score anchor 선형보간과 자연수 퍼센트
round-half-up을 그대로 사용한다. 같은 상태 안에서는 posterior가 51%이든
99.999%이든 같은 기본 점수와 비중을 사용한다.

## 유지하는 안전장치

- Fast Overlay 8개 중 3개 확인 시 한 번만 `-20%p`, 최저 25%
- 감축은 즉시 반영
- 증액은 기존 3-of-4 re-entry와 one-step ladder를 통과할 때만 허용
- 입력 오류와 상태/posterior 불일치는 `NO_DECISION`
- advisory-only이며 자동 주문과 연결하지 않음

## 승격 조건

이 후보는 현재 HMM의 수치 수렴이나 posterior 포화를 고치는 모델이 아니다.
포화가 정책 비중으로 증폭되는 연결만 제거한다. Historical OOS, 24시간
cooling-off, clean rerun, 126거래일 shadow를 통과하기 전에는 POLICY role로
승격하지 않는다.

Fast Overlay와 Re-entry를 제외한 `SLOW_POLICY_ONLY` OOS만으로는 위 Historical
gate를 통과한 것으로 보지 않으며, POLICY 승격 또는 cooling-off 시작을 허용하지
않는다.
