# Market Lens Posterior-Weighted Sequential Risk Policy Candidate v3

Status: **PROVISIONAL / NOT APPROVED**  
Effective date: 2026-08-11  
Policy version: `risk-policy-v5.1-posterior-weighted-candidate`

## 변경 산식

HMM의 가장 큰 확률만 고르던 v2 점수 대신 세 사후확률을 모두 반영한다.

`Risk Score = round_half_up(10×p_bull + 50×p_sideways + 80×p_bear)`

- 세 확률은 각각 `[0, 1]`이며 합계가 1이어야 한다.
- 표시용 Regime은 가장 큰 확률의 `BULL/SIDEWAYS/BEAR`를 계속 사용한다.
- Score→Long 비중은 기존 anchor `0→100, 20→90, 40→75, 60→60,
  80→40, 100→25%` 사이를 선형보간한 뒤 자연수 %로 round-half-up한다.
- 순수 상태의 기본 비중은 `BULL=95%, SIDEWAYS=68%, BEAR=40%`다.

| HMM 확률 (BULL / SIDEWAYS / BEAR) | Risk Score | 기본 Long 비중 |
|---|---:|---:|
| 85% / 10% / 5% | 18 | 91% |
| 20% / 80% / 0% | 42 | 74% |
| 18% / 47% / 35% | 53 | 65% |
| 0% / 0% / 100% | 80 | 40% |

## 승계 규칙

`MODEL_SPEC_ROUTER_STATE_v2.md`의 Fast 3-of-8 감축, 단계식 재진입, 같은 날
멱등성, 날짜순 replay, t+1 수익률 적용, 결측치 처리, OOS 및 승격 gate를 그대로
승계한다. 정책 상태 직렬화 구조는 바뀌지 않아 `state_version=1`을 유지한다.

## 버전 격리와 제한

- v5.0 정책 상태·과거 이력은 v5.1의 시작 상태로 재사용하지 않는다.
- 배포 전 새 series를 최초 결정일부터 전체 날짜순으로 재생한다.
- HMM이 한 상태에 거의 100% 포화되면 가중 점수도 같은 비중에 머물 수 있다.
- 1% 단위 raw target 변화가 정상 복귀 3일 확인을 과도하게 지연하는지 OOS에서
  별도로 확인한다.
- Historical OOS, cooling-off, 126거래일 shadow를 통과하기 전에는
  CHAMPION으로 승격하거나 주문에 연결하지 않는다.
