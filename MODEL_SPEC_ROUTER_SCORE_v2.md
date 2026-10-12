# Market Lens NQ Rule-Based Stress Score Candidate v2

Status: **RESEARCH SHADOW / NOT APPROVED**  
Effective date: 2026-08-13  
Model version: `nq-rule-stress-score-v2-w756-m252`

## 목적

HMM 확률 포화와 무관하게 NQ 추세·변동성·낙폭의 실제 변화를 매일 표시한다.
평범한 시장을 위험 50점으로 간주했던 v1의 구조적 편향을 제거한다. HMM은 국면
표시와 재진입 더블체크로만 남긴다.

## 고정 산식

v1과 동일하게 현재 값을 제외한 직전 756개 관측치에서 백분위 순위를 계산하며,
최소 과거 252개가 필요하다. Trend·Volatility·Drawdown 세 블록도 동일하다.

각 블록의 50분위까지는 정상 범위로 보고 0점, 50분위 초과분만 0~100으로
선형 변환한다.

각 원자료의 직전 이력에 서로 다른 유효값이 20개 미만이면 동률로 인한 잘못된
극단 백분위를 막기 위해 해당 날짜는 점수를 내지 않는다.

- `Block Stress = max(0, 2 × Block Percentile - 1)`
- `Risk Score = round_half_up(100 × mean(Trend Stress, Volatility Stress, Drawdown Stress))`

세 블록은 동일 가중치다. 결과값이나 미래 수익률로 계수·중립선·가중치를
학습하지 않는다. 점수는 0~100 자연수이며 높을수록 위험하다.

## 비중·재진입 규칙

- 기존 0~100 점수별 Long 비중 anchor와 Fast Overlay는 유지한다.
- 위험 완화 중 raw 목표가 전일보다 같거나 높으면 회복 확인일을 이어간다.
- 3거래일 연속 확인 후 기존 25→40→60→80→100% 계단으로 한 단계씩 늘린다.
- raw 목표가 낮아지거나 Fast 위험이 재발하면 확인일을 0으로 초기화한다.
- 권고비중과 raw 목표가 모두 Green(90% 이상)에 도달하면 위험감축 episode를 종료한다.

## 승격 제한

- v1·현재 공개 정책과 운영 DB를 덮어쓰지 않는다.
- 미래정보 미사용, 비용 10/25bp, MDD·수익률·Sharpe·회전율, 재진입 실패율을
  Historical OOS로 다시 검증한다.
- 24시간 cooling-off와 126거래일 shadow 전에는 공개 CHAMPION으로 승격하지 않는다.
- posterior clipping·temperature scaling은 사용하지 않는다.
