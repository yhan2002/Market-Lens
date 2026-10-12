# Market Lens NQ Rule-Based Risk Score Candidate v1

Status: **RESEARCH SHADOW / NOT APPROVED**  
Effective date: 2026-08-13  
Model version: `nq-rule-score-v1-w756-m252`

## 목적

HMM posterior가 한 상태에 장기간 포화될 때도 시장 추세·변동성·낙폭의 변화를
연속적으로 표시한다. 이 점수는 미래 하락확률이 아니며, 결과값으로 계수나
임계값을 학습하지 않는다. HMM은 국면 표시와 Risk-off 재진입 더블체크에 남긴다.

## 고정 산식

NQ 가격으로 다음 여섯 원자료를 계산한다.

- 위험 방향 20일·60일 모멘텀: `-NQ return`
- 위험 방향 200일선 거리·20일 기울기: 각 값의 음수
- 20일 연율 변동성
- 위험 방향 60일 낙폭: `-drawdown`

각 날짜의 원자료는 **해당 날짜를 제외한** 직전 756개 유효 관측치에서 percentile
rank로 바꾼다. 최소 252개 과거 관측치가 없으면 점수를 내지 않는다.

- `TREND = mean(momentum20 rank, momentum60 rank, MA200 distance rank, MA200 slope rank)`
- `VOLATILITY = volatility20 rank`
- `DRAWDOWN = -drawdown60 rank`
- `Risk Score = round_half_up(100 × mean(TREND, VOLATILITY, DRAWDOWN))`

세 블록은 동일 가중치다. 점수는 0~100 자연수이며 높을수록 위험하다. 기존 Long
비중 anchor와 Fast Overlay·단계식 재진입 규칙은 변경하지 않는다.

## 버전·승격 제한

- 현재 공개 정책과 HMM 테이블을 덮어쓰지 않는다.
- Historical OOS에서 미래정보 미사용, 1,000일 이상, 향후 20일 낙폭 AUC CI,
  단순 변동성 비교, 비용 10/25bp, MDD·수익률·Sharpe·회전율을 검증한다.
- 신호 gate는 AUC CI 하한 `>0.50`, top-quintile lift `>=1.25`, HMM 표시점수
  대비 paired AUC point `>=0`·CI 하한 `>=-0.02`, 20일 변동성 AUC 대비 point
  `>=-0.02`로 고정한다. 최근 63일 점수 고유값은 8개 이상, 변화일은 12일
  이상, 최빈 점수 비중은 50% 이하여야 한다.
- 24시간 cooling-off와 126거래일 shadow 전에는 공개 CHAMPION으로 승격하지 않는다.
- posterior clipping·temperature scaling은 사용하지 않는다.
