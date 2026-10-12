# Independent HMM A2-G Four-State Preflight

Status: **PREFLIGHT_FAIL — FULL WALK-FORWARD NOT RUN**  
Planned benchmark: `NQ=F`

## Frozen design

A2-F를 기준으로 latent state 수만 3개에서 4개로 늘리는 SHADOW ablation이었다.
학습구간의 composite risk centroid 순서로 Goldilocks / Normal / Inflationary Bear /
Crash를 정의하고, 공개 확률은 `BULL=Goldilocks`, `SIDEWAYS=Normal`,
`BEAR=Inflationary Bear+Crash`로 합산하도록 사전 고정했다.

숨은 상태 붕괴를 기존 3확률 validator가 놓치지 않도록 각 refit에서 다음 조건을
모두 만족하는 seed만 허용하도록 구현했다.

- 각 latent hard-state 학습 점유율 5~80%
- 인접 normalized risk centroid gap 0.25 이상
- 수렴 및 확률합 1

## 결과를 보지 않은 사전 적합

최초·중간·최신 refit 창에서 seed 42·43·44 총 9회만 적합했다.

| Refit date | Train rows | Latent occupancy | Minimum centroid gap | Admissible seeds |
|---|---:|---|---:|---:|
| 2008-06-23 | 252 | 9.1% / 26.2% / 32.1% / 32.5% | 0.066 | 0/3 |
| 2017-05-31 | 1,260 | 15.8% / 20.8% / 21.9% / 41.5% | 0.037 | 0/3 |
| 2026-08-07 | 1,260 | 12.5% / 13.6% / 27.1% / 46.8% | 0.429 | 3/3 |

네 상태의 표본 점유율은 충분했지만 과거 두 창에서 인접 상태의 위험 centroid가
거의 같았다. 최신 시장에서만 네 상태가 분리되므로, 전 역사에 네 상태를 강제하면
경제적 의미가 없는 중복 상태를 만드는 결과다.

## 결정

사전동결 guard를 결과에 맞춰 완화하지 않았다. 전체 229-refit 실행과 DB 저장을
시작하지 않고 A2-G를 기각했다. 다음 연구는 상태 수가 아니라 새로운 독립 정보축을
추가하는 방향으로 이동한다.

## 재현 명령과 해시

```powershell
.\.venv\Scripts\python.exe .\research_four_state_preflight.py `
  .\outputs\shadow_hmm_a2f\a2f_input_20260813.db `
  --json .\outputs\shadow_hmm_a2g\preflight.json
```

- input clone SHA256: `81514361AAB6E81394ACB02AFA6D95E59AC3E71DBC15B1ACD43D05D426A8387D`
- preflight script SHA256: `D0D6FA7CFA305EF2AFE9B1F69373462ACF6BB673E1A3C3109F35FEEE8619D1B3`
- result JSON SHA256: `BC7BB441720D04CB7441E78391C858DF14056431D0C15C42B650246083A39A8B`
- run log SHA256: `1E0663C75981E47500BB9F4CA9E7693146C6FC13351A4DAE75C8193F9F257D1E`
