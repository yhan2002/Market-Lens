# Model Promotion & Rollback Runbook

Registry: `MODEL_GOVERNANCE.json`  
Rule engine: `model_governance.py`

현재 상태는 다음과 같다.

- Active: `hmm3-sticky-nq-core-v1-k25-m252-w1260-r20-s42-n3`
- Rollback 1: `hmm3-sticky-v2-spy-k25-m252-w1260-r20-s42-n3`
- Rollback 2: `hmm3-sticky-v2-spy-k25-m252-w1260-r20-s42`
- q20: `REJECTED` (Historical gate 실패)
- Router-State policy v4: `REJECTED`; slow gate `FAIL`, full PDF gate `NOT_EVALUATED`
- Router-State policy v5: `CANDIDATE`; 동일 순차 엔진 구현 완료, Historical replay 실행 대기
- PDF8: `DATA_BLOCKED`
- Independent A2 HMM: `RESEARCH SHADOW`; 구현 완료, Historical 검증 전

현재 실패 근거는 `outputs/hmm_nq_core_v1_validation.md`,
`outputs/hmm_nq_tied4_v2_validation.md`, `outputs/q20_validation.md`,
`outputs/policy_oos.md`, `outputs/policy_v4_router_state/policy_oos.md`에 보존되어
있습니다. `REJECTED` 또는 `DATA_BLOCKED` 후보에는 아래 cooling-off/Shadow 명령을
실행하지 않습니다.

v5는 `POLICY_REPLAY_v5_RUNBOOK.md`에 따라 별도 결과 디렉터리에 실행한다. 기존 v4
보고서는 v5 승격 또는 투자설명회 성과 근거로 대체 사용할 수 없다.

Independent A2 HMM은 `MODEL_SPEC_HMM_A2.md`의 3개 입력만 사용한다. 운영 HMM을
교체하지 않으며, 별도 DB·model version에서 적합한 뒤 현재 HMM과 Historical
검증을 비교한다. 장시간 적합 명령은 해당 사양서에 고정한다.

Registry 명령은 승인 원장을 갱신할 뿐 운영 코드의 `HMM_MODEL_VERSION`을 자동으로
바꾸지 않는다. 실제 모델 교체는 승인 뒤 별도 코드 변경·smoke test로 수행한다.

## 1. 신규 후보 등록

Historical OOS JSON의 `status`가 `HISTORICAL_PASS`이고 모든 `gates`가 true여야
`COOLING_OFF`로 등록된다. 아니면 자동으로 `REJECTED`다.

보고서에 기록된 역할과 정확히 같은 소스만 넘긴다. 예를 들어 q20은 다음과 같다.

```powershell
.\.venv\Scripts\python.exe .\model_governance.py register .\MODEL_GOVERNANCE.json .\outputs\q20_validation.json `
  --source .\MODEL_SPEC_v1.md --source .\research_transition_risk.py
```

## 2. 24시간 cooling-off 후 clean 재실행

최소 24시간 뒤 같은 frozen command로 Historical OOS를 새 파일에 다시 실행한다.
그동안 source hash 또는 model design이 하나라도 바뀌면 `start-shadow`가 실패한다.

```powershell
.\.venv\Scripts\python.exe .\model_governance.py start-shadow .\MODEL_GOVERNANCE.json <MODEL_ID> <CLEAN_REPORT.json> `
  --source .\MODEL_SPEC_v1.md --source <해당 역할의 고정 소스.py>
```

## 3. 126거래일 Shadow 기록

매 거래일 한 행만 기록한다. 당일 검증이 하나라도 실패하면 모델은 즉시
`SHADOW_FAILED`가 된다. 기간 중 `DE_RISK`와 `REENTRY`가 각각 최소 1회 필요하다.
126거래일에 cycle이 부족하면 관찰을 연장하되, 252번째 거래일까지 두 cycle이
완성되지 않으면 `SHADOW_WINDOW_EXPIRED`로 실패 처리하고 새 candidate version으로
Historical 검증부터 다시 시작한다. 253번째 기록과 뒤늦은 승격은 허용하지 않는다.

`<DAILY_EVIDENCE.json>`에는 같은 날짜의 `"as_of": "YYYY-MM-DD"`가 있어야 한다.

```powershell
.\.venv\Scripts\python.exe .\model_governance.py record .\MODEL_GOVERNANCE.json <MODEL_ID> 2026-08-11 `
  --event NONE --result PASS --evidence <DAILY_EVIDENCE.json>
```

`--event`는 `NONE`, `DE_RISK`, `REENTRY` 중 하나다.

## 4. 승격 승인

126~252개의 고유 거래일이 모두 PASS이고 두 cycle이 모두 있어야 성공한다.

```powershell
.\.venv\Scripts\python.exe .\model_governance.py promote .\MODEL_GOVERNANCE.json <MODEL_ID> `
  --source .\MODEL_SPEC_v1.md --source <해당 역할의 고정 소스.py>
```

성공 후에만 운영 상수 변경, 전체 단위 테스트, `site_publisher.py --skip-refresh
--no-upload` smoke test를 수행한다.

## 5. Rollback

원장은 최근 정상 모델 두 개만 유지한다.

```powershell
.\.venv\Scripts\python.exe .\model_governance.py rollback .\MODEL_GOVERNANCE.json `
  --role CURRENT_REGIME
```

명령 성공 후 원장에 표시된 active version으로 운영 상수를 되돌리고 같은 smoke
test를 수행한다. DB의 해당 version 행은 삭제하지 않는다.

## 6. 상태 확인

```powershell
.\.venv\Scripts\python.exe .\model_governance.py show .\MODEL_GOVERNANCE.json
```
