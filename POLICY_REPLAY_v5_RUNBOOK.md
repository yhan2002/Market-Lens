# 동일 규칙 과거 성과 산출 절차

## 목적

실시간 화면, 과거 날짜 조회, 과거 성과검증이 모두 같은 순차 비중 조절 규칙을 사용한다.
신호가 나온 날의 종가를 미리 체결한 것으로 보지 않고 다음 NQ 거래일부터 비중을 반영한다.

고정 시리즈:

`risk-policy-v5.1-posterior-weighted-candidate__hmm3-sticky-nq-core-v1-k25-m252-w1260-r20-s42-n3__r3`

`r3`는 2026-01-02부터 point-in-time HY OAS와 포트폴리오 상관 집계를 포함해
전체 이력을 다시 게시한 시리즈다. `r2`와 `r1`은 롤백용으로 보존한다.

기존 v4 보고서는 현재 화면과 다른 계산 경로를 사용했으므로 투자설명회 근거로 재사용하지 않는다.

## 실행

가장 간단한 방법은 `Create_Same_Logic_Performance_Report.cmd`를 더블클릭하는 것이다.

PowerShell에서는 다음 명령을 사용한다.

```powershell
.\.venv\Scripts\python.exe .\policy_oos.py .\regime_router_publish.db --output-dir .\outputs\policy_v5_1_weighted
```

실행 환경만 빠르게 확인하려면 다음 명령을 사용한다. 정상이라면 `preflight=OK`와
가격·확률 행 수가 표시된다.

```powershell
cmd.exe /d /c call ".\Create_Same_Logic_Performance_Report.cmd" --check
```

- 외부 API 호출 및 사이트 업로드 없음
- 게시 DB는 읽기 전용으로 열림
- 약 4천~5천 거래일을 순서대로 계산하므로 PC에 따라 수 분 이상 소요 가능

## 결과물

- `policy_oos_presentation.html`: 투자설명회 캡처용 한 페이지
- `policy_oos.md`: 검증 결과 요약
- `policy_oos.json`: 판정·버전·원본 파일 해시
- `policy_oos_daily.csv`: 날짜별 비중·수익률·누적성과·낙폭

대표 화면은 거래비용 10bp 적용 결과와 NQ 기준 성과를 비교한다. 누적성과뿐 아니라 최대
낙폭과 위험조정 성과를 함께 제시한다.

## 문구 사용 기준

- `Full PDF gate = PASS`: “현재 규칙과 전체 입력을 동일하게 적용한 과거 순차 검증”이라고 표시 가능
- `Full PDF gate = NOT_EVALUATED`: “가용 데이터 기준 동일 규칙 순차 재생”으로만 표시
- `Historical metric gate = FAIL`: 홍보 수치로 사용하지 않고 원인 분석 대상으로 분류

Bloomberg 이력이 없는 Fast·재진입 지표는 임의 프록시로 채우지 않는다. 따라서 완전한 입력
검증 전까지 산출 화면에도 `가용 데이터 기준 참고 결과` 배지가 자동 표시된다.

## 게시 전 절차

1. HTML·MD·JSON의 기간, 비용, 데이터 완비율을 확인한다.
2. 동일 명령을 24시간 뒤 다시 실행해 결과가 같은지 확인한다.
3. 승인된 시리즈만 사이트의 `ACTIVE_POLICY_SERIES`로 지정한다.
4. 새 시리즈의 최초 행은 전체 재생, 이후 행은 저장된 직전 상태만 이어받는다.
5. 과거 행은 덮어쓰지 않는다. 규칙이 바뀌면 새 시리즈 버전으로 다시 산출한다.

사이트 백필과 배포는 외부 상태를 변경하므로 성과검증 승인 뒤 별도로 실행한다.
