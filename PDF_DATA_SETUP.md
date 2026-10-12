# PDF Risk Model exact 데이터 준비

새 정책은 QQQ, HYG/IEF, RSP/SPY 같은 proxy를 사용하지 않습니다. 입력이 없으면
해당 조건을 임의로 채우지 않고 `UNAVAILABLE`로 둡니다. 아래 작업은 코드와 테스트만
준비했으며 실제 외부 다운로드는 실행하지 않았습니다.

## 1. 현재 운용 화면을 완성하는 초기 데이터

### HY OAS

FRED의 `BAMLH0A0HYM2`를 percentage-point 단위로 저장합니다. 현재 FRED 공개
범위는 최근 3년으로 제한되므로 최신 Fast/Re-entry 신호에는 쓸 수 있지만 2008년
이후 HMM 검증 원장으로는 부족합니다.

일반 `fetch-hy-oas`는 실행한 시각을 모든 내려받은 행의 `available_at`으로 보존합니다.
따라서 실행일 이후의 live 수집에는 적합하지만 과거 화면 복구용으로 소급하지 않습니다.
과거 공개일은 공식 ALFRED의 new/revised vintage(`output_type=3`)를 사용합니다.
웹 다운로드 경로는 API key 없이 동작하며, HTML에서 실제 제공 vintage 날짜를 확인한
뒤 공식 ZIP CSV만 받아 저장합니다. HTML 오류 화면이나 잘못된 ZIP은 DB에 쓰지 않습니다.

```powershell
.\.venv\Scripts\python.exe .\macro_data.py fetch-hy-oas .\regime_router_publish.db --start 2023-08-10 --end 2026-08-10

.\.venv\Scripts\python.exe .\macro_data.py backfill-hy-oas-alfred .\regime_router_publish.db --start 2025-12-01 --end 2026-08-11

# 선택 경로: 무료 FRED API key가 있을 때 동일 output_type=3을 직접 조회
$env:FRED_API_KEY="발급받은 키"
.\.venv\Scripts\python.exe .\macro_data.py backfill-hy-oas-vintages .\regime_router_publish.db --start 2025-12-01 --end 2026-08-11
Remove-Item Env:FRED_API_KEY
```

vintage 백필은 ALFRED 열의 vintage 날짜 또는 FRED API의 `realtime_start`를 날짜
단위 `available_at`으로 저장합니다.
예를 들어 7월 1일 관측값이 7월 2일 공개됐다면 7월 1일 판단에는 6월 30일 값을,
7월 2일부터 7월 1일 값을 사용합니다. 기존 과거 웹 이력은 수정하지 않습니다. 백필
검증 후 정책/HMM 버전은 유지하고 snapshot `series_version`만 새 suffix로 올려
2026년 1월부터 순차 재생하며, 기존 시리즈는 rollback용으로 보존합니다.

### S&P 500 Breadth

Bloomberg 연결 전에는 PDDL 공개 데이터셋의 현재 구성종목 목록을 매일 저장하고,
그날부터 Yahoo 수정주가로 live breadth를 계산할 수 있습니다. 공개 목록은 과거 구성
이력이 아니므로 처음 관측한 날짜보다 앞선 구간으로 소급하지 않습니다. 일반 사이트
최신화는 이후 구성종목 변경을 자동으로 누적합니다.

최초 1회 아래 순서로 가격을 채웁니다. 약 500종목 × 2년 다운로드이므로 이 명령은
사용자가 명시적으로 실행합니다. 구성표 최초 수집일의 미국 종가가 들어오면 breadth
level이 생기고, 그 다음 거래일 종가부터 `시장폭 악화/반전` 판단이 가능합니다.

```powershell
.\.venv\Scripts\python.exe .\breadth_metrics.py sync-current-open-data .\regime_router_publish.db
.\.venv\Scripts\python.exe .\breadth_metrics.py register-active .\regime_router_publish.db
.\.venv\Scripts\python.exe .\market_data.py download .\regime_router_publish.db --group-name SP500_BREADTH --years 2 --backfill
.\.venv\Scripts\python.exe .\breadth_metrics.py compute .\regime_router_publish.db
```

Bloomberg 또는 라이선스 PIT 이력이 들어오면 아래 정식 경로가 공개 현재목록보다
우선합니다. 공개 목록을 사용하기 시작한 기간 전체를 490종목 이상으로 완전하게
덮는 PIT 파일만 자동 전환되며, 일부 종목 파일은 기존 live 기록을 보호하기 위해
거부됩니다.

`data_templates\sp500_membership_template.csv`와 같은 여섯 열을 가진 파일이
필요합니다. `effective_from/effective_to`는 실제 편입 구간이고 `effective_to`는
포함일입니다. `membership_as_of`는 구성 정보의 기준일, `available_at`은 그 정보가
당시 이용 가능해진 날짜입니다. source는 `LICENSED_PIT:` 또는 `BLOOMBERG_PIT:`로
시작해야 하며 행마다 보존합니다. 이 provenance가 없는 기존 3열 파일은 exact로
받지 않습니다. 현재 구성종목만 받은 경우에는 확인 가능한 시작일 이후의 **live
지표**만 만들고, 현재 구성종목을 2008년까지 소급하지 않습니다. Yahoo 표기와 다른
클래스 주식 기호는 가격 공급자 기호(예: `BRK-B`)로 맞춰야 합니다.

```powershell
.\.venv\Scripts\python.exe .\breadth_metrics.py import-membership .\regime_router_publish.db "<LOCAL_DATA_FILE>" --source "LICENSED_PIT:vendor_export_YYYYMMDD"
.\.venv\Scripts\python.exe .\breadth_metrics.py register-range .\regime_router_publish.db --start 2024-08-10 --end 2026-08-10
.\.venv\Scripts\python.exe .\market_data.py download .\regime_router_publish.db --group-name SP500_BREADTH --years 2 --backfill
.\.venv\Scripts\python.exe .\breadth_metrics.py compute .\regime_router_publish.db
```

`register-range`는 현재 구성종목뿐 아니라 해당 2년 사이 편출된 종목도 포함합니다.
현재 종목만 등록하면 과거 breadth가 생존편향되므로 `register-active`는 과거 계산에
사용하지 않습니다. exact PIT breadth는 한 종목이라도 계산 이력이 빠지면 저장하지
않고, 공개 live breadth는 계산 가능 종목이 98% 이상일 때 실제 coverage와 함께
저장합니다. 초기 가격 수집 예상 범위는 약 500~600종목 × 2년(약 25만~30만
행)이며 환경에 따라 대략
5~20분입니다. 무료 Yahoo 요청이지만 rate limit이나 일부 종목 누락이 날 수 있습니다.
초기 수집 뒤 일일 작업은 최근 며칠만 증분 갱신합니다.

### Portfolio Correlation

`ss 가공용 v2`는 포지션이 바뀔 때만 갱신되는 event snapshot입니다.
MP 원본 `<LOCAL_PROJECT_PATH>`만 사용합니다. 이전 수동 복사본과 U: 드라이브 원본은 읽지 않으며 WRAP 입력은 연결하지 않았습니다. 파일의 저장일이 10일을 넘으면 확인 경고를 표시합니다.
Windows 예약 작업은 원본 접근을 최대 3회 확인한 뒤 Excel을 읽기 전용으로 열어
`%LOCALAPPDATA%\RegimeRouter\private\SS 가공용 v2.xlsx`에 비공개 캐시를 만든 뒤
실제 최신 포지션이 바뀌었는지 확인하여 `regime_router_shadow.db`에 반영합니다. 새 캐시는 별도 임시 파일에서 양식·기준일 검증을 통과해야 교체됩니다. 자동 동기화는 입력 날짜 밖의 기존 백필 이력을 보존합니다. 원본
연결이 끊기면 마지막 확인 포지션을 계속 사용하고 부분 완료 경고를 남깁니다. 보유내역 기준일은 시장 가격일과 구분해 표시합니다. 보유종목 가격은
매일 비공개 DB에서 갱신해 20일 평균 상관과 beta를 다시 계산합니다.
보유종목·수량·비중·내부 경로는 게시 DB와 JSON에 복사하지 않습니다. 가격 weight
coverage가 90%보다 낮으면 계산하지 않습니다. 일일 snapshot으로 지정된 다른
보유내역 source에는 기존 7일 제한을 유지합니다. 미국 종목의 평가금액이 0이면
보유수량×최신 조정주가로 비중을 복원하며 외국통화 종목에는 이 보정을 적용하지
않습니다.

```powershell
.\.venv\Scripts\python.exe .\portfolio_risk.py compute .\regime_router_shadow.db --prices-database .\regime_router_publish.db --as-of 2026-08-07
```

일반 게시 실행에서는 private holdings와 private prices를 결합해 aggregate만 자동
동기화하므로 위 명령은 진단용입니다.

## 2. 2008년 이후 8변수 HMM 검증에 추가로 필요한 원자료

다음 두 파일은 무료 공개 데이터만으로 point-in-time 완전성을 보장할 수 없습니다.

1. 2008년 이후 S&P 500 종목별 실제 편입·편출 구간과 상장폐지 종목을 포함한
   수정주가 이력
2. 2008년 이후 HY OAS의 `date`, `value`(percentage points), 실제
   `available_at`이 포함된 전체 이력

두 원자료를 받기 전에는 PDF 8변수 HMM을 현행 HMM 대신 승격하지 않습니다. 현행
대시보드의 최신 Fast/Re-entry 계산은 최근 exact 데이터가 준비된 날짜부터 동작합니다.

HY OAS 라이선스 파일은 다음처럼 가져올 수 있습니다.

```powershell
.\.venv\Scripts\python.exe .\macro_data.py import-hy-oas .\regime_router_publish.db "<LOCAL_DATA_FILE>" --source "ICE or Bloomberg export"
```

템플릿은 `data_templates\hy_oas_pit_template.csv`입니다. `available_at`이 없는
과거값은 당시 알 수 있었던 값인지 확인할 수 없어 HMM 검증에는 넣지 않습니다.

## 3. Position Quality Score

Bloomberg 결합 전에는 Fundamental Momentum과 Earnings Revision을 운용자가
0~100으로 입력합니다. Relative Strength와 Liquidity도 같은 척도로 입력하며,
코드는 PDF의 40% / 25% / 20% / 15% 가중치로 자연수 점수를 계산합니다.

```powershell
.\.venv\Scripts\python.exe .\portfolio_risk.py score-quality "<LOCAL_DATA_FILE>" "<LOCAL_DATA_FILE>"
```

이 파일은 보유종목 정보이므로 게시 DB나 Git에 넣지 않습니다.
