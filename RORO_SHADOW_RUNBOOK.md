# Market Lens RORO SHADOW 재현 절차

상태: **LOCAL SHADOW ONLY — 외부 게시와 운영 배분 변경 금지**  
기준 시장: `NQ=F`  
표시 시작일: `2026-01-07`

아래 명령은 프로젝트 폴더의 PowerShell에서 실행한다. `--as-of`는 확인된 최신 NQ
거래일로 바꾼다. 원천 DB는 읽기 전용으로 사용하고 결과는 `outputs` 아래의 별도
SHADOW DB에만 저장한다.

## 0. 안전 실행 래퍼(권장)

아래 첫 명령은 **계획만 표시**한다. 폴더 생성, 테스트, 다운로드, DB 처리, 업로드,
배포를 전혀 시작하지 않는다.

```powershell
.\RUN_RORO_SHADOW_LOCAL.ps1
```

사용자가 전체 로컬 실행을 명시적으로 승인한 뒤에만 `-Execute`를 붙인다. 실행마다
`outputs` 아래에 새 staging 폴더를 만들며, 같은 이름이 이미 있으면 덮어쓰지 않고
중단한다.

```powershell
.\RUN_RORO_SHADOW_LOCAL.ps1 -Execute -AsOf 2026-08-12 -DataEnd 2026-08-12 -FetchAsOf 2026-08-14T00:00:00Z -BootstrapSamples 500
```

래퍼는 아래 1~7절의 명령을 순서대로 실행하고 비정상 종료 코드를 만나면 즉시
중단한다. 원본 DB 전후 해시, 복제본, 코드·명세 해시, 환경, 정확한 인자와 단계별
종료 코드, 산출물 해시를 `run.log`, `logs\`, `run_manifest.json`에 남긴다.
실행 전에 `site\node_modules`와 사이트 build/test 필수 입력을 확인한다. 코드 스냅샷은
`requirements.txt`, 사이트 설정, app/lib/db/build/worker/tests 소스, Drizzle SQL·meta,
public 자산을 포함하며 데이터 카탈로그 CSV가 있으면 명세 스냅샷에도 포함한다.
게시 단계는 항상 `--no-upload`이며 외부 업로드와 사이트 배포 명령은 포함하지 않는다.
실패한 staging 폴더도 삭제하지 않아 원인을 확인할 수 있다.

## 1. 빠른 코드 검증

```powershell
Get-ChildItem -File .\roro_*.py, .\sector_*shadow*.py | ForEach-Object { & .\.venv\Scripts\python.exe -m py_compile $_.FullName }
.\.venv\Scripts\python.exe -m unittest test_roro_data.py test_roro_engine.py test_roro_ew_shadow.py test_roro_portfolio.py test_roro_axis.py test_roro_phase.py test_roro_q20.py test_roro_policy.py test_roro_price_observations.py test_roro_shadow.py test_roro_site_publisher.py test_roro_validation_report.py test_roro_data_catalog.py test_run_roro_shadow_local.py test_sector_rotation.py test_sector_rotation_forecast.py test_sector_roro_shadow.py test_sector_shadow_store.py test_sector_site_publisher.py
```

정식 카탈로그가 동결된 뒤에는 실제 파일도 실행 전에 검증한다.

```powershell
.\.venv\Scripts\python.exe .\roro_data_catalog.py .\RORO_DATA_CATALOG.csv
```

## 2. 복제 DB와 무료 데이터 준비

운영 DB를 직접 수정하지 않는다. 아래 전용 staging 폴더가 이미 있으면 중단하고 새
이름을 사용한다.

```powershell
$stage = Join-Path (Get-Location) 'outputs\roro_data_fill_20260814'
if (Test-Path -LiteralPath $stage) { throw "staging folder already exists: $stage" }
New-Item -ItemType Directory -Path $stage | Out-Null
Copy-Item -LiteralPath '.\regime_router_publish.db' -Destination "$stage\publish_clone.db"
Copy-Item -LiteralPath '.\regime_router_shadow.db' -Destination "$stage\private_clone.db"
$sourceHashesBefore = Get-FileHash '.\regime_router_publish.db', '.\regime_router_shadow.db' -Algorithm SHA256
$sourceHashesBefore | Format-Table Path, Hash
```

HY OAS는 공식 무료 ALFRED vintage를 복제 DB에만 채운다. 예상 신규 330~350행,
1~5분, API key와 비용은 모두 0원이다.

```powershell
.\.venv\Scripts\python.exe .\macro_data.py backfill-hy-oas-alfred "$stage\publish_clone.db" --start 2024-08-01 --end 2026-08-12
```

FX·Gold 세션 관측치를 처음 채울 때 사용할 준비된 명령은 아래와 같다. Yahoo Chart
2종목 약 1,020~1,070행, 보통 10~60초, API 비용 0원으로 예상한다. 현재 작업에서는 실제로
실행하지 않았다.

```powershell
.\.venv\Scripts\python.exe .\roro_price_observations.py download "$stage\roro_price_observations.db" --start 2024-08-01 --end 2026-08-12 --as-of 2026-08-14T00:00:00Z
```

`--start/--end`는 양 끝을 포함하고 `--as-of`는 UTC timezone이 명시돼야 한다. 같은
명령의 재실행만 idempotent다. 같은 과거 구간을 다른 `--as-of`로 다시 받으면
`fetched_at`이 달라 불변행 충돌로 중단된다. 이 경우 기존 행을 덮어쓰지 말고 새
구간만 이어받거나, revision이 필요하면 새 contract version을 먼저 승인한다.

r6는 이 관측 DB가 명시되고 동결 계약 행이 하나 이상 있을 때만 생성된다.
미지정·빈 DB는 fail-closed하고, 개별 날짜의 관측 결손은 임의로 채우지 않아
해당 블록을 `DATA_BLOCKED`로 남긴다. r3~r5는 같은 이름으로 재계산하지
않고 명시적 조회용으로 계속 보존한다.

## 3. RORO 불변 SHADOW 생성

전체 역사 재생성은 150개 NQ 거래일, 약 1~3분, 외부 API 비용 0원이다.

```powershell
$stage = Join-Path (Get-Location) 'outputs\roro_data_fill_20260814'
.\.venv\Scripts\python.exe .\roro_shadow.py build "$stage\publish_clone.db" "$stage\private_clone.db" "$stage\roro_shadow_r6.db" --price-observation-database "$stage\roro_price_observations.db" --as-of 2026-08-12
```

동일 series/date를 같은 내용으로 다시 실행하면 idempotent이고, 다른 내용으로
덮어쓰려 하면 실패한다. 현재 r6는 날짜별 q20 readiness와 Fast/re-entry 정책 audit,
FX·Gold 관측 계약 버전을 함께 저장하며 r3~r5는 그대로 보존한다. 새 사양은
기존 series를 수정하지 않고 새 version으로 쓴다.

## 4. RORO 인과 검증

500회 moving-block bootstrap은 약 1~3분, 외부 API 비용은 0원이다.

```powershell
$stage = Join-Path (Get-Location) 'outputs\roro_data_fill_20260814'
.\.venv\Scripts\python.exe .\roro_validation_report.py "$stage\publish_clone.db" "$stage\roro_shadow_r6.db" --as-of 2026-08-12 --bootstrap-samples 500 --output-directory "$stage\validation"
```

`status`, 공통 표본, q20 readiness, comparator와 10bp 결과를 확인한다. q20 최소
표본 `252 labels / 20 independent events`가 미달이면 확률은 반드시 `null`이다. 현재
q20은 readiness-only 단계이므로 기준을 채워도 실제 fitter가 승인되기 전까지 확률은
계속 `null`이다.

## 5. Sector RORO 검증과 불변 저장

검증은 약 1~3분, 저장은 보통 1분 이내이며 외부 API 비용은 0원이다.

```powershell
$stage = Join-Path (Get-Location) 'outputs\roro_data_fill_20260814'
.\.venv\Scripts\python.exe .\sector_roro_shadow.py validate "$stage\publish_clone.db" "$stage\roro_shadow_r6.db" --as-of 2026-08-12 --bootstrap-samples 500 --output "$stage\sector_validation.json"
.\.venv\Scripts\python.exe .\sector_shadow_store.py build "$stage\publish_clone.db" "$stage\roro_shadow_r6.db" "$stage\sector_shadow_r6.db" --as-of 2026-08-12
```

52주와 SELECT 30건이 채워지기 전에는 `INSUFFICIENT_EVIDENCE`가 정상이다. 학습이
불가능해 건너뛴 주는 날짜와 사유를 보고서에 남긴다.

## 6. 공개용 JSON 생성 — 업로드 없음

```powershell
$stage = Join-Path (Get-Location) 'outputs\roro_data_fill_20260814'
.\.venv\Scripts\python.exe .\roro_site_publisher.py --publish-database "$stage\publish_clone.db" --private-database "$stage\private_clone.db" --shadow-database "$stage\roro_shadow_r6.db" --price-observation-database "$stage\roro_price_observations.db" --as-of 2026-08-12 --no-upload --output "$stage\roro_site_payload.json"
.\.venv\Scripts\python.exe .\sector_site_publisher.py --shadow-database "$stage\sector_shadow_r6.db" --as-of 2026-08-12 --no-upload --output "$stage\sector_site_payload.json"
```

개별 보유종목 identity가 JSON에 포함되지 않았는지 확인한다. `--no-upload`를 빼도
`roro_publish_consent.json`에 명시 동의가 없으면 두 게시자는 외부 전송을 보류한다.

## 7. 로컬 사이트 검증

```powershell
Set-Location .\site
npm test
```

마지막으로 프로젝트 폴더로 돌아와 원본 DB가 바뀌지 않았는지 해시를 강제 확인한다.
같은 PowerShell 세션이 아니라면 위 `$sourceHashesBefore` 값을 별도 파일에 보존한 뒤
비교해야 한다.

```powershell
Set-Location ..
$sourceHashesAfter = Get-FileHash '.\regime_router_publish.db', '.\regime_router_shadow.db' -Algorithm SHA256
foreach ($before in $sourceHashesBefore) {
    $after = $sourceHashesAfter | Where-Object Path -eq $before.Path
    if ($null -eq $after -or $after.Hash -ne $before.Hash) { throw "source DB changed: $($before.Path)" }
}
```

외부 D1 업로드, ACTIVE series 변경, Sites 배포는 별도 사용자 승인 후에만 진행한다.
