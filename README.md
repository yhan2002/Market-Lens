# Market Lens

시장 국면, 위험 신호, 포트폴리오 상태와 섹터 순환을 연구하는 Python 도구와 보호된 웹 대시보드입니다. HMM 국면 추정, 위험 정책 재생, RORO/섹터 shadow 모델, 데이터 품질 검사와 게시 도구를 포함합니다. `site/`에는 별도 뉴스 파이프라인이 전송한 결과를 표시하는 화면도 있습니다.

이 저장소는 기존 작업의 **소스 스냅샷**입니다. 모델 계산과 연구 코드는 보존했지만 운영 DB, 실계좌·보유내역, 기사, 모델 승인 이력, 게시 동의, 비밀 설정, 캐시와 생성 보고서는 포함하지 않습니다. 입력과 설정 없이 운영 결과를 재현하거나 운영 서비스에 바로 연결할 수 없습니다. 연구 신호·shadow 결과는 주문 실행이나 수익을 보장하지 않습니다.

날짜가 고정된 내부 운영현황·회의 내용을 본문에 내장한 보고서 생성기도 제외했습니다. 나머지 보고서 도구는 연구 방법·입력·계산을 설명하는 소스이며 생성 결과물은 포함하지 않습니다.

## 구조

| 경로 | 역할 |
| --- | --- |
| `regime_router.py`, `regime_router_app.py` | 국면 추정과 로컬 UI |
| `market_data.py`, `macro_data.py`, `breadth_metrics.py` | 시장·거시·시장폭 데이터와 SQLite 스키마 |
| `risk_*.py`, `portfolio_risk.py`, `roro_*.py`, `sector_*.py` | 위험·포트폴리오·섹터 연구 및 shadow 계산 |
| `model_governance.py`, `MODEL_SPEC*.md` | 연구 모델 등록 규칙과 설계 문서 |
| `reports/*.py` | 보고서·연구 도구 소스; 결과물은 제외 |
| `data_templates/` | 열 이름만 담긴 입력 CSV 템플릿 |
| `site/` | React/Vinext 화면, Cloudflare worker, D1 migration, 웹 테스트 |
| `tools/run_offline_tests.py`, `test_*.py` | 외부 연결을 차단하는 Python 단위 테스트 |

## 설치와 오프라인 확인

Apple Silicon M1 Mac에서는 [macOS 설치·검증 안내](docs/MACOS.md)를 먼저 읽으세요. Python 연구 코드와 로컬 웹 화면의 준비 절차, Windows 전용 통합 갱신기의 제한을 구분해 설명합니다.

Python **3.12 이상**이 필요합니다. 일부 연구 모듈은 Python 3.12 f-string 문법을 사용합니다. Windows PowerShell에서:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -B tools/run_offline_tests.py
.\.venv\Scripts\python.exe regime_router.py --help
```

테스트는 합성 입력·임시 DB·mock을 사용하며 네트워크와 외부 프로세스 실행을 차단합니다. 실제 PowerShell을 실행하는 `test_run_roro_shadow_local.py`는 이 runner에서 제외합니다. PDF/Word 보고서 도구에는 선택적으로 `reportlab`, `pypdf`, `Pillow`, `python-docx`가 필요할 수 있고 Bloomberg 도구에는 공급자 SDK가 필요합니다. 기본 requirements는 모든 선택 보고서·상용 연동 도구의 설치를 보장하지 않습니다.

게시 준비 검증에서는 Python 3.12.14의 기존 의존성 환경을 읽기 전용으로 사용했습니다. Python suite의 기존 실패 두 건은 그대로 남아 있습니다: `test_risk_signals`의 JPY 신호 상태 기대값과 `test_site_publisher`의 정책 버전 suffix 기대값이 현재 코드와 일치하지 않습니다. 원본 파일을 복사한 별도 격리본에서도 같은 실패를 재현했으며 전략 의미를 바꿔 통과시키지 않았습니다. 별도 `News Alarm` 프로젝트의 helper가 없으면 관련 테스트 클래스 한 개는 skip합니다.

웹은 Node.js **22.13 이상**이 필요합니다. 정확한 패키지 버전은 lockfile을 사용합니다.

```powershell
Set-Location site
npm ci
npm run build
node --test tests/*.test.mjs
npm run dev
```

`npm run dev`는 로컬 서버를 엽니다. 토큰·DB가 없으면 보호 화면과 미설정 상태만 확인할 수 있습니다. `npm run build`는 로컬 `dist/`, `.wrangler/` 산출물을 생성하며 배포 명령이 아닙니다. 웹 글꼴 처리 등 빌드 도구가 별도 네트워크 접근을 요구할 수 있습니다. `site/build/sites-vite-plugin.ts`, `site/.openai/hosting.json`, `site/drizzle/`는 빌드에 필요한 소스입니다. hosting 설정의 프로젝트 식별자는 로컬 예제이며 배포 대상이 아닙니다.

## 입력과 비밀 설정

- `.env.example`에는 필요한 환경변수 이름만 있습니다. 일반 Python 모듈은 `.env`를 자동 로드하지 않으므로 PowerShell 환경변수로 설정합니다. `roro_narrative.py`만 선택적으로 로컬 dotenv를 읽습니다.
- `REGIME_ROUTER_HOLDINGS_SOURCE`는 운영자가 제공하는 개인 workbook 경로입니다. `REGIME_ROUTER_PRIVATE_DIR` 기본값은 이 프로젝트의 `private/`이며 캐시는 `holdings.xlsx`입니다. 입력을 지정하지 않으면 존재하지 않는 `private/holdings-source.xlsx`를 가리킵니다. 실제 파일을 저장소에 넣지 마세요.
- 게시를 별도 설정할 때만 `site_upload_config.example.json`을 `site_upload_config.json`으로 복사하고 자신의 HTTPS URL·인증값을 채웁니다. 실제 값은 Git에서 제외합니다.
- 웹 로컬 비밀 이름은 `site/.dev.vars.example`을 참고하세요. 빈 비밀로 인증을 우회하지 않습니다. 뉴스 표시에는 별도 수집 파이프라인과 NEWS 설정이 필요하며 이 저장소에는 기존 기사가 없습니다.
- DB 스키마는 Python 모듈과 `site/db/schema.ts`, `site/drizzle/*.sql`에 있습니다. 시장 DB 초기화·계산에는 사용자가 선택한 유니버스 설정과 적법하게 확보한 시장 입력이 별도로 필요합니다. 입력 형식은 `data_templates/`, [데이터 준비 문서](PDF_DATA_SETUP.md), [RORO 데이터 카탈로그](RORO_DATA_CATALOG.md)를 참고하세요.
- 실제 승인 상태는 제외했습니다. 빈 registry 형식은 `MODEL_GOVERNANCE.example.json`이며 [모델 관리 규칙](MODEL_OPERATIONS.md)은 연구 절차를 설명합니다. 예제는 운영 승인이나 게시 동의를 부여하지 않습니다.

## 외부 연동과 수동 운영 도구

Yahoo/yfinance, FRED/ALFRED, Bloomberg 및 선택적인 OpenRouter 모델 호출을 사용하는 코드가 있습니다. 공급자의 이용 조건·접근 권한·호출 제한이 적용되며 Bloomberg, OpenRouter, Cloudflare 등의 사용에는 계약이나 비용이 발생할 수 있습니다. 이 스냅샷 준비에서는 데이터·모델 다운로드, 유료 호출, 실제 발송·게시·배포를 실행하지 않았습니다.

`refresh_pipeline.py`, `refresh_request_agent.py`, `UPDATE_SITE*.cmd`, `RUN_RORO_SHADOW_LOCAL.ps1`은 설정과 옵션에 따라 수집·계산·원격 게시를 수행합니다. `INSTALL_*_TASK.ps1`은 Windows 예약 작업을 등록하며 `EXPORT_CLOUDIUM_HOLDINGS.vbs`는 Excel을 통해 개인 입력을 복사합니다. 기본 설치·테스트 과정에서는 이 도구들을 실행하지 않습니다. `.github/workflows`는 포함하지 않아 저장소 생성만으로 CI나 배포를 시작하지 않습니다.

데이터 시점·누락·구성종목 편향과 연구 모델의 한계를 개별 `MODEL_SPEC*.md` 및 runbook에서 확인해야 합니다. 기존 문서의 운영 설명은 코드의 원래 사용 맥락이며 이 소스 스냅샷의 서비스 운영이나 모델 승인을 의미하지 않습니다. 별도 배포 라이선스 파일은 원본에 없었으며 새 라이선스를 임의로 부여하지 않았습니다.
