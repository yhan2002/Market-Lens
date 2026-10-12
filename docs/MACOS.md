# Market-Lens: M1 Mac 설치와 확인

대상은 **Apple Silicon M1 / arm64**입니다. [공통 M1 시작 가이드](MACOS-M1-START.md)에서 GitHub 로그인과 기본 도구를 준비한 뒤 진행하세요. 아래 명령은 Terminal의 zsh/bash에서 실행합니다. Rosetta나 Intel Python을 기본으로 사용하지 않습니다.

이 안내는 Windows에서 소스와 테스트를 확인해 작성했습니다. **실제 M1/macOS 실행, 새 Mac의 Python 의존성 설치, 화면·글꼴 품질은 아직 검증하지 않았습니다.** macOS 버전도 확인되지 않았으므로 아래 웹 도구의 OS 조건을 먼저 확인하세요.

## 사용할 수 있는 범위

| 기능 | 이번 안내의 범위 |
| --- | --- |
| Python 연구 코드 | Python 3.12 환경, import/CLI 도움말, 합성 입력을 쓰는 오프라인 테스트 |
| `site/` 웹 | arm64 Node로 설치·로컬 빌드·테스트·수동 개발 서버; 운영 DB 없이 보호 화면 확인 |
| Tk 로컬 UI | Tk 설치 확인과 수동 실행 안내; 실제 데이터와 화면은 Mac에서 별도 확인 |
| 통합 갱신·Excel·예약 작업 | Windows 전용 부분이 남아 있음. Mac에서 전체 운영 파이프라인 실행 대상으로 보지 않음 |
| Bloomberg·PDF/Word 연구 보고서 | 별도 SDK·입력·선택 의존성·글꼴이 필요하며 M1 호환성 미검증 |

회사 PC의 `.venv`, `node_modules`, DB, 보유내역, 기사, `.env`, 게시 토큰을 복사하지 않습니다. 이 저장소에는 운영 입력·승인 상태가 포함되지 않습니다.

## Python 설치와 오프라인 확인

저장소 루트(`Market-Lens`)에서 실행합니다. 프로젝트별 `.venv`를 새로 만들고 다른 저장소의 가상환경과 공유하지 않습니다. 기존 `.venv`가 있다면 먼저 그 Python의 버전·아키텍처를 확인하고, 잘못된 환경을 덮어쓰지 마세요.

```bash
brew install python@3.12 python-tk@3.12
python3.12 -c 'import platform, sys; print(sys.version); print(platform.machine()); assert platform.machine() == "arm64"'
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --only-binary=:all: -r requirements.txt
python -m pip check
python -c 'import numpy, pandas, sklearn, hmmlearn, yfinance, openpyxl, tkinter; print("imports OK")'
python regime_router.py --help
python -B tools/run_offline_tests.py
```

Python 3.12 이상이 필요한 소스 문법이 있어 3.12를 기준으로 잡았습니다. 기본 의존성은 [requirements.txt](../requirements.txt)의 NumPy, pandas, scikit-learn, hmmlearn, yfinance, openpyxl입니다. 하한 버전만 지정되어 있어 설치 시점에 선택되는 조합은 달라질 수 있습니다. `--only-binary=:all:`은 호환 wheel이 없을 때 임의의 소스 빌드 대신 설치를 멈추게 합니다. 실패하면 해당 패키지명, `sw_vers`, Python 버전과 `platform.machine()` 결과를 확인하세요. 이 안내를 이유로 의존성을 일괄 업그레이드하거나 Rosetta로 바꾸지는 않습니다.

Tk는 GUI뿐 아니라 `site_publisher.py`가 가져오는 `regime_router_app.py`에도 필요합니다. Homebrew Python의 Tk 패키지는 별도 설치입니다. [Python 3.12](https://formulae.brew.sh/formula/python@3.12), [python-tk@3.12](https://formulae.brew.sh/formula/python-tk@3.12) 공식 설치 정보를 참고하세요. Homebrew가 현재 macOS를 지원하는지도 공통 가이드에서 확인해야 합니다.

오프라인 runner는 네트워크, 외부 프로세스 실행과 소켓 생성을 차단하며 합성 입력·임시 DB를 사용합니다. Windows PowerShell을 실제 실행하는 `test_run_roro_shadow_local.py`는 제외합니다. 테스트 중 표시되는 모델/API 실패 메시지는 mock 또는 차단된 테스트 경로일 수 있으므로 마지막 테스트 요약을 확인하세요.

기존 전체 suite에는 두 실패가 남아 있어 현재 종료 코드는 1입니다. 두 기대값은 이번 Mac 안내에서 바꾸지 않았습니다.

- `test_risk_signals.FastSignalSnapshotTests.test_session_aligned_yahoo_jpy_overrides_fred_fallback`: JPY 상태 기대값 불일치
- `test_site_publisher.SnapshotRiskPolicyTests.test_adds_optional_policy_without_hiding_stale_raw_channel`: 정책 버전 suffix 기대값 불일치

별도 `News Alarm` helper가 없으면 관련 테스트 클래스 한 개가 skip됩니다. 이 이름의 이웃 폴더를 만들거나 운영 데이터를 복사해 skip을 해소할 필요는 없습니다. Windows timeout 테스트는 실제 호스트의 경로 클래스를 유지해 macOS에서 Windows 분기를 모의 실행하도록 수정했습니다.

## 로컬 웹 설치·실행

`site/package.json`은 Node.js **22.13 이상**을 요구합니다. 공통 가이드에서 arm64 Node를 설치한 뒤 아래를 확인하세요. 웹 런타임에 포함된 Wrangler/workerd의 공식 지원 조건은 **macOS 13.5 이상**입니다. 이 조건만으로 모든 의존성의 M1 실행이 검증된 것은 아닙니다. [Cloudflare 시스템 요구사항](https://developers.cloudflare.com/workers/wrangler/install-and-update/)

```bash
sw_vers -productVersion
node -p 'process.version + " " + process.platform + " " + process.arch'
cd site
npm ci
npm run build
node --test tests/*.test.mjs
npm run dev -- --hostname 127.0.0.1
```

Node 출력이 `darwin arm64`인지 확인합니다. `package-lock.json`에는 workerd, esbuild, sharp, rolldown, Tailwind의 Darwin arm64 패키지가 포함되어 있습니다. Mac에서 `npm ci`를 새로 실행해 해당 플랫폼 패키지를 설치하고 Windows의 `node_modules`를 가져오지 마세요. 프로젝트 lockfile을 그대로 사용하며 `npm update`나 전역 Wrangler 설치는 필요하지 않습니다.

빌드 후 전체 `tests/*.test.mjs`를 실행합니다. `npm test`는 렌더링 테스트 일부만 실행하므로 같은 범위가 아닙니다. `npm ci`는 패키지 다운로드가 필요하고, 빌드도 도구·글꼴 처리에 따라 네트워크 접근이 필요할 수 있습니다. Python runner처럼 웹 명령 전체가 네트워크를 강제로 차단하는 것은 아닙니다.

개발 서버가 출력한 `127.0.0.1` 주소를 브라우저에서 열고 종료할 때 `Ctrl+C`를 누릅니다. 토큰·DB 없이 확인할 수 있는 것은 보호 화면과 미설정 상태입니다. 빌드는 로컬 `dist/`, `.wrangler/` 파일을 만들지만 원격 배포를 실행하지 않습니다. `.openai/hosting.json`은 로컬 예제이며 운영 배포 대상이 아닙니다. 기존 운영 토큰이나 DB를 넣어 화면을 채우지 마세요.

## Tk UI를 수동으로 확인할 때

웹 작업을 마치면 저장소 루트로 돌아가 `.venv`를 활성화한 상태에서 실행합니다.

```bash
python -m tkinter
python regime_router_app.py
```

첫 명령은 Tk 자체의 테스트 창입니다. 두 번째는 프로젝트 UI이며, 입력 DB가 없으면 결과 대신 데이터 부족 오류가 나올 수 있습니다. 창의 **전체 최신화** 기능은 네트워크 수집·DB 쓰기를 수행하므로 설치 확인 중 누르지 않습니다. UI에는 이웃 `QuantBot/QuantBotSystem/config.py`를 기대하는 기존 경로도 남아 있습니다. 개인 Mac의 clone 이름이 `QuantBotSystem`이라면 자동으로 연결되지 않으며, 회사 설정을 복사해 맞추는 절차는 제공하지 않습니다.

## Windows 전용·선택 기능의 제한

- `refresh_pipeline.py`는 `.venv/Scripts/python.exe`, `msvcrt` 파일 잠금, Windows Script Host와 Excel VBS 내보내기에 의존합니다. `refresh_request_agent.py`도 이 파이프라인을 사용합니다. Mac용 통합 갱신·게시 경로는 이번 변경에 포함하지 않습니다.
- `START_REGIME_ROUTER.cmd`, `UPDATE_SITE*.cmd`, `RUN_RORO_SHADOW_LOCAL.ps1`, `INSTALL_*_TASK.ps1`은 Windows 실행·예약 도구입니다. `cron`/`launchd` 등록이나 자동 변환은 하지 않습니다.
- `EXPORT_CLOUDIUM_HOLDINGS.vbs`는 Windows Excel COM을 사용합니다. Mac Excel로 치환되지 않습니다. `trading_log.read_holdings_workbook()`의 openpyxl 읽기는 계산된 값 읽기이며 Excel 재계산·서식·매크로 보존 저장을 보장하지 않습니다.
- `reports/build_sector_turning_alert_backtest_pdf.py`는 Windows의 Malgun 글꼴 파일 경로를 직접 사용합니다. 이 보고서는 기본 Mac 실행 대상이 아닙니다. 다른 보고서도 기존 DB·중간 산출물과 `reportlab`, `pypdf`, `Pillow`, `python-docx` 등 선택 의존성을 확인해야 합니다.
- Bloomberg 관련 도구는 `blpapi` SDK, 서비스 접근 권한과 입력이 별도로 필요합니다. M1/arm64 및 사용자의 계약·연결 환경은 검증하지 않았습니다. 기본 설치에 SDK를 추가하지 않습니다.
- `.env.example`를 `.env`로 복사하는 것만으로 일반 모듈에 설정이 적용되지는 않습니다. 대부분은 프로세스 환경변수를 읽고 `roro_narrative.py`만 선택적으로 dotenv를 읽습니다. 기본 오프라인 검사에는 실제 API 키가 필요 없습니다.
- `site_publisher.py --no-upload`는 업로드만 생략합니다. 다른 옵션에 따라 수집·계산·DB 변경을 할 수 있으므로 오프라인 진단 명령으로 사용하지 않습니다.

## 검증 기록과 Mac에서 남은 확인

이번 안내 준비에서 Windows의 기존 Python 의존성 환경과 Node를 사용했습니다. 새 Python 의존성 설치와 실제 macOS/arm64 실행을 통과했다고 주장하지 않습니다. Windows 기준 Python 전체 suite는 **548 통과 / 1 skip / 위 기존 실패 2개**, 웹 전체 테스트는 **37 통과**였습니다. 실제 데이터 수집, 모델 API, 알림, 게시, 운영 DB 이관, 배포·재시작은 실행하지 않았습니다.

M1에서는 macOS 버전, Python·Node의 arm64 출력, 새 `.venv` 설치·import, 동일 테스트 결과, 로컬 웹 화면을 순서대로 확인하면 됩니다. 테스트가 다른 원인으로 실패하면 오류와 버전만 공유하고 비밀 설정이나 운영 입력을 첨부하지 마세요.
