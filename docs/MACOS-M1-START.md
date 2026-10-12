# M1 Mac에서 시작하기

대상은 Apple Silicon **M1 / arm64**입니다. macOS 버전은 아직 확인되지 않았습니다. 아래 안내와 각 저장소의 보완은 `codex/macos-m1-20261012` 작업 브랜치에 있으며, draft PR 검토 중에는 `main`에 포함되지 않습니다. Windows에서 소스·오프라인 테스트·셸 구문을 점검했으며 실제 M1 설치/실행을 인증한 문서가 아닙니다.

## 1. Mac 환경 확인

기본 터미널에서 다음을 확인합니다.

```bash
uname -m
sw_vers -productVersion
```

`uname -m` 결과는 `arm64`여야 합니다. `x86_64`라면 터미널의 Rosetta 실행 여부부터 확인하세요. Intel용 Python/Node와 arm64 패키지를 같은 가상환경에 섞지 않습니다.

[Homebrew 공식 설치 요건](https://docs.brew.sh/Installation)을 먼저 확인하세요. 문서 확인 시점에는 macOS 15 이상을 지원 대상으로 안내합니다. Mac 버전이 이보다 낮으면 아래 Homebrew 경로가 지원된다고 가정하지 말고, 해당 OS와 호환되는 공식 설치 프로그램부터 확인하세요. 여기서는 OS 업그레이드나 Rosetta 설치를 자동 수행하지 않습니다.

## 2. 공통 도구 설치

지원되는 macOS에서는 [Homebrew 공식 홈페이지](https://brew.sh/)의 설치 안내를 따라 설치하고, 설치기가 출력한 shellenv 설정을 적용합니다. Apple Silicon 기본 경로는 `/opt/homebrew`입니다. 아래는 그 기본 경로를 쓰는 경우의 **현재 터미널 세션** 설정입니다.

```bash
eval "$(/opt/homebrew/bin/brew shellenv)"
brew --prefix
brew install git gh python@3.12
```

`brew --prefix`가 `/opt/homebrew`인지 확인합니다. 소스 빌드가 필요하다는 메시지가 나오면 [Apple Command Line Tools 안내](https://developer.apple.com/xcode/resources/)를 확인하고 `xcode-select --install`로 설치합니다. 시스템 Python을 덮어쓰거나 `sudo pip`를 사용하지 않습니다. 다른 설치 경로가 필요하면 [Python의 macOS 안내](https://docs.python.org/3.12/using/mac.html)에서 지원 버전과 universal2 설치 프로그램을 확인하세요.

QuantBot의 문서/검증 기준 Python 3.11이 필요하면 별도로 설치합니다.

```bash
brew install python@3.11
```

Market-Lens 또는 News-Alarm의 별도 웹 프로젝트를 사용할 때만 Node 24를 설치합니다. 두 웹 프로젝트는 Node 22.13 이상을 요구합니다.

```bash
brew install node@24
export PATH="$(brew --prefix node@24)/bin:$PATH"
node -p "process.version + ' ' + process.arch"
npm --version
```

Node 아키텍처는 `arm64`여야 합니다. 위 PATH 설정 역시 현재 세션에만 적용됩니다. [Python 3.12](https://formulae.brew.sh/formula/python@3.12), [Node 24](https://formulae.brew.sh/formula/node@24)의 현재 OS/arm64 바이너리 지원은 설치 시 공식 페이지에서 확인하세요. 모든 Python 하위 의존성의 M1 wheel 설치를 이번 Windows 점검에서 검증한 것은 아닙니다.

## 3. 개인 Mac에서 직접 인증하고 내려받기

비공개 저장소 접근 권한이 있는 본인 계정으로 [GitHub CLI 로그인](https://cli.github.com/manual/gh_auth_login)을 합니다. 회사 PC의 인증 파일, API 키, `.env`, DPAPI 파일, 가상환경, DB, 포트폴리오 파일을 옮기지 않습니다.

```bash
gh auth login --hostname github.com --git-protocol https --web
mkdir -p "$HOME/Projects"
cd "$HOME/Projects"
```

설치 안내와 호환성 변경이 담긴 **검토 브랜치**를 확인하려면 필요한 저장소만 복제합니다. 아래 명령은 기존 같은 이름 폴더가 없는 새 위치용입니다.

```bash
gh repo clone yhan2002/QuantBotSystem -- --branch codex/macos-m1-20261012
gh repo clone yhan2002/LS-Portfolio -- --branch codex/macos-m1-20261012
gh repo clone yhan2002/Market-Lens -- --branch codex/macos-m1-20261012
gh repo clone yhan2002/News-Alarm -- --branch codex/macos-m1-20261012
gh repo clone yhan2002/Reversion-Signal -- --branch codex/macos-m1-20261012
```

기존 체크아웃이 있으면 먼저 `git status --short`와 현재 브랜치를 확인합니다. dirty 파일을 덮어쓰는 reset/checkout 명령은 사용하지 않습니다. `main`만 복제하려면 `-- --branch ...` 부분을 생략할 수 있지만, PR이 병합되기 전에는 이 안내와 보완이 빠집니다.

## 4. 저장소별로 따로 설치

각 저장소에 진입한 뒤 해당 안내를 따릅니다. Python 가상환경은 저장소마다 새로 만들고 재사용·복사하지 않습니다. 예를 들어 Python 3.12 프로젝트에서는:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -c "import platform, sys; print(sys.version); print(platform.machine()); print(sys.executable)"
# 해당 프로젝트 안내에 있는 requirements 파일만 설치합니다.
# 다음 프로젝트로 이동하기 전에:
deactivate
```

실행 파일이 현재 저장소의 `.venv/bin/python`이고 아키텍처가 `arm64`인지 확인합니다. 이미 `.venv`가 있다면 먼저 확인하고 새 환경을 별도 이름으로 만들며, 기존 환경을 삭제하지 않습니다.

| 저장소 | 첫 확인 경로 | macOS에서 남는 제한 |
| --- | --- | --- |
| [QuantBotSystem](https://github.com/yhan2002/QuantBotSystem/blob/codex/macos-m1-20261012/docs/MACOS.md) | Python 3.11, 설정 점검·오프라인 테스트 | selector 실행 관리의 macOS 프로세스 검증 미지원; 실제 봇은 별도 개인 설정·외부 freshness 모듈 필요 |
| [LS-Portfolio](https://github.com/yhan2002/LS-Portfolio/blob/codex/macos-m1-20261012/docs/MACOS.md) | Python 3.12, 합성 입력 self-test·HTML preview | 원본 Tracker의 Excel COM 저장은 Windows 전용; Mac Excel이 대체 경로가 되지 않음 |
| [Market-Lens](MACOS.md) | Python 3.12+, 오프라인 연구 테스트; 선택적 Node 웹 | Windows Excel exporter·운영 refresh 경로 제약; 기존 Python 회귀 실패 2건 별도 명시 |
| [News-Alarm](https://github.com/yhan2002/News-Alarm/blob/codex/macos-m1-20261012/docs/MACOS.md) | Python 3.12, 로컬 데모 화면·dry-run | DPAPI 키 저장과 Windows 잠금 작업기 미지원; 선택적 모델은 OS/torch 호환성 추가 확인 |
| [Reversion-Signal](https://github.com/yhan2002/Reversion-Signal/blob/codex/macos-m1-20261012/docs/MACOS.md) | Git·편집기로 문서 확인 | 연구 앱/전략/테스트 런타임이 없는 scaffold; 기존 게시 도우미는 Windows PowerShell용 |

[공개 momentum-factor-screener](https://github.com/yhan2002/momentum-factor-screener)는 별도 노트북 프로젝트입니다. 이 Mac 작업에서는 의존성·노트북을 수정하지 않았습니다. OpenSmilee는 이번 안내 대상에서 제외했습니다.

## 5. 첫 실행의 범위

먼저 각 프로젝트의 오프라인 테스트, 샘플 입력, 로컬 화면만 확인합니다. 패키지 설치는 인터넷 다운로드를 수반하지만 테스트/미리보기 명령은 안내에서 구분했습니다. API 호출·RSS 수집·모델 다운로드·알림·배포·스케줄러는 각각 별도 동작이며 설치 성공을 확인하기 위해 실행할 필요가 없습니다.

실제 M1에서 남은 확인은 **macOS 버전 → arm64 Python/Node → 새 venv 의존성 설치 → 저장소별 오프라인 테스트/로컬 화면** 순서입니다. 이 결과 전에는 Mac 실운영 호환성을 확정하지 않습니다. 기존 회사 봇이나 원본 작업 폴더에는 이번 변경을 배포하지 않았습니다.
