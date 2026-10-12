@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONUTF8=1"
title Market Lens Same Logic Performance Report

echo 현재 운용 화면과 동일한 규칙으로 과거 성과를 계산합니다.
echo 로컬 게시 DB만 읽으며 외부 API를 호출하거나 사이트를 변경하지 않습니다.
echo 수천 거래일을 순서대로 재생하므로 수 분 이상 걸릴 수 있습니다.
echo.

if not exist ".venv\Scripts\python.exe" (
  echo [오류] .venv Python을 찾을 수 없습니다.
  pause
  exit /b 1
)

if not exist "regime_router_publish.db" (
  echo [오류] regime_router_publish.db를 찾을 수 없습니다.
  pause
  exit /b 1
)

echo 실행 환경과 데이터 파일을 확인합니다.
".venv\Scripts\python.exe" -u "policy_oos.py" "regime_router_publish.db" --check-only
if errorlevel 1 (
  echo.
  echo [실패] 사전점검 오류를 확인해 주세요.
  pause
  exit /b 1
)
if /i "%~1"=="--check" exit /b 0

echo.
".venv\Scripts\python.exe" -u "policy_oos.py" "regime_router_publish.db" --output-dir "outputs\policy_v5_1_weighted"
if errorlevel 1 (
  echo.
  echo [실패] 위 오류를 확인해 주세요.
  pause
  exit /b 1
)

echo.
echo [완료] 투자설명회용 결과 화면을 엽니다.
set "REPORT=%CD%\outputs\policy_v5_1_weighted\policy_oos_presentation.html"
set "BROWSER=%ProgramFiles%\Google\Chrome\Application\chrome.exe"
if not exist "%BROWSER%" set "BROWSER=%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"
if not exist "%BROWSER%" set "BROWSER=%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"
if exist "%BROWSER%" (
  start "" "%BROWSER%" "%REPORT%"
) else (
  echo [안내] Chrome 또는 Edge를 찾지 못했습니다. 아래 파일을 브라우저에서 열어 주세요.
  echo %REPORT%
)
pause
