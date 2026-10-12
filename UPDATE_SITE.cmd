@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

call "UPDATE_SITE_SCHEDULED.cmd"
if errorlevel 1 (
  echo.
  echo [실패] 갱신 로그를 확인하세요: %LOCALAPPDATA%\RegimeRouter\logs\site_update.log
  pause
  exit /b 1
)

echo.
echo [종료] HTTPS 대시보드 갱신 처리가 끝났습니다. 위의 경고 및 부분 완료 내용을 확인하세요.
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoLogo -NoProfile -ExecutionPolicy Bypass -File ".\OPEN_SITE.ps1"
pause
