@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONUTF8=1"

if not exist ".venv\Scripts\python.exe" exit /b 1
if "%MARKET_LENS_AS_OF%"=="" (
  ".venv\Scripts\python.exe" "refresh_pipeline.py"
) else (
  ".venv\Scripts\python.exe" "refresh_pipeline.py" --as-of "%MARKET_LENS_AS_OF%"
)
exit /b %ERRORLEVEL%
