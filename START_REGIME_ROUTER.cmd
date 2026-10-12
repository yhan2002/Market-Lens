@echo off
chcp 65001 >nul
set "APP_DIR=%~dp0"
set "APP_PYTHON=%APP_DIR%.venv\Scripts\pythonw.exe"

if not exist "%APP_PYTHON%" (
    powershell -NoProfile -Command "Add-Type -AssemblyName PresentationFramework; [System.Windows.MessageBox]::Show('실행 환경(.venv)이 없습니다. 관리자에게 초기 설치를 요청하세요.','Regime Router')"
    exit /b 1
)

start "" "%APP_PYTHON%" "%APP_DIR%regime_router_app.py"
