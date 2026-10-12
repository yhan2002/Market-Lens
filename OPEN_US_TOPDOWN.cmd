@echo off
cd /d "%~dp0"
"%~dp0.venv\Scripts\python.exe" "%~dp0us_topdown_screener.py" --open
if errorlevel 1 pause
