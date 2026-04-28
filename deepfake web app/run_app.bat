@echo off
setlocal
cd /d "%~dp0"
echo Starting Verifixia API on port 3001...
py -3.12 app.py
endlocal
