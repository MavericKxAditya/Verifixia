@echo off
setlocal
cd /d "%~dp0"

py -3.12 tester.py
if errorlevel 1 (
  echo.
  echo If this fails, make sure Python 3.12 has torch, torchvision, opencv-python, and pillow installed.
)

endlocal
