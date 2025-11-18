@echo off
REM Helper: run Barcode_Sorter_RC1.py using the project's .venv if present.
REM This script now prints diagnostics and pauses so errors remain visible when double-clicked.
cd /d %~dp0

echo Working directory: %CD%

set rc=0
if exist .venv (
  echo Activating virtual environment .venv
  call ".venv\Scripts\activate.bat"
  echo "Python executable:"
  where python || echo "(where python failed)"
  python --version 2>nul || echo "(python --version failed)"
  echo Running Barcode_Sorter_RC1.py
  python Barcode_Sorter_RC1.py %*
  set rc=%ERRORLEVEL%
) else (
  echo No .venv found - attempting to use system Python
  echo Checking for python on PATH...
  where python 2>nul
  if %ERRORLEVEL% neq 0 (
    echo Python not found on PATH. Please install Python 3.12+ or run setup_windows.bat.
    echo Press any key to close this window.
    pause
    exit /b 1
  )
  python --version
  echo Running Barcode_Sorter_RC1.py
  python Barcode_Sorter_RC1.py %*
  set rc=%ERRORLEVEL%
)

echo.
echo Script exited with code %rc%
if %rc% neq 0 (
  echo There was an error running the Python script. See messages above for details.
) else (
  echo Script completed successfully.
)

echo.
echo Press any key to close this window.
pause
