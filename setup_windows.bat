@echo off
setlocal ENABLEDELAYEDEXPANSION

REM Barcode Sorter Windows Setup (no VC++ runtime install)
REM - Installs Python if missing (winget or python.org)
REM - Creates .venv and installs pip dependencies
REM - Skips Microsoft Visual C++ Redistributable by design

cd /d %~dp0

set VENV_DIR=.venv
set REQ_FILE=requirements.txt
set PY_MIN_VER=3.12
set PY_INSTALLER_VER=3.12.7
set PY_INSTALLER_URL=https://www.python.org/ftp/python/%PY_INSTALLER_VER%/python-%PY_INSTALLER_VER%-amd64.exe
set PY_INSTALLER=%TEMP%\python-%PY_INSTALLER_VER%-amd64.exe

:print_header
echo ============================================================
echo BARCODE SORTER - WINDOWS SETUP
echo ============================================================

REM 1) Notify about VC++ runtime prerequisite
echo.
echo NOTE: This setup does NOT install the Microsoft VC++ Runtime.
echo If pyzbar fails to import later, install the Visual C++ Redistributable (x64):
echo   https://www.microsoft.com/en-us/download/details.aspx?id=40784&msockid=2027dabdab8d696a3192ccf8aa2068d3

echo.
echo Checking for Python...

REM 2) Detect Python (prefer py launcher)
set PY_CMD=
where py >nul 2>&1 && set PY_CMD=py
if not defined PY_CMD (
  where python >nul 2>&1 && set PY_CMD=python
)

if not defined PY_CMD (
  echo Python not found. Attempting to install...
  REM 2a) Try winget first
  where winget >nul 2>&1
  if %ERRORLEVEL%==0 (
    echo Using winget to install Python %PY_MIN_VER%...
    winget install -e --id Python.Python.%PY_MIN_VER% --accept-package-agreements --accept-source-agreements
    if %ERRORLEVEL%==0 (
      where py >nul 2>&1 && set PY_CMD=py
      if not defined PY_CMD where python >nul 2>&1 && set PY_CMD=python
    ) else (
      echo winget install failed or was cancelled. Falling back to python.org installer...
    )
  ) else (
    echo winget not available. Falling back to python.org installer...
  )

  if not defined PY_CMD (
    echo Downloading Python installer %PY_INSTALLER_VER% ...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Invoke-WebRequest -Uri '%PY_INSTALLER_URL%' -OutFile '%PY_INSTALLER%'" || (
      echo Failed to download Python installer.
      echo Please install Python manually from https://www.python.org/downloads/windows/
      exit /b 1
    )
    echo Running Python installer (per-user, silent)...
    "%PY_INSTALLER%" /quiet InstallAllUsers=0 PrependPath=1 Include_launcher=1 Include_pip=1 SimpleInstall=1 || (
      echo Python installer failed or was cancelled.
      exit /b 1
    )
    del "%PY_INSTALLER%" >nul 2>&1
    where py >nul 2>&1 && set PY_CMD=py
    if not defined PY_CMD where python >nul 2>&1 && set PY_CMD=python
  )
)

if not defined PY_CMD (
  echo Could not locate Python after installation attempts.
  echo Please install Python 3.12+ and re-run this script.
  exit /b 1
)

echo Found Python launcher: %PY_CMD%

REM 3) Create virtual environment if missing
if not exist "%VENV_DIR%" (
  echo Creating virtual environment at %VENV_DIR% ...
  %PY_CMD% -m venv "%VENV_DIR%" || (
    echo Failed to create virtual environment.
    exit /b 1
  )
) else (
  echo Virtual environment already exists: %VENV_DIR%
)

REM 4) Use venv Python for all installs
set VENV_PY="%VENV_DIR%\Scripts\python.exe"
set VENV_PIP="%VENV_DIR%\Scripts\pip.exe"

REM Ensure venv Python exists
if not exist %VENV_PY% (
  echo venv Python not found at %VENV_PY%.
  exit /b 1
)

REM 5) Upgrade pip and tooling
%VENV_PY% -m pip install --upgrade pip setuptools wheel
if %ERRORLEVEL% NEQ 0 (
  echo Pip upgrade failed. Continuing...
)

REM 6) Install dependencies
if exist "%REQ_FILE%" (
  echo Installing packages from %REQ_FILE% ...
  %VENV_PIP% install -r "%REQ_FILE%" || (
    echo Package installation failed.
    exit /b 1
  )
) else (
  echo requirements.txt not found. Installing required packages individually...
  %VENV_PIP% install Pillow pyzbar opencv-python numpy pytesseract psutil || (
    echo Package installation failed.
    exit /b 1
  )
)

echo.
echo ============================================================
echo SETUP COMPLETE

echo To run the sorter in this environment:
echo   PowerShell:
echo     .\%VENV_DIR%\Scripts\Activate.ps1
echo     python .\Barcode_Sorter_RC1.py

echo   CMD:
echo     .\%VENV_DIR%\Scripts\activate.bat && python .\Barcode_Sorter_RC1.py

echo If you see DLL errors about MSVCP/VCRUNTIME when importing pyzbar,
echo install the Visual C++ Redistributable (x64):
echo   https://www.microsoft.com/en-us/download/details.aspx?id=40784&msockid=2027dabdab8d696a3192ccf8aa2068d3

echo Optionally install Tesseract-OCR to enable OCR fallback:
echo   https://github.com/UB-Mannheim/tesseract/wiki

echo ============================================================
exit /b 0
