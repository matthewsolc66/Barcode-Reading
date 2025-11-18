@echo off
REM Barcode Sorter Windows Setup
REM Creates .venv and installs pip dependencies

cd /d %~dp0

echo ============================================================
echo BARCODE SORTER - WINDOWS SETUP
echo ============================================================
echo.
echo NOTE: This setup does NOT install the Microsoft VC++ Runtime.
echo If pyzbar fails to import later, install it from:
echo   https://www.microsoft.com/en-us/download/details.aspx?id=40784
echo.
echo Checking for Python...

REM Find Python
set PYTHON_EXE=
where py >nul 2>&1
if %ERRORLEVEL%==0 (
    set PYTHON_EXE=py
    echo Found Python launcher: py
    goto :create_venv
)

where python >nul 2>&1
if %ERRORLEVEL%==0 (
    set PYTHON_EXE=python
    echo Found Python: python
    goto :create_venv
)

echo.
echo ERROR: Python not found on PATH.
echo Please install Python 3.12+ from https://www.python.org/downloads/windows/
echo Make sure to check "Add Python to PATH" during installation.
echo.
echo Press any key to close...
pause >nul
exit /b 1

:create_venv
echo.
if exist .venv (
    echo Virtual environment .venv already exists.
) else (
    echo Creating virtual environment .venv ...
    %PYTHON_EXE% -m venv .venv
    if %ERRORLEVEL% NEQ 0 (
        echo.
        echo ERROR: Failed to create virtual environment.
        echo Make sure Python venv module is installed.
        echo.
        echo Press any key to close...
        pause >nul
        exit /b 1
    )
    echo Virtual environment created successfully.
)

REM Use venv Python
set VENV_PYTHON=.venv\Scripts\python.exe
set VENV_PIP=.venv\Scripts\pip.exe

if not exist "%VENV_PYTHON%" (
    echo.
    echo ERROR: Virtual environment Python not found at %VENV_PYTHON%
    echo.
    echo Press any key to close...
    pause >nul
    exit /b 1
)

echo.
echo Upgrading pip...
"%VENV_PYTHON%" -m pip install --upgrade pip --quiet
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: pip upgrade failed, continuing anyway...
)

echo.
if exist requirements.txt (
    echo Installing packages from requirements.txt ...
    "%VENV_PIP%" install -r requirements.txt
    if %ERRORLEVEL% NEQ 0 (
        echo.
        echo ERROR: Package installation failed.
        echo Check your internet connection and try again.
        echo.
        echo Press any key to close...
        pause >nul
        exit /b 1
    )
) else (
    echo requirements.txt not found - installing core packages...
    "%VENV_PIP%" install Pillow pyzbar opencv-python numpy pytesseract psutil
    if %ERRORLEVEL% NEQ 0 (
        echo.
        echo ERROR: Package installation failed.
        echo Check your internet connection and try again.
        echo.
        echo Press any key to close...
        pause >nul
        exit /b 1
    )
)

echo.
echo ============================================================
echo SETUP COMPLETE!
echo ============================================================
echo.
echo Virtual environment created at: .venv
echo.
echo To run the sorter:
echo   1. Double-click run_sorter.bat
echo   OR
echo   2. Run manually:
echo      .venv\Scripts\activate.bat
echo      python Barcode_Sorter_RC1.py
echo.
echo IMPORTANT REMINDERS:
echo   - Install VC++ Redistributable if you see pyzbar DLL errors:
echo     https://www.microsoft.com/en-us/download/details.aspx?id=40784
echo.
echo   - Optionally install Tesseract-OCR for better OCR fallback:
echo     https://github.com/UB-Mannheim/tesseract/wiki
echo.
echo ============================================================
echo.
echo Press any key to close...
pause >nul
exit /b 0
