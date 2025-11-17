@echo off
REM Simple helper to run Barcode_Sorter_RC1.py using the project's venv if present
cd /d %~dp0

if exist .venv ( 
  echo Activating virtual environment .venv
  call .venv\Scripts\activate.bat
  echo Running Barcode_Sorter_RC1.py
  python Barcode_Sorter_RC1.py %*
) else (
  echo No .venv found - running system Python
  python Barcode_Sorter_RC1.py %*
)
