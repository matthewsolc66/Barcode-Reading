<#
PowerShell helper to run Barcode_Sorter_RC1.py using the project's .venv if present.
Usage:
  - Double-click this file (may require running PowerShell with ExecutionPolicy Bypass),
  - Or run from an elevated/non-elevated PowerShell prompt:
      .\run_sorter.ps1
  - To pass arguments to the Python script, append them after the script name:
      .\run_sorter.ps1 -- --my-arg value
    (Use `--` to separate PS parameters from script args if needed.)

Behavior:
 - If `.venv\Scripts\python.exe` exists, this script will use that Python executable.
 - Otherwise it will try to use the system `python` on PATH.
 - The script forwards any provided arguments to `Barcode_Sorter_RC1.py`.
 - After Python exits, the script waits for ENTER so double-click runs don't close immediately.
#>

param(
    [switch]$NoPause
)

# Ensure script runs from its containing directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $scriptDir

Write-Host "[INFO] Starting Barcode Sorter runner in: $scriptDir" -ForegroundColor Cyan

$venvPy = Join-Path $scriptDir ".venv\Scripts\python.exe"
$scriptToRun = Join-Path $scriptDir "Barcode_Sorter_RC1.py"

if (Test-Path $venvPy) {
    Write-Host "[INFO] Found virtual environment Python: $venvPy" -ForegroundColor Green
    $pythonExe = $venvPy
} else {
    Write-Host "[WARN] No .venv found. Falling back to system Python." -ForegroundColor Yellow
    $py = Get-Command python -ErrorAction SilentlyContinue
    if (-not $py) {
        Write-Host "[ERROR] No Python found on PATH. Run setup_windows.bat to install Python and dependencies." -ForegroundColor Red
        if (-not $NoPause) { Write-Host "Press ENTER to close..."; [void][System.Console]::ReadLine() }
        exit 1
    }
    $pythonExe = $py.Source
    Write-Host "[INFO] Using system Python: $pythonExe" -ForegroundColor Green
}

# Build argument list: forward all remaining args to the Python script
$forwardArgs = @()
if ($args.Count -gt 0) {
    $forwardArgs = $args
}

# Run the Python script
Write-Host "[INFO] Running: $pythonExe $scriptToRun $($forwardArgs -join ' ')" -ForegroundColor Cyan
& $pythonExe $scriptToRun @forwardArgs
$exitCode = $LASTEXITCODE

Write-Host "[INFO] Python exited with code: $exitCode" -ForegroundColor Cyan

if (-not $NoPause) {
    Write-Host "Press ENTER to close..." -NoNewline
    [void][System.Console]::ReadLine()
}

exit $exitCode
