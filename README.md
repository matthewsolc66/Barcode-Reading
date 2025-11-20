# Barcode Sorter

Simple image sorter that scans images for Part Numbers (P####-#####) and Serial Numbers (S900...) and organizes images into `Sorted_Images/`.

## Prerequisites (Windows)

- Microsoft Visual C++ Redistributable (x64) — REQUIRED for `pyzbar` on Windows:
  https://www.microsoft.com/en-us/download/details.aspx?id=40784&msockid=2027dabdab8d696a3192ccf8aa2068d3

- Python 3.12+ (64-bit) — The included `setup_windows.bat` will install Python if missing.

- (Optional, for OCR) Tesseract-OCR — install to enable OCR fallback:
  https://github.com/UB-Mannheim/tesseract/wiki

## Quick setup (Windows)

1. Install the Visual C++ Redistributable (x64) from the link above (required by `pyzbar`).

2. In this project folder, run the setup script to install Python (if missing), create a virtual environment, and install dependencies:

```powershell
.\setup_windows.bat
```

This will create a `.venv` folder and install these Python packages:
- Pillow
- pyzbar
- opencv-python
- numpy
- pytesseract
- psutil

If `requirements.txt` is present, the script installs from it.

3. Run the sorter using the helper script (activates venv if present):

CMD:
```cmd
run_sorter.bat
```

PowerShell (activate manually or use `run_sorter.bat`):
```powershell
.\.venv\Scripts\Activate.ps1
python .\Barcode_Sorter_RC1.py
```

## Useful files

- `Barcode_Sorter_RC1.py` — main script (three-pass: barcode scan, OCR fallback, organize)
- `setup_windows.bat` — Windows setup script (installs Python, creates venv, installs pip deps)
- `requirements.txt` — Python dependency list
- `run_sorter.bat` — convenience runner that activates `.venv` and runs the sorter
- `ocr_region_tester.py` — interactive ROI OCR tester (useful during tuning)
- `part_numbers_config.txt` — optional whitelist of expected part numbers

## Troubleshooting
If none of these resolve the issue, please create an issue at https://github.com/matthewsolc66/Barcode-Reading and the developer will try to help as soon as possible.

 
- If you see an import error for `pyzbar` mentioning `MSVCP*.dll` or `VCRUNTIME*.dll`:
  - Install the Microsoft Visual C++ Redistributable (x64) (link above) and reboot.

- If OCR isn't working (no `pytesseract` import error or empty OCR results):
  - Install Tesseract-OCR (UB-Mannheim build recommended for Windows). Make sure its `tesseract.exe` is on `PATH`, or set `TESSDATA_PREFIX` appropriately.

- If package installation fails during `setup_windows.bat`:
  - Try running the script as an administrator, or open a PowerShell prompt with admin rights and rerun the `pip install` line manually.

 
## Workflow Diagram

```mermaid
flowchart TD
  ST([Start Script]) --> INIT[Initialize: imports, config, GUI, handlers]
  INIT --> LOAD_PARTS[Load expected part numbers - optional]
  LOAD_PARTS --> SEL_DIALOG[Show dialog: Select parts, workers, quick modes]
  SEL_DIALOG -->|Quick: Small Batch| SMALL_BATCH[Run small batch test] --> END
  SEL_DIALOG -->|Quick: OCR Tester| OCR_TESTER[Launch OCR region tester] --> END
  SEL_DIALOG -->|Normal| FOLDER[User selects input image folder]
  FOLDER --> FIND_FILES[Find all images in folder]
  FIND_FILES --> PASS1["Pass 1: Scan barcodes in images<br/>(parallel, with retries/rotations)<br/>"]
  PASS1 --> CHECK_PASS2{Images missing\n part/serial?}
  CHECK_PASS2 -->|Yes| PASS2["Pass 2: OCR fallback<br/>(parallel, CLAHE/Otsu/PSMs/union crops)"]
  CHECK_PASS2 -->|No| PASS3["Pass 3: Organize/copy images by Part/Serial"]
  PASS2 --> PASS3
  PASS3 --> REPORT["Generate concise report"]
  REPORT --> OPEN_FOLDER["Show output folder, completion dialog"]
  OPEN_FOLDER --> END([End])
```

