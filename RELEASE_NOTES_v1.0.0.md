# Barcode Sorter v1.0.0 Release Notes

Release Date: 2025-11-20

## Overview
Initial public/stable release of the Windows-only Barcode & OCR Sorter. Automates organizing label photos by extracting Part Numbers (pattern `P####-#####`) and Serial Numbers (`S900` + 15 digits) primarily via barcode scanning with an OCR fallback for incomplete or damaged barcodes.

## Key Features
- Multi-pass barcode scanning (rotations + image enhancement) for robust detection.
- OCR fallback pipeline (CLAHE, Otsu thresholding, multiple PSM modes, union crops) when barcodes are unreadable.
- Intelligent character normalization (e.g., O→0, I→1) and regex-based extraction for part and serial patterns.
- Output sorting into structured folders: `Sorted_Images/<Part>/<Serial>/`.
- Configurable known parts list (`part_numbers_config.txt`) to reinforce recognition.
- Debug/inspection tooling (`ocr_region_tester.py`) for tuning OCR regions.
- Clear Windows setup with `setup_windows.bat` and simple execution via `run_sorter.bat`.
- Photo capture best practices documented (angle, focus, lighting) to improve success rates.
- Consolidated troubleshooting guidance with link for issue reporting.

## Included Files
- `Barcode_Sorter_RC1.py` – Main processing script.
- `run_sorter.bat` – Activates virtual environment and launches sorter.
- `setup_windows.bat` – Creates `.venv` and installs dependencies.
- `requirements.txt` – Dependency list (Pillow, pyzbar, opencv-python, numpy, pytesseract, psutil).
- `ocr_region_tester.py` – Optional OCR tuning utility.
- `part_numbers_config.txt` – Optional known part numbers.
- `images/` – Placeholder for documentation images (angle examples).
- `README.md` – Full usage, workflow, and troubleshooting documentation.

## Requirements
- Windows 64-bit
- Python 3.12+ (64-bit)
- Visual C++ Redistributable (x64) for `pyzbar`
- Optional: Tesseract-OCR (UB Mannheim build) for OCR fallback quality

## Installation Summary
1. Install VC++ Redistributable (x64).
2. Install Tesseract (optional but recommended).
3. Run `setup_windows.bat` (creates `.venv` + installs deps).
4. Run `run_sorter.bat` and choose your image folder.

## Usage Flow
1. Select image folder.
2. Barcode pass attempts part/serial extraction.
3. OCR fallback on unresolved items.
4. Validation and normalization of extracted values.
5. Images organized; summary report displayed.

## Known Limitations
- Only tested on Windows (no cross-platform batch scripts in this release).
- Severe blur or extreme glare may still defeat both barcode and OCR passes.
- Handwritten or heavily stylized fonts may degrade OCR accuracy.
- Requires manual addition of angle guidance images to `images/` if not already present.

## Recommended Image Guidelines
- Perpendicular camera angle.
- Sharp focus, adequate diffuse lighting.
- Full label visible (do not crop too aggressively).

## Troubleshooting Highlights
- Missing barcode DLL errors: Reinstall VC++ redistributable.
- Poor OCR output: Confirm Tesseract installed; retake photo (angle/focus). 
- Immediate window close: Launch via an already-open PowerShell or CMD to read output.

## Security & Privacy
- Processes local image files only; no network upload.
- Does not store sensitive data beyond sorted filenames and optional summary report.

## Versioning Strategy
- Semantic versioning: MAJOR.MINOR.PATCH.
- Future minor versions may add cross-platform scripts or enhanced reporting.

## Next Planned Improvements
- Optional CSV/JSON export of extracted part/serial pairs.
- Logging to a dedicated file for audit/debug.
- Batch integrity check (duplicate serial detection).

## How to Report Issues
Open an issue at: https://github.com/matthewsolc66/Barcode-Reading/issues
Please attach sample images (if permissible) and a brief description of the failure mode.

## Checks Prior to Tagging
- Confirm `DEBUG` flag set to False in `Barcode_Sorter_RC1.py` distribution build.
- Confirm `.venv` excluded from ZIP.
- Test a fresh setup on a clean Windows VM (optional but recommended).

## Tag & Release Instructions (Reference)
```powershell
# Create and push tag
git tag -a v1.0.0 -m "Barcode Sorter v1.0.0 initial release"
git push origin v1.0.0

# (Optional) Regenerate ZIP if changes occurred after last archive
Compress-Archive -Path .\Barcode_Sorter_RC1.py, .\run_sorter.bat, .\setup_windows.bat, .\requirements.txt, .\ocr_region_tester.py, .\part_numbers_config.txt, .\README.md, .\images -DestinationPath "Packaging Photo Sorter.zip" -Force
```

## Final Note
Thank you for using Barcode Sorter v1.0.0. Early feedback is invaluable—please report successes, failures, and suggestions.
