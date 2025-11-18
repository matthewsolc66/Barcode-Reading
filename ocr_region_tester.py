"""
Interactive OCR Region Tester
Click and drag to select a rectangle on the image, then press ENTER to OCR that region.
Press 'r' to reset selection, 'q' to quit.
"""

import cv2
import numpy as np
import os
import re
from datetime import datetime
from PIL import Image
import pytesseract
from tkinter import Tk, filedialog

# Global variables for rectangle selection
drawing = False
ix, iy = -1, -1
fx, fy = -1, -1
image_copy = None


def load_expected_part_numbers(config_file='part_numbers_config.txt'):
    """Read list of expected P####-##### from config file (same as main script)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, config_file)
    parts = []
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith('#'):
                        continue
                    if s.startswith('P'):
                        s = s[1:]
                    if re.fullmatch(r'\d{4}-\d{5}', s):
                        parts.append(s)
        except Exception as e:
            print(f"[WARNING] Could not read config file: {e}")
    return parts if parts else None


def is_valid_serial_number(serial: str) -> bool:
    """Validate serial format S900########### with valid WWYY window (same as main script)."""
    if not re.fullmatch(r'^S900\d{15}$', serial):
        return False
    date_code = serial[11:15]
    week = int(date_code[:2])
    year = int(date_code[2:])
    if not (1 <= week <= 52):
        return False
    cy = datetime.now().year % 100
    return 24 <= year <= (cy + 1)


def correct_ocr_part_number(detected_part: str, expected_parts: list[str] | None):
    """Correct small OCR mistakes by matching to expected_parts (same as main script)."""
    if not expected_parts:
        return detected_part
    x = detected_part[1:] if detected_part.startswith('P') else detected_part
    if x in expected_parts:
        return x
    # Expanded OCR confusion matrix with common visual substitutions
    ocr = {
        '0': ['8', 'O', 'o', 'Q', 'D'],
        '1': ['7', 'I', 'i', 'l', '|', '!'],
        '2': ['Z', 'z'],
        '3': ['8', 'B'],
        '4': ['A', 'a'],
        '5': ['S', 's'],
        '6': ['8', '5', 'G', 'b', '\u00b0'],  # degree symbol often confused with 6
        '7': ['1', 'T', 't'],
        '8': ['0', '3', '6', 'B', 'b'],
        '9': ['8', 'g', 'q']
    }
    best, min_err = None, 1e9
    for exp in expected_parts:
        if len(x) != len(exp):
            continue
        err = 0.0
        for a, b in zip(x, exp):
            if a != b:
                err += 0.5 if (b in ocr.get(a, []) or a in ocr.get(b, [])) else 1.0
        if err <= 2 and err < min_err:
            min_err, best = err, exp
    return best if best else None


def correct_ocr_serial_number(detected_serial: str) -> str | None:
    """Try to salvage a nearly-correct serial by fixing plausible digit confusions in WWYY (same as main script)."""
    s = detected_serial[1:] if detected_serial.startswith('S') else detected_serial
    if not (s.startswith('900') and len(s) == 18):
        return None
    # Expanded OCR confusion matrix with common visual substitutions
    ocr = {
        '0': ['8', 'O', 'o', 'Q', 'D'],
        '1': ['7', 'I', 'i', 'l', '|', '!'],
        '2': ['Z', 'z'],
        '3': ['8', 'B'],
        '4': ['A', 'a'],
        '5': ['S', 's'],
        '6': ['8', '5', 'G', 'b', '\u00b0'],  # degree symbol often confused with 6
        '7': ['1', 'T', 't'],
        '8': ['0', '3', '6', 'B', 'b'],
        '9': ['8', 'g', 'q']
    }
    chars = list(s)
    week = s[10:12]
    year = s[12:14]
    cy = datetime.now().year % 100
    
    corrected = False

    # Correct week digits
    week_valid = False
    for i, ch in enumerate(week):
        wk_int = int(week)
        if 1 <= wk_int <= 52:
            week_valid = True
            break
        for d in '0123456789':
            if d == ch:
                continue
            if ch in ocr.get(d, []) or d in ocr.get(ch, []):
                test_week = (week[:i] + d + week[i+1:])
                test_wk_int = int(test_week)
                if 1 <= test_wk_int <= 52:
                    chars[10 + i] = d
                    week = test_week
                    week_valid = True
                    corrected = True
                    break
        if week_valid:
            break

    # Correct year digits
    year_valid = False
    for i, ch in enumerate(year):
        yr_int = int(year)
        if 24 <= yr_int <= cy:
            year_valid = True
            break
        for d in '0123456789':
            if d == ch:
                continue
            if ch in ocr.get(d, []) or d in ocr.get(ch, []):
                test_year = (year[:i] + d + year[i+1:])
                test_yr_int = int(test_year)
                if 24 <= test_yr_int <= cy:
                    chars[12 + i] = d
                    year = test_year
                    year_valid = True
                    corrected = True
                    break
        if year_valid:
            break

    if not corrected:
        return None
        
    corrected_serial = 'S' + ''.join(chars)
    if is_valid_serial_number(corrected_serial):
        return corrected_serial
    else:
        return None


def clean_ocr_text_aggressive(text: str) -> str:
    """Aggressively clean OCR text by replacing common symbol->digit confusions.
    This normalizes text BEFORE pattern matching to catch more variations.
    """
    # Replace common OCR symbol mistakes with their digit equivalents
    replacements = {
        '\u00b0': '6',  # degree symbol → 6
        'O': '0', 'o': '0', 'Q': '0', 'D': '0',  # O variations → 0
        'I': '1', 'i': '1', 'l': '1', '|': '1', '!': '1',  # I variations → 1
        'Z': '2', 'z': '2',  # Z → 2
        'B': '8', 'b': '8',  # B → 8 (also could be 3 or 6, but 8 most common)
        'S': '5', 's': '5',  # S → 5
        'G': '6', 'g': '9',  # G → 6, g → 9
        'T': '7', 't': '7',  # T → 7
        'A': '4', 'a': '4',  # A → 4
        'q': '9',  # q → 9
    }
    cleaned = text
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
    return cleaned


def extract_identifiers(text: str, expected_parts: list[str] | None):
    """Extract and validate part numbers and serial numbers from OCR text."""
    results = {
        'part_numbers': [],
        'serial_numbers': [],
        'raw_parts': [],
        'raw_serials': []
    }
    
    # Clean text aggressively (replace common symbol->digit confusions)
    cleaned = clean_ocr_text_aggressive(text)
    # Strip spaces from cleaned text
    s = cleaned.replace(' ', '')
    
    # Find part numbers
    for m in re.findall(r"\d{4}-?\d{5}", s):
        norm = m if '-' in m else f"{m[:4]}-{m[4:]}"
        results['raw_parts'].append(norm)
        
        # Try fuzzy correction if expected parts provided
        if expected_parts:
            corr = correct_ocr_part_number(norm, expected_parts)
            if corr:
                results['part_numbers'].append(f"P{corr}")
        else:
            # No expected list, accept if valid format
            if re.fullmatch(r"^\d{4}-\d{5}$", norm):
                results['part_numbers'].append(f"P{norm}")
    
    # Find serial numbers - primary search for clean 900... matches
    for m in re.findall(r"900\d{15}", s):
        S = f"S{m}"
        results['raw_serials'].append(S)
        
        if is_valid_serial_number(S):
            results['serial_numbers'].append(S)
        else:
            # Try fuzzy correction
            corr = correct_ocr_serial_number(S)
            if corr:
                results['serial_numbers'].append(corr)
    
    # Fallback: long digit runs containing 900... substring
    for long_digits in re.findall(r"\d{18,24}", s):
        sub = re.search(r"900\d{15}", long_digits)
        if sub:
            m = sub.group(0)
            S = f"S{m}"
            if S not in results['raw_serials']:
                results['raw_serials'].append(S)
            
            if is_valid_serial_number(S):
                if S not in results['serial_numbers']:
                    results['serial_numbers'].append(S)
            else:
                corr = correct_ocr_serial_number(S)
                if corr and corr not in results['serial_numbers']:
                    results['serial_numbers'].append(corr)
    
    return results


def select_roi(event, x, y, flags, param):
    """Mouse callback for selecting rectangle region"""
    global ix, iy, fx, fy, drawing, image_copy, display_image
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        fx, fy = x, y
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            fx, fy = x, y
            # Redraw image with current rectangle
            display_image = image_copy.copy()
            cv2.rectangle(display_image, (ix, iy), (fx, fy), (0, 255, 0), 2)
            cv2.imshow('OCR Region Tester', display_image)
    
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        fx, fy = x, y
        # Draw final rectangle
        display_image = image_copy.copy()
        cv2.rectangle(display_image, (ix, iy), (fx, fy), (0, 255, 0), 2)
        cv2.imshow('OCR Region Tester', display_image)


def main():
    global image_copy, display_image, ix, iy, fx, fy
    
    # Load expected part numbers from config
    expected_parts = load_expected_part_numbers()
    if expected_parts:
        print(f"[INFO] Loaded {len(expected_parts)} expected part numbers from config")
    else:
        print("[INFO] No part numbers config found - accepting all formats")
    
    # Variables for tracking original image and scale
    scale_x = 1.0
    scale_y = 1.0
    original_image = None
    
    # Open file dialog to select image
    Tk().withdraw()  # Hide root window
    file_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
            ("All files", "*.*")
        ]
    )
    
    if not file_path:
        print("No file selected. Exiting.")
        return
    
    # Load image
    original_image = cv2.imread(file_path)
    if original_image is None:
        print(f"Failed to load image: {file_path}")
        return
    
    # Get screen resolution and calculate window size (1/4 of screen)
    import ctypes
    user32 = ctypes.windll.user32
    screen_width = user32.GetSystemMetrics(0)
    screen_height = user32.GetSystemMetrics(1)
    window_width = screen_width // 4
    window_height = screen_height // 4
    
    # Resize image to fit window while maintaining aspect ratio
    h, w = original_image.shape[:2]
    aspect = w / h
    
    if w > window_width or h > window_height:
        if aspect > 1:  # Wide image
            new_w = window_width
            new_h = int(window_width / aspect)
        else:  # Tall image
            new_h = window_height
            new_w = int(window_height * aspect)
        
        # Store the scale factor for coordinate mapping
        scale_x = w / new_w
        scale_y = h / new_h
        
        # Resize for display
        display_scale = (new_w, new_h)
        image_display = cv2.resize(original_image, display_scale, interpolation=cv2.INTER_AREA)
    else:
        scale_x = 1.0
        scale_y = 1.0
        image_display = original_image.copy()
    
    # Make a copy for drawing
    image_copy = image_display.copy()
    display_image = image_display.copy()
    
    # Create window and set mouse callback
    cv2.namedWindow('OCR Region Tester', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('OCR Region Tester', image_display.shape[1], image_display.shape[0])
    cv2.setMouseCallback('OCR Region Tester', select_roi)
    
    # Make sure window is visible
    cv2.moveWindow('OCR Region Tester', 100, 100)
    
    print("\n=== OCR Region Tester ===")
    print(f"Loaded: {file_path}")
    print("\nInstructions:")
    print("  - Click and drag to select a rectangle")
    print("  - Press ENTER to run OCR on selected region")
    print("  - Press 'r' to reset selection")
    print("  - Press 'q' to quit")
    print("=" * 40)
    
    try:
        while True:
            cv2.imshow('OCR Region Tester', display_image)
            key = cv2.waitKey(1) & 0xFF
            
            # Quit
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            
            # Reset selection
            elif key == ord('r'):
                ix, iy, fx, fy = -1, -1, -1, -1
                display_image = image_copy.copy()
                cv2.imshow('OCR Region Tester', display_image)
                print("\nSelection reset.")
            
            # Run OCR on selected region
            elif key == 13:  # Enter key
                if ix >= 0 and iy >= 0 and fx >= 0 and fy >= 0:
                    # Ensure coordinates are in correct order
                    x1, x2 = min(ix, fx), max(ix, fx)
                    y1, y2 = min(iy, fy), max(iy, fy)
                    
                    if x2 - x1 < 5 or y2 - y1 < 5:
                        print("\nRegion too small! Please select a larger area.")
                        continue
                    
                    # Map display coordinates back to original image coordinates
                    orig_x1 = int(x1 * scale_x)
                    orig_x2 = int(x2 * scale_x)
                    orig_y1 = int(y1 * scale_y)
                    orig_y2 = int(y2 * scale_y)
                    
                    # Crop the region from original image
                    roi = original_image[orig_y1:orig_y2, orig_x1:orig_x2]
                    
                    # Convert to PIL Image for pytesseract
                    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(roi_rgb)
                    
                    print("\n" + "=" * 60)
                    print(f"Selected region (display): ({x1}, {y1}) to ({x2}, {y2})")
                    print(f"Original image region: ({orig_x1}, {orig_y1}) to ({orig_x2}, {orig_y2})")
                    print(f"Region size: {orig_x2-orig_x1} x {orig_y2-orig_y1} pixels")
                    print("-" * 60)
                    
                    # Run OCR with different configurations
                    configs = [
                        ('Default (All chars)', '--psm 6'),
                        ('Single line (All chars)', '--psm 7'),
                        ('Single word (All chars)', '--psm 8'),
                    ]
                    
                    all_results = []
                    
                    for config_name, config_str in configs:
                        try:
                            text = pytesseract.image_to_string(pil_image, config=config_str).strip()
                            print(f"\n{config_name}:")
                            if text:
                                print(f"  Raw OCR: '{text}'")
                                print(f"  Repr:    {repr(text)}")
                                
                                # Extract and validate identifiers
                                extracted = extract_identifiers(text, expected_parts)
                                
                                # Show extracted data
                                if extracted['part_numbers'] or extracted['serial_numbers']:
                                    print(f"  ✅ EXTRACTED:")
                                    for pn in extracted['part_numbers']:
                                        print(f"     Part Number: {pn}")
                                        all_results.append(('part', pn))
                                    for sn in extracted['serial_numbers']:
                                        valid_mark = "✓" if is_valid_serial_number(sn) else "✗"
                                        print(f"     Serial Number: {sn} {valid_mark}")
                                        all_results.append(('serial', sn))
                                
                                # Show raw matches that didn't validate
                                rejected_parts = [p for p in extracted['raw_parts'] if f"P{p}" not in extracted['part_numbers']]
                                rejected_serials = [s for s in extracted['raw_serials'] if s not in extracted['serial_numbers']]
                                
                                if rejected_parts or rejected_serials:
                                    print(f"  ⚠️  REJECTED (didn't validate):")
                                    for p in rejected_parts:
                                        print(f"     Part-like: P{p} (not in expected list or fuzzy match failed)")
                                    for s in rejected_serials:
                                        print(f"     Serial-like: {s} (invalid date code)")
                                
                                if not extracted['part_numbers'] and not extracted['serial_numbers'] and not extracted['raw_parts'] and not extracted['raw_serials']:
                                    print(f"  ℹ️  No part/serial patterns found")
                            else:
                                print("  (no text detected)")
                        except Exception as e:
                            print(f"  Error: {e}")
                    
                    # Summary
                    print("\n" + "=" * 60)
                    print("SUMMARY:")
                    unique_parts = list(set([r[1] for r in all_results if r[0] == 'part']))
                    unique_serials = list(set([r[1] for r in all_results if r[0] == 'serial']))
                    
                    if unique_parts:
                        print(f"  ✅ Found {len(unique_parts)} unique part number(s):")
                        for p in unique_parts:
                            print(f"     • {p}")
                    else:
                        print(f"  ❌ No valid part numbers found")
                    
                    if unique_serials:
                        print(f"  ✅ Found {len(unique_serials)} unique serial number(s):")
                        for s in unique_serials:
                            print(f"     • {s}")
                    else:
                        print(f"  ❌ No valid serial numbers found")
                    
                    print("=" * 60)
                    
                    # Show the cropped region in a separate window
                    cv2.imshow('Selected Region', roi)
                else:
                    print("\nNo region selected! Click and drag to select an area first.")
    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C detected. Closing windows...")
    finally:
        cv2.destroyAllWindows()
        print("\nExiting.")


if __name__ == "__main__":
    main()
