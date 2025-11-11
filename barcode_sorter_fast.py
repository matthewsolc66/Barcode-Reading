"""
Barcode Scanner and Image Sorter - FAST Version with Parallel OCR

Optimizations:
- Early exit when P & S found in barcode scan
- Early exit when P & S found in OCR
- Reduced OCR attempts from 8 to 2 (4x faster)
- Parallel OCR processing using all CPU cores
- Graceful Ctrl+C handling - sorts processed images before exit
"""

import os
import re
import shutil
import warnings
import signal
import sys
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from tkinter import Tk, filedialog, simpledialog, messagebox, Toplevel, Checkbutton, IntVar, Button, Label, Frame, Scale, HORIZONTAL
from PIL import Image, ImageOps
from pyzbar.pyzbar import decode, ZBarSymbol
import cv2
import numpy as np

# Try to import pytesseract for OCR fallback
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'


def select_part_numbers_dialog(part_numbers):
    """Show GUI dialog with checkboxes to select which part numbers to process."""
    if not part_numbers:
        return None, None
    
    selected_parts = []
    selected_workers = None
    max_workers = os.cpu_count() or 4
    
    # Create dialog window
    dialog = Toplevel()
    dialog.title("Select Part Numbers and Workers")
    dialog.geometry("450x500")
    dialog.resizable(False, False)
    
    # Center the window
    dialog.update_idletasks()
    width = dialog.winfo_width()
    height = dialog.winfo_height()
    x = (dialog.winfo_screenwidth() // 2) - (width // 2)
    y = (dialog.winfo_screenheight() // 2) - (height // 2)
    dialog.geometry(f'{width}x{height}+{x}+{y}')
    
    # Title label
    title_label = Label(dialog, text="Select which part numbers to process:", 
                       font=("Arial", 11, "bold"), pady=10)
    title_label.pack()
    
    # Frame for checkboxes with scrollbar if needed
    checkbox_frame = Frame(dialog)
    checkbox_frame.pack(pady=10, padx=20, fill="both", expand=True)
    
    # Create checkboxes
    checkbox_vars = []
    for part in part_numbers:
        var = IntVar(value=1)  # 1 = checked by default
        cb = Checkbutton(checkbox_frame, text=f"P{part}", variable=var, 
                        font=("Arial", 10), anchor="w")
        cb.pack(fill="x", pady=2)
        checkbox_vars.append((part, var))
    
    # Separator
    separator = Frame(dialog, height=2, bd=1, relief="sunken")
    separator.pack(fill="x", padx=20, pady=10)
    
    # Worker selection frame
    worker_frame = Frame(dialog)
    worker_frame.pack(pady=10, padx=20)
    
    Label(worker_frame, text="Parallel Workers (CPU Cores):", 
          font=("Arial", 10, "bold")).pack()
    
    # Slider value display
    worker_value_label = Label(worker_frame, text=f"{max_workers}", 
                               font=("Arial", 12))
    worker_value_label.pack(pady=5)
    
    def update_worker_label(val):
        worker_value_label.config(text=f"{int(float(val))}")
    
    # Slider
    worker_slider = Scale(worker_frame, from_=1, to=max_workers, 
                         orient=HORIZONTAL, length=350,
                         command=update_worker_label,
                         showvalue=False, tickinterval=1,
                         resolution=1)
    worker_slider.set(max_workers)  # Default to max
    worker_slider.pack()
    
    # Button frame
    button_frame = Frame(dialog)
    button_frame.pack(pady=15)
    
    def on_ok():
        nonlocal selected_parts, selected_workers
        selected_parts = [part for part, var in checkbox_vars if var.get() == 1]
        selected_workers = worker_slider.get()
        dialog.destroy()
    
    def on_cancel():
        nonlocal selected_parts, selected_workers
        selected_parts = None
        selected_workers = None
        dialog.destroy()
    
    def select_all():
        for _, var in checkbox_vars:
            var.set(1)
    
    def deselect_all():
        for _, var in checkbox_vars:
            var.set(0)
    
    # Buttons
    Button(button_frame, text="Select All", command=select_all, width=12).pack(side="left", padx=5)
    Button(button_frame, text="Deselect All", command=deselect_all, width=12).pack(side="left", padx=5)
    Button(button_frame, text="OK", command=on_ok, width=12, bg="#4CAF50", fg="white").pack(side="left", padx=5)
    Button(button_frame, text="Cancel", command=on_cancel, width=12).pack(side="left", padx=5)
    
    # Make dialog modal
    dialog.transient()
    dialog.grab_set()
    dialog.wait_window()
    
    return selected_parts, selected_workers


def load_expected_part_numbers(config_file='part_numbers_config.txt'):
    """Load expected part numbers from config file."""
    expected_parts = []
    
    # Try to load from config file in script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, config_file)
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines and comments
                    if line and not line.startswith('#'):
                        # Remove P prefix if present
                        if line.startswith('P'):
                            line = line[1:]
                        # Validate format
                        if re.fullmatch(r'\d{4}-\d{5}', line):
                            expected_parts.append(line)
            
            if expected_parts:
                print(f"[INFO] Loaded {len(expected_parts)} expected part numbers from config:")
                for part in expected_parts:
                    print(f"       - P{part}")
            else:
                print(f"[INFO] Config file empty - accepting all part numbers")
        except Exception as e:
            print(f"[WARNING] Could not read config file: {e}")
            print("[INFO] Accepting all part numbers")
    else:
        print(f"[INFO] No config file found at {config_path}")
        print("[INFO] Accepting all part numbers")
    
    return expected_parts if expected_parts else None


def correct_ocr_part_number(detected_part, expected_parts):
    """
    Try to correct OCR errors by finding closest match in expected parts.
    Returns corrected part number or None if no good match found.
    """
    if not expected_parts:
        return detected_part
    
    # Remove P prefix if present
    if detected_part.startswith('P'):
        detected_part = detected_part[1:]
    
    # Exact match
    if detected_part in expected_parts:
        return detected_part
    
    # Common OCR character substitutions based on visual similarity
    ocr_mistakes = {
        '0': ['8', 'O'],      # 0 looks like 8 when blurry/partial, or letter O
        '1': ['7', 'I'],      # 1 looks like 7 at angles, or letter I
        '2': ['Z'],           # 2 can look like Z
        '3': ['8'],           # 3 looks like 8 when middle closes
        '4': ['A'],           # 4 can look like A
        '5': ['S'],           # 5 can look like letter S
        '6': ['8', '5'],      # 6 looks like 8 or 5 when partial
        '7': ['1'],           # 7 looks like 1 at angles
        '8': ['0', '3', '6', 'B'],  # 8 is commonly confused with 0, 3, 6, and B
        '9': ['8']            # 9 looks like 8 when bottom closes
    }
    
    best_match = None
    min_errors = float('inf')
    
    for expected in expected_parts:
        if len(detected_part) != len(expected):
            continue
        
        errors = 0
        for i, (detected_char, expected_char) in enumerate(zip(detected_part, expected)):
            if detected_char != expected_char:
                # Check if it's a common OCR mistake
                if expected_char in ocr_mistakes.get(detected_char, []) or \
                   detected_char in ocr_mistakes.get(expected_char, []):
                    errors += 0.5  # Half penalty for common OCR mistake
                else:
                    errors += 1  # Full penalty for uncommon difference
        
        # Accept if at most 2 character differences (or 4 OCR mistakes)
        if errors < min_errors and errors <= 2:
            min_errors = errors
            best_match = expected
    
    if best_match:
        if best_match != detected_part:
            print(f"       [CORRECTED] {detected_part} → {best_match}")
        return best_match
    
    return None  # No good match found


def apply_exif_orientation(pil_image):
    """Apply EXIF orientation to correctly orient the image."""
    try:
        orientation = pil_image.getexif().get(274, 1)
        if orientation != 1:
            pil_image = ImageOps.exif_transpose(pil_image)
    except Exception:
        pass
    return pil_image


def enhance_for_barcode_reading(image_array):
    """Apply CLAHE enhancement for better barcode detection."""
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY) if len(image_array.shape) == 3 else image_array
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def is_valid_serial_number(serial):
    """Validate serial number format and date code.
    Format: S + 10-digit vendor code + 4-digit date (WWYY) + 4-digit part ID
    Example: S9000167952 0825 8026 where 08=week, 25=year
    """
    if not re.fullmatch(r'^S900\d{15}$', serial):
        return False
    
    # Extract date code (characters 11-14, 0-indexed: positions 11,12,13,14)
    # S=0, vendor=1-10, date=11-14, part=15-18
    date_code = serial[11:15]
    week = int(date_code[0:2])
    year = int(date_code[2:4])
    
    # Validate week (01-52)
    if week < 1 or week > 52:
        return False
    
    # Validate year (24 to current_year+1)
    current_year = datetime.now().year % 100  # Get last 2 digits (e.g., 2025 -> 25)
    if year < 24 or year > (current_year + 1):
        return False
    
    return True


def has_required_barcodes(barcodes_found):
    """Check if we have both part number and valid serial number."""
    has_part = any(re.fullmatch(r'^P\d{4}-\d{5}$', b) for b in barcodes_found)
    has_serial = any(is_valid_serial_number(b) for b in barcodes_found)
    return has_part and has_serial


def decode_barcodes(pil_image):
    """Decode Code 128 barcodes using multiple preprocessing techniques with early exit."""
    barcodes_found = []
    barcode_regions = []  # Store barcode locations for zoomed retry
    img_array = np.array(pil_image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
    
    # Try 1: Direct scan - also collect barcode locations
    decoded_objects = decode(pil_image, symbols=[ZBarSymbol.CODE128])
    for barcode in decoded_objects:
        data = barcode.data.decode('utf-8')
        if data not in barcodes_found:
            barcodes_found.append(data)
        # Store region for potential zoom-in
        barcode_regions.append(barcode.rect)
    if has_required_barcodes(barcodes_found):
        return barcodes_found
    
    # Try 2: Glare removal
    _, glare_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    glare_mask = cv2.dilate(glare_mask, np.ones((3,3), np.uint8), iterations=1)
    deglared = cv2.inpaint(gray, glare_mask, 3, cv2.INPAINT_TELEA)
    decoded_objects = decode(Image.fromarray(deglared), symbols=[ZBarSymbol.CODE128])
    for barcode in decoded_objects:
        data = barcode.data.decode('utf-8')
        if data not in barcodes_found:
            barcodes_found.append(data)
        if barcode.rect not in barcode_regions:
            barcode_regions.append(barcode.rect)
    if has_required_barcodes(barcodes_found):
        return barcodes_found
    
    # Try 3: CLAHE enhancement
    enhanced = enhance_for_barcode_reading(img_array)
    decoded_objects = decode(Image.fromarray(enhanced), symbols=[ZBarSymbol.CODE128])
    for barcode in decoded_objects:
        data = barcode.data.decode('utf-8')
        if data not in barcodes_found:
            barcodes_found.append(data)
        if barcode.rect not in barcode_regions:
            barcode_regions.append(barcode.rect)
    if has_required_barcodes(barcodes_found):
        return barcodes_found
    
    # Try 4: Binary thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    for barcode in decode(Image.fromarray(binary), symbols=[ZBarSymbol.CODE128]):
        data = barcode.data.decode('utf-8')
        if data not in barcodes_found:
            barcodes_found.append(data)
    if has_required_barcodes(barcodes_found):
        return barcodes_found
    
    # Try 5: Adaptive thresholding
    for block_size in [11, 21, 51]:
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 10)
        for barcode in decode(Image.fromarray(adaptive), symbols=[ZBarSymbol.CODE128]):
            data = barcode.data.decode('utf-8')
            if data not in barcodes_found:
                barcodes_found.append(data)
        if has_required_barcodes(barcodes_found):
            return barcodes_found
    
    # Try 6: Inverted binary
    for barcode in decode(Image.fromarray(cv2.bitwise_not(binary)), symbols=[ZBarSymbol.CODE128]):
        data = barcode.data.decode('utf-8')
        if data not in barcodes_found:
            barcodes_found.append(data)
    if has_required_barcodes(barcodes_found):
        return barcodes_found
    
    # Try 7: Morphological operations
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((2,2), np.uint8))
    for barcode in decode(Image.fromarray(morph), symbols=[ZBarSymbol.CODE128]):
        data = barcode.data.decode('utf-8')
        if data not in barcodes_found:
            barcodes_found.append(data)
    if has_required_barcodes(barcodes_found):
        return barcodes_found
    
    # Try 8: Zoom into detected barcode regions and retry
    # If we found some barcodes but not P&S, try zooming into those areas
    if barcode_regions and not has_required_barcodes(barcodes_found):
        for rect in barcode_regions:
            # Expand region by 50% in all directions to get context
            x, y, w, h = rect.left, rect.top, rect.width, rect.height
            expand_factor = 0.5
            x_expand = int(w * expand_factor)
            y_expand = int(h * expand_factor)
            
            x1 = max(0, x - x_expand)
            y1 = max(0, y - y_expand)
            x2 = min(gray.shape[1], x + w + x_expand)
            y2 = min(gray.shape[0], y + h + y_expand)
            
            # Extract and upscale the region 2x
            region = gray[y1:y2, x1:x2]
            if region.size == 0:
                continue
            
            zoomed = cv2.resize(region, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            
            # Try multiple preprocessing on zoomed region
            for attempt_img in [zoomed, 
                               cv2.threshold(zoomed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                               enhance_for_barcode_reading(zoomed)]:
                for barcode in decode(Image.fromarray(attempt_img), symbols=[ZBarSymbol.CODE128]):
                    data = barcode.data.decode('utf-8')
                    if data not in barcodes_found:
                        barcodes_found.append(data)
                if has_required_barcodes(barcodes_found):
                    return barcodes_found
    
    return barcodes_found


def extract_text_with_ocr(pil_image, expected_part_numbers=None):
    """Use OCR to extract part/serial numbers with early exit - OPTIMIZED."""
    if not TESSERACT_AVAILABLE:
        return []
    
    found_codes = []
    img_array = np.array(pil_image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
    
    # Apply CLAHE enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # OPTIMIZED: Only 2 preprocessing methods (reduced from 4)
    preprocessed = [
        enhanced,
        cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    ]
    
    # OPTIMIZED: Only PSM 6 (removed PSM 11)
    for img in preprocessed:
        try:
            text = pytesseract.image_to_string(Image.fromarray(img), 
                config=f'--psm 6 -c tessedit_char_whitelist=0123456789-')
            
            # Extract part numbers
            for match in re.findall(r'\d{4}-?\d{5}', text):
                normalized = match if '-' in match else f"{match[:4]}-{match[4:]}"
                
                # Try to correct OCR errors if expected parts provided
                if expected_part_numbers:
                    corrected = correct_ocr_part_number(normalized, expected_part_numbers)
                    if corrected:
                        normalized = corrected
                    else:
                        continue  # Skip if no good match
                
                normalized = f"P{normalized}"
                if normalized not in found_codes and re.fullmatch(r'^P\d{4}-\d{5}$', normalized):
                    found_codes.append(normalized)
            
            # Extract serial numbers (must start with 900 and have valid date code)
            for match in re.findall(r'900\d{15}', text):
                normalized = f"S{match}"
                # Validate serial number before adding
                if normalized not in found_codes and is_valid_serial_number(normalized):
                    found_codes.append(normalized)
            
            # EARLY EXIT: Stop as soon as we have both part and serial
            if has_required_barcodes(found_codes):
                return found_codes
                
        except Exception:
            continue
    
    return found_codes


def classify_barcode(data):
    """Classify barcode based on pattern and validate serial numbers."""
    if re.fullmatch(r'^P\d{4}-\d{5}$', data):
        return "Part Number"
    elif is_valid_serial_number(data):
        return "Serial Number"
    elif re.fullmatch(r'^Q\d+$', data):
        return "Quantity"
    elif re.fullmatch(r'^\d{1,2}L[A-Z]{1,2}$', data):
        return "Origin"
    elif re.fullmatch(r'^\d+T$', data):
        return "Lot Number"
    return "Other"


def extract_identifiers(barcodes):
    """Extract part number and serial number from barcodes."""
    part_number = None
    serial_number = None
    
    for barcode_data in barcodes:
        barcode_type = classify_barcode(barcode_data)
        if barcode_type == "Part Number" and not part_number:
            part_number = barcode_data
        elif barcode_type == "Serial Number" and not serial_number:
            serial_number = barcode_data
    
    return part_number, serial_number


def create_folder_structure(base_folder, part_number, serial_number):
    """Create folder structure based on what was detected."""
    # If no part number (even if serial exists), put in unsorted
    if not part_number:
        target_folder = os.path.join(base_folder, "_Unsorted")
    # If we have part number and serial number
    elif part_number and serial_number:
        part_folder = part_number[1:] if part_number.startswith('P') else part_number
        serial_folder = serial_number[-8:] if len(serial_number) >= 8 else serial_number
        target_folder = os.path.join(base_folder, part_folder, serial_folder)
    # If only part number (no serial)
    elif part_number:
        part_folder = part_number[1:] if part_number.startswith('P') else part_number
        target_folder = os.path.join(base_folder, part_folder, "_No_Serial")
    else:
        # Fallback (shouldn't reach here based on logic above)
        target_folder = os.path.join(base_folder, "_Unsorted")
    
    os.makedirs(target_folder, exist_ok=True)
    return target_folder





def process_image_ocr(image_path, expected_part_numbers):
    """Process a single image with OCR - used for parallel processing."""
    start_time = datetime.now()
    try:
        pil_image = Image.open(image_path)
        pil_image = apply_exif_orientation(pil_image)
        ocr_codes = extract_text_with_ocr(pil_image, expected_part_numbers)
        processing_time = (datetime.now() - start_time).total_seconds()
        return {
            'image_path': image_path,
            'ocr_codes': ocr_codes,
            'success': True,
            'processing_time': processing_time
        }
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        return {
            'image_path': image_path,
            'ocr_codes': [],
            'success': False,
            'error': str(e),
            'processing_time': processing_time
        }


def process_image_barcode_only(image_path):
    """Quick barcode scan only - no OCR fallback."""
    start_time = datetime.now()
    try:
        pil_image = Image.open(image_path)
        pil_image = apply_exif_orientation(pil_image)
        
        # Scan for barcodes
        all_barcodes = decode_barcodes(pil_image)
        part_number, serial_number = extract_identifiers(all_barcodes)
        
        return {
            'image_path': image_path,
            'part_number': part_number,
            'serial_number': serial_number,
            'all_barcodes': all_barcodes,
            'success': True,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'processing_time': (datetime.now() - start_time).total_seconds()
        }
    except Exception as e:
        return {
            'image_path': image_path,
            'part_number': None,
            'serial_number': None,
            'all_barcodes': [],
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'processing_time': (datetime.now() - start_time).total_seconds()
        }


def generate_report(results, output_folder, total_time=None):
    """Generate a text report of all processed images."""
    report_path = os.path.join(output_folder, "_sorting_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("BARCODE SORTING REPORT\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        total = len(results)
        successful = sum(1 for r in results if r['success'])
        with_part = sum(1 for r in results if r.get('part_number'))
        with_serial = sum(1 for r in results if r.get('serial_number'))
        with_both = sum(1 for r in results if r.get('part_number') and r.get('serial_number'))
        
        f.write(f"Total images processed: {total}\n")
        f.write(f"Successfully processed: {successful}\n")
        f.write(f"Images with part number: {with_part}\n")
        f.write(f"Images with serial number: {with_serial}\n")
        f.write(f"Images with both: {with_both}\n")
        f.write(f"Unidentified: {total - max(with_part, with_serial)}\n")
        if total_time:
            f.write(f"\nTotal processing time: {total_time:.1f}s ({total_time/60:.1f} minutes)\n")
        f.write("\n" + "=" * 80 + "\n\n")
        
        f.write("DETAILED RESULTS:\n\n")
        for result in results:
            f.write(f"Filename: {result['filename']}\n")
            f.write(f"  Timestamp: {result.get('timestamp', 'N/A')}\n")
            f.write(f"  Processing Time: {result.get('processing_time', 0):.2f}s\n")
            if result['success']:
                f.write(f"  Part Number: {result['part_number'] or 'Not found'}\n")
                f.write(f"  Serial Number: {result['serial_number'] or 'Not found'}\n")
                f.write(f"  All Barcodes: {', '.join(result['all_barcodes']) if result['all_barcodes'] else 'None'}\n")
                f.write(f"  Sorted to: {result['target_folder']}\n")
            else:
                f.write(f"  ERROR: {result.get('error', 'Unknown error')}\n")
            f.write("\n")
    
    print(f"\n[INFO] Report saved to: {report_path}")


def main():
    """Main function with graceful Ctrl+C handling."""
    script_start_time = datetime.now()
    print("=" * 80)
    print("BARCODE IMAGE SORTER - FAST VERSION")
    print(f"Started at: {script_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print("[INFO] Press Ctrl+C at any time to stop and sort processed images")
    
    # Variables to store state for Ctrl+C handling
    all_results = []
    output_folder = None
    interrupted = False
    
    def signal_handler(sig, frame):
        nonlocal interrupted
        interrupted = True
        print("\n\n[INFO] Ctrl+C detected! Finishing current operations and sorting processed images...")
    
    # Register Ctrl+C handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Load expected part numbers from config file
    all_expected_parts = load_expected_part_numbers()
    
    # Select input folder
    root = Tk()
    root.withdraw()
    
    # Default max workers
    max_workers = os.cpu_count() or 4
    
    # If we have expected parts, show selection dialog
    if all_expected_parts:
        expected_part_numbers, max_workers = select_part_numbers_dialog(all_expected_parts)
        if expected_part_numbers is None:
            print("[INFO] Cancelled. Exiting.")
            return
        if not expected_part_numbers:
            messagebox.showwarning("No Selection", "No part numbers selected. Exiting.")
            print("[INFO] No part numbers selected. Exiting.")
            return
        print(f"\n[INFO] Selected {len(expected_part_numbers)} part numbers to process:")
        for part in expected_part_numbers:
            print(f"       - P{part}")
        print(f"[INFO] Using {max_workers} parallel workers (CPU cores)")
    else:
        expected_part_numbers = None
        print("[INFO] No expected part numbers configured - accepting all part numbers")
        print(f"[INFO] Using {max_workers} parallel workers (CPU cores)")
    
    input_folder = filedialog.askdirectory(title="Select Folder Containing Images to Sort")
    
    if not input_folder:
        print("[INFO] No folder selected. Exiting.")
        return
    
    print(f"\n[INFO] Input folder: {input_folder}")
    
    # Create output folder
    output_folder = os.path.join(input_folder, "Sorted_Images")
    os.makedirs(output_folder, exist_ok=True)
    print(f"[INFO] Output folder: {output_folder}")
    
    # Find image files
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    image_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) 
                   if f.lower().endswith(image_extensions)]
    
    if not image_files:
        print(f"\n[ERROR] No image files found")
        return
    
    print(f"\n[INFO] Found {len(image_files)} image(s)")
    print("=" * 80)
    
    # PASS 1: Parallel barcode scanning
    print("\n[INFO] Pass 1: Quick barcode scanning (PARALLEL PROCESSING)...")
    print(f"[INFO] Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    pass1_start = datetime.now()
    
    # Use the max_workers selected by user (or default)
    print(f"[INFO] Using {max_workers} parallel workers (CPU cores)")
    print("=" * 80)
    
    completed = 0
    executor = None
    try:
        executor = ProcessPoolExecutor(max_workers=max_workers)
        # Submit all barcode scanning jobs
        future_to_path = {
            executor.submit(process_image_barcode_only, image_path): image_path
            for image_path in image_files
        }
        
        # Process completed jobs as they finish
        for future in as_completed(future_to_path):
            if interrupted:
                print("\n[INFO] Stopping Pass 1 early...")
                executor.shutdown(wait=False, cancel_futures=True)
                break
                
            completed += 1
            image_path = future_to_path[future]
            
            try:
                result = future.result()
                all_results.append(result)
                
                print(f"[{completed}/{len(image_files)}] Scanned: {os.path.basename(image_path)}", end=" ")
                
                if result.get('skipped'):
                    print(f"⊘ Skipped (plain packaging) ({result['processing_time']:.2f}s)")
                elif result['success']:
                    if result['part_number'] and result['serial_number']:
                        print(f"✓ Part & Serial ({result['processing_time']:.1f}s)")
                    elif result['part_number']:
                        print(f"✓ Part only ({result['processing_time']:.1f}s)")
                    elif result['serial_number']:
                        print(f"✓ Serial only ({result['processing_time']:.1f}s)")
                    elif result['all_barcodes']:
                        print(f"⚠ {len(result['all_barcodes'])} barcode(s) ({result['processing_time']:.1f}s)")
                    else:
                        print(f"⚠ No barcodes ({result['processing_time']:.1f}s)")
                else:
                    print(f"✗ Error ({result['processing_time']:.1f}s)")
            except Exception as e:
                print(f"[{completed}/{len(image_files)}] Scanned: {os.path.basename(image_path)} ✗ Failed")
                all_results.append({
                    'image_path': image_path,
                    'part_number': None,
                    'serial_number': None,
                    'all_barcodes': [],
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'processing_time': 0
                })
    except KeyboardInterrupt:
        interrupted = True
        print("\n[INFO] Keyboard interrupt in Pass 1")
    finally:
        if executor:
            executor.shutdown(wait=False, cancel_futures=True)
    
    pass1_duration = (datetime.now() - pass1_start).total_seconds()
    print("=" * 80)
    print(f"[INFO] Pass 1 completed in {pass1_duration:.1f}s ({pass1_duration/60:.1f} minutes)")
    print(f"[INFO] Processed {len(all_results)} images")
    print(f"[INFO] Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Skip Pass 2 if interrupted
    if interrupted:
        print("\n[INFO] Skipping Pass 2 due to interruption")
    else:
        # PASS 2: Parallel OCR fallback
        images_needing_ocr = [r for r in all_results 
                              if not has_required_barcodes(r['all_barcodes'])]
        
        if images_needing_ocr:
            print(f"\n[INFO] Pass 2: OCR fallback for {len(images_needing_ocr)} images (PARALLEL PROCESSING)...")
            print(f"[INFO] Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 80)
            pass2_start = datetime.now()
            
            # Create a mapping of image_path to result for quick lookup
            result_map = {r['image_path']: r for r in images_needing_ocr}
            
            # Use the same max_workers from Pass 1
            print(f"[INFO] Using {max_workers} parallel workers (CPU cores)")
            print("=" * 80)
            
            completed = 0
            executor2 = None
            try:
                executor2 = ProcessPoolExecutor(max_workers=max_workers)
                # Submit all OCR jobs
                future_to_path = {
                    executor2.submit(process_image_ocr, r['image_path'], expected_part_numbers): r['image_path']
                    for r in images_needing_ocr
                }
                
                # Process completed jobs as they finish
                for future in as_completed(future_to_path):
                    if interrupted:
                        print("\n[INFO] Stopping Pass 2 early...")
                        executor2.shutdown(wait=False, cancel_futures=True)
                        break
                    
                    completed += 1
                    image_path = future_to_path[future]
                    result = result_map[image_path]
                    
                    try:
                        ocr_result = future.result()
                        ocr_codes = ocr_result['ocr_codes']
                        processing_time = ocr_result.get('processing_time', 0)
                        
                        print(f"[{completed}/{len(images_needing_ocr)}] OCR: {os.path.basename(image_path)}", end=" ")
                        
                        if ocr_codes:
                            for code in ocr_codes:
                                if code not in result['all_barcodes']:
                                    result['all_barcodes'].append(code)
                            result['part_number'], result['serial_number'] = extract_identifiers(result['all_barcodes'])
                            print(f"✓ Found P/S ({processing_time:.1f}s)")
                        else:
                            print(f"⚠ Nothing found ({processing_time:.1f}s)")
                    except Exception as e:
                        print(f"[{completed}/{len(images_needing_ocr)}] OCR: {os.path.basename(image_path)} ✗ Failed")
            except KeyboardInterrupt:
                interrupted = True
                print("\n[INFO] Keyboard interrupt in Pass 2")
            finally:
                if executor2:
                    executor2.shutdown(wait=False, cancel_futures=True)
            
            pass2_duration = (datetime.now() - pass2_start).total_seconds()
            print("=" * 80)
            print(f"[INFO] Pass 2 completed in {pass2_duration:.1f}s ({pass2_duration/60:.1f} minutes)")
            print(f"[INFO] Processed {completed}/{len(images_needing_ocr)} images")
            print(f"[INFO] Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print("\n[INFO] Pass 2: Skipped (all images have barcodes)")
    
    # Build serial-to-part mapping
    serial_to_part = {r['serial_number']: r['part_number'] 
                      for r in all_results if r['part_number'] and r['serial_number']}
    
    # PASS 3: Organize and copy (always run, even if interrupted)
    print("\n[INFO] Pass 3: Organizing and copying images...")
    if interrupted:
        print("[INFO] Sorting images processed before interruption...")
    print(f"[INFO] Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    pass3_start = datetime.now()
    final_results = []
    
    try:
        for idx, result in enumerate(all_results, 1):
            if interrupted:
                print(f"\n[INFO] Stopping Pass 3 early at {idx}/{len(all_results)} images...")
                break
            
            copy_start = datetime.now()
            print(f"[{idx}/{len(all_results)}] Organizing: {os.path.basename(result['image_path'])}", end=" ")
            
            part_number = result['part_number']
            serial_number = result['serial_number']
            
            # Match serial to part if needed
            if serial_number and not part_number and serial_number in serial_to_part:
                part_number = serial_to_part[serial_number]
            
            # Copy file with appropriate folder structure
            target_folder = create_folder_structure(output_folder, part_number, serial_number)
            target_path = os.path.join(target_folder, os.path.basename(result['image_path']))
            
            copy_time = (datetime.now() - copy_start).total_seconds()
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            try:
                shutil.copy2(result['image_path'], target_path)
                
                # Show appropriate message based on destination
                if not part_number:
                    print(f"→ Unsorted ({copy_time:.2f}s)")
                else:
                    print(f"✓ ({copy_time:.2f}s)")
                
                final_results.append({
                    'filename': os.path.basename(result['image_path']),
                    'part_number': part_number,
                    'serial_number': serial_number,
                    'all_barcodes': result['all_barcodes'],
                    'target_folder': target_folder,
                    'success': True,
                    'timestamp': timestamp,
                    'processing_time': result.get('processing_time', 0)
                })
            except Exception as e:
                print(f"✗ ({copy_time:.2f}s)")
                final_results.append({
                    'filename': os.path.basename(result['image_path']),
                    'part_number': part_number,
                    'serial_number': serial_number,
                    'all_barcodes': result['all_barcodes'],
                    'target_folder': None,
                    'success': False,
                    'error': str(e),
                    'timestamp': timestamp,
                    'processing_time': result.get('processing_time', 0)
                })
    except KeyboardInterrupt:
        interrupted = True
        print("\n[INFO] Keyboard interrupt in Pass 3")
    
    pass3_duration = (datetime.now() - pass3_start).total_seconds()
    print("=" * 80)
    print(f"[INFO] Pass 3 completed in {pass3_duration:.1f}s ({pass3_duration/60:.1f} minutes)")
    print(f"[INFO] Copied {len(final_results)}/{len(all_results)} images")
    print(f"[INFO] Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Generate report
    print("\n" + "=" * 80)
    total_duration = (datetime.now() - script_start_time).total_seconds()
    generate_report(final_results, output_folder, total_duration)
    
    # Calculate total time
    print("\n" + "=" * 80)
    if interrupted:
        print("[INFO] Processing interrupted by user!")
        print(f"[INFO] Processed and sorted {len(final_results)}/{len(all_results)} images")
        print(f"[INFO] Total time: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")
        print(f"[INFO] Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[INFO] Images organized in: {output_folder}")
        print("=" * 80)
        # Force exit when interrupted to release terminal
        sys.exit(0)
    else:
        print("[INFO] Processing complete!")
    print(f"[INFO] Total time: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")
    print(f"[INFO] Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[INFO] Images organized in: {output_folder}")
    print("=" * 80)
    
    # Open output folder
    try:
        os.startfile(output_folder)
    except Exception:
        pass
    
    # Show completion dialog
    messagebox.showinfo(
        "Processing Complete",
        f"Successfully processed {len(final_results)} images!\n\n"
        f"Total time: {total_duration/60:.1f} minutes\n\n"
        f"Results saved to:\n{output_folder}\n\n"
        f"Check '_sorting_report.txt' for details."
    )


if __name__ == "__main__":
    main()
