"""
Barcode Scanner and Image Sorter - FAST Version with Parallel OCR

Optimizations:
- Early exit when P & S found in barcode scan
- Early exit when P & S found in OCR
- Reduced OCR attempts from 8 to 2 (4x faster)
- Parallel OCR processing using all CPU cores
"""

import os
import re
import shutil
import warnings
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from tkinter import Tk, filedialog, simpledialog, messagebox
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
    img_array = np.array(pil_image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
    
    # Try 1: Direct scan
    for barcode in decode(pil_image, symbols=[ZBarSymbol.CODE128]):
        data = barcode.data.decode('utf-8')
        if data not in barcodes_found:
            barcodes_found.append(data)
    if has_required_barcodes(barcodes_found):
        return barcodes_found
    
    # Try 2: Glare removal
    _, glare_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    glare_mask = cv2.dilate(glare_mask, np.ones((3,3), np.uint8), iterations=1)
    deglared = cv2.inpaint(gray, glare_mask, 3, cv2.INPAINT_TELEA)
    for barcode in decode(Image.fromarray(deglared), symbols=[ZBarSymbol.CODE128]):
        data = barcode.data.decode('utf-8')
        if data not in barcodes_found:
            barcodes_found.append(data)
    if has_required_barcodes(barcodes_found):
        return barcodes_found
    
    # Try 3: CLAHE enhancement
    enhanced = enhance_for_barcode_reading(img_array)
    for barcode in decode(Image.fromarray(enhanced), symbols=[ZBarSymbol.CODE128]):
        data = barcode.data.decode('utf-8')
        if data not in barcodes_found:
            barcodes_found.append(data)
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
    
    return barcodes_found


def extract_text_with_ocr(pil_image, expected_part_number=None):
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
    
    # Determine matching mode
    match_exact = expected_part_number and '-' in expected_part_number
    match_prefix = expected_part_number and '-' not in expected_part_number
    
    # OPTIMIZED: Only PSM 6 (removed PSM 11)
    for img in preprocessed:
        try:
            text = pytesseract.image_to_string(Image.fromarray(img), 
                config=f'--psm 6 -c tessedit_char_whitelist=0123456789-')
            
            # Extract part numbers
            for match in re.findall(r'\d{4}-?\d{5}', text):
                normalized = match if '-' in match else f"{match[:4]}-{match[4:]}"
                
                if match_exact and normalized != expected_part_number:
                    continue
                if match_prefix and not normalized.startswith(expected_part_number + '-'):
                    continue
                
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


def create_folder_structure(base_folder, part_number, serial_number, is_skipped=False):
    """Create folder structure based on what was detected."""
    # If image was skipped (plain packaging/security seal)
    if is_skipped:
        target_folder = os.path.join(base_folder, "_Security_Seals")
    # If no part number (even if serial exists), put in unsorted
    elif not part_number:
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


def is_plain_packaging_image(pil_image):
    """Detect plain packaging images (no labels/barcodes visible) to skip processing."""
    try:
        # Convert to numpy array and resize for speed
        img_array = np.array(pil_image)
        small_img = cv2.resize(img_array, (400, 400))
        
        # Convert to grayscale
        if len(small_img.shape) == 3:
            gray = cv2.cvtColor(small_img, cv2.COLOR_RGB2GRAY)
        else:
            gray = small_img
        
        # Look for bright regions (white/beige labels)
        # Labels are typically 150-255 in grayscale (lowered threshold to catch more labels)
        _, bright_mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        bright_percentage = (np.sum(bright_mask > 0) / bright_mask.size) * 100
        
        # VERY STRICT: Only skip if there's almost NOTHING bright
        # This image has a large white label, so bright_percentage should be >10%
        # We only skip if <5% is bright (basically just security seals/small logos)
        should_skip = (bright_percentage < 5)
        
        return should_skip
        
    except Exception:
        # If analysis fails, process the image normally
        return False


def process_image_ocr(image_path, expected_part_number):
    """Process a single image with OCR - used for parallel processing."""
    try:
        pil_image = Image.open(image_path)
        pil_image = apply_exif_orientation(pil_image)
        ocr_codes = extract_text_with_ocr(pil_image, expected_part_number)
        return {
            'image_path': image_path,
            'ocr_codes': ocr_codes,
            'success': True
        }
    except Exception as e:
        return {
            'image_path': image_path,
            'ocr_codes': [],
            'success': False,
            'error': str(e)
        }


def process_image_barcode_only(image_path):
    """Quick barcode scan only - no OCR fallback. Skips plain packaging images."""
    start_time = datetime.now()
    try:
        pil_image = Image.open(image_path)
        pil_image = apply_exif_orientation(pil_image)
        
        # Check if this is a plain packaging image
        if is_plain_packaging_image(pil_image):
            return {
                'image_path': image_path,
                'part_number': None,
                'serial_number': None,
                'all_barcodes': [],
                'success': True,
                'skipped': True,
                'skip_reason': 'Plain packaging detected',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
        
        barcodes = decode_barcodes(pil_image)
        part_number, serial_number = extract_identifiers(barcodes)
        
        return {
            'image_path': image_path,
            'part_number': part_number,
            'serial_number': serial_number,
            'all_barcodes': barcodes,
            'success': True,
            'skipped': False,
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
            'skipped': False,
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
    """Main function."""
    script_start_time = datetime.now()
    print("=" * 80)
    print("BARCODE IMAGE SORTER - FAST VERSION")
    print(f"Started at: {script_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Select input folder
    root = Tk()
    root.withdraw()
    input_folder = filedialog.askdirectory(title="Select Folder Containing Images to Sort")
    
    if not input_folder:
        print("[INFO] No folder selected. Exiting.")
        return
    
    print(f"\n[INFO] Input folder: {input_folder}")
    
    # Ask for expected part number
    expected_part_number = simpledialog.askstring(
        "Part Number Filter",
        "Enter the expected part number to filter OCR results.\n\n"
        "Format: ####-##### (e.g., '0042-68152')\n"
        "Or just prefix: #### (e.g., '0042')\n\n"
        "Press OK without entering anything to use default (0042 prefix)",
        initialvalue=""
    )
    
    if expected_part_number is None:
        print("[INFO] Cancelled. Exiting.")
        return
    
    expected_part_number = expected_part_number.strip()
    if not expected_part_number:
        expected_part_number = "0042"
        print("[INFO] Using default: 0042 prefix")
    else:
        if expected_part_number.startswith('P'):
            expected_part_number = expected_part_number[1:]
        
        if re.fullmatch(r'\d{4}-\d{5}', expected_part_number):
            print(f"[INFO] Filtering for exact part: {expected_part_number}")
        elif re.fullmatch(r'\d{4}', expected_part_number):
            print(f"[INFO] Filtering for prefix: {expected_part_number}-")
        else:
            messagebox.showwarning("Invalid Format", f"Invalid format. Using default 0042 prefix.")
            expected_part_number = "0042"
    
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
    
    # Process images in parallel using all available CPU cores
    max_workers = os.cpu_count() or 4
    print(f"[INFO] Using {max_workers} parallel workers (CPU cores)")
    print("=" * 80)
    
    all_results = []
    completed = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all barcode scanning jobs
        future_to_path = {
            executor.submit(process_image_barcode_only, image_path): image_path
            for image_path in image_files
        }
        
        # Process completed jobs as they finish
        for future in as_completed(future_to_path):
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
    
    pass1_duration = (datetime.now() - pass1_start).total_seconds()
    skipped_count = sum(1 for r in all_results if r.get('skipped'))
    print("=" * 80)
    print(f"[INFO] Pass 1 completed in {pass1_duration:.1f}s ({pass1_duration/60:.1f} minutes)")
    print(f"[INFO] Skipped {skipped_count} plain packaging images")
    print(f"[INFO] Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # PASS 2: Parallel OCR fallback (exclude skipped images)
    images_needing_ocr = [r for r in all_results 
                          if not r.get('skipped') and not has_required_barcodes(r['all_barcodes'])]
    
    if images_needing_ocr:
        print(f"\n[INFO] Pass 2: OCR fallback for {len(images_needing_ocr)} images (PARALLEL PROCESSING)...")
        print(f"[INFO] Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        pass2_start = datetime.now()
        
        # Create a mapping of image_path to result for quick lookup
        result_map = {r['image_path']: r for r in images_needing_ocr}
        
        # Process images in parallel using all available CPU cores
        max_workers = os.cpu_count() or 4
        print(f"[INFO] Using {max_workers} parallel workers (CPU cores)")
        print("=" * 80)
        
        completed = 0
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all OCR jobs
            future_to_path = {
                executor.submit(process_image_ocr, r['image_path'], expected_part_number): r['image_path']
                for r in images_needing_ocr
            }
            
            # Process completed jobs as they finish
            for future in as_completed(future_to_path):
                completed += 1
                image_path = future_to_path[future]
                result = result_map[image_path]
                
                try:
                    ocr_result = future.result()
                    ocr_codes = ocr_result['ocr_codes']
                    
                    print(f"[{completed}/{len(images_needing_ocr)}] OCR: {os.path.basename(image_path)}", end=" ")
                    
                    if ocr_codes:
                        for code in ocr_codes:
                            if code not in result['all_barcodes']:
                                result['all_barcodes'].append(code)
                        result['part_number'], result['serial_number'] = extract_identifiers(result['all_barcodes'])
                        print(f"✓ Found P/S")
                    else:
                        print(f"⚠ Nothing found")
                except Exception as e:
                    print(f"[{completed}/{len(images_needing_ocr)}] OCR: {os.path.basename(image_path)} ✗ Failed")
        
        pass2_duration = (datetime.now() - pass2_start).total_seconds()
        print("=" * 80)
        print(f"[INFO] Pass 2 completed in {pass2_duration:.1f}s ({pass2_duration/60:.1f} minutes)")
        print(f"[INFO] Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print("\n[INFO] Pass 2: Skipped (all images have barcodes)")
    
    # Build serial-to-part mapping
    serial_to_part = {r['serial_number']: r['part_number'] 
                      for r in all_results if r['part_number'] and r['serial_number']}
    
    # PASS 3: Organize and copy
    print("\n[INFO] Pass 3: Organizing and copying images...")
    print(f"[INFO] Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    pass3_start = datetime.now()
    final_results = []
    
    for idx, result in enumerate(all_results, 1):
        copy_start = datetime.now()
        print(f"[{idx}/{len(all_results)}] Organizing: {os.path.basename(result['image_path'])}", end=" ")
        
        part_number = result['part_number']
        serial_number = result['serial_number']
        is_skipped = result.get('skipped', False)
        
        # Match serial to part if needed (only if image wasn't skipped)
        if not is_skipped and serial_number and not part_number and serial_number in serial_to_part:
            part_number = serial_to_part[serial_number]
        
        # Copy file with appropriate folder structure
        target_folder = create_folder_structure(output_folder, part_number, serial_number, is_skipped)
        target_path = os.path.join(target_folder, os.path.basename(result['image_path']))
        
        copy_time = (datetime.now() - copy_start).total_seconds()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        try:
            shutil.copy2(result['image_path'], target_path)
            
            # Show appropriate message based on destination
            if is_skipped:
                print(f"→ Security Seals ({copy_time:.2f}s)")
            elif not part_number:
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
                'processing_time': result.get('processing_time', 0),
                'skipped': is_skipped
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
                'processing_time': result.get('processing_time', 0),
                'skipped': is_skipped
            })
    
    pass3_duration = (datetime.now() - pass3_start).total_seconds()
    print("=" * 80)
    print(f"[INFO] Pass 3 completed in {pass3_duration:.1f}s ({pass3_duration/60:.1f} minutes)")
    print(f"[INFO] Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Generate report
    print("\n" + "=" * 80)
    total_duration = (datetime.now() - script_start_time).total_seconds()
    generate_report(final_results, output_folder, total_duration)
    
    # Calculate total time
    print("\n" + "=" * 80)
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
