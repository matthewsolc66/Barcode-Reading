"""
Barcode Scanner and Image Sorter - Clean Version

Scans images for barcodes, extracts part numbers and serial numbers,
and organizes images into folders based on those identifiers.
"""

import os
import re
import shutil
import warnings
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


def has_required_barcodes(barcodes_found):
    """Check if we have both part number and serial number."""
    has_part = any(re.fullmatch(r'^P\d{4}-\d{5}$', b) for b in barcodes_found)
    has_serial = any(re.fullmatch(r'^S\d{18}$', b) for b in barcodes_found)
    return has_part and has_serial


def decode_barcodes(pil_image):
    """Decode Code 128 barcodes using multiple preprocessing techniques."""
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
    
    # Try 8: Brightness/contrast adjustments (only if nothing found)
    if len(barcodes_found) == 0:
        from PIL import ImageEnhance
        for factor in [1.5, 2.0, 2.5, 3.0, 0.5, 0.7, 0.85, 1.3]:
            enhanced_img = ImageEnhance.Contrast(pil_image).enhance(factor)
            for barcode in decode(enhanced_img, symbols=[ZBarSymbol.CODE128]):
                data = barcode.data.decode('utf-8')
                if data not in barcodes_found:
                    barcodes_found.append(data)
            if has_required_barcodes(barcodes_found):
                return barcodes_found
    
    return barcodes_found


def extract_text_with_ocr(pil_image, expected_part_number=None):
    """Use OCR to extract part/serial numbers from text."""
    if not TESSERACT_AVAILABLE:
        return []
    
    found_codes = []
    img_array = np.array(pil_image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
    
    # Apply CLAHE enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Try multiple preprocessing methods
    preprocessed = [
        enhanced,
        cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10),
        cv2.resize(enhanced, (enhanced.shape[1]*2, enhanced.shape[0]*2), interpolation=cv2.INTER_CUBIC)
    ]
    
    # Determine matching mode
    match_exact = expected_part_number and '-' in expected_part_number
    match_prefix = expected_part_number and '-' not in expected_part_number
    
    # Run OCR on each preprocessed version
    for img in preprocessed:
        for psm in [6, 11]:
            try:
                text = pytesseract.image_to_string(Image.fromarray(img), 
                    config=f'--psm {psm} -c tessedit_char_whitelist=0123456789-')
                
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
                
                # Extract serial numbers
                for match in re.findall(r'\d{18}', text):
                    normalized = f"S{match}"
                    if normalized not in found_codes:
                        found_codes.append(normalized)
            except Exception:
                continue
    
    return found_codes


def classify_barcode(data):
    """Classify barcode based on pattern."""
    if re.fullmatch(r'^P\d{4}-\d{5}$', data):
        return "Part Number"
    elif re.fullmatch(r'^S\d{18}$', data):
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
    """Create folder structure: base_folder/part_number_without_P/last_8_digits_of_serial"""
    if not part_number and not serial_number:
        target_folder = os.path.join(base_folder, "_Unidentified")
    elif part_number and serial_number:
        part_folder = part_number[1:] if part_number.startswith('P') else part_number
        serial_folder = serial_number[-8:] if len(serial_number) >= 8 else serial_number
        target_folder = os.path.join(base_folder, part_folder, serial_folder)
    elif part_number:
        part_folder = part_number[1:] if part_number.startswith('P') else part_number
        target_folder = os.path.join(base_folder, part_folder, "_No_Serial")
    else:
        serial_folder = serial_number[-8:] if len(serial_number) >= 8 else serial_number
        target_folder = os.path.join(base_folder, "_No_Part", serial_folder)
    
    os.makedirs(target_folder, exist_ok=True)
    return target_folder


def process_image_barcode_only(image_path):
    """Quick barcode scan only - no OCR fallback."""
    try:
        pil_image = Image.open(image_path)
        pil_image = apply_exif_orientation(pil_image)
        barcodes = decode_barcodes(pil_image)
        part_number, serial_number = extract_identifiers(barcodes)
        
        return {
            'image_path': image_path,
            'part_number': part_number,
            'serial_number': serial_number,
            'all_barcodes': barcodes,
            'success': True
        }
    except Exception as e:
        return {
            'image_path': image_path,
            'part_number': None,
            'serial_number': None,
            'all_barcodes': [],
            'success': False,
            'error': str(e)
        }


def generate_report(results, output_folder):
    """Generate a text report of all processed images."""
    report_path = os.path.join(output_folder, "_sorting_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("BARCODE SORTING REPORT\n")
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
        f.write("\n" + "=" * 80 + "\n\n")
        
        f.write("DETAILED RESULTS:\n\n")
        for result in results:
            f.write(f"Filename: {result['filename']}\n")
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
    print("=" * 80)
    print("BARCODE IMAGE SORTER")
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
    
    # PASS 1: Quick barcode scanning
    print("\n[INFO] Pass 1: Quick barcode scanning...")
    print("=" * 80)
    all_results = []
    for idx, image_path in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] Scanning: {os.path.basename(image_path)}", end=" ")
        result = process_image_barcode_only(image_path)
        all_results.append(result)
        
        if result['success']:
            if result['part_number'] and result['serial_number']:
                print(f"✓ Part & Serial")
            elif result['part_number']:
                print(f"✓ Part only")
            elif result['serial_number']:
                print(f"✓ Serial only")
            elif result['all_barcodes']:
                print(f"⚠ {len(result['all_barcodes'])} barcode(s)")
            else:
                print(f"⚠ No barcodes")
        else:
            print(f"✗ Error")
    
    print("=" * 80)
    
    # PASS 2: OCR fallback
    images_needing_ocr = [r for r in all_results if not has_required_barcodes(r['all_barcodes'])]
    
    if images_needing_ocr:
        print(f"\n[INFO] Pass 2: OCR fallback for {len(images_needing_ocr)} images...")
        print("=" * 80)
        
        for idx, result in enumerate(images_needing_ocr, 1):
            print(f"[{idx}/{len(images_needing_ocr)}] OCR: {os.path.basename(result['image_path'])}", end=" ")
            
            try:
                pil_image = Image.open(result['image_path'])
                pil_image = apply_exif_orientation(pil_image)
                ocr_codes = extract_text_with_ocr(pil_image, expected_part_number)
                
                if ocr_codes:
                    for code in ocr_codes:
                        if code not in result['all_barcodes']:
                            result['all_barcodes'].append(code)
                    result['part_number'], result['serial_number'] = extract_identifiers(result['all_barcodes'])
                    print(f"✓ Found P/S")
                else:
                    print(f"⚠ Nothing found")
            except Exception:
                print(f"✗ Failed")
        
        print("=" * 80)
    else:
        print("\n[INFO] Pass 2: Skipped (all images have barcodes)")
    
    # Build serial-to-part mapping
    serial_to_part = {r['serial_number']: r['part_number'] 
                      for r in all_results if r['part_number'] and r['serial_number']}
    
    # PASS 3: Organize and copy
    print("\n[INFO] Pass 3: Organizing and copying images...")
    print("=" * 80)
    final_results = []
    
    for idx, result in enumerate(all_results, 1):
        print(f"[{idx}/{len(all_results)}] Organizing: {os.path.basename(result['image_path'])}", end=" ")
        
        part_number = result['part_number']
        serial_number = result['serial_number']
        
        # Match serial to part if needed
        if serial_number and not part_number and serial_number in serial_to_part:
            part_number = serial_to_part[serial_number]
        
        # Copy file
        target_folder = create_folder_structure(output_folder, part_number, serial_number)
        target_path = os.path.join(target_folder, os.path.basename(result['image_path']))
        
        try:
            shutil.copy2(result['image_path'], target_path)
            print(f"✓")
            
            final_results.append({
                'filename': os.path.basename(result['image_path']),
                'part_number': part_number,
                'serial_number': serial_number,
                'all_barcodes': result['all_barcodes'],
                'target_folder': target_folder,
                'success': True
            })
        except Exception as e:
            print(f"✗")
            final_results.append({
                'filename': os.path.basename(result['image_path']),
                'part_number': part_number,
                'serial_number': serial_number,
                'all_barcodes': result['all_barcodes'],
                'target_folder': None,
                'success': False,
                'error': str(e)
            })
    
    # Generate report
    print("\n" + "=" * 80)
    generate_report(final_results, output_folder)
    
    print("\n[INFO] Processing complete!")
    print(f"[INFO] Images organized in: {output_folder}")
    
    # Open output folder
    try:
        os.startfile(output_folder)
    except Exception:
        pass
    
    # Show completion dialog
    messagebox.showinfo(
        "Processing Complete",
        f"Successfully processed {len(final_results)} images!\n\n"
        f"Results saved to:\n{output_folder}\n\n"
        f"Check '_sorting_report.txt' for details."
    )


if __name__ == "__main__":
    main()
