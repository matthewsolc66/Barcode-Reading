"""
Barcode Scanner and Image Sorter

Scans images for barcodes, extracts part numbers and serial numbers,
and organizes images into folders based on those identifiers.
"""

import os
import re
import shutil
import warnings
from pathlib import Path
from tkinter import Tk, filedialog
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
    print("[WARN] pytesseract not available. Install with: pip install pytesseract")
    print("[WARN] Also install Tesseract-OCR from: https://github.com/UB-Mannheim/tesseract/wiki")

# Suppress pyzbar warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'


def get_exif_orientation(pil_image):
    """Return EXIF orientation tag (1 if missing)."""
    try:
        exif = pil_image.getexif()
        return exif.get(274, 1)  # 274 = Orientation tag
    except Exception:
        return 1


def apply_exif_orientation(pil_image):
    """Apply EXIF orientation to correctly orient the image."""
    try:
        orientation = get_exif_orientation(pil_image)
        if orientation != 1:
            pil_image = ImageOps.exif_transpose(pil_image)
        return pil_image
    except Exception as e:
        print(f"[WARN] Failed to apply EXIF orientation: {e}")
        return pil_image


def enhance_for_barcode_reading(image_array):
    """Apply image enhancements to improve barcode detection."""
    # Convert to grayscale if needed
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_array.copy()
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    return enhanced


def has_required_barcodes(barcodes_found):
    """Check if we have both part number and serial number."""
    has_part = any(re.fullmatch(r'^P\d{4}-\d{5}$', b) for b in barcodes_found)
    has_serial = any(re.fullmatch(r'^S\d{18}$', b) for b in barcodes_found)
    return has_part and has_serial


def decode_barcodes(pil_image):
    """
    Decode all Code 128 barcodes from an image using multiple techniques.
    Returns list of decoded barcode data strings.
    """
    barcodes_found = []
    
    # Convert PIL to numpy array
    img_array = np.array(pil_image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
    
    # Try 1: Direct scan on color image (Code 128 only)
    results = decode(pil_image, symbols=[ZBarSymbol.CODE128])
    for barcode in results:
        data = barcode.data.decode('utf-8')
        if data not in barcodes_found:
            barcodes_found.append(data)
    
    if has_required_barcodes(barcodes_found):
        return barcodes_found
    
    # Try 2: Remove glare/reflections before processing
    # Detect bright spots and inpaint them
    _, glare_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3,3), np.uint8)
    glare_mask = cv2.dilate(glare_mask, kernel, iterations=1)
    deglared = cv2.inpaint(gray, glare_mask, 3, cv2.INPAINT_TELEA)
    deglared_pil = Image.fromarray(deglared)
    results = decode(deglared_pil, symbols=[ZBarSymbol.CODE128])
    for barcode in results:
        data = barcode.data.decode('utf-8')
        if data not in barcodes_found:
            barcodes_found.append(data)
    
    if has_required_barcodes(barcodes_found):
        return barcodes_found
    
    # Try 3: Enhanced grayscale with CLAHE
    enhanced = enhance_for_barcode_reading(img_array)
    enhanced_pil = Image.fromarray(enhanced)
    results = decode(enhanced_pil, symbols=[ZBarSymbol.CODE128])
    for barcode in results:
        data = barcode.data.decode('utf-8')
        if data not in barcodes_found:
            barcodes_found.append(data)
    
    if has_required_barcodes(barcodes_found):
        return barcodes_found
    
    # Try 4: Multiple scales (resize image to look for small barcodes)
    for scale in [1.5, 2.0, 2.5]:
        new_w = int(pil_image.width * scale)
        new_h = int(pil_image.height * scale)
        scaled = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        results = decode(scaled, symbols=[ZBarSymbol.CODE128])
        for barcode in results:
            data = barcode.data.decode('utf-8')
            if data not in barcodes_found:
                barcodes_found.append(data)
        
        if has_required_barcodes(barcodes_found):
            return barcodes_found
        
        # Also try enhanced version at this scale
        scaled_array = np.array(scaled)
        scaled_enhanced = enhance_for_barcode_reading(scaled_array)
        scaled_enhanced_pil = Image.fromarray(scaled_enhanced)
        results = decode(scaled_enhanced_pil, symbols=[ZBarSymbol.CODE128])
        for barcode in results:
            data = barcode.data.decode('utf-8')
            if data not in barcodes_found:
                barcodes_found.append(data)
        
        if has_required_barcodes(barcodes_found):
            return barcodes_found
    
    # Try 5: Binary thresholding (Otsu's method)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_pil = Image.fromarray(binary)
    results = decode(binary_pil, symbols=[ZBarSymbol.CODE128])
    for barcode in results:
        data = barcode.data.decode('utf-8')
        if data not in barcodes_found:
            barcodes_found.append(data)
    
    if has_required_barcodes(barcodes_found):
        return barcodes_found
    
    # Try 6: Adaptive thresholding with multiple block sizes
    for block_size in [11, 21, 51]:
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, block_size, 10)
        adaptive_pil = Image.fromarray(adaptive)
        results = decode(adaptive_pil, symbols=[ZBarSymbol.CODE128])
        for barcode in results:
            data = barcode.data.decode('utf-8')
            if data not in barcodes_found:
                barcodes_found.append(data)
        
        if has_required_barcodes(barcodes_found):
            return barcodes_found
    
    # Try 7: Inverted binary (white text on black background)
    inverted = cv2.bitwise_not(binary)
    inverted_pil = Image.fromarray(inverted)
    results = decode(inverted_pil, symbols=[ZBarSymbol.CODE128])
    for barcode in results:
        data = barcode.data.decode('utf-8')
        if data not in barcodes_found:
            barcodes_found.append(data)
    
    if has_required_barcodes(barcodes_found):
        return barcodes_found
    
    # Try 8: Morphological operations to clean up noise
    kernel = np.ones((2,2), np.uint8)
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    morph_pil = Image.fromarray(morph)
    results = decode(morph_pil, symbols=[ZBarSymbol.CODE128])
    for barcode in results:
        data = barcode.data.decode('utf-8')
        if data not in barcodes_found:
            barcodes_found.append(data)
    
    if has_required_barcodes(barcodes_found):
        return barcodes_found
    
    # Try 9: Multiple rotations (only if still nothing found)
    if len(barcodes_found) == 0:
        for angle in [90, 180, 270]:
            rotated = pil_image.rotate(angle, expand=True)
            results = decode(rotated, symbols=[ZBarSymbol.CODE128])
            for barcode in results:
                data = barcode.data.decode('utf-8')
                if data not in barcodes_found:
                    barcodes_found.append(data)
            
            if has_required_barcodes(barcodes_found):
                return barcodes_found
            
            # Try rotated + scaled
            for scale in [1.5, 2.0]:
                new_w = int(rotated.width * scale)
                new_h = int(rotated.height * scale)
                scaled_rot = rotated.resize((new_w, new_h), Image.Resampling.LANCZOS)
                results = decode(scaled_rot, symbols=[ZBarSymbol.CODE128])
                for barcode in results:
                    data = barcode.data.decode('utf-8')
                    if data not in barcodes_found:
                        barcodes_found.append(data)
                
                if has_required_barcodes(barcodes_found):
                    return barcodes_found
    
    # Try 10: Brightness/contrast adjustments (only if still nothing found)
    if len(barcodes_found) == 0:
        from PIL import ImageEnhance
        
        # Higher contrast
        enhancer = ImageEnhance.Contrast(pil_image)
        for factor in [1.5, 2.0, 2.5, 3.0]:
            contrast_img = enhancer.enhance(factor)
            results = decode(contrast_img, symbols=[ZBarSymbol.CODE128])
            for barcode in results:
                data = barcode.data.decode('utf-8')
                if data not in barcodes_found:
                    barcodes_found.append(data)
            
            if has_required_barcodes(barcodes_found):
                return barcodes_found
        
        # Lower contrast (sometimes helps with overexposed images)
        contrast_img = enhancer.enhance(0.5)
        results = decode(contrast_img, symbols=[ZBarSymbol.CODE128])
        for barcode in results:
            data = barcode.data.decode('utf-8')
            if data not in barcodes_found:
                barcodes_found.append(data)
        
        if has_required_barcodes(barcodes_found):
            return barcodes_found
        
        # Brightness adjustments
        enhancer = ImageEnhance.Brightness(pil_image)
        for factor in [0.7, 0.85, 1.3, 1.5]:
            bright_img = enhancer.enhance(factor)
            results = decode(bright_img, symbols=[ZBarSymbol.CODE128])
            for barcode in results:
                data = barcode.data.decode('utf-8')
                if data not in barcodes_found:
                    barcodes_found.append(data)
            
            if has_required_barcodes(barcodes_found):
                return barcodes_found
        
        # Increase sharpness
        enhancer = ImageEnhance.Sharpness(pil_image)
        for factor in [2.0, 3.0]:
            sharp_img = enhancer.enhance(factor)
            results = decode(sharp_img, symbols=[ZBarSymbol.CODE128])
            for barcode in results:
                data = barcode.data.decode('utf-8')
                if data not in barcodes_found:
                    barcodes_found.append(data)
            
            if has_required_barcodes(barcodes_found):
                return barcodes_found
    
    return barcodes_found


def extract_text_with_ocr(pil_image):
    """
    Use OCR to extract text from image and look for part/serial numbers.
    Returns list of found identifiers.
    """
    if not TESSERACT_AVAILABLE:
        return []
    
    found_codes = []
    
    try:
        # Convert to grayscale and enhance
        img_array = np.array(pil_image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
        
        # Apply CLAHE for better text recognition
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Try multiple preprocessing methods
        preprocessed_images = []
        
        # Original enhanced
        preprocessed_images.append(enhanced)
        
        # Binary threshold
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed_images.append(binary)
        
        # Adaptive threshold
        adaptive = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 21, 10)
        preprocessed_images.append(adaptive)
        
        # Upscale for better OCR (2x)
        h, w = enhanced.shape
        scaled = cv2.resize(enhanced, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
        preprocessed_images.append(scaled)
        
        # Run OCR on each preprocessed version
        for img in preprocessed_images:
            pil_img = Image.fromarray(img)
            
            # Use multiple PSM modes (Page Segmentation Modes)
            # PSM 6 = Assume a single uniform block of text
            # PSM 11 = Sparse text. Find as much text as possible in no particular order
            for psm in [6, 11]:
                try:
                    text = pytesseract.image_to_string(
                        pil_img, 
                        config=f'--psm {psm} -c tessedit_char_whitelist=0123456789-'
                    )
                    
                    # Look for part numbers: ####-##### (without P prefix in text)
                    part_matches = re.findall(r'\d{4}-?\d{5}', text)
                    for match in part_matches:
                        # Normalize (ensure hyphen is present and add P prefix)
                        if '-' in match:
                            normalized = f"P{match}"
                        else:
                            normalized = f"P{match[:4]}-{match[4:]}"
                        if normalized not in found_codes and re.fullmatch(r'^P\d{4}-\d{5}$', normalized):
                            found_codes.append(normalized)
                    
                    # Look for serial numbers: ################## (without S prefix in text)
                    serial_matches = re.findall(r'\d{18}', text)
                    for match in serial_matches:
                        # Add S prefix
                        normalized = f"S{match}"
                        if normalized not in found_codes:
                            found_codes.append(normalized)
                    
                except Exception as e:
                    continue
        
    except Exception as e:
        print(f"[WARN] OCR processing failed: {e}")
    
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
    else:
        return "Other"


def extract_identifiers(barcodes):
    """
    Extract part number and serial number from list of barcodes.
    Returns (part_number, serial_number) tuple.
    """
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
    """
    Create folder structure for organizing images.
    Structure: base_folder / part_number_without_P / last_8_digits_of_serial
    Returns the target folder path.
    """
    if not part_number and not serial_number:
        # No identifiers found - use "Unidentified" folder
        target_folder = os.path.join(base_folder, "_Unidentified")
    elif part_number and serial_number:
        # Both found - organize by part then serial
        # Remove 'P' prefix from part number
        part_folder = part_number[1:] if part_number.startswith('P') else part_number
        # Extract last 8 digits from serial number (date code + part identifier)
        serial_folder = serial_number[-8:] if len(serial_number) >= 8 else serial_number
        target_folder = os.path.join(base_folder, part_folder, serial_folder)
    elif part_number:
        # Only part number found
        part_folder = part_number[1:] if part_number.startswith('P') else part_number
        target_folder = os.path.join(base_folder, part_folder, "_No_Serial")
    else:
        # Only serial number found (unusual)
        serial_folder = serial_number[-8:] if len(serial_number) >= 8 else serial_number
        target_folder = os.path.join(base_folder, "_No_Part", serial_folder)
    
    os.makedirs(target_folder, exist_ok=True)
    return target_folder


def process_image(image_path, output_base_folder):
    """
    Process a single image: scan for barcodes, extract identifiers,
    and copy to organized folder structure.
    """
    print(f"\n[INFO] Processing: {os.path.basename(image_path)}")
    
    try:
        # Load and orient image
        pil_image = Image.open(image_path)
        pil_image = apply_exif_orientation(pil_image)
        
        # Decode all barcodes
        barcodes = decode_barcodes(pil_image)
        
        # If no barcodes found or missing required ones, try OCR fallback
        if not has_required_barcodes(barcodes):
            print(f"  Trying OCR fallback...")
            ocr_codes = extract_text_with_ocr(pil_image)
            if ocr_codes:
                print(f"  OCR found: {ocr_codes}")
                # Merge with existing barcodes
                for code in ocr_codes:
                    if code not in barcodes:
                        barcodes.append(code)
        
        if barcodes:
            print(f"  Found {len(barcodes)} barcode(s): {barcodes}")
        else:
            print(f"  No barcodes found")
        
        # Extract identifiers
        part_number, serial_number = extract_identifiers(barcodes)
        
        print(f"  Part Number: {part_number if part_number else 'Not found'}")
        print(f"  Serial Number: {serial_number if serial_number else 'Not found'}")
        
        # Create folder structure and copy image
        target_folder = create_folder_structure(output_base_folder, part_number, serial_number)
        target_path = os.path.join(target_folder, os.path.basename(image_path))
        
        # Copy image to target location
        shutil.copy2(image_path, target_path)
        print(f"  Copied to: {target_path}")
        
        return {
            'filename': os.path.basename(image_path),
            'part_number': part_number,
            'serial_number': serial_number,
            'all_barcodes': barcodes,
            'target_folder': target_folder,
            'success': True
        }
        
    except Exception as e:
        print(f"  [ERROR] Failed to process image: {e}")
        return {
            'filename': os.path.basename(image_path),
            'part_number': None,
            'serial_number': None,
            'all_barcodes': [],
            'target_folder': None,
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
        
        # Summary statistics
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
        
        # Detailed results
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
    """Main function to run the barcode sorter."""
    print("=" * 80)
    print("BARCODE IMAGE SORTER")
    print("=" * 80)
    print("\nThis tool will:")
    print("  1. Scan all images in a folder for barcodes")
    print("  2. Extract part numbers (P####-#####) and serial numbers (S##################)")
    print("  3. Organize images into folders: PartNumber/SerialNumber/")
    print()
    
    # Select input folder
    root = Tk()
    root.withdraw()
    input_folder = filedialog.askdirectory(title="Select Folder Containing Images to Sort")
    
    if not input_folder:
        print("[INFO] No folder selected. Exiting.")
        return
    
    print(f"\n[INFO] Input folder: {input_folder}")
    
    # Create output folder automatically in the input directory
    output_folder = os.path.join(input_folder, "Sorted_Images")
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"[INFO] Output folder: {output_folder}")
    
    # Find all image files
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    image_files = []
    for file in os.listdir(input_folder):
        if file.lower().endswith(image_extensions):
            image_files.append(os.path.join(input_folder, file))
    
    if not image_files:
        print(f"\n[ERROR] No image files found in {input_folder}")
        return
    
    print(f"\n[INFO] Found {len(image_files)} image(s) to process")
    print("=" * 80)
    
    # Process all images
    results = []
    for image_path in image_files:
        result = process_image(image_path, output_folder)
        results.append(result)
    
    # Generate report
    print("\n" + "=" * 80)
    generate_report(results, output_folder)
    
    print("\n[INFO] Processing complete!")
    print(f"[INFO] Images organized in: {output_folder}")


if __name__ == "__main__":
    main()
