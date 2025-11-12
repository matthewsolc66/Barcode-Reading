"""
Interactive OCR Region Tester
Click and drag to select a rectangle on the image, then press ENTER to OCR that region.
Press 'r' to reset selection, 'q' to quit.
"""

import cv2
import numpy as np
from PIL import Image
import pytesseract
from tkinter import Tk, filedialog

# Global variables for rectangle selection
drawing = False
ix, iy = -1, -1
fx, fy = -1, -1
image_copy = None


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
                        ('Default', '--psm 6'),
                        ('Single line', '--psm 7'),
                        ('Single word', '--psm 8'),
                        ('Digits only', '--psm 6 -c tessedit_char_whitelist=0123456789'),
                        ('Digits and dash', '--psm 6 -c tessedit_char_whitelist=0123456789-'),
                    ]
                    
                    for config_name, config_str in configs:
                        try:
                            text = pytesseract.image_to_string(pil_image, config=config_str).strip()
                            print(f"\n{config_name}:")
                            if text:
                                print(f"  '{text}'")
                                print(f"  Raw: {repr(text)}")
                            else:
                                print("  (no text detected)")
                        except Exception as e:
                            print(f"  Error: {e}")
                    
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
