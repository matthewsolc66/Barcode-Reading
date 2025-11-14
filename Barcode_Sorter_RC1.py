"""
Barcode Scanner and Image Sorter - RC1 (Release Candidate 1)

Purpose:
- Production-ready version with all OCR improvements and interactive features.
- Streamlined codebase with only necessary features.
- Clear comments explaining each feature.

Key features kept:
- GUI selection of part numbers with scrollbar + workers slider (uses physical cores)
- Quick actions: Small Batch Test and OCR Region Test
- Pass 1: Parallel barcode scan with early exit and zoomed retries
- Pass 2: Parallel OCR fallback (CLAHE + Otsu) with PSM6 then PSM7
- OCR helpers: strip spaces, fuzzy part correction (against expected list),
  serial fuzzy correction around WWYY with dynamic year range
- Graceful Ctrl+C: stop after current work, sort what we have, exit cleanly
- Pass 3: Organize copies by P/Serial; generate concise report

Dependencies: Pillow, pyzbar, OpenCV, numpy, pytesseract, psutil, tkinter
"""

import os
import io
import re
import shutil
import warnings
import signal
import sys
import subprocess
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from tkinter import (
    Tk, filedialog, messagebox, Toplevel, Checkbutton, IntVar, Button,
    Label, Frame, Scale, HORIZONTAL, Canvas, Scrollbar, VERTICAL
)
from PIL import Image, ImageOps
from pyzbar.pyzbar import decode, ZBarSymbol
import cv2
import numpy as np
import psutil

# Optional OCR (gracefully disabled if not available)
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except Exception:
    TESSERACT_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'


# -------------------------- Debug toggles ---------------------------
# Set to True to save ROI overlays and zoom crops for Pass 1 verification
DEBUG_ZOOM_ROI = True
DEBUG_ZOOM_DIR = None  # Will be set to output_dir/_zoom_debug when output_dir is known


# ------------------------------ UI ---------------------------------

def select_part_numbers_dialog(part_numbers):
    """Modal dialog to select part numbers and workers.
    Returns (selected_parts, workers, mode) where mode ∈ {None, 'small_batch', 'ocr_region'}.
    """
    if not part_numbers:
        return None, None, None

    # Use physical cores for default maximum workers
    try:
        max_workers = psutil.cpu_count(logical=False) or (os.cpu_count() or 4)
    except Exception:
        max_workers = os.cpu_count() or 4

    selected_parts = []
    mode = None
    # Default to 1/2 of max workers, at least 1
    selected_workers = max(1, int(max_workers * 0.5))

    dialog = Toplevel()
    dialog.title("Select Parts and Options")
    dialog.geometry("450x600")
    dialog.resizable(False, False)

    # Center window
    dialog.update_idletasks()
    x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
    y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
    dialog.geometry(f"{dialog.winfo_width()}x{dialog.winfo_height()}+{x}+{y}")

    # Quick action buttons at the top right
    button_row = Frame(dialog)
    button_row.pack(fill="x", pady=(10, 5), padx=20)

    def on_small_batch():
        nonlocal mode
        mode = "small_batch"
        dialog.destroy()

    def on_ocr_region():
        nonlocal mode
        mode = "ocr_region"
        dialog.destroy()

    Button(button_row, text="Small Batch Test", command=on_small_batch, bg="#2196F3", fg="white", width=14).pack(side="right", padx=3)
    Button(button_row, text="OCR Region Test", command=on_ocr_region, bg="#9C27B0", fg="white", width=14).pack(side="right", padx=3)

    # Title below the buttons
    title_frame = Frame(dialog)
    title_frame.pack(fill="x", pady=(5, 10), padx=20)
    Label(title_frame, text="Select which part numbers to process:", font=("Arial", 11, "bold")).pack(anchor="w")

    # Checkbox list with scrollbar
    container = Frame(dialog, height=250)
    container.pack(pady=10, padx=20, fill="both")
    container.pack_propagate(False)

    canvas = Canvas(container, highlightthickness=0)
    vscroll = Scrollbar(container, orient=VERTICAL, command=canvas.yview)
    box = Frame(canvas)
    canvas.configure(yscrollcommand=vscroll.set)
    vscroll.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    w = canvas.create_window((0, 0), window=box, anchor="nw")

    checkbox_vars = []
    for part in part_numbers:
        var = IntVar(value=1)
        Checkbutton(box, text=f"P{part}", variable=var, anchor="w", font=("Arial", 10)).pack(fill="x", pady=2)
        checkbox_vars.append((part, var))

    def cfg_scroll(_evt=None):
        canvas.configure(scrollregion=canvas.bbox("all"))
        canvas.itemconfig(w, width=canvas.winfo_width())

    box.bind("<Configure>", cfg_scroll)
    canvas.bind("<Configure>", cfg_scroll)

    # Select/Deselect buttons directly below checkbox list
    def sel_all():
        for _, v in checkbox_vars: v.set(1)
    def desel_all():
        for _, v in checkbox_vars: v.set(0)
    
    select_btns = Frame(dialog)
    select_btns.pack(pady=10)
    Button(select_btns, text="Select All", width=12, command=sel_all).pack(side="left", padx=5)
    Button(select_btns, text="Deselect All", width=12, command=desel_all).pack(side="left", padx=5)

    # Workers section
    workers_frame = Frame(dialog)
    workers_frame.pack(pady=10)
    Label(workers_frame, text="Parallel Workers:", font=("Arial", 10, "bold")).pack()
    Label(workers_frame, text=f"(Physical CPU cores: {max_workers})", font=("Arial", 8), fg="gray").pack()
    workers_label = Label(workers_frame, text=f"{selected_workers}", font=("Arial", 12))
    workers_label.pack(pady=4)
    def _upd(val): workers_label.config(text=f"{int(float(val))}")
    slider = Scale(workers_frame, from_=1, to=max_workers, orient=HORIZONTAL, length=350, 
                   showvalue=False, command=_upd, tickinterval=1, resolution=1)
    slider.set(selected_workers)
    _upd(selected_workers)
    
    # Make clicking/dragging on the trough snap to that position (and prevent auto-repeat to min/max)
    def _set_from_x(x):
        length = max(1, slider.winfo_width())
        raw = (x / length) * (max_workers - 1) + 1  # Map to [1, max_workers]
        value = int(round(raw))
        value = 1 if value < 1 else (max_workers if value > max_workers else value)
        slider.set(value)
        _upd(value)

    def on_trough_click(event):
        _set_from_x(event.x)
        return "break"  # stop default auto-repeat behavior

    def on_trough_drag(event):
        _set_from_x(event.x)
        return "break"

    slider.bind("<Button-1>", on_trough_click)
    slider.bind("<B1-Motion>", on_trough_drag)
    slider.pack()

    # OK/Cancel buttons centered at the bottom
    def on_ok():
        nonlocal selected_parts, selected_workers
        selected_parts = [p for p, v in checkbox_vars if v.get() == 1]
        selected_workers = slider.get()
        dialog.destroy()

    def on_cancel():
        nonlocal selected_parts
        selected_parts = None
        dialog.destroy()

    btns = Frame(dialog)
    btns.pack(pady=15)
    Button(btns, text="OK", width=15, bg="#4CAF50", fg="white", command=on_ok).pack(side="left", padx=5)
    Button(btns, text="Cancel", width=15, bg="#f44336", fg="white", command=on_cancel).pack(side="left", padx=5)

    dialog.transient()
    dialog.grab_set()
    dialog.wait_window()

    return selected_parts, selected_workers, mode


def load_expected_part_numbers(config_file='part_numbers_config.txt'):
    """Read list of expected P####-##### from a simple text file (one per line)."""
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
    if parts:
        print(f"[INFO] Loaded {len(parts)} expected part numbers from config")
    else:
        print("[INFO] No expected part numbers configured - accepting all part numbers")
        parts = None
    return parts


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


def is_valid_serial_number(serial: str) -> bool:
    """Validate serial format S900########### with valid WWYY window."""
    if not re.fullmatch(r'^S900\d{15}$', serial):
        return False
    date_code = serial[11:15]
    week = int(date_code[:2])
    year = int(date_code[2:])
    if not (1 <= week <= 52):
        return False
    cy = datetime.now().year % 100
    return 24 <= year <= (cy + 1)


def has_required_barcodes(barcodes_found):
    """Check if we have both part number and valid serial number."""
    has_part = any(re.fullmatch(r'^P\d{4}-\d{5}$', b) for b in barcodes_found)
    has_serial = any(is_valid_serial_number(b) for b in barcodes_found)
    return has_part and has_serial


def decode_barcodes(pil_image, debug_name: str | None = None):
    """Multi-try Code 128 decode with early exits + limited zoom retries."""
    if DEBUG_ZOOM_ROI and debug_name:
        print(f"[DEBUG] Processing {debug_name} with debug enabled")
    
    found = []
    union_images: list[np.ndarray] = []  # in-memory zoom-union crops (grayscale, 8-bit)
    best_rotation_img: np.ndarray | None = None  # Best rotated image based on barcode count
    regions = []
    barcodes = []  # list of tuples (data, rect, polygon_points)
    arr = np.array(pil_image)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if len(arr.shape) == 3 else arr

    def _collect(objs, M_inv=None, record_regions=True):
        from types import SimpleNamespace
        H, W = arr.shape[0], arr.shape[1]

        def _aff_apply(M, px, py):
            if M is None:
                return float(px), float(py)
            x0 = float(M[0,0]) * px + float(M[0,1]) * py + float(M[0,2])
            y0 = float(M[1,0]) * px + float(M[1,1]) * py + float(M[1,2])
            return x0, y0

        for bc in objs:
            data = bc.data.decode('utf-8')
            # Prefer polygon bounds if available for a fuller region; fall back to rect
            x = getattr(bc, 'rect', None).left if hasattr(bc, 'rect') else 0
            y = getattr(bc, 'rect', None).top if hasattr(bc, 'rect') else 0
            w = getattr(bc, 'rect', None).width if hasattr(bc, 'rect') else 0
            h = getattr(bc, 'rect', None).height if hasattr(bc, 'rect') else 0
            poly = getattr(bc, 'polygon', None)
            poly_pts = None
            if poly and len(poly) >= 3:
                try:
                    # Transform polygon back to original image if needed
                    tpts = []
                    for p in poly:
                        tx, ty = _aff_apply(M_inv, float(p.x), float(p.y))
                        tpts.append((tx, ty))
                    poly_pts = [(int(round(px)), int(round(py))) for px, py in tpts]
                    xs = [px for px, _ in tpts]
                    ys = [py for _, py in tpts]
                    x = max(0, int(round(min(xs))))
                    y = max(0, int(round(min(ys))))
                    w = int(round(min(W - x, max(xs) - min(xs))))
                    h = int(round(min(H - y, max(ys) - min(ys))))
                except Exception:
                    pass
            # Inflate region a bit to better cover the whole barcode
            if w <= 0 or h <= 0:
                # Fallback: transform rect corners if polygon not available
                cx1, cy1 = _aff_apply(M_inv, x, y)
                cx2, cy2 = _aff_apply(M_inv, x + w, y)
                cx3, cy3 = _aff_apply(M_inv, x + w, y + h)
                cx4, cy4 = _aff_apply(M_inv, x, y + h)
                xs = [cx1, cx2, cx3, cx4]
                ys = [cy1, cy2, cy3, cy4]
                x = max(0, int(round(min(xs))))
                y = max(0, int(round(min(ys))))
                w = int(round(min(W - x, max(xs) - min(xs))))
                h = int(round(min(H - y, max(ys) - min(ys))))

            infx = int(max(6, w * 0.2))
            infy = int(max(6, h * 0.35))
            x = max(0, x - infx)
            y = max(0, y - infy)
            w = min(W - x, w + 2 * infx)
            h = min(H - y, h + 2 * infy)
            rect = SimpleNamespace(left=int(x), top=int(y), width=int(w), height=int(h))
            if data not in found:
                found.append(data)
            if record_regions:
                # Avoid duplicates by simple overlap check
                dup = False
                for r in regions:
                    if abs(r.left - rect.left) < 3 and abs(r.top - rect.top) < 3 and abs(r.width - rect.width) < 5 and abs(r.height - rect.height) < 5:
                        dup = True; break
                if not dup:
                    regions.append(rect)
                    barcodes.append((data, rect, poly_pts))
    
    def _save_debug_overlay():
        """Save debug overlay showing detected barcode regions.
        - Thin blue lines: Raw ZBar polygon detections (exact decode regions)
        - Thick green boxes: Inflated regions we actually use for zoom crops
        - Yellow union box: Combined region per unique barcode value
        Only saved when required P/S not yet found.
        """
        if DEBUG_ZOOM_ROI and DEBUG_ZOOM_DIR and debug_name and regions and not has_required_barcodes(found):
            print(f"[DEBUG] {debug_name}: Found {len(regions)} barcode region(s), saving overlay...")
            try:
                overlay = cv2.cvtColor(arr.copy(), cv2.COLOR_RGB2BGR)
                H, W = arr.shape[0], arr.shape[1]
                
                # First, draw all ZBar polygon outlines (thin blue) to show exact tiny detection regions
                for data, r, poly_pts in barcodes:
                    if poly_pts and len(poly_pts) >= 3:
                        pts = np.array(poly_pts, dtype=np.int32)
                        cv2.polylines(overlay, [pts], isClosed=True, color=(255, 100, 0), thickness=1)
                
                # Draw the inflated rectangles (thick green) - these are what we use for zoom crops
                for data, r, poly_pts in barcodes:
                    x, y, w, h = r.left, r.top, r.width, r.height
                    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Group by decoded data and draw union boxes (yellow) to show logical grouping
                groups: dict[str, list] = {}
                for data, r, poly_pts in barcodes:
                    groups.setdefault(data, []).append(r)

                color = (0, 255, 255)
                idx = 0
                for data, rects in groups.items():
                    if len(rects) > 1:  # Only draw union if multiple regions for same code
                        lx = min(r.left for r in rects)
                        ty = min(r.top for r in rects)
                        rx = max(r.left + r.width for r in rects)
                        by = max(r.top + r.height for r in rects)
                        lx, ty, rx, by = int(lx), int(ty), int(rx), int(by)
                        cv2.rectangle(overlay, (lx, ty), (rx, by), color, 3)
                    
                    # Label with index and data preview
                    first_rect = rects[0]
                    label_x = first_rect.left
                    label_y = max(0, first_rect.top - 5)
                    label = f"R{idx}: {data[:12]}" + ('…' if len(data) > 12 else '')
                    cv2.putText(overlay, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                    idx += 1

                overlay_path = os.path.join(DEBUG_ZOOM_DIR, f"{os.path.splitext(debug_name)[0]}_regions.png")
                cv2.imwrite(overlay_path, overlay)
                print(f"[DEBUG] Saved region overlay: {os.path.basename(overlay_path)}")
            except Exception as ex:
                print(f"[DEBUG WARNING] Failed to save region overlay: {ex}")

    _collect(decode(pil_image, symbols=[ZBarSymbol.CODE128]))
    if has_required_barcodes(found):
        return found, union_images

    # Deglare
    _, glare = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    glare = cv2.dilate(glare, np.ones((3, 3), np.uint8), 1)
    _collect(decode(Image.fromarray(cv2.inpaint(gray, glare, 3, cv2.INPAINT_TELEA)), symbols=[ZBarSymbol.CODE128]))
    if has_required_barcodes(found):
        return found, union_images

    # CLAHE
    enh = enhance_for_barcode_reading(arr)
    _collect(decode(Image.fromarray(enh), symbols=[ZBarSymbol.CODE128]))
    if has_required_barcodes(found):
        return found, union_images

    # Otsu
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _collect(decode(Image.fromarray(bin_img), symbols=[ZBarSymbol.CODE128]))
    if has_required_barcodes(found):
        return found, union_images

    # Adaptive
    for blk in [11, 21, 51]:
        adap = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blk, 10)
        _collect(decode(Image.fromarray(adap), symbols=[ZBarSymbol.CODE128]))
        if has_required_barcodes(found):
            return found, union_images

    # Invert
    _collect(decode(Image.fromarray(cv2.bitwise_not(bin_img)), symbols=[ZBarSymbol.CODE128]))
    if has_required_barcodes(found):
        _save_debug_overlay()
        return found, union_images

    # Morph close
    morph = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    _collect(decode(Image.fromarray(morph), symbols=[ZBarSymbol.CODE128]))
    if has_required_barcodes(found):
        return found, union_images

    # Rotation retries if we found barcodes but missing P/S
    if not has_required_barcodes(found) and found:
        # Try rotating ±14°, ±28°, ±42° to handle tilted labels (ZBar reads ±7° from each)
        # Track which rotation finds the most barcodes
        max_barcode_count = len(found)
        best_rotation_angle = 0  # 0 means original image
        
        for angle in [14, -14, 28, -28, 42, -42]:
            h, w = arr.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(arr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            rotated_pil = Image.fromarray(rotated)
            M_inv = cv2.invertAffineTransform(M)
            
            # Track count before rotation
            count_before = len(found)
            _collect(decode(rotated_pil, symbols=[ZBarSymbol.CODE128]), M_inv=M_inv, record_regions=True)
            count_after = len(found)
            
            # Update best rotation if this one found more barcodes
            if count_after > max_barcode_count:
                max_barcode_count = count_after
                best_rotation_angle = angle
                best_rotation_img = cv2.cvtColor(rotated, cv2.COLOR_RGB2GRAY) if len(rotated.shape) == 3 else rotated
            
            if has_required_barcodes(found):
                return found, union_images
            # Try enhanced version too
            rotated_gray = cv2.cvtColor(rotated, cv2.COLOR_RGB2GRAY) if len(rotated.shape) == 3 else rotated
            rotated_enh = enhance_for_barcode_reading(rotated)
            
            count_before_enh = len(found)
            _collect(decode(Image.fromarray(rotated_enh), symbols=[ZBarSymbol.CODE128]), M_inv=M_inv, record_regions=True)
            count_after_enh = len(found)
            
            # Check enhanced version too for best rotation
            if count_after_enh > max_barcode_count:
                max_barcode_count = count_after_enh
                best_rotation_angle = angle
                best_rotation_img = rotated_enh
            
            if has_required_barcodes(found):
                return found, union_images
        
        # Log best rotation found
        if best_rotation_angle != 0 and DEBUG_ZOOM_ROI and debug_name:
            print(f"[DEBUG] Best rotation: {best_rotation_angle}° ({max_barcode_count} barcodes)")
            # Save best rotation image
            if DEBUG_ZOOM_DIR and best_rotation_img is not None:
                try:
                    rotation_path = os.path.join(DEBUG_ZOOM_DIR, f"{os.path.splitext(debug_name)[0]}_best_rotation_{best_rotation_angle}deg.png")
                    cv2.imwrite(rotation_path, best_rotation_img)
                    print(f"[DEBUG] Saved best rotation image: {os.path.basename(rotation_path)}")
                except Exception as ex:
                    print(f"[DEBUG WARNING] Failed to save best rotation image: {ex}")

    # Save debug overlay before zoom attempts (only if P/S not found yet)
    if DEBUG_ZOOM_ROI and debug_name:
        if regions:
            _save_debug_overlay()
        else:
            print(f"[DEBUG] {debug_name}: No barcode regions detected in initial passes")

    # Zoom retries on first few regions
    if regions and not has_required_barcodes(found):
        # Calculate union rectangle based on per-code grouping ("yellow" boxes)
        groups: dict[str, list] = {}
        for data, r, poly_pts in barcodes:
            groups.setdefault(data, []).append(r)
        if groups:
            best = None  # (area, (min_x, min_y, max_x, max_y), data)
            for data, rects in groups.items():
                gx1 = min(rr.left for rr in rects)
                gy1 = min(rr.top for rr in rects)
                gx2 = max(rr.left + rr.width for rr in rects)
                gy2 = max(rr.top + rr.height for rr in rects)
                area = max(1, (gx2 - gx1)) * max(1, (gy2 - gy1))
                if best is None or area > best[0]:
                    best = (area, (gx1, gy1, gx2, gy2), data)
            (min_x, min_y, max_x, max_y) = best[1]
            if DEBUG_ZOOM_ROI and debug_name:
                try:
                    print(f"[DEBUG] Using group-union for zoom: '{best[2]}' area={best[0]} bbox=({min_x},{min_y})-({max_x},{max_y})")
                except Exception:
                    pass
        else:
            # Fallback to union of all green boxes
            min_x = min(r.left for r in regions)
            min_y = min(r.top for r in regions)
            max_x = max(r.left + r.width for r in regions)
            max_y = max(r.top + r.height for r in regions)
        
        # Build anchor-aware union crop fitted to 3:2 aspect
        base_w = max_x - min_x
        base_h = max_y - min_y
        pad_w = int(base_w * 0.85)
        pad_h = int(base_h * 1.5)
        target_aspect = 3.0 / 2.0

        img_h, img_w = gray.shape[:2]

        # Find rects for specific labels using decoded data->rect mapping
        def find_rect(predicate):
            for data, r, poly_pts in barcodes:
                u = data.upper()
                if predicate(u):
                    return r
            return None

        qty_rect = find_rect(lambda s: s.startswith('Q') or 'QTY' in s)
        lot_rect = find_rect(lambda s: '1T' in s or 'LOT' in s)
        origin_rect = find_rect(lambda s: s.startswith('4L') or 'ORIGIN' in s)
        pn_rect = find_rect(lambda s: s.startswith('P') and re.match(r'^P\d{4}-\d{5}$', s))
        sn_rect = find_rect(lambda s: s.startswith('S') and re.match(r'^S900\d{15}$', s))

        # Compute target w/h based on chosen union plus padding, adjusted to 3:2
        # Minimum dimensions: 750px wide x 500px tall
        w = max(750, int((base_w + 2 * pad_w) * 1.15))
        h = max(500, int((base_h + 2 * pad_h) * 1.15))
        if w / max(1, h) < target_aspect:
            w = int(h * target_aspect)
        else:
            h = int(w / target_aspect)
        # Ensure minimums are still met after aspect ratio adjustment
        if w < 750:
            w = 750
            h = int(w / target_aspect)
        if h < 500:
            h = 500
            w = int(h * target_aspect)

        def clamp(val, lo, hi):
            return max(lo, min(hi, val))

        # Anchoring strategies - use specific anchor if only 1 barcode, otherwise loosely follow while ensuring all regions fit
        single_barcode = len(barcodes) == 1
        
        if qty_rect is not None:
            # align QTY barcode near bottom-left with some padding below
            x1 = max(0, qty_rect.left - int(w * 0.05))  # Small left padding
            y2 = min(img_h, qty_rect.top + qty_rect.height + int(h * 0.1))  # Add padding below
            x2 = x1 + w
            y1 = y2 - h
            if x2 > img_w:
                shift = x2 - img_w
                x1 -= shift; x2 -= shift
            if y1 < 0:
                y1 = 0; y2 = h
        elif lot_rect is not None:
            # align LOT barcode to left edge mid-height
            x1 = lot_rect.left
            x2 = x1 + w
            y_center = lot_rect.top + lot_rect.height // 2
            y1 = int(y_center - h / 2)
            y2 = y1 + h
            if x2 > img_w:
                shift = x2 - img_w
                x1 -= shift; x2 -= shift
            if y1 < 0:
                y1 = 0; y2 = h
            if y2 > img_h:
                y2 = img_h; y1 = img_h - h
        elif origin_rect is not None:
            # align ORIGIN barcode along bottom edge, shifted 10% left to show more of left edge
            x_center = origin_rect.left
            x1 = int(x_center - w * 0.6)  # Shift left by using 0.6 instead of 0.5
            x2 = x1 + w
            y2 = origin_rect.top + origin_rect.height
            y1 = y2 - h
            if x1 < 0:
                x1 = 0; x2 = w
            if x2 > img_w:
                x2 = img_w; x1 = img_w - w
            if y1 < 0:
                y1 = 0; y2 = h
        elif pn_rect is not None:
            # align PN barcode to top edge with horizontal centering and minimal padding
            x_center = pn_rect.left + pn_rect.width // 2
            x1 = int(x_center - w / 2)
            x2 = x1 + w
            y1 = max(0, pn_rect.top - 10)  # Just 10px top padding
            y2 = y1 + h
            if x1 < 0:
                x1 = 0; x2 = w
            if x2 > img_w:
                x2 = img_w; x1 = img_w - w
            if y2 > img_h:
                y2 = img_h; y1 = max(0, img_h - h)
            # Ensure y1 doesn't go below 0
            if y1 < 0:
                y1 = 0; y2 = h
        elif sn_rect is not None:
            # center SN barcode in crop
            x_center = sn_rect.left + sn_rect.width // 2
            y_center = sn_rect.top + sn_rect.height // 2
            x1 = int(x_center - w / 2)
            y1 = int(y_center - h / 2)
            x2 = x1 + w
            y2 = y1 + h
            if x1 < 0:
                x1 = 0; x2 = w
            if x2 > img_w:
                x2 = img_w; x1 = img_w - w
            if y1 < 0:
                y1 = 0; y2 = h
            if y2 > img_h:
                y2 = img_h; y1 = img_h - h
        else:
            # Fallback: center on union if no anchor barcode found
            cx = (min_x + max_x) // 2
            cy = (min_y + max_y) // 2
            x1 = clamp(cx - w // 2, 0, img_w - w)
            y1 = clamp(cy - h // 2, 0, img_h - h)
            x2 = x1 + w
            y2 = y1 + h
        
        # CRITICAL: Ensure ALL detected regions are fully included in crop (priority for multi-barcode)
        # For multiple barcodes, expand/shift crop to include everything with margin
        if not single_barcode:
            margin = 20  # pixels
            if min_x < x1 + margin:
                # Regions extend left - adjust crop
                needed = (x1 + margin) - min_x
                x1 = max(0, x1 - needed)
                x2 = min(img_w, x1 + w)
            if max_x > x2 - margin:
                # Regions extend right - adjust crop
                needed = max_x - (x2 - margin)
                x2 = min(img_w, x2 + needed)
                x1 = max(0, x2 - w)
            if min_y < y1 + margin:
                # Regions extend above - adjust crop
                needed = (y1 + margin) - min_y
                y1 = max(0, y1 - needed)
                y2 = min(img_h, y1 + h)
            if max_y > y2 - margin:
                # Regions extend below - adjust crop
                needed = max_y - (y2 - margin)
                y2 = min(img_h, y2 + needed)
                y1 = max(0, y2 - h)
        
        x1 = int(clamp(x1, 0, img_w - 1)); x2 = int(clamp(x2, 1, img_w))
        y1 = int(clamp(y1, 0, img_h - 1)); y2 = int(clamp(y2, 1, img_h))
        
        # Use best rotation image if available, otherwise use original gray
        source_gray = best_rotation_img if best_rotation_img is not None else gray
        crop = source_gray[y1:y2, x1:x2]
        if crop.size == 0:
            return found, union_images
        zoom = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        # Store union zoom in memory for potential OCR usage later
        try:
            union_images.append(zoom)
        except Exception:
            pass
        
        # Save zoom union image if debug mode enabled
        if DEBUG_ZOOM_ROI and DEBUG_ZOOM_DIR and debug_name:
            try:
                zoom_path = os.path.join(DEBUG_ZOOM_DIR, f"{os.path.splitext(debug_name)[0]}_zoom_union.png")
                cv2.imwrite(zoom_path, zoom)
                print(f"[DEBUG] Saved zoom union crop: {os.path.basename(zoom_path)}")
            except Exception as ex:
                print(f"[DEBUG WARNING] Failed to save zoom union crop: {ex}")
        
        # Re-run the barcode scanner on the zoom-union crop (raw, Otsu, CLAHE)
        for z in [zoom, cv2.threshold(zoom, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1], enhance_for_barcode_reading(zoom)]:
            _collect(decode(Image.fromarray(z), symbols=[ZBarSymbol.CODE128]), record_regions=False)
            if has_required_barcodes(found):
                return found, union_images

        # Also try small rotations on the zoom union image to handle slight tilt
        if not has_required_barcodes(found):
            base_variants = [
                ("zoom", zoom),
                ("zoom_enh", enhance_for_barcode_reading(zoom))
            ]
            for name, base in base_variants:
                for ang in [14, -14, 28, -28, 42, -42]:
                    h2, w2 = base.shape[:2]
                    M2 = cv2.getRotationMatrix2D((w2 // 2, h2 // 2), ang, 1.0)
                    rot = cv2.warpAffine(base, M2, (w2, h2), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                    _collect(decode(Image.fromarray(rot), symbols=[ZBarSymbol.CODE128]), record_regions=False)
                    if has_required_barcodes(found):
                        return found, union_images
                    # Try Otsu on rotated as well
                    rot_bin = cv2.threshold(rot, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                    _collect(decode(Image.fromarray(rot_bin), symbols=[ZBarSymbol.CODE128]), record_regions=False)
                    if has_required_barcodes(found):
                        return found, union_images
    return found, union_images


def correct_ocr_part_number(detected_part: str, expected_parts: list[str] | None):
    """Correct small OCR mistakes by matching to expected_parts. Returns corrected value or None."""
    if not expected_parts:
        return detected_part
    x = detected_part[1:] if detected_part.startswith('P') else detected_part
    if x in expected_parts:
        return x
    ocr = {
        '0': ['8', 'O'], '1': ['7', 'I'], '2': ['Z'], '3': ['8'], '4': ['A'],
        '5': ['S'], '6': ['8', '5'], '7': ['1'], '8': ['0', '3', '6', 'B'], '9': ['8']
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
    if best and best != x:
        print(f"       [CORRECTED PART] P{x} → P{best}")
    return best if best else None


def correct_ocr_serial_number(detected_serial: str) -> str | None:
    """Try to salvage a nearly-correct serial by fixing plausible digit confusions in WWYY."""
    s = detected_serial[1:] if detected_serial.startswith('S') else detected_serial
    if not (s.startswith('900') and len(s) == 18):
        return None
    ocr = {
        '0': ['8', 'O'], '1': ['7', 'I'], '2': ['Z'], '3': ['8'], '4': ['A'],
        '5': ['S'], '6': ['8', '5'], '7': ['1'], '8': ['0', '3', '6', 'B'], '9': ['8']
    }
    chars = list(s)
    week = s[10:12]
    year = s[12:14]
    cy = datetime.now().year % 100
    
    # Track if we made any corrections
    corrected = False

    # Correct week digits - but only if year is already valid or will be made valid
    for i, ch in enumerate(week):
        wk_int = int(week)
        if 1 <= wk_int <= 52:
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
                    corrected = True
                    print(f"       [CORRECTED SERIAL] S{s} → S{''.join(chars)} (week char {i}: {ch}→{d})")
                    break

    # Correct year digits
    for i, ch in enumerate(year):
        yr_int = int(year)
        if 24 <= yr_int <= cy:
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
                    corrected = True
                    print(f"       [CORRECTED SERIAL] S{s} → S{''.join(chars)} (year char {i}: {ch}→{d})")
                    break

    if not corrected:
        return None
        
    corrected_serial = 'S' + ''.join(chars)
    # Ensure corrected serial passes full validation
    if is_valid_serial_number(corrected_serial):
        return corrected_serial
    else:
        return None


def extract_text_with_ocr(pil_image: Image.Image, expected_parts: list[str] | None, preferred_images: list[Image.Image] | None = None) -> list[str]:
    """OCR fallback to find part and serial.
    Tries union crops (preferred_images) first, then the original if still missing P/S.
    Each image is processed with CLAHE and Otsu variants, with PSM6 then PSM7 if needed.
    """
    if not TESSERACT_AVAILABLE:
        return []

    def build_variants(img: Image.Image) -> list[np.ndarray]:
        arr = np.array(img)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if len(arr.shape) == 3 else arr
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_img = clahe.apply(gray)
        otsu = cv2.threshold(enhanced_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return [enhanced_img, otsu]

    found: list[str] = []
    candidates: list[np.ndarray] = []

    # Prefer union crops first if provided
    if preferred_images:
        for pim in preferred_images:
            try:
                candidates.extend(build_variants(pim))
            except Exception:
                continue

    # Finally add the original image variants
    candidates.extend(build_variants(pil_image))

    psm_modes = [6]
    for img in candidates:
        for psm in psm_modes:
            try:
                text = pytesseract.image_to_string(Image.fromarray(img), config=f"--psm {psm} -c tessedit_char_whitelist=0123456789-")
                s = text.replace(' ', '')
                # Part numbers
                for m in re.findall(r"\d{4}-?\d{5}", s):
                    norm = m if '-' in m else f"{m[:4]}-{m[4:]}"
                    if expected_parts:
                        corr = correct_ocr_part_number(norm, expected_parts)
                        if not corr:
                            continue
                        norm = corr
                    P = f"P{norm}"
                    if P not in found and re.fullmatch(r"^P\d{4}-\d{5}$", P):
                        found.append(P)
                # Serials
                for m in re.findall(r"900\d{15}", s):
                    S = f"S{m}"
                    if S not in found and is_valid_serial_number(S):
                        found.append(S)
                    elif S not in found:
                        corr = correct_ocr_serial_number(S)
                        if corr and corr not in found:
                            found.append(corr)
                if has_required_barcodes(found):
                    return found
            except Exception:
                continue
        if not any(c.startswith('S') for c in found) and 7 not in psm_modes:
            psm_modes.append(7)
    return found


def classify_barcode(data: str) -> str:
    if re.fullmatch(r"^P\d{4}-\d{5}$", data):
        return "Part Number"
    if is_valid_serial_number(data):
        return "Serial Number"
    return "Other"


def extract_identifiers(barcodes: list[str]):
    part = serial = None
    for c in barcodes:
        t = classify_barcode(c)
        if t == "Part Number" and not part:
            part = c
        elif t == "Serial Number" and not serial:
            serial = c
    return part, serial


def create_folder_structure(base: str, part: str | None, serial: str | None) -> str:
    if not part:
        target = os.path.join(base, "_Unsorted")
    elif part and serial:
        target = os.path.join(base, part[1:], (serial[-8:] if len(serial) >= 8 else serial))
    else:
        target = os.path.join(base, part[1:] if part.startswith('P') else part, "_No_Serial")
    os.makedirs(target, exist_ok=True)
    return target


# Multiprocessing worker helpers - these set the global DEBUG_ZOOM_DIR in worker processes
def _process_barcode_with_debug(args):
    """Wrapper for multiprocessing that sets DEBUG_ZOOM_DIR in worker process."""
    image_path, debug_dir = args
    global DEBUG_ZOOM_DIR
    DEBUG_ZOOM_DIR = debug_dir
    return process_image_barcode_only(image_path)

def _process_ocr_with_debug(args):
    """Wrapper for multiprocessing that sets DEBUG_ZOOM_DIR in worker process."""
    image_path, expected_parts, debug_dir, union_bytes = args
    global DEBUG_ZOOM_DIR
    DEBUG_ZOOM_DIR = debug_dir
    return process_image_ocr(image_path, expected_parts, union_bytes)


def process_image_ocr(image_path: str, expected_parts: list[str] | None, union_bytes: list[bytes] | None = None):
    start = datetime.now()
    try:
        pimg = Image.open(image_path)
        pimg = apply_exif_orientation(pimg)
        # Build preferred images from in-memory union crop bytes
        preferred_images: list[Image.Image] | None = None
        if union_bytes:
            preferred_images = []
            for b in union_bytes:
                try:
                    preferred_images.append(Image.open(io.BytesIO(b)))
                except Exception:
                    continue

        ocr_codes = extract_text_with_ocr(pimg, expected_parts, preferred_images=preferred_images)
        return {
            'image_path': image_path,
            'ocr_codes': ocr_codes,
            'success': True,
            'processing_time': (datetime.now() - start).total_seconds(),
        }
    except Exception as e:
        return {
            'image_path': image_path,
            'ocr_codes': [],
            'success': False,
            'error': str(e),
            'processing_time': (datetime.now() - start).total_seconds(),
        }


def process_image_barcode_only(image_path: str):
    start = datetime.now()
    try:
        pimg = Image.open(image_path)
        pimg = apply_exif_orientation(pimg)
        # Optional debug: pass image name so ROI overlays can be saved if enabled
        dbg_name = os.path.basename(image_path) if DEBUG_ZOOM_ROI else None
        all_codes, union_imgs = decode_barcodes(pimg, debug_name=dbg_name)
        part, serial = extract_identifiers(all_codes)
        # Encode union crops to PNG bytes for cross-process handoff
        union_bytes: list[bytes] = []
        try:
            for z in union_imgs or []:
                ok, buf = cv2.imencode('.png', z)
                if ok:
                    union_bytes.append(buf.tobytes())
        except Exception:
            pass
        return {
            'image_path': image_path,
            'part_number': part,
            'serial_number': serial,
            'all_barcodes': all_codes,
            'union_crops': union_bytes,
            'success': True,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'processing_time': (datetime.now() - start).total_seconds(),
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
            'processing_time': (datetime.now() - start).total_seconds(),
        }


def generate_report(results: list[dict], out_dir: str, total_time: float | None):
    path = os.path.join(out_dir, "_sorting_report.txt")
    with open(path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("BARCODE SORTING REPORT\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        total = len(results)
        ok = sum(1 for r in results if r.get('success'))
        with_part = sum(1 for r in results if r.get('part_number'))
        with_serial = sum(1 for r in results if r.get('serial_number'))
        with_both = sum(1 for r in results if r.get('part_number') and r.get('serial_number'))
        f.write(f"Total images processed: {total}\nSuccessfully processed: {ok}\n")
        f.write(f"Images with part number: {with_part}\n")
        f.write(f"Images with serial number: {with_serial}\n")
        f.write(f"Images with both: {with_both}\n")
        if total_time is not None:
            f.write(f"\nTotal processing time: {total_time:.1f}s ({total_time/60:.1f} minutes)\n")
        f.write("\nDETAILS:\n\n")
        for r in results:
            f.write(f"{r['filename']} | P={r.get('part_number') or '—'} | S={r.get('serial_number') or '—'} | Codes={', '.join(r.get('all_barcodes', [])) or '—'}\n")
    print(f"\n[INFO] Report saved to: {path}")


# --------------------------- Quick modes ----------------------------

def run_small_batch_test(expected_parts: list[str] | None):
    print("\n" + "=" * 80)
    print("SMALL BATCH TEST - Quick Debug Check")
    print("=" * 80)
    root = Tk(); root.withdraw()
    files = filedialog.askopenfilenames(title="Select 1-10 Images to Test",
                                        filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif"), ("All files", "*.*")])
    if not files:
        print("[INFO] No images selected. Exiting small batch test.")
        return
    if len(files) > 10:
        messagebox.showwarning("Too Many Images", f"Selected {len(files)} images. Limiting to first 10.")
        files = files[:10]

    results = []
    for i, p in enumerate(files, 1):
        name = os.path.basename(p)
        print(f"\n[{i}/{len(files)}] {name}")
        r = {'filename': name, 'part_number': None, 'serial_number': None, 'all_barcodes': [], 'success': True}
        try:
            b = process_image_barcode_only(p)
            r['all_barcodes'] = b['all_barcodes']
            r['part_number'], r['serial_number'] = extract_identifiers(r['all_barcodes'])
            if not (r['part_number'] and r['serial_number']):
                print("  → Trying OCR fallback…")
                o = process_image_ocr(p, expected_parts)
                for code in o['ocr_codes']:
                    if code not in r['all_barcodes']:
                        r['all_barcodes'].append(code)
                r['part_number'], r['serial_number'] = extract_identifiers(r['all_barcodes'])
        except Exception as e:
            r['success'] = False
            r['error'] = str(e)
        finally:
            print(f"  Part: {r['part_number'] or '❌'}  |  Serial: {r['serial_number'] or '❌'}")
            results.append(r)

    print("\n" + "=" * 80)
    print("SMALL BATCH TEST SUMMARY")
    print("=" * 80)
    both = sum(1 for r in results if r['part_number'] and r['serial_number'])
    print(f"Total: {len(results)} | Both: {both} | Part only: {sum(1 for r in results if r['part_number']) - both} | Serial only: {sum(1 for r in results if r['serial_number']) - both}")
    print("=" * 80)


def launch_ocr_region_tester():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tester = os.path.join(script_dir, 'ocr_region_tester.py')
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location('ocr_region_tester', tester)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, 'main'):
                mod.main(); return
    except Exception:
        pass
    if os.path.exists(tester):
        try:
            subprocess.run([sys.executable, tester], check=False)
        except Exception as e:
            print(f"[ERROR] Could not launch OCR Region Tester: {e}")
    else:
        print("[ERROR] OCR Region Tester not found.")


def main():
    start = datetime.now()
    print("=" * 80)
    print("BARCODE IMAGE SORTER - RC1")
    print(f"Started at: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print("[INFO] Press Ctrl+C to stop early; processed images will still be sorted")

    all_results = []
    interrupted = False

    def on_sigint(_sig, _frm):
        nonlocal interrupted
        interrupted = True
        print("\n[INFO] Ctrl+C detected. Finishing in-flight work…")
    signal.signal(signal.SIGINT, on_sigint)

    expected = load_expected_part_numbers()

    # Input selection (or quick modes)
    Tk().withdraw()
    parts, workers, mode = select_part_numbers_dialog(expected if expected else []) if expected else (None, max(1, int((os.cpu_count() or 4) * 0.5)), None)
    if expected:
        if mode == 'small_batch':
            run_small_batch_test(parts if parts else expected)
            return
        if mode == 'ocr_region':
            print("\n[INFO] Launching OCR Region Tester…")
            launch_ocr_region_tester()
            return
        if parts is None:
            print("[INFO] Cancelled. Exiting.")
            return
        if not parts:
            print("[INFO] No part numbers selected. Exiting.")
            return
        selected_parts = parts
        max_workers = workers
        print(f"[INFO] Selected {len(selected_parts)} part numbers")
    else:
        selected_parts = None
        max_workers = os.cpu_count() or 4

    input_dir = filedialog.askdirectory(title="Select Folder Containing Images to Sort")
    if not input_dir:
        print("[INFO] No folder selected. Exiting.")
        return

    out_dir = os.path.join(input_dir, "Sorted_Images"); os.makedirs(out_dir, exist_ok=True)
    
    # Set debug directory if debug mode is enabled
    global DEBUG_ZOOM_DIR
    if DEBUG_ZOOM_ROI:
        DEBUG_ZOOM_DIR = os.path.join(out_dir, "_zoom_debug")
        os.makedirs(DEBUG_ZOOM_DIR, exist_ok=True)
        print(f"[DEBUG] Zoom ROI debug enabled. Images will be saved to: {DEBUG_ZOOM_DIR}")

    # Discover images
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(exts)]
    if not files:
        print("[ERROR] No image files found")
        return
    print(f"\n[INFO] Found {len(files)} image(s) in: {input_dir}")

    # Pass 1
    print("\n[INFO] Pass 1: Quick barcode scanning (parallel)…")
    t1 = datetime.now(); completed = 0
    executor = None
    try:
        executor = ProcessPoolExecutor(max_workers=max_workers)
        fut2path = {executor.submit(_process_barcode_with_debug, (p, DEBUG_ZOOM_DIR)): p for p in files}
        for fut in as_completed(fut2path):
            if interrupted:
                print("\n[INFO] Stopping Pass 1 early…")
                # Cancel anything pending and stop waiting
                executor.shutdown(wait=False, cancel_futures=True)
                # Drain any futures that already finished but haven't been recorded yet
                for f, pth in fut2path.items():
                    if f.done():
                        try:
                            r = f.result()
                            if r not in all_results:
                                all_results.append(r)
                        except Exception:
                            pass
                break
            completed += 1
            path = fut2path[fut]
            try:
                r = fut.result(); all_results.append(r)
                msg = "✅ Part & Serial" if (r['part_number'] and r['serial_number']) else (
                      "✅ Part only" if r['part_number'] else (
                      "✅ Serial only" if r['serial_number'] else (
                      f"⚠️  {len(r['all_barcodes'])} barcode(s)" if r['all_barcodes'] else "⚠️  No barcodes")))
                print(f"[{completed}/{len(files)}] {os.path.basename(path)} → {msg} ({r['processing_time']:.1f}s)")
            except Exception as e:
                print(f"[{completed}/{len(files)}] {os.path.basename(path)} ❌ {e}")
    except KeyboardInterrupt:
        interrupted = True
        print("\n[INFO] Keyboard interrupt in Pass 1")
    finally:
        if executor:
            executor.shutdown(wait=False, cancel_futures=True)
    print(f"[INFO] Pass 1 completed in {(datetime.now()-t1).total_seconds():.1f}s")

    # Pass 2
    need_ocr = [r for r in all_results if not has_required_barcodes(r['all_barcodes'])]
    if not interrupted and need_ocr:
        print(f"\n[INFO] Pass 2: OCR fallback for {len(need_ocr)} image(s) (parallel)…")
        t2 = datetime.now(); completed = 0
        executor2 = None
        try:
            executor2 = ProcessPoolExecutor(max_workers=max_workers)
            fut2path = {executor2.submit(_process_ocr_with_debug, (r['image_path'], selected_parts, DEBUG_ZOOM_DIR, r.get('union_crops'))): r['image_path'] for r in need_ocr}
            for fut in as_completed(fut2path):
                if interrupted:
                    print("\n[INFO] Stopping Pass 2 early…")
                    executor2.shutdown(wait=False, cancel_futures=True)
                    # Drain finished futures and attach any OCR results we already have
                    for f, pth in fut2path.items():
                        if f.done():
                            try:
                                o = f.result(); codes = o.get('ocr_codes', [])
                                base = next((r for r in all_results if r['image_path'] == pth), None)
                                if base is not None and codes:
                                    for c in codes:
                                        if c not in base['all_barcodes']:
                                            base['all_barcodes'].append(c)
                                    base['part_number'], base['serial_number'] = extract_identifiers(base['all_barcodes'])
                            except Exception:
                                pass
                    break
                completed += 1
                p = fut2path[fut]
                try:
                    o = fut.result(); codes = o['ocr_codes']
                    base = next((r for r in all_results if r['image_path'] == p), None)
                    if base is not None and codes:
                        for c in codes:
                            if c not in base['all_barcodes']:
                                base['all_barcodes'].append(c)
                        base['part_number'], base['serial_number'] = extract_identifiers(base['all_barcodes'])
                        print(f"[{completed}/{len(need_ocr)}] {os.path.basename(p)} → ✅ Found P/S ({o['processing_time']:.1f}s)")
                    else:
                        print(f"[{completed}/{len(need_ocr)}] {os.path.basename(p)} → ⚠️  Nothing found ({o['processing_time']:.1f}s)")
                except Exception as e:
                    print(f"[{completed}/{len(need_ocr)}] {os.path.basename(p)} ❌ {e}")
        except KeyboardInterrupt:
            interrupted = True
            print("\n[INFO] Keyboard interrupt in Pass 2")
        finally:
            if executor2:
                executor2.shutdown(wait=False, cancel_futures=True)
        print(f"[INFO] Pass 2 completed in {(datetime.now()-t2).total_seconds():.1f}s")
    elif not need_ocr:
        print("\n[INFO] Pass 2: Skipped (all images have barcodes)")

    # Pass 3 (organize/copy)
    print("\n[INFO] Pass 3: Organizing and copying images…")
    final = []
    try:
        for i, r in enumerate(all_results, 1):
            t = datetime.now()
            target = create_folder_structure(out_dir, r['part_number'], r['serial_number'])
            dest = os.path.join(target, os.path.basename(r['image_path']))
            try:
                shutil.copy2(r['image_path'], dest)
                print(f"[{i}/{len(all_results)}] {os.path.basename(r['image_path'])} → ✅ ({(datetime.now()-t).total_seconds():.2f}s)")
                final.append({'filename': os.path.basename(r['image_path']),
                              'part_number': r['part_number'],
                              'serial_number': r['serial_number'],
                              'all_barcodes': r['all_barcodes'],
                              'target_folder': target,
                              'success': True,
                              'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                              'processing_time': r.get('processing_time', 0)})
            except Exception as e:
                print(f"[{i}/{len(all_results)}] {os.path.basename(r['image_path'])} → ❌ {e}")
                final.append({'filename': os.path.basename(r['image_path']),
                              'part_number': r['part_number'],
                              'serial_number': r['serial_number'],
                              'all_barcodes': r['all_barcodes'],
                              'target_folder': None,
                              'success': False,
                              'error': str(e),
                              'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                              'processing_time': r.get('processing_time', 0)})
    except KeyboardInterrupt:
        interrupted = True
        print("\n[INFO] Keyboard interrupt in Pass 3")

    total_time = (datetime.now() - start).total_seconds()
    print("\n" + "=" * 80)
    if interrupted:
        print("[INFO] Processing interrupted by user!")
    print(f"[INFO] Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"[INFO] Images organized in: {out_dir}")
    print("=" * 80)

    generate_report(final, out_dir, total_time)

    # Try to open the output folder (Windows)
    if not interrupted:
        try:
            os.startfile(out_dir)
        except Exception:
            pass
        
        # Show completion dialog with auto-close after 30 seconds
        root = Tk()
        root.withdraw()
        dialog = Toplevel(root)
        dialog.title("Processing Complete")
        dialog.geometry("450x180")
        dialog.resizable(False, False)
        
        # Center the dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"{dialog.winfo_width()}x{dialog.winfo_height()}+{x}+{y}")
        
        msg = f"Processed {len(final)} images.\n\nResults saved to:\n{out_dir}\n\n(Auto-closing in 30 seconds...)"
        Label(dialog, text=msg, wraplength=400, justify="left", padx=20, pady=20).pack(expand=True)
        
        def close_dialog():
            dialog.destroy()
            root.quit()
            root.destroy()
        
        Button(dialog, text="OK", command=close_dialog, width=10, bg="#4CAF50", fg="white").pack(pady=(0, 15))
        
        # Auto-close after 30 seconds
        dialog.after(30000, close_dialog)
        dialog.mainloop()
    else:
        # On interrupt, exit immediately after writing report to free the terminal faster
        print("[INFO] Exiting immediately due to Ctrl+C (report already written).")
        sys.exit(0)


if __name__ == "__main__":
    main()
