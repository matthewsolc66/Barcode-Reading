"""
Dependency Installer for Barcode Sorter

This script automatically installs all required Python packages for the barcode_sorter.py script.
Double-click to run.
"""

import subprocess
import sys
import os
from tkinter import Tk, messagebox

def install_package(package_name, display_name=None):
    """Install a Python package using pip."""
    if display_name is None:
        display_name = package_name
    
    print(f"\n{'='*60}")
    print(f"Installing {display_name}...")
    print('='*60)
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"✓ {display_name} installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install {display_name}")
        print(f"Error: {e}")
        return False

def main():
    """Main installation function."""
    print("="*60)
    print("BARCODE SORTER - DEPENDENCY INSTALLER")
    print("="*60)
    print("\nThis script will install the following packages:")
    print("  - Pillow (Image processing)")
    print("  - pyzbar (Barcode reading)")
    print("  - opencv-python (Image preprocessing)")
    print("  - numpy (Numerical operations)")
    print("  - pytesseract (OCR fallback)")
    print("  - psutil (System information)")
    print("\nNote: You will also need to install Tesseract-OCR separately")
    print("for OCR functionality. Download from:")
    print("https://github.com/UB-Mannheim/tesseract/wiki")
    print("\n" + "="*60 + "\n")
    
    input("Press Enter to begin installation...")
    
    # List of packages to install
    packages = [
        ("Pillow", "Pillow"),
        ("pyzbar", "pyzbar"),
        ("opencv-python", "opencv-python (cv2)"),
        ("numpy", "numpy"),
        ("pytesseract", "pytesseract"),
        ("psutil", "psutil")
    ]
    
    results = []
    for package_name, display_name in packages:
        success = install_package(package_name, display_name)
        results.append((display_name, success))
    
    # Summary
    print("\n" + "="*60)
    print("INSTALLATION SUMMARY")
    print("="*60)
    
    all_successful = True
    for display_name, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{display_name:30} {status}")
        if not success:
            all_successful = False
    
    print("\n" + "="*60)
    
    if all_successful:
        print("\n✓ All packages installed successfully!")
        print("\nREMINDER: Don't forget to install Tesseract-OCR:")
        print("https://github.com/UB-Mannheim/tesseract/wiki")
        print("\nYou can now run barcode_sorter.py")
        
        # Show success dialog
        root = Tk()
        root.withdraw()
        messagebox.showinfo(
            "Installation Complete",
            "All Python packages installed successfully!\n\n"
            "IMPORTANT: You still need to install Tesseract-OCR separately\n"
            "for OCR functionality. Download from:\n"
            "https://github.com/UB-Mannheim/tesseract/wiki\n\n"
            "You can now run barcode_sorter.py"
        )
    else:
        print("\n✗ Some packages failed to install.")
        print("Please check the error messages above and try again.")
        print("You may need to run this script as administrator.")
        
        # Show error dialog
        root = Tk()
        root.withdraw()
        messagebox.showerror(
            "Installation Failed",
            "Some packages failed to install.\n\n"
            "Please check the console output for details.\n"
            "You may need to run as administrator."
        )
    
    print("\n")
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()
