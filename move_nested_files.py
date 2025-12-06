#!/usr/bin/env python3
"""
Script to move files from nested diag/ and rx/ folders to top-level folders.

This script scans all subdirectories in ./cir_files/ and moves files from:
- ./cir_files/{folder_name}/diag/ -> ./cir_files/diag/
- ./cir_files/{folder_name}/rx/ -> ./cir_files/rx/

Files are renamed to include the parent folder name to avoid conflicts.
"""

import os
import shutil
from pathlib import Path


def move_nested_files(base_dir="cir_files"):
    """
    Move files from nested diag/ and rx/ folders to top-level folders.
    
    Args:
        base_dir: The base directory containing the subdirectories (default: "cir_files")
    """
    # Get absolute path to base directory
    script_dir = Path(__file__).parent
    base_path = script_dir / base_dir
    
    if not base_path.exists():
        print(f"Error: Directory {base_path} does not exist!")
        return
    
    # Create top-level diag and rx folders if they don't exist
    top_diag = base_path / "diag"
    top_rx = base_path / "rx"
    top_diag.mkdir(exist_ok=True)
    top_rx.mkdir(exist_ok=True)
    
    print(f"Scanning {base_path} for nested diag/ and rx/ folders...")
    print("-" * 70)
    
    moved_count = 0
    
    # Iterate through all subdirectories in cir_files
    for item in base_path.iterdir():
        if not item.is_dir():
            continue
        
        # Skip the top-level diag, rx, and tx folders
        if item.name in ["diag", "rx", "tx"]:
            continue
        
        folder_name = item.name
        
        # Check for diag/ subfolder
        diag_folder = item / "diag"
        if diag_folder.exists() and diag_folder.is_dir():
            files = list(diag_folder.iterdir())
            if files:
                print(f"\nFound {len(files)} file(s) in {folder_name}/diag/")
                for file_path in files:
                    if file_path.is_file():
                        # Create new filename with parent folder prefix
                        new_filename = f"{file_path.name}"
                        dest_path = top_diag / new_filename
                        
                        # Move the file
                        try:
                            shutil.move(str(file_path), str(dest_path))
                            print(f"  ✓ Moved: {file_path.name} -> diag/{new_filename}")
                            moved_count += 1
                        except Exception as e:
                            print(f"  ✗ Error moving {file_path.name}: {e}")
        
        # Check for rx/ subfolder
        rx_folder = item / "rx"
        if rx_folder.exists() and rx_folder.is_dir():
            files = list(rx_folder.iterdir())
            if files:
                print(f"\nFound {len(files)} file(s) in {folder_name}/rx/")
                for file_path in files:
                    if file_path.is_file():
                        # Create new filename with parent folder prefix
                        new_filename = f"{file_path.name}"
                        dest_path = top_rx / new_filename
                        
                        # Move the file
                        try:
                            shutil.move(str(file_path), str(dest_path))
                            print(f"  ✓ Moved: {file_path.name} -> rx/{new_filename}")
                            moved_count += 1
                        except Exception as e:
                            print(f"  ✗ Error moving {file_path.name}: {e}")
    
    print("\n" + "=" * 70)
    print(f"Summary: Moved {moved_count} file(s) total")
    print("=" * 70)


if __name__ == "__main__":
    move_nested_files()
