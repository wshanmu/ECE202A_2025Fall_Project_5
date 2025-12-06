#!/usr/bin/env python3
"""
Batch processing script for multiple CIR data folders.
Usage:
    python batch_process_folders.py --base-dir ./tdma_sensing/cir_files --pattern "20251103_*"
    python batch_process_folders.py --folders folder1 folder2 folder3
"""

import os
import sys
import glob
import argparse
import subprocess
from pathlib import Path


def find_folders(base_dir, pattern="*"):
    """Find all folders matching the pattern in base_dir."""
    search_path = os.path.join(base_dir, pattern)
    folders = [f for f in glob.glob(search_path) if os.path.isdir(f)]
    return sorted(folders)


def process_folder(folder_path):
    """Process a single folder using process_sensing_data.py."""
    print(f"\n{'='*60}")
    print(f"Processing: {folder_path}")
    print(f"{'='*60}")

    try:
        # Run the processing script with Hydra override
        result = subprocess.run(
            ["python", "process_sensing_data.py", f"input_folder={folder_path}"],
            check=True,
            capture_output=False  # Show output in real-time
        )
        print(f"✓ Successfully processed: {folder_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to process: {folder_path}")
        print(f"   Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Batch process CIR data folders")

    # Option 1: Auto-discover folders
    parser.add_argument(
        "--base-dir",
        type=str,
        default="./tdma_sensing/cir_files",
        help="Base directory containing data folders"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="20251103_*",
        help="Pattern to match folder names (e.g., '20251103_*', '*occupied*')"
    )

    # Option 2: Manually specify folders
    parser.add_argument(
        "--folders",
        nargs="+",
        help="List of specific folders to process"
    )

    # Additional options
    parser.add_argument(
        "--exclude",
        nargs="+",
        help="Patterns to exclude from processing"
    )

    args = parser.parse_args()

    # Determine which folders to process
    if args.folders:
        # Use manually specified folders
        folders = args.folders
        print(f"Processing {len(folders)} manually specified folders...")
    else:
        # Auto-discover folders
        folders = find_folders(args.base_dir, args.pattern)
        print(f"Found {len(folders)} folders matching pattern '{args.pattern}'")

        # Apply exclusion filters if specified
        if args.exclude:
            original_count = len(folders)
            for exclude_pattern in args.exclude:
                folders = [f for f in folders if exclude_pattern not in f]
            print(f"Excluded {original_count - len(folders)} folders")

    if not folders:
        print("No folders to process!")
        return

    # Display folders to be processed
    print("\nFolders to process:")
    for i, folder in enumerate(folders, 1):
        print(f"  {i}. {folder}")

    # Ask for confirmation
    response = input(f"\nProcess {len(folders)} folders? [Y/n]: ")
    if response.lower() not in ['y', 'yes', '']:
        print("Cancelled.")
        return

    # Process each folder
    successful = 0
    failed = 0

    for folder in folders:
        if process_folder(folder):
            successful += 1
        else:
            failed += 1

    # Summary
    print(f"\n{'='*60}")
    print("Batch Processing Summary")
    print(f"{'='*60}")
    print(f"Total folders: {len(folders)}")
    print(f"✓ Successful: {successful}")
    print(f"✗ Failed: {failed}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
