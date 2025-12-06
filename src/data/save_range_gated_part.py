"""
Script to extract range-gated parts from CIR data files.

For each .npy file in the data directory:
1. Check if the filename contains a top-level key from the YAML (e.g., "Horizontal", "Vertical")
2. For matching files, extract +-35 bins around each range bin center defined in the YAML
3. Concatenate the extracted data across all subentries (e.g., Desk 1, Desk 2)
4. Save the range-gated data

Input shape: (T, 4, 3, 2, 768)
Output shape: (T, 4, 3, 2, 70 * num_subentries)
"""

import os
import numpy as np
import yaml
from pathlib import Path


def load_range_mapping(yaml_path: str) -> dict:
    """Load range mapping from YAML file."""
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def extract_range_gated_data(data: np.ndarray, range_centers: np.ndarray, half_window: int = 35) -> np.ndarray:
    """
    Extract range-gated data around specified centers.

    Args:
        data: Input data of shape (T, 4, 3, 2, 768)
        range_centers: Array of shape (4, 3) containing range bin centers
        half_window: Half window size (default 35 for +-35 bins = 70 total)

    Returns:
        Extracted data of shape (T, 4, 3, 2, 70)
    """
    T, num_nodes, num_pairs, num_channels, num_bins = data.shape
    window_size = 2 * half_window  # 70 bins

    output = np.zeros((T, num_nodes, num_pairs, num_channels, window_size), dtype=data.dtype)

    for node_idx in range(num_nodes):
        for pair_idx in range(num_pairs):
            center = int(range_centers[node_idx, pair_idx])
            start = max(0, center - half_window)
            end = min(num_bins, center + half_window)

            # Handle boundary cases
            out_start = half_window - (center - start)
            out_end = out_start + (end - start)

            output[:, node_idx, pair_idx, :, out_start:out_end] = data[:, node_idx, pair_idx, :, start:end]

    return output


def process_files(data_dir: str, yaml_path: str, output_dir: str = None, half_window: int = 35):
    """
    Process all .npy files in data_dir using range mapping from yaml_path.

    Args:
        data_dir: Directory containing .npy files
        yaml_path: Path to YAML file with range mappings
        output_dir: Directory to save output files (default: same as data_dir)
        half_window: Half window size for range gating (default 35)
    """
    if output_dir is None:
        output_dir = data_dir

    os.makedirs(output_dir, exist_ok=True)

    # Load range mapping
    range_mapping = load_range_mapping(yaml_path)
    top_level_keys = list(range_mapping.keys())  # e.g., ["Horizontal", "Vertical"]

    # Get all .npy files
    npy_files = list(Path(data_dir).glob("*.npy"))

    for npy_file in npy_files:
        filename = npy_file.name

        # Find which top-level key matches this file
        matched_key = None
        for key in top_level_keys:
            if key in filename:
                matched_key = key
                break

        if matched_key is None:
            print(f"Skipping {filename}: no matching key found in YAML")
            continue

        print(f"Processing {filename} with key '{matched_key}'")

        # Load data
        data = np.load(npy_file)
        # Expected shape: (T, 4, 3, 2, 768)

        # Get subentries for this key (e.g., Desk 1, Desk 2)
        subentries = range_mapping[matched_key]

        # Extract range-gated data for each subentry and concatenate
        gated_parts = []
        for subentry in subentries:
            # subentry is a dict like {"Desk 1": [[...], [...], [...], [...]]}
            subentry_name = list(subentry.keys())[0]
            range_centers_list = subentry[subentry_name]

            # Convert to numpy array of shape (4, 3)
            range_centers = np.array(range_centers_list)

            # Extract range-gated data
            gated_data = extract_range_gated_data(data, range_centers, half_window)
            gated_parts.append(gated_data)
            print(f"  Extracted {subentry_name}: shape {gated_data.shape}")

        # Concatenate along the last dimension
        # Each part is (T, 4, 3, 2, 70), concatenated to (T, 4, 3, 2, 70*num_subentries)
        output_data = np.concatenate(gated_parts, axis=-1)

        # Save output
        output_filename = npy_file.stem + "_range_gated.npy"
        output_path = os.path.join(output_dir, output_filename)
        np.save(output_path, output_data)
        print(f"  Saved: {output_path}, shape: {output_data.shape}")


if __name__ == "__main__":
    # Default paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    yaml_path = os.path.join(script_dir, "range_mapping_oneLongDesk.yaml")

    process_files(data_dir, yaml_path)
