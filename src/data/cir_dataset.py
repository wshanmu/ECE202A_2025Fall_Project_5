"""
PyTorch Dataset and DataLoader implementation for CIR (Channel Impulse Response) data.

This module provides a complete data loading pipeline for .npy files containing complex-valued
CIR measurements from multiple nodes. It supports:
- Sliding window segmentation with overlap
- Complex-to-polar conversion (magnitude + unwrapped phase)
- Min-max normalization per segment
- YAML-based configuration for train/val/test splits
- Optional data augmentation (Gaussian noise, time warping)
- Flexible sender node selection (single sender or all senders)
- Easy-to-use DataLoader creation

Data Format:
-----------
Input .npy files: shape (3, 2, X, 896)
    - 3: number of nodes (senders)
    - 2: sender-listener pairs (we use the 2 listening nodes)
    - X: number of frames (~4000-5000)
    - 896: feature dimension
    - dtype: complex128

Output samples: shape (2, 2, 1500, 896)
    - 1st dim (2): the 2 listening nodes
    - 2nd dim (2): magnitude and unwrapped phase
    - 3rd dim (1500): consecutive frames (window size)
    - 4th dim (896): feature dimension
    - dtype: float32 (min-max normalized)

Processing Pipeline:
-------------------
1. Load full .npy file (3, 2, X, 896)
2. Apply low-pass filter (if enabled) to full file
3. Extract sender data (2, X, 896)
4. Apply augmentations to FULL file before segmentation:
   - Time warp: Operates on entire temporal sequence
   - Gaussian noise: Applied to full measurement
5. Segment into windows (window_size=1500, stride=750, 50% overlap)
6. Apply range gating (if enabled)
7. Complex-to-polar conversion (magnitude + unwrapped phase)
8. Min-max normalization per segment

Segmentation:
------------
- Window size: 1500 frames (default)
- Stride: 750 frames (50% overlap, default)
- Frame rate: 100 Hz (100 frames/second)

Usage:
------
    from cir_dataset import get_dataloaders

    # Option 1: Use only a specific sender node (e.g., sender 0)
    train_loader, val_loader, test_loader = get_dataloaders(
        yaml_path='config.yaml',
        batch_size=32,
        num_workers=4,
        sender_idx=0,  # Use only sender 0
        augmentations_train=['gaussian_noise', 'time_warp']
    )

    # Option 2: Use all sender nodes (3x more data)
    train_loader, val_loader, test_loader = get_dataloaders(
        yaml_path='config.yaml',
        batch_size=32,
        sender_idx=None,  # Use all senders (0, 1, 2)
        augmentations_train=['gaussian_noise']
    )

    for batch_data, batch_labels in train_loader:
        # batch_data: (batch_size, 2, 2, 1500, 896) float32 [magnitude, phase]
        # batch_labels: (batch_size,) int64
        ...
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import yaml
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Callable
import warnings
from scipy.signal import butter, filtfilt


# ============================================================================
# Preprocessing Functions
# ============================================================================

def butter_lowpass_filter(data: np.ndarray, cutoff: float, fs: float, order: int = 5, axis: int = -2) -> np.ndarray:
    """
    Apply Butterworth low-pass filter to complex-valued data along specified axis.

    This filter is applied along the slow time axis (frame dimension) to remove
    high-frequency noise and smooth the temporal evolution of the CIR.

    Args:
        data: Input array (complex-valued). Shape: (..., frames, ...)
        cutoff: Cutoff frequency in Hz
        fs: Sampling frequency in Hz (default: 100 Hz for CIR data)
        order: Filter order (higher = sharper cutoff, default: 5)
        axis: Axis along which to apply filter (default: -2 for slow time)

    Returns:
        Filtered data with same shape and dtype as input

    Note:
        Uses filtfilt for zero-phase filtering (no time shift).
        Filters real and imaginary parts independently.

    Example:
        >>> data = np.load('cir_data.npy')  # Shape: (3, 2, 4000, 896)
        >>> # Filter along axis 2 (slow time) with 2 Hz cutoff
        >>> filtered = butter_lowpass_filter(data, cutoff=2.0, fs=100, order=5, axis=2)
    """
    # Design Butterworth low-pass filter
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    # Apply filter to real and imaginary parts separately
    if np.iscomplexobj(data):
        # Filter real part
        real_filtered = filtfilt(b, a, data.real, axis=axis)
        # Filter imaginary part
        imag_filtered = filtfilt(b, a, data.imag, axis=axis)
        # Combine back to complex
        return real_filtered + 1j * imag_filtered
    else:
        # Real-valued data
        return filtfilt(b, a, data, axis=axis)


# ============================================================================
# Data Augmentation Classes
# ============================================================================

class GaussianNoise:
    """
    Add Gaussian noise to complex-valued data.

    Noise is added independently to real and imaginary parts.

    Args:
        std: Standard deviation of the Gaussian noise (relative to data std if relative=True)
        relative: If True, std is multiplied by the standard deviation of the input data
    """
    def __init__(self, std: float = 0.01, relative: bool = True):
        self.std = std
        self.relative = relative

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Complex tensor of shape (2, T, R) where T is time frames, R is range bins
               Applied to full file before segmentation

        Returns:
            Noisy tensor with same shape
        """
        if self.relative:
            # Calculate std of real and imaginary parts
            data_std = torch.std(torch.abs(x))
            noise_std = self.std * data_std
        else:
            noise_std = self.std

        # Create complex noise
        noise_real = torch.randn_like(x.real) * noise_std
        noise_imag = torch.randn_like(x.imag) * noise_std
        noise = torch.complex(noise_real, noise_imag)

        return x + noise


class TimeWarp:
    """
    Apply time warping (speed variation) along the frame axis.

    This resamples the temporal dimension by a random factor to simulate
    heart rate variations. The output length CHANGES - this is the key difference
    from typical augmentations.

    Args:
        warp_factor_range: Tuple (min, max) for random warping factor.
                          1.0 means no change, <1.0 compresses time (faster), >1.0 stretches time (slower).
                          Example: (0.9, 1.1) for ±10% speed variation

    Example:
        Input: (2, 4000, 896) with warp_factor=0.9
        Output: (2, 3600, 896)  # 10% faster (compressed)

        Input: (2, 4000, 896) with warp_factor=1.1
        Output: (2, 4400, 896)  # 10% slower (expanded)
    """
    def __init__(self, warp_factor_range: Tuple[float, float] = (0.95, 1.05)):
        self.warp_factor_range = warp_factor_range

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Complex tensor of shape (2, T, R) where T is time frames, R is range bins
               Applied to full file before segmentation

        Returns:
            Time-warped tensor with CHANGED length: (2, T', R) where T' = int(T * warp_factor)
        """
        # Random warp factor
        warp_factor = np.random.uniform(*self.warp_factor_range)

        # New length after warping
        new_length = int(x.shape[1] * warp_factor)

        # Separate real and imaginary parts for interpolation
        # Shape: (2, T, R) -> (2, R, T) for torch interpolate
        x_real = x.real.permute(0, 2, 1)  # (2, R, T)
        x_imag = x.imag.permute(0, 2, 1)

        # Interpolate to new length along time dimension
        x_real_warped = torch.nn.functional.interpolate(
            x_real.unsqueeze(0),  # Add batch dim: (1, 2, R, T)
            size=(x_real.shape[1], new_length),  # (R, new_length)
            mode='bilinear',
            align_corners=False
        ).squeeze(0)  # Remove batch dim: (2, R, new_length)

        x_imag_warped = torch.nn.functional.interpolate(
            x_imag.unsqueeze(0),
            size=(x_imag.shape[1], new_length),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        # Permute back: (2, R, new_length) -> (2, new_length, R)
        x_real_final = x_real_warped.permute(0, 2, 1)
        x_imag_final = x_imag_warped.permute(0, 2, 1)

        return torch.complex(x_real_final, x_imag_final)


class RangeShift:
    """
    Apply random range shift augmentation for range-gated data.

    This augmentation perturbs the range center index with Gaussian noise
    to make the model robust to range estimation errors.

    Args:
        shift_std: Standard deviation of the Gaussian shift (in range bins)
        max_shift: Maximum absolute shift allowed (prevents extreme shifts)

    Note:
        This augmentation is applied BEFORE range gating in the dataset.
        It returns the shift value to be added to the center index.
    """
    def __init__(self, shift_std: float = 5.0, max_shift: int = 15):
        self.shift_std = shift_std
        self.max_shift = max_shift

    def __call__(self) -> int:
        """
        Returns:
            Integer shift to be added to the range center index
        """
        shift = np.random.normal(0, self.shift_std)
        shift = int(np.round(shift))
        # Clamp to max_shift
        shift = np.clip(shift, -self.max_shift, self.max_shift)
        return shift


class Compose:
    """Compose multiple transforms together."""
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            x = t(x)
        return x


# ============================================================================
# Main Dataset Class
# ============================================================================

class CIRDataset(Dataset):
    """
    PyTorch Dataset for CIR (Channel Impulse Response) data from .npy files.

    This dataset loads complex-valued CIR measurements, segments them into overlapping
    windows, and provides optional data augmentation for training.

    Args:
        yaml_path: Path to YAML configuration file containing file paths and labels
        split: Which data split to use ('Train', 'Val', or 'Test')
        window_size: Number of frames per segment (default: 1500)
        stride: Step size for sliding window (default: 750, i.e., 50% overlap)
        augmentations: Optional list of augmentation transforms to apply (only for training)
        sender_idx: Which sender node to use (0, 1, 2, or None).
                   - If 0, 1, or 2: use only that specific sender node
                   - If None: use segments from all sender nodes (increases dataset size 3x)
                   Default: 0 (first sender)
        range_gated: If True, apply range gating to reduce range dimension (default: False)
        range_window_size: Size of the range window when range_gated=True (default: 64)
        range_index_mapping_path: Path to YAML/JSON file with range center indices per location
        enable_range_shift: If True, apply random range shift augmentation (only for training)
        range_shift_std: Standard deviation of range shift in bins (default: 5.0)
        apply_lowpass_filter: If True, apply Butterworth low-pass filter along slow time axis
        lowpass_cutoff: Cutoff frequency for low-pass filter in Hz (default: 2.0)
        lowpass_order: Order of the Butterworth filter (default: 5)
        lowpass_fs: Sampling frequency in Hz (default: 100.0)
        complex_to_polar: If True, convert complex data to [magnitude, unwrapped_phase] (default: True)
        apply_minmax_norm: If True, apply min-max normalization over (frames, range) dims (default: True)
        cache_size: Number of files to keep in memory cache (default: 10, 0 to disable)
        verbose: If True, print loading information

    Attributes:
        samples: List of tuples (file_path, segment_start_idx, sender_idx, label)
        num_samples: Total number of segments across all files
        range_mapping: Dict mapping location IDs to 3x2 arrays of range center indices

    Output Shape:
        When complex_to_polar=True (default):
            - Output shape: (2, 2, T, R) where:
                - First 2: listening pairs
                - Second 2: [magnitude, unwrapped_phase]
                - T: time frames (window_size, default 1500)
                - R: range bins (896 or range_window_size if range_gated)
            - dtype: float32
        When complex_to_polar=False:
            - Output shape: (2, T, R)
            - dtype: complex64

    Range Gating:
        When range_gated=False (default):
            - Range dimension: 896 bins
        When range_gated=True:
            - Range dimension: range_window_size bins (e.g., 64)
            - Requires range_index_mapping_path to be specified
            - Extracts a window of size range_window_size centered at location-specific indices
    """

    def __init__(
        self,
        yaml_path: str,
        split: str,
        window_size: int = 1500,
        stride: int = 750,
        augmentations: Optional[List[Callable]] = None,
        sender_idx: Optional[int] = 0,
        range_gated: bool = False,
        range_window_size: int = 64,
        range_index_mapping_path: Optional[str] = None,
        enable_range_shift: bool = True,
        range_shift_std: float = 5.0,
        apply_lowpass_filter: bool = False,
        lowpass_cutoff: float = 3.0,
        lowpass_order: int = 5,
        lowpass_fs: float = 100.0,
        complex_to_polar: bool = True,
        apply_minmax_norm: bool = True,
        cache_size: int = 10,
        verbose: bool = False
    ):
        self.yaml_path = Path(yaml_path)
        self.split = split
        self.window_size = window_size
        self.stride = stride
        self.sender_idx = sender_idx
        self.verbose = verbose

        # Range gating parameters
        self.range_gated = range_gated
        self.range_window_size = range_window_size
        self.range_index_mapping_path = range_index_mapping_path
        self.enable_range_shift = enable_range_shift
        self.range_shift_std = range_shift_std

        # Low-pass filter parameters
        self.apply_lowpass_filter = apply_lowpass_filter
        self.lowpass_cutoff = lowpass_cutoff
        self.lowpass_order = lowpass_order
        self.lowpass_fs = lowpass_fs

        # Complex-to-polar and normalization parameters
        self.complex_to_polar = complex_to_polar
        self.apply_minmax_norm = apply_minmax_norm

        # File caching for performance (LRU cache)
        self.cache_size = cache_size
        self.file_cache = {}  # Maps file_path -> preprocessed data
        self.cache_order = []  # LRU tracking

        # Range shift augmentation (only for training when range gating is enabled)
        self.range_shift_aug = None
        if self.range_gated and self.enable_range_shift and self.split == 'Train':
            self.range_shift_aug = RangeShift(shift_std=self.range_shift_std)

        # Augmentation pipeline (for general augmentations like noise, time warp)
        if augmentations is not None and len(augmentations) > 0:
            self.transform = Compose(augmentations)
        else:
            self.transform = None

        # Load range index mapping if range gating is enabled
        self.range_mapping = None
        self.file_location_map = {}  # Maps file_path -> location_id
        if self.range_gated:
            self._load_range_mapping()

        # Load configuration and create segment index
        self._load_config()
        self._create_segment_index()

        if self.verbose:
            if self.sender_idx is None:
                print(f"[{self.split}] Loaded {len(self.samples)} segments from {len(self.file_label_map)} files (all senders)")
            else:
                print(f"[{self.split}] Loaded {len(self.samples)} segments from {len(self.file_label_map)} files (sender {self.sender_idx})")

            if self.range_gated:
                print(f"[{self.split}] Range gating enabled: {self.range_window_size} bins (from 896)")
                if self.range_shift_aug is not None:
                    print(f"[{self.split}] Range shift augmentation enabled (std={self.range_shift_std})")

            if self.apply_lowpass_filter:
                print(f"[{self.split}] Low-pass filter enabled: cutoff={self.lowpass_cutoff} Hz, order={self.lowpass_order}, fs={self.lowpass_fs} Hz")

            if self.cache_size > 0:
                print(f"[{self.split}] File caching enabled: max {self.cache_size} files in memory")

    def _load_range_mapping(self):
        """
        Load range index mapping from YAML/JSON file.

        The mapping file format:
            loc1:
              - [120, 135]   # node 0, 2 listening pairs
              - [200, 210]   # node 1, 2 listening pairs
              - [300, 320]   # node 2, 2 listening pairs
            loc2:
              - [100, 115]
              - [180, 190]
              - [260, 275]

        Each location has a 3x2 array (list of lists) specifying range center indices
        for [node_idx, pair_idx].
        """
        if self.range_index_mapping_path is None:
            raise ValueError(
                "range_gated=True requires range_index_mapping_path to be specified"
            )

        mapping_path = Path(self.range_index_mapping_path)
        if not mapping_path.is_absolute():
            # Make path relative to YAML config file location
            mapping_path = (self.yaml_path.parent / mapping_path).resolve()

        if not mapping_path.exists():
            raise FileNotFoundError(f"Range mapping file not found: {mapping_path}")

        # Load mapping file (supports YAML and JSON)
        with open(mapping_path, 'r') as f:
            if mapping_path.suffix in ['.yaml', '.yml']:
                import yaml
                self.range_mapping = yaml.safe_load(f)
            elif mapping_path.suffix == '.json':
                import json
                self.range_mapping = json.load(f)
            else:
                raise ValueError(
                    f"Unsupported range mapping file format: {mapping_path.suffix}. "
                    f"Supported: .yaml, .yml, .json"
                )

        # Validate mapping structure
        for loc_id, indices in self.range_mapping.items():
            if not isinstance(indices, list):
                raise ValueError(
                    f"Location '{loc_id}' mapping must be a list, got {type(indices)}"
                )
            if len(indices) != 3:
                raise ValueError(
                    f"Location '{loc_id}' must have 3 node entries (got {len(indices)})"
                )
            for node_idx, node_pairs in enumerate(indices):
                if not isinstance(node_pairs, list):
                    raise ValueError(
                        f"Location '{loc_id}' node {node_idx} must be a list, "
                        f"got {type(node_pairs)}"
                    )
                if len(node_pairs) != 2:
                    raise ValueError(
                        f"Location '{loc_id}' node {node_idx} must have 2 pair entries "
                        f"(got {len(node_pairs)})"
                    )

        if self.verbose:
            print(f"[{self.split}] Loaded range mapping with {len(self.range_mapping)} locations: "
                  f"{list(self.range_mapping.keys())}")

    def _load_config(self):
        """Load YAML configuration and extract file paths and labels for the specified split."""
        if not self.yaml_path.exists():
            raise FileNotFoundError(f"YAML config file not found: {self.yaml_path}")

        with open(self.yaml_path, 'r') as f:
            config = yaml.safe_load(f)

        if self.split not in config:
            raise ValueError(f"Split '{self.split}' not found in YAML. Available: {list(config.keys())}")

        split_config = config[self.split]

        # Build mapping: file_path -> label
        self.file_label_map: Dict[Path, int] = {}

        for label_str, file_paths in split_config.items():
            try:
                label = int(label_str)
            except ValueError:
                raise ValueError(f"Label '{label_str}' cannot be converted to integer")

            for file_path in file_paths:
                file_path = Path(file_path)
                if not file_path.is_absolute():
                    # Make path relative to YAML file location
                    file_path = (self.yaml_path.parent / file_path).resolve()

                self.file_label_map[file_path] = label

    def _detect_location(self, file_path: Path) -> Optional[str]:
        """
        Detect location ID from filename by matching location keys in the filename.

        Args:
            file_path: Path to the .npy file

        Returns:
            Location ID (str) if found, None otherwise

        The function checks if any of the location IDs from the range mapping
        appear in the filename string.
        """
        if self.range_mapping is None:
            return None

        filename = file_path.name.lower()  # Case-insensitive matching

        # Try to find a location key that appears in the filename
        matched_locations = []
        for loc_id in self.range_mapping.keys():
            if loc_id.lower() in filename:
                matched_locations.append(loc_id)

        if len(matched_locations) == 0:
            return None
        elif len(matched_locations) == 1:
            return matched_locations[0]
        else:
            # Multiple matches - prefer the longest match (most specific)
            return max(matched_locations, key=len)

    def _create_segment_index(self):
        """
        Create an index of all segments across all files.

        Each entry is a tuple: (file_path, segment_start_idx, sender_idx, label)
        When self.sender_idx is None, segments from all senders are included.

        Note: segment_start_idx is based on ORIGINAL file length. When augmentation
        (especially time warp) is applied, the actual start index will be adjusted
        in __getitem__ based on the warped file length.
        """
        self.samples: List[Tuple[Path, int, int, int]] = []

        for file_path, label in self.file_label_map.items():
            if not file_path.exists():
                warnings.warn(f"File not found, skipping: {file_path}")
                continue

            # Detect location for range gating
            if self.range_gated:
                location_id = self._detect_location(file_path)
                if location_id is None:
                    warnings.warn(
                        f"Could not detect location for {file_path.name} "
                        f"(no matching key from {list(self.range_mapping.keys())}). Skipping."
                    )
                    continue
                self.file_location_map[file_path] = location_id

            try:
                # Load data to get shape
                data = np.load(file_path)  # Shape: (3, 2, X, 896)

                if data.ndim != 4:
                    warnings.warn(f"Unexpected data shape {data.shape} in {file_path}, expected 4D. Skipping.")
                    continue

                num_nodes, num_pairs, num_frames, feat_dim = data.shape

                # Validate shape
                if num_nodes != 3 or num_pairs != 2 or feat_dim != 896:
                    warnings.warn(
                        f"Unexpected shape {data.shape} in {file_path}. "
                        f"Expected (3, 2, X, 896). Skipping."
                    )
                    continue

                # Validate sender_idx if specified
                if self.sender_idx is not None and self.sender_idx >= num_nodes:
                    warnings.warn(
                        f"sender_idx={self.sender_idx} >= num_nodes={num_nodes} "
                        f"in {file_path}. Skipping."
                    )
                    continue

                # Check if file is long enough
                if num_frames < self.window_size:
                    warnings.warn(
                        f"File {file_path} has only {num_frames} frames, "
                        f"less than window_size={self.window_size}. Skipping."
                    )
                    continue

                # Calculate number of segments based on original file length
                num_segments = (num_frames - self.window_size) // self.stride + 1

                # Determine which sender indices to use
                if self.sender_idx is None:
                    # Use all senders
                    sender_indices = list(range(num_nodes))
                else:
                    # Use only the specified sender
                    sender_indices = [self.sender_idx]

                # Add all segments from this file for each sender
                for sender in sender_indices:
                    for seg_idx in range(num_segments):
                        start_idx = seg_idx * self.stride
                        self.samples.append((file_path, start_idx, sender, label))

                if self.verbose:
                    if self.sender_idx is None:
                        print(f"  {file_path.name}: {num_frames} frames -> {num_segments} segments x {num_nodes} senders = {num_segments * num_nodes} total")
                    else:
                        print(f"  {file_path.name}: {num_frames} frames -> {num_segments} segments")

            except Exception as e:
                warnings.warn(f"Error loading {file_path}: {e}. Skipping.")
                continue

    def __len__(self) -> int:
        """Return total number of segments."""
        return len(self.samples)

    def _load_and_preprocess_file(self, file_path: Path) -> np.ndarray:
        """
        Load and preprocess a file (with caching).

        This method loads the file and applies preprocessing that doesn't change
        across multiple segments (low-pass filtering).

        Args:
            file_path: Path to the .npy file

        Returns:
            Preprocessed data of shape (3, 2, X, 896)
        """
        # Check cache first
        if self.cache_size > 0 and file_path in self.file_cache:
            # Update LRU order
            self.cache_order.remove(file_path)
            self.cache_order.append(file_path)
            return self.file_cache[file_path]

        # Load the full file
        data = np.load(file_path)  # Shape: (3, 2, X, 896)

        # Apply low-pass filter if enabled
        # Filter is applied BEFORE segmentation to avoid edge effects at segment boundaries
        if self.apply_lowpass_filter:
            data = butter_lowpass_filter(
                data,
                cutoff=self.lowpass_cutoff,
                fs=self.lowpass_fs,
                order=self.lowpass_order,
                axis=2  # Slow time axis (frames dimension)
            )

        # Add to cache if caching is enabled
        if self.cache_size > 0:
            # Evict oldest entry if cache is full
            if len(self.file_cache) >= self.cache_size:
                oldest_file = self.cache_order.pop(0)
                del self.file_cache[oldest_file]

            # Add to cache
            self.file_cache[file_path] = data
            self.cache_order.append(file_path)

        return data

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single segment.

        Args:
            idx: Index of the segment

        Returns:
            Tuple of (segment_tensor, label) where:
                - segment_tensor: torch.Tensor with shape:
                    - (2, 2, T, R) if complex_to_polar=True (default)
                        where dim 1 is [magnitude, unwrapped_phase]
                    - (2, T, R) if complex_to_polar=False
                - dtype: float32 (if complex_to_polar=True) or complex64
                - label: int (0, 1, or higher)
        """
        file_path, start_idx, sender_idx, label = self.samples[idx]

        # Load and preprocess file (with caching)
        data = self._load_and_preprocess_file(file_path)

        # Extract data for the specified sender BEFORE augmentation
        # This allows time warp to operate on the full temporal sequence
        # Shape: (2, X, 896) where X is the full length
        sender_data = data[sender_idx, :, :, :]

        # Store original length before augmentation
        original_length = sender_data.shape[1]

        # Apply augmentations to the FULL sender data before segmentation
        # This is critical for time warp to work properly
        if self.transform is not None:
            # Convert to torch tensor for augmentation
            sender_data_tensor = torch.from_numpy(sender_data).to(torch.complex64)
            # Apply augmentations (time warp, Gaussian noise, etc.)
            sender_data_tensor = self.transform(sender_data_tensor)
            # Convert back to numpy
            sender_data = sender_data_tensor.cpu().numpy()

        # Now extract the segment from the augmented data
        # Note: After time warp, the length may have changed!
        current_length = sender_data.shape[1]

        # Adjust segment index based on the length change from time warp
        if current_length != original_length:
            # Time warp changed the length - adjust the segment index proportionally
            # scale_factor = new_length / original_length
            scale_factor = current_length / original_length
            adjusted_start_idx = int(start_idx * scale_factor)

            # Ensure we have enough frames for a complete window
            if adjusted_start_idx + self.window_size > current_length:
                # Adjust to extract from the end if needed
                adjusted_start_idx = max(0, current_length - self.window_size)

            start_idx = adjusted_start_idx
        else:
            # No length change (no time warp or warp_factor=1.0)
            # Check if pre-computed index is still valid
            if start_idx + self.window_size > current_length:
                # If not enough frames, extract from the end
                start_idx = max(0, current_length - self.window_size)

        # Extract the segment
        # We select:
        # - [:]: both listening pairs (2 listeners)
        # - [start_idx:start_idx+window_size]: the time window
        # - [:]: all features (896)
        segment = sender_data[:, start_idx:start_idx+self.window_size, :]
        # Shape: (2, 1500, 896)

        # Apply range gating if enabled
        if self.range_gated:
            # Get location-specific range indices
            location_id = self.file_location_map[file_path]
            range_indices = self.range_mapping[location_id][sender_idx]  # Shape: (2,) for 2 pairs

            # Apply range gating to each listening pair
            gated_pairs = []
            for pair_idx in range(2):
                # Get center index for this pair
                center = range_indices[pair_idx]

                # Apply range shift augmentation if enabled (training only)
                if self.range_shift_aug is not None:
                    shift = self.range_shift_aug()
                    center = center + shift

                # Calculate range window boundaries
                # Centered window: [center - window_size//2, center + window_size//2)
                half_window = self.range_window_size // 2
                start_range = center - half_window
                end_range = start_range + self.range_window_size

                # Handle boundaries by clamping
                # Strategy: Clamp to valid range [0, 896), maintaining window size
                if start_range < 0:
                    start_range = 0
                    end_range = self.range_window_size
                elif end_range > 896:
                    end_range = 896
                    start_range = 896 - self.range_window_size

                # Final clamp to ensure valid indices
                start_range = max(0, start_range)
                end_range = min(896, end_range)

                # Extract range window for this pair
                # segment shape: (2, 1500, 896)
                # pair_data shape: (1500, range_window_size)
                pair_data = segment[pair_idx, :, start_range:end_range]
                gated_pairs.append(pair_data)

            # Stack the two pairs back together
            # Result shape: (2, 1500, range_window_size)
            segment = np.stack(gated_pairs, axis=0)

        # Convert to torch tensor (complex64 for efficiency)
        segment_tensor = torch.from_numpy(segment).to(torch.complex64)

        # Note: Augmentations (time warp, Gaussian noise) have already been applied
        # to the full file before segmentation (see above)

        # Convert complex to polar (magnitude + unwrapped phase)
        if self.complex_to_polar:
            # Compute magnitude and phase for each listening pair
            # segment_tensor shape: (2, T, R) complex64
            magnitude = torch.abs(segment_tensor)  # Shape: (2, T, R)
            phase = torch.angle(segment_tensor)    # Shape: (2, T, R)

            # Unwrap phase along range axis (axis=-1) for each pair
            # Optimized: Process all pairs at once, keep on same device
            phase_np = phase.numpy()  # Shape: (2, T, R)
            # Unwrap along range axis (last dimension) for all pairs simultaneously
            phase_unwrapped_np = np.unwrap(phase_np, axis=-1)  # Unwrap along range axis
            phase_unwrapped = torch.from_numpy(phase_unwrapped_np)

            # Stack magnitude and unwrapped phase
            # Shape: (2, 2, T, R) where dim 1 is [magnitude, phase]
            segment_tensor = torch.stack([magnitude, phase_unwrapped], dim=1).float()

            # Apply min-max normalization over (T, R) dimensions for each pair and channel
            if self.apply_minmax_norm:
                # segment_tensor shape: (2, 2, T, R)
                # Vectorized normalization: normalize each (pair, channel) independently
                # Reshape to (2, 2, T*R) for easier min/max computation
                original_shape = segment_tensor.shape
                T, R = original_shape[2], original_shape[3]
                segment_flat = segment_tensor.reshape(2, 2, -1)  # Shape: (2, 2, T*R)

                # Compute min and max along the spatial-temporal dimension
                data_min = segment_flat.min(dim=-1, keepdim=True)[0]  # Shape: (2, 2, 1)
                data_max = segment_flat.max(dim=-1, keepdim=True)[0]  # Shape: (2, 2, 1)

                # Compute range and handle constant values
                data_range = data_max - data_min
                data_range = torch.where(data_range > 1e-8, data_range, torch.ones_like(data_range))

                # Normalize
                segment_flat = (segment_flat - data_min) / data_range

                # Reshape back to original shape
                segment_tensor = segment_flat.reshape(original_shape)

        return segment_tensor, label


# ============================================================================
# Helper Functions
# ============================================================================

def create_augmentations(aug_list: Optional[List[str]] = None) -> Optional[List[Callable]]:
    """
    Create augmentation transforms from a list of augmentation names.

    Args:
        aug_list: List of augmentation names. Supported:
            - 'gaussian_noise': Add Gaussian noise
            - 'time_warp': Apply time warping
            - None or []: No augmentations

    Returns:
        List of augmentation callables, or None if no augmentations

    Example:
        >>> augs = create_augmentations(['gaussian_noise', 'time_warp'])
    """
    if aug_list is None or len(aug_list) == 0:
        return None

    transforms = []

    for aug_name in aug_list:
        if aug_name == 'gaussian_noise':
            transforms.append(GaussianNoise(std=0.01, relative=True))
        elif aug_name == 'time_warp':
            transforms.append(TimeWarp(warp_factor_range=(0.95, 1.05)))
        else:
            warnings.warn(f"Unknown augmentation '{aug_name}', ignoring.")

    return transforms if len(transforms) > 0 else None


def get_dataloaders(
    yaml_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    augmentations_train: Optional[List[str]] = None,
    window_size: int = 1536,
    stride: int = 768,
    sender_idx: Optional[int] = 0,
    range_gated: bool = False,
    range_window_size: int = 64,
    range_index_mapping_path: Optional[str] = None,
    enable_range_shift: bool = True,
    range_shift_std: float = 5.0,
    apply_lowpass_filter: bool = False,
    lowpass_cutoff: float = 3.0,
    lowpass_order: int = 5,
    lowpass_fs: float = 100.0,
    complex_to_polar: bool = True,
    apply_minmax_norm: bool = True,
    cache_size: int = 10,
    shuffle_train: bool = True,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: Optional[int] = 4,
    verbose: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for train, validation, and test splits.

    Args:
        yaml_path: Path to YAML configuration file
        batch_size: Batch size for all DataLoaders
        num_workers: Number of worker processes for data loading
        augmentations_train: List of augmentation names for training data only
                           (e.g., ['gaussian_noise', 'time_warp'])
        window_size: Number of frames per segment (default: 1500)
        stride: Step size for sliding window (default: 750)
        sender_idx: Which sender node to use (0, 1, 2, or None).
                   - If 0, 1, or 2: use only that specific sender node
                   - If None: use segments from all sender nodes (increases dataset size 3x)
                   Default: 0
        range_gated: If True, apply range gating to reduce range dimension
        range_window_size: Size of the range window when range_gated=True (default: 64)
        range_index_mapping_path: Path to YAML/JSON file with range center indices per location
        enable_range_shift: If True, apply random range shift augmentation (training only)
        range_shift_std: Standard deviation of range shift in bins (default: 5.0)
        apply_lowpass_filter: If True, apply Butterworth low-pass filter along slow time axis
        lowpass_cutoff: Cutoff frequency for low-pass filter in Hz (default: 2.0)
        lowpass_order: Order of the Butterworth filter (default: 5)
        lowpass_fs: Sampling frequency in Hz (default: 100.0)
        complex_to_polar: If True, convert complex data to [magnitude, unwrapped_phase] (default: True)
        apply_minmax_norm: If True, apply min-max normalization over (frames, range) dims (default: True)
        cache_size: Number of files to keep in memory cache (default: 10, 0 to disable)
        shuffle_train: Whether to shuffle training data
        pin_memory: Whether to pin memory for faster GPU transfer
        persistent_workers: Keep worker processes alive between epochs (default: True)
        prefetch_factor: Number of batches to prefetch per worker (default: 4, None to disable)
        verbose: If True, print dataset information

    Returns:
        Tuple of (train_loader, val_loader, test_loader)

    Example:
        >>> # Use only sender 0
        >>> train_loader, val_loader, test_loader = get_dataloaders(
        ...     yaml_path='config.yaml',
        ...     batch_size=32,
        ...     num_workers=4,
        ...     sender_idx=0,
        ...     augmentations_train=['gaussian_noise', 'time_warp']
        ... )
        >>>
        >>> # Use all senders (3x more data)
        >>> train_loader, val_loader, test_loader = get_dataloaders(
        ...     yaml_path='config.yaml',
        ...     batch_size=32,
        ...     sender_idx=None
        ... )
        >>>
        >>> # Use range gating
        >>> train_loader, val_loader, test_loader = get_dataloaders(
        ...     yaml_path='config.yaml',
        ...     batch_size=32,
        ...     range_gated=True,
        ...     range_window_size=64,
        ...     range_index_mapping_path='range_mapping.yaml',
        ...     augmentations_train=['gaussian_noise']
        ... )
    """
    # Create augmentations for training
    train_augs = create_augmentations(augmentations_train)

    # Create datasets
    train_dataset = CIRDataset(
        yaml_path=yaml_path,
        split='Train',
        window_size=window_size,
        stride=stride,
        augmentations=train_augs,
        sender_idx=sender_idx,
        range_gated=range_gated,
        range_window_size=range_window_size,
        range_index_mapping_path=range_index_mapping_path,
        enable_range_shift=enable_range_shift,
        range_shift_std=range_shift_std,
        apply_lowpass_filter=apply_lowpass_filter,
        lowpass_cutoff=lowpass_cutoff,
        lowpass_order=lowpass_order,
        lowpass_fs=lowpass_fs,
        complex_to_polar=complex_to_polar,
        apply_minmax_norm=apply_minmax_norm,
        cache_size=cache_size,
        verbose=verbose
    )

    val_dataset = CIRDataset(
        yaml_path=yaml_path,
        split='Val',
        window_size=window_size,
        stride=stride,
        augmentations=None,  # No augmentations for validation
        sender_idx=sender_idx,
        range_gated=range_gated,
        range_window_size=range_window_size,
        range_index_mapping_path=range_index_mapping_path,
        enable_range_shift=False,  # No range shift for validation
        range_shift_std=range_shift_std,
        apply_lowpass_filter=apply_lowpass_filter,
        lowpass_cutoff=lowpass_cutoff,
        lowpass_order=lowpass_order,
        lowpass_fs=lowpass_fs,
        complex_to_polar=complex_to_polar,
        apply_minmax_norm=apply_minmax_norm,
        cache_size=cache_size,
        verbose=verbose
    )

    test_dataset = CIRDataset(
        yaml_path=yaml_path,
        split='Test',
        window_size=window_size,
        stride=stride,
        augmentations=None,  # No augmentations for test
        sender_idx=sender_idx,
        range_gated=range_gated,
        range_window_size=range_window_size,
        range_index_mapping_path=range_index_mapping_path,
        enable_range_shift=False,  # No range shift for test
        range_shift_std=range_shift_std,
        apply_lowpass_filter=apply_lowpass_filter,
        lowpass_cutoff=lowpass_cutoff,
        lowpass_order=lowpass_order,
        lowpass_fs=lowpass_fs,
        complex_to_polar=complex_to_polar,
        apply_minmax_norm=apply_minmax_norm,
        cache_size=cache_size,
        verbose=verbose
    )

    # DataLoader kwargs (persistent_workers requires num_workers > 0)
    dataloader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
    }

    # Add persistent_workers and prefetch_factor only if num_workers > 0
    if num_workers > 0:
        dataloader_kwargs['persistent_workers'] = persistent_workers
        if prefetch_factor is not None:
            dataloader_kwargs['prefetch_factor'] = prefetch_factor

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        shuffle=shuffle_train,
        **dataloader_kwargs
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **dataloader_kwargs
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **dataloader_kwargs
    )

    return train_loader, val_loader, test_loader


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == '__main__':
    """
    Example usage of the CIRDataset and DataLoader creation.
    """

    # Example 1: Create individual dataset
    print("Example 1: Creating individual dataset")
    print("-" * 50)

    dataset = CIRDataset(
        yaml_path='src/data/data_config.yaml',
        split='Train',
        augmentations=create_augmentations(['gaussian_noise', 'time_warp']),
        verbose=True,
        window_size=1536,
        stride=768,
        sender_idx=0,
        range_gated=False,
        range_index_mapping_path='example_range_mapping.yaml',
        range_window_size=64,
        complex_to_polar=True,
        apply_minmax_norm=True
    )

    print(f"Dataset size: {len(dataset)} segments")

    if len(dataset) > 0:
        sample_data, sample_label = dataset[0]
        print(f"Sample shape: {sample_data.shape}")  # (2 RX nodes, 2 (mag+angle), 1500 frames, 896 bins)
        print(f"Sample dtype: {sample_data.dtype}")  # float32
        print(f"Sample label: {sample_label}")

    # Example 2: Create all DataLoaders at once
    print("Example 2: Creating DataLoaders")
    print("-" * 50)

    train_loader, val_loader, test_loader = get_dataloaders(
        yaml_path='src/data/data_config.yaml',
        batch_size=16,
        num_workers=2,
        augmentations_train=['gaussian_noise', 'time_warp'],
        verbose=True
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Iterate through one batch
    if len(train_loader) > 0:
        for batch_data, batch_labels in train_loader:
            print(f"\nBatch data shape: {batch_data.shape}")  # (batch_size, 2, 2, 1500, 896)
            print(f"Batch labels shape: {batch_labels.shape}")  # (batch_size,)
            print(f"Batch data dtype: {batch_data.dtype}")  # float32
            print(f"Batch labels: {batch_labels}")
            break
