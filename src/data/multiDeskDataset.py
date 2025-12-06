# =============================================================================
# PyTorch Dataset for Multi-Desk UWB Radar Data
# =============================================================================

"""
YAML Label File Structure (example: dataset_labels.yaml):
---------------------------------------------------------
# Number of desks (for validation)
num_desks: 2

# Data splits
train:
  - filename: "data_01_range_gated.npy"
    labels: [0, 1]  # Desk 0: class 0, Desk 1: class 1
  - filename: "data_02_range_gated.npy"
    labels: [1, 0]

val:
  - filename: "data_03_range_gated.npy"
    labels: [0, 0]

test:
  - filename: "data_04_range_gated.npy"
    labels: [1, 1]
"""

import torch
from torch.utils.data import Dataset
from typing import Optional, List, Tuple, Union
from dataclasses import dataclass
import numpy as np
import yaml
from pathlib import Path
from scipy.interpolate import interp1d


# =============================================================================
# Augmentation Functions
# =============================================================================

def apply_gaussian_noise(data: np.ndarray, noise_std: float = 0.01) -> np.ndarray:
    """
    Add Gaussian noise to data.

    Args:
        data: Input array of any shape
        noise_std: Standard deviation of noise relative to data std

    Returns:
        Data with added Gaussian noise
    """
    noise = np.random.randn(*data.shape).astype(data.dtype) * noise_std * np.std(data)
    return data + noise


def apply_time_warp(data: np.ndarray, sigma: float = 0.2, num_knots: int = 4) -> np.ndarray:
    """
    Apply time warping augmentation (speed variation) using smooth random warping.

    Args:
        data: Input array of shape (T, ...) where T is the time dimension
        sigma: Standard deviation for random warp magnitude
        num_knots: Number of knots for the warping spline

    Returns:
        Time-warped data with same shape as input
    """
    T = data.shape[0]
    orig_steps = np.arange(T)

    # Generate random warp path using cubic spline
    knot_positions = np.linspace(0, T - 1, num_knots + 2)
    knot_values = np.linspace(0, T - 1, num_knots + 2)

    # Add random perturbations to interior knots
    knot_values[1:-1] += np.random.randn(num_knots) * sigma * T / num_knots

    # Ensure monotonicity by sorting and clipping
    knot_values = np.sort(knot_values)
    knot_values = np.clip(knot_values, 0, T - 1)

    # Create smooth warping function
    warp_func = interp1d(knot_positions, knot_values, kind='cubic', fill_value='extrapolate')
    warped_steps = warp_func(orig_steps)

    # Clip to valid range
    warped_steps = np.clip(warped_steps, 0, T - 1)

    # Interpolate data along time axis
    # Flatten all non-time dimensions for efficient interpolation
    original_shape = data.shape
    data_flat = data.reshape(T, -1)
    num_features = data_flat.shape[1]

    warped_data = np.zeros_like(data_flat)
    for i in range(num_features):
        interp_func = interp1d(orig_steps, data_flat[:, i], kind='linear', fill_value='extrapolate')
        warped_data[:, i] = interp_func(warped_steps)

    return warped_data.reshape(original_shape)


def min_max_normalize(data: np.ndarray, feature_range: Tuple[float, float] = (0, 1)) -> np.ndarray:
    """
    Apply per-feature min-max normalization to data.

    For CIR data with shape (T, 3, 2), each of the 3*2=6 features (time series)
    is normalized independently along the time axis.

    Args:
        data: Input array of shape (T, ...) where T is the time dimension
        feature_range: Target range (min, max)

    Returns:
        Normalized data in feature_range, with each (T,) feature normalized separately
    """
    T = data.shape[0]
    original_shape = data.shape

    # Reshape to (T, num_features) for per-feature normalization
    data_flat = data.reshape(T, -1)
    num_features = data_flat.shape[1]

    result = np.empty_like(data_flat)

    for f in range(num_features):
        feature_data = data_flat[:, f]
        min_val = np.min(feature_data)
        max_val = np.max(feature_data)

        if max_val - min_val < 1e-8:
            result[:, f] = feature_range[0]
        else:
            scale = (feature_range[1] - feature_range[0]) / (max_val - min_val)
            result[:, f] = (feature_data - min_val) * scale + feature_range[0]

    return result.reshape(original_shape)


@dataclass
class SampleMetadata:
    """Metadata for a single dataset sample."""
    file_path: str
    start_t: int
    desk_id: int
    sender_id: int
    label: int
    layout: str = ""  # "Horizontal" or "Vertical" for single_antenna mode


class MultiDeskCIRDataset(Dataset):
    """
    PyTorch Dataset for Multi-Desk UWB Radar Time-Series Data.

    Input Data Shape: (T, 4, 3, 2, N*70)
        - T: Time steps
        - 4: Senders (Nodes)
        - 3: Receivers
        - 2: Amp/Phase channels
        - N*70: Range bins (N desks x 70 bins each)

    Output Shape:
        - Default: (window_size, 3, 2) after mean reduction over range bins
        - single_antenna=True: (window_size, 2) - specific antenna pairs based on layout

    Args:
        data_dir: Directory containing .npy files
        label_yaml_path: Path to YAML file with labels and split info
        split: One of 'train', 'val', 'test'
        window_size: Sliding window size in time dimension (default: 1500)
        stride: Stride for sliding window (default: 750)
        sender_idx: Specific sender index (0-3) or None for all senders
        is_train: Whether to apply training augmentations
        bins_per_desk: Number of range bins per desk (default: 70)
        crop_size: Final number of bins after cropping (default: 64)
        use_minmax_norm: Whether to apply min-max normalization (default: False)
        norm_range: Target range for min-max normalization (default: (0, 1))
        use_gaussian_noise: Whether to add Gaussian noise augmentation (default: False)
        noise_std: Standard deviation for Gaussian noise (default: 0.01)
        use_time_warp: Whether to apply time warping augmentation (default: False)
        time_warp_sigma: Magnitude of time warping (default: 0.2)
        time_warp_knots: Number of knots for time warping spline (default: 4)
        single_antenna: Whether to extract single antenna per sender (default: False)
            - Horizontal: sender0->receiver1, sender1->receiver2
            - Vertical: sender0->receiver0, sender2->receiver2
    """

    # Single antenna extraction mapping: layout -> {sender_id: receiver_idx}
    SINGLE_ANTENNA_MAP = {
        "Horizontal": {0: 1, 1: 2},  # sender0->node2, sender1->node3
        "Vertical": {0: 0, 2: 2},    # sender0->node1, sender2->node3
    }

    def __init__(
        self,
        data_dir: str,
        label_yaml_path: str,
        split: str = 'train',
        window_size: int = 1500,
        stride: int = 750,
        sender_idx: Optional[int] = None,
        is_train: bool = True,
        bins_per_desk: int = 70,
        crop_size: int = 64,
        # Normalization
        use_minmax_norm: bool = False,
        norm_range: Tuple[float, float] = (0, 1),
        # Augmentation
        use_gaussian_noise: bool = False,
        noise_std: float = 0.01,
        use_time_warp: bool = False,
        time_warp_sigma: float = 0.2,
        time_warp_knots: int = 4,
        # Single antenna mode
        single_antenna: bool = False,
    ):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.stride = stride
        self.sender_idx = sender_idx
        self.is_train = is_train
        self.bins_per_desk = bins_per_desk
        self.crop_size = crop_size
        self.crop_margin = bins_per_desk - crop_size  # 6 for default values
        self.single_antenna = single_antenna

        # Normalization settings
        self.use_minmax_norm = use_minmax_norm
        self.norm_range = norm_range

        # Augmentation settings (only applied when is_train=True)
        self.use_gaussian_noise = use_gaussian_noise
        self.noise_std = noise_std
        self.use_time_warp = use_time_warp
        self.time_warp_sigma = time_warp_sigma
        self.time_warp_knots = time_warp_knots

        # Validate sender_idx
        if sender_idx is not None and not (0 <= sender_idx <= 3):
            raise ValueError(f"sender_idx must be None or in range [0, 3], got {sender_idx}")

        # Load label configuration
        with open(label_yaml_path, 'r') as f:
            label_config = yaml.safe_load(f)

        if split not in label_config:
            raise ValueError(f"Split '{split}' not found in label YAML. Available: {list(label_config.keys())}")

        # Build sample index
        self.samples: List[SampleMetadata] = []
        self._build_sample_index(label_config[split])

    def _build_sample_index(self, split_config: List[dict]) -> None:
        """
        Build the expanded sample index from configuration.

        For each file, creates entries for all combinations of:
        - Valid time windows
        - Desks (0 to N-1)
        - Senders (all 4 or just sender_idx, or layout-specific for single_antenna)
        """
        for file_entry in split_config:
            filename = file_entry['filename']
            labels = file_entry['labels']
            file_path = self.data_dir / filename

            if not file_path.exists():
                raise FileNotFoundError(f"Data file not found: {file_path}")

            # Detect layout from filename for single_antenna mode
            layout = ""
            if self.single_antenna:
                if "Horizontal" in filename:
                    layout = "Horizontal"
                elif "Vertical" in filename:
                    layout = "Vertical"
                else:
                    raise ValueError(
                        f"File {filename}: single_antenna=True requires 'Horizontal' or "
                        f"'Vertical' in filename to determine antenna mapping"
                    )

            # Memory-map file to read shape without loading
            data = np.load(file_path, mmap_mode='r')
            T, num_senders, num_receivers, num_channels, total_bins = data.shape

            # Determine number of desks dynamically
            num_desks = total_bins // self.bins_per_desk
            if total_bins % self.bins_per_desk != 0:
                raise ValueError(
                    f"File {filename}: total_bins ({total_bins}) not divisible by "
                    f"bins_per_desk ({self.bins_per_desk})"
                )

            # Validate labels match number of desks
            if len(labels) != num_desks:
                raise ValueError(
                    f"File {filename}: label count ({len(labels)}) doesn't match "
                    f"detected num_desks ({num_desks})"
                )

            # Calculate valid time windows
            num_windows = max(0, (T - self.window_size) // self.stride + 1)

            # Determine which senders to use
            if self.single_antenna:
                # Use only the senders defined in the antenna map for this layout
                sender_list = list(self.SINGLE_ANTENNA_MAP[layout].keys())
            elif self.sender_idx is not None:
                sender_list = [self.sender_idx]
            else:
                sender_list = list(range(4))

            # Expansion loop: create entries for all combinations
            for window_idx in range(num_windows):
                start_t = window_idx * self.stride

                for desk_id in range(num_desks):
                    label = labels[desk_id]

                    for sender_id in sender_list:
                        self.samples.append(SampleMetadata(
                            file_path=str(file_path),
                            start_t=start_t,
                            desk_id=desk_id,
                            sender_id=sender_id,
                            label=label,
                            layout=layout,
                        ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Load and process a single sample.

        Processing pipeline:
            1. Load data (mmap)
            2. Slice time window, sender, desk
            3. Single antenna extraction (if enabled)
            4. Random/center crop (70 -> 64 bins)
            5. Mean reduction over range bins
            6. Time warping (if enabled, train only)
            7. Gaussian noise (if enabled, train only)
            8. Min-max normalization (if enabled)

        Returns:
            Tuple of (data_tensor, label)
            - data_tensor: shape (window_size, 3, 2) or (window_size, 2) for single_antenna
            - label: integer class label
        """
        meta = self.samples[idx]

        # Lazy load with memory mapping
        data = np.load(meta.file_path, mmap_mode='r')

        # Slice time window: (window_size, 4, 3, 2, N*70)
        time_slice = data[meta.start_t : meta.start_t + self.window_size]

        # Slice sender: (window_size, 3, 2, N*70)
        sender_slice = time_slice[:, meta.sender_id, :, :, :]

        # Slice desk bins: (window_size, 3, 2, 70)
        bin_start = meta.desk_id * self.bins_per_desk
        bin_end = bin_start + self.bins_per_desk
        desk_slice = sender_slice[:, :, :, bin_start:bin_end]

        # Single antenna extraction: (window_size, 2, 70)
        if self.single_antenna:
            receiver_idx = self.SINGLE_ANTENNA_MAP[meta.layout][meta.sender_id]
            desk_slice = desk_slice[:, receiver_idx:receiver_idx+1, :, :]  # (window_size, 1, 2, 70)

        # Random crop augmentation (70 -> 64 bins)
        if self.is_train:
            crop_start = np.random.randint(0, self.crop_margin + 1)  # 0 to 6 inclusive
        else:
            crop_start = self.crop_margin // 2  # Center crop: 3

        cropped = desk_slice[:, :, :, crop_start : crop_start + self.crop_size]

        # Mean reduction over range bins: (window_size, 3, 2) or (window_size, 1, 2)
        reduced = np.mean(cropped, axis=-1, dtype=np.float32)

        # For single_antenna mode, squeeze the receiver dimension: (window_size, 2)
        if self.single_antenna:
            reduced = reduced.squeeze(axis=1)

        # Copy to detach from mmap before augmentation
        reduced = reduced.copy()

        # Apply augmentations (only during training)
        if self.is_train:
            # Time warping (speed variation)
            if self.use_time_warp:
                reduced = apply_time_warp(
                    reduced,
                    sigma=self.time_warp_sigma,
                    num_knots=self.time_warp_knots
                )

            # Gaussian noise
            if self.use_gaussian_noise:
                reduced = apply_gaussian_noise(reduced, noise_std=self.noise_std)

        # Min-max normalization (applied to both train and eval)
        if self.use_minmax_norm:
            reduced = min_max_normalize(reduced, feature_range=self.norm_range)
        
        # perm = np.random.permutation(3)
        # reduced = reduced[:, perm, :]
        # reduced = reduced.reshape(1, self.window_size, 3*2)
        # reduced = reduced.transpose(2, 0, 1)
        
        # Convert to tensor
        tensor = torch.from_numpy(reduced).float()

        return tensor, meta.label


# =============================================================================
# Full Range CIR Dataset (768 bins with windowed averaging)
# =============================================================================

@dataclass
class FullRangeSampleMetadata:
    """Metadata for a full range dataset sample."""
    file_path: str
    start_t: int
    sender_id: int
    labels: List[int]  # Multi-label: [desk0_label, desk1_label, ...]
    layout: str = ""  # "Horizontal" or "Vertical" for single_antenna mode


class FullRangeCIRDataset(Dataset):
    """
    PyTorch Dataset for Full Range UWB Radar Time-Series Data.

    Input Data Shape: (T, 4, 3, 2, 768)
        - T: Time steps
        - 4: Senders (Nodes)
        - 3: Receivers
        - 2: Amp/Phase channels
        - 768: Full range bins

    Processing:
        1. Average across range bins with window_size=32, stride=32 -> 24 range windows
        2. Extract single antenna pair based on layout
        3. Apply augmentations and normalization

    Output Shape: (T_window, 2, 24) for single_antenna mode
        - T_window: Time window size
        - 2: Amp/Phase channels
        - 24: Range windows (768 / 32)

    Args:
        data_dir: Directory containing .npy files
        label_yaml_path: Path to YAML file with labels and split info
        split: One of 'train', 'val', 'test'
        window_size: Sliding window size in time dimension (default: 1500)
        stride: Stride for sliding window in time (default: 750)
        range_window: Window size for averaging range bins (default: 32)
        range_stride: Stride for range bin windowing (default: 32)
        is_train: Whether to apply training augmentations
        use_minmax_norm: Whether to apply min-max normalization (default: True)
        norm_range: Target range for min-max normalization (default: (0, 1))
        use_gaussian_noise: Whether to add Gaussian noise augmentation (default: False)
        noise_std: Standard deviation for Gaussian noise (default: 0.01)
        use_time_warp: Whether to apply time warping augmentation (default: False)
        time_warp_sigma: Magnitude of time warping (default: 0.2)
        time_warp_knots: Number of knots for time warping spline (default: 4)
    """

    # Single antenna extraction mapping: layout -> {sender_id: receiver_idx}
    SINGLE_ANTENNA_MAP = {
        "Horizontal": {0: 1, 1: 2},  # sender0->rx1, sender1->rx2
        "Vertical": {0: 0, 2: 2},    # sender0->rx0, sender2->rx2
    }

    def __init__(
        self,
        data_dir: str,
        label_yaml_path: str,
        split: str = 'train',
        window_size: int = 1500,
        stride: int = 750,
        range_window: int = 32,
        range_stride: int = 32,
        is_train: bool = True,
        # Normalization
        use_minmax_norm: bool = True,
        norm_range: Tuple[float, float] = (0, 1),
        # Augmentation
        use_gaussian_noise: bool = False,
        noise_std: float = 0.01,
        use_time_warp: bool = False,
        time_warp_sigma: float = 0.2,
        time_warp_knots: int = 4,
    ):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.stride = stride
        self.range_window = range_window
        self.range_stride = range_stride
        self.is_train = is_train

        # Normalization settings
        self.use_minmax_norm = use_minmax_norm
        self.norm_range = norm_range

        # Augmentation settings (only applied when is_train=True)
        self.use_gaussian_noise = use_gaussian_noise
        self.noise_std = noise_std
        self.use_time_warp = use_time_warp
        self.time_warp_sigma = time_warp_sigma
        self.time_warp_knots = time_warp_knots

        # Load label configuration
        with open(label_yaml_path, 'r') as f:
            label_config = yaml.safe_load(f)

        if split not in label_config:
            raise ValueError(f"Split '{split}' not found in label YAML. Available: {list(label_config.keys())}")

        # Build sample index
        self.samples: List[FullRangeSampleMetadata] = []
        self._build_sample_index(label_config[split])

    def _build_sample_index(self, split_config: List[dict]) -> None:
        """
        Build the expanded sample index from configuration.

        For each file, creates entries for all combinations of:
        - Valid time windows
        - Senders (based on layout-specific antenna mapping)
        """
        for file_entry in split_config:
            filename = file_entry['filename']
            labels = file_entry['labels']  # Multi-label: [desk0, desk1, ...]
            file_path = self.data_dir / filename

            if not file_path.exists():
                raise FileNotFoundError(f"Data file not found: {file_path}")

            # Detect layout from filename
            if "Horizontal" in filename:
                layout = "Horizontal"
            elif "Vertical" in filename:
                layout = "Vertical"
            else:
                raise ValueError(
                    f"File {filename}: requires 'Horizontal' or 'Vertical' in filename"
                )

            # Memory-map file to read shape without loading
            data = np.load(file_path, mmap_mode='r')
            T = data.shape[0]

            # Calculate valid time windows
            num_windows = max(0, (T - self.window_size) // self.stride + 1)

            # Use only the senders defined in the antenna map for this layout
            sender_list = list(self.SINGLE_ANTENNA_MAP[layout].keys())

            # Expansion loop: create entries for all combinations
            for window_idx in range(num_windows):
                start_t = window_idx * self.stride

                for sender_id in sender_list:
                    self.samples.append(FullRangeSampleMetadata(
                        file_path=str(file_path),
                        start_t=start_t,
                        sender_id=sender_id,
                        labels=labels,
                        layout=layout,
                    ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and process a single sample.

        Processing pipeline:
            1. Load data (mmap)
            2. Slice time window, sender
            3. Single antenna extraction
            4. Windowed averaging over range bins (768 -> 24)
            5. Time warping (if enabled, train only)
            6. Gaussian noise (if enabled, train only)
            7. Min-max normalization (if enabled)

        Returns:
            Tuple of (data_tensor, label_tensor)
            - data_tensor: shape (window_size, 2, num_range_windows)
            - label_tensor: shape (num_desks,) with binary labels
        """
        meta = self.samples[idx]

        # Lazy load with memory mapping
        data = np.load(meta.file_path, mmap_mode='r')

        # Slice time window: (window_size, 4, 3, 2, 768)
        time_slice = data[meta.start_t : meta.start_t + self.window_size]

        # Slice sender: (window_size, 3, 2, 768)
        sender_slice = time_slice[:, meta.sender_id, :, :, :]

        # Single antenna extraction: (window_size, 2, 768)
        receiver_idx = self.SINGLE_ANTENNA_MAP[meta.layout][meta.sender_id]
        antenna_slice = sender_slice[:, receiver_idx, :, :]  # (window_size, 2, 768)

        # Windowed averaging over range bins: (window_size, 2, num_windows)
        # 768 bins with window=32, stride=32 -> 24 windows
        num_range_bins = antenna_slice.shape[-1]
        num_range_windows = (num_range_bins - self.range_window) // self.range_stride + 1

        # Efficient windowed mean using reshape
        reduced_list = []
        for i in range(num_range_windows):
            start_bin = i * self.range_stride
            end_bin = start_bin + self.range_window
            window_mean = np.mean(antenna_slice[:, :, start_bin:end_bin], axis=-1, keepdims=True)
            reduced_list.append(window_mean)

        # Stack along last dimension: (window_size, 2, 24)
        reduced = np.concatenate(reduced_list, axis=-1).astype(np.float32)

        # Copy to detach from mmap before augmentation
        reduced = reduced.copy()

        # Apply augmentations (only during training)
        if self.is_train:
            # Time warping (speed variation)
            if self.use_time_warp:
                reduced = apply_time_warp(
                    reduced,
                    sigma=self.time_warp_sigma,
                    num_knots=self.time_warp_knots
                )

            # Gaussian noise
            if self.use_gaussian_noise:
                reduced = apply_gaussian_noise(reduced, noise_std=self.noise_std)

        # Min-max normalization (applied to both train and eval)
        if self.use_minmax_norm:
            reduced = min_max_normalize(reduced, feature_range=self.norm_range)

        # Convert to tensor
        tensor = torch.from_numpy(reduced).float()

        # Labels as tensor: (num_desks,)
        label_tensor = torch.tensor(meta.labels, dtype=torch.float32)

        return tensor, label_tensor


def create_full_range_dataloaders(
    data_dir: str,
    label_yaml_path: str,
    window_size: int = 1500,
    stride: int = 750,
    range_window: int = 32,
    range_stride: int = 32,
    batch_size: int = 32,
    num_workers: int = 4,
    # Normalization
    use_minmax_norm: bool = True,
    norm_range: Tuple[float, float] = (0, 1),
    # Augmentation
    use_gaussian_noise: bool = False,
    noise_std: float = 0.01,
    use_time_warp: bool = False,
    time_warp_sigma: float = 0.2,
    time_warp_knots: int = 4,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train, validation, and test dataloaders for full range CIR data.

    Args:
        data_dir: Directory containing .npy files
        label_yaml_path: Path to YAML file with labels and split info
        window_size: Sliding window size in time (default: 1500)
        stride: Stride for sliding window in time (default: 750)
        range_window: Window size for averaging range bins (default: 32)
        range_stride: Stride for range bin windowing (default: 32)
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        use_minmax_norm: Enable min-max normalization
        norm_range: Target range for normalization
        use_gaussian_noise: Enable Gaussian noise augmentation (train only)
        noise_std: Noise standard deviation
        use_time_warp: Enable time warping augmentation (train only)
        time_warp_sigma: Time warp magnitude
        time_warp_knots: Number of knots for time warp spline

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = FullRangeCIRDataset(
        data_dir=data_dir,
        label_yaml_path=label_yaml_path,
        split='train',
        window_size=window_size,
        stride=stride,
        range_window=range_window,
        range_stride=range_stride,
        is_train=True,
        use_minmax_norm=use_minmax_norm,
        norm_range=norm_range,
        use_gaussian_noise=use_gaussian_noise,
        noise_std=noise_std,
        use_time_warp=use_time_warp,
        time_warp_sigma=time_warp_sigma,
        time_warp_knots=time_warp_knots,
    )

    val_dataset = FullRangeCIRDataset(
        data_dir=data_dir,
        label_yaml_path=label_yaml_path,
        split='val',
        window_size=window_size,
        stride=stride,
        range_window=range_window,
        range_stride=range_stride,
        is_train=False,
        use_minmax_norm=use_minmax_norm,
        norm_range=norm_range,
    )

    test_dataset = FullRangeCIRDataset(
        data_dir=data_dir,
        label_yaml_path=label_yaml_path,
        split='test',
        window_size=window_size,
        stride=stride,
        range_window=range_window,
        range_stride=range_stride,
        is_train=False,
        use_minmax_norm=use_minmax_norm,
        norm_range=norm_range,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


# =============================================================================
# Multi-Desk CIR Dataloaders (original)
# =============================================================================

def create_dataloaders(
    data_dir: str,
    label_yaml_path: str,
    window_size: int = 1500,
    stride: int = 750,
    sender_idx: Optional[int] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    bins_per_desk: int = 70,
    crop_size: int = 64,
    # Normalization
    use_minmax_norm: bool = False,
    norm_range: Tuple[float, float] = (0, 1),
    # Augmentation
    use_gaussian_noise: bool = False,
    noise_std: float = 0.01,
    use_time_warp: bool = False,
    time_warp_sigma: float = 0.2,
    time_warp_knots: int = 4,
    # Single antenna mode
    single_antenna: bool = False,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train, validation, and test dataloaders.

    Args:
        data_dir: Directory containing .npy files
        label_yaml_path: Path to YAML file with labels and split info
        window_size: Sliding window size (default: 1500)
        stride: Stride for sliding window (default: 750)
        sender_idx: Specific sender (0-3) or None for all
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        bins_per_desk: Range bins per desk (default: 70)
        crop_size: Final bins after cropping (default: 64)
        use_minmax_norm: Enable min-max normalization
        norm_range: Target range for normalization
        use_gaussian_noise: Enable Gaussian noise augmentation (train only)
        noise_std: Noise standard deviation
        use_time_warp: Enable time warping augmentation (train only)
        time_warp_sigma: Time warp magnitude
        time_warp_knots: Number of knots for time warp spline
        single_antenna: Extract single antenna per sender based on layout

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = MultiDeskCIRDataset(
        data_dir=data_dir,
        label_yaml_path=label_yaml_path,
        split='train',
        window_size=window_size,
        stride=stride,
        sender_idx=sender_idx,
        is_train=True,
        bins_per_desk=bins_per_desk,
        crop_size=crop_size,
        use_minmax_norm=use_minmax_norm,
        norm_range=norm_range,
        use_gaussian_noise=use_gaussian_noise,
        noise_std=noise_std,
        use_time_warp=use_time_warp,
        time_warp_sigma=time_warp_sigma,
        time_warp_knots=time_warp_knots,
        single_antenna=single_antenna,
    )

    val_dataset = MultiDeskCIRDataset(
        data_dir=data_dir,
        label_yaml_path=label_yaml_path,
        split='val',
        window_size=window_size,
        stride=stride,
        sender_idx=sender_idx,
        is_train=False,
        bins_per_desk=bins_per_desk,
        crop_size=crop_size,
        use_minmax_norm=use_minmax_norm,
        norm_range=norm_range,
        single_antenna=single_antenna,
        # Augmentation disabled for val (is_train=False)
    )

    test_dataset = MultiDeskCIRDataset(
        data_dir=data_dir,
        label_yaml_path=label_yaml_path,
        split='test',
        window_size=window_size,
        stride=stride,
        sender_idx=sender_idx,
        is_train=False,
        bins_per_desk=bins_per_desk,
        crop_size=crop_size,
        use_minmax_norm=use_minmax_norm,
        norm_range=norm_range,
        single_antenna=single_antenna,
        # Augmentation disabled for test (is_train=False)
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Example usage
    data_directory = "./data/"
    label_yaml = "./dataset_labels_oneLongDesk.yaml"

    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_directory,
        label_yaml_path=label_yaml,
        window_size=1500,
        stride=750,
        sender_idx=None,
        batch_size=32,
        num_workers=4,
        # Normalization
        use_minmax_norm=True,
        norm_range=(0, 1),
        # Augmentation (train only)
        use_gaussian_noise=True,
        noise_std=0.01,
        use_time_warp=True,
        time_warp_sigma=0.2,
        time_warp_knots=4,
    )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    for data_batch, label_batch in train_loader:
        print(f"Data batch shape: {data_batch.shape}, Label batch shape: {label_batch.shape}")
        print(f"Data range: [{data_batch.min():.4f}, {data_batch.max():.4f}]")
        break