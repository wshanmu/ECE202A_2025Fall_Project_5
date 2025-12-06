#!/usr/bin/env python3
"""
Sensing Data Processing Pipeline
Processes CIR data from multiple nodes, performs alignment and phase correction,
and saves waterfall plots for each TX-RX pair.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, medfilt, butter, filtfilt
import os
import re
import hydra
from omegaconf import DictConfig, OmegaConf

from correlation_alignment import correlation_align_cir

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
        out = real_filtered + 1j * imag_filtered
    else:
        # Real-valued data
        out = filtfilt(b, a, data, axis=axis)
    return out.astype(data.dtype, copy=False)

def convert_to_amp_and_phase(data):
    """
    Convert complex64 data to amplitude and phase representation.

    Parameters:
    data (np.ndarray): Input complex64 data.

    Returns:
    tuple: A tuple containing two np.ndarrays - amplitude and phase.
    """
    if not isinstance(data, np.ndarray) or data.dtype != np.complex64:
        return data
    node_tx, node_rx, t_index, tap_index = data.shape
    new_data = np.empty((node_tx, node_rx, 2, t_index, tap_index), dtype=np.float32)
    amplitude = np.abs(data)
    phase = np.unwrap(np.angle(data), axis=-2)
    new_data[:, :, 0, :, :] = amplitude
    new_data[:, :, 1, :, :] = phase
    return new_data

@hydra.main(version_base=None, config_path="config", config_name="cir_sensing_processing")
def main(config: DictConfig):
    """Main processing function with Hydra config management."""

    print("=" * 60)
    print("CIR Data Processing Pipeline")
    print("=" * 60)

    print(f"\nLoading data from: {config.input_folder}")
    print(f"Upsample factor: {config.upsample_factor}")

    INPUT_FOLDER = config.input_folder

    # Automatically derive input_file from input_folder
    folder_name = os.path.basename(INPUT_FOLDER)
    # Extract everything after the timestamp pattern (YYYYMMDD_HHMMSS_)
    suffix_match = re.search(r'\d{8}_\d{6}_(.*)', folder_name)
    if suffix_match:
        suffix = suffix_match.group(1)
    else:
        # Fallback: use entire folder name if pattern doesn't match
        suffix = folder_name

    INPUT_FILE_NAME = f"rx/cir_data_rx_{suffix}"
    print(f"Auto-derived input file: {INPUT_FILE_NAME}")

    board_id = [0, 1, 2, 3]
    INPUT_FILES = [os.path.join(INPUT_FOLDER, INPUT_FILE_NAME + f'_Node{i}.npy') for i in board_id]
    DIAG_FILES = [os.path.join(INPUT_FOLDER, 'diag', f'diag_data_rx_{suffix}_Node{i}.npy') for i in board_id]

    upsample_factor = config.upsample_factor

    # Load CIR arrays
    print("\n[1/7] Loading CIR data...")
    rx_cir_list = [np.load(f, allow_pickle=True)[config.discard_reading:] for f in INPUT_FILES]
    min_len = min([cir.shape[0] for cir in rx_cir_list])
    rx_cir_list = [cir[-(min_len - config.discard_reading):, :] for cir in rx_cir_list]
    rx_cir_files = np.stack(rx_cir_list, axis=0)

    # Load diagnostic data
    diag_data_files = [np.load(f, allow_pickle=True)[-(min_len - config.discard_reading):] for f in DIAG_FILES]
    diag_data_files = np.array(diag_data_files, dtype=object)
    print(f"   CIR shape: {rx_cir_files.shape} (nodes, frames, taps)")

    frame_idx_list = np.array([np.array([diag['D'] for diag in diag_data]) for diag_data in diag_data_files])
    sender_idx_list = np.array([np.array([diag['sender_ID'] for diag in diag_data]) for diag_data in diag_data_files])

    # Calculate fractional part shifts
    print("\n[2/7] Computing fractional shifts...")
    index_fp = np.array([np.array([i['index_fp_u32']/64 for i in diag_data], dtype=np.float32) for diag_data in diag_data_files])
    index_fp = index_fp / 64 - index_fp // 64
    shift_64 = (index_fp * upsample_factor)
    shift_64 = np.round(shift_64).astype(int)
    print(f"   Shift values computed for {len(board_id)} nodes")

    # Preprocess CIR data
    num_nodes, num_frames, num_taps = rx_cir_files.shape
    zero_padded_num = config.zero_padded_num
    zero_padded_rx_cir = np.zeros((num_nodes, num_frames, num_taps + 2*zero_padded_num), dtype=np.complex64)
    zero_padded_rx_cir[..., zero_padded_num:zero_padded_num+num_taps] = rx_cir_files
    rx_cir = zero_padded_rx_cir
    print(f"   Zero-padded CIR shape: {rx_cir.shape}, dtype: {rx_cir.dtype}")

    # Upsample CIR data using FFT
    print(f"\n[3/7] Upsampling CIR by {upsample_factor}x using FFT...")
    cir_upsampled = np.zeros((num_nodes, num_frames, (num_taps+zero_padded_num*2) * upsample_factor), dtype=np.complex64)
    for i in range(num_nodes):
        for j in range(num_frames):
            X = np.fft.fft(rx_cir[i, j])
            X_upsampled = np.zeros(X.shape[0] * upsample_factor, dtype=np.complex64)
            X_upsampled[:X.shape[0]//2] = X[:X.shape[0]//2]
            X_upsampled[-X.shape[0]//2:] = X[-X.shape[0]//2:]
            cir_upsampled[i, j] = np.fft.ifft(X_upsampled)[0:cir_upsampled.shape[2]] * upsample_factor
    print(f"   Upsampled CIR shape: {cir_upsampled.shape}, dtype: {cir_upsampled.dtype}")
    # cir_upsampled = cir_upsampled[:, 0:cir_upsampled.shape[1]//(num_nodes-1)*(num_nodes-1), :]

    # Shift and normalize
    print("\n[4/7] Applying fractional shifts and normalizing...")
    accum_list = np.array([[i.get('accumCount', 1) for i in diag_data] for diag_data in diag_data_files])

    cir_shifted = np.zeros_like(cir_upsampled, dtype=np.complex64)
    upsampled_taps = cir_upsampled.shape[-1]

    idx = (np.arange(upsampled_taps)[None, None, :] - shift_64[..., None]) % upsampled_taps
    rolled = np.take_along_axis(cir_upsampled, idx, axis=-1)
    cir_shifted = rolled / accum_list[..., None]
    print(f"   CIR shifted and normalized")

    # Reorganize CIR data by sender nodes
    print("\n[5/7] Reorganizing CIR by TX-RX pairs...")
    cir_reorg = None
    num_of_nodes_except_tx = len(board_id) - 1
    # seq_num = cir_shifted.shape[1] // num_of_nodes_except_tx
    seq_num = 1e9
    for tx_node in board_id:
        sender_node_is_tx = (sender_idx_list == tx_node)
        for i in board_id:
            if i == tx_node:
                continue
            seq_num = min(seq_num, cir_shifted[i, sender_node_is_tx[i], :].shape[0])
    print(f"   Each TX-RX pair has {seq_num} frames")
    for tx_node in board_id:
        sender_node_is_tx = (sender_idx_list == tx_node)
        if cir_reorg is None:
            cir_reorg = np.zeros((len(board_id), num_of_nodes_except_tx, seq_num, cir_shifted.shape[2]), dtype=np.complex64)
        for i in board_id:
            if i == tx_node:
                continue
            cir_reorg[tx_node, i - (1 if i > tx_node else 0)] = cir_shifted[i, sender_node_is_tx[i], :][:seq_num]
    print(f"   Reorganized CIR shape: {cir_reorg.shape}, dtype: {cir_reorg.dtype} (tx_nodes, rx_nodes, frames, taps)")

    cir_shifted = cir_reorg
    B0, B1, N, L = cir_shifted.shape
    los_indices = np.zeros((B0, B1), dtype=float)
    cir_aligned = np.zeros_like(cir_shifted, dtype=np.complex64)

    # Load alignment parameters from config
    los_search_start = int((zero_padded_num + config.los_search_offset_start) * upsample_factor)
    los_search_end = int((zero_padded_num + config.los_search_offset_end) * upsample_factor)
    peak_win_start = zero_padded_num * upsample_factor
    peak_win_end = (zero_padded_num + config.peak_win_offset_end) * upsample_factor

    # Align CIR data using zero-crossing method
    print("\n[6/7] Aligning CIR using zero-crossing method...")
    print(f"   Savgol filter: window={config.savgol_window_length}, poly={config.savgol_polyorder}, deriv={config.savgol_deriv}")
    for b0 in range(B0):
        for b1 in range(B1):
            cir_slice = cir_shifted[b0, b1]
            aligned_cir, shifts, combined_outliers = correlation_align_cir(
                cir_data=cir_slice, 
                los_search_start=los_search_start, 
                los_search_end=los_search_end,
                sg_window=5, sg_poly=2,
                corr_factor=0.5,
                derivative_type='2nd',
                raw_corr_threshold=0.7,
                raw_corr_search_window=50,
                interpolate_outliers=True, visualize=False)

            cir_aligned[b0, b1] = aligned_cir

            aligned_los_part = np.abs(aligned_cir[:, peak_win_start:peak_win_end])
            peak_indices = np.argmax(aligned_los_part, axis=1)
            los_index = peak_indices.mean() + peak_win_start
            los_indices[b0, b1] = los_index

    # Roll LoS index to 100
    for b0 in range(B0):
        for b1 in range(B1):
            los_diff = 100 - los_indices[b0, b1]
            cir_aligned[b0, b1] = np.roll(cir_aligned[b0, b1], int(np.round(los_diff)), axis=-1)
            los_indices[b0, b1] += los_diff
    print(f"   Alignment complete. LoS indices computed for all {B0}x{B1} TX-RX pairs")

    # Zero reference phase
    print("\n[7/7] Applying phase correction...")
    B0, B1, N, L = cir_aligned.shape
    start = int((zero_padded_num + 1) * upsample_factor)
    end = (zero_padded_num + 2) * upsample_factor + int(los_index)

    cir_aligned_los_part = cir_aligned[..., start:end]
    cir_aligned_los_part_phase = np.unwrap(np.angle(cir_aligned_los_part), axis=-1)

    ref_phase = cir_aligned_los_part_phase.mean(axis=-1, keepdims=True)
    cir_phased = cir_aligned * np.exp(-1j * ref_phase)
    print(f"   Phase correction applied, CIR shape: {cir_phased.shape}, dtype: {cir_phased.dtype}")

    print("\n[8/8] Applying Low pass filtering at 3Hz...")
    cir_lpf = butter_lowpass_filter(
                cir_phased,
                cutoff=3.0,
                fs=100,
                order=5,
                axis=2  # Slow time axis (frames dimension)
            )
    print("LPFed: ", cir_lpf.shape, cir_lpf.dtype)
    # covert to amplitude and phase representation
    cir_lpf = convert_to_amp_and_phase(cir_lpf)
    print(f"   Converted to amplitude and phase representation, new shape: {cir_lpf.shape}, dtype: {cir_lpf.dtype}")
    new_data = np.transpose(cir_lpf, (3, 0, 1, 2, 4))

    # Save processed CIR data
    saved_path = f'./tdma_sensing/cir_files/processed_cir/{config.input_folder.split("/")[-1]}_cir_lpf3.npy'
    np.save(saved_path, new_data)
    print(f"\n   Processed CIR saved to: {saved_path}")

    # Import waterfall_cir function
    from cir_utils import waterfall_cir

    # Generate and save waterfall plots for all TX-RX pairs
    print("\n" + "=" * 60)
    print("Generating Waterfall Plots")
    print("=" * 60)

    output_folder = config.input_folder
    plot_count = 0
    for tx_node in board_id:
        for i in range(len(board_id) - 1):
            dynamic = np.abs(cir_phased[tx_node, i])
            rx_nodes = [n for n in board_id if n != tx_node]
            rx_node = rx_nodes[i]

            plot_count += 1
            print(f"\n[{plot_count}/6] Saving waterfall plot: TX Node {tx_node} -> RX Node {rx_node}")

            waterfall_cir(
                dynamic,
                db=False,
                time_real=np.arange(dynamic.shape[0]),
                time_stamp_suffix=f"Node {tx_node} to Node {rx_node}",
                save=True,
                path=output_folder,
                show=False
            )
            plt.close()

    print("\n" + "=" * 60)
    print(f"Processing Complete!")
    print(f"   Total waterfall plots saved: {plot_count}")
    print(f"   Output folder: {output_folder}")
    print("=" * 60)


if __name__ == "__main__":
    main()
