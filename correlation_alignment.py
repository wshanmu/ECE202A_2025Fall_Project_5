#!/usr/bin/env python3
"""
Correlation-based CIR Alignment using 1st and 2nd derivative correlation.

This version calculates shifts using correlation on 1st and 2nd derivatives
separately, then averages them. Outliers are detected based on maximum
correlation power.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, correlate


def compute_derivative_correlations(cir_los_part, reference_frame, sg_window=25, sg_poly=2):
    """
    Compute correlations using 1st and 2nd derivatives separately.

    Parameters:
    -----------
    cir_los_part : ndarray (N_frames, N_taps)
        LoS portion of CIR data (amplitude)
    reference_frame : ndarray (N_taps,)
        Reference frame for correlation
    sg_window : int
        Savitzky-Golay filter window length
    sg_poly : int
        Polynomial order for SG filter

    Returns:
    --------
    shifts_1st : ndarray (N_frames,)
        Shifts calculated from 1st derivative correlation
    shifts_2nd : ndarray (N_frames,)
        Shifts calculated from 2nd derivative correlation
    corr_max_1st : ndarray (N_frames,)
        Maximum correlation values from 1st derivative
    corr_max_2nd : ndarray (N_frames,)
        Maximum correlation values from 2nd derivative
    """
    N, L = cir_los_part.shape
    sg_bound = sg_window // 2 + 1

    # Compute derivatives for all frames
    cir_1st_deriv = savgol_filter(cir_los_part, window_length=sg_window,
                                  polyorder=sg_poly, deriv=1, axis=1)[:, sg_bound:-sg_bound]
    cir_2nd_deriv = savgol_filter(cir_los_part, window_length=sg_window,
                                  polyorder=sg_poly, deriv=2, axis=1)[:, sg_bound:-sg_bound]

    # Compute reference derivatives
    ref_1st_deriv = savgol_filter(reference_frame, window_length=sg_window,
                                  polyorder=sg_poly, deriv=1)[sg_bound:-sg_bound]
    ref_2nd_deriv = savgol_filter(reference_frame, window_length=sg_window,
                                  polyorder=sg_poly, deriv=2)[sg_bound:-sg_bound]

    shifts_1st = np.zeros(N)
    shifts_2nd = np.zeros(N)
    corr_max_1st = np.zeros(N)
    corr_max_2nd = np.zeros(N)

    # Constrained search window around center (±10 taps)
    search_window = 10
    center_idx = L // 2

    for i in range(N):
        # 1st derivative correlation
        corr_1st = correlate(cir_1st_deriv[i], ref_1st_deriv, mode='same')
        # Search only in window around center
        search_start = max(0, center_idx - search_window)
        search_end = min(len(corr_1st), center_idx + search_window)
        max_idx_1st = np.argmax(corr_1st[search_start:search_end]) + search_start
        corr_max_1st[i] = corr_1st[max_idx_1st]
        shifts_1st[i] = L // 2 - max_idx_1st - sg_bound

        # 2nd derivative correlation
        corr_2nd = correlate(cir_2nd_deriv[i], ref_2nd_deriv, mode='same')
        # Search only in window around center
        max_idx_2nd = np.argmax(corr_2nd[search_start:search_end]) + search_start
        corr_max_2nd[i] = corr_2nd[max_idx_2nd]
        shifts_2nd[i] = L // 2 - max_idx_2nd - sg_bound

    return shifts_1st, shifts_2nd, corr_max_1st, corr_max_2nd


def detect_outliers_by_correlation(corr_max_1st, corr_max_2nd, factor=0.5):
    """
    Detect outlier frames based on correlation power relative to mean.

    Processes 1st and 2nd derivative correlations separately, then merges results.
    A frame is an outlier if its correlation < mean(correlation) * factor.

    Parameters:
    -----------
    corr_max_1st : ndarray (N_frames,)
        Maximum correlation values from 1st derivative
    corr_max_2nd : ndarray (N_frames,)
        Maximum correlation values from 2nd derivative
    factor : float
        Threshold factor (0-1). Frames with corr < mean*factor are outliers.
        Default: 0.5 (50% of mean correlation)

    Returns:
    --------
    outlier_indices : list
        Indices of detected outlier frames (union of both derivatives)
    mean_corr : ndarray (N_frames,)
        Mean correlation power per frame (for visualization)
    """
    # Compute mean correlation for each derivative
    mean_corr_1st = np.mean(corr_max_1st)
    mean_corr_2nd = np.mean(corr_max_2nd)

    # Compute thresholds
    threshold_1st = mean_corr_1st * factor
    threshold_2nd = mean_corr_2nd * factor

    # Detect outliers for 1st derivative
    outliers_1st = np.where(corr_max_1st < threshold_1st)[0].tolist()

    # Detect outliers for 2nd derivative
    outliers_2nd = np.where(corr_max_2nd < threshold_2nd)[0].tolist()

    # Merge outliers (union)
    outlier_indices = list(set(outliers_1st + outliers_2nd))
    outlier_indices.sort()

    # Compute mean correlation for visualization
    mean_corr = (corr_max_1st + corr_max_2nd) / 2

    return outlier_indices, mean_corr


def interpolate_outlier_frames(cir_data, outlier_indices, max_search_distance=5):
    """
    Replace outlier frames with interpolated values using neighboring frames.

    For each outlier, computes mean amplitude and mean phase from nearest
    non-outlier frames before and after, then reconstructs the complex frame.

    Parameters:
    -----------
    cir_data : ndarray (N_frames, L_taps)
        Complex CIR data with outliers
    outlier_indices : list or array
        Indices of frames to interpolate
    max_search_distance : int
        Maximum distance to search for non-outlier neighbors (default: 5)

    Returns:
    --------
    interpolated_cir : ndarray (N_frames, L_taps)
        CIR data with outliers replaced by interpolated values
    """
    interpolated_cir = cir_data.copy()
    N_frames = cir_data.shape[0]

    if len(outlier_indices) == 0:
        print("   No outliers to interpolate")
        return interpolated_cir

    outlier_set = set(outlier_indices)
    outlier_indices = np.array(outlier_indices)
    interpolated_count = 0
    skipped_count = 0

    for idx in outlier_indices:
        # Find nearest non-outlier frame before
        frame_before_idx = None
        for offset in range(1, max_search_distance + 1):
            candidate_idx = idx - offset
            if candidate_idx < 0:
                break
            if candidate_idx not in outlier_set:
                frame_before_idx = candidate_idx
                break

        # Find nearest non-outlier frame after
        frame_after_idx = None
        for offset in range(1, max_search_distance + 1):
            candidate_idx = idx + offset
            if candidate_idx >= N_frames:
                break
            if candidate_idx not in outlier_set:
                frame_after_idx = candidate_idx
                break

        # Check if we found valid neighbors
        if frame_before_idx is None or frame_after_idx is None:
            print(f"   Warning: Cannot find non-outlier neighbors for frame {idx} "
                  f"within {max_search_distance} frames, skipping")
            skipped_count += 1
            continue

        # Get neighboring frames
        frame_before = cir_data[frame_before_idx]
        frame_after = cir_data[frame_after_idx]

        # Compute mean amplitude
        amp_before = np.abs(frame_before)
        amp_after = np.abs(frame_after)
        mean_amplitude = (amp_before + amp_after) / 2

        # Compute mean phase (using circular mean for phase)
        phase_before = np.angle(frame_before)
        phase_after = np.angle(frame_after)
        mean_phase = np.arctan2(
            (np.sin(phase_before) + np.sin(phase_after)) / 2,
            (np.cos(phase_before) + np.cos(phase_after)) / 2
        )

        # Reconstruct complex frame
        interpolated_cir[idx] = mean_amplitude * np.exp(1j * mean_phase)
        interpolated_count += 1

    print(f"   Interpolated {interpolated_count} outlier frames")
    if skipped_count > 0:
        print(f"   Skipped {skipped_count} frames (no valid neighbors)")

    return interpolated_cir


def correlation_align_cir(cir_data, los_search_start, los_search_end,
                          sg_window=25, sg_poly=2,
                          corr_factor=0.5,
                          raw_corr_threshold=0.7,
                          raw_corr_search_window=10,
                          derivative_type='both',
                          interpolate_outliers=True, visualize=False):
    """
    Perform CIR alignment using multi-stage correlation pipeline.

    Pipeline stages:
    1. Initial raw amplitude correlation (using mean of first 10 frames)
    2. Frame-by-frame correlation with min-max normalization
    3. Outlier detection based on max correlation power
    4. Outlier interpolation
    5. Derivative-based correlation alignment

    Parameters:
    -----------
    cir_data : ndarray (N_frames, L_taps)
        Full CIR data to be aligned (complex)
    cir_los_part : ndarray (N_frames, los_length)
        LoS portion extracted from cir_data (amplitude)
    sg_window : int
        Savitzky-Golay filter window length
    sg_poly : int
        Savitzky-Golay polynomial order
    corr_factor : float
        Outlier threshold factor (0-1). Frames with correlation < mean*factor
        are outliers. Default: 0.5 (50% of mean)
    derivative_type : str
        Type of derivative to use: '1st', '2nd', or 'both' (default)
        - '1st': Use only 1st derivative correlation
        - '2nd': Use only 2nd derivative correlation
        - 'both': Average of 1st and 2nd derivative correlations
    interpolate_outliers : bool
        Whether to interpolate outlier frames using neighbors
    visualize : bool
        Whether to plot diagnostic figures

    Returns:
    --------
    aligned_cir : ndarray (N_frames, L_taps)
        Aligned CIR data (with outliers interpolated if requested)
    shifts : ndarray (N_frames,)
        Applied shift amounts from derivative correlation
    outlier_indices : list
        Detected outlier frame indices
    """
    N, L = cir_data.shape

    # ===== STAGE 1: Initial Raw Amplitude Correlation =====
    print("=" * 60)
    print("STAGE 1: Initial raw amplitude correlation alignment")
    print("=" * 60)

    # Use mean of first 10 frames as reference
    K = min(1, N)
    cir_los_part = np.abs(cir_data[:, los_search_start:los_search_end])
    ref_raw = cir_los_part[0]
    # Min-max normalize reference
    ref_raw_norm = (ref_raw - ref_raw.min()) / (ref_raw.max() - ref_raw.min() + 1e-8)
    print(f"   Using first frame as reference")

    # ===== STAGE 2: Frame-by-frame correlation with min-max normalization =====
    print("\nSTAGE 2: Frame-by-frame correlation with min-max normalization")
    print("-" * 60)

    cir_data_shifted = cir_data.copy()
    corr_max_raw = np.zeros(N)
    raw_shifts = np.zeros(N)
    half_window = cir_los_part.shape[1] // 2

    for i in range(N):
        # Min-max normalize each frame
        curr_frame = cir_los_part[i]
        curr_frame_norm = (curr_frame - curr_frame.min()) / (curr_frame.max() - curr_frame.min() + 1e-8)

        # Correlate with reference
        # find the maximum and shift within a search window
        center_idx = len(curr_frame_norm) // 2
        search_start = max(0, center_idx - raw_corr_search_window)
        search_end = min(len(curr_frame_norm), center_idx + raw_corr_search_window)
        corr = correlate(ref_raw_norm, curr_frame_norm, mode='same')
        # Search only in window around center
        corr = corr[search_start:search_end]
        aligned_idx = np.argmax(corr)
        corr_max_raw[i] = corr[aligned_idx]

        # Calculate shift
        shift = -half_window + aligned_idx + search_start
        raw_shifts[i] = shift

        # Apply shift
        if shift != 0:
            cir_data_shifted[i] = np.roll(cir_data_shifted[i], shift)

    print(f"   Raw correlation shifts applied: mean={np.mean(raw_shifts):.2f}, std={np.std(raw_shifts):.2f}")
    print(f"   Correlation power range: [{corr_max_raw.min():.2e}, {corr_max_raw.max():.2e}]")

    # ===== STAGE 3: Outlier Detection =====
    print("\nSTAGE 3: Outlier detection based on correlation power")
    print("-" * 60)

    # Use percentile-based threshold
    # threshold = np.max(corr_max_raw) - 0.3 * (np.max(corr_max_raw) - np.min(corr_max_raw))
    # threshold = np.max(corr_max_raw) * raw_corr_threshold
    threshold = np.quantile(corr_max_raw, 0.9) * raw_corr_threshold
    percentile_outliers = np.where(corr_max_raw < threshold)[0].tolist()

    print(f"   Threshold: {threshold:.2e}")
    print(f"   Found {len(percentile_outliers)} outlier frames based on raw correlation")
    if len(percentile_outliers) > 0 and len(percentile_outliers) <= 20:
        print(f"   Outlier indices: {percentile_outliers}")

    # ===== STAGE 4: Outlier Interpolation =====
    print("\nSTAGE 4: Interpolating outlier frames")
    print("-" * 60)
    cir_data_interp = interpolate_outlier_frames(cir_data_shifted,
                                                  percentile_outliers,
                                                  max_search_distance=10)

    # Update LoS part after interpolation
    cir_los_part_interp = np.abs(cir_data_interp[:, los_search_start:los_search_end])

    # ===== STAGE 5: Derivative-based Correlation Alignment =====
    print("\nSTAGE 5: Derivative-based correlation alignment")
    print("-" * 60)
    print(f"   Using derivative type: {derivative_type}")

    # Use mean of first K frames as reference
    reference_frame = np.mean(cir_los_part_interp[:K], axis=0)
    print(f"   Using mean of first {K} frames as reference")

    # Compute derivative correlations
    shifts_1st, shifts_2nd, corr_max_1st, corr_max_2nd = compute_derivative_correlations(
        cir_los_part_interp, reference_frame, sg_window, sg_poly
    )

    # Select derivative based on parameter
    if derivative_type == '1st':
        shifts = shifts_1st
        corr_max_used = corr_max_1st
        print(f"   1st derivative: shift range [{shifts_1st.min():.1f}, {shifts_1st.max():.1f}], "
              f"corr range [{corr_max_1st.min():.2e}, {corr_max_1st.max():.2e}]")
    elif derivative_type == '2nd':
        shifts = shifts_2nd
        corr_max_used = corr_max_2nd
        print(f"   2nd derivative: shift range [{shifts_2nd.min():.1f}, {shifts_2nd.max():.1f}], "
              f"corr range [{corr_max_2nd.min():.2e}, {corr_max_2nd.max():.2e}]")
    else:  # 'both'
        shifts = (shifts_1st + shifts_2nd) / 2
        corr_max_used = (corr_max_1st + corr_max_2nd) / 2
        print(f"   1st derivative: shift range [{shifts_1st.min():.1f}, {shifts_1st.max():.1f}], "
              f"corr range [{corr_max_1st.min():.2e}, {corr_max_1st.max():.2e}]")
        print(f"   2nd derivative: shift range [{shifts_2nd.min():.1f}, {shifts_2nd.max():.1f}], "
              f"corr range [{corr_max_2nd.min():.2e}, {corr_max_2nd.max():.2e}]")

    print(f"   Final shift statistics: mean={np.mean(shifts):.2f}, std={np.std(shifts):.2f}")

    # Apply derivative-based shifts
    print("\nApplying derivative-based shifts...")
    aligned_cir = np.zeros_like(cir_data_interp, dtype=cir_data_interp.dtype)
    for i in range(N):
        aligned_cir[i] = np.roll(cir_data_interp[i], int(np.round(shifts[i])))

    # Detect derivative-based outliers
    print("\nDetecting derivative-based outliers...")
    print(f"   Using correlation factor: {corr_factor}")
    outlier_indices, mean_corr = detect_outliers_by_correlation(
        corr_max_1st, corr_max_2nd, corr_factor
    )
    print(f"   Found {len(outlier_indices)} additional outlier frames from derivative correlation")

    # Combine outliers
    combined_outliers = list(set(percentile_outliers + outlier_indices))
    combined_outliers.sort()
    print(f"\nTotal unique outliers detected: {len(combined_outliers)}")

    # Visualization
    if visualize:
        plot_correlation_diagnostics(shifts_1st, shifts_2nd, shifts,
                                     corr_max_1st, corr_max_2nd, mean_corr,
                                     combined_outliers,
                                     raw_shifts, corr_max_raw, percentile_outliers)

    print("=" * 60)
    print("Alignment complete!")
    print("=" * 60)

    return aligned_cir, shifts, combined_outliers


def plot_correlation_diagnostics(shifts_1st, shifts_2nd, shifts_mean,
                                 corr_max_1st, corr_max_2nd, corr_mean,
                                 outlier_indices,
                                 raw_shifts=None, corr_max_raw=None, raw_outliers=None):
    """
    Plot diagnostic figures for multi-stage correlation-based alignment.

    Parameters:
    -----------
    shifts_1st, shifts_2nd, shifts_mean : ndarray
        Shifts from derivative correlations
    corr_max_1st, corr_max_2nd, corr_mean : ndarray
        Correlation powers
    outlier_indices : list
        Combined outlier indices
    raw_shifts : ndarray, optional
        Shifts from raw amplitude correlation (Stage 2)
    corr_max_raw : ndarray, optional
        Correlation powers from raw amplitude (Stage 2)
    raw_outliers : list, optional
        Outlier indices from raw correlation (Stage 3)
    """

    N = len(shifts_mean)
    frame_indices = np.arange(N)
    normal_mask = np.ones(N, dtype=bool)
    normal_mask[outlier_indices] = False

    # Determine number of rows based on whether raw data is provided
    n_rows = 3 if raw_shifts is not None else 2
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))

    # Make axes indexable in a consistent way
    if n_rows == 2:
        axes_flat = axes
    else:
        axes_flat = axes

    # ===== Row 1: Raw Correlation (if provided) =====
    if raw_shifts is not None and corr_max_raw is not None:
        row_idx = 0

        # Create mask for raw outliers
        raw_normal_mask = np.ones(N, dtype=bool)
        if raw_outliers is not None and len(raw_outliers) > 0:
            raw_normal_mask[raw_outliers] = False

        # Plot 1: Raw correlation shifts
        axes_flat[row_idx, 0].plot(frame_indices[raw_normal_mask], raw_shifts[raw_normal_mask],
                       'c.', alpha=0.5, label='Normal')
        if raw_outliers is not None and len(raw_outliers) > 0:
            axes_flat[row_idx, 0].plot(raw_outliers, raw_shifts[raw_outliers],
                           'rx', markersize=10, label='Outliers')
        axes_flat[row_idx, 0].set_xlabel('Frame Index')
        axes_flat[row_idx, 0].set_ylabel('Shift (taps)')
        axes_flat[row_idx, 0].set_title('Stage 2: Raw Amplitude Correlation Shifts')
        axes_flat[row_idx, 0].legend()
        axes_flat[row_idx, 0].grid(True, alpha=0.3)

        # Plot 2: Raw correlation power
        axes_flat[row_idx, 1].plot(frame_indices[raw_normal_mask], corr_max_raw[raw_normal_mask],
                        'c.', alpha=0.5, label='Normal')
        if raw_outliers is not None and len(raw_outliers) > 0:
            axes_flat[row_idx, 1].plot(raw_outliers, corr_max_raw[raw_outliers],
                           'rx', markersize=10, label='Outliers')
            # Draw threshold line
            threshold = np.max(corr_max_raw) - 0.3 * (np.max(corr_max_raw) - np.min(corr_max_raw))
            axes_flat[row_idx, 1].axhline(y=threshold, color='r', linestyle='--',
                              alpha=0.5, label=f'Threshold')
        axes_flat[row_idx, 1].set_xlabel('Frame Index')
        axes_flat[row_idx, 1].set_ylabel('Correlation Power')
        axes_flat[row_idx, 1].set_title('Stage 2: Raw Correlation Power')
        axes_flat[row_idx, 1].legend()
        axes_flat[row_idx, 1].grid(True, alpha=0.3)

        # Plot 3: Histogram of raw correlation power
        axes_flat[row_idx, 2].hist(corr_max_raw, bins=50, alpha=0.7, color='cyan', edgecolor='black')
        if raw_outliers is not None and len(raw_outliers) > 0:
            threshold = np.max(corr_max_raw) - 0.3 * (np.max(corr_max_raw) - np.min(corr_max_raw))
            axes_flat[row_idx, 2].axvline(x=threshold, color='r', linestyle='--',
                              linewidth=2, label=f'Outlier Threshold')
        axes_flat[row_idx, 2].set_xlabel('Correlation Power')
        axes_flat[row_idx, 2].set_ylabel('Count')
        axes_flat[row_idx, 2].set_title('Stage 2: Raw Correlation Distribution')
        axes_flat[row_idx, 2].legend()
        axes_flat[row_idx, 2].grid(True, alpha=0.3)

        deriv_row = 1
    else:
        deriv_row = 0

    # ===== Derivative Correlation Row 1: Shifts =====
    # Plot 1: Shifts from 1st derivative
    axes_flat[deriv_row, 0].plot(frame_indices[normal_mask], shifts_1st[normal_mask],
                   'b.', alpha=0.5, label='Normal')
    if len(outlier_indices) > 0:
        axes_flat[deriv_row, 0].plot(outlier_indices, shifts_1st[outlier_indices],
                       'rx', markersize=10, label='Outliers')
    axes_flat[deriv_row, 0].set_xlabel('Frame Index')
    axes_flat[deriv_row, 0].set_ylabel('Shift (taps)')
    axes_flat[deriv_row, 0].set_title('Stage 5: Shifts from 1st Derivative')
    axes_flat[deriv_row, 0].legend()
    axes_flat[deriv_row, 0].grid(True, alpha=0.3)

    # Plot 2: Shifts from 2nd derivative
    axes_flat[deriv_row, 1].plot(frame_indices[normal_mask], shifts_2nd[normal_mask],
                   'g.', alpha=0.5, label='Normal')
    if len(outlier_indices) > 0:
        axes_flat[deriv_row, 1].plot(outlier_indices, shifts_2nd[outlier_indices],
                       'rx', markersize=10, label='Outliers')
    axes_flat[deriv_row, 1].set_xlabel('Frame Index')
    axes_flat[deriv_row, 1].set_ylabel('Shift (taps)')
    axes_flat[deriv_row, 1].set_title('Stage 5: Shifts from 2nd Derivative')
    axes_flat[deriv_row, 1].legend()
    axes_flat[deriv_row, 1].grid(True, alpha=0.3)

    # Plot 3: Mean shifts
    axes_flat[deriv_row, 2].plot(frame_indices[normal_mask], shifts_mean[normal_mask],
                   'k.', alpha=0.5, label='Normal')
    if len(outlier_indices) > 0:
        axes_flat[deriv_row, 2].plot(outlier_indices, shifts_mean[outlier_indices],
                       'rx', markersize=10, label='Outliers')
    axes_flat[deriv_row, 2].set_xlabel('Frame Index')
    axes_flat[deriv_row, 2].set_ylabel('Shift (taps)')
    axes_flat[deriv_row, 2].set_title('Stage 5: Final Derivative Shifts')
    axes_flat[deriv_row, 2].legend()
    axes_flat[deriv_row, 2].grid(True, alpha=0.3)

    # ===== Derivative Correlation Row 2: Correlation Powers =====
    corr_row = deriv_row + 1

    # Plot 4: Correlation power from 1st derivative
    axes_flat[corr_row, 0].semilogy(frame_indices[normal_mask], corr_max_1st[normal_mask],
                        'b.', alpha=0.5, label='Normal')
    if len(outlier_indices) > 0:
        axes_flat[corr_row, 0].semilogy(outlier_indices, corr_max_1st[outlier_indices],
                           'rx', markersize=10, label='Outliers')
    axes_flat[corr_row, 0].set_xlabel('Frame Index')
    axes_flat[corr_row, 0].set_ylabel('Max Correlation')
    axes_flat[corr_row, 0].set_title('Stage 5: Correlation Power (1st Derivative)')
    axes_flat[corr_row, 0].legend()
    axes_flat[corr_row, 0].grid(True, alpha=0.3)

    # Plot 5: Correlation power from 2nd derivative
    axes_flat[corr_row, 1].semilogy(frame_indices[normal_mask], corr_max_2nd[normal_mask],
                        'g.', alpha=0.5, label='Normal')
    if len(outlier_indices) > 0:
        axes_flat[corr_row, 1].semilogy(outlier_indices, corr_max_2nd[outlier_indices],
                           'rx', markersize=10, label='Outliers')
    axes_flat[corr_row, 1].set_xlabel('Frame Index')
    axes_flat[corr_row, 1].set_ylabel('Max Correlation')
    axes_flat[corr_row, 1].set_title('Stage 5: Correlation Power (2nd Derivative)')
    axes_flat[corr_row, 1].legend()
    axes_flat[corr_row, 1].grid(True, alpha=0.3)

    # Plot 6: Mean correlation power
    axes_flat[corr_row, 2].semilogy(frame_indices[normal_mask], corr_mean[normal_mask],
                        'k.', alpha=0.5, label='Normal')
    if len(outlier_indices) > 0:
        axes_flat[corr_row, 2].semilogy(outlier_indices, corr_mean[outlier_indices],
                           'rx', markersize=10, label='Outliers')
        # Draw threshold line
        threshold = np.percentile(corr_mean, 10)
        axes_flat[corr_row, 2].axhline(y=threshold, color='r', linestyle='--',
                          alpha=0.5, label=f'Threshold (10%ile)')
    axes_flat[corr_row, 2].set_xlabel('Frame Index')
    axes_flat[corr_row, 2].set_ylabel('Mean Correlation')
    axes_flat[corr_row, 2].set_title('Stage 5: Mean Correlation Power')
    axes_flat[corr_row, 2].legend()
    axes_flat[corr_row, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('correlation_alignment_diagnostics.png', dpi=150, bbox_inches='tight')
    print("   Saved diagnostic plot: correlation_alignment_diagnostics.png")
    plt.show()


# For backward compatibility, provide a similar interface to adaptive_align_cir
def adaptive_align_cir_correlation(cir_data, cir_los_part, los_search_start, los_search_end,
                                   sg_window=25, sg_poly=2,
                                   corr_factor=0.5,
                                   interpolate_outliers=True, visualize=False,
                                   **kwargs):
    """
    Wrapper function with similar interface to adaptive_align_cir.

    This uses correlation-based alignment instead of feature-based alignment.
    Note: los_search_start and los_search_end are ignored (for compatibility only).
    """
    return correlation_align_cir(
        cir_data, cir_los_part,
        sg_window, sg_poly, corr_factor,
        interpolate_outliers, visualize
    )
