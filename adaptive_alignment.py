#!/usr/bin/env python3
"""
Adaptive CIR Alignment using derivative-based feature detection.
Handles variable number of detected features and outlier frames.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.ndimage import correlate1d


def find_alignment_features(cir_los_part, sg_window=25, sg_poly=2,
                            slope_threshold=-2.2, value_threshold_1st=30,
                            value_threshold_2nd=28):
    """
    Find alignment features from CIR LoS part using derivatives.

    Parameters:
    -----------
    cir_los_part : ndarray (N_frames, N_taps)
        LoS portion of CIR data
    sg_window : int
        Savitzky-Golay filter window length
    sg_poly : int
        Polynomial order for SG filter
    slope_threshold : float
        Threshold for steep negative slopes (< threshold)
    value_threshold_1st : float
        Amplitude threshold for 1st derivative features
    value_threshold_2nd : float
        Amplitude threshold for 2nd derivative features

    Returns:
    --------
    all_features : list of ndarrays
        Detected feature indices for each frame
    feature_types : list of lists
        Type of each feature ('1st' or '2nd')
    """
    N, L = cir_los_part.shape
    sg_bound = (sg_window - 1) // 2

    # Compute derivatives
    cir_1st_deriv = savgol_filter(cir_los_part, window_length=sg_window,
                                  polyorder=sg_poly, deriv=1, axis=1)
    cir_2nd_deriv = savgol_filter(cir_los_part, window_length=sg_window,
                                  polyorder=sg_poly, deriv=2, axis=1)

    all_features = []
    feature_types = []

    for i in range(N):
        frame_features = []
        frame_types = []

        # ===== 1st Derivative Features =====
        # Find zero crossings in 1st derivative
        zero_cross_1st = np.where(np.diff(np.sign(cir_1st_deriv[i, sg_bound:-sg_bound])))[0]

        if len(zero_cross_1st) > 0:
            # Calculate slopes at zero crossings
            slopes = 100 * np.gradient(cir_1st_deriv[i, sg_bound:-sg_bound])[zero_cross_1st]
            zero_cross_1st += sg_bound

            # Filter by steep negative slope
            steep_neg_mask = slopes < slope_threshold
            steep_indices = zero_cross_1st[steep_neg_mask]

            # Filter by original amplitude
            if len(steep_indices) > 0:
                values = cir_los_part[i, steep_indices]
                high_value_mask = values > value_threshold_1st
                valid_indices = steep_indices[high_value_mask]

                frame_features.extend(valid_indices.tolist())
                frame_types.extend(['1st'] * len(valid_indices))

        # ===== 2nd Derivative Features =====
        # Find zero crossings in 2nd derivative
        zero_cross_2nd = np.where(np.diff(np.sign(cir_2nd_deriv[i, sg_bound:-sg_bound])))[0]

        if len(zero_cross_2nd) > 0:
            zero_cross_2nd += sg_bound

            # Filter by original amplitude
            values = cir_los_part[i, zero_cross_2nd]
            high_value_mask = values > value_threshold_2nd
            high_indices = zero_cross_2nd[high_value_mask]

            # Filter by sign of 1st derivative
            slopes_original = cir_1st_deriv[i, high_indices]
            valid_indices = high_indices[slopes_original < 0]

            frame_features.extend(valid_indices.tolist())
            frame_types.extend(['2nd'] * len(valid_indices))

        all_features.append(np.array(frame_features))
        feature_types.append(frame_types)

    return all_features, feature_types


def align_variable_features(all_features, feature_types, cir_los_part, corr_threshold=0.7):
    """
    Align features using most common feature configuration as reference.

    Logic:
    1. Find most common number of features and feature types
    2. For frames with fewer features: test correlation → outlier or fill with mean
    3. For frames with more features: select closest features to mean

    Parameters:
    -----------
    all_features : list of ndarrays
        Feature indices for each frame (variable length)
    feature_types : list of lists
        Type of each feature ('1st' or '2nd')
    cir_los_part : ndarray (N_frames, N_taps)
        LoS portion of CIR for correlation test
    corr_threshold : float
        Correlation threshold for outlier detection

    Returns:
    --------
    aligned_features : ndarray (N_frames, n_features)
        Aligned features (NaN for outliers)
    n_features : int
        Most common number of features
    outlier_indices : list
        Detected outlier frame indices
    most_common_types : tuple
        Most common feature type configuration
    """
    from collections import Counter

    N = len(all_features)

    # Step 1: Find most common feature length
    all_features_len = [len(f) for f in all_features]
    feature_count = Counter(all_features_len)
    most_common_num_features = feature_count.most_common(1)[0][0]
    num_most_common = feature_count.most_common(1)[0][1]
    percentage_most_common = num_most_common / N * 100

    print(f"   Most common number of features: {most_common_num_features}")
    print(f"   Percentage of frames with {most_common_num_features} features: {percentage_most_common:.2f}%")

    # Step 2: Find most common feature types for the most common length
    feature_types_filtered = [tuple(ft) for i, ft in enumerate(feature_types)
                              if len(all_features[i]) == most_common_num_features]

    if len(feature_types_filtered) > 0:
        feature_types_count = Counter(feature_types_filtered)
        most_common_types = feature_types_count.most_common(1)[0][0]
        print(f"   Most common feature types: {most_common_types}")
    else:
        most_common_types = None
        print(f"   Warning: No frames have {most_common_num_features} features")

    # Step 3: Compute mean features from frames with most common config
    reference_indices = [i for i in range(N)
                        if len(all_features[i]) == most_common_num_features
                        and (most_common_types is None or tuple(feature_types[i]) == most_common_types)]

    if len(reference_indices) > 0:
        reference_features = np.array([all_features[i] for i in reference_indices])
        mean_features = np.mean(reference_features, axis=0)
        print(f"   Reference mean features computed from {len(reference_indices)} frames")
    else:
        # Fallback: use all frames with most common length
        reference_indices = [i for i in range(N) if len(all_features[i]) == most_common_num_features]
        if len(reference_indices) > 0:
            reference_features = np.array([all_features[i] for i in reference_indices])
            mean_features = np.mean(reference_features, axis=0)
            print(f"   Warning: Using all {len(reference_indices)} frames with length {most_common_num_features} as reference")
        else:
            # Ultimate fallback
            mean_features = np.zeros(most_common_num_features)
            print(f"   Warning: No reference frames found, using zeros")

    # Step 4: Initialize aligned features array
    aligned_features = np.full((N, most_common_num_features), np.nan)
    outlier_indices = []

    # Step 5: Separate most common types into 1st and 2nd parts
    if most_common_types is None:
        # Fallback: treat all as 2nd derivative
        most_common_types = ('2nd',) * most_common_num_features

    most_common_1st_indices = [j for j, t in enumerate(most_common_types) if t == '1st']
    most_common_2nd_indices = [j for j, t in enumerate(most_common_types) if t == '2nd']
    mean_features_1st = mean_features[most_common_1st_indices] if len(most_common_1st_indices) > 0 else np.array([])
    mean_features_2nd = mean_features[most_common_2nd_indices] if len(most_common_2nd_indices) > 0 else np.array([])

    # Step 6: Process each frame
    reference_frame = cir_los_part[reference_indices[0]] if len(reference_indices) > 0 else cir_los_part[0]

    for i in range(N):
        current_features = all_features[i]
        current_types = feature_types[i]
        n_features = len(current_features)

        # Check if feature types match most common
        if n_features == most_common_num_features and tuple(current_types) == most_common_types:
            # Exact match - use as is
            aligned_features[i] = current_features
            continue

        # Separate current features by type
        current_1st_features = np.array([current_features[j] for j, t in enumerate(current_types) if t == '1st'])
        current_2nd_features = np.array([current_features[j] for j, t in enumerate(current_types) if t == '2nd'])

        # Check if this is a potential outlier (too few features overall)
        if n_features < most_common_num_features * 0.5:  # Less than half expected features
            # Test correlation
            corr = np.correlate(reference_frame, cir_los_part[i], mode='valid')[0]
            corr_norm = corr / (np.linalg.norm(reference_frame) * np.linalg.norm(cir_los_part[i]) + 1e-10)

            if corr_norm < corr_threshold:
                # True outlier - mark as NaN
                outlier_indices.append(i)
                # aligned_features[i] stays as NaN
                continue

        # Process 1st derivative features
        aligned_1st = np.full(len(most_common_1st_indices), np.nan)
        if len(mean_features_1st) > 0:
            if len(current_1st_features) == len(mean_features_1st):
                # Same length - use as is
                aligned_1st = current_1st_features
            elif len(current_1st_features) > len(mean_features_1st):
                # More features - pick closest ones
                for j, mean_val in enumerate(mean_features_1st):
                    distances = np.abs(current_1st_features - mean_val)
                    closest_idx = np.argmin(distances)
                    aligned_1st[j] = current_1st_features[closest_idx]
            elif len(current_1st_features) > 0:
                # Fewer features (but not zero) - calculate offset and fill missing
                # Match current features to mean positions
                matched_indices = []
                matched_values = []
                for feat_val in current_1st_features:
                    distances = np.abs(mean_features_1st - feat_val)
                    best_match = np.argmin(distances)
                    matched_indices.append(best_match)
                    matched_values.append(feat_val)

                # Calculate offset from matched features
                offsets = []
                for j, idx in enumerate(matched_indices):
                    offset = matched_values[j] - mean_features_1st[idx]
                    offsets.append(offset)
                mean_offset = np.mean(offsets) if len(offsets) > 0 else 0

                # Fill aligned features
                for j in range(len(mean_features_1st)):
                    if j in matched_indices:
                        # Use matched value
                        match_pos = matched_indices.index(j)
                        aligned_1st[j] = matched_values[match_pos]
                    else:
                        # Fill with mean + offset
                        aligned_1st[j] = mean_features_1st[j] + mean_offset
            # else: Zero 1st features - will be filled from 2nd derivative offset later

        # Process 2nd derivative features
        aligned_2nd = np.full(len(most_common_2nd_indices), np.nan)
        mean_offset_2nd = 0  # Initialize offset

        if len(mean_features_2nd) > 0:
            if len(current_2nd_features) == len(mean_features_2nd):
                # Same length - use as is
                aligned_2nd = current_2nd_features
                # Calculate offset even when lengths match (for filling 1st if needed)
                mean_offset_2nd = np.mean(current_2nd_features - mean_features_2nd)
            elif len(current_2nd_features) > len(mean_features_2nd):
                # More features - pick closest ones
                for j, mean_val in enumerate(mean_features_2nd):
                    distances = np.abs(current_2nd_features - mean_val)
                    closest_idx = np.argmin(distances)
                    aligned_2nd[j] = current_2nd_features[closest_idx]
                # Calculate offset from selected features
                mean_offset_2nd = np.mean(aligned_2nd - mean_features_2nd)
            elif len(current_2nd_features) > 0:
                # Fewer features (but not zero) - calculate offset and fill missing
                # Match current features to mean positions
                matched_indices = []
                matched_values = []
                for feat_val in current_2nd_features:
                    distances = np.abs(mean_features_2nd - feat_val)
                    best_match = np.argmin(distances)
                    matched_indices.append(best_match)
                    matched_values.append(feat_val)

                # Calculate offset from matched features
                offsets = []
                for j, idx in enumerate(matched_indices):
                    offset = matched_values[j] - mean_features_2nd[idx]
                    offsets.append(offset)
                mean_offset_2nd = np.mean(offsets) if len(offsets) > 0 else 0

                # Fill aligned features
                for j in range(len(mean_features_2nd)):
                    if j in matched_indices:
                        # Use matched value
                        match_pos = matched_indices.index(j)
                        aligned_2nd[j] = matched_values[match_pos]
                    else:
                        # Fill with mean + offset
                        aligned_2nd[j] = mean_features_2nd[j] + mean_offset_2nd
            # else: Zero 2nd features - will stay as NaN

        # If 1st derivative features are missing, use 2nd derivative offset to fill
        if len(current_1st_features) == 0 and len(mean_features_1st) > 0 and len(current_2nd_features) > 0:
            aligned_1st = mean_features_1st + mean_offset_2nd

        # Combine 1st and 2nd derivative features back into original order
        result = np.full(most_common_num_features, np.nan)
        result[most_common_1st_indices] = aligned_1st
        result[most_common_2nd_indices] = aligned_2nd
        aligned_features[i] = result

    return aligned_features, most_common_num_features, outlier_indices, most_common_types


def detect_outliers(cir_los_part, all_features, corr_threshold=0.7):
    """
    Detect outlier frames using correlation with reference frame.

    Parameters:
    -----------
    cir_los_part : ndarray (N_frames, N_taps)
        LoS portion of CIR data
    all_features : list of ndarrays
        Feature indices for each frame
    corr_threshold : float
        Correlation threshold (< threshold = outlier)

    Returns:
    --------
    outlier_indices : list
        Indices of outlier frames
    """
    N = cir_los_part.shape[0]
    reference = cir_los_part[0]
    outlier_indices = []

    for i in range(N):
        # If no features detected, check correlation
        if len(all_features[i]) == 0:
            # Compute normalized cross-correlation
            corr = np.correlate(reference, cir_los_part[i], mode='valid')[0]
            corr_norm = corr / (np.linalg.norm(reference) * np.linalg.norm(cir_los_part[i]))

            if corr_norm < corr_threshold:
                outlier_indices.append(i)

    return outlier_indices


def fill_missing_features(aligned_features, outlier_indices):
    """
    Fill missing features with mean values, excluding outliers.

    Parameters:
    -----------
    aligned_features : ndarray (N_frames, max_n_features)
        Aligned features with NaN for missing
    outlier_indices : list
        Indices of outlier frames to exclude from mean

    Returns:
    --------
    filled_features : ndarray (N_frames, max_n_features)
        Features with NaN filled by column-wise mean
    """
    filled_features = aligned_features.copy()
    N, n_features = aligned_features.shape

    # Create mask for valid (non-outlier) frames
    valid_mask = np.ones(N, dtype=bool)
    valid_mask[outlier_indices] = False

    # Fill each feature column with mean of valid frames
    for j in range(n_features):
        col = aligned_features[:, j]
        valid_col = col[valid_mask]

        # Compute mean excluding NaN
        col_mean = np.nanmean(valid_col)

        # Fill NaN values with mean
        nan_mask = np.isnan(filled_features[:, j])
        filled_features[nan_mask, j] = col_mean

    return filled_features


def compute_shifts(filled_features):
    """
    Compute shift amounts for alignment based on filled features.

    Parameters:
    -----------
    filled_features : ndarray (N_frames, max_n_features)
        Feature positions for each frame

    Returns:
    --------
    shifts : ndarray (N_frames,)
        Shift amount for each frame
    """
    # Compute reference as mean across all frames
    reference = np.nanmean(filled_features, axis=0)

    # Compute shift for each frame as mean deviation from reference
    shifts = np.nanmean(reference - filled_features, axis=1)

    return shifts


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


def adaptive_align_cir(cir_data, los_search_start, los_search_end,
                       sg_window=25, sg_poly=2, slope_threshold=-2.2,
                       value_threshold_1st=30, value_threshold_2nd=28,
                       corr_threshold=0.9,
                       interpolate_outliers=True, visualize=False):
    """
    Perform adaptive CIR alignment with outlier detection.

    Parameters:
    -----------
    cir_data : ndarray (N_frames, L_taps), complex value
        Full CIR data to be aligned
    los_search_start : int
        Start index of LoS portion in full CIR
    los_search_end : int
        End index of LoS portion in full CIR
    sg_window : int
        Savitzky-Golay filter window length
    sg_poly : int
        Savitzky-Golay polynomial order
    slope_threshold : float
        Slope threshold for 1st derivative peak detection
    value_threshold_1st : float
        Amplitude threshold for 1st derivative features
    value_threshold_2nd : float
        Amplitude threshold for 2nd derivative features
    corr_threshold : float
        Correlation threshold for outlier detection
    interpolate_outliers : bool
        Whether to interpolate outlier frames using neighbors
    visualize : bool
        Whether to plot diagnostic figures

    Returns:
    --------
    aligned_cir : ndarray (N_frames, L_taps)
        Aligned CIR data (with outliers interpolated if requested)
    shifts : ndarray (N_frames,)
        Applied shift amounts
    outlier_indices : list
        Detected outlier frame indices
    """
    N, L = cir_data.shape
    cir_los_part = np.abs(cir_data[:, los_search_start:los_search_end])

    # Step 1: Find alignment features
    print("Finding alignment features from derivatives...")
    all_features, feature_types = find_alignment_features(
        cir_los_part, sg_window, sg_poly, slope_threshold,
        value_threshold_1st, value_threshold_2nd
    )

    print(f"   Detected features: min={min(len(f) for f in all_features)}, "
          f"max={max(len(f) for f in all_features)}, "
          f"mean={np.mean([len(f) for f in all_features]):.1f}")

    # Step 2: Align features across frames (includes outlier detection)
    print("Aligning features using most common configuration...")
    aligned_features, n_features, outlier_indices, most_common_types = align_variable_features(
        all_features, feature_types, cir_los_part, corr_threshold
    )
    print(f"   Most common configuration: {n_features} features with types {most_common_types}")
    print(f"   Found {len(outlier_indices)} outlier frames: {outlier_indices}")

    # Step 3: Fill missing features
    print("Filling missing features with mean values...")
    filled_features = fill_missing_features(aligned_features, outlier_indices)

    # Step 4: Compute shifts
    print("Computing shift amounts...")
    shifts = compute_shifts(filled_features)

    # Step 5: Apply shifts
    print("Applying shifts to align CIR data...")
    aligned_cir = np.zeros_like(cir_data, dtype=cir_data.dtype)
    for i in range(N):
        aligned_cir[i] = np.roll(cir_data[i], int(np.round(shifts[i])))

    # Step 6: Interpolate outliers (optional)
    if interpolate_outliers and len(outlier_indices) > 0:
        print("Interpolating outlier frames using neighbors...")
        aligned_cir = interpolate_outlier_frames(aligned_cir, outlier_indices)

    # Visualization
    if visualize:
        plot_alignment_diagnostics(cir_los_part, all_features, aligned_features,
                                   filled_features, shifts, outlier_indices)

    return aligned_cir, shifts, outlier_indices


def plot_alignment_diagnostics(cir_los_part, all_features, aligned_features,
                               filled_features, shifts, outlier_indices):
    """Plot diagnostic figures for alignment process."""

    N = cir_los_part.shape[0]

    # Plot 1: Feature distribution
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    feature_counts = [len(f) for f in all_features]
    plt.hist(feature_counts, bins=range(max(feature_counts) + 2), edgecolor='black')
    plt.xlabel('Number of Features Detected')
    plt.ylabel('Number of Frames')
    plt.title('Feature Count Distribution')
    plt.grid(True, alpha=0.3)

    # Plot 2: Feature positions over frames
    plt.subplot(1, 3, 2)
    for i in range(N):
        if i in outlier_indices:
            color = 'red'
            marker = 'x'
            alpha = 0.5
        else:
            color = 'blue'
            marker = 'o'
            alpha = 0.3

        if len(all_features[i]) > 0:
            plt.scatter([i] * len(all_features[i]), all_features[i],
                       c=color, marker=marker, alpha=alpha, s=20)

    plt.xlabel('Frame Index')
    plt.ylabel('Feature Position (tap index)')
    plt.title('Detected Features (red = outliers)')
    plt.grid(True, alpha=0.3)

    # Plot 3: Shift amounts
    plt.subplot(1, 3, 3)
    plt.plot(shifts, 'b-', linewidth=1, alpha=0.7)
    if len(outlier_indices) > 0:
        plt.scatter(outlier_indices, shifts[outlier_indices],
                   c='red', marker='x', s=100, zorder=5, label='Outliers')
    plt.xlabel('Frame Index')
    plt.ylabel('Shift Amount (taps)')
    plt.title('Computed Shifts')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Plot 4: Aligned CIR LoS part
    plt.figure(figsize=(10, 6))
    plt.imshow(cir_los_part, aspect='auto', cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Amplitude')
    plt.xlabel('Tap Index')
    plt.ylabel('Frame Index')
    plt.title('CIR LoS Part (Before Alignment)')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Adaptive CIR Alignment Module")
    print("Import this module to use adaptive alignment functions.")
