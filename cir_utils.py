import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pywt
from scipy.signal import savgol_filter
import os
import time
import scienceplots
from scipy.signal import max_len_seq
from scipy.signal import find_peaks, stft, medfilt
from scipy.fft import fft, fftfreq
from ssqueezepy import ssq_cwt, ssq_stft
from ssqueezepy.experimental import scale_to_freq
from scipy.signal import butter, filtfilt
from numpy.fft import fft, ifft, fftshift, fftfreq
from scipy.interpolate import UnivariateSpline, interp1d
import re
from pathlib import Path
from types import SimpleNamespace

DEFAULT_CIR_CONFIG = {
    "input_folder": "./cir_files/",
    "input_file": None,
    "discard_reading": 0,
    "upsample_factor": 32,
    "pn_code": "./kasami_code/kasami_packed_16bit_120.h",
    "shift_amount": 1,
    "starting_idx": 0,
    "peak_wnd": 3000,
    "SHOW_FIG": False,
}


class ConfigNamespace(SimpleNamespace):
    """Attribute-based config with a dict-like ``get`` helper."""
    def get(self, key, default=None):  # pragma: no cover - trivial
        return getattr(self, key, default)

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf, DictConfig
from typing import Union, Any

def _normalize_config(config_obj: Union[dict, Any]) -> Union[dict, Any, ConfigNamespace]:
    """Merge defaults and ensure attribute access for the returned config."""
    base_cfg = OmegaConf.create(DEFAULT_CIR_CONFIG)
    cfg_obj = config_obj
    if not isinstance(cfg_obj, DictConfig):
        cfg_obj = OmegaConf.create(cfg_obj)
    merged_cfg = OmegaConf.merge(base_cfg, cfg_obj)
    return merged_cfg

def load_cir_config(
    config_name: str = "cir_processing",
    config_dir: str | os.PathLike = "config",
) -> dict:
    """Load CIR processing parameters, preferring Hydra/OmegaConf."""
    config_path = Path(config_dir)
    if not config_path.exists():
        raise FileNotFoundError(f"Missing CIR config directory: {config_path}")

    # Hydra is global-stateful; make sure we clean before reusing (e.g. notebooks).
    gh = GlobalHydra.instance()
    if gh.is_initialized():  # pragma: no branch - simple guard
        gh.clear()
    with initialize_config_dir(version_base=None, config_dir=str(config_path.resolve())):
        cfg = compose(config_name=config_name)
    
    config = _normalize_config(cfg)
    
    # Dynamically set corr_indices based on SHOW_FIG value
    if hasattr(config, 'SHOW_FIG') and hasattr(config, 'corr_indices_range') and hasattr(config, 'corr_indices_single'):
        # Disable struct mode to allow adding new keys
        OmegaConf.set_struct(config, False)
        
        if config.SHOW_FIG:
            config.corr_indices = config.corr_indices_single
        else:
            # Use corr_indices_range to create np.arange and convert to list
            start, stop, step = config.corr_indices_range
            config.corr_indices = np.arange(start, stop, step).tolist()
    return config

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

def butter_highpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = filtfilt(b, a, data)
    return y

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs  # Nyquist Frequency
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def autocorr_same_length(raw, sampled, showFig=False, title=None):
    """Compute the autocorrelation of a signal."""
    n = len(raw)
    raw_spec = fft(raw, n)
    corr = np.abs(ifft(raw_spec * np.conj(fft(sampled[0:n], n))))

    # corr /= n
    # normalize the correlation by power
    norm_factor = np.linalg.norm(sampled[0:n]) * np.linalg.norm(raw)
    corr = corr / 1
    if showFig:
        plt.figure(figsize=(5, 3))
        # plt.plot(np.arange(-n/2+1, n/2+1), fftshift(corr), '.-')
        plt.plot((corr), '.-')
        plt.margins(0.1, 0.1)
        plt.grid(True)
        if title is not None:
            plt.savefig(title, dpi=300)
        else:
            plt.show()
        plt.close()
    return corr

def reverse_autocorr_same_length(corr, raw):
    """
    Reverse function of autocorr_same_length.
    Recovers the sampled signal from correlation result and raw signal.
    
    Parameters:
    - corr: correlation result (real-valued array)
    - raw: original raw signal (real-valued array)
    
    Returns:
    - sampled: recovered sampled signal (real-valued array)
    
    Note: This assumes all signals are real-valued and uses a simple deconvolution approach.
    The recovery may not be perfect due to information loss in the absolute value operation.
    """
    n = len(raw)
    
    # Compute FFT of raw signal
    raw_spec = fft(raw, n)
    
    # Compute FFT of correlation result
    corr_spec = fft(corr, n)
    
    # Attempt to recover the sampled signal spectrum
    # Since corr = |ifft(raw_spec * conj(sampled_spec))|, we need to invert this
    # This is an approximation since we lost phase information in the abs() operation
    
    # Avoid division by zero
    raw_spec_safe = raw_spec + 1e-12
    
    # Simple deconvolution: sampled_spec ≈ conj(corr_spec / raw_spec)
    # This is a rough approximation due to the absolute value operation
    sampled_spec = np.conj(corr_spec / raw_spec_safe)
    
    # Convert back to time domain
    sampled = np.real(ifft(sampled_spec))
    
    return sampled

def xcorr_linear(x, y, showFig=False, title=None):
    x = np.asarray(x); y = np.asarray(y)
    x = x - np.mean(x); y = y - np.mean(y)     # remove DC
    n = len(x); nfft = 1 << int(np.ceil(np.log2(2*n-1)))
    X = np.fft.fft(x, nfft); Y = np.fft.fft(y, nfft)
    r = np.fft.ifft(X * np.conj(Y), nfft)      # linear corr via zero-padding
    r = np.fft.ifftshift(r)                    # center zero lag
    # r = r / (np.linalg.norm(x)*np.linalg.norm(y) + 1e-12)  # normalize
    lags = np.arange(-(nfft//2), nfft//2)
    half_r = r[0:nfft//2+1] + r[nfft//2-1:]
    if showFig:
        plt.figure(figsize=(5, 3))
        plt.plot(np.abs(half_r), '.-')
        # plt.xlim(np.argmax(np.abs(half_r))-50, np.argmax(np.abs(half_r))+50)
        plt.title('Cross-correlation')
        plt.xlabel('Lag')
        plt.ylabel('Normalized correlation')
        plt.grid()
        plt.show()
    return half_r

def detect_periodic_pn(s, p):
    L = len(p)
    N = len(s)
    K = N // L  # 使用完整周期数
    s = s[:K*L].reshape(K, L)
    s_fold = s.sum(axis=0)                # 按周期折叠

    # L 点循环相关: r[m] = sum_r s_fold[r] * conj(p[(r-m) mod L])
    R = np.fft.ifft(np.fft.fft(s_fold) * np.conj(np.fft.fft(p)))
    r = np.asarray(R)
    m_hat = int(np.argmax(np.abs(r)))
    rho = np.abs(r[m_hat]) / (np.linalg.norm(s_fold) * np.linalg.norm(p))

    # 估计 PSR
    off = np.delete(np.abs(r), m_hat)
    psr = np.abs(r[m_hat]) / (off.mean() + 3*off.std() + 1e-12)
    return np.abs(r)**2, rho, psr

def calculate_pmr(corr_power, window_size):
    # roll the correlation result to center the peak
    peak_idx = np.argmax(corr_power)
    full_len = len(corr_power)
    corr_power = np.roll(corr_power, -peak_idx + full_len // 2)
    peak_idx = full_len // 2  # Now the peak is at the center
    window_amplitude = np.concatenate((corr_power[peak_idx-window_size:peak_idx-1], corr_power[peak_idx+1:peak_idx+window_size]))
    pmr = corr_power[peak_idx] / np.mean(window_amplitude)
    return pmr

def generate_resampled_template_with_phase(
    pn_sequence: np.ndarray,
    drift_ppm: float,
    samples_per_symbol: int,
    num_output_samples: int,
    initial_phase: float = 0.0,
    integration_duration: int = 0,
) -> np.ndarray:
    """
    Generates a resampled PN sequence template to account for clock drift and initial phase.

    Args:
        pn_sequence: The original, ideal bipolar (+1/-1) PN sequence.
        drift_ppm: The hypothetical clock drift in Parts Per Million (PPM).
        samples_per_symbol: The number of samples per symbol at zero drift.
        num_output_samples: The total number of samples for the output template.
        initial_phase: The initial phase offset as a fraction of one symbol period (0.0 to 1.0).
                       This accounts for sub-symbol alignment.
        integration_duration: The duration over which to integrate samples (default is 0, meaning no integration).

    Returns:
        A resampled bipolar (+1/-1) numpy array of length num_output_samples.
    """
    drift_factor = 1.0 + (drift_ppm / 1_000_000.0)
    
    # Convert the fractional phase into an offset in units of samples.
    phase_offset_samples = initial_phase * samples_per_symbol
    
    # Create an array of receiver sample indices: [0, 1, 2, ...]
    n = np.arange(num_output_samples) # start_sampled_time

    # apply drift and initial phase
    shifted_drifted_start_time = (n + phase_offset_samples) / (samples_per_symbol * drift_factor)
    shifted_drifted_end_time = shifted_drifted_start_time + (integration_duration / drift_factor)

    # for those int(floor(start_time)) == int(floor(end_time)), we can directly assign the value
    # otherwise, we need to do a linear interpolation between the two symbols
    resampled_sequence = np.zeros(num_output_samples)
    same_indices = (np.floor(shifted_drifted_start_time) == np.floor(shifted_drifted_end_time))
    resampled_sequence[same_indices] = pn_sequence[np.floor(shifted_drifted_start_time[same_indices]).astype(int) % len(pn_sequence)]
    
    # For the rest, we do linear interpolation
    diff_indices = ~same_indices
    start_indices = np.floor(shifted_drifted_start_time[diff_indices]).astype(int) % len(pn_sequence)
    end_indices = np.floor(shifted_drifted_end_time[diff_indices]).astype(int) % len(pn_sequence)
    start_weights = 1 - (shifted_drifted_start_time[diff_indices] - np.floor(shifted_drifted_start_time[diff_indices]))
    end_weights = shifted_drifted_end_time[diff_indices] - np.floor(shifted_drifted_end_time[diff_indices])
    resampled_sequence[diff_indices] = (pn_sequence[start_indices] * start_weights + pn_sequence[end_indices] * end_weights) / (start_weights + end_weights)
    return resampled_sequence

def plot_specrum(signal, fs=500, title='Spectrum', xlim=None):
    """
    Plot the spectrum of a 1D time series.

    Parameters:
    - signal: 1D np.array, complex or real time series
    - fs: Sampling frequency (Hz), default is 1.0 (for normalized time index)
    - title: Title for the plot
    """
    signal = np.asarray(signal)

    # Apply Hanning window
    window = np.hanning(len(signal))
    signal = signal * window

    # Compute FFT
    N = len(signal)
    dt = 1.0 / fs  # Sampling interval
    freqs = fftfreq(N, dt)[:N//2]  # Frequency axis for real FFT
    FFT_vals = fft(signal)[:N//2]  # Take only the positive frequencies

    # Plot
    plt.figure(figsize=(5, 3))
    if xlim is not None:
        freq_min = xlim[0]
        freq_max = xlim[1]
        freq_mask = (freqs >= freq_min) & (freqs <= freq_max)
        plt.plot(freqs[freq_mask], 20*np.log10(np.abs(FFT_vals[freq_mask])), label='Magnitude')
    else:
        plt.plot(freqs, 20*np.log10(np.abs(FFT_vals)), label='Magnitude')
    plt.title(title)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()

def plot_raw_cir_tap_series(cir_eval, tap):
    plt.figure(figsize=(11, 3))
    plt.plot(np.real(cir_eval[:, tap]), label='Real Part')
    plt.legend()
    plt.show()
    plt.close()

    plt.figure(figsize=(11, 3))
    plt.plot(np.imag(cir_eval[:, tap]), label='Imag Part')
    plt.legend()
    plt.show()
    plt.close()

    plot_specrum(np.real(cir_eval[:, tap]), fs=1000)


def get_latest_file(folder):
    files = [f for f in os.listdir(folder) if f.endswith('.npy')]
    if not files:
        return None
    latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(folder, x)))
    return os.path.join(folder, latest_file)


def get_PN(bits=7, spacing=1):
    seq = max_len_seq(bits)[0]*2-1
    # if spacing > 1:
    #     enlarged_seq = np.zeros(len(seq) * spacing, dtype=int)
    #     for i in range(len(seq)):
    #         enlarged_seq[i*spacing:(i+1)*spacing] = [seq[i]] * spacing
    #     enlarged_seq[0::2] = enlarged_seq[0::2] * -1  # Invert every other bit
    #     print(seq)
    #     plt.plot(enlarged_seq, label=f"PN Sequence (bits={bits}, spacing={spacing})")
    #     plt.show()
    return seq


def convert_clock_offset(clock_offset: np.ndarray) -> list[np.ndarray]:
    """
    Convert int16 array of clock offset (in PPM) to parts-per-hundred-million (PPHM).
    Equivalent to:
        cfo_pphm = (int)((float)cfo * (CLOCK_OFFSET_PPM_TO_RATIO * 1e6 * 100));

    Parameters:
    - cfo_array: np.ndarray of dtype int16

    Returns:
    - np.ndarray of converted values (float)
    """
    CLOCK_OFFSET_PPM_TO_RATIO = 1.0 / (1 << 26)
    factor = CLOCK_OFFSET_PPM_TO_RATIO * 1e6 * 100
    HERTZ_TO_PPM_MULTIPLIER_CHAN_5 = (-1.0e6 / 6489.6e6)
    HERTZ_TO_PPM_MULTIPLIER_CHAN_9 = (-1.0e6 / 7987.2e6)
    clock_offset_ppm = clock_offset.astype(np.float32) * factor
    clock_offset_hz = clock_offset_ppm / HERTZ_TO_PPM_MULTIPLIER_CHAN_9 / 100
    clock_offset_hz = - clock_offset * (7987.2e6) / (2**26)
    return clock_offset_ppm, clock_offset_hz

def generate_mls_lfsr_10(length=(2**10 - 1)):
    """
    Generate a maximum length sequence using a 10-bit LFSR.
    Primitive polynomial: x^10 + x^3 + 1
    """
    lfsr = 0x3FF  # Non-zero seed (10 bits all set)
    mask = (1 << 10) - 1
    sequence = np.zeros(length, dtype=int)

    for i in range(length):
        output_bit = lfsr & 1
        sequence[i] = output_bit
        # Feedback from bit positions 10 and 3 (9 and 2 in 0-indexed)
        feedback = ((lfsr >> 9) ^ (lfsr >> 2)) & 1
        lfsr = ((lfsr << 1) | feedback) & mask

    bipolar_sequence = 2 * sequence - 1
    return bipolar_sequence

def generate_mls_lfsr_11(length=(2**11 - 1)):
    """
    Generate a maximum length sequence using an 11-bit LFSR.
    Primitive polynomial: x^11 + x^2 + 1
    """
    lfsr = 0x7FF  # Non-zero seed (11 bits all set)
    mask = (1 << 11) - 1
    sequence = np.zeros(length, dtype=int)

    for i in range(length):
        output_bit = lfsr & 1
        sequence[i] = output_bit
        # Feedback from bit positions 11 and 2 (10 and 1 in 0-indexed)
        feedback = ((lfsr >> 10) ^ (lfsr >> 1)) & 1
        lfsr = ((lfsr << 1) | feedback) & mask

    bipolar_sequence = 2 * sequence - 1
    return bipolar_sequence

def generate_mls_lfsr_12(length=(2**12 - 1)):
    """
    Generate a maximum length sequence using a 12-bit LFSR.
    Primitive polynomial: x^12 + x^6 + x^4 + x + 1
    """
    lfsr = 0xFFF  # Non-zero seed (12 bits all set)
    mask = (1 << 12) - 1
    sequence = np.zeros(length, dtype=int)

    for i in range(length):
        output_bit = lfsr & 1
        sequence[i] = output_bit
        # Feedback from bit positions 12, 6, 4, 1 (11, 5, 3, 0 in 0-indexed)
        feedback = ((lfsr >> 11) ^ (lfsr >> 5) ^ (lfsr >> 3) ^ lfsr) & 1
        lfsr = ((lfsr << 1) | feedback) & mask

    bipolar_sequence = 2 * sequence - 1
    return bipolar_sequence

def generate_mls_lfsr_13(length=(2**13 - 1)):
    """
    Generate a maximum length sequence using a 13-bit LFSR.
    Primitive polynomial: x^13 + x^4 + x^3 + x + 1
    """
    lfsr = 0x1FFF  # Non-zero seed (13 bits all set)
    mask = (1 << 13) - 1
    sequence = np.zeros(length, dtype=int)

    for i in range(length):
        output_bit = lfsr & 1
        sequence[i] = output_bit
        # Feedback from bit positions 13, 4, 3, 1 (12, 3, 2, 0 in 0-indexed)
        feedback = ((lfsr >> 12) ^ (lfsr >> 3) ^ (lfsr >> 2) ^ lfsr) & 1
        lfsr = ((lfsr << 1) | feedback) & mask

    bipolar_sequence = 2 * sequence - 1
    return bipolar_sequence

def generate_mls_lfsr_14():
    sequence = []
    lfsr = 0x3FFF  # Non-zero seed for 14-bit LFSR
    mask = 0x3FFF  # 14-bit mask
    sequence_length = (1 << 14) - 1  # 16383

    for _ in range(sequence_length):
        current_bit = lfsr & 0x1
        sequence.append(current_bit)

        # Primitive poly: x^14 + x^13 + x^12 + x^2 + 1
        bit = ((lfsr >> 13) ^ (lfsr >> 12) ^ (lfsr >> 11) ^ (lfsr >> 1)) & 0x1
        lfsr = ((lfsr << 1) | bit) & mask

    bipolar_sequence = 2 * np.array(sequence) - 1
    return bipolar_sequence

def generate_mls_lfsr_16():
    sequence = []
    lfsr = 0xFFFF  # Non-zero seed for 16-bit LFSR
    mask = 0xFFFF  # 16-bit mask to limit to 16 bits
    sequence_length = (1 << 16) - 1  # 65535

    for _ in range(sequence_length):
        # Extract output bit (usually the LSB)
        current_bit = lfsr & 0x1
        sequence.append(current_bit)

        # Compute feedback bit using primitive poly: x^16 + x^14 + x^13 + x^11 + 1
        bit = ((lfsr >> 15) ^ (lfsr >> 13) ^ (lfsr >> 12) ^ (lfsr >> 10)) & 0x1

        # Shift left and insert the feedback bit
        lfsr = ((lfsr << 1) | bit) & mask
    bipolar_sequence = 2 * np.array(sequence) - 1
    return bipolar_sequence

def generate_mls_lfsr_17():
    sequence = []
    lfsr = 0x1FFFF  # Non-zero 17-bit initial value
    mask = 0x1FFFF  # 17-bit mask
    sequence_length = (1 << 17) - 1  # 131071

    for _ in range(sequence_length):
        current_bit = lfsr & 0x1
        sequence.append(current_bit)

        # Primitive polynomial: x^17 + x^3 + 1
        bit = ((lfsr >> 16) ^ (lfsr >> 2)) & 0x1
        lfsr = ((lfsr << 1) | bit) & mask

    bipolar_sequence = 2 * np.array(sequence) - 1
    return bipolar_sequence

def generate_mls_lfsr_18(length=(2**18 - 1)):
    """
    Generate a maximum length sequence using a 18-bit LFSR with taps at bit 18 and 7
    Polynomial: x^18 + x^7 + 1
    """
    lfsr = 0x1FFFF  # Non-zero seed (18 bits all set)
    mask = (1 << 18) - 1
    sequence = np.zeros(length, dtype=int)

    for i in range(length):
        output_bit = lfsr & 1
        sequence[i] = output_bit
        # Feedback from bit positions 18 and 7 (17 and 6 in 0-indexed Python)
        feedback = ((lfsr >> 17) ^ (lfsr >> 6)) & 1
        lfsr = ((lfsr << 1) | feedback) & mask

    # Convert to bipolar: {0,1} → {-1,+1}
    bipolar_sequence = 2 * sequence - 1
    return bipolar_sequence

def read_packed_kasami_h(filename):
    with open(filename, 'r') as f:
        content = f.read()

    # Extract all hex bytes in the array
    hex_bytes = re.findall(r'0x([0-9A-Fa-f]{2})', content)
    byte_array = np.array([int(b, 16) for b in hex_bytes], dtype=np.uint8)

    # Extract declared bit length from the header file
    match = re.search(r'const\s+uint32_t\s+kasami_bit_len\s*=\s*(\d+);', content)
    bit_len = int(match.group(1)) if match else len(byte_array) * 8

    # Unpack bits
    unpacked = np.unpackbits(byte_array)[:bit_len].astype(int)

    return unpacked  # array of 0s and 1s

def read_packed_kasami_h_lsb(filename):
    with open(filename, 'r') as f:
        content = f.read()

    # Extract all hex bytes in the array
    hex_bytes = re.findall(r'0x([0-9A-Fa-f]{2})', content)
    byte_array = np.array([int(b, 16) for b in hex_bytes], dtype=np.uint8)

    # Extract declared bit length from the header file
    match = re.search(r'const\s+uint32_t\s+kasami_bit_len\s*=\s*(\d+);', content)
    bit_len = int(match.group(1)) if match else len(byte_array) * 8

    # --- CHANGED: Added bitorder='little' ---
    # Input 0x88 -> [0, 0, 0, 1, 0, 0, 0, 1] 
    # This matches reading (byte >> 0) & 1, then (byte >> 1) & 1...
    unpacked = np.unpackbits(byte_array, bitorder='little')[:bit_len].astype(int)

    return unpacked

def load_file(INPUT_FILE=None, discard_secs=2, sampling_rate=500, show_frame_interval=False):
    INPUT_FOLDER = "./cir_files/"
    if INPUT_FILE is None:
        INPUT_FILE = get_latest_file(INPUT_FOLDER + 'rx/')
    print(f"Loading file: {INPUT_FILE}")
    time_stamp_suffix = INPUT_FILE.split('cir_data_')[-1].split('.')[0]  # Extract timestamp from filename
    rx_cir = np.load(INPUT_FILE, allow_pickle=True)
    diag_data = np.load(INPUT_FOLDER + 'diag/diag_data_' + time_stamp_suffix + '_rx.npy', allow_pickle=True)
    cir_datas = rx_cir[0:len(diag_data)] # if RX else tx_cir[0:len(diag_data)]
    discard_points = int(discard_secs * sampling_rate)
    cir_datas = cir_datas[discard_points:]
    diag_data = diag_data[discard_points:]
    start_time = diag_data[discard_points]['timestamp']
    tx_start_time = diag_data[discard_points]['tx_timestamp']
    frame_idx_list = [diag['D'] for diag in diag_data]
    clockOffset = [diag['CO'] for diag in diag_data]
    carrierInterger = np.array([diag['CI'] for diag in diag_data])
    clockOffsetPPM, clockOffsetHertz = convert_clock_offset(np.array(clockOffset))
    relative_times = [d['timestamp'] - start_time for d in diag_data]
    tx_relative_times = [d['tx_timestamp'] - tx_start_time for d in diag_data]
    
    frame_interval = np.array([i%17208 for i in np.diff(relative_times)])
    tx_frame_interval = np.array([i%17208 for i in np.diff(tx_relative_times)])

    p_los_list = np.array([diag['p_los']/(2**11) for diag in diag_data])

    accumulated_phase = 0.0
    corrected_CIRs = np.zeros_like(cir_datas, dtype=complex)

    for t in range(len(clockOffsetHertz)):
        if t > 0:
            dt = (relative_times[t] - relative_times[t-1])/1000  # Time difference between frames in seconds
        else:
            dt = 0.0  # Initial time
        accumulated_phase += 2 * np.pi * clockOffsetHertz[t] * dt
        correction = np.exp(-1j * accumulated_phase)
        corrected_CIRs[t, :] = cir_datas[t, :] * correction

    # corrected_CIRs_angles = np.angle(corrected_CIRs)
    # corrected_CIRs_magnitude = np.abs(corrected_CIRs)
    # corrected_CIRs_angles = np.insert(np.diff(corrected_CIRs_angles, axis=0), 0, 0, axis=0)
    # corrected_CIRs = corrected_CIRs_magnitude * np.exp(1j * corrected_CIRs_angles)  # Reconstruct the CIRs with the corrected phase

    Delta_phi = - clockOffsetHertz[1:] * frame_interval * 2 * np.pi / 1000
    Delta_phi = (Delta_phi) % (2 * np.pi) - np.pi  # Normalize to [-pi, pi]
    phase_exmaples = np.angle(cir_datas[:, 1])
    # plt.plot(carrierInterger / np.max(carrierInterger), label='CI')
    # plt.plot(Delta_phi, label='Delta Phi')

    frame_idx_interval = np.array([i%255 for i in np.diff(frame_idx_list)])
    print(f"Frame index interval: {frame_idx_interval[-10:]}")
    # exit()
    missing_frame_idx = np.where(frame_interval > (1500 / sampling_rate))[0] 
    print(f"Missing frame index: {missing_frame_idx}")
    time_real = np.cumsum(frame_interval) / 1000  # Convert to seconds
    time_real = np.insert(time_real, 0, 0.0)  # Add a zero as the first element
    time_real = np.array(time_real, dtype=np.float64)
    
    # corrected_phase_unwrapped_ref = [np.unwrap(np.angle(corrected_CIRs[:, i])) for i in range(1)]
    # corrected_phase_unwrapped = np.mean(corrected_phase_unwrapped_ref, axis=0)
    # slope, _ = np.polyfit(time_real, corrected_phase_unwrapped, 1)
    # plt.plot(time_real, corrected_phase_unwrapped, label='Unwrapped Phase')
    # plt.plot(time_real, slope * time_real, label='Slope')
    # plt.plot(time_real, corrected_phase_unwrapped - slope * time_real, label='Residual Phase')
    # plt.show()
    # slope /= (2 * np.pi)  # Convert to normalized frequency offset
    # print(f"Residual Slope {slope:.4e}")
    # for t in range(len(corrected_CIRs)):
    #     corrected_CIRs[t, :] *= np.exp(-1j * slope * time_real[t] * 2 * np.pi)
    
    # plt.style.use(['science', 'no-latex'])
    # plt.plot((np.unwrap(phase_exmaples)), label='Raw Upwrapped Phase')
    # # plt.plot((np.cumsum(Delta_phi)), label='Cumsum of Delta Phi')
    # plt.plot((np.unwrap(np.angle(corrected_CIRs[:, 1]))), label='Corrected Unwrapped Phase')

    # plt.xlabel("Frame Index")
    # plt.ylabel("Phase Upwrapped (radians)")
    # # plt.xlim(0, 50)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(f"phase_correction_{time_stamp_suffix}.png", dpi=300)
    # plt.show()
    # plt.close()
    # exit()

    if show_frame_interval:
        plt.style.use(['science', 'no-latex'])
        plt.figure(figsize=(8, 3))
        plt.plot(frame_interval[2:], label='RX timestamp inteval')
        # plt.plot(tx_frame_interval[2:], label='TX timestamp inteval')
        # plt.plot(frame_idx_interval[2:], label='frame index inteval')
        plt.xlabel("Frame Index")
        plt.ylabel("Frame Interval (ms)")
        plt.title("Frame Interval Over Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("frame_interval.png", dpi=300)
        plt.show()
        plt.close()
    
    return corrected_CIRs, diag_data, time_real, time_stamp_suffix, p_los_list



def load_txrx_file(INPUT_FILE=None, discard_secs=2, sampling_rate=500, show_frame_interval=False):
    INPUT_FOLDER = "./cir_files/"
    if INPUT_FILE is None:
        INPUT_FILE = get_latest_file(INPUT_FOLDER + 'rx/')
    print(f"Loading file: {INPUT_FILE}")
    time_stamp_suffix = INPUT_FILE.split('cir_data_rx_')[-1].split('.')[0]  # Extract timestamp from filename
    rx_cir = np.load(INPUT_FILE, allow_pickle=True)
    tx_cir = np.load(INPUT_FOLDER + 'tx/cir_data_tx_' + time_stamp_suffix + '.npy', allow_pickle=True)
    rx_diag_data = np.load(INPUT_FOLDER + 'diag/diag_data_rx_' + time_stamp_suffix + '.npy', allow_pickle=True)
    tx_diag_data = np.load(INPUT_FOLDER + 'diag/diag_data_tx_' + time_stamp_suffix + '.npy', allow_pickle=True)
    print(len(rx_diag_data), len(tx_diag_data))
    print(len(rx_cir), len(tx_cir))
    # exit()
    rx_cir = rx_cir[-len(tx_diag_data):]
    rx_diag_data = rx_diag_data[-len(tx_diag_data):]
    # assert len(rx_diag_data) == len(tx_diag_data), "Length of RX and TX diag data do not match"
    # assert len(rx_cir) == len(rx_diag_data), "Length of RX CIR and diag data do not match"

    discard_points = int(discard_secs * sampling_rate)
    rx_cir = rx_cir[discard_points:]
    tx_cir = tx_cir[discard_points:]
    rx_diag_data = rx_diag_data[discard_points:]
    tx_diag_data = tx_diag_data[discard_points:]

    diag_data = rx_diag_data.copy()
    rx_start_time = rx_diag_data[discard_points]['timestamp']
    tx_start_time = tx_diag_data[discard_points]['timestamp']
    
    frame_idx_list = [diag['D'] for diag in diag_data]
    clockOffset = [diag['CO'] for diag in diag_data]
    
    rx_relative_times = [d['timestamp'] - rx_start_time for d in rx_diag_data]
    tx_relative_times = [d['timestamp'] - tx_start_time for d in tx_diag_data]
    
    rx_frame_interval = np.array([i%17208 for i in np.diff(rx_relative_times)])
    tx_frame_interval = np.array([i%17208 for i in np.diff(tx_relative_times)])



    tx_phase = (np.angle(tx_cir))
    rx_phase = (np.angle(rx_cir))
    plt.plot(np.unwrap(tx_phase[:, 1]), label='TX Phase')
    plt.plot(np.unwrap(rx_phase[:, 1]), label='RX Phase')
    plt.plot((np.unwrap(rx_phase[:, 1] + tx_phase[:, 1])), label='Corrected Phase')
    plt.xlabel("Frame Index")
    plt.ylabel("Phase Upwrapped (radians)")
    plt.legend()
    plt.show()
    plt.close()



    frame_idx_interval = np.array([i%255 for i in np.diff(frame_idx_list)])
    print(f"Frame index interval: {frame_idx_interval[-10:]}")
    # exit()
    missing_frame_idx = np.where(rx_frame_interval > (1500 / sampling_rate))[0] 
    print(f"Missing frame index: {missing_frame_idx}")
    time_real = np.cumsum(rx_frame_interval) / 1000  # Convert to seconds
    time_real = np.insert(time_real, 0, 0.0)  # Add a zero as the first element
    time_real = np.array(time_real, dtype=np.float64)
    

    if show_frame_interval:
        plt.style.use(['science', 'no-latex'])
        plt.figure(figsize=(8, 3))
        plt.plot(tx_frame_interval[2:], label='TX timestamp inteval')
        plt.plot(rx_frame_interval[2:], label='RX timestamp inteval')
        # plt.plot(frame_idx_interval[2:], label='frame index inteval')
        plt.xlabel("Frame Index")
        plt.ylabel("Frame Interval (ms)")
        plt.title("Frame Interval Over Time")
        plt.legend()
        # plt.ylim(1000/sampling_rate-1, 1000/sampling_rate+1)
        plt.grid(True)
        plt.tight_layout()
        # plt.savefig("frame_interval.png", dpi=300)
        plt.show()
        plt.close()
    
    return None

def load_rx_file(INPUT_FILE=None, discard_secs=2, sampling_rate=500, show_frame_interval=False):
    INPUT_FOLDER = "./cir_files/"
    if INPUT_FILE is None:
        INPUT_FILE = get_latest_file(INPUT_FOLDER + 'rx/')
    print(f"Loading file: {INPUT_FILE}")
    time_stamp_suffix = INPUT_FILE.split('cir_data_rx_')[-1].split('.')[0]  # Extract timestamp from filename
    rx_cir = np.load(INPUT_FILE, allow_pickle=True)
    rx_diag_data = np.load(INPUT_FOLDER + 'diag/diag_data_rx_' + time_stamp_suffix + '.npy', allow_pickle=True)
    print(len(rx_diag_data))
    # exit()
    discard_points = int(discard_secs * sampling_rate)
    rx_cir = rx_cir[discard_points:]
    rx_diag_data = rx_diag_data[discard_points:]

    diag_data = rx_diag_data
    rx_start_time = rx_diag_data[discard_points]['timestamp']
    
    frame_idx_list = [diag['D'] for diag in diag_data]
    clockOffset = [diag['CO'] for diag in diag_data]
    
    rx_relative_times = [d['timestamp'] - rx_start_time for d in rx_diag_data] # 0, 1, 2, ...
    
    rx_frame_interval = np.array([i%17208 for i in np.diff(rx_relative_times)]) # 1, 1, ...
    missing_frame_idx = np.where((rx_frame_interval > (1800 / sampling_rate)) & (rx_frame_interval < (2500 / sampling_rate)))[0] # if idx=0, meaning frame 1 appear 1.8ms after frame 0 but less than 2.5ms, which means we need to insert a new frame after frame 0
    missing_two_frame_idx = np.where((rx_frame_interval > (2500 / sampling_rate)))[0]
    print(f"Missing frame index: {missing_frame_idx}")
    print(f"Missing two frame index: {missing_two_frame_idx}")

    p_los_list = np.array([diag['p_los']/(2**11) for diag in rx_diag_data])


    # corr_p_los_and_shift_64 = np.correlate(p_los_list, shift_64, mode='full')
    # plt.plot(corr_p_los_and_shift_64, label='Correlation of p_los and shift_64')
    # plt.show()
    # plt.close()

    plt.figure(figsize=(10, 6))
    # plt.plot(shift_64[1000:1100], label='Shift 64', marker='o')
    # plt.plot(np.unwrap(p_los_list[1000:1100]), label='p_los', marker='o')
    # plt.plot(np.unwrap(np.angle(rx_cir[1000:1100, 1])), label='RX CIR', marker='o')
    # plt.plot(np.diff(np.unwrap(p_los_list[1000:1100])), label='p_los', marker='o')
    rx_frame_interval_filtered = medfilt(rx_frame_interval[2:], kernel_size=3)
    plt.plot(rx_frame_interval_filtered[0:100], label='RX timestamp interval')
    plt.plot(((np.diff(rx_frame_interval_filtered) * 2 * np.pi * 7.9872e6)%(2*np.pi))[0:100], label='Phase change')
    plt.legend()
    plt.show()
    exit()
    # processed_cir_data, cir_datas, accum_list, shift_64 = process_cir(np.abs(rx_cir), rx_diag_data, remove_mean=False, missing_frame_idx=missing_frame_idx)
    # return processed_cir_data
    
    # waterfall_cir(processed_cir_data, rx_relative_times, time_stamp_suffix, save=True)
    exit()

    # for slow_time in range(processed_cir_data.shape[0]):
    #     corr_delay = np.correlate(np.abs(processed_cir_data[slow_time]), np.abs(processed_cir_data[0]), mode='full')
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(np.abs(processed_cir_data[slow_time]), label=f"Slow Time {slow_time}")
    #     plt.plot(np.abs(processed_cir_data[0]), label=f"Slow Time 0")
    #     plt.title(f"Slow Time {slow_time} - max index {np.argmax(corr_delay)-1280+1}")
    #     plt.legend()
    #     plt.show()
    #     plt.close()


    # exit()
    # rx_phase = (np.angle(processed_cir_data))
    # # for i in range(1, 64*20, 64):
    # #     f, t_stft, Zxx = stft(np.abs(processed_cir_data[:, i]), fs=1000, nperseg=512, noverlap=256)
    # #     plt.pcolormesh(t_stft, f, 10*np.log10(np.abs(Zxx)), shading='gouraud')
    # #     plt.title(f'STFT Magnitude {i}')
    # #     plt.ylabel('Frequency [Hz]')
    # #     plt.xlabel('Time [sec]')
    # #     plt.colorbar(label='Magnitude')
    # #     plt.show()
    # #     plt.close()
    # #     # plt.figure(figsize=(10, 6))
    # #     # # plt.plot((np.unwrap(p_los_list)), label=f'los phase estimated')
    # #     # plt.plot((np.unwrap(rx_phase[:, i]-rx_phase[:, 64])), label=f'Tap-{i} Phase')
    # #     # # plt.plot(np.diff(np.unwrap(rx_phase[:, i] - rx_phase[:, 64] + p_los_list)), label=f'My correct Tap-{i} Phase')
    # #     # plt.xlabel("Frame Index")
    # #     # plt.ylabel("Phase Upwrapped (radians)")
    # #     # plt.legend()
    # #     # plt.show()
    # #     # plt.close()


    # # plt.style.use(['science', 'no-latex'])
    # # # 153 cm, 1 tap: 60/64=0.9375 cm, tap-64: LoS, 64+164=tap 228
    # # for i in range(199, 220, 1):
    # #     # plot_specrum((rx_phase[:, i]-rx_phase[:, 64]), fs=1000, title=f'Tap {i} RX Phase Spectrum')
    # #     plot_specrum((np.abs(processed_cir_data[:, i])), fs=1000, title=f'Tap {i} RX Phase Spectrum')
    
    # max_power = -100
    # for i in range(0, 20*64):
    #     amp_series = np.diff(np.abs(processed_cir_data[:, i]))
    #     fft_phase = fft(amp_series)
    #     freq = fftfreq(len(amp_series), 1/1000)
    #     # only see 240 to 300 Hz
    #     valid_mask = (freq > 50) & (freq < 135)
    #     freq = freq[valid_mask]
    #     fft_phase = fft_phase[valid_mask]
    #     current_max_power = np.max(20*np.log10(np.abs(fft_phase)))
    #     if current_max_power > max_power:
    #         max_power = current_max_power
    #         print(f"Max power: {max_power:.2f} dB, Tap-{i}, at frequency {freq[np.argmax(20*np.log10(np.abs(fft_phase)))]} Hz")
    # exit()
    
    # # plt.plot(np.unwrap(rx_phase[:, 1]), label='LoS Phase')
    # # plt.plot(np.unwrap(rx_phase[:, 2]), label='Tap-2 Phase Raw')
    # for i in range(20):
    #     plt.plot(np.unwrap(rx_phase[:, i] - rx_phase[:, 0]), label=f'Tap-{i} Phase')
    # plt.xlabel("Frame Index")
    # plt.ylabel("Phase Upwrapped (radians)")
    # plt.legend()
    # plt.show()
    # plt.close()




    frame_idx_interval = np.array([i%255 for i in np.diff(frame_idx_list)])
    print(f"Frame index interval: {frame_idx_interval[-10:]}")
    # exit()
    missing_frame_idx = np.where(rx_frame_interval > (1500 / sampling_rate))[0] 
    print(f"Missing frame index: {missing_frame_idx}")
    time_real = np.cumsum(rx_frame_interval) / 1000  # Convert to seconds
    time_real = np.insert(time_real, 0, 0.0)  # Add a zero as the first element
    time_real = np.array(time_real, dtype=np.float64)
    

    if show_frame_interval:
        plt.style.use(['science', 'no-latex'])
        plt.figure(figsize=(8, 3))
        rx_frame_interval_filtered = medfilt(rx_frame_interval[2:], kernel_size=3)
        plt.plot(rx_frame_interval_filtered, label='RX timestamp interval')
        # plt.plot(frame_idx_interval[2:], label='frame index inteval')
        plt.xlabel("Frame Index")
        plt.ylabel("Frame Interval (ms)")
        plt.title("Frame Interval Over Time")
        plt.legend()
        # plt.ylim(1000/sampling_rate-1, 1000/sampling_rate+1)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("frame_interval_5min.png", dpi=300)
        plt.show()
        plt.close()
    
    return None



def process_cir(cir_datas, diag_data, remove_mean=False, missing_frame_idx=None):
    upsample_factor = 64
    index_fp = [i['index_fp_u32'] for i in diag_data]
    index_fp = np.array(index_fp, dtype=np.uint32)
    index_fp = [i/64 - i//64 for i in index_fp]  # Fractional part of the index_fp
    shift_64 = np.array([i * upsample_factor for i in index_fp], dtype=np.uint32)  # Shifted index_fp for upsampling

    # upsample the CIR by 64 using FFT
    cir_datas_upsampled = np.zeros((cir_datas.shape[0], cir_datas.shape[1] * upsample_factor), dtype=complex)
    for i in range(cir_datas.shape[0]):
        # cir_data = cir_datas[i]
        # N = len(cir_data)
        # x = np.arange(N)
        # x_interp = np.linspace(0, N-1, N * upsample_factor)
        # amp = np.abs(cir_data)
        # phase = np.angle(cir_data)
        # phase_unwrapped = np.unwrap(phase)
        # delta_phi = 2 * np.pi * 7987.2e6 / (2 * 499.2e6)
        # phase_phys = phase_unwrapped + delta_phi * x
        # amp_spline = UnivariateSpline(x, amp, s=0)
        # amp_interp = amp_spline(x_interp)

        # phase_interp_func = interp1d(x, phase_phys, kind='cubic')
        # phase_interp_phys = phase_interp_func(x_interp)
        # phase_interp_final = phase_interp_phys - delta_phi * x_interp

        # cir_datas_upsampled[i] = amp_interp * np.exp(1j * phase_interp_final)

        X = np.fft.fft(cir_datas[i])
        # zero-padding in the frequency domain
        X_upsampled = np.zeros(X.shape[0] * upsample_factor, dtype=complex)
        X_upsampled[:X.shape[0]//2] = X[:X.shape[0]//2]
        X_upsampled[-X.shape[0]//2:] = X[-X.shape[0]//2:]
        cir_datas_upsampled[i] = np.fft.ifft(X_upsampled)
    # plt.plot(np.unwrap(np.angle(cir_datas_upsampled[:, 1])))
    # plt.show()

    # Shift and normalize
    accum_list = [i.get('accumCount', 1) for i in diag_data]
    # plot_specrum(accum_list, fs=1000, title='Accumulation Count Spectrum')
    processed_cir_data = np.zeros_like(cir_datas_upsampled, dtype=complex)
    for i, diag in enumerate(diag_data):
        shift_amount = - int(shift_64[i])  # Positive: move right, Negative: move left    
        shifted = np.roll(cir_datas_upsampled[i], shift_amount)

        accum = accum_list[i]
        if accum == 0:  # avoid divide-by-zero
            exit()
        # accum = 1
        processed_cir_data[i] = shifted / accum
        cir_datas[i] = cir_datas[i] / accum  # Normalize original CIR data as well
        # processed_cir_data[i] = processed_cir_data[i] / np.max(np.abs(processed_cir_data[i]))  # Normalize by max amplitude
    
    if remove_mean:
        processed_cir_data = processed_cir_data - np.mean(processed_cir_data, axis=0)
        cir_datas = cir_datas - np.mean(cir_datas, axis=0)
    
    for i in missing_frame_idx:
        # interpolate the missing data: idx=0, insert a frame after frame 0
        data_interp = processed_cir_data.copy()
        new_row = (np.abs(data_interp[i]) + np.abs(data_interp[i + 1])) / 2.0 * np.exp(1j * (np.angle(data_interp[i]) + np.angle(data_interp[i + 1])) / 2.0)
        data_interp = np.insert(data_interp, i + 1, new_row, axis=0)

    return data_interp, cir_datas, accum_list, shift_64

def waterfall_cir(data, time_real, time_stamp_suffix, save=False, db=False, path=None, show=True):
    if data.dtype == np.complex128:
        data = np.abs(data)
    plt.style.use(['science', 'no-latex'])
    if db:
        data = 10 * np.log10(data)
    plt.figure(figsize=(4, 3))
    extent = [0, data.shape[1], time_real[-1], time_real[0]]  # flip y-axis to match time flow
    plt.imshow(data, aspect='auto', cmap='viridis', extent=extent)
    plt.xlabel("Fast Time Index")
    plt.ylabel("Slow Time (s)")
    if db:
        plt.colorbar(label="Amplitude (dB)")
    else:
        plt.colorbar(label="Amplitude")
    plt.title(f"CIR Waterfall Plot - {time_stamp_suffix}")
    plt.tight_layout()
    if save:
        if path is not None:
            plt.savefig(f"{path}/cir_waterfall_plot_{time_stamp_suffix}.png", dpi=300)
        else:
            plt.savefig(f"./figures/cir_waterfall_plot_{time_stamp_suffix}.png", dpi=300)
    if show:
        plt.show()
    plt.close()

def excess_length_to_tap(excess_length, los_index, upsample_factor=32):
    return (excess_length / 30.4589137328) * upsample_factor + los_index + 72.5

def tap_to_excess_length(tap_idx, los_index, upsample_factor=32):
    return (tap_idx - los_index - 72.5) * (30.4589137328 / upsample_factor)

def insert_interpolated_frames(data, missing_frame_idx):
    T, N = data.shape
    K = len(missing_frame_idx)
    if K == 0:
        return data
    
    new_T = T + K
    result = np.zeros((new_T, N), dtype=data.dtype)
    
    # Sort to ensure order
    missing_set = set(missing_frame_idx)
    sorted_missing = sorted(missing_frame_idx)
    
    src_idx = 0  # Index in original data
    dst_idx = 0  # Index in new array
    
    while src_idx < T:
        # Copy the original row
        result[dst_idx] = data[src_idx]
        dst_idx += 1

        # Check if we need to insert after this row
        if src_idx in missing_set:
            if src_idx + 1 >= T:
                raise ValueError(f"Cannot interpolate at boundary index {src_idx}")
            # Insert interpolated row between data[src_idx] and data[src_idx+1]
            interp_row = (data[src_idx] + data[src_idx + 1]) / 2.0
            result[dst_idx] = interp_row
            dst_idx += 1
        
        src_idx += 1
    
    return result

def insert_zeros_frames(data, missing_frame_idx):
    """
    Insert zero frames at specified indices.
    
    Args:
        data: Input array of shape (T, N)
        missing_frame_idx: List of indices where zero frames should be inserted after
        
    Returns:
        Array with zero frames inserted, shape (T + len(missing_frame_idx), N)
    """
    if len(missing_frame_idx) == 0:
        return data
    
    T, N = data.shape
    new_T = T + len(missing_frame_idx)
    result = np.zeros((new_T, N), dtype=data.dtype)
    
    src_idx = 0
    dst_idx = 0
    
    for src_idx in range(T):
        result[dst_idx] = data[src_idx]
        dst_idx += 1
        
        if src_idx in missing_frame_idx:
            # Zero frame already in place, just advance destination
            dst_idx += 1
    
    return result

def plotting_cir_amp_and_angle(cir_datas, phase_unwarp=True):
    plt.figure(figsize=(5, 3))
    for i in range(cir_datas.shape[0]):
        plt.plot(np.abs(cir_datas[i, :]), label=f'cir_data[{i}]')
    # plt.legend()
    plt.show()
    plt.close()
    plt.figure(figsize=(5, 3))
    for i in range(cir_datas.shape[0]):
        if phase_unwarp:
            # Unwrap the phase to avoid discontinuities
            plt.plot(np.unwrap(np.angle(cir_datas[i, :])), label=f'cir_data[{i}]')
        else:
            # Plot the raw phase
            plt.plot(np.angle(cir_datas[i, :]), label=f'cir_data[{i}]')
    # plt.legend()
    plt.show()
    plt.close()

def plot_constellation(x):
    plt.figure(figsize=(3, 3))
    plt.plot(x.real, x.imag, 'o', alpha=0.2)  # 'o' means circular markers
    plt.axhline(0, color='gray', lw=0.5)
    plt.axvline(0, color='gray', lw=0.5)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlabel('In-Phase (I)')
    plt.ylabel('Quadrature (Q)')
    plt.title('Constellation Diagram')
    plt.axis('equal')  # Ensure x and y scales are equal
    plt.show()
    
    plt.figure(figsize=(3, 3))
    plt.hist2d(x.real, x.imag, bins=50, cmap='viridis')
    plt.colorbar(label='Count')
    plt.xlabel('In-Phase (I)')
    plt.ylabel('Quadrature (Q)')
    plt.title('Constellation 2D Histogram')
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

def plot_constellation_remove_DC(x):
    x_dc_removed = x - np.mean(x)  # Remove DC component
    plt.figure(figsize=(3, 3))
    plt.plot(x.real, x.imag, 'o', alpha=0.2)  # 'o' means circular markers
    plt.plot(x_dc_removed.real, x_dc_removed.imag, 'o', alpha=0.2)  # Plot DC removed constellation
    plt.axhline(0, color='gray', lw=0.5)
    plt.axvline(0, color='gray', lw=0.5)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlabel('In-Phase (I)')
    plt.ylabel('Quadrature (Q)')
    plt.title('Constellation Diagram')
    plt.axis('equal')  # Ensure x and y scales are equal
    plt.show()

def static_background_removal(cir_data, beta=0.9):
    """
    Perform static background removal on the CIR data.
    
    Parameters:
    - cir_data: The CIR data to process.
    - beta: The decay factor for the exponential moving average.
    
    Returns:
    - The processed CIR data with static background removed.
    """
    # Initialize the background estimate
    background = np.zeros_like(cir_data[0], dtype=complex)
    
    # Initialize the processed data array
    processed_data = np.zeros_like(cir_data, dtype=complex)
    
    for i in range(len(cir_data)):
        # Update the background estimate using exponential moving average
        background = beta * background + (1 - beta) * (cir_data[i])
        
        # Subtract the background from the current CIR data
        processed_data[i] = cir_data[i] - background
    
    return processed_data


def save_processed_data(data, input_file):
    file_name = input_file.split('/')[-1].split('.')[0]
    np.save(f"./processed_data/{file_name}.npy", data)

def estimate_cfo_from_cir(h_raw, h_upsampled, peak_index=None):
    # Step 1: Find the index of the peak (use average over time to reduce noise)
    avg_amplitude = np.abs(h_raw).mean(axis=0)
    # variance_amplitude = np.var(np.abs(h), axis=0)
    # svr = avg_amplitude / variance_amplitude
    if peak_index is None:
        peak_index = np.argmax(avg_amplitude) 
    print(f"Peak index: {peak_index}, Peak value: {avg_amplitude[peak_index]:.2f}")

    # Step 2: Extract phase over time at that tap
    phase_series = np.angle(h_upsampled[:, peak_index])  # shape [T]

    plt.plot(np.diff(np.unwrap(np.angle(h_upsampled[:, 1]))), label='unwrapped phase')
    plt.legend()
    plt.legend()
    plt.show()
    plt.close()
    exit()

    # Step 3: Unwrap phase to prevent 2π discontinuity
    unwrapped_phase = np.unwrap(phase_series)
    correction = np.exp(-1j * unwrapped_phase).reshape(-1, 1) # shape: [T, 1], to correct the phase offset
    h_corrected = h_upsampled * correction  # shape: [T, N]


    # Step 4: Fit a line to phase vs time (slope gives angular frequency offset)
    time = np.arange(len(unwrapped_phase))
    slope, _ = np.polyfit(time, unwrapped_phase, 1)

    # CFO in radians/sample; divide by 2π to get normalized frequency offset
    cfo_normalized = slope / (2 * np.pi)
    print(f"Estimated CFO (normalized): {cfo_normalized:.4e}")

    # for vis_idx in range(0, h_upsampled.shape[1], 64):
    #     # plt.figure()
    #     # plt.plot(np.unwrap(np.angle(h_upsampled[:, vis_idx])), label='Unwrapped Phase')
    #     # plt.plot(np.unwrap(np.angle(h_corrected[:, vis_idx])), label='Corrected Phase')
    #     # plt.title(f"Phase over Time @ tap {vis_idx}, CFO = {cfo_normalized:.4e}")
    #     # plt.xlabel("Time Index")
    #     # plt.ylabel("Unwrapped Phase (radians)")
    #     # plt.legend()
    #     # plt.grid(True)
    #     # plt.ylim(-5, 5)
    #     # plt.show()
    #     # plt.close()

    #     plt.plot(np.abs(h_upsampled[:, vis_idx]), label='Magnitude')
    #     plt.title(f"Magnitude over Time @ tap {vis_idx}, CFO = {cfo_normalized:.4e}")
    #     plt.xlabel("Time Index")
    #     plt.ylabel("Magnitude")
    #     plt.legend()
    #     plt.grid(True)
    #     plt.show()
    #     plt.close()
    return h_corrected, cfo_normalized, peak_index, unwrapped_phase

def plot_cwt_spectrum(signal, fs=500, wavelet='cmor1.5-1.0', title='CWT Spectrum'):
    """
    Plot the CWT spectrum of a 1D time series.

    Parameters:
    - signal: 1D np.array, complex or real time series
    - fs: Sampling frequency (Hz), default is 1.0 (for normalized time index)
    - wavelet: Wavelet type (default is complex Morlet 'cmor1.5-1.0')
    - title: Title for the plot
    """
    signal = np.asarray(signal)

    # Define scales (inversely related to frequency)
    widths = np.arange(1, 128)

    # Compute CWT
    cwt_matrix, freqs = pywt.cwt(signal, widths, wavelet, sampling_period=1.0/fs)

    # Get time axis
    time = np.arange(len(signal)) / fs

    # Plot
    plt.figure(figsize=(10, 6))
    plt.imshow(np.abs(cwt_matrix), extent=[time[0], time[-1], freqs[-1], freqs[0]],
               aspect='auto', cmap='jet')
    plt.colorbar(label='Magnitude')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()
    plt.show()
    plt.close()


def detect_periodic_cir(cir, fs_time, max_lag_seconds=5, prominence=0.1):
    """
    Process CIR array to detect periodic changes using ACF.
    
    Args:
        cir: T*N array, CIR snapshots over time (T frames, N delay taps)
        fs_time: Sampling rate in time domain (Hz)
        max_lag_seconds: Maximum lag to compute ACF (seconds)
        prominence: Minimum peak prominence for detection
        
    Returns:
        dominant_period: Detected dominant period (seconds)
        combined_acf: Combined ACF across subcarriers
        freqs: Frequency subcarriers
    """
    # Convert CIR to CSI (Time → Frequency Domain)
    csi = fft(cir, axis=1)  # (T, N_subcarriers)
    n_subcarriers = csi.shape[1]
    
    # Compute ACF for each subcarrier
    max_lag = int(max_lag_seconds * fs_time)
    acfs = []
    weights = []
    
    for f in range(n_subcarriers):
        # Get CSI time series for this subcarrier
        h_ts = csi[:, f]
        
        # Compute autocorrelation (normalized)
        acf = np.correlate(h_ts, h_ts, mode='full')[len(h_ts)-1:]  # Raw ACF
        acf = acf / np.max(acf)  # Normalize
        
        # Truncate to max_lag
        acf = acf[:max_lag]
        acfs.append(acf)
        
        # Calculate weight (g(f) from first lag)
        weights.append(acf[1] if len(acf) > 1 else 0)
    
    # Maximal Ratio Combining (MRC)
    weights = np.array(weights)
    weights += 1e-6  # Avoid division by zero
    weights /= np.sum(weights)  # Normalize
    
    combined_acf = np.sum([w * a for w, a in zip(weights, acfs)], axis=0)
    
    # Find peaks in combined ACF
    lags = np.arange(len(combined_acf)) / fs_time
    peaks, properties = find_peaks(combined_acf, prominence=prominence)
    
    if len(peaks) > 0:
        # Get lag of most prominent peak
        main_peak_idx = np.argmax(properties['prominences'])
        dominant_lag = lags[peaks[main_peak_idx]]
        return dominant_lag, combined_acf, lags
    else:
        return None, combined_acf, lags

def correlation_PN_analysis(processed_cir_data, seq, tap_list=[], PLOT=True, THRD=0.0):
    max_correlation_results = -100
    if len(tap_list) == 0:
        tap_list = [i for i in range(processed_cir_data.shape[1])]
    for n in tap_list:
        # # # apply Savitzky-Golay filter to smooth the data
        # smoothed_cir_data = savgol_filter(processed_cir_data[:, n], 10, 1, deriv=0)  # window size 51, polynomial order 3
        # correlate the CIR with the PN sequence
        corr_results = np.correlate(processed_cir_data[:, n], seq, mode='valid')
        # find the max correlation value
        max_corr = np.max(corr_results)
        if max_corr > max_correlation_results:
            max_correlation_results = max_corr
            print(f"Max correlation for tap {n}: {max_corr:.2f}")
            if PLOT and max_corr > THRD:
                plt.plot(processed_cir_data[:, n], label=f"CIR Tap {n}")
                plt.xlabel("Time")
                plt.ylabel("Amplitude")
                plt.title(f"CIR Tap {n} - 64 Times Upsampled")
                plt.legend()
                plt.show()
                plt.close()

                plt.plot(np.arange(len(corr_results))/500, corr_results, label=f"CIR Tap {n}")
                plt.xlabel("Time (s)")
                plt.ylabel("Amplitude")
                plt.title("Correlation Results")
                # plt.axhline(y=0, color='k', linestyle='--', label='Zero Line')
                plt.legend()
                plt.show()
                plt.close()

                # do fft on the correlation results
                N = len(corr_results)
                time_stamps = np.arange(N) / 500  # Ensure time stamps match the length of the correlation results
                dt = np.mean(np.diff(time_stamps))    # sampling interval (e.g., 0.03 s)
                freqs = np.fft.rfftfreq(N, d=dt)      # frequency axis for real FFT
                FFT_vals = np.fft.rfft(corr_results * np.hamming(N))  # apply Hamming window to reduce edge effects
                power_spectrum = np.abs(FFT_vals)**2
                power_spectrum = 10*np.log10(power_spectrum[:])
                # freqs = freqs[10:]
                plt.plot(freqs, power_spectrum, label=f"CIR Tap {n}")
                plt.xlim(0, 10)
                plt.xlabel("Frequency (Hz)")
                plt.ylabel("Power (Linear Scale)")
                plt.title(f"Power Spectrum of CIR Taps {n} - 64 Times Upsampled")
                plt.show()
                plt.close()

def fft_analysis(processed_cir_data, time_real):
    max_power = -100
    for i in range(processed_cir_data.shape[1]):
        ts = (processed_cir_data[:, i])  # Use the magnitude of the CIR tap for FFT analysis
        N = len(ts)
        time_stamps = time_real[:N]  # Ensure time stamps match the length of the CIR tap
        dt = np.mean(np.diff(time_stamps))    # sampling interval (e.g., 0.03 s)
        freqs = np.fft.fftfreq(N, d=dt)      # frequency axis for real FFT
        FFT_vals = np.fft.fft(ts * np.hamming(N))  # apply Hamming window to reduce edge effects
        FFT_vals = FFT_vals[:N//2]  # take only the positive frequencies
        freqs = freqs[:N//2]
        power_spectrum = np.abs(FFT_vals)**2
        power_spectrum = 10*np.log10(power_spectrum[10:])
        freqs = freqs[10:]
        current_max_power = np.max(power_spectrum)
        if current_max_power > max_power:
            max_power = current_max_power  
            print(f"Max power for tap {i}: frequency {freqs[np.argmax(power_spectrum)]:.2f} Hz, power {current_max_power:.2f}")
            if current_max_power > 41.85:
                plt.plot(freqs, 10**(power_spectrum/10), label=f"CIR Tap {i}")
                # plt.xlim(0, 10)
                # plt.ylim(0, 9)
                plt.xlabel("Frequency (Hz)")
                plt.ylabel("Power (Linear Scale)")
                plt.title(f"Power Spectrum of CIR Taps {i} - 64 Times Upsampled")
                plt.tight_layout()
                # plt.savefig(f"cir_fft_tap_{i}_{current_max_power:.2f}db_freq{freqs[np.argmax(power_spectrum)]:.2f}.png", dpi=300)
                plt.show()
                plt.close()

        # plt.plot(time_stamps, ts, label=f"CIR Tap {i}")
        # plt.xlabel("Time (s)")
        # plt.ylabel("Amplitude")
        # plt.title("CIR Tap Over Time")
        # plt.legend()
        # plt.show()
        # plt.close()

def periodic_autocorr(x):
    N = len(x)
    return np.array([np.dot(x, np.roll(x, -k)) for k in range(N)])

def corrupt_pn_seq_with_missing(seq, missing_idx):
    corrupted_seq = np.concatenate((seq[0:missing_idx], seq[missing_idx+1:]), axis=0)
    return corrupted_seq

def autocorr(raw, sampled, showFig=False):
    """Compute the autocorrelation of a signal."""
    n = len(raw)
    raw_spec = fft(raw, n)
    repeat_times = int(np.ceil(len(sampled) / n))
    avg_acorrcirc = np.mean([
        ifft(raw_spec * np.conj(fft(sampled[i*n:(i+1)*n], n))).real
        for i in range(repeat_times)
    ], axis=0)
    if showFig:
        plt.figure()
        plt.plot(np.arange(-n/2+1, n/2+1), fftshift(avg_acorrcirc), '.-')
        plt.margins(0.1, 0.1)
        plt.grid(True)
        plt.show()
        plt.close()
    return avg_acorrcirc

def modulate_seq(seq):
    # turn +1 to 1010 and -1 to 0101
    seq_modulated = np.zeros(len(seq) * 4, dtype=int)
    for i in range(len(seq)):
        if seq[i] == 1:
            seq_modulated[i*4:(i+1)*4] = [1, -1, 1, -1]
        else:
            seq_modulated[i*4:(i+1)*4] = [-1, 1, -1, 1]
    return seq_modulated


if __name__ == "__main__":
    INPUT_FILE = None
    # INPUT_FILE = "cir_files/rx/cir_data_20250421_1440_EN256SW_h8.npy"
    # INPUT_FILE = "cir_files/rx/cir_data_rx_20250507_1809_ENesp32pn10bit1ms.npy"
    cir_datas = load_rx_file(INPUT_FILE=INPUT_FILE, discard_secs=2, sampling_rate=1000, show_frame_interval=True)
    np.save("cir_datas_en_5minutes_3m.npy", cir_datas)
    np.save(f"./processed_data/{INPUT_FILE.split('cir_data_rx_')[1].split('.')[0]}.npy", cir_datas)
    exit()

    # cir_datas = np.load("cir_datas_5minutes_disable.npy", allow_pickle=True)
    # cir_datas = np.load("cir_datas.npy", allow_pickle=True)
  
    # for idx in range(0, 20*64, 64):
    #     plt.style.use(['science', 'no-latex'])
    #     f, t_stft, Zxx = stft(np.abs(cir_datas[:, idx]), fs=1000, nperseg=256, noverlap=128)
    #     plt.figure(figsize=(5, 3))
    #     plt.pcolormesh(t_stft, f, 10*np.log10(np.abs(Zxx)), shading='gouraud')
    #     plt.title('STFT Magnitude')
    #     plt.ylabel('Frequency [Hz]')
    #     plt.xlabel('Time [sec]')
    #     plt.colorbar(label='Magnitude')
    #     plt.title(f"STFT Magnitude of CIR Tap {idx} - 64 Times Upsampled")
    #     plt.savefig(f"./figures/disable_5minutes_stft_cir_tap_{idx}.png", dpi=300)
    #     plt.show()
    #     plt.close()
    # exit()


    bits = 10 
    PLOT = False
    THRD = 0.0
    tap_list = []  # List of taps to analyze, empty means all taps
    PLOT = True if len(tap_list) > 0 else PLOT
    RM_DC = False
    fs = 1000
    # seq = get_PN(bits=bits)  # Generate the PN sequence
    # pn_length = len(seq)  # Length of PN sequence

    # repeat_times = int(len(cir_datas)/pn_length)  # Repeat the PN sequence 4 times
    # symbol_rate = 1000  # Hz → symbol_duration = 1/1000 s
    # symbol_duration = 1 / symbol_rate
    # t_total = pn_length * symbol_duration * repeat_times

    # # Time resolution for waveform
    # base_fs = 256000  # High-res simulation (128kHz)
    # t = np.arange(0, t_total, 1 / base_fs)

    # # Generate PN waveform
    # pn_waveform = np.repeat(np.tile(seq, repeat_times), len(t) // pn_length + 1)[:len(t)]

    # location: 101+90-57=134cm, 134/60*64+64+1=208
    pn_18bit = generate_mls_lfsr_18()
    max_corr_list = []
    for tap in range(218, 219):
        corr_res = autocorr_same_length(pn_18bit, np.abs(cir_datas[:, tap]), showFig=True) 
        print(f"Max correlation for tap {tap}: {np.max(corr_res):.2f}")
        max_corr_list.append(np.round(np.max(corr_res), 2).item())
    print(f"starting from tap 0 to 400, max correlation: {max_corr_list}")
    exit()

    # --- Sampling with clock drift ---
    epsilon_list = [0.006]  # Clock drift values
    initial_phase_list = [4]
    for epsilon in epsilon_list:
        for initial_phase in initial_phase_list:
            # epsilon = 0.0001  # 5% clock drift → sampling interval = 1ms * (1 + epsilon)
            ideal_sample_interval = 0.001  # 1ms
            actual_sample_interval = ideal_sample_interval * (1 + epsilon)
            init_phase = initial_phase / 8 * actual_sample_interval  # start sampling at 3/8 of the interval
            print(f"Clock drift: {epsilon:.4f}, Initial phase: {init_phase:.4f}")

            # Create drifting sample times
            sampling_times = np.arange(init_phase, t_total, actual_sample_interval)
            pn_sequence_sampled = np.interp(sampling_times, t, pn_waveform)

            max_corr = -100
            tap_list = np.arange(120, 250) if len(tap_list) == 0 else tap_list
            for tap in tap_list:
                avg_corr = (autocorr(pn_sequence_sampled, np.abs(cir_datas[:, tap])))
                current_max_corr = np.max(avg_corr)
                if current_max_corr > max_corr:
                    max_corr = current_max_corr
                    print(f"Max correlation for tap {tap}: {max_corr:.2f}")
                    if PLOT and max_corr > THRD:
                        autocorr(pn_sequence_sampled, np.abs(cir_datas[:, tap]), showFig=True)
    exit()


    cir_datas, diag_data, time_real, time_stamp_suffix, p_los_list = load_file(INPUT_FILE=INPUT_FILE, discard_secs=0, sampling_rate=fs, show_frame_interval=True)
    processed_cir_data, cir_datas, accum_list = process_cir(cir_datas, diag_data, remove_mean=False)
    
    waterfall_cir(processed_cir_data, time_real, time_stamp_suffix, save=False)
    # exit()
    # h_corrected, cfo_normalized, peak_index, unwrapped_phase = estimate_cfo_from_cir(processed_cir_data, peak_index=64)
    h_corrected, cfo_normalized, peak_index, unwrapped_phase = estimate_cfo_from_cir(cir_datas, processed_cir_data, peak_index=1)
    

    max_fft_result = -100
    tap_list = np.arange(0, processed_cir_data.shape[1]) if len(tap_list) == 0 else tap_list
    for tap_idx in tap_list:
        tap_signal = (processed_cir_data[:, tap_idx])
        N = len(tap_signal)
        dt = 0.001
        freqs = np.fft.fftfreq(N, d=dt)  # frequency axis for real FFT
        FFT_vals = np.fft.fft(tap_signal)  # apply Hamming window to reduce edge effects
        FFT_vals = FFT_vals[:N//2]  # take only the positive frequencies
        freqs = freqs[10:N//2]
        fft_results = np.abs(FFT_vals)[10:]

        # only focus on power at 0.4 to 0.6 normalized frequency
        # fft_results = fft_results[(freqs > 225) & (freqs < 280)]
        # freqs = freqs[(freqs > 225) & (freqs < 280)]
        current_max_power = np.max(fft_results)
        if current_max_power > max_fft_result:
            max_fft_result = current_max_power
            print(f"Max power for tap {tap_idx}: frequency {freqs[np.argmax(fft_results)]:.2f} Hz, power {current_max_power:.2f}")
            if PLOT and current_max_power > THRD:
                plt.figure(figsize=(5, 3))
                plt.plot(freqs, fft_results, label=f"CIR Tap {tap_idx}")
                plt.xlabel("Frequency (Hz)")
                plt.ylabel("Power (Linear Scale)")
                plt.title(f"Power Spectrum of CIR Taps {tap_idx} - 64 Times Upsampled")
                plt.tight_layout()
                # plt.savefig(f"cir_fft_tap_{tap_idx}_{current_max_power:.2f}db_freq{freqs[np.argmax(fft_results)]:.2f}.png", dpi=300)
                plt.show()
                plt.close()

    # max_correlation = -100
    # for tap_idx in tap_list:  
    #     tap_signal = (h_corrected[:, tap_idx])
    #     tap_length = len(tap_signal)
        
    #     # # amp_tap_signal = np.abs(tap_signal)
    #     # # angle_tap_signal = np.angle(tap_signal)
    #     # # pad zeros to the seq signal
    #     # # seq = get_PN(bits=bits)  # Generate the PN sequence
    #     # # # seq = np.pad(seq, (0, len(seq)), 'constant', constant_values=0)  # Pad zeros to the PN sequence
    #     # # enlarged_seq = np.tile(seq, (tap_length // len(seq) + 1))[:tap_length]  # Repeat the PN sequence to match tap length
    #     # # ratio = (0.05 * np.min(amp_tap_signal))
    #     # # enlarged_seq = ratio * np.min(amp_tap_signal) * enlarged_seq 
    #     # # tap_signal = (amp_tap_signal + enlarged_seq) * np.exp(1j * angle_tap_signal)  # Modulate the tap signal with the PN sequence

    #     # tap_signal = (butter_highpass_filter(tap_signal, cutoff=50, fs=fs, order=5))
    #     # tap_signal = np.diff(tap_signal)
    #     # pn_signal = np.diff(get_PN(bits=bits)) 
    #     # correlation_results = np.correlate(np.real(tap_signal), pn_signal, mode='valid') + 1j * np.correlate(np.imag(tap_signal), pn_signal, mode='valid')
    #     # period_length = (2**bits - 1)
    #     # correlation_results_ = correlation_results[0: len(correlation_results)//period_length*period_length]
    #     # correlation_results_ = np.abs(correlation_results_.reshape(-1, period_length).sum(axis=0))
    #     # current_max_correlation = np.max(correlation_results_)

    #     circular_corr = circular_correlation(tap_signal, seq)
    #     current_max_correlation = np.max(circular_corr)
    #     if current_max_correlation > max_correlation:
    #         max_correlation = current_max_correlation
    #         print(f"Max correlation for tap {tap_idx}: {max_correlation:.2f}")
    #         if PLOT and max_correlation > THRD:
    #             plt.plot(np.arange(len(circular_corr)), circular_corr, label=f"CIR Tap {tap_idx}")
    #             plt.xlabel("Time (s)")
    #             plt.ylabel("Amplitude")
    #             plt.title("Correlation Results")
    #             plt.axhline(y=0, color='k', linestyle='--', label='Zero Line')
    #             plt.legend()
    #             plt.show()
    #             plt.close()
    # plt.plot(correlation_results, label='Raw Correlation')
    # plt.plot(correlation_results_, label='Integrated Correlation')
    # plt.xlabel("Time (s)")
    # plt.ylabel("Amplitude")
    # plt.title("Correlation Results")
    # plt.axhline(y=0, color='k', linestyle='--', label='Zero Line')
    # plt.legend()
    # plt.show()
    # plt.close()
    # exit()

    # plot_specrum(tap_signal, fs=fs, title='Spectrum of Tap Signal')

    # # apply a low-pass filter (fs/2)
    # filtered_signal = butter_lowpass_filter(tap_signal, cutoff=245, fs=fs, order=4)

    # f, t_stft, Zxx = stft(np.diff(np.abs(filtered_signal)), fs=fs, nperseg=256, noverlap=128)
    # plt.pcolormesh(t_stft, f, 10*np.log10(np.abs(Zxx)), shading='gouraud')
    # plt.title('STFT Magnitude')
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.colorbar(label='Magnitude')
    # plt.show()

    # fft_analysis(((h_corrected)), time_real)

    # correlation_PN_analysis(np.abs(processed_cir_data), seq, tap_list=tap_list, PLOT=PLOT, THRD=THRD)    
    # waterfall_cir(processed_cir_data, time_real, time_stamp_suffix, save=False)
