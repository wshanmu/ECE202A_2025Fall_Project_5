import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cir_utils import *
# import scienceplots
plt.style.use(['science', 'no-latex'])
import matplotlib
matplotlib.rc('font', family='Helvetica')
import re


def generate_small_kasami_set(m_seq, d=257):
    L = len(m_seq)
    n_half = int(np.log2(d - 1))  # for 257, this gives 8
    size = 2 ** n_half

    kasami_set = []
    for i in range(size):
        # m(dt + i) modulo L
        indices = (d * np.arange(L) + i) % L
        m_d_i = m_seq[indices]
        kasami_seq = m_seq * m_d_i  # XOR in bipolar is multiplication
        kasami_set.append(kasami_seq)
    return np.stack(kasami_set, axis=0)


def pack_bits_to_bytes(bits):
    bits = (bits > 0).astype(np.uint8)  # Ensure binary 0/1
    # Pad to multiple of 8
    pad_len = (-len(bits)) % 8
    if pad_len > 0:
        bits = np.concatenate([bits, np.zeros(pad_len, dtype=np.uint8)])
    packed = np.packbits(bits)
    return packed


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


def generate_and_save_kasami_codes(m_seq, d=257, save_to_files=True, multiple=8, bit_length=16):
    """
    Generate Kasami code set and optionally save to header files.

    Args:
        m_seq: Maximum length sequence
        d: Decimation parameter (default 257)
        save_to_files: Whether to save codes to C header files
        multiple: Number of times to tile the sequence (default 8)
        bit_length: Bit length for naming files (default 16)

    Returns:
        kasami_set: Array of Kasami sequences
    """
    print(f"Generating {bit_length}-bit Kasami code set with d={d}...")
    kasami_set = generate_small_kasami_set(m_seq, d=d)
    if save_to_files:
        print(f"Saving {len(kasami_set)} Kasami sequences to header files...")
        for kasami_index in range(len(kasami_set)):
            pn = kasami_set[kasami_index, :]
            pn_multiple = np.tile(pn, multiple)
            # Save the kasami set to a file for ESP32 use
            packed_seq = pack_bits_to_bytes(pn_multiple)
            # Export to C header
            with open(f"./pn_code_generation_and_simulation/kasami_code/kasami_packed_{bit_length}bit_dup{multiple}_{kasami_index}.h", "w") as f:
                f.write("#include <stdint.h>\n\n")
                f.write("const uint8_t kasami_sequence[] = {\n")
                for i, byte in enumerate(packed_seq):
                    f.write(f"0x{byte:02X}, ")
                    if (i + 1) % 16 == 0:
                        f.write("\n")
                f.write("};\n")
                f.write(f"const uint32_t kasami_bit_len = {len(pn_multiple)};\n")

        # Verify the first saved sequence
        pn_bits = read_packed_kasami_h(f"./pn_code_generation_and_simulation/kasami_code/kasami_packed_{bit_length}bit_dup{multiple}_0.h")
        print(f"First 20 bits of saved sequence: {pn_bits[:20]}")
        print(f"Verification - matching bits: {np.sum(np.tile(kasami_set[0, :], multiple) == 2*pn_bits-1)}")

    return kasami_set


def generate_all_kasami_codes(bit_lengths=[10, 12], multiple=1, save_to_files=True):
    """
    Generate Kasami codes for multiple bit lengths.

    Small Kasami sets require even n (MLS order).
    Decimation factor d = 2^(n/2) + 1
    Number of sequences = 2^(n/2)

    Args:
        bit_lengths: List of bit lengths to generate (must be even for small Kasami)
        multiple: Number of times to tile the sequence
        save_to_files: Whether to save to header files

    Returns:
        Dictionary of kasami_sets keyed by bit_length
    """
    # MLS generators for each bit length
    mls_generators = {
        10: generate_mls_lfsr_10,
        12: generate_mls_lfsr_12,
        14: generate_mls_lfsr_14,
        16: generate_mls_lfsr_16,
        18: generate_mls_lfsr_18,
    }

    results = {}

    for n in bit_lengths:
        if n % 2 != 0:
            print(f"Skipping {n}-bit: Small Kasami sets require even n")
            continue

        if n not in mls_generators:
            print(f"Skipping {n}-bit: No MLS generator available")
            continue

        # Calculate decimation factor for small Kasami set
        d = 2 ** (n // 2) + 1
        num_sequences = 2 ** (n // 2)
        seq_length = 2 ** n - 1

        print(f"\n{'='*50}")
        print(f"Generating {n}-bit Kasami codes:")
        print(f"  Sequence length: {seq_length}")
        print(f"  Decimation factor d: {d}")
        print(f"  Number of sequences: {num_sequences}")
        print(f"{'='*50}")

        # Generate MLS
        m_seq = mls_generators[n]()
        print(f"  M-sequence length: {len(m_seq)}, sum: {np.sum(m_seq)}")

        # Generate and save Kasami codes
        kasami_set = generate_and_save_kasami_codes(
            m_seq, d=d, save_to_files=save_to_files,
            multiple=multiple, bit_length=n
        )
        results[n] = kasami_set

        print(f"  Generated {len(kasami_set)} sequences of length {kasami_set.shape[1]}")

    return results


def simulate_signal_with_interference(pn_sequence, bit=16, noise=10):
    """
    Create a simulated signal with interference, noise, and the PN sequence.
    
    Args:
        pn_sequence: The PN sequence to embed in the signal
    
    Returns:
        sampled: The complete simulated signal
        static_vector: The interference component
    """
    print("Simulating signal with interference and noise...")
    
    # Create static interference vector
    static_vector = np.zeros(2**bit-1+100, dtype=complex)
    
    # Add a sine wave to the static vector
    static_vector[0:len(pn_sequence)+100] = 10 * np.exp(1j * 2 * np.pi * 0.1 * np.arange(len(pn_sequence)+100))
    static_vector[0:len(pn_sequence)+100] += 3+4j
    
    # Create tiled PN sequence
    pn_noisy = np.tile(pn_sequence, 2)[0:len(static_vector)]
    pn_noisy = np.roll(pn_noisy, 0)
    
    # Combine signal components with noise
    sampled = (static_vector + 
               pn_noisy * (1+0.5j) + 
               (np.random.normal(0, noise, len(static_vector)) + 
                1j * np.random.normal(0, noise, len(static_vector))))
    
    return sampled, static_vector


def calculate_missing_indices(pn_length, initial_offset=1.0, per_sample_duration_offset=20e-6):
    """
    Calculate indices where samples might be missing due to timing offsets.
    
    Args:
        pn_length: Length of the PN sequence
        initial_offset: Initial timing offset
        per_sample_duration_offset: Per-sample duration offset
    
    Returns:
        missing_indices: List of indices where samples are missing
    """
    missing_indices = []
    missing_idx = (initial_offset) // per_sample_duration_offset + 1
    missing_indices.append(missing_idx)
    print(f"Missing index: {missing_idx}")

    L = pn_length
    if missing_idx < L:
        normal_missing_length = 1 // per_sample_duration_offset + 1
        print(f"Normal missing length: {normal_missing_length}")
        L -= missing_idx
        leftover = L // normal_missing_length
        if leftover > 0:
            for i in range(leftover):
                missing_indices.append(missing_idx + i * normal_missing_length)

    print(f"Missing indices: {missing_indices}")
    return missing_indices


def perform_correlation_analysis(pn_sequence, sampled_signal):
    """
    Perform various correlation analyses on the signal.
    
    Args:
        pn_sequence: The reference PN sequence
        sampled_signal: The sampled signal to analyze
    """
    print("\nPerforming correlation analysis...")
    
    # Random indices selection (for potential future use)
    random_indices = np.random.choice(len(sampled_signal), 1, replace=False)
    print(f"Randomly picked indices: {random_indices}")
    
    # Use the signal as-is for interpolation
    sampled_interpolated = sampled_signal.copy()
    
    # Apply high-pass filtering
    pn_hp = butter_highpass_filter(pn_sequence, cutoff=250, fs=1000, order=5)
    sampled_interpolated_hp = butter_highpass_filter(sampled_interpolated, cutoff=250, fs=1000, order=5)
    
    # Correlation analysis with original signal
    print("\n1. Correlation with original interpolated signal:")
    corr_real = autocorr_same_length(pn_sequence, sampled_interpolated, showFig=True)
    snr = np.max(corr_real) / np.mean(corr_real[:])
    print(f"SNR: {snr:.2f}")
    
    # Correlation analysis with high-pass filtered signal
    print("\n2. Correlation with high-pass filtered signal:")
    corr_real = autocorr_same_length(pn_sequence, np.real(sampled_interpolated_hp), showFig=True)
    snr = np.max(corr_real) / np.mean(corr_real[:])
    print(f"SNR: {snr:.2f}")
    
    # Correlation analysis with misaligned signals
    print("\n3. Correlation with misaligned signal (removed index 15000):")
    misaligned_sampled = np.append(np.delete(sampled_interpolated, [15000]), 0)
    misaligned_pn = np.append(np.delete(pn_sequence, [15000]), 0)
    corr_real = autocorr_same_length(misaligned_pn, np.real(misaligned_sampled), showFig=True)
    snr = np.max(corr_real) / np.mean(corr_real[:])
    print(f"SNR: {snr:.2f}")
    
    print("\n4. Correlation with misaligned signal (removed index 10000):")
    misaligned_sampled = np.append(np.delete(sampled_interpolated, [10000]), 0)
    corr_real = autocorr_same_length(pn_sequence, np.real(misaligned_sampled), showFig=True)
    snr = np.max(corr_real) / np.mean(corr_real[:])
    print(f"SNR: {snr:.2f}")


def main():
    """
    Main function to execute Kasami code generation and signal simulation.
    """
    # Set random seed for reproducibility
    np.random.seed(1)

    print("=== Kasami Code Generation ===")

    # Generate Kasami codes for 10-bit and 12-bit (even numbers only for small Kasami)
    # 11-bit and 13-bit are odd, so they don't support small Kasami sets
    results = generate_all_kasami_codes(
        bit_lengths=[14],
        multiple=8,
        save_to_files=True
    )

    print("\n=== Summary ===")
    for bit_len, kasami_set in results.items():
        print(f"{bit_len}-bit: {len(kasami_set)} sequences, length {kasami_set.shape[1]}")

    print("\n=== Kasami Code Generation Complete ===")


if __name__ == "__main__":
    main()