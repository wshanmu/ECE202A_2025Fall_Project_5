from cir_utils import *

bits = 10
seq = get_PN(bits)

# --- Step 1: Parameters ---
pn_length = len(seq)  # Length of PN sequence
symbol_rate = 1000  # Hz → symbol_duration = 1/64 s
symbol_duration = 1 / symbol_rate
mod_freq = 256  # Hz (modulation frequency)
t_total = pn_length * symbol_duration
print(f"Total time: {t_total:.2f} s")

# Time resolution for waveform
base_fs = 64000  # High-res simulation (64kHz)
t = np.arange(0, t_total, 1 / base_fs)

# Generate PN waveform
pn_waveform = np.repeat(seq, len(t) // pn_length + 1)[:len(t)]
print(len(pn_waveform), len(t))

# Modulate with sgn(sin(...))
mod_wave = np.sign(np.sin(2 * np.pi * mod_freq * t))
mixed_signal = pn_waveform * mod_wave
mixed_signal = pn_waveform
# mixed_signal = mod_wave # square wave

# --- Sampling with clock drift ---
epsilon = 0.00  # 5% clock drift → sampling interval = 1ms * (1 + epsilon)
ideal_sample_interval = 0.001  # 1ms
actual_sample_interval = ideal_sample_interval * (1 + epsilon)
init_phase = 0 / 8 * actual_sample_interval  # start sampling at 3/8 of the interval

# Create drifting sample times
sampling_times = np.arange(init_phase, t_total, actual_sample_interval)
sampled_signal = np.interp(sampling_times, t, mixed_signal) + np.random.normal(0, 0, len(sampling_times))


# For circular correlation
t = np.arange(0, t_total, 0.001)
# Generate PN waveform
pn_waveform = np.repeat(seq, len(t) // pn_length + 1)[:len(t)]
# Modulate with sgn(sin(...))
mod_wave = np.sign(np.sin(2 * np.pi * (mod_freq/(1-epsilon)) * (t - 1/8 * 0.001)))
mixed_signal = pn_waveform

print(len(sampled_signal), len(mixed_signal))
N = len(mixed_signal)
tap_spec = fft(sampled_signal, n=N)
seq_spec = fft(mixed_signal, n=N)
acorrcirc = ifft(tap_spec * np.conj(seq_spec)).real
print(f"Max correlation: {np.max(acorrcirc)}")
plt.plot(np.arange(-N/2+1, N/2+1), fftshift(acorrcirc), '.-')
plt.show()
# exit()


# # --- Plot the result ---
# plt.figure(figsize=(10, 4))
# plt.plot(t[:8000], mixed_signal[:8000], label='Mixed Signal (Zoomed)')
# plt.plot(sampling_times[:80], sampled_signal[:80], 'ro', label='Samples with Drift')
# plt.xlabel("Time (s)")
# plt.title("Clock Drift Sampling (5% offset)")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# perform FFT on the sampled signal
N = len(sampled_signal)
dt = 0.001  # sampling interval (consider it 1ms)
freqs = np.fft.rfftfreq(N, d=dt)      # frequency axis for real FFT
FFT_vals = np.fft.rfft(sampled_signal * np.hamming(N))  # apply Hamming window to reduce edge effects
# freqs = np.fft.fftshift(freqs)  # shift zero frequency to center
# FFT_vals = np.fft.fftshift(FFT_vals)  # shift zero frequency to center
plt.plot(freqs, 10*np.log10(np.abs(FFT_vals)**2), label='FFT of Sampled Signal')
plt.xlabel("Frequency (Hz)")
plt.show()

