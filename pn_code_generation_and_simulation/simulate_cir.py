import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

import scienceplots
from palettable.cartocolors.diverging import Geyser_4
colors = Geyser_4.mpl_colors

# fix random seed for reproducibility
np.random.seed(16)

# Simulation settings
Ts = 1.016e-9       # 1 ns sampling interval
Tmax = 30e-9   # 100 ns total duration
upsample_factor = 8
N = int(Tmax / Ts)
t = np.linspace(0, Tmax, N)
t_upsampled = np.linspace(0, Tmax, N * upsample_factor)
t_precise = np.linspace(0, Tmax, 1000 * N)

# Parameters of CIR
L = 5                           # number of paths
delay_times = np.sort(np.random.uniform(5e-9, Tmax, L - 1))  # delay times of paths
delay_times = np.insert(delay_times, 0, 5e-9)  # add the first path at t=0
amplitudes = np.random.rayleigh(scale=1.0, size=L)
phases = np.random.uniform(-np.pi, np.pi, size=L)
complex_gains = amplitudes * np.exp(1j * phases)

def root_raised_cosine_pulse(t, T, beta, center):
    # Root Raised Cosine Pulse
    t = t - center  # Center the pulse at the specified time
    pulse = np.zeros_like(t)
    factor = 4 * beta / np.pi / np.sqrt(T)
    small_t = np.abs(t) < 1e-12
    regular_t = ~small_t
    t_reg = t[regular_t]
    nominator = np.cos((1 + beta) * np.pi * t_reg / T) + np.sin((1 - beta) * np.pi * t_reg / T) / (4 * beta * t_reg / T)
    denominator = 1 - (4 * beta * t / T)**2
    pulse[regular_t] = nominator / denominator[regular_t]
    pulse[small_t] = 1 + (1 - beta) * np.pi / 4 / beta
    pulse /= np.sqrt(np.sum(pulse**2)/Ts)  # Normalize the pulse energy
    return factor * pulse


def upsample_with_rrc(h_samples, Ts, t_upsampled, T_pulse=2e-9, beta=0.5):
    N = len(h_samples)
    t_original = np.arange(N) * Ts
    h_upsampled = np.zeros_like(t_upsampled, dtype=complex)

    # Use the RRC as the interpolation kernel
    for i, h_i in enumerate(h_samples):
        center = t_original[i]
        h_upsampled += h_i * root_raised_cosine_pulse(t_upsampled, T=T_pulse, beta=beta, center=center)
    
    return h_upsampled

def upsample_with_fft(h_samples, upsample_factor):
    # upsample the CIR by 64 using FFT
    cir_datas_upsampled = np.zeros(len(h_samples) * upsample_factor, dtype=complex)
    X = np.fft.fft(h_samples)
    # zero-padding in the frequency domain
    X_upsampled = np.zeros(X.shape[0] * upsample_factor, dtype=complex)
    X_upsampled[:X.shape[0]//2] = X[:X.shape[0]//2]
    X_upsampled[-X.shape[0]//2:] = X[-X.shape[0]//2:]
    cir_datas_upsampled = np.fft.ifft(X_upsampled)
    return cir_datas_upsampled * upsample_factor


# Build CIR
h = np.zeros(N, dtype=complex)
h_precise = np.zeros(1000 * N, dtype=complex)
for i in range(L):
    h += complex_gains[i] * root_raised_cosine_pulse(t, T=2e-9, beta=0.5, center=delay_times[i])
    h_precise += complex_gains[i] * root_raised_cosine_pulse(t_precise, T=2e-9, beta=0.5, center=delay_times[i])

h_samples = h
h_upsampled = upsample_with_rrc(h_samples, Ts, t_upsampled) * 8
h_upsampled_fft = upsample_with_fft(h_samples, upsample_factor)

# Plot
plt.style.use(['science', 'no-latex'])
plt.rcParams['figure.figsize'] = (5, 3)
plt.figure()
discrete_t = np.arange(len(h_samples) * upsample_factor) 
plt.plot(discrete_t[::upsample_factor], np.abs(h), label='Magnitude', marker='o')
plt.plot(discrete_t, np.abs(h_upsampled), '-', label='Upsampled (RRC-based)')
plt.plot(discrete_t, np.abs(h_upsampled_fft), '-', label='Upsampled (FFT-based)')
# plt.plot(t_precise * 1e9, np.abs(h_precise), label='Magnitude (High Res)', linestyle='--')
# plt.plot(t * 1e9, np.angle(h), label='Phase')
plt.xlabel("Time [ns]")
plt.legend()
plt.grid()
plt.show()
plt.close()

# Plot the phase of the CIR
plt.figure()
plt.plot(discrete_t[::upsample_factor], np.angle(h), label='Phase', marker='o')
plt.plot(discrete_t, np.angle(h_upsampled), '-', label='Upsampled (RRC-based)')
plt.plot(discrete_t, np.angle(h_upsampled_fft), '-', label='Upsampled (FFT-based)')
plt.plot(np.linspace(0, len(discrete_t), len(h_precise)), np.angle(h_precise), label='Phase (High Res)', linestyle='--')
plt.xlabel("Time [ns]")
plt.legend()
plt.grid()
plt.show()
plt.close()
