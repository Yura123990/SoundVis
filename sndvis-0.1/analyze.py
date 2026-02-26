import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

def search(filename, start_sec, duration, search_band, min_periods_required, peak_ratio_threshold):
    # 1. Reading a file
    data, fs = sf.read(filename)
    if data.ndim > 1:
        data = data.mean(axis=1)

    start_sample = int(start_sec * fs)
    N = int(duration * fs)
    frag = data[start_sample:start_sample + N].astype(float)

    # 2. Signal conversion in FFT
    X = np.abs(rfft(frag))
    freqs = rfftfreq(len(frag), 1 / fs)

    # 3. Searching for a peak in a specified range
    mask = (freqs >= search_band[0]) & (freqs <= search_band[1])
    if not np.any(mask):
        raise ValueError("There are no frequencies for analysis in the specified range!")

    X_band = X[mask]
    f_band = freqs[mask]

    peak_idx = np.argmax(X_band)
    f_peak = f_band[peak_idx]
    amp_peak = X_band[peak_idx]

    mean_amp = np.mean(X_band)
    if amp_peak < peak_ratio_threshold * mean_amp:
        f_peak = None
        print("No peak")
    else:
        print(f"Repetitive signal detected, peak frequency ≈ {f_peak:.2f} Hz")

    if f_peak is not None and f_peak > 0:
        samples_per_period = int(round(fs / f_peak))
        num_periods = len(frag) // samples_per_period

        if num_periods < min_periods_required:
            print("Not enough periods for confident repetition.")
        else:
            stack = np.zeros(samples_per_period)
            for i in range(num_periods):
                chunk = frag[i * samples_per_period:(i + 1) * samples_per_period]
                stack += chunk
            stack /= num_periods

            plt.figure(figsize=(10, 6))

            plt.subplot(2, 1, 1)
            plt.semilogy(f_band, X_band)
            plt.axvline(f_peak, color='r', linestyle='--', label=f'peak ≈ {f_peak:.2f} Hz')
            plt.xlabel("Frequency, Hz")
            plt.ylabel("Amplitude")
            plt.title("Spectrum (FFT)")
            plt.legend()
            plt.grid(True)

            plt.subplot(2, 1, 2)
            t = np.arange(samples_per_period) / fs
            plt.plot(t, stack)
            plt.xlabel("Time, s (1 period)")
            plt.ylabel("Amplitude")
            plt.title("Applied signal (averaged period)")
            plt.grid(True)

            plt.tight_layout()
            plt.show(block=False)
