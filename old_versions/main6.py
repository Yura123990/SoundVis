import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

filename = "input7.wav"
start_sec = 0.0
duration = 55.0
search_band = (1, 200)
min_periods_required = 10

data, fs = sf.read(filename)
if data.ndim > 1:
    data = data.mean(axis=1)

start_sample = int(start_sec * fs)
N = int(duration * fs)
frag = data[start_sample:start_sample + N].astype(float)

X = np.abs(rfft(frag))
freqs = rfftfreq(len(frag), 1/fs)

mask = (freqs >= search_band[0]) & (freqs <= search_band[1])
if not np.any(mask):
    raise ValueError("У заданому діапазоні немає частот для аналізу!")

peak_idx = np.argmax(X[mask])
f_peak = freqs[mask][peak_idx]
amp_peak = X[mask][peak_idx]

if amp_peak < 0.05 * np.max(X):
    print("Повторюваного сигналу не виявлено (піковий сигнал занадто слабкий).")
    f_peak = None
else:
    print(f"Пікова частота: {f_peak:.2f} Hz")

if f_peak is not None and f_peak > 0:
    samples_per_period = int(round(fs / f_peak))
    num_periods = len(frag) // samples_per_period

    if num_periods < min_periods_required:
        print("Недостатньо періодів у сигналі для впевненого повторення.")
    else:
        stack = np.zeros(samples_per_period)
        for i in range(num_periods):
            chunk = frag[i*samples_per_period:(i+1)*samples_per_period]
            stack += chunk
        stack /= num_periods

        plt.figure(figsize=(10,6))

        plt.subplot(2,1,1)
        plt.semilogy(freqs[mask], X[mask])
        plt.axvline(f_peak, color='r', linestyle='--', label=f'peak ≈ {f_peak:.2f} Hz')
        plt.xlabel("Частота, Hz")
        plt.ylabel("Амплітуда")
        plt.title("Спектр (FFT)")
        plt.legend()
        plt.grid(True)

        plt.subplot(2,1,2)
        t = np.arange(samples_per_period) / fs
        plt.plot(t, stack)
        plt.xlabel("Час, s (1 період)")
        plt.ylabel("Амплітуда")
        plt.title("Накладений сигнал (усереднений період)")
        plt.grid(True)

        plt.tight_layout()
        plt.show()
