import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

filename = "input11.wav"
duration = 55.0
search_band = (1, 200)
peak_ratio_threshold = 5

data, fs = sf.read(filename)
if data.ndim > 1:
    data = data.mean(axis=1)

N = int(duration * fs)
frag = data[:N].astype(float)

X = np.abs(rfft(frag))
freqs = rfftfreq(len(frag), 1/fs)

mask = (freqs >= search_band[0]) & (freqs <= search_band[1])
band_freqs = freqs[mask]
band_ampl = X[mask]

if len(band_ampl) == 0:
    raise ValueError("У діапазоні немає частот для аналізу!")

peak_idx = np.argmax(band_ampl)
f_peak = band_freqs[peak_idx]
amp_peak = band_ampl[peak_idx]
amp_mean = np.mean(band_ampl)

if amp_peak > peak_ratio_threshold * amp_mean:
    print(f"Повторюваний сигнал виявлено: {f_peak:.2f} Hz")
else:
    print("Повторюваний сигнал не виявлено")

plt.figure(figsize=(8,4))
plt.semilogy(band_freqs, band_ampl)
plt.axvline(f_peak, color='r', linestyle='--', label=f'peak ≈ {f_peak:.2f} Hz')
plt.xlabel("Частота, Hz")
plt.ylabel("Амплітуда")
plt.title("Спектр (FFT)")
plt.legend()
plt.grid(True)
plt.show()
