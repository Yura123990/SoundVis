import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import windows
from scipy.fft import rfft, rfftfreq

filename = "input17.wav"
start_sec = 0.0
duration = 5.0
band = (1.0, 500.0)

data, fs = sf.read(filename)
if data.ndim > 1:
    data = data.mean(axis=1)

start_sample = int(start_sec * fs)
N = int(duration * fs)
frag = data[start_sample:start_sample + N].astype(float)

win = windows.hann(len(frag))
X = np.abs(rfft(frag * win))
freqs = rfftfreq(len(frag), 1/fs)

low, high = band
mask = (freqs >= low) & (freqs <= high)
idx_band = np.where(mask)[0]
peak_rel = np.argmax(X[idx_band])
peak_idx = idx_band[peak_rel]
f_peak = freqs[peak_idx]

print(f"Найвища амплітуда: {f_peak:.3f} Hz")

T = 1.0 / f_peak
samples_per_period = int(round(fs * T))

num_periods = len(frag) // samples_per_period
stack = np.zeros(samples_per_period)

for i in range(num_periods):
    chunk = frag[i*samples_per_period:(i+1)*samples_per_period]
    if len(chunk) == samples_per_period:
        stack += chunk

stack /= num_periods

plt.figure(figsize=(12,6))

plt.subplot(2,1,1)
plt.plot(freqs[mask], X[mask])
plt.axvline(f_peak, color='r', linestyle='--', label=f'peak {f_peak:.2f} Hz')
plt.xlabel("Частота, Hz")
plt.ylabel("Амплітуда")
plt.title("FFT спектр")
plt.legend()
plt.grid(True)

plt.subplot(2,1,2)
t = np.linspace(0, T, samples_per_period, endpoint=False)
plt.plot(t, stack)
plt.xlabel("Час, s (1 період)")
plt.ylabel("Амплітуда")
plt.title("Когерентне усереднення (накладання періодів)")
plt.grid(True)

plt.tight_layout()
plt.show()
