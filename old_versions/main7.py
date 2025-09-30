import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

filename = "input17.wav"
start_sec = 0.0
duration = 35.0
search_band = (10, 40)
win_size = 2.0
min_repeats = 5


data, fs = sf.read(filename)
if data.ndim > 1:
    data = data.mean(axis=1)  # моно

start_sample = int(start_sec * fs)
N = int(duration * fs)
frag = data[start_sample:start_sample + N].astype(float)

Nwin = int(win_size * fs)
n_windows = len(frag) // Nwin

found_freqs = []

for i in range(n_windows):
    chunk = frag[i*Nwin:(i+1)*Nwin]
    X = np.abs(rfft(chunk))
    freqs = rfftfreq(len(chunk), 1/fs)

    mask = (freqs >= search_band[0]) & (freqs <= search_band[1])
    if not np.any(mask):
        continue

    peak_idx = np.argmax(X[mask])
    f_peak = freqs[mask][peak_idx]
    found_freqs.append(f_peak)


if not found_freqs:
    print("Жодного піку у цільовому діапазоні не знайдено.")
else:
    # округляємо до 0.5 Гц, щоб об’єднати близькі
    rounded = np.round(found_freqs, 1)
    vals, counts = np.unique(rounded, return_counts=True)
    best_idx = np.argmax(counts)
    f_peak = vals[best_idx]
    repeats = counts[best_idx]

    if repeats < min_repeats:
        print("Сигнал занадто нестабільний.")
        f_peak = None
    else:
        print(f"Стабільна пікова частота: {f_peak:.2f} Hz (зустрічається {repeats} разів)")

if f_peak is not None:
    samples_per_period = int(round(fs / f_peak))
    num_periods = len(frag) // samples_per_period

    stack = np.zeros(samples_per_period)
    for i in range(num_periods):
        chunk = frag[i*samples_per_period:(i+1)*samples_per_period]
        stack += chunk
    stack /= num_periods

    plt.figure(figsize=(10,6))

    plt.subplot(2,1,1)
    plt.hist(rounded, bins=np.arange(search_band[0], search_band[1]+0.5, 0.5), color="skyblue", edgecolor="k")
    plt.axvline(f_peak, color='r', linestyle='--', label=f"f_peak ≈ {f_peak:.2f} Hz")
    plt.xlabel("Частота, Hz")
    plt.ylabel("Кількість появ")
    plt.title("Частоти, знайдені у вікнах")
    plt.legend()
    plt.grid(True)

    plt.subplot(2,1,2)
    t = np.arange(samples_per_period) / fs
    plt.plot(t, stack)
    plt.xlabel("Час, s (1 період)")
    plt.ylabel("Амплітуда")
    plt.title("Усереднений період сигналу")
    plt.grid(True)

    plt.tight_layout()
    plt.show()
