import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

# Задаю параметри
filename = "testsounds-1/input10.wav"
start_sec = 0.0           # початок фрагменту
duration = 30.0           # довжина фрагмента в секундах
search_band = (1, 200)    # діапазон пошуку піку (Hz)
min_periods_required = 600 # мінімальна кількість періодів для знайдення повторюваного сигналу
peak_ratio_threshold = 5  # у скільки разів пік має бути вищим за середнє

# 1. Зчитування файлу
data, fs = sf.read(filename)
if data.ndim > 1:
    data = data.mean(axis=1)

start_sample = int(start_sec * fs)
N = int(duration * fs)
frag = data[start_sample:start_sample + N].astype(float)

# 2. Перетворення сигналу у FFT
X = np.abs(rfft(frag))
freqs = rfftfreq(len(frag), 1/fs)

# 3. Пошук піку у заданому діапазоні
mask = (freqs >= search_band[0]) & (freqs <= search_band[1])
if not np.any(mask):
    raise ValueError("У заданому діапазоні немає частот для аналізу!")

X_band = X[mask]
f_band = freqs[mask]

peak_idx = np.argmax(X_band)
f_peak = f_band[peak_idx]
amp_peak = X_band[peak_idx]

# критерій: пік має значно перевищувати середнє по діапазону
mean_amp = np.mean(X_band)
if amp_peak < peak_ratio_threshold * mean_amp:
    print("Повторюваного сигналу не виявлено.")
    f_peak = None
else:
    print(f"Виявлено повторюваний сигнал, пікова частота ≈ {f_peak:.2f} Hz")

# 4. Якщо сигнал знайдено — усереднення періоду і графіки
if f_peak is not None and f_peak > 0:
    samples_per_period = int(round(fs / f_peak))
    num_periods = len(frag) // samples_per_period

    if num_periods < min_periods_required:
        print("Недостатньо періодів для впевненого повторення.")
    else:
        # Накладення шматків
        stack = np.zeros(samples_per_period)
        for i in range(num_periods):
            chunk = frag[i*samples_per_period:(i+1)*samples_per_period]
            stack += chunk
        stack /= num_periods  # усереднення

        # Візуалізація
        plt.figure(figsize=(10,6))

        # верхній графік — спектр
        plt.subplot(2,1,1)
        plt.semilogy(f_band, X_band)
        plt.axvline(f_peak, color='r', linestyle='--', label=f'peak ≈ {f_peak:.2f} Hz')
        plt.xlabel("Частота, Hz")
        plt.ylabel("Амплітуда")
        plt.title("Спектр (FFT)")
        plt.legend()
        plt.grid(True)

        # нижній графік — накладений сигнал
        plt.subplot(2,1,2)
        t = np.arange(samples_per_period) / fs
        plt.plot(t, stack)
        plt.xlabel("Час, s (1 період)")
        plt.ylabel("Амплітуда")
        plt.title("Накладений сигнал (усереднений період)")
        plt.grid(True)

        plt.tight_layout()
        plt.show()
