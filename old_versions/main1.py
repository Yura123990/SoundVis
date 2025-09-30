import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import rfft, rfftfreq

filename = "input2.mp3"

y, sr = librosa.load(filename, sr=None, mono=True)

center_time = 13
window = 5

start_sample = int((center_time - window/2) * sr)
end_sample   = int((center_time + window/2) * sr)
y_chunk = y[start_sample:end_sample]

segment_length = int(sr / 300)
num_segments = len(y_chunk) // segment_length
segments = y_chunk[:num_segments * segment_length].reshape(num_segments, segment_length)

y_sum = np.sum(segments, axis=0)

window_func = np.hanning(len(y_sum))
y_sum_windowed = y_sum * window_func

N_fft = max(8192, len(y_sum_windowed))
y_sum_padded = np.pad(y_sum_windowed, (0, N_fft - len(y_sum_windowed)), 'constant')

fft_spectrum = rfft(y_sum_padded)
freqs = rfftfreq(N_fft, 1/sr)
magnitude = np.abs(fft_spectrum)

time_sum = np.arange(len(y_sum)) / sr

plt.figure(figsize=(14, 6))

plt.subplot(2, 1, 1)
plt.plot(time_sum, y_sum, color="darkred")
plt.title("Сумований сигнал (відрізки по 1/88 с)")
plt.xlabel("Час [с]")
plt.ylabel("Амплітуда")

plt.subplot(2, 1, 2)
plt.plot(freqs, magnitude, color="navy")
plt.xlim(0, 750)
plt.title("Амплітудний спектр (до 750 Гц) з високою точністю")
plt.xlabel("Частота [Гц]")
plt.ylabel("Амплітуда")

plt.tight_layout()
plt.show()
