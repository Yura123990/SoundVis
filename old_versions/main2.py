import numpy as np
import matplotlib.pyplot as plt
import librosa

filename = "input.mp3"
segment_duration = 0.006711 # секунда
num_segments = 100

y, sr = librosa.load(filename, sr=None)
N = len(y)

mid_point = N // 2
segment_length = int(segment_duration * sr)
start_index = mid_point - (num_segments // 2) * segment_length

sum_signal = np.zeros(segment_length)

for i in range(num_segments):
    start = start_index + i * segment_length
    end = start + segment_length
    segment = y[start:end]

    sum_signal += segment

t = np.arange(len(sum_signal)) / sr

plt.figure(figsize=(12, 4))
plt.plot(t, sum_signal)
plt.title(f"Підсумований сигнал 10 сегментів по {segment_duration}s")
plt.xlabel("Час (секунди)")
plt.ylabel("Амплітуда")
plt.show()
