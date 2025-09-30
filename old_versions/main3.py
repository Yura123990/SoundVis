import numpy as np
import matplotlib.pyplot as plt
import librosa


filename = "input17.wav"
start_time = 60.0
duration = 0.0134
max_plot_freq = 750
show_db = True

y, sr = librosa.load(filename, sr=None, mono=True)
total_seconds = len(y) / sr
print(f"Loaded '{filename}': {len(y)} samples, sr = {sr} Hz, duration = {total_seconds:.2f} s")

start_sample = int(np.round(start_time * sr))
segment_length = int(np.round(duration * sr))
end_sample = start_sample + segment_length

if start_sample < 0:
    start_sample = 0
if end_sample > len(y):
    end_sample = len(y)
segment = y[start_sample:end_sample]
seg_len = len(segment)
if seg_len == 0:
    raise ValueError("Відрізок порожній — перевірте start_time та duration відносно довжини файлу.")

print(f"Segment: start {start_sample} (≈{start_sample/sr:.4f}s), length {seg_len} samples (≈{seg_len/sr:.4f}s)")

window = np.hanning(seg_len)

n_fft = 1 if seg_len == 0 else 2 ** int(np.ceil(np.log2(seg_len)))
y_win = segment * window

fft_vals = np.fft.rfft(y_win, n=n_fft)
fft_freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
magnitude = np.abs(fft_vals)

freq_mask = fft_freqs <= max_plot_freq

if show_db:
    magnitude_db = 20 * np.log10(magnitude + 1e-12)  # +eps, щоб уникнути log(0)
    y_plot_fft = magnitude_db[freq_mask]
    y_label_fft = "Magnitude (dB)"
else:
    y_plot_fft = magnitude[freq_mask]
    y_label_fft = "Magnitude (linear)"

freqs_plot = fft_freqs[freq_mask]

f_res = sr / n_fft
print(f"FFT: n_fft = {n_fft}, frequency resolution ≈ {f_res:.2f} Hz")

nonzero_mask = freqs_plot > 0
idx_top = np.argsort(y_plot_fft[nonzero_mask])[-5:][::-1]
# map back to original indices
nonzero_indices = np.where(nonzero_mask)[0]
top_freqs = freqs_plot[nonzero_indices[idx_top]]
top_mags = y_plot_fft[nonzero_indices[idx_top]]

t = np.arange(start_sample, start_sample + seg_len) / sr

frame_length = int(0.02 * sr)   # 20 ms
hop_length = int(0.01 * sr)     # 10 ms

if frame_length > seg_len:
    frame_length = seg_len
if hop_length < 1:
    hop_length = 1

rms = librosa.feature.rms(y=segment, frame_length=frame_length, hop_length=hop_length)[0]
rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length) + (start_sample / sr)

if np.max(rms) > 0:
    rms_scaled = rms / np.max(rms) * (0.9 * np.max(np.abs(segment)))
else:
    rms_scaled = rms

plt.figure(figsize=(12, 7))

plt.subplot(2, 1, 1)
plt.plot(freqs_plot, y_plot_fft, linewidth=1)
plt.scatter(top_freqs, top_mags, s=20, color='red', zorder=3)
for f, m in zip(top_freqs[:3], top_mags[:3]):  # підписати 3 найвищі
    plt.annotate(f"{f:.1f} Hz", xy=(f, m), xytext=(f, m + (np.max(y_plot_fft)-np.min(y_plot_fft))*0.05),
                 ha='center', fontsize=8, arrowprops=dict(arrowstyle='->', lw=0.5))
plt.title("FFT-спектр відрізка" + (f" ({duration:.4f} s)"))
plt.xlabel("Частота (Hz)")
plt.ylabel(y_label_fft)
plt.grid(alpha=0.25)

plt.subplot(2, 1, 2)
plt.plot(t, segment, linewidth=0.6, label="Waveform")
plt.fill_between(t, segment, alpha=0.12)
plt.plot(rms_times, rms_scaled, linewidth=1.5, label="RMS envelope (20 ms)", linestyle='--')
plt.title("Часовий сигнал (waveform) з RMS-енвелопою")
plt.xlabel("Час (s)")
plt.ylabel("Амплітуда")
plt.legend(loc='upper right')
plt.grid(alpha=0.2)

plt.tight_layout()
plt.show()
