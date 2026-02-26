import torch
import torch.nn as nn
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 8, 9, 2, 4), nn.ReLU(),
            nn.Conv1d(8, 16, 9, 2, 4), nn.ReLU(),
            nn.Conv1d(16, 32, 9, 2, 4), nn.ReLU(),
            nn.AdaptiveAvgPool1d(64),
        )
        self.det_head = nn.Sequential(nn.Linear(32*64, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
        self.freq_head = nn.Sequential(nn.Linear(32*64, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        f = self.cnn(x)
        f = f.flatten(1)
        det = self.det_head(f)
        freq = self.freq_head(f)
        return det, freq

def predict_whole_wav(file_path, model, window_size=44100, hop_size=22050):

    # === 1. Завантаження ===
    data, sr = sf.read(file_path)
    if data.ndim > 1:
        data = data.mean(axis=1)

    num_windows = (len(data) - window_size) // hop_size + 1
    print(f"Total windows: {num_windows}")

    det_probs = []
    freq_preds = []

    # === 2. Обхід вікон ===
    for i in range(num_windows):
        start = i * hop_size
        end = start + window_size
        window = data[start:end]

        window = window * np.hanning(len(window))
        fft_vals = np.abs(rfft(window)).astype(np.float32)

        # padding / trimming
        max_len = 44100
        if len(fft_vals) > max_len:
            fft_vals = fft_vals[:max_len]
        elif len(fft_vals) < max_len:
            fft_vals = np.pad(fft_vals, (0, max_len - len(fft_vals)))

        x_tensor = torch.tensor(fft_vals).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            det_pred, freq_pred = model(x_tensor)

        det_probs.append(det_pred.item())
        freq_preds.append(freq_pred.item())

    # === 3. Агрегація по всьому файлу ===
    mean_prob = np.mean(det_probs)
    mean_freq = np.mean(freq_preds)

    print("\n=== FINAL RESULT ===")
    print(f"Signal probability (avg): {mean_prob:.3f}")
    print(f"Predicted frequency (avg): {mean_freq:.2f} Hz")

    # === 4. Глобальний FFT всього сигналу ===
    full_fft = np.abs(rfft(data * np.hanning(len(data))))
    freqs = rfftfreq(len(data), 1/sr)

    plt.figure(figsize=(12, 4))
    plt.plot(freqs, full_fft)
    plt.axvline(mean_freq, color='red', linestyle='--',
                label=f'AI prediction ≈ {mean_freq:.2f} Hz')
    plt.title("Full signal spectrum with AI prediction")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.xlim(0, 500)
    plt.legend()
    plt.grid(True)
    plt.show()

    return mean_prob, mean_freq

model_path = "best_model.pth"
model = SimpleNet()
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

wav_path = "C:/Users/yurga/PycharmProjects/SoundVis/testsounds-2/10hz-range.wav"

prob, freq = predict_whole_wav(wav_path, model)