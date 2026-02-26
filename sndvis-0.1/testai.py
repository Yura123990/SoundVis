import torch
import torch.nn as nn
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

def load_model(model_path):
    import torch
    import torch.nn as nn

    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv1d(1, 8, 9, 2, 4), nn.ReLU(),
                nn.Conv1d(8, 16, 9, 2, 4), nn.ReLU(),
                nn.Conv1d(16, 32, 9, 2, 4), nn.ReLU(),
                nn.AdaptiveAvgPool1d(64),
            )
            self.det_head = nn.Sequential(
                nn.Linear(32*64, 64), nn.ReLU(),
                nn.Linear(64, 1), nn.Sigmoid()
            )
            self.freq_head = nn.Sequential(
                nn.Linear(32*64, 64), nn.ReLU(),
                nn.Linear(64, 1)
            )

        def forward(self, x):
            f = self.cnn(x)
            f = f.flatten(1)
            det = self.det_head(f)
            freq = self.freq_head(f)
            return det, freq

    model = SimpleNet()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    return model

def predict_whole_wav(file_path,
                      model,
                      start_sec=0,
                      duration=None,
                      window_size=44100,
                      hop_size=22050):
    import torch
    import numpy as np
    import soundfile as sf
    from scipy.fft import rfft
    # === 1. Завантаження ===
    data, sr = sf.read(file_path)
    if data.ndim > 1:
        data = data.mean(axis=1)

    total_duration = len(data) / sr

    # === 2. Обрізання по часу ===
    start_sample = int(start_sec * sr)

    if duration is not None:
        end_sample = int((start_sec + duration) * sr)
    else:
        end_sample = len(data)

    # захист від виходу за межі файлу
    end_sample = min(end_sample, len(data))

    data = data[start_sample:end_sample]

    print(f"Analyzing from {start_sec:.2f}s "
          f"to {end_sample/sr:.2f}s "
          f"({len(data)/sr:.2f} seconds)")

    if len(data) < window_size:
        print("Fragment too short for one window.")
        return None, None

    # === 3. Вікна ===
    num_windows = (len(data) - window_size) // hop_size + 1
    print(f"Total windows: {num_windows}")

    det_probs = []
    freq_preds = []

    for i in range(num_windows):
        start = i * hop_size
        end = start + window_size
        window = data[start:end]

        window = window * np.hanning(len(window))
        fft_vals = np.abs(rfft(window)).astype(np.float32)

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

    # === 4. Агрегація ===
    mean_prob = np.mean(det_probs)
    mean_freq = np.mean(freq_preds)

    print("\n=== FINAL RESULT ===")
    print(f"Signal probability (avg): {mean_prob:.3f}")
#    print(f"Predicted frequency (avg): {mean_freq:.2f} Hz")

    # === 5. FFT всього фрагмента ===
    full_fft = np.abs(rfft(data * np.hanning(len(data))))
    freqs = rfftfreq(len(data), 1/sr)

    plt.figure(figsize=(12, 4))
    plt.plot(freqs, full_fft)
    plt.axvline(mean_freq, color='red', linestyle='--',
               label=f'AI prediction ≈ {mean_freq:.2f} Hz')
    plt.title("Fragment spectrum with AI prediction")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.xlim(0, 500)
    plt.legend()
    plt.grid(True)
    plt.show(block=False)

    return mean_prob, mean_freq
