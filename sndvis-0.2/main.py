# diagnostic_inference.py
# Використання:
# python diagnostic_inference.py path/to/test.wav path/to/model.pth

import sys
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
import math
import os

# ---- Налаштування ----
WINDOW_SIZE_DEFAULT = 44100   # якщо модель навчалась на 1-сек вікні
SR = 44100

# ---- Завантаження моделі (використовуємо строгий load_state для видимості помилок) ----
def load_model_auto(model_path, device):
    state = torch.load(model_path, map_location=device)
    # Спроба вгадати архітектуру: якщо в state_dict є 'cnn.0.weight' та інші ключі —
    # користувачу краще дати зрозуміти, що модель потрібно завантажити у ту ж архітектуру.
    # Тут робимо універсальний wrapper: якщо model.pth містить entire model (saved model),
    # пробуємо завантажити як nn.Module (unsafe) — але спочатку просто повернемо state_dict.
    return state

# ---- Прості допоміжні моделі (контейнери) ----
# Ми не знаємо точно архітектуру, тому не намагаємось відтворити її тут.
# Ми зробимо inference методом: якщо state_dict зберігає повний torch model (scriptmodule), то завантажимо його.
def try_load_full_model(model_path, device):
    # Спробуємо просто torch.load -> якщо це ScriptModule або full model, воно буде callable
    m = None
    try:
        m = torch.load(model_path, map_location=device)
        if isinstance(m, torch.nn.Module):
            m.to(device)
            m.eval()
            return m
    except Exception:
        return None
    return None

# ---- Аналітичні функції ----
def dominant_freq_fft(wave, sr):
    N = len(wave)
    yf = np.fft.rfft(wave * np.hanning(N))
    mag = np.abs(yf)
    freqs = np.fft.rfftfreq(N, 1.0/sr)
    peak = np.argmax(mag)
    return freqs[peak], freqs, mag

def autocorr_peak_freq(wave, sr, fmin=1, fmax=500):
    # простий автокор функція для низьких частот
    x = wave - np.mean(wave)
    corr = np.correlate(x, x, mode='full')
    corr = corr[corr.size//2:]
    # знайдемо перший пік після нуля, конвертуємо у частоту
    peaks = np.where((corr[1:-1] > corr[:-2]) & (corr[1:-1] > corr[2:]))[0] + 1
    if len(peaks)==0:
        return None
    # перший пік -> період_samples
    period = peaks[0]
    freq = sr / period if period>0 else None
    if freq is None or freq < fmin or freq > fmax:
        # якщо не підходить, повернемо None
        return None
    return freq

def pad_or_trim(arr, target_len):
    if len(arr) > target_len:
        return arr[:target_len]
    elif len(arr) < target_len:
        return np.pad(arr, (0, target_len - len(arr)))
    else:
        return arr

# ---- Варианти препроцесингу і inference (якщо модель можна викликати) ----
def infer_with_model_callable(model, x_tensor):
    # model має повертати або (det, freq) або single output; намагаємось обробити
    with torch.no_grad():
        out = model(x_tensor)
    # нормалізуємо повернення
    if isinstance(out, tuple) or isinstance(out, list):
        # припускаємо (det, freq)
        det = out[0]
        freq = out[1]
        det_v = det.cpu().numpy().ravel()
        freq_v = freq.cpu().numpy().ravel()
        return det_v, freq_v
    else:
        # один тензор — трактуємо як freq предикт
        v = out.cpu().numpy().ravel()
        return None, v

# ---- Основна діагностика для одного файлу ----
def diagnose_file(wav_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # завантаження wav
    sig, sr = sf.read(wav_path)
    if sig.ndim > 1:
        sig = sig.mean(axis=1)
    print(f"Loaded {wav_path}: {len(sig)} samples, sr={sr}")

    # аналітична частота (FFT на повному файлі)
    peak_freq, freqs, mag = dominant_freq_fft(sig, sr)
    ac_freq = autocorr_peak_freq(sig, sr)
    print(f"Analytic dominant FFT peak: {peak_freq:.3f} Hz  (autocorr: {ac_freq})")

    # Спроба завантажити модель повністю (ScriptModule або повний модуль)
    model_full = try_load_full_model(model_path, device)
    if model_full is not None:
        print("Model loaded as full torch module (callable). Will try direct inference variants.")
    else:
        print("Model is state_dict (not full module) or could not be auto-loaded as Module. We'll still try call if possible.")

    # Якщо model_full є — використовуємо. Інакше пробуємо завантажити state_dict і спробувати викликати (ризик)
    # Але тут без точної архітектури не можемо load_state_dict -> module. Тому зосередимось на діагностиці препроцесингу.
    # ВАРІАНТ A: feed raw time-domain trimmed/padded to 44100 as [1,1,44100]
    td = pad_or_trim(sig, WINDOW_SIZE_DEFAULT)
    x_td = torch.tensor(td.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,44100]

    # ВАРІАНТ B: feed FFT magnitude (rfft) padded to WINDOW_SIZE_DEFAULT (we will use length = WINDOW_SIZE_DEFAULT)
    fft_mag = np.abs(np.fft.rfft(td))  # length ~ 22051
    fft_len = len(fft_mag)
    # make vector length = fft_len (we won't pad to 44100 since rfft length differs) but keep parity
    x_fft = torch.tensor(pad_or_trim(fft_mag, fft_len).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

    # ВАРІАНТ C: slice into windows of length WINDOW_SIZE_DEFAULT with hop WINDOW_SIZE_DEFAULT/2 and average model outputs per window
    hop = WINDOW_SIZE_DEFAULT // 2
    windows = []
    for i in range(0, max(1, len(sig) - WINDOW_SIZE_DEFAULT + 1), hop):
        w = sig[i:i+WINDOW_SIZE_DEFAULT]
        w = pad_or_trim(w, WINDOW_SIZE_DEFAULT)
        windows.append(w)
    if len(windows) == 0:
        windows = [pad_or_trim(sig, WINDOW_SIZE_DEFAULT)]
    X_windows = torch.tensor(np.stack(windows).astype(np.float32)).unsqueeze(1).to(device)  # [B,1,44100]

    # Тепер пробуємо викликати модель різними способами, якщо model_full є. Якщо ні — повідомимо, що потрібно надати архітектуру.
    if model_full is None:
        print("\nУвага: автоматично відтворити архітектуру з state_dict без коду тренування неможливо.")
        print("Надішли, будь ласка, файл з кодом, яким модель була збережена (архітектура class + torch.save).")
        print("Тим часом — ось локальні аналітичні значення та підготовлені варіанти входу для тесту (їх можна передати у модель вручну):")
        print(f" - time-domain input shape: {x_td.shape} (use this if model expects raw waveform length 44100)")
        print(f" - fft mag shape: {x_fft.shape} (use this if model expects FFT magnitude)")
        print(f" - windows batch shape: {X_windows.shape} (use this if model був навчений на вікнах)")
        # Також виведемо піки вікон (аналітичні), щоб порівняти:
        print("\nDominant freq per window (analytic):")
        for idx,w in enumerate(windows[:10]):
            f,_,_ = dominant_freq_fft(w, sr)
            print(f" window {idx}: {f:.3f} Hz")
        return {
            "analytic_peak": peak_freq,
            "autocorr_peak": ac_freq,
            "prepared": {
                "time_domain_tensor_shape": tuple(x_td.shape),
                "fft_tensor_shape": tuple(x_fft.shape),
                "windows_tensor_shape": tuple(X_windows.shape)
            }
        }

    # Якщо маємо callable модель — зробимо інференс трьома способами
    results = {}
    try:
        print("\nTrying inference on full model with raw time-domain padded-> [1,1,44100]")
        det_td, freq_td = infer_with_model_callable(model_full, x_td)
        results['time_domain'] = (det_td, freq_td)
        print(" time-domain -> det:", det_td, "freq:", freq_td)
    except Exception as e:
        print(" time-domain inference failed:", e)
    try:
        print("\nTrying inference with FFT magnitude (rfft) as [1,1,L]")
        det_fft, freq_fft = infer_with_model_callable(model_full, x_fft)
        results['fft_mag'] = (det_fft, freq_fft)
        print(" fft-mag -> det:", det_fft, "freq:", freq_fft)
    except Exception as e:
        print(" fft-mag inference failed:", e)
    try:
        print("\nTrying inference on all windows (batch) and aggregating")
        dets = []
        freqs = []
        B = 32
        for i in range(0, X_windows.shape[0], B):
            xb = X_windows[i:i+B]
            detb, freqb = model_full(xb)
            dets.append(detb.cpu().numpy())
            freqs.append(freqb.cpu().numpy())
        dets = np.concatenate(dets, axis=0)
        freqs = np.concatenate(freqs, axis=0)
        print(" windows -> dets mean/max:", dets.mean(), dets.max(), " freq mean (masked):", np.mean(freqs))
        results['windows'] = (dets, freqs)
    except Exception as e:
        print(" windows inference failed:", e)

    # Повернемо результати
    return {
        "analytic_peak": peak_freq,
        "autocorr_peak": ac_freq,
        "inference_results": results
    }

# ---- Виконання ----
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python diagnostic_inference.py path/to/test.wav path/to/model.pth")
        sys.exit(1)
    wav = sys.argv[1]
    model_path = sys.argv[2]
    if not os.path.exists(wav):
        print("WAV not found:", wav); sys.exit(1)
    if not os.path.exists(model_path):
        print("Model not found:", model_path); sys.exit(1)

    res = diagnose_file(wav, model_path)
    print("\n--- DIAGNOSTIC SUMMARY ---")
    from pprint import pprint
    pprint(res)
