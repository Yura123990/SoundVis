import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

def plot_standard_audio_analysis(filename):
    # 1. Read the file
    data, fs = sf.read(filename)
    if data.ndim > 1:
        data = data.mean(axis=1) # Convert stereo to mono

    duration_sec = 30
    num_samples = int(duration_sec * fs)
    data = data[:num_samples]

    # 2. Prepare FFT (Spectrum)
    n = len(data)
    yf = rfft(data)
    xf = rfftfreq(n, 1 / fs)
    amplitude_spectrum = np.abs(yf)

    # 3. Plotting
    plt.figure(figsize=(10, 8))

    time_axis = np.linspace(0, len(data) / fs, num=len(data))
    plt.plot(time_axis, data)
    plt.title("Waveform (Time Domain)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.show()

plot_standard_audio_analysis("testsounds-2/30-TWO_SAWTOOTH.wav")