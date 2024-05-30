## This script is used to plot the signal in different domains
import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt
import scipy.signal as signal

class WaveVisualizer:
    """
    * This class is used to plot the signal in the different domains, namely time domain, frequency domain,
    * PSD, SPL, MelSpectrogram, STFT domain
    """
    def __init__(self, waveform, sample_rate, label=None):
        if isinstance(waveform, torch.Tensor):
            self.waveform = waveform.numpy().flatten()
        elif isinstance(waveform, np.ndarray):
            self.waveform = waveform.flatten()
        else:
            raise TypeError("Waveform must be a numpy array or a torch tensor")
        self.sample_rate = sample_rate
        self.label = label

    def plot_time_domain(self):
        t = np.arange(0, len(self.waveform)) / self.sample_rate
        plt.plot(t, self.waveform, label=self.label)
        plt.xlabel("Times [s]")
        plt.ylabel('Amplitude')
        plt.title('Waveform in Time domain')
        plt.grid(True)
        if self.label:
            plt.legend()

    def plot_frequency_domain(self):
        f = np.fft.fftfreq(len(self.waveform), d=1/self.sample_rate)
        X = np.fft.fft(self.waveform)
        plt.plot(f[:len(f)//2], np.abs(X)[:len(X)//2], label=self.label)
        plt.xlabel('Freuqency [Hz]')
        plt.ylabel('Amplitude')
        plt.xlim([0 ,5000])
        plt.title('Waveform in frequency domain')
        plt.grid(True)
        if self.label:
            plt.legend()

    def plot_PSD(self):
        f, Pxx_den = signal.periodogram(self.waveform, self.sample_rate)
        plt.semilogy(f, Pxx_den, label=self.label)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Power/Frequency [dB/Hz]')
        plt.title('The PSD of the waveform')
        plt.grid(True)
        if self.label:
            plt.legend()
    
    def plot_PSD_welch(self):
        f, Pxx_den = signal.welch(self.waveform, self.sample_rate, nperseg=1024)
        plt.semilogy(f, Pxx_den, label=self.label)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Power/Frequency [dB/Hz]')
        plt.title("The PSD of the waveform (welch)")
        plt.grid(True)
        if self.label:
            plt.legend()

    def compute_RMS_SPL(self, window_size=4096, overlap=1024):
        hop_size = window_size - overlap
        rms_values = []
        for i in range(0, len(self.waveform) - window_size, hop_size):
            waveform_window = self.waveform[i:i+window_size]
            rms = np.sqrt(np.mean(waveform_window ** 2))
            rms_values.append(rms)

        rms_values = np.array(rms_values)
        spl_values = 20 * np.log10(rms_values / 20e-6)
        spl_avg = np.mean(spl_values)
        return spl_avg, spl_values

    def plot_RMS_SPL(self, window_size=4096, overlap=1024):
        hop_size = window_size - overlap
        spl_avg, spl_values = self.compute_RMS_SPL(window_size, overlap)
        t = np.arange(len(spl_values)) * hop_size / self.sample_rate
        plt.plot(t, spl_values, label=self.label)
        plt.xlabel('Time [s]')
        plt.ylabel('SPL [dB]')
        plt.title(f'The RMS SPL of the waveform {spl_avg:.2f} dB')
        print(f"The SPL is {spl_avg:.2f} dB")
        plt.grid(True)
        if self.label:
            plt.legend()

    def plot_spectrogram(self):
        f, t, Sxx = signal.spectrogram(self.waveform, self.sample_rate)
        plt.pcolormesh(t, f, 10 * np.log10(Sxx))
        plt.xlabel('Time[s]')
        plt.ylabel('Frequency[Hz]')
        plt.title('Waveform in spectrogram domain')
        plt.colorbar(label = 'Magnitude[dB]')
        plt.grid(True)

    def plot_mel_spectrogram(self):
        waveform_tensor = torch.from_numpy(self.waveform).float()
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate = self.sample_rate,
            n_fft = 2048,
            hop_length = 512,
            n_mels = 128
        )
        waveform_mel = mel_transform(waveform_tensor)

        plt.imshow(waveform_mel.log2().detach().numpy(), cmap='viridis', origin='lower', aspect='auto')
        plt.xlabel('Time [s]')
        plt.ylabel('Mel Frequency [Hz]')
        plt.title('Waveform in Mel-spectrogram domain')
        plt.colorbar(format='%+2.0f dB')
        plt.grid(True)


def design_a_weighting(fs):
    """
    * This function is used to design an a-weighting filter according to the sampling rate
    @ params:
    - fs: sampling rate [float]
    @ returns:
    - b: Numerator polynomials of the IIR [ndarray]
    - a: Denominator polynomials of the IIR [ndarray]
    """
    f1 = 20.598997
    f2 = 107.65265
    f3 = 737.86223
    f4 = 12194.217
    A1000 = 1.9997

    numerators = [(2*np.pi * f4)**2 * (10**(A1000/20)), 0, 0, 0, 0]
    denominators = np.convolve([1, 4*np.pi * f4, (2*np.pi * f4)**2],
                               [1, 4*np.pi * f1, (2*np.pi * f1)**2])
    denominators = np.convolve(np.convolve(denominators, 
                                           [1, 2*np.pi * f3]),
                               [1, 2*np.pi * f2])
    
    b, a = signal.bilinear(numerators, denominators, fs)
    return b, a

def a_weighting(waveform, fs):
    b, a = design_a_weighting(fs)
    filtered_wave = signal.lfilter(b,a,waveform)
    return filtered_wave


if __name__ == "__main__":
    noise_path1 = r"D:/Coding/Selective_ANC_CNN/Real_Noise/Aircraft.wav"
    waveform1, sample_rate1 = torchaudio.load(noise_path1)
    noise_path2 = r"D:/Coding/Selective_ANC_CNN/Real_Noise/Speech@24kHz.wav"
    waveform2, sample_rate2 = torchaudio.load(noise_path2)
    waveform1_a = a_weighting(waveform1, sample_rate1)

    visualizer1 = WaveVisualizer(waveform1, sample_rate1, label="Aircraft Noise")
    visualizer2 = WaveVisualizer(waveform1_a, sample_rate1, label="Speech noise")

    spl_avg1, _ = visualizer1.compute_RMS_SPL()
    spl_avg2, _ = visualizer2.compute_RMS_SPL()

    visualizer1.label = f"Aircraft Noise: {spl_avg1:.2f} dB"
    visualizer2.label = f"A-weighted Noise: {spl_avg2:.2f} dB"


    plt.figure(figsize=(10,4))
    visualizer1.plot_RMS_SPL()
    visualizer2.plot_RMS_SPL()
    plt.legend()
    plt.show()
