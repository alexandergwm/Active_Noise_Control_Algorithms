import numpy as np
from scipy.signal import butter, filtfilt
import scipy.io as sio
import os
import plot_tools


def generate_bandpass_wn(samples, fs, freq_low, freq_high):
    """
    Generates a bandpass-filtered white noise signal within a specified frequency range.

    @ params:
    - samples: int, number of samples to generate.
    - fs: int, sampling rate in Hz.
    - freq_low: float, low cutoff frequency of the bandpass filter in Hz.
    - freq_high: float, high cutoff frequency of the bandpass filter in Hz.

    @ returns:
    - np.array, the bandpass-filtered white noise signal.
    """
    # Generate random white noise
    noise = np.random.randn(samples)

    # Calculate filter coefficients
    nyq = 0.5 * fs  # Nyquist frequency
    low = freq_low / nyq
    high = freq_high / nyq
    b, a = butter(4, [low, high], btype='band')

    # Apply the filter
    bandpass_noise = filtfilt(b, a, noise)

    return bandpass_noise

def loading_paths_from_MAT(folder = r'D:/Coding/Gavin/Frequency-domain fast LMS'
                           ,subfolder = 'Primary and Secondary Path'
                           ,Pri_path_file_name = 'Primary_path.mat'
                           ,Sec_path_file_name ='Secondary_path.mat'):
    """
    * This function is used to load the primary path and secondary path from .mat files
    """
    Primay_path_file, Secondary_path_file = os.path.join(folder, subfolder, Pri_path_file_name), os.path.join(folder,subfolder, Sec_path_file_name)
    Pri_dfs, Secon_dfs = sio.loadmat(Primay_path_file), sio.loadmat(Secondary_path_file)
    Pri_path, Secon_path = Pri_dfs['Pz1'].squeeze(), Secon_dfs['S'].squeeze()
    return Pri_path, Secon_path




# Example usage
if __name__ == '__main__':
    fs = 48000 
    samples = int(30 * fs)  
    freq_low = 30  
    freq_high = 800  
    filtered_noise = generate_bandpass_wn(samples, fs, freq_low, freq_high)

    plot1 = plot_tools.WaveVisualizer(filtered_noise,fs)
    plot1.plot_frequency_domain()