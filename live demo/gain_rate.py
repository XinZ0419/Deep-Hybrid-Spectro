from scipy import signal
from scipy.signal import find_peaks
import pandas as pd


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band', analog=False)
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def gain_freq(rate_file):
    temp_data = pd.read_csv(rate_file, sep="\t")
    temp_data.drop([len(temp_data) - 1], inplace=True)
    sampling_rate = 1000
    # based on resistance
    temp_data['Resistance'] = temp_data['Volts'] / temp_data['Amps']
    time_slot = len(temp_data['Resistance']) / sampling_rate

    filtered_wave_pulse1 = butter_lowpass_filter(temp_data['Resistance'].values, 1.5, sampling_rate, 5)
    filtered_wave_pulse2 = butter_highpass_filter(filtered_wave_pulse1, 1.5, sampling_rate, 2)
    peak_index_pulse, _ = find_peaks(filtered_wave_pulse2, distance=330)
    freq_p = 60 * len(peak_index_pulse) / time_slot

    filtered_wave_breath1 = butter_lowpass_filter(temp_data['Resistance'].values, 1, sampling_rate, 5)
    filtered_wave_breath2 = butter_highpass_filter(filtered_wave_breath1, 0.1, sampling_rate, 2)
    peak_index_breath, _ = find_peaks(filtered_wave_breath2, distance=2000)
    freq_b = len(peak_index_breath) / time_slot

    return round(freq_p, 2), freq_b
