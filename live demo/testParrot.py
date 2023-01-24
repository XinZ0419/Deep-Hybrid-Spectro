import socket
from time import sleep
from scipy import signal
from scipy.signal import find_peaks, stft
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
import torch.nn as nn


# %% rate detection
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
    sampling_rate = 1000
    # based on resistance
    temp_data['Resistance'] = temp_data['Potential (V)'] / temp_data['Amps']
    time_slot = len(temp_data['Resistance']) / sampling_rate

    filtered_wave_pulse1 = butter_lowpass_filter(temp_data['Resistance'].values, 1.5, sampling_rate, 5)
    filtered_wave_pulse2 = butter_highpass_filter(filtered_wave_pulse1, 1.5, sampling_rate, 2)
    peak_index_pulse, _ = find_peaks(filtered_wave_pulse2, distance=330)
    freq_p = len(peak_index_pulse) / time_slot

    # =========================================================================================
    filtered_wave_breath1 = butter_lowpass_filter(temp_data['Resistance'].values, 1, sampling_rate, 5)
    filtered_wave_breath2 = butter_highpass_filter(filtered_wave_breath1, 0.1, sampling_rate, 2)
    peak_index_breath, _ = find_peaks(filtered_wave_breath2, distance=2000)
    freq_b = len(peak_index_breath) / time_slot

    return freq_p, freq_b

#%% predict classification
class Classifier(Dataset):
    def __init__(self, temp_array):
        self.data_raw = temp_array[:, :3000]

        ## STFT
        f, t, Zxx = stft(self.data_raw, fs=1000, window='hann', nperseg=512, noverlap=256, nfft=None, boundary=None,
                         padded=False, axis=-1)
        intens = 20 * np.log10(np.abs(Zxx), where=np.abs(Zxx) > 0)
        self.data_freq = intens

    def __len__(self):
        return len(self.data_raw)

    def __getitem__(self, idx):
        return self.data_raw[idx, :], self.data_freq[idx, :]


# ===================================================================================================================
class LinearModel_fused_stftBased(nn.Module):
    def __init__(self, input_signal_size, num_classes, num_channels=1, drop_prob=0.3):
        super().__init__()
        self.input_signal_size = input_signal_size
        self.num_classes = num_classes
        self.raw = nn.Sequential(
            nn.Linear(input_signal_size, 1024), nn.BatchNorm1d(1024), nn.LeakyReLU(),
            nn.Linear(1024, 256), nn.BatchNorm1d(256), nn.LeakyReLU())

        self.stft = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=8, kernel_size=3, padding=1), nn.LeakyReLU(),
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, padding=1), nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(82240, 1024), nn.LeakyReLU(), nn.Dropout(drop_prob),   ## for 512
            nn.Linear(1024, 256), nn.LeakyReLU(), nn.Dropout(drop_prob))

        self.visu = nn.Sequential(
            nn.Linear(2 * 256, 128), nn.LeakyReLU(),
            nn.BatchNorm1d(128), nn.Linear(128, 32))

        self.out = nn.Sequential(
            nn.Linear(32, num_classes))

    def get_weights(self):
        return self.weight

    def forward(self, t_raw, t_stft):
        raw_out = self.raw(t_raw)
        fft_out = self.stft(t_stft)
        t_in = torch.cat([raw_out, fft_out], dim=1)
        t_visu = self.visu(t_in)
        out = self.out(t_visu)
        return out


def test(model, device, single_data):
    # model in eval mode skips Dropout etc
    model.eval()

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    # set the requires_grad flag to false as in the test mode
    with torch.no_grad():
        data_raw = torch.tensor(single_data.data_raw)
        data_freq = torch.tensor(single_data.data_freq)
        data_raw, data_freq = data_raw.to(device), data_freq.to(device)
        data_freq = torch.unsqueeze(data_freq, dim=1)

        # the model on the data
        starter.record()
        output = model(data_raw.float(), data_freq.float())
        ender.record()
        torch.cuda.synchronize()
        run_time = starter.elapsed_time(ender)

        y_pred = output.argmax()
        predict = nn.functional.one_hot(y_pred, 11)
        # predict = y_pred.item()

    return predict, run_time


def prediction(class_file):
    temp_data = pd.read_csv(class_file, sep="\t")
    num_samples_to_keep_before = 1500
    num_samples_to_keep_after = 1500
    total_num_samples = 3000
    num_classes = 11
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # cut 3000 points
    indices_to_keep = temp_data['Amps'].idxmin()

    if indices_to_keep - num_samples_to_keep_before < 0:
        fix = num_samples_to_keep_before - indices_to_keep
        samples_to_keep = pd.Series(temp_data['Amps'][0])
        for q in range(fix - 1):
            samples_to_keep = samples_to_keep.append(pd.Series(temp_data['Amps'][0]), ignore_index=True)
        samples_to_keep = samples_to_keep.append(
            temp_data['Amps'][0:indices_to_keep + num_samples_to_keep_after], ignore_index=True)

    elif indices_to_keep + num_samples_to_keep_after > len(temp_data['Amps']):
        fix = indices_to_keep + num_samples_to_keep_after - len(temp_data['Amps'])
        samples_to_keep = temp_data['Amps'][
                          indices_to_keep - num_samples_to_keep_before:len(temp_data['Amps']) - 1]
        for q in range(fix + 1):
            samples_to_keep = samples_to_keep.append(
                pd.Series(temp_data['Amps'][len(temp_data['Amps']) - 1]), ignore_index=True)

    else:
        samples_to_keep = temp_data['Amps'][
                          indices_to_keep - num_samples_to_keep_before:indices_to_keep + num_samples_to_keep_after]

    # raw+fft
    normed_wave = samples_to_keep / samples_to_keep.max()
    temp_fft = np.fft.fft(normed_wave).real
    temp_array = np.expand_dims(np.concatenate([samples_to_keep, temp_fft]), 0)

    single_data = Classifier(temp_array)

    # prediction
    model = LinearModel_fused_stftBased(total_num_samples, num_classes, num_channels=1, drop_prob=0.3).to(device)
    state_dict = torch.load('Model1_fused_stft_Fold_0_allclass_twoinputs_net_parameter.pth')
    model.load_state_dict(state_dict)

    predict, run_time = test(model, device, single_data)

    return predict

#%% Send data to Visulization
def sendData(HOST, PORT, data):
    s = socket.socket()
    s.connect((HOST, PORT))
    s.send(data.encode())
    s.close()
 

# %% main
if __name__ == "__main__":

    HOST, PORT = socket.gethostname(), 11000

    rate = gain_freq('D://Xin Zhang/Sensor/conbination/NewData/All/move-shaking/MOVE-SHAKING - Copy (10).txt')
    gesture = prediction('D://Xin Zhang/Sensor/conbination/NewData/All/move-shaking/MOVE-SHAKING - Copy (10).txt')

    s = str(rate) + "," + str(gesture)
    print(s)
    sendData(HOST, PORT, s)
    sleep(3)