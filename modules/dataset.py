import numpy as np
from scipy.signal import stft
from torch.utils.data import Dataset


class Classifier(Dataset):
    def __init__(self, opt):
        self.temp_data = np.load(opt.processed_data_file, allow_pickle=True)

        self.data_raw = self.temp_data[:, :opt.num_samples_to_keep]

        ## STFT
        f, t, Zxx = stft(self.data_raw, fs=opt.sampling_rate, window='hann', nperseg=512, noverlap=256, nfft=None, boundary=None,
                         padded=False, axis=-1)
        intens = 20 * np.log10(np.abs(Zxx), where=np.abs(Zxx) > 0)
        self.data_freq = intens

        self.label = self.temp_data[:, opt.num_samples_to_keep]
        self.partic = self.temp_data[:, opt.num_samples_to_keep + 1]

        self.pl = self.temp_data[:, -1]

    def __len__(self):
        return len(self.data_raw)

    def __getitem__(self, idx):
        return self.data_raw[idx, :], self.data_freq[idx, :], self.label[idx], self.partic[idx], self.pl[idx]
