from torch.utils.data import Dataset
import numpy as np


length_signal = 3000
num_class_level = 1
num_sub_class_level = 0


class Classifier(Dataset):
    def __init__(self, file_name):
        self.temp_data = np.load(file_name, allow_pickle=True)

        self.data_raw = self.temp_data[:, :length_signal]
        self.data_fft = self.temp_data[:, length_signal:-(num_class_level + num_sub_class_level)]
        self.label = self.temp_data[:, -(num_class_level + num_sub_class_level):]

    def __len__(self):
        return len(self.data_raw)

    def __getitem__(self, idx):
        return self.data_raw[idx, :], self.data_fft[idx, :], self.label[idx]
