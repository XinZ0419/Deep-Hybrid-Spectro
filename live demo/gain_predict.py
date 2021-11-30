import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
import torch.nn as nn


class Classifier(Dataset):
    def __init__(self, temp_array):
        self.data_raw = temp_array[:, :3000]
        self.data_fft = temp_array[:, 3000:]

    def __len__(self):
        return len(self.data_raw)

    def __getitem__(self, idx):
        return self.data_raw[idx, :], self.data_fft[idx, :]


class LinearModel_1(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.raw = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),

            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU())

        self.fft = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout(0.3),

            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.3))

        self.visu = nn.Sequential(
            nn.Linear(2 * 256, 128),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 32))

        self.out = nn.Sequential(
            nn.Linear(32, num_classes))

    def get_weights(self):
        return self.weight

    def forward(self, t_raw, t_fft):
        raw_out = self.raw(t_raw)
        fft_out = self.fft(t_fft)
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
        data_fft = torch.tensor(single_data.data_fft)
        data_raw, data_fft = data_raw.to(device), data_fft.to(device)
        # data_raw = torch.squeeze(data_raw)
        # data_fft = torch.squeeze(data_fft)

        # the model on the data
        starter.record()
        output = model(data_raw.float(), data_fft.float())
        ender.record()
        torch.cuda.synchronize()
        run_time = starter.elapsed_time(ender)

        y_pred = output.argmax()
        # print(y_pred)
        # predict = nn.functional.one_hot(y_pred, 11)
        predict = y_pred.item()

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
    model = LinearModel_1(total_num_samples, num_classes).to(device)
    state_dict = torch.load('Model1_comb_Fold_4_allclass_twoinputs_net_parameter.pth')
    model.load_state_dict(state_dict)

    predict, run_time = test(model, device, single_data)

    return predict, run_time
