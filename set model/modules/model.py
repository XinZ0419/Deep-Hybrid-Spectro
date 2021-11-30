import torch
import torch.nn as nn


class LinearModel_fft(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fft = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout(0.3),

            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes))

    def forward(self, t_fft):
        fft_out = self.fft(t_fft)
        return fft_out


class LinearModel_raw(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.raw = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),

            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, num_classes))

    def forward(self, t_raw):
        raw_out = self.raw(t_raw)
        return raw_out


class LinearModel_fused(nn.Module):
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
