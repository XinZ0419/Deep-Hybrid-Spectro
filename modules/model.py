import torch
import torch.nn as nn


class LinearModel_raw(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.raw = nn.Sequential(
            nn.Linear(input_size, 1024), nn.BatchNorm1d(1024), nn.LeakyReLU(),
            nn.Linear(1024, 256), nn.BatchNorm1d(256), nn.LeakyReLU(),
            nn.Linear(256, num_classes))

    def forward(self, t_raw):
        raw_out = self.raw(t_raw)
        return raw_out


class LinearModel_stft(nn.Module):
    def __init__(self, num_classes, num_channels=1, drop_prob=0.3):
        super().__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.stft = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=8, kernel_size=3, padding=1), nn.LeakyReLU(),
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, padding=1), nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(82240, 1024), nn.LeakyReLU(), nn.Dropout(drop_prob),   ## for 512
            # nn.Linear(90816, 1024), nn.LeakyReLU(), nn.Dropout(drop_prob),   ## for 256
            nn.Linear(1024, 256), nn.LeakyReLU(), nn.Dropout(drop_prob),
            nn.Linear(256, num_classes))

    def forward(self, t_fft):
        fft_out = self.stft(t_fft)
        return fft_out


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
            # nn.Linear(90816, 1024), nn.LeakyReLU(), nn.Dropout(drop_prob),  ## for 256
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


class LinearModel_fft(nn.Module):
    def __init__(self, input_size, num_classes, drop_prob=0.3):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.fft = nn.Sequential(
            nn.Linear(input_size, 1024), nn.BatchNorm1d(1024), nn.LeakyReLU(), nn.Dropout(drop_prob),
            nn.Linear(1024, 256), nn.BatchNorm1d(256), nn.LeakyReLU(), nn.Dropout(drop_prob),
            nn.Linear(256, num_classes))

    def forward(self, t_fft):
        fft_out = self.fft(t_fft)
        return fft_out


class LinearModel_fused(nn.Module):
    def __init__(self, input_signal_size, input_freq_size, num_classes, drop_prob=0.3):
        super().__init__()
        self.input_signal_size = input_signal_size
        self.input_freq_size = input_freq_size
        self.num_classes = num_classes
        self.raw = nn.Sequential(
            nn.Linear(input_signal_size, 1024), nn.BatchNorm1d(1024), nn.LeakyReLU(),
            nn.Linear(1024, 256), nn.BatchNorm1d(256), nn.LeakyReLU())

        self.fft = nn.Sequential(
            nn.Linear(input_freq_size, 1024), nn.BatchNorm1d(1024), nn.LeakyReLU(), nn.Dropout(drop_prob),
            nn.Linear(1024, 256), nn.BatchNorm1d(256), nn.LeakyReLU(), nn.Dropout(drop_prob))

        self.visu = nn.Sequential(
            nn.Linear(2 * 256, 128), nn.LeakyReLU(), nn.BatchNorm1d(128), nn.Linear(128, 32))

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
