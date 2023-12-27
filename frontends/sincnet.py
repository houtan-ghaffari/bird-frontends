import torch
import torchaudio
from torch import nn
import numpy as np
import math
from frontends.pcen import PCENLayer
from frontends.poolings import GaussianLowPass
from frontends.normalizers import minmax_normalizer, standardizer


class SincConv1d(nn.Module):

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, n_filters=80, kernel_size=801, sr=32000, min_freq=150., min_band_hz=50., max_freq=16000.):
        super(SincConv1d, self).__init__()
        assert kernel_size % 2 != 0, 'kernel size should be odd'
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.sr = sr
        self.max_freq = sr / 2 if max_freq is None else max_freq
        self.min_freq = min_freq
        self.min_band_hz = min_band_hz
        high_hz = self.max_freq - (min_freq + min_band_hz)
        mel = np.linspace(self.to_mel(min_freq), self.to_mel(high_hz), n_filters + 1)
        hz = self.to_hz(mel)
        self.low_hz_ = nn.Parameter(torch.tensor(hz[:-1], dtype=torch.float32).view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.tensor(np.diff(hz), dtype=torch.float32).view(-1, 1))
        self.register_buffer('window_', torch.hamming_window(kernel_size - 1)[:(kernel_size - 1) // 2])
        self.register_buffer('n_', 2 * math.pi * torch.arange(-(kernel_size - 1) // 2, 0).view(1, -1) / sr)

    def forward(self, waveforms):
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """
        low = self.min_freq  + torch.abs(self.low_hz_)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_), self.min_freq, self.max_freq)
        band = (high - low)[: ,0]
        f_times_t_low = low * self.n_  # out_channels, (kernel_size-1) / 2
        f_times_t_high = high * self.n_  # out_channels, (kernel_size-1) / 2
        # algebraicly simplified by Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET)
        band_pass_left = ((torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (self.n_ / 2)) * self.window_
        band_pass_center = 2 * band.view(-1 ,1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])
        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)
        band_pass = band_pass / (2 * band[:, None])  # normalizes to [-1, +1]
        filters = band_pass.view(self.n_filters, 1, self.kernel_size)
        f_map = nn.functional.conv1d(waveforms, filters, stride=1, padding="same", bias=None)
        return f_map

    def _get_filters(self):
        low = self.min_freq  + torch.abs(self.low_hz_)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_), self.min_freq, self.max_freq)
        band = (high - low)[: ,0]
        f_times_t_low = low * self.n_  # out_channels, (kernel_size-1) / 2
        f_times_t_high = high * self.n_  # out_channels, (kernel_size-1) / 2
        # algebraicly simplified by Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET)
        band_pass_left = ((torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (self.n_ / 2)) * self.window_
        band_pass_center = 2 * band.view(-1 ,1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])
        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)
        band_pass = band_pass / (2 * band[:, None])  # normalizes to [-1, +1]
        filters = band_pass.view(self.n_filters, 1, self.kernel_size)
        return filters


class SincNet(nn.Module):
    def __init__(self, n_filters=80, sr=32000, window_len=25., window_stride=10., min_freq=150.,
                 min_band_hz=50., max_freq=16000., use_pcen=True, normalizer=None):
        super(SincNet, self).__init__()
        window_size = int(sr * window_len // 1000 + 1)
        window_stride = int(sr * window_stride // 1000)
        self._conv = SincConv1d(n_filters=n_filters, kernel_size=window_size, sr=sr, min_freq=min_freq,
                                min_band_hz=min_band_hz, max_freq=max_freq)
        self._activation = nn.ReLU()
        self._pooling = GaussianLowPass(n_filters, kernel_size=window_size, stride=window_stride, padding="same",
                                        use_bias=True)
        if use_pcen:
            self._compression = PCENLayer(n_filters, alpha=0.96, smooth_coef=0.04, delta=2.0, floor=1e-12)
        else:
            self._compression = torchaudio.transforms.AmplitudeToDB('magnitude', top_db=80)

        if normalizer == 'minmax':
            self.normalizer = minmax_normalizer
        elif normalizer == 'standard':
            self.normalizer = standardizer
        else:
            self.normalizer = nn.Identity()

    def forward(self, x):
        x = self._conv(x)
        x = self._activation(x)
        x = self._pooling(x)
        x = torch.maximum(x, torch.tensor(1e-5, device=x.device))
        x = self._compression(x)
        x = self.normalizer(x)
        return x
