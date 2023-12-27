import numpy as np
import torch
from torch import nn
from frontends.pcen import PCENLayer
from frontends.poolings import GaussianLowPass
import math
import torchaudio
from frontends.normalizers import minmax_normalizer, standardizer


class SquaredModulus(nn.Module):
    """this helps to simulate complex convolution using 2*num_filters and summing their squared values pair-wise instead
     of using complex convolution and modulus directly.
     """
    def __init__(self):
        super(SquaredModulus, self).__init__()
        self._pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = x.transpose(1, 2)  # B, 2C, T -> B, T, 2C
        output = 2 * self._pool(x ** 2.)  # multiply by two to remove the effect of averaging
        output = output.transpose(1, 2)  # B, C, T
        return output


class GaborConv1d(nn.Module):
    def __init__(self, n_filters=None, kernel_size=801, stride=1, padding='same', n_fft=512, min_freq=0, max_freq=16000,
                 sample_rate=32000):
        super(GaborConv1d, self).__init__()
        self.n_filters = n_filters // 2  # half of them will come from imaginary values in complex gabor filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.register_buffer('mu_lower', torch.tensor(0.))
        self.register_buffer('mu_upper', torch.tensor(math.pi))
        self.register_buffer('sigma_lower', 4 * torch.sqrt(2. * torch.log(torch.tensor(2.))) / math.pi)
        self.register_buffer('sigma_upper', self.kernel_size * torch.sqrt(2. * torch.log(torch.tensor(2.))) / math.pi)
        self.n_fft = n_fft
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.sample_rate = sample_rate
        self.kernel_params = nn.Parameter(self._init_weights())

    def forward(self, x):
        kernel_params = self._constraint(self.kernel_params)
        filters = self.gabor_filters(kernel_params, self.kernel_size)
        filters = torch.view_as_real(filters)
        filters = torch.stack([filters[:, :, 0], filters[:, :, 1]], dim=1).reshape(2 * self.n_filters, 1,
                                                                                   self.kernel_size)
        fmap = nn.functional.conv1d(x, filters, stride=self.stride, padding=self.padding)
        return fmap

    def _constraint(self, filters_params):
        idx = torch.argsort(filters_params[:, 0])
        filters_params = filters_params[idx]
        clipped_mu = torch.clamp(filters_params[:, 0], self.mu_lower, self.mu_upper)
        clipped_sigma = torch.clamp(filters_params[:, 1], self.sigma_lower, self.sigma_upper)
        return torch.vstack([clipped_mu, clipped_sigma]).t()

    def _init_weights(self):
        coeff = torch.sqrt(2. * torch.log(torch.tensor(2.))) * self.n_fft
        mel_filters = torchaudio.functional.melscale_fbanks(n_freqs=self.n_fft // 2 + 1, f_min=self.min_freq,
                                                            f_max=self.max_freq, n_mels=self.n_filters,
                                                            sample_rate=self.sample_rate).transpose(1, 0)
        sqrt_filters = torch.sqrt(mel_filters)
        center_frequencies = torch.argmax(sqrt_filters, dim=1)
        peaks, _ = torch.max(sqrt_filters, dim=1, keepdim=True)
        half_magnitudes = peaks / 2.
        fwhms = torch.sum((sqrt_filters >= half_magnitudes).float(), dim=1)
        output = torch.cat(
            [(center_frequencies * 2 * np.pi / self.n_fft).unsqueeze(1), (coeff / (np.pi * fwhms)).unsqueeze(1)],
            dim=-1)
        return output

    def gabor_impulse_response(self, t, center, fwhm):
        denominator = 1. / (torch.sqrt(torch.tensor(2.0) * math.pi) * fwhm)
        gaussian = torch.exp(torch.tensordot(1.0 / (2. * fwhm.unsqueeze(1) ** 2), (-t ** 2.).unsqueeze(0), dims=1))
        center_frequency_complex = center.type(torch.complex64)
        t_complex = t.type(torch.complex64)
        sinusoid = torch.exp(
            torch.complex(torch.tensor(0.), torch.tensor(1.)) * torch.tensordot(center_frequency_complex.unsqueeze(1),
                                                                                t_complex.unsqueeze(0), dims=1))
        denominator = denominator.type(torch.complex64).unsqueeze(1)
        gaussian = gaussian.type(torch.complex64)
        return denominator * sinusoid * gaussian

    def gabor_filters(self, kernel_params, size=801):
        t = torch.arange(-(size // 2), (size + 1) // 2, dtype=kernel_params.dtype, device=kernel_params.device)
        return self.gabor_impulse_response(t, center=kernel_params[:, 0], fwhm=kernel_params[:, 1])


class Leaf(nn.Module):
    def __init__(self, n_filters=80, sr=32000, window_len=25., window_stride=10., min_freq=150,
                 max_freq=16000, use_pcen=True, normalizer=None):
        super(Leaf, self).__init__()
        window_size = int(sr * window_len // 1000 + 1)
        window_stride = int(sr * window_stride // 1000)
        self._complex_conv = GaborConv1d(n_filters=2 * n_filters, kernel_size=window_size, stride=1, padding="same",
                                         n_fft=512, min_freq=min_freq, max_freq=max_freq, sample_rate=sr)
        self._activation = SquaredModulus()
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
        x = self._complex_conv(x)
        x = self._activation(x)
        x = self._pooling(x)
        x = torch.maximum(x, torch.tensor(1e-5, device=x.device))
        x = self._compression(x)
        x = self.normalizer(x)
        return x
