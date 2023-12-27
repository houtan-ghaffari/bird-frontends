import torch
from torch import nn


class GaussianLowPass(nn.Module):
    def __init__(self, in_channels, kernel_size=801, stride=320, padding="same", use_bias=True):
        super(GaussianLowPass, self).__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = kernel_size
        self.stride = stride
        self.use_bias = use_bias
        self.in_channels = in_channels
        self.padding = kernel_size // 2 if padding == 'same' else 0
        w = torch.ones((1, 1, in_channels, 1)) * 0.4
        self.weights = nn.Parameter(w)
        if use_bias:
            self._bias = torch.nn.Parameter(torch.ones(in_channels, ))
        else:
            self._bias = None

    def forward(self, x):
        kernel = self.gaussian_lowpass(self.weights, self.kernel_size)
        kernel = kernel.reshape(-1, self.kernel_size, self.in_channels)
        kernel = kernel.permute(2, 0, 1)
        outputs = nn.functional.conv1d(x, kernel, bias=self._bias, stride=self.stride, padding=self.padding,
                                       groups=self.in_channels)
        return outputs

    @staticmethod
    def gaussian_lowpass(sigma, filter_size):
        sigma = torch.clamp(sigma, min=(2. / filter_size), max=0.5)
        t = torch.arange(0, filter_size, dtype=sigma.dtype, device=sigma.device)
        t = torch.reshape(t, (1, filter_size, 1, 1))
        numerator = t - 0.5 * (filter_size - 1)
        denominator = sigma * 0.5 * (filter_size - 1)
        return torch.exp(-0.5 * (numerator / denominator) ** 2)
