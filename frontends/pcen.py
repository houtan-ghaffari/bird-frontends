import torch
from torch import nn


class ExponentialMovingAverage(nn.Module):
    def __init__(self, in_channels, coeff_init):
        super(ExponentialMovingAverage, self).__init__()
        self.weights = nn.Parameter(torch.ones(in_channels) * coeff_init)

    def forward(self, x):
        w = torch.clamp(self.weights, min=0.02, max=1.)
        initial_state = x[:, :, 0]  # x is B, C, T
        return self.scan(initial_state, x, w)

    def scan(self, init_state, x, w):
        x = x.permute(2, 0, 1)  # T, B, C
        acc = init_state
        results = []
        for ix in range(len(x)):  # for each t in T
            acc = (w * x[ix]) + ((1.0 - w) * acc)
            results.append(acc.unsqueeze(0))  # each one is 1, B, C
        results = torch.cat(results, dim=0)
        results = results.permute(1, 2, 0)  # B, C, T
        return results


class PCENLayer(nn.Module):
    def __init__(self, in_channels, alpha=0.96, smooth_coef=0.04, delta=2.0, root=1., floor=1e-6):
        super(PCENLayer, self).__init__()
        self.floor = floor
        self.alpha = nn.Parameter(torch.ones(in_channels) * alpha)
        self.delta = nn.Parameter(torch.ones(in_channels) * delta)
        self.root = nn.Parameter(torch.ones(in_channels) * root)
        self.ema = ExponentialMovingAverage(in_channels, coeff_init=smooth_coef)

    def forward(self, x):
        alpha = torch.min(self.alpha, torch.tensor(1.0, dtype=x.dtype, device=x.device))
        root = torch.max(self.root, torch.tensor(1.0, dtype=x.dtype, device=x.device))
        ema_smoother = self.ema(x)
        one_over_root = 1. / root
        output = (((x / ((self.floor + ema_smoother) ** alpha.view(1, -1, 1))) + self.delta.view(1, -1, 1)) ** one_over_root.view(1, -1, 1)) - (self.delta.view(1, -1, 1) ** one_over_root.view(1, -1, 1))
        return output
