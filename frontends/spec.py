import torch
import torchaudio
from torch import nn
from frontends.pcen import PCENLayer
from frontends.normalizers import minmax_normalizer, standardizer


class TFFE(nn.Module):
    def __init__(self, sr=32000, n_fft=512, hop_length=320, f_min=150, f_max=16000, n_mels=80,
                 use_pcen=False, normalizer=None, tf_mode='mel'):
        super(TFFE, self).__init__()
        self.tf_mode = tf_mode
        if tf_mode == 'mel':
            self.tf_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=n_fft, hop_length=hop_length,
                                                                     f_min=f_min, f_max=f_max, n_mels=n_mels,
                                                                     center=True)
            self.n_filters = n_mels
        elif tf_mode == 'stft':
            self.tf_transform = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=2.0,
                                                                  center=True)
            self.n_filters = n_fft // 2
        else:
            raise ValueError(f'Time-Frequency mode {tf_mode} is not available.')
        if use_pcen:
            self.compression = PCENLayer(self.n_filters, alpha=0.96, smooth_coef=0.04, delta=2.0, floor=1e-12)
        else:
            self.compression = torchaudio.transforms.AmplitudeToDB('magnitude', top_db=80)
        if normalizer == 'minmax':
            self.normalizer = minmax_normalizer
        elif normalizer == 'standard':
            self.normalizer = standardizer
        else:
            self.normalizer = nn.Identity()

    def forward(self, x):
        """
        :param x: batch of waveforms of shape (B, 1, T)
        :type x: torch.Tensor
        :return: tensor of stft or mel with shape (B, F, T)
        :rtype: torch.Tensor
        """
        x = x.squeeze(1)
        x = self.tf_transform(x)
        if self.tf_mode == 'stft':
            x = x[:, 1:, :]  # drop the 0th bin
        x = torch.maximum(x, torch.tensor(1e-5, device=x.device))
        x = self.compression(x)
        x = self.normalizer(x)
        return x
