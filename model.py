import torch
from torch import nn
import torchvision
from frontends.leaf import Leaf
from frontends.sincnet import SincNet
from frontends.spec import TFFE


class Net(nn.Module):

    def __init__(self, n_filters=80, window_len=25., window_stride=10., sr=32000, n_fft=512, hop_length=320,
                 min_freq=150, max_freq=16000, num_classes=10, frontend='mel', use_pcen=False, normalizer=None):
        super(Net, self).__init__()
        if frontend == 'leaf':
            self.frontend = Leaf(n_filters=n_filters, sr=sr, window_len=window_len, window_stride=window_stride,
                                 min_freq=150, max_freq=sr/2, use_pcen=use_pcen, normalizer=normalizer)
        elif frontend == 'sinc':
            self.frontend = SincNet(n_filters=n_filters, sr=sr, window_len=window_len, window_stride=window_stride,
                                    min_freq=150, max_freq=sr/2, use_pcen=use_pcen, normalizer=normalizer)
        elif frontend in ['mel', 'stft']:
            # hop_length = int(sr * window_stride // 1000)
            # n_fft = sr / 1000 * window_len
            self.frontend = TFFE(sr=sr, n_fft=n_fft, hop_length=hop_length, f_min=min_freq, f_max=max_freq,
                                 n_mels=n_filters, use_pcen=use_pcen, normalizer=normalizer, tf_mode=frontend)
        else:
            raise ValueError(f'frontend {frontend} is not available.')
        self.backend = torchvision.models.efficientnet_b0(
            weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        w = self.backend.features[0][0].weight.data.clone().mean(dim=1, keepdim=True)
        self.backend.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2),
                                                padding=(1, 1), bias=False)
        self.backend.features[0][0].weight.data = w.clone()
        self.backend.classifier[1] = nn.Identity()
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        """
        :param x: batch of waveforms of shape (B, 1, T)
        :type x: torch.Tensor
        :return: batch of logits for predictions of shape (B, num_classes)
        :rtype: torch.Tensor
        """
        x = self.frontend(x)  # B, F, T
        x.unsqueeze_(1)  # B, 1, F, T
        x = self.backend(x)  # B, 1280
        logits = self.fc(x)  # B, num_classes
        return logits

    # @torch.inference_mode()
    # def _embed(self, x):
    #     front_z = self.frontend(x)
    #     front_z.unsqueeze_(1)
    #     backend_z = self.backend(front_z)
    #     return front_z, backend_z
    