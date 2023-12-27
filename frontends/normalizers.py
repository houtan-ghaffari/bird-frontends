def minmax_normalizer(x):
    """
    :param x: mel/spectrogram of shape (B, F, T)
    :type x: torch.Tensor
    :return: normalized mel/spectrogram of shape (B, F, T)
    :rtype: torch.Tensor
    """
    shape = x.shape
    x = x.flatten(1)
    min_ = x.min(1)[0].unsqueeze(1)
    max_ = x.max(1)[0].unsqueeze(1)
    x = (x - min_) / (max_ - min_ + 1e-6)
    x = x.reshape(shape)
    return x


def standardizer(x):
    """
    :param x: mel/spectrogram of shape (B, F, T)
    :type x: torch.Tensor
    :return: standard scaled mel/spectrogram of shape (B, F, T)
    :rtype: torch.Tensor
    """
    shape = x.shape
    x = x.flatten(1)
    m = x.mean(1).unsqueeze(1)
    s = x.std(1).unsqueeze(1)
    x = (x - m) / (s + 1e-6)
    x = x.reshape(shape)
    return x
