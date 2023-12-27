import torch
import random
import math
import torchaudio


class WaveData(torch.utils.data.Dataset):

    def __init__(self, df, noise_files=None, sample_dur=2, sr=32000, validation=False, augment=False, snr_db=0,
                 add_noise=False, full_len=False, overlap_dur=1):
        super().__init__()
        self.sample_dur = sample_dur
        self.sr = sr
        self.df = df
        self.validation = validation
        self.augment = False if validation else augment
        self.add_noise = False if not validation else add_noise
        self.noise_files = noise_files
        self.snr = 10. ** (snr_db / 10)
        self.full_len = False if not validation else full_len
        self.shift = (sample_dur - overlap_dur) * sr
        self.chunk_size = sample_dur * sr

    def __getitem__(self, idx):
        x, y = self.load_one(idx)
        if self.add_noise:
            x = self.add_background_noise(x)
        if self.augment:
            x = self.wave_augment(x)
        if self.full_len:
            x = x.unsqueeze(1)  # N, 1, T
        else:
            x = x.permute(1, 0)  # T, 1
        return x, y

    def __len__(self):
        return len(self.df)

    def load_one(self, idx):
        filename, y, dur, sr = self.df.iloc[idx][['path', 'label', 'length', 'smp']]
        if (
                dur < self.sample_dur) or self.validation:
            offset = 0
        else:
            offset = int(random.randint(0, max(0, int(math.fabs(dur - self.sample_dur)) - 1)) * sr)
        num_frames = -1 if self.full_len else int(self.sample_dur * sr)
        try:
            x, sr_ = torchaudio.load(filepath=filename, frame_offset=offset, num_frames=num_frames)
        except Exception as e:
            print(filename, y, dur, sr, offset)
            raise e
        x = x.mean(dim=0, keepdim=True)
        x = torchaudio.functional.resample(x, sr, self.sr, lowpass_filter_width=64, rolloff=0.9475937167399596,
                                           resampling_method="sinc_interp_kaiser", beta=14.769656459379492)
        if self.full_len:
            x = [x[0, i * self.shift:i * self.shift + self.chunk_size] for i in
                 range(math.ceil(x.shape[1] / self.shift))]
            last_len = x[-1].shape[0]
            if (last_len < (self.chunk_size - self.shift)) and (len(x) > 1):
                x = x[:-1]
            last_len = x[-1].shape[0]
            pad = self.chunk_size - last_len
            if pad > 0:
                x[-1] = torch.cat([x[-1], torch.zeros((pad,))], dim=0)
            x = torch.stack(x)
        return x, torch.tensor(y, dtype=torch.float)

    def add_background_noise(self, signal):
        random_noise_file = random.choice(self.noise_files)
        noise, sr_ = torchaudio.load(filepath=random_noise_file)
        noise = noise.mean(dim=0, keepdim=True)  # mono
        noise = torchaudio.functional.resample(noise, sr_, self.sr, lowpass_filter_width=64, rolloff=0.9475937167399596,
                                               resampling_method="sinc_interp_kaiser", beta=14.769656459379492)
        signal_length = signal.shape[-1]
        noise_length = noise.shape[-1]
        if noise_length > signal_length:
            offset = random.randint(0, noise_length - signal_length)
            noise = noise[..., offset:offset + signal_length]
        elif noise_length < signal_length:
            noise = torch.cat([noise, torch.zeros((1, signal_length - noise_length))], dim=-1)
        signal_power = (signal ** 2).sum(dim=1, keepdim=True)  # N, 1 if full_length else 1, 1
        noise_power = (noise ** 2).sum(dim=1, keepdim=True)  # 1, 1
        scale = (signal_power / (self.snr * noise_power)) ** 0.5
        noisy_signal = (signal + scale * noise) / 2
        return noisy_signal

    def random_speed_change(self, signal):
        speed = random.choice([0.9, 1., 1.1])
        sox_effects = [["speed", str(speed)], ["rate", str(self.sr)]]
        modified_signal, _ = torchaudio.sox_effects.apply_effects_tensor(signal, self.sr, sox_effects)
        return modified_signal

    def random_pitch_change(self, signal):
        pitch = str(random.randint(-100, 100))
        sox_effects = [['pitch', pitch]]
        modified_signal, _ = torchaudio.sox_effects.apply_effects_tensor(signal, self.sr, sox_effects)
        return modified_signal

    def random_gain(self, signal):
        gain = random.uniform(0.8, 1.2)
        modified_signal = gain * signal
        return modified_signal

    def wave_augment(self, waveform):
        x = self.random_speed_change(waveform)
        if random.random() > 0.2:
            x = self.random_pitch_change(x)
        x = self.random_gain(x)
        x = x + (x.mean() + torch.randn(x.shape) * x.std() * random.uniform(0, 0.2))
        return x

    @staticmethod
    def collate_fn(batch):
        x, y = zip(*batch)
        x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=0.)
        x = x.permute(0, 2, 1)  # (B, T, 1) -> (B, 1, T)
        y = torch.stack(y)
        return x, y

    @staticmethod
    def collate_fn_full(batch):
        x, y = zip(*batch)
        y = torch.stack(y)
        return x, y
