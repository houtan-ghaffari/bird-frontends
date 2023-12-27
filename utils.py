import math
from sklearn import metrics
import numpy as np
import torch
import matplotlib.pyplot as plt
import random
import pandas as pd
from datautils.datapipe import WaveData
from tqdm import tqdm
import os


class RiseRunDecayScheduler(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, steps_in_epoch=None, warmup=1, constant=0, total_epochs=100, lowest_lr=1e-6):

        self.warmup = warmup * steps_in_epoch
        self.constant = self.warmup + (constant * steps_in_epoch)
        self.final_step = total_epochs * steps_in_epoch
        self.decay_interval = self.final_step - self.constant
        self.lowest_lr = lowest_lr
        super().__init__(optimizer)

    def get_lr(self):
        current_iteration = self.last_epoch
        if current_iteration <= self.warmup:
            factor = current_iteration / self.warmup
        elif current_iteration <= self.constant:
            factor = 1.0
        else:
            current_iteration = self.last_epoch - self.constant
            factor = 0.5 * (1 + math.cos(math.pi * current_iteration / self.decay_interval))

        return [lr * factor if (lr * factor) > self.lowest_lr else self.lowest_lr for lr in self.base_lrs]


def plot_leaf_filters(model, sr, path2save):
    model.to('cpu')
    trained_center = model.frontend._complex_conv._constraint(model.frontend._complex_conv.kernel_params).data[:,
                     0] / math.pi * (sr / 2)
    init_center = model.frontend._complex_conv._constraint(model.frontend._complex_conv._init_weights()).data[:,
                  0] / math.pi * (sr / 2)
    with plt.style.context('seaborn-v0_8-colorblind'):
        _ = plt.figure(figsize=(6.4, 4.), tight_layout=True)
        plt.plot(range(len(init_center)), init_center, alpha=1., linewidth=12, label='Initial frequency response',
                 c='k')
        plt.plot(range(len(trained_center)), trained_center, alpha=1., linewidth=4, label='Trained frequency response',
                 c='C5')
        plt.title("Gabor Filters Frequency Response", fontsize=14, fontweight='bold')
        plt.ylabel("Hz", fontsize=14, fontweight='bold')
        plt.grid()
        plt.xticks(np.arange(len(trained_center))[::10], np.arange(len(trained_center))[::10] + 1, fontsize=14,
                   fontweight='bold')
        plt.xlim([-1., len(trained_center)])
        plt.yticks(fontsize=14, fontweight='bold')
        plt.xlabel('Filter Index', fontsize=14, fontweight='bold')
        plt.legend(fontsize=16)
        plt.savefig(path2save, dpi=300, format='jpg', bbox_inches='tight')
        plt.close()


def plot_sinc_filters(model, init_model, sr, path2save):
    model.to('cpu')
    filters = model.frontend._conv._get_filters().squeeze().detach().data
    init_filters = init_model.frontend._conv._get_filters().squeeze().detach().data
    m_f = torch.fft.rfft(filters, n=sr).abs().numpy()
    i_f = torch.fft.rfft(init_filters, n=sr).abs().numpy()

    peaks = []
    init_peaks = []
    for i in range(m_f.shape[0]):
        h = np.where(m_f[i] > m_f[i].mean())[0][-1]
        l = np.where(m_f[i] > m_f[i].mean())[0][0]
        peaks.append((h + l) / 2)
        h = np.where(i_f[i] > i_f[i].mean())[0][-1]
        l = np.where(i_f[i] > i_f[i].mean())[0][0]
        init_peaks.append((h + l) / 2)
    with plt.style.context('seaborn-v0_8-colorblind'):
        _ = plt.figure(figsize=(6.4, 4.), tight_layout=True)
        plt.plot(range(len(init_peaks)), init_peaks, alpha=1., linewidth=12, label='Initial frequency response',
                 c='k')
        plt.plot(range(len(peaks)), peaks, alpha=1., linewidth=4, label='Trained frequency response',
                 c='C5')
        plt.title("Sinc Filters Frequency Response", fontsize=14, fontweight='bold')
        plt.ylabel("Hz", fontsize=14, fontweight='bold')
        plt.grid()

        plt.xticks(np.arange(len(peaks))[::10], np.arange(len(peaks))[::10] + 1, fontsize=14,
                   fontweight='bold')
        plt.xlim([-1., len(peaks)])
        plt.yticks(fontsize=14, fontweight='bold')
        plt.xlabel('Filter Index', fontsize=14, fontweight='bold')
        plt.legend(fontsize=16)
        plt.savefig(path2save, dpi=300, format='jpg', bbox_inches='tight')
        plt.close()


def plot_cm(y_true, y_pred, path2save):
    names = ['CC', 'ER', 'FC', 'LM', 'PM', 'PC', 'SA', 'TT', 'TM', 'TP']
    fig, ax = plt.subplots(figsize=(6.4 * 1, 6.4 * 1), tight_layout=True)
    ax.tick_params(labelsize=14)
    ax.xaxis.label.set_size(14)
    ax.yaxis.label.set_size(14)
    metrics.ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=np.arange(10),
                                                    display_labels=names, ax=ax, xticks_rotation='horizontal',
                                                    values_format=None, colorbar=False, normalize=None, cmap='Blues',
                                                    text_kw={'fontsize': 14, 'fontweight': 'bold'})
    plt.savefig(path2save, dpi=300, format='jpg', bbox_inches="tight")
    plt.close()


def df_split(df):
    np.random.seed(29)
    random.seed(29)
    train_frac, val_frac, test_frac = 0.7, 0.1, 0.2
    df_g = df.groupby('gensp')
    train_df, test_df, val_df = [], [], []
    for n, g in df_g:
        size = len(g)
        start = 0
        end = int(math.ceil(train_frac * size))
        train_df.append(g.iloc[start:end])
        start = end
        end = end + int(math.floor(val_frac * size))
        val_df.append(g.iloc[start:end])
        test_df.append(g.iloc[end:])

    train_df = pd.concat(train_df, ignore_index=True)
    val_df = pd.concat(val_df, ignore_index=True)
    test_df = pd.concat(test_df, ignore_index=True)

    return train_df, val_df, test_df


def load_data(path2df, data_dir):
    df = pd.read_csv(path2df)
    df.loc[:, 'path'] = df.apply(lambda row: os.path.join(data_dir, row.path), axis=1)
    df = df.reset_index(drop=True)
    train_df, val_df, test_df = df_split(df)
    return train_df, val_df, test_df


def train_step(model, optimizer, scheduler, data_loader, device='cuda'):
    model.train()
    y_true, y_pred, losses = [], [], []
    for x, y in data_loader:
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device).long()
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.item()))
        y_true.extend(y.cpu().tolist())
        y_pred.extend(logits.detach().cpu().argmax(dim=1).tolist())
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    f1_macro_ = metrics.f1_score(y_true, y_pred, average='macro') * 100
    f1_micro_ = metrics.f1_score(y_true, y_pred, average='micro') * 100
    loss_ = np.mean(losses)
    return f1_macro_, f1_micro_, loss_


@torch.inference_mode()
def validate(model, data_loader, device='cuda', return_labels=False, path2save=None):
    model.eval()
    y_true, y_pred, losses = [], [], []
    for x, y in data_loader:
        x, y = x.to(device), y.to(device).long()
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        losses.append(float(loss.detach().item()))
        y_true.extend(y.cpu().tolist())
        y_pred.append(logits.detach().cpu().softmax(dim=1).numpy())
    y_pred = np.concatenate(y_pred)
    y_pred_hard = y_pred.argmax(axis=1)
    y_true = np.array(y_true)
    test_f1_macro = metrics.f1_score(y_true, y_pred_hard, average='macro') * 100
    test_f1_micro = metrics.f1_score(y_true, y_pred_hard, average='micro') * 100
    test_loss = np.mean(losses)
    if path2save is not None:
        prec = metrics.precision_score(y_true, y_pred_hard, average='macro')
        recal = metrics.recall_score(y_true, y_pred_hard, average='macro')
        results = {'f1_macro': test_f1_macro, 'f1_micro': test_f1_micro, 'precision': prec, 'recall': recal}
        for k in range(1, 6):
            results[f'top{k}'] = metrics.top_k_accuracy_score(y_true, y_pred, k=k) * 100
        _ = pd.DataFrame(results, index=[0]).to_csv(path2save)
    if return_labels:
        return test_f1_macro, test_f1_micro, test_loss, y_true, y_pred
    return test_f1_macro, test_f1_micro, test_loss


@torch.inference_mode()
def validate_long_files(model, data_loader, batch_size=64, device='cuda', return_labels=False, path2save=None):
    model.eval()
    y_true, y_pred, logits, counts = [], [], [], []
    for x, y in tqdm(data_loader):
        y_true.extend(y.cpu().tolist())
        counts.extend([xi.shape[0] for xi in x])
        xc = torch.cat(x).to(device)
        for i in range(math.ceil(xc.shape[0] / batch_size)):
            xb = xc[i*batch_size:(i+1)*batch_size]
            logits.append(model(xb).cpu())
    logits = torch.cat(logits, dim=0)
    start = 0
    for i, c in enumerate(counts):
        end = start + c
        y_pred.append(logits[start:end].exp().mean(dim=0, keepdim=False).log())
        start = end
    del logits
    assert len(y_pred) == len(y_true)
    y_pred = torch.softmax(torch.stack(y_pred), dim=1).numpy()
    assert len(y_pred.shape) == 2
    y_pred_hard = y_pred.argmax(axis=1)
    assert len(y_pred_hard.shape) == 1
    y_true = np.array(y_true)
    test_f1_macro = metrics.f1_score(y_true, y_pred_hard, average='macro') * 100
    test_f1_micro = metrics.f1_score(y_true, y_pred_hard, average='micro') * 100
    test_loss = None
    if path2save is not None:
        prec = metrics.precision_score(y_true, y_pred_hard, average='macro') * 100
        recal = metrics.recall_score(y_true, y_pred_hard, average='macro') * 100
        results = {'f1_macro': test_f1_macro, 'f1_micro': test_f1_micro, 'precision': prec, 'recall': recal}
        for k in range(1, 6):
            results[f'top{k}'] = metrics.top_k_accuracy_score(y_true, y_pred, k=k) * 100
        _ = pd.DataFrame(results, index=[0]).to_csv(path2save)
    if return_labels:
        return test_f1_macro, test_f1_micro, test_loss, y_true, y_pred
    return test_f1_macro, test_f1_micro, test_loss


def noisy_validate(model, test_df, noise_classes=None, path2save=None, device='cuda', sr=32000, full_len=False,
                   noise_dir='ESC-50-master', nw=1):
    noise_df_path = os.path.join(noise_dir, 'meta', 'esc50.csv')
    noise_df = pd.read_csv(noise_df_path)
    noise_df = noise_df[noise_df.category.isin(noise_classes)]
    noise_df.loc[:, 'filename'] = noise_df.apply(lambda row: os.path.join(noise_dir, 'audio', row.filename), axis=1)
    noise_files = noise_df.filename.values.tolist()
    f1_at_snrs = []
    SNR_DB = [10, 5, -5]
    for snr_db in SNR_DB:
        noisy_test_loader = torch.utils.data.DataLoader(
            WaveData(test_df, noise_files, sample_dur=5, sr=sr, validation=True, snr_db=snr_db, add_noise=True, full_len=full_len),
            shuffle=False, batch_size=8, pin_memory=True, num_workers=nw, persistent_workers=False,
            collate_fn=WaveData.collate_fn_full if full_len else WaveData.collate_fn)
        if full_len:
            noisy_test_f1_macro, *_ = validate_long_files(model, noisy_test_loader, batch_size=16, device=device,
                                                          return_labels=False, path2save=None)
        else:
            noisy_test_f1_macro, *_ = validate(model, noisy_test_loader, device=device, return_labels=False,
                                               path2save=None)
        f1_at_snrs.append(noisy_test_f1_macro)
        del noisy_test_loader
    with open(path2save, 'w') as f:
        f.write('10,5,-5\n')
        for x in f1_at_snrs:
            f.write(str(round(x, 2)) + ',')


