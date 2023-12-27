"""
author: houtan ghaffari
email: houtan.ghaffari@ugent.be
"""
import sys
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm
import cli
from datetime import datetime
from datautils.datapipe import WaveData
from model import Net
from utils import (RiseRunDecayScheduler, plot_leaf_filters, plot_sinc_filters, plot_cm, load_data, train_step,
                   validate, noisy_validate, validate_long_files)


if __name__ == '__main__':
    parser = cli.create_parser()
    args = parser.parse_args()
    print(args)
    Path.mkdir(Path('logs/cm'), parents=True, exist_ok=True)
    Path.mkdir(Path('logs/history'), parents=True, exist_ok=True)
    Path.mkdir(Path('logs/snr'), parents=True, exist_ok=True)
    Path.mkdir(Path('logs/test'), parents=True, exist_ok=True)
    Path.mkdir(Path('logs/filters'), parents=True, exist_ok=True)
    Path.mkdir(Path(f'states/{args.fe}'), parents=True, exist_ok=True)
    device = args.device
    epochs = args.epochs
    train_df, val_df, test_df = load_data('xc.csv', args.data_dir)
    test_df = test_df[test_df.length < 300].reset_index(drop=True)

    train_loader = torch.utils.data.DataLoader(WaveData(train_df, sample_dur=2, sr=args.sr, augment=args.augment),
                                               shuffle=True, batch_size=32, num_workers=args.trnw, pin_memory=True,
                                               collate_fn=WaveData.collate_fn, persistent_workers=True)
    val_loader = torch.utils.data.DataLoader(WaveData(val_df, sample_dur=5, sr=args.sr, validation=True),
                                             shuffle=False, batch_size=32, num_workers=args.vanw, pin_memory=True,
                                             collate_fn=WaveData.collate_fn, persistent_workers=True)
    test_loader = torch.utils.data.DataLoader(WaveData(test_df, sample_dur=5, sr=args.sr, validation=True,
                                                       full_len=True, overlap_dur=1), shuffle=False, batch_size=8,
                                              num_workers=args.tenw, pin_memory=True, persistent_workers=False,
                                              collate_fn=WaveData.collate_fn_full)
    # for run in range(args.runs):
    model_name = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_")
    model_name += f'{args.fe}'
    if args.normalizer is not None:
        model_name += f'_{args.normalizer}'
    if args.pcen:
        model_name += '_pcen'
    else:
        model_name += '_log'
    if args.augment:
        model_name += f'_augment'
    history_path = f'logs/history/{model_name}_history.csv'
    state_path = f'states/{args.fe}/{model_name}.pt'
    result_path = f'logs/test/{model_name}_test.csv'
    natural_snr_path = f'logs/snr/{model_name}_natural_snr.txt'
    urban_snr_path = f'logs/snr/{model_name}_urban_snr.txt'
    cm_fig_path = f'logs/cm/{model_name}_cm.jpg'
    filter_fig_path = f'logs/filters/{model_name}_filters.jpg'

    model = Net(n_filters=args.n_filters, window_len=25., window_stride=10., sr=args.sr, n_fft=args.fft,
                hop_length=args.hop, min_freq=150, max_freq=args.sr//2, num_classes=10, frontend=args.fe,
                use_pcen=args.pcen, normalizer=args.normalizer)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = RiseRunDecayScheduler(optimizer, steps_in_epoch=len(train_loader), warmup=10, constant=20,
                                      total_epochs=epochs, lowest_lr=1e-5)

    history = {'loss': [], 'f1_macro': [], 'f1_micro': [], 'val_loss': [], 'val_f1_macro': [], 'val_f1_micro': []}
    pd.DataFrame(history).to_csv(history_path)
    torch.save(model.state_dict(), state_path)
    best_score = 0
    best_epoch = 0
    pbar = tqdm(range(epochs), colour='blue')
    for e in pbar:
        f1_macro_, f1_micro_, loss_ = train_step(model, optimizer, scheduler, train_loader, device)
        f1_macro_val, f1_micro_val, loss_val = validate(model, val_loader, device)
        for k, v in zip(['loss', 'f1_macro', 'f1_micro', 'val_loss', 'val_f1_macro', 'val_f1_micro'],
                        [loss_, f1_macro_, f1_micro_, loss_val, f1_macro_val, f1_micro_val]):
            history[k].append(v)
        if f1_macro_val > best_score:
            torch.save(model.state_dict(), state_path)
            best_score = f1_macro_val
            best_epoch = e + 1
        message = ' '.join([f'{k}:{v[e]:.4f}' for k, v in history.items()]) + f' best: {best_score} at {best_epoch}'
        pbar.set_description(message)
        pd.DataFrame(history).to_csv(history_path)

    # ****************************** Test ******************************
    model.load_state_dict(torch.load(state_path))
    model.to(device)
    model.eval()
    test_f1_macro, test_f1_micro, test_loss, test_y_true, test_y_pred = \
        validate_long_files(model, test_loader, 64, device, True, result_path)
    plot_cm(test_y_true, test_y_pred.argmax(axis=1), cm_fig_path)
    # ****************************** Noisy Test ******************************
    noisy_validate(model, test_df, ['thunderstorm', 'wind', 'rain', 'crickets'], natural_snr_path,
                   device, args.sr, True, args.noise_dir, args.tenw)
    noisy_validate(model, test_df, ['train', 'airplane', 'engine', 'helicopter'], urban_snr_path,
                   device, args.sr, True, args.noise_dir, args.tenw)
    # ****************************** save filter plots ******************************
    if args.fe == 'leaf':
        plot_leaf_filters(model, args.sr, filter_fig_path)
    if args.fe == 'sinc':
        init_model = Net(n_filters=args.n_filters, window_len=25., window_stride=10., sr=args.sr, n_fft=args.fft,
                         hop_length=args.hop, min_freq=150, max_freq=args.sr//2, num_classes=10, frontend=args.fe,
                         use_pcen=args.pcen, normalizer=args.normalizer)
        plot_sinc_filters(model, init_model, args.sr, filter_fig_path)
        del init_model

    model.cpu()
    del optimizer, scheduler, model
    torch.cuda.empty_cache()
