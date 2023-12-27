import argparse


def create_parser():
    parser = argparse.ArgumentParser(prog='audio frontend',
                                     fromfile_prefix_chars='@',
                                     description='AudioFrontends for Bioacoustics.',
                                     )
    parser.add_argument('--sr', default=32000, type=int)
    parser.add_argument('--fft', default=512, type=int)
    parser.add_argument('--hop', default=320, type=int)
    parser.add_argument('--n-filters', default=80, type=int)
    parser.add_argument('--fe', type=str, choices=['leaf', 'sinc', 'mel', 'stft'],
                        default='mel', help='select the frontend to use.')
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--trnw', default=8, type=int,
                        help='number of workers for train set DataLoader (default: %(default)i).')
    parser.add_argument('--tenw', default=8, type=int,
                        help='number of workers for test set DataLoader (default: %(default)i).')
    parser.add_argument('--vanw', default=8, type=int,
                        help='number of workers for validation set DataLoader (default: %(default)i).')
    parser.add_argument('--augment', type=bool, default=True)
    parser.add_argument('--pcen', type=bool, default=True)
    parser.add_argument('--normalizer', default='minmax', choices=['minmax', 'standard', None])
    parser.add_argument('--noise-dir', type=str, default='ESC-50-master',
                        help='path to ESC-50 recordings.')
    parser.add_argument('--data-dir', type=str, default='xc_recordings',
                        help='path to Xeno-Canto recordings directory.')
    return parser
