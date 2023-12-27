# Role of Audio Frontends in Bird Species Recognition
This repository contains code for the paper Role of Audio Frontends in Bird Species Recognition.

## Dataset
For the bird recordings, please use the **xc.csv** metadata to download them from [Xeno-Canto](https://xeno-canto.org/). They provide an API to download the files using code. You can easily find online python codes for downloading from Xeno-Canto.
Afterward, put all the recordings in one folder, you can name it `xc_recordings` or something else. Then, open the `cli.py` file and change the default value of `--data-dir` argument  to the absolute path where your `xc_recordings` folder is. For example, `/username/home/datasets/xc_recordings`. 

For the environmental noise recordings, download the [ESC-50](https://github.com/karolpiczak/ESC-50) dataset. After unzipping the file, open the `cli.py` and change the default value of `--noise-dir` argument to the absolute path where your unzipped `ESC-50-master` is located. For example, `/username/home/datasets/ESC-50-master`.

## How to Use
It would be easier if you open the project in an IDE like PyCharm, and change the arguments in `cli.py`. Then, you only need to run the `main.py`. However, it is also possible to run it from terminal/command line like this:
```
python main.py --fe leaf --n-filters 80 --sr 32000 --fft 512 --hop 320 --augment --pcen --normalizer minmax --device cuda --epochs 100 --trnw 54 --vanw 16 --tenw 16
```
You can edit the default value of the arguments in `cli.py` and simply run:
```
python main.py
```

Th arguments are:
* `--fe`:  The frontend to use. choices are [leaf, sinc, mel, stft]. sinc is SincNet.
* `--n-filters`: The number of learnable filters in leaf or sinc.
* `--sr`: All audio files will be resampled to this sampling rate (your original files will remain intact), and frontends will operate using this sampling rate.
* `--fft`: The fft window size for Short-Time-Fourier-Transform (STFT).
* `--hop`: The hop length of STFT.
* `--augment`: Data augmentation will be used. The default is off.
* `--pcen`: PCEN will be used. The default is log-compression.
* `--normalizer`: The data normalization method. Choices are [minmax, standard, None].
* `--device`: The device to train the model on (e.g., cuda or cpu).
* `--epochs`: The number of training epochs.
* `--trnw`: The number of DataLoader workers for the training set. Set it to a number less than your machine's CPU cores. If you have many CPUs, 54 is a good choice. The sum of train, validation, and test dataloader workers do not need to be less than your CPUs, but each individual dataloader should be less.
* `--vanw`: The number of DataLoader workers for the validation set. If you have anough CPUs, 16 is a good choice.
* `--tenw`: The number of DataLoader workers for the test set. If you have anough CPUs, 16 is a good choice.

## How the code is structured
fronend contains codes for each type of the audio frontend 
