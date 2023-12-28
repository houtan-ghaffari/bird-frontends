# On the Role of Audio Frontends in Bird Species Recognition
This repository contains code for the paper On the Role of Audio Frontends in Bird Species Recognition by Houtan Ghaffari and Paul Devos.

## Conclusion
A detailed ablation study of the learnable and static audio frontends showed little benefit from data-driven frequency selectivity for bird vocalization. Nonetheless, the functional form of the learnable filters impacted the performance despite the homogeneity of frequency channels across the frontends. However, adequate normalization and compression operations reduced the performance gap between the frontends. In particular, PCEN, min-max normalization, and standardization made the models resilient against unseen environmental noise and consistently made the (mel-)spectrogram comparable to modern audio frontends.

An in-depth explanation was provided for each experiment, followed by a thorough discussion of all the results to summarize the observations and practical findings. This work concludes that the (mel-)spectrogram combined with PCEN and a global normalization method is on par with learnable audio frontends that operate on the waveform. The findings may not apply to all animals and scenarios but should be valid for typical bird species recognition tasks.
% An in-depth explanation was provided for each experiment, followed by a thorough discussion of all the results to summarize the observations and practical findings. This work concludes that the (mel-)spectrogram combined with PCEN and a global normalization method is as suitable for bird species classification as learnable audio frontends that operate on the waveform. The findings may not apply to all animals and scenarios but should be valid for typical bird species classification tasks.

Audio frontends that use waveforms might significantly increase computation time and latency. Therefore, a marginal improvement compared to using the (mel-)spectrogram should be necessary for the task to be considered a proper trade-off. Regardless, adapting the time-frequency representation to the dataset's characteristics and using physically informed filters are intriguing ideas for the discussed reasons. The topic deserves further research to build efficient learnable frontends for bioacoustics and identify the circumstances where they are appropriate.

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
The name of the files are self-explanatory, so I make it brief. The audio frontends are in the `fronend` folder. LEAF is implemented in `leaf.py`, SincNet in 'sincnet.py', and mel-spectrogram and spectrogram in `spec.py`. Other files implement the PCEN (`pcen.py`), normalization methods (`normalizers.py`), and Gaussian pooling (`poolings.py`).

The full model leverages a frontend and a pretrained EfficientNet-B0. It is implemented in 'model.py'. The training and model configurations can be set in `cli.py` or by command line as described above. The `main.py` glues everything together and runs the model.

The `utils.py` contains helper functions:
* `RiseRunDecayScheduler`: learning rate scheduler. You can have a linear warmup (Rise), a constant period (Run), and then a cosie decay curve (Decay). Example usage with a linear warmup of 10 epochs to reach the optimizer default/set learning rate, a constant period of 20 epochs with the default learning rate, and then decaying to 0 in 100 epochs with the possibility of maintaining a minimum learning rate set by user:`scheduler = RiseRunDecayScheduler(optimizer, steps_in_epoch=len(train_loader), warmup=10, constant=20, total_epochs=100, lowest_lr=1e-5)`.
* `plot_leaf_filters`: creates and saves the LEAF frontend filters, like figures 5 and 7 in the paper. It runs by default when using leaf frontend, unless you modify the `main.py`.
* `plot_sinc_filters`: Same as the one above but for SincNet.
* `plot_cm`: creates and saves confusion matrix.
* `load_data`: loads the **xc.csv** and splits it into train, validation, and test sets.
* `df_split`: helps the above function for splitting.
* `train_step`: performs one epoch of training.
* `validate`: evaluation using validation set dataloader.
* `validate_long_files`: evaluation of the full-length recording for test set dataloader. See the paper methodology section to see why there are different evaluation functions.
* `noisy_validate`: mixes the ESC-50 dataset with bird recordings test set at multiple SNR ratios for noise robustness analysis.

# Citation
Consider citing the following paper if you used our results in your work:
`to be filled upon publication.`
