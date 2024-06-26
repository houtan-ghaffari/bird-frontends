# On the Role of Audio Frontends in Bird Species Recognition
This repository contains code for the paper [On the role of audio frontends in bird species recognition](https://eur03.safelinks.protection.outlook.com/?url=https%3A%2F%2Fdoi.org%2F10.1016%2Fj.ecoinf.2024.102573&data=05%7C02%7CHoutan.Ghaffari%40UGent.be%7Cdc48c4f2ab784ead468a08dc4f7ba07c%7Cd7811cdeecef496c8f91a1786241b99c%7C1%7C0%7C638472640549558453%7CUnknown%7CTWFpbGZsb3d8eyJWIjoiMC4wLjAwMDAiLCJQIjoiV2luMzIiLCJBTiI6Ik1haWwiLCJXVCI6Mn0%3D%7C0%7C%7C%7C&sdata=yjuEyf89WWj%2Fj4UUwbkRjWPtrs3MfoF7j0cPHFGOH3I%3D&reserved=0) by Houtan Ghaffari and Paul Devos.

## Summary
It has been of concern that audio features such as mel-filterbanks might be inadequate for analyzing bird vocalization since they are designed based on human psychoacoustics. A detailed ablation study of the learnable and static audio frontends showed little benefit from data-driven frequency selectivity for bird vocalization. Nonetheless, the functional form of the learnable filters impacted the performance despite the homogeneity of frequency channels across the frontends. However, adequate normalization and compression operations reduced the performance gap between the frontends. In particular, PCEN, min-max normalization, and standardization made the models resilient against unseen environmental noise and consistently made the (mel-)spectrogram comparable to modern audio frontends.

An in-depth explanation was provided for each experiment, followed by a thorough discussion of all the results to summarize the observations and practical findings. This work concludes that the (mel-)spectrogram combined with PCEN and a global normalization method is on par with learnable audio frontends that operate on the waveform. The findings may not apply to all animals and scenarios but should be valid for typical bird species recognition tasks.

Audio frontends that use waveforms might significantly increase computation time and latency. Therefore, a marginal improvement compared to using the (mel-)spectrogram should be necessary for the task to be considered a proper trade-off. Regardless, adapting the time-frequency representation to the dataset's characteristics and using physically informed filters are intriguing ideas for the discussed reasons in the paper. The topic deserves further research to build efficient learnable frontends for bioacoustics and identify the circumstances where they are appropriate.

## Dataset
Download the bird recordings (~34 GiB) from [Xeno-Canto](https://xeno-canto.org/) using the `xc.csv` metadata. Put the recordings in a directory called `xc_recordings` inside the project repository. If you relocated the `xc_recordings`, open the `cli.py` and change the default value of the `--data-dir` argument to the absolute path of your recordings folder. For example, `/username/home/datasets/xc_recordings`. Xeno-Canto allows to download by code for reasonable usage. However, as far as I understood, they don't appreciate it to spread such codes. Therefore, please search online for ways to download from Xeno-Canto or contact them for advice.

Download the [ESC-50](https://github.com/karolpiczak/ESC-50) dataset for the environmental noise recordings. Either put the `ESC-50-master` inside the project directory or open the `cli.py` and change the default value of the `--noise-dir` argument to the absolute path where your unzipped `ESC-50-master` is located. For example, `/username/home/datasets/ESC-50-master`.

## How the code is structured
The names of the files are self-explanatory, so I make it brief.
* The data loading and processing is in `datautils\datapipe.py`.
* The audio frontends are in the `frontend` folder. LEAF is implemented in `leaf.py`, SincNet in `sincnet.py`, and mel-spectrogram and spectrogram in `spec.py`. Other files implement the PCEN (`pcen.py`), normalization methods (`normalizers.py`), and Gaussian pooling (`poolings.py`).
* The full model leverages a frontend and a pretrained EfficientNet-B0. It is implemented in `model.py`. Some of the training and model configurations can be set in `cli.py` or by command line as described in the next section. The `main.py` glues everything together and runs the model. 
* The `utils.py` contains helper functions:
  * `RiseRunDecayScheduler`: learning rate scheduler.
  * `plot_leaf_filters`: creates and saves the LEAF frontend filters, like figures 5 and 7 in the paper. It runs by default when using the leaf frontend unless you modify the `main.py`.
  * `plot_sinc_filters`: Same as the one above but for SincNet.
  * `plot_cm`: creates and saves the confusion matrix.
  * `load_data`: loads the **xc.csv** and splits it into train, validation, and test sets.
  * `df_split`: helps the above function for splitting.
  * `train_step`: performs one epoch of training.
  * `validate`: evaluation using validation set dataloader.
  * `validate_long_files`: evaluation of the full-length recordings for test set. See the paper methodology section to see why there are different evaluation functions.
  * `noisy_validate`: mixes the ESC-50 dataset with bird recordings test set at multiple SNR ratios for noise robustness analysis.
 
## How to use
It would be easier if you open the project in an IDE like PyCharm, and change the arguments in `cli.py`. Then, you only need to run the `main.py`. However, it is also possible to run it from the terminal/command line like this:
```
python main.py --fe leaf --n-filters 80 --sr 32000 --fft 512 --hop 320 --augment --pcen --normalizer minmax --device cuda --epochs 100 --trnw 54 --vanw 16 --tenw 16
```
You can edit the default value of the arguments in `cli.py` and simply run this from the command line:
```
python main.py
```

The arguments are:
* `--fe`:  The frontend to use. choices are [leaf, sinc, mel, stft]. sinc is SincNet.
* `--n-filters`: The number of learnable filters in leaf or sinc.
* `--sr`: All audio files will be resampled to this sampling rate (your original files will remain intact), and frontends will operate using this sampling rate.
* `--fft`: The fft window size for Short-Time-Fourier-Transform (STFT).
* `--hop`: The hop length of STFT.
* `--augment`: Data augmentation will be used. The default is off.
* `--pcen`: PCEN will be used. The default is log compression.
* `--normalizer`: The data normalization method. Choices are [minmax, standard, None].
* `--device`: The device to train the model on (e.g., cuda or cpu).
* `--epochs`: The number of training epochs.
* `--trnw`: The number of DataLoader workers for the training set. Set it to a number less than your machine's CPU cores. If you have many CPUs, 54 is a good choice. The sum of train, validation, and test dataloader workers does not need to be less than your CPUs, but each dataloader should have less.
* `--vanw`: The number of DataLoader workers for the validation set. If you have enough CPUs, 16 is a good choice.
* `--tenw`: The number of DataLoader workers for the test set. If you have enough CPUs, 16 is a good choice.

The names for saving the models, logs, and plots are automatically and uniquely generated at each run. The state with highest macro averaged f1-score on the validation set will be saved.

## Using another dataset
If you want to use another dataset, you have to first provide the metadata as a CSV file that can be loaded into a Pandas DataFrame. It should at least contain the following information and columns:
* The indices of the DataFrame should be ordered integers without a gap.
* A column for `path`: name of each recording file (e.g., rec1.mp3) if you put all of the recordings in one directory. If they are in subfolders, i.e., each recording in a subfolder with its species name, then `path` should be the path to that recording inside the parent directory, e.g., 'turdus_merula/rec1.mp3'.
* A column for `label`: numerical training label of the recording. Each species should have a unique integer label starting from 0.
* A column for `length`: duration of each recording in seconds.
* A column for `smp`: sampling rate of each recording.
* A column for `gensp`: The gen and species of each recording label, e.g., 'erithacus_rubecula', 'turdus_merula', etc.

When you prepared this metadata, go to line 30 of `main.py` and change the `'xc.csv'` with your metadata path in `train_df, val_df, test_df = load_data('xc.csv', args.data_dir)`. Make sure the `--data-dir` argument in `cli.py` is set to your recordings parent directory.
## Libraries
The libraries that we used are:
* python: 3.10.10
* pytorch: 2.0.0
* numpy: 1.23.5
* pandas: 1.5.3
* torchaudio: 2.0.0
* torchvision: 0.15.0
* matplotlib: 3.7.1
* tqdm: 3.7.1
* scikit-learn: 1.2.2

# Citation
Kindly consider citing the following paper if our results were helpful to your work:

```
@article{GHAFFARI2024102573,
title = {On the role of audio frontends in bird species recognition},
journal = {Ecological Informatics},
volume = {81},
pages = {102573},
year = {2024},
issn = {1574-9541},
doi = {https://doi.org/10.1016/j.ecoinf.2024.102573},
url = {https://www.sciencedirect.com/science/article/pii/S1574954124001158},
author = {Houtan Ghaffari and Paul Devos}
```
