# Role of Audio Frontends in Bird Species Recognition
This repository contains code for the paper Role of Audio Frontends in Bird Species Recognition.

## Dataset
For the bird recordings, please use the _xc.csv_ metadata to download them from [Xeno-Canto](https://xeno-canto.org/). They provide an API to download the files using code. You can easily find online python codes for downloading from Xeno-Canto.
Afterward, put all the recordings in one folder, you can name it _xc_recordings_ or something else. Then, open the `cli.py` file and change the `--data-dir` argyment default value to the absolute path where your _xc_recordings_ folder is. For example, `/username/home/datasets/xc_recordings`. 

Instructions for preparing the dataset.
Use xc.csv for downloading the files from xeno-canto.
download esc-50 for testing noisy conditions.

## How to Use
* select a frontend and provide these necessary arguments. You can run the file directly in an ide and modify the cli.py file.
