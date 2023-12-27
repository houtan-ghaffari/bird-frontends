# Role of Audio Frontends in Bird Species Recognition
This repository contains code for the paper Role of Audio Frontends in Bird Species Recognition.

## Dataset
For the bird recordings, please use the **xc.csv** metadata to download them from [Xeno-Canto](https://xeno-canto.org/). They provide an API to download the files using code. You can easily find online python codes for downloading from Xeno-Canto.
Afterward, put all the recordings in one folder, you can name it `xc_recordings` or something else. Then, open the `cli.py` file and change the default value of `--data-dir` argument  to the absolute path where your `xc_recordings` folder is. For example, `/username/home/datasets/xc_recordings`. 

For the environmental noise recordings, download the [ESC-50](https://github.com/karolpiczak/ESC-50) dataset. After unzipping the file, open the `cli.py` and change the default value of `--noise-dir` argument to the absolute path where your unzipped `ESC-50-master` is located. For example, `/username/home/datasets/ESC-50-master`.

## How to Use
* select a frontend and provide these necessary arguments. You can run the file directly in an ide and modify the cli.py file.
