import os
import time
import concurrent.futures
import pandas as pd
import requests
import random
import numpy as np
from tqdm import tqdm


def thread_download(sub_df, root_dir=None):
    """
    :param sub_df: a slice of metadata that current thread will download.
    :param path: the directory for saving audio files.
    """
    time.sleep(random.uniform(1, 10))
    sub_df.reset_index(inplace=True, drop=True)
    for i in tqdm(range(len(sub_df))):
        time.sleep(1) # do not remove this
        path2save = os.path.join(root_dir, sub_df.iloc[i].path)
        link = sub_df.iloc[i].file
        if not os.path.isfile(path2save):  # do not re-download if the file exists
            try:
                response = requests.get(link, timeout=30, allow_redirects=True)
            except Exception as e:
                print(f'failed to download {link}')
                raise
            if response.status_code == 200 and response.content:
                with open(path2save, 'wb') as f:
                    f.write(response.content)
            else:
                print(f'failed to download {link}')
                raise

def download_xc(df, root_dir='xc_recordings', num_threads=8):
    """
    :param df: metadata to downloaded from xeno-canto.
    :param path: directory for saving audio files.
    :param num_threads: number of threads for downloading.
    """
    os.makedirs(root_dir, exist_ok=True)
    assert os.path.isdir(root_dir), f'can not create a directory at: {root_dir}'
    thread_share = int(np.ceil(len(df) / num_threads))
    sub_dfs = [df[thread_share * i:thread_share * (i + 1)] for i in range(num_threads)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        executor.map(thread_download, sub_dfs, [root_dir] * num_threads)


if __name__ == '__main__':
    df = pd.read_csv('xc.csv')
    download_xc(df, root_dir='xc_recordings', num_threads=8)
