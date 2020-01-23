# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
    Utility functions for downloading, extracting, and reading the
    Multi-Genre NLI (MultiNLI) Corpus.
    https://www.nyu.edu/projects/bowman/multinli/
"""
import pandas as pd
from interpret_text.common.dataset.utils_data_shared import maybe_download

DATA_URLS = {
    "train": "https://github.com/AcademiaSinicaNLPLab/sentiment_dataset/raw/master/data/stsa.binary.train",
    "dev": "https://github.com/AcademiaSinicaNLPLab/sentiment_dataset/raw/master/data/stsa.binary.dev",
    "test": "https://github.com/AcademiaSinicaNLPLab/sentiment_dataset/raw/master/data/stsa.binary.test",
}

def load_sst2_pandas_df(file_split, local_cache_path="."):
    """Downloads and extracts dataset into a pandas dataframe
    Args:
        file_split (str): The subset to load.
            One of: {"train", "dev", "split"}
        local_cache_path (str, optional): Defaults to current working directory.
    Returns:
        pd.DataFrame: pandas DataFrame containing the specified
            SST2 subset.
    """
    try:
        URL = DATA_URLS[file_split]
        file_name = URL.split("/")[-1]
        file_path = maybe_download(URL, file_name, local_cache_path)
    except Exception as e:
        raise e
    
    return load_data(file_path)

def load_data(fpath):
    """Loads data from a given file into pandas
    Args:
        fpath (str): The file to load data from.
    Returns:
        pd.DataFrame: pandas DataFrame containing data from the
            specified file.
    """
    label_col = 'labels'
    text_col = 'sentences'
    df_dict = {label_col: [], text_col: []}
    with open(fpath, 'r') as f:
        label_start = 0
        sentence_start = 2
        for line in f:
            label = int(line[label_start])
            sentence = line[sentence_start:]
            df_dict[label_col].append(label)
            df_dict[text_col].append(sentence)
    return pd.DataFrame.from_dict(df_dict)