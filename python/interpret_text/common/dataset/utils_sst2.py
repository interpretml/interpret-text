# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
    Utility functions for downloading, extracting, and reading the
    Multi-Genre NLI (MultiNLI) Corpus.
    https://www.nyu.edu/projects/bowman/multinli/
"""
import pandas as pd
from interpret_text.common.dataset.utils_data_shared import maybe_download

host = "https://github.com/AcademiaSinicaNLPLab/"\
        "sentiment_dataset/raw/master/data/"
DATA_URLS = {
    "train": host + "stsa.binary.train",
    "dev": host + "stsa.binary.dev",
    "test": host + "stsa.binary.test",
}


def load_sst2_pandas_df(file_split, local_cache_path="."):
    """Downloads and extracts dataset into a pandas dataframe
    :param file_split: The subset to load.
        One of: {"train", "dev", "split"}
    :type X_tokens: string
    :param local_cache_path: path to folder to store downloaded data files.
        Defaults to current working directory.
    :type string, optional
    :return: pd.DataFrame containing the specified
        SST2 subset.
    :rtype: pandas DataFrame
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
    :param fpath: Path to the file to load data from.
    :type fpath: string
    :return: pd.DataFrame containing data from the
            specified file.
    :rtype: pandas DataFrame
    """
    label_col = "labels"
    text_col = "sentences"
    df_dict = {label_col: [], text_col: []}
    with open(fpath, "r") as f:
        label_start = 0
        sentence_start = 2
        for line in f:
            label = int(line[label_start])
            sentence = line[sentence_start:]
            df_dict[label_col].append(label)
            df_dict[text_col].append(sentence)
    return pd.DataFrame.from_dict(df_dict)
