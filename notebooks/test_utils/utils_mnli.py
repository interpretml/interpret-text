# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This script reuses code from
# Microsoft NLP recipes repo:
# https://github.com/microsoft/nlp-recipes/tree/master/utils_nlp

"""
    Utility functions for downloading, extracting, and reading the
    Multi-Genre NLI (MultiNLI) Corpus.
    https://cims.nyu.edu/~sbowman/multinli/
"""
import os
import pandas as pd
from notebooks.test_utils.utils_data_shared import maybe_download, extract_zip

URL = "https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip"
DATA_FILES = {
    "train": "multinli_1.0/multinli_1.0_train.jsonl",
    "dev_matched": "multinli_1.0/multinli_1.0_dev_matched.jsonl",
    "dev_mismatched": "multinli_1.0/multinli_1.0_dev_mismatched.jsonl",
}


def download_file_and_extract(
    local_cache_path: str = ".", file_split: str = "train"
) -> None:
    """Download and extract the test_utils files

    Args:
        local_cache_path (str [optional]) -- Directory to cache files to.
            Defaults to current working directory (default: {"."})
        file_split {str} -- [description] (default: {"train"})
    """
    file_name = URL.split("/")[-1]
    maybe_download(URL, file_name, local_cache_path)

    if not os.path.exists(os.path.join(local_cache_path, DATA_FILES[file_split])):
        extract_zip(os.path.join(local_cache_path, file_name), local_cache_path)


def load_mnli_pandas_df(local_cache_path=".", file_split="train"):
    """Loads extracted test_utils into pandas
    Args:
        local_cache_path ([type], optional): [description].
            Defaults to current working directory.
        file_split (str, optional): The subset to load.
            One of: {"train", "dev_matched", "dev_mismatched"}
            Defaults to "train".
    Returns:
        pd.DataFrame: pandas DataFrame containing the specified
            MultiNLI subset.
    """
    try:
        download_file_and_extract(local_cache_path, file_split)
    except Exception as e:
        raise e
    return pd.read_json(
        os.path.join(local_cache_path, DATA_FILES[file_split]), lines=True
    )
