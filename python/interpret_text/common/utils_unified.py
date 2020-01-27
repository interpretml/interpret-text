# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This script reuses some code from
# Microsoft NLP recipes repo:
# https://github.com/microsoft/nlp-recipes/tree/master/utils_nlp

"""
    Utility functions for downloading, extracting, and reading the
    Multi-Genre NLI (MultiNLI) Corpus.
    https://www.nyu.edu/projects/bowman/multinli/
"""

import os

import pandas as pd
import logging

from tempfile import TemporaryDirectory

import torch
from pytorch_pretrained_bert import BertTokenizer
from interpret_text.common.utils_bert import Language, Tokenizer

import math
import zipfile
from contextlib import contextmanager

import requests
from tqdm import tqdm

log = logging.getLogger(__name__)


URL = "http://www.nyu.edu/projects/bowman/multinli/multinli_1.0.zip"
DATA_FILES = {
    "train": "multinli_1.0/multinli_1.0_train.jsonl",
    "dev_matched": "multinli_1.0/multinli_1.0_dev_matched.jsonl",
    "dev_mismatched": "multinli_1.0/multinli_1.0_dev_mismatched.jsonl",
}


def maybe_download(url, filename=None, work_directory=".", expected_bytes=None):
    """Download a file if it is not already downloaded.

    Args:
        filename (str): File name.
        work_directory (str): Working directory.
        url (str): URL of the file to download.
        expected_bytes (int): Expected file size in bytes.
    Returns:
        str: File path of the file downloaded.
    """
    if filename is None:
        filename = url.split("/")[-1]
    os.makedirs(work_directory, exist_ok=True)
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        if not os.path.isdir(work_directory):
            os.makedirs(work_directory)
        r = requests.get(url, stream=True)
        total_size = int(r.headers.get("content-length", 0))
        block_size = 1024
        num_iterables = math.ceil(total_size / block_size)

        with open(filepath, "wb") as file:
            for data in tqdm(
                r.iter_content(block_size),
                total=num_iterables,
                unit="KB",
                unit_scale=True,
            ):
                file.write(data)
    else:
        log.debug("File {} already downloaded".format(filepath))
    if expected_bytes is not None:
        statinfo = os.stat(filepath)
        if statinfo.st_size != expected_bytes:
            os.remove(filepath)
            raise IOError("Failed to verify {}".format(filepath))

    return filepath


def extract_zip(file_path, dest_path="."):
    """Extracts all contents of a zip archive file.
    Args:
        file_path (str): Path of file to extract.
        dest_path (str, optional): Destination directory. Defaults to ".".
    """
    if not os.path.exists(file_path):
        raise IOError("File doesn't exist")
    if not os.path.exists(dest_path):
        raise IOError("Destination directory doesn't exist")
    with zipfile.ZipFile(file_path) as z:
        z.extractall(dest_path, filter(lambda f: not f.endswith("\r"), z.namelist()))


def download_file_and_extract(
    local_cache_path: str = ".", file_split: str = "train"
) -> None:
    """Download and extract the dataset files

    Args:
        local_cache_path (str [optional]) -- Directory to cache files to. Defaults to current working
        directory (default: {"."})
        file_split {str} -- [description] (default: {"train"})

    Returns:
        None -- Nothing is returned
    """
    file_name = URL.split("/")[-1]
    maybe_download(URL, file_name, local_cache_path)

    if not os.path.exists(os.path.join(local_cache_path, DATA_FILES[file_split])):
        extract_zip(os.path.join(local_cache_path, file_name), local_cache_path)


def load_pandas_df(local_cache_path=".", file_split="train"):
    """Loads extracted dataset into pandas
    Args:
        local_cache_path ([type], optional): [description]. Defaults to current working directory.
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


@contextmanager
def download_path(path):
    tmp_dir = TemporaryDirectory()
    if path is None:
        path = tmp_dir.name
    else:
        path = os.path.realpath(path)

    try:
        yield path
    finally:
        tmp_dir.cleanup()


def get_single_embedding(model, text, device):
    """Get the bert embedding for a single sentence
    :param text: The current sentence
    :type text: str
    :param device: A pytorch device
    :type device: torch.device
    :param model: a pytorch model
    :type model: torch.nn
    :return: A bert embedding of the single sentence
    :rtype: torch.embedding
    """
    tokenizer = BertTokenizer.from_pretrained(Language.ENGLISH)
    words = ["[CLS]"] + tokenizer.tokenize(text) + ["[SEP]"]
    tokenized_ids = tokenizer.convert_tokens_to_ids(words)
    token_tensor = torch.tensor([tokenized_ids], device=device)
    embedding = model.bert.embeddings(token_tensor)[0]
    return embedding, words


def make_bert_embeddings(
    train_data,
    model,
    device,
    LANGUAGE=Language.ENGLISH,
    TO_LOWER=True,
    BERT_CACHE_DIR="./temp",
    MAX_LEN=150,
):
    """Get the bert embedding for multiple sentences
    :param train_data: A list of sentences
    :type df: list
    :param device: A pytorch device
    :type device: torch.device
    :param model: a pytorch model
    :type model: torch.nn
    :param LANGUAGE: The pretrained model's language. Defaults to Language.ENGLISH
    :type LANGUAUGE: str
    :param BERT_CACHE_DIR: Location of BERT's cache directory. Defaults to "'/temp".
    :type BERT_CACHE_DIR: str
    :return: A bert embedding of all the sentences
    :rtype: torch.embedding
    """
    tokenizer = Tokenizer(LANGUAGE, to_lower=TO_LOWER, cache_dir=BERT_CACHE_DIR)
    tokens = tokenizer.tokenize(train_data)
    tokens, mask, _ = tokenizer.preprocess_classification_tokens(tokens, MAX_LEN)
    tokens_tensor = torch.tensor(tokens, device=device)
    mask_tensor = torch.tensor(mask, device=device)
    embeddings = model.bert.embeddings(tokens_tensor, mask_tensor)
    return embeddings
