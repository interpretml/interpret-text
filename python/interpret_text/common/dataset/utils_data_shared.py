# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This script reuses code from
# Microsoft NLP recipes repo:
# https://github.com/microsoft/nlp-recipes/tree/master/utils_nlp


"""
    Generic utility functions for downloading and extracting datasets.
"""

import os
import requests
from tqdm import tqdm
import math
import logging
import zipfile

from tempfile import TemporaryDirectory
from contextlib import contextmanager

log = logging.getLogger(__name__)


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

def download_and_unzip(URL, file_name, local_cache_path="."):
    zip_name = URL.split("/")[-1]

    if not os.path.exists(os.path.join(local_cache_path, file_name)):
        if not os.path.exists(os.path.join(local_cache_path, zip_name)):
            try:
                zip_path = maybe_download(URL, zip_name, local_cache_path)
            except Exception as e:
                raise e
        extract_zip(zip_path, dest_path=".")
    return os.path.join(local_cache_path, file_name)