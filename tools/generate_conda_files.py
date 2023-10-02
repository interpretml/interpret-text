#!/usr/bin/python

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This script creates yaml files to build conda environments
# For generating a conda file for running only python code:
# $ python generate_conda_file.py


import argparse
import textwrap
from sys import platform


HELP_MSG = """
To create the conda environment:
$ conda env create -f {conda_env}.yaml
To update the conda environment:
$ conda env update -f {conda_env}.yaml
To register the conda environment in Jupyter:
$ conda activate {conda_env}
$ python -m ipykernel install --user --name {conda_env} \
--display-name "Python ({conda_env})"
"""

CHANNELS = ["conda-forge", "pytorch"]

CONDA_BASE = {
    "python": "python=3.8",
    "pip": "pip>=19.1.1",
    "ipykernel": "ipykernel>=4.6.1",
    "jupyter": "jupyter>=1.0.0",
    "matplotlib": "matplotlib>=2.2.2",
    "numpy": "numpy>=1.13.3",
    "pandas": "pandas>=0.24.2",
    "pytest": "pytest>=3.6.4",
    "pytorch": "pytorch>=1.0.0",
    "scipy": "scipy>=1.0.0",
    "tensorflow": "tensorflow",
    "tensorflow-estimator": "tensorflow-estimator",
    "h5py": "h5py>=2.8.0",
    "xgboost": "xgboost"
}
CONDA_GPU = {
    "numba": "numba>=0.38.1",
    "pytorch": "pytorch>=1.0.0",
    "tensorflow": "tensorflow-gpu==1.14.0",
    "cudatoolkit": "cudatoolkit==9.2",
}

CONDA_CPU = {
    "cpuonly": "cpuonly"
}

PIP_BASE = {
    "interpret-community": "interpret-community>=0.16.0",
    "cached-property": "cached-property==1.5.2",
    "papermill": "papermill>=2.3.3",
    "nteract-scrapbook": "nteract-scrapbook>=0.2.1",
    "pytorch-pretrained-bert": "pytorch-pretrained-bert>=0.6",
    "tqdm": "tqdm>=4.62.3",
    "scikit-learn": "scikit-learn>=0.19.0,<=1.3.1",
    "nltk": "nltk>=3.4",
    "pre-commit": "pre-commit>=1.20.0",
    "spacy": "spacy>=2.2.3",
    "transformers": "transformers>=4.17.0"
}
PIP_GPU = {}

PIP_DARWIN = {}
PIP_DARWIN_GPU = {}

PIP_LINUX = {}
PIP_LINUX_GPU = {}

PIP_WIN32 = {}
PIP_WIN32_GPU = {}

CONDA_DARWIN = {}
CONDA_DARWIN_GPU = {}

CONDA_LINUX = {}
CONDA_LINUX_GPU = {}

CONDA_WIN32 = {}
CONDA_WIN32_GPU = {"pytorch": "pytorch==1.0.0", "cudatoolkit": "cuda90"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(
            """
        This script generates a conda file for different environments.
        Plain python is the default,
        but flags can be used to support GPU functionality."""
        ),
        epilog=HELP_MSG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--name", help="specify name of conda environment")
    parser.add_argument(
        "--gpu", action="store_true", help="include packages for GPU support"
    )
    parser.add_argument(
        "--python-version", help="Specify Python version", required=True
    )
    args = parser.parse_args()

    # set name for environment and output yaml file
    conda_env = "interpret_cpu"
    if args.gpu:
        conda_env = "interpret_gpu"

    # Set the Python version
    CONDA_BASE["python"] = "python={0}".format(args.python_version)

    # overwrite environment name with user input
    if args.name is not None:
        conda_env = args.name

    # add conda and pip base packages
    conda_packages = CONDA_BASE
    pip_packages = PIP_BASE

    # update conda and pip packages based on flags provided
    if args.gpu:
        conda_packages.update(CONDA_GPU)
        pip_packages.update(PIP_GPU)
    else:
        conda_packages.update(CONDA_CPU)

    # update conda and pip packages based on os platform support
    if platform == "darwin":
        conda_packages.update(CONDA_DARWIN)
        pip_packages.update(PIP_DARWIN)
        if args.gpu:
            conda_packages.update(CONDA_DARWIN_GPU)
            pip_packages.update(PIP_DARWIN_GPU)
    elif platform.startswith("linux"):
        conda_packages.update(CONDA_LINUX)
        pip_packages.update(PIP_LINUX)
        if args.gpu:
            conda_packages.update(CONDA_LINUX_GPU)
            pip_packages.update(PIP_LINUX_GPU)
    elif platform == "win32":
        conda_packages.update(CONDA_WIN32)
        pip_packages.update(PIP_WIN32)
        if args.gpu:
            conda_packages.update(CONDA_WIN32_GPU)
            pip_packages.update(PIP_WIN32_GPU)
    else:
        raise Exception("Unsupported platform. Must be Windows, Linux, or macOS")

    # write out yaml file
    conda_file = "{}.yaml".format(conda_env)
    with open(conda_file, "w") as f:
        for line in HELP_MSG.format(conda_env=conda_env).split("\n"):
            f.write("# {}\n".format(line))
        f.write("name: {}\n".format(conda_env))
        f.write("channels:\n")
        for channel in CHANNELS:
            f.write("- {}\n".format(channel))
        f.write("dependencies:\n")
        for conda_package in conda_packages.values():
            f.write("- {}\n".format(conda_package))
        f.write("- pip:\n")
        for pip_package in pip_packages.values():
            f.write("  - {}\n".format(pip_package))

    print("Generated conda file: {}".format(conda_file))
    print(HELP_MSG.format(conda_env=conda_env))
