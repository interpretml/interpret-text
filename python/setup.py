# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Setup file for interpret-community package."""
from setuptools import setup, find_packages
import os

_major = "0.1"
_minor = "0.8"

if os.path.exists("../major.version"):
    with open("../major.version", "rt") as bf:
        _major = str(bf.read()).strip()

if os.path.exists("../minor.version"):
    with open("../minor.version", "rt") as bf:
        _minor = str(bf.read()).strip()

VERSION = "{}.{}".format(_major, _minor)
SELFVERSION = VERSION
if os.path.exists("patch.version"):
    with open("patch.version", "rt") as bf:
        _patch = str(bf.read()).strip()
        SELFVERSION = "{}.{}".format(VERSION, _patch)


CLASSIFIERS = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.5",
    "Programming Language :: Python :: 3.6",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: MacOS",
    "Operating System :: POSIX :: Linux",
]

DEPENDENCIES = [
    "numpy",
    "pandas",
    "pydantic",
    "spacy",
    "ipywidets",
    "transformers",
    "scipy",
    "scikit-learn",
    "tqdm",
    "torch",
    "interpret-community",
    "shap>=0.20.0, <=0.29.3",
]

EXTRAS = {"sample": ["hdbscan"], "deep": ["tensorflow"], "mimic": ["lightgbm"]}

setup(
    name="interpret-text",
    version=SELFVERSION,
    description="Microsoft Interpret Text SDK for Python",
    long_description="",  # README,
    long_description_content_type="text/markdown",
    author="Microsoft Corp",
    author_email="ilmat@microsoft.com",
    license="MIT License",
    url="https://docs.microsoft.com/en-us/azure/machine-learning/service/",
    classifiers=CLASSIFIERS,
    packages=find_packages(exclude=["*.tests"]),
    install_requires=DEPENDENCIES,
    extras_require=EXTRAS,
    include_package_data=True,
    data_files=[
        (
            "share/jupyter/nbextensions/interpret-text-widget",
            [
                "interpret_text/widget/static/extension.js",
                "interpret_text/widget/static/extension.js.map",
                "interpret_text/widget/static/index.js",
                "interpret_text/widget/static/index.js.map",
            ],
        ),
        (
            "etc/jupyter/nbconfig/notebook.d",
            ["jupyter-config/nbconfig/notebook.d/interpret-text-widget.json"],
        ),
        (
            "share/jupyter/lab/extensions",
            [
                "interpret_text/widget/js/"
                "interpret_text_widget/labextension/interpret-text-widget-0.1.7.tgz"
            ],
        ),
    ],
    zip_safe=False,
)
