# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Setup file for interpret-text package."""
from setuptools import setup, find_packages

import interpret_text

CLASSIFIERS = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
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
    "ipywidgets",
    "transformers>=4.17.0",
    "scipy",
    "scikit-learn",
    "tqdm",
    "torch",
    "pytorch_pretrained_bert",
    "cached_property",
    "interpret-community",
    "shap>=0.20.0, <=0.42.1",
]

setup(
    name=interpret_text.__name__,
    version=interpret_text.__version__,
    description="Microsoft Interpret Text SDK for Python",
    long_description="",
    long_description_content_type="text/markdown",
    author="Microsoft Corp",
    author_email="ilmat@microsoft.com",
    license="MIT License",
    url="https://github.com/interpretml/interpret-text",
    classifiers=CLASSIFIERS,
    packages=find_packages(exclude=["*.tests"]),
    install_requires=DEPENDENCIES,
    include_package_data=True,
    data_files=[
        (
            "share/jupyter/nbextensions/interpret-text-widget",
            [
                "interpret_text/experimental/widget/static/extension.js",
                "interpret_text/experimental/widget/static/extension.js.map",
                "interpret_text/experimental/widget/static/index.js",
                "interpret_text/experimental/widget/static/index.js.map",
            ],
        ),
        (
            "etc/jupyter/nbconfig/notebook.d",
            ["jupyter-config/nbconfig/notebook.d/interpret-text-widget.json"],
        ),
        (
            "share/jupyter/lab/extensions",
            [
                "interpret_text/experimental/widget/js/"
                "interpret_text_widget/labextension/interpret-text-widget-0.1.7.tgz"
            ],
        ),
    ],
    zip_safe=False,
)
