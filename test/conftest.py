import os
import pytest
from tempfile import TemporaryDirectory


def path_notebooks():
    """Returns the path of the notebooks folder"""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, "notebooks"))

@pytest.fixture(scope="module")
def notebooks():
    folder_notebooks = path_notebooks()

    # Path for the notebooks
    paths = {
        "tc_mnli_bert": os.path.join(folder_notebooks, "text_classification", "text_classification_unified_information_explainer.ipynb"),
        "text_classification_mnli_bow_lr": os.path.join(folder_notebooks, "text_classification", "text_classification_classical_text_explainer.ipynb")
    }
    return paths

@pytest.fixture
def tmp(tmp_path_factory):
    td = TemporaryDirectory(dir=tmp_path_factory.getbasetemp())
    try:
        yield td.name
    finally:
        td.cleanup()