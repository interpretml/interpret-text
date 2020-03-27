import os
import sys
import pytest
from tempfile import TemporaryDirectory

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def path_notebooks():
    """Returns the path of the notebooks folder"""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, "notebooks"))

@pytest.fixture(scope="module")
def notebooks():
    folder_notebooks = path_notebooks()

    # Path for the notebooks
    paths = {
        "tc_unified_information": os.path.join(folder_notebooks, "text_classification", "text_classification_unified_information_explainer.ipynb"),
        "tc_classical_text": os.path.join(folder_notebooks, "text_classification", "text_classification_classical_text_explainer.ipynb"),
        "tc_introspective_rationale": os.path.join(folder_notebooks, "text_classification", "text_classification_introspective_rationale_explainer.ipynb")
    }
    return paths

@pytest.fixture
def tmp(tmp_path_factory):
    td = TemporaryDirectory(dir=tmp_path_factory.getbasetemp())
    try:
        yield td.name
    finally:
        td.cleanup()