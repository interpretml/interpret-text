# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import pytest
import papermill as pm
import scrapbook as sb

ABS_TOL = 0.1
KERNEL_NAME = "python3"
OUTPUT_NOTEBOOK = "output.ipynb"


@pytest.mark.integration
def test_text_classification_classical_text(notebooks, tmp):
    notebook_path = notebooks["tc_classical_text"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
        parameters=dict(DATA_FOLDER=tmp),
    )
    result = sb.read_notebook(OUTPUT_NOTEBOOK).scraps.data_dict
    assert pytest.approx(result["accuracy"], 0.83, abs=ABS_TOL)
    assert pytest.approx(result["precision"], 0.83, abs=ABS_TOL)
    assert pytest.approx(result["recall"], 0.83, abs=ABS_TOL)
    assert pytest.approx(result["f1"], 0.83, abs=ABS_TOL)
