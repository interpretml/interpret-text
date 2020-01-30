# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import pytest
import papermill as pm
import scrapbook as sb

ABS_TOL = 0.1
KERNEL_NAME = "python3"
OUTPUT_NOTEBOOK = "output.ipynb"


@pytest.mark.integration
def test_text_classification_three_player_explainer(notebooks, tmp):
    notebook_path = notebooks["text_classification_sst2_three_player"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
        parameters=dict(
            DATA_FOLDER=tmp,
            CUDA=True,
            QUICK_RUN=False,
            MODEL_SAVE_DIR=tmp
        ),
    )
    result = sb.read_notebook(OUTPUT_NOTEBOOK).scraps.data_dict
    print(result)
    assert pytest.approx(result["accuracy"], 0.93, abs=ABS_TOL)
    assert pytest.approx(result["precision"], 0.93, abs=ABS_TOL)
    assert pytest.approx(result["recall"], 0.93, abs=ABS_TOL)
    assert pytest.approx(result["f1"], 0.93, abs=ABS_TOL)