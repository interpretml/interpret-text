# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import pytest
import papermill as pm
import scrapbook as sb

ABS_TOL = 0.2
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
    assert pytest.approx(result["accuracy"], 0.72, abs=ABS_TOL)
    assert pytest.approx(result["anti_accuracy"], 0.69, abs=ABS_TOL)
    assert pytest.approx(result["sparsity"], 0.17, abs=ABS_TOL)
