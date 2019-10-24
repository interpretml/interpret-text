# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import json
import shutil
import pytest
import papermill as pm
import scrapbook as sb
#from common import OUTPUT_NOTEBOOK, KERNEL_NAME


ABS_TOL = 0.1
KERNEL_NAME = "python3"
OUTPUT_NOTEBOOK = "output.ipynb"

#@pytest.mark.gpu
#@pytest.mark.integration
def test_text_classification_mnli_bert(notebook_path, tmp):
    #notebook_path = notebooks
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
        parameters=dict(
            DATA_FOLDER=tmp,
            BERT_CACHE_DIR=tmp,
            BATCH_SIZE=32,
            BATCH_SIZE_PRED=512,
            NUM_EPOCHS=1,
            TEST = True
        ),
    )
    result = sb.read_notebook(OUTPUT_NOTEBOOK).scraps.data_dict
    assert pytest.approx(result["accuracy"], 0.93, abs=ABS_TOL)
    assert pytest.approx(result["precision"], 0.93, abs=ABS_TOL)
    assert pytest.approx(result["recall"], 0.93, abs=ABS_TOL)
    assert pytest.approx(result["f1"], 0.93, abs=ABS_TOL)

test_text_classification_mnli_bert('C:\\Users\\ehnosakh\\Documents\\03InterpretText\\developer\\Repo\\interpret-community-text\\notebooks\\text_classification_mnli_bert.ipynb',
 'C:\\Users\\ehnosakh\\Documents\\03InterpretText\\developer\\Repo\\interpret-community-text\\notebooks\\temp' )



