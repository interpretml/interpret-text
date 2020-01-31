# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

# Unit tests for model explainability SDK. Doesn't test visualization method
import os
import torch
import pandas as pd

from interpret_text.three_player_introspective.three_player_introspective_explainer import (
    ThreePlayerIntrospectiveExplainer
)
from interpret_text.common.dataset.utils_sst2 import load_sst2_pandas_df
from interpret_text.common.utils_three_player import GlovePreprocessor, ModelArguments, load_glove_embeddings


class TestThreePlayerExplainer(object):
    def test_working(self):
        assert True



