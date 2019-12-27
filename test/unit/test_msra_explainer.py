# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

# Tests for model explainability SDK
import numpy as np
import torch
from numpy import dot
from numpy.linalg import norm

from interpret_text.msra.MSRAExplainer import MSRAExplainer
from utils_test import get_mnli_test_dataset, get_bert_model


class TestMSRAExplainer(object):
    def test_working(self):
        assert True

    def test_explain_model_BERT_seq_classification(self):
        device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
        mnli_test_dataset = get_mnli_test_dataset()
        text = "rare bird has more than enough charm to make it memorable."
        model = get_bert_model()
        model.to(device)
        interpreter_msra = MSRAExplainer(
            model=model,
            train_dataset=list(mnli_test_dataset),
            device=device,
            target_layer=14,
        )
        explanation_msra = interpreter_msra.explain_local(text)
        valid_imp_vals = np.array(
            [
                0.2620866596698761,
                0.16004231572151184,
                0.17308972775936127,
                0.18205846846103668,
                0.26146841049194336,
                0.25957807898521423,
                0.3549807369709015,
                0.23873654007911682,
                0.2826242744922638,
                0.2700383961200714,
                0.3673151433467865,
                0.3899800479412079,
                0.20173774659633636,
                0.260466068983078,
            ]
        )
        print(explanation_msra.local_importance_values)
        local_importance_values = np.array(explanation_msra.local_importance_values)
        cos_sim = dot(valid_imp_vals, local_importance_values) / (
            norm(valid_imp_vals) * norm(local_importance_values)
        )
        assert cos_sim >= 0.80
