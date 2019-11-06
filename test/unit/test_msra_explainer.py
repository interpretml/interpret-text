# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import pytest

# Tests for model explainability SDK
import shutil
import numpy as np
import logging
import shap
import torch
import json
from numpy import dot
from numpy.linalg import norm
from urllib import request
from pytorch_pretrained_bert import BertModel, BertTokenizer

from interpret_text.msra.MSRAExplainer import MSRAExplainer

class TestMSRAExplainer(object):
    def test_working(self):
        assert True

    def test_explain_model_BERT(self):
        device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
        text = "rare bird has more than enough charm to make it memorable."
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        words = ["[CLS]"] + tokenizer.tokenize(text) + ["[SEP]"]
        model = BertModel.from_pretrained("bert-base-uncased").to(device)
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        tokenized_ids = tokenizer.convert_tokens_to_ids(words)
        segment_ids = [0 for _ in range(len(words))]
        token_tensor = torch.tensor([tokenized_ids], device=device)
        segment_tensor = torch.tensor([segment_ids], device=device)
        x_bert = model.embeddings(token_tensor, segment_tensor)[0]
        data = request.urlopen("https://nlpbp.blob.core.windows.net/data/regular.json").read()
        regularization_bert = json.loads(data)
        interpreter_simple = MSRAExplainer(input_embeddings=x_bert, regularization=regularization_bert, words=words)
        explanation = interpreter_simple.explain_local(model=model,evaluation_examples=text,device=device)
        print(explanation.local_importance_values)
        valid_imp_vals = np.array([0.1741795390844345, 0.14556556940078735, 0.14939342439174652, 0.23016422986984253, 0.20574013888835907, 0.20974034070968628, 0.18763116002082825, 0.13564500212669373, 0.26105546951293945, 0.22772076725959778, 0.24030782282352448, 0.12845061719417572, 0.27545174956321716, 0.3505420684814453])
        local_importance_values = np.array(explanation.local_importance_values)
        cos_sim = cos_sim = dot(valid_imp_vals, local_importance_values)/(norm(valid_imp_vals)*norm(local_importance_values))
        assert (cos_sim >= .99)