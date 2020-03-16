import torch
from interpret_text.common.utils_unified import (
    _get_single_embedding,
    make_bert_embeddings,
)
from utils_test import get_mnli_test_dataset, get_bert_model


class TestMSRAUtils(object):
    train_dataset = get_mnli_test_dataset()

    def test_working(self):
        assert True

    def test_get_single_embedding(self):
        model = get_bert_model()
        device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
        text = "rare bird has more than enough charm to make it memorable."
        embedded_input = _get_single_embedding(model, text, device)
        assert embedded_input is not None

    def test_make_bert_embeddings(self):
        model = get_bert_model()
        device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
        train_dataset = get_mnli_test_dataset()
        training_embeddings = make_bert_embeddings(train_dataset, model, device)
        assert training_embeddings is not None
