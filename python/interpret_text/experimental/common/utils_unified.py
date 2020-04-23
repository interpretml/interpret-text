# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This script reuses some code from
# Microsoft NLP recipes repo:
# https://github.com/microsoft/nlp-recipes/tree/master/utils_nlp

"""
    Utility functions useful across explainers
"""
import logging

import torch
from pytorch_pretrained_bert import BertTokenizer
from interpret_text.experimental.common.utils_bert import Language, Tokenizer
from interpret_text.experimental.common.constants import BertTokens

log = logging.getLogger(__name__)


def _get_single_embedding(model, text, device):
    """Get the bert embedding for a single sentence
    :param text: The current sentence
    :type text: str
    :param device: A pytorch device
    :type device: torch.device
    :param model: a pytorch model
    :type model: torch.nn
    :return: A bert embedding of the single sentence
    :rtype: torch.embedding
    """
    tokenizer = BertTokenizer.from_pretrained(Language.ENGLISH)
    words = [BertTokens.CLS] + tokenizer.tokenize(text) + [BertTokens.SEP]
    tokenized_ids = tokenizer.convert_tokens_to_ids(words)
    token_tensor = torch.tensor([tokenized_ids], device=device)
    embedding = model.bert.embeddings(token_tensor)[0]
    return embedding, words


def make_bert_embeddings(
    train_data,
    model,
    device,
    LANGUAGE=Language.ENGLISH,
    TO_LOWER=True,
    BERT_CACHE_DIR="./temp",
    MAX_LEN=150,
):
    """Get the bert embedding for multiple sentences
    :param train_data: A list of sentences
    :type df: list
    :param device: A pytorch device
    :type device: torch.device
    :param model: a pytorch model
    :type model: torch.nn
    :param LANGUAGE: The pretrained model's language. Defaults to Language.ENGLISH
    :type LANGUAUGE: str
    :param BERT_CACHE_DIR: Location of BERT's cache directory. Defaults to "'/temp".
    :type BERT_CACHE_DIR: str
    :return: A bert embedding of all the sentences
    :rtype: torch.embedding
    """
    tokenizer = Tokenizer(LANGUAGE, to_lower=TO_LOWER, cache_dir=BERT_CACHE_DIR)
    tokens = tokenizer.tokenize(train_data)
    tokens, mask, _ = tokenizer.preprocess_classification_tokens(tokens, MAX_LEN)
    tokens_tensor = torch.tensor(tokens, device=device)
    mask_tensor = torch.tensor(mask, device=device)
    embeddings = model.bert.embeddings(tokens_tensor, mask_tensor)
    return embeddings
