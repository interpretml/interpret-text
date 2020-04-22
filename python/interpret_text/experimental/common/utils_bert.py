# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This script reuses some code from
# https://github.com/huggingface/transformers/blob/master/examples
# /run_glue.py
# and
# Microsoft NLP recipes repo:
# https://github.com/microsoft/nlp-recipes/tree/master/utils_nlp

import logging
import warnings
from collections.abc import Iterable
from enum import Enum

import torch
from pytorch_pretrained_bert.tokenization import BertTokenizer
from tqdm import tqdm

import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam

from interpret_text.experimental.common.utils_pytorch import get_device, move_to_device
from interpret_text.experimental.common.constants import BertTokens

from collections import namedtuple
from cached_property import cached_property

# Max supported sequence length
BERT_MAX_LEN = 512


class Language(str, Enum):
    """An enumeration of the supported pretrained models and languages."""

    ENGLISH: str = "bert-base-uncased"
    ENGLISHCASED: str = "bert-base-cased"
    ENGLISHLARGE: str = "bert-large-uncased"
    ENGLISHLARGECASED: str = "bert-large-cased"
    ENGLISHLARGEWWM: str = "bert-large-uncased-whole-word-masking"
    ENGLISHLARGECASEDWWM: str = "bert-large-cased-whole-word-masking"
    CHINESE: str = "bert-base-chinese"
    MULTILINGUAL: str = "bert-base-multilingual-cased"


class Tokenizer:
    def __init__(self, language=Language.ENGLISH, to_lower=False, cache_dir="."):
        """Initializes the underlying pretrained BERT tokenizer.

        Args:
            language (Language, optional): The pretrained model's language.
                                           Defaults to Language.ENGLISH.
            cache_dir (str, optional): Location of BERT's cache directory.
                Defaults to ".".
        """
        self.tokenizer = BertTokenizer.from_pretrained(
            language, do_lower_case=to_lower, cache_dir=cache_dir
        )
        self.language = language

    def tokenize(self, text):
        """Tokenizes a list of documents using a BERT tokenizer

        Args:
            text (list): List of strings (one sequence) or
                tuples (two sequences).

        Returns:
            [list]: List of lists. Each sublist contains WordPiece tokens
                of the input sequence(s).
        """
        if isinstance(text[0], str):
            return [self.tokenizer.tokenize(x) for x in tqdm(text)]
        else:
            return [
                [self.tokenizer.tokenize(x) for x in sentences]
                for sentences in tqdm(text)
            ]

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length.
        This is a simple heuristic which will always truncate the longer
        sequence one token at a time. This makes more sense than
        truncating an equal percent of tokens from each, since if one
        sequence is very short then each token that's truncated likely
        contains more information than a longer sequence."""

        if not tokens_b:
            max_length += 1

        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

        tokens_a.append(BertTokens.SEP)

        if tokens_b:
            tokens_b.append(BertTokens.SEP)

        return [tokens_a, tokens_b]

    def preprocess_classification_tokens(self, tokens, max_len=BERT_MAX_LEN):
        """Preprocessing of input tokens:
            - add BERT sentence markers ([CLS] and [SEP])
            - map tokens to token indices in the BERT vocabulary
            - pad and truncate sequences
            - create an input_mask
            - create token type ids, aka. segment ids

        Args:
            tokens (list): List of token lists to preprocess.
            max_len (int, optional): Maximum number of tokens
                            (documents will be truncated or padded).
                            Defaults to 512.
        Returns:
            tuple: A tuple containing the following three lists
                list of preprocessed token lists
                list of input mask lists
                list of token type id lists
        """
        if max_len > BERT_MAX_LEN:
            logging.info(
                "setting max_len to max allowed tokens: {}".format(BERT_MAX_LEN)
            )
            max_len = BERT_MAX_LEN

        if isinstance(tokens[0][0], str):
            tokens = [x[0: max_len - 2] + [BertTokens.SEP] for x in tokens]
            token_type_ids = None
        else:
            # get tokens for each sentence [[t00, t01, ...] [t10, t11,... ]]
            tokens = [
                self._truncate_seq_pair(sentence[0], sentence[1], max_len - 3)
                for sentence in tokens
            ]

            # construct token_type_ids
            # [[0, 0, 0, 0, ... 0, 1, 1, 1, ... 1], [0, 0, 0, ..., 1, 1, ]
            token_type_ids = [
                [[i] * len(sentence) for i, sentence in enumerate(example)]
                for example in tokens
            ]
            # merge sentences
            tokens = [
                [token for sentence in example for token in sentence]
                for example in tokens
            ]
            # prefix with [0] for [CLS]
            token_type_ids = [
                [0] + [i for sentence in example for i in sentence]
                for example in token_type_ids
            ]
            # pad sequence
            token_type_ids = [x + [0] * (max_len - len(x)) for x in token_type_ids]

        tokens = [[BertTokens.CLS] + x for x in tokens]
        # convert tokens to indices
        tokens = [self.tokenizer.convert_tokens_to_ids(x) for x in tokens]
        # pad sequence
        tokens = [x + [0] * (max_len - len(x)) for x in tokens]
        # create input mask
        input_mask = [[min(1, x) for x in y] for y in tokens]
        return tokens, input_mask, token_type_ids

    def preprocess_encoder_tokens(self, tokens, max_len=BERT_MAX_LEN):
        """Preprocessing of input tokens:
            - add BERT sentence markers ([CLS] and [SEP])
            - map tokens to token indices in the BERT vocabulary
            - pad and truncate sequences
            - create an input_mask
            - create token type ids, aka. segment ids

        Args:
            tokens (list): List of token lists to preprocess.
            max_len (int, optional): Maximum number of tokens
                            (documents will be truncated or padded).
                            Defaults to 512.
        Returns:
            tuple: A tuple containing the following four lists
                list of preprocessed token lists
                list of input id lists
                list of input mask lists
                list of token type id lists
        """
        if max_len > BERT_MAX_LEN:
            logging.info(
                "setting max_len to max allowed tokens: {}".format(BERT_MAX_LEN)
            )
            max_len = BERT_MAX_LEN

        if isinstance(tokens[0][0], str):
            tokens = [x[0: max_len - 2] + ["[SEP]"] for x in tokens]
            token_type_ids = None
        else:
            # get tokens for each sentence [[t00, t01, ...] [t10, t11,... ]]
            tokens = [
                self._truncate_seq_pair(sentence[0], sentence[1], max_len - 3)
                for sentence in tokens
            ]

            # construct token_type_ids
            # [[0, 0, 0, 0, ... 0, 1, 1, 1, ... 1], [0, 0, 0, ..., 1, 1, ]
            token_type_ids = [
                [[i] * len(sentence) for i, sentence in enumerate(example)]
                for example in tokens
            ]
            # merge sentences
            tokens = [
                [token for sentence in example for token in sentence]
                for example in tokens
            ]
            # prefix with [0] for [CLS]
            token_type_ids = [
                [0] + [i for sentence in example for i in sentence]
                for example in token_type_ids
            ]
            # pad sequence
            token_type_ids = [x + [0] * (max_len - len(x)) for x in token_type_ids]

        tokens = [[BertTokens.CLS] + x for x in tokens]
        # convert tokens to indices
        input_ids = [self.tokenizer.convert_tokens_to_ids(x) for x in tokens]
        # pad sequence
        input_ids = [x + [0] * (max_len - len(x)) for x in input_ids]
        # create input mask
        input_mask = [[min(1, x) for x in y] for y in input_ids]
        return tokens, input_ids, input_mask, token_type_ids

    def tokenize_ner(
        self,
        text,
        max_len=BERT_MAX_LEN,
        labels=None,
        label_map=None,
        trailing_piece_tag="X",
    ):
        """
        Tokenize and preprocesses input word lists, involving the following steps
            0. WordPiece tokenization.
            1. Convert string tokens to token ids.
            2. Convert input labels to label ids, if labels and label_map are
                provided.
            3. If a word is tokenized into multiple pieces of tokens by the
                WordPiece tokenizer, label the extra tokens with
                trailing_piece_tag.
            4. Pad or truncate input text according to max_seq_length
            5. Create input_mask for masking out padded tokens.

        Args:
            text (list): List of lists. Each sublist is a list of words in an
                input sentence.
            max_len (int, optional): Maximum length of the list of
                tokens. Lists longer than this are truncated and shorter
                ones are padded with "O"s. Default value is BERT_MAX_LEN=512.
            labels (list, optional): List of word label lists. Each sublist
                contains labels corresponding to the input word list. The lengths
                of the label list and word list must be the same. Default
                value is None.
            label_map (dict, optional): Dictionary for mapping original token
                labels (which may be string type) to integers. Default value
                is None.
            trailing_piece_tag (str, optional): Tag used to label trailing
                word pieces. For example, "criticize" is broken into "critic"
                and "##ize", "critic" preserves its original label and "##ize"
                is labeled as trailing_piece_tag. Default value is "X".

        Returns:
            tuple: A tuple containing the following four lists.
                1. input_ids_all: List of lists. Each sublist contains
                    numerical values, i.e. token ids, corresponding to the
                    tokens in the input text data.
                2. input_mask_all: List of lists. Each sublist
                    contains the attention mask of the input token id list,
                    1 for input tokens and 0 for padded tokens, so that
                    padded tokens are not attended to.
                3. trailing_token_mask: List of lists. Each sublist is
                    a boolean list, True for the first word piece of each
                    original word, False for the trailing word pieces,
                    e.g. "##ize". This mask is useful for removing the
                    predictions on trailing word pieces, so that each
                    original word in the input text has a unique predicted
                    label.
                4. label_ids_all: List of lists of numerical labels,
                    each sublist contains token labels of a input
                    sentence/paragraph, if labels is provided. If the `labels`
                    argument is not provided, the value of this is None.
        """

        def _is_iterable_but_not_string(obj):
            return isinstance(obj, Iterable) and not isinstance(obj, str)

        if max_len > BERT_MAX_LEN:
            warnings.warn(
                "setting max_len to max allowed tokens: {}".format(BERT_MAX_LEN)
            )
            max_len = BERT_MAX_LEN

        if not _is_iterable_but_not_string(text):
            # The input text must be an non-string Iterable
            raise ValueError("Input text must be an iterable and not a string.")
        else:
            # If the input text is a single list of words, convert it to
            # list of lists for later iteration
            if not _is_iterable_but_not_string(text[0]):
                text = [text]
        if labels is not None:
            if not _is_iterable_but_not_string(labels):
                raise ValueError("labels must be an iterable and not a string.")
            else:
                if not _is_iterable_but_not_string(labels[0]):
                    labels = [labels]

        label_available = True
        if labels is None:
            label_available = False
            # create an artificial label list for creating trailing token mask
            labels = [["O"] * len(t) for t in text]

        input_ids_all = []
        input_mask_all = []
        label_ids_all = []
        trailing_token_mask_all = []
        for t, t_labels in zip(text, labels):

            if len(t) != len(t_labels):
                raise ValueError(
                    "The number of words is {0}, but the number of labels is {1}.".format(
                        len(t), len(t_labels)
                    )
                )

            new_labels = []
            new_tokens = []
            if label_available:
                for word, tag in zip(t, t_labels):
                    sub_words = self.tokenizer.tokenize(word)
                    for count, sub_word in enumerate(sub_words):
                        if count > 0:
                            tag = trailing_piece_tag
                        new_labels.append(tag)
                        new_tokens.append(sub_word)
            else:
                for word in t:
                    sub_words = self.tokenizer.tokenize(word)
                    for count, sub_word in enumerate(sub_words):
                        if count > 0:
                            tag = trailing_piece_tag
                        else:
                            tag = "O"
                        new_labels.append(tag)
                        new_tokens.append(sub_word)

            if len(new_tokens) > max_len:
                new_tokens = new_tokens[:max_len]
                new_labels = new_labels[:max_len]
            input_ids = self.tokenizer.convert_tokens_to_ids(new_tokens)

            # The mask has 1 for real tokens and 0 for padding tokens.
            # Only real tokens are attended to.
            input_mask = [1.0] * len(input_ids)

            # Zero-pad up to the max sequence length.
            padding = [0.0] * (max_len - len(input_ids))
            label_padding = ["O"] * (max_len - len(input_ids))

            input_ids += padding
            input_mask += padding
            new_labels += label_padding

            trailing_token_mask_all.append(
                [True if label != trailing_piece_tag else False for label in new_labels]
            )

            if label_map:
                label_ids = [label_map[label] for label in new_labels]
            else:
                label_ids = new_labels

            input_ids_all.append(input_ids)
            input_mask_all.append(input_mask)
            label_ids_all.append(label_ids)

        if label_available:
            return (
                input_ids_all,
                input_mask_all,
                trailing_token_mask_all,
                label_ids_all,
            )
        else:
            return input_ids_all, input_mask_all, trailing_token_mask_all, None


class BERTSequenceClassifier:
    """BERT-based sequence classifier"""

    def __init__(self, language=Language.ENGLISH, num_labels=2, cache_dir="."):
        """Initializes the classifier and the underlying pretrained model.

        Args:
            language (Language, optional): The pretrained model's language.
                                           Defaults to Language.ENGLISH.
            num_labels (int, optional): The number of unique labels in the
                training data. Defaults to 2.
            cache_dir (str, optional): Location of BERT's cache directory.
                Defaults to ".".
        """
        if num_labels < 2:
            raise ValueError("Number of labels should be at least 2.")

        self.language = language
        self.num_labels = num_labels
        self.cache_dir = cache_dir

        # create classifier
        self.model = BertForSequenceClassification.from_pretrained(
            language, cache_dir=cache_dir, num_labels=num_labels
        )
        self.has_cuda = self.cuda

    @cached_property
    def cuda(self):
        """ cache the output of torch.cuda.is_available() """

        self.has_cuda = torch.cuda.is_available()
        return self.has_cuda

    def fit(
        self,
        token_ids,
        input_mask,
        labels,
        token_type_ids=None,
        num_gpus=None,
        num_epochs=1,
        batch_size=32,
        lr=2e-5,
        warmup_proportion=None,
        verbose=True,
    ):
        """Fine-tunes the BERT classifier using the given training data.

        Args:
            token_ids (list): List of training token id lists.
            input_mask (list): List of input mask lists.
            labels (list): List of training labels.
            token_type_ids (list, optional): List of lists. Each sublist
                contains segment ids indicating if the token belongs to
                the first sentence(0) or second sentence(1). Only needed
                for two-sentence tasks.
            num_gpus (int, optional): The number of gpus to use.
                                      If None is specified, all available GPUs
                                      will be used. Defaults to None.
            num_epochs (int, optional): Number of training epochs.
                Defaults to 1.
            batch_size (int, optional): Training batch size. Defaults to 32.
            lr (float): Learning rate of the Adam optimizer. Defaults to 2e-5.
            warmup_proportion (float, optional): Proportion of training to
                perform linear learning rate warmup for. E.g., 0.1 = 10% of
                training. Defaults to None.
            verbose (bool, optional): If True, shows the training progress and
                loss values. Defaults to True.
        """

        device, num_gpus = get_device(num_gpus)

        self.model = move_to_device(self.model, device, num_gpus)

        token_ids_tensor = torch.tensor(token_ids, dtype=torch.long)
        input_mask_tensor = torch.tensor(input_mask, dtype=torch.long)
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        if token_type_ids:
            token_type_ids_tensor = torch.tensor(token_type_ids, dtype=torch.long)
            train_dataset = TensorDataset(
                token_ids_tensor,
                input_mask_tensor,
                token_type_ids_tensor,
                labels_tensor,
            )
        else:
            train_dataset = TensorDataset(
                token_ids_tensor, input_mask_tensor, labels_tensor
            )
        train_sampler = RandomSampler(train_dataset)

        train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler, batch_size=batch_size
        )
        # define optimizer and model parameters
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        num_batches = len(train_dataloader)
        num_train_optimization_steps = num_batches * num_epochs

        if warmup_proportion is None:
            opt = BertAdam(optimizer_grouped_parameters, lr=lr)
        else:
            opt = BertAdam(
                optimizer_grouped_parameters,
                lr=lr,
                t_total=num_train_optimization_steps,
                warmup=warmup_proportion,
            )

        # define loss function
        loss_func = nn.CrossEntropyLoss().to(device)

        # train
        self.model.train()  # training mode

        for epoch in range(num_epochs):
            training_loss = 0
            for i, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                if token_type_ids:
                    x_batch, mask_batch, token_type_ids_batch, y_batch = tuple(
                        t.to(device) for t in batch
                    )
                else:
                    token_type_ids_batch = None
                    x_batch, mask_batch, y_batch = tuple(t.to(device) for t in batch)

                opt.zero_grad()

                y_h = self.model(
                    input_ids=x_batch,
                    token_type_ids=token_type_ids_batch,
                    attention_mask=mask_batch,
                    labels=None,
                )
                loss = loss_func(y_h, y_batch).mean()

                training_loss += loss.item()

                loss.backward()
                opt.step()
                if verbose:
                    if i % ((num_batches // 10) + 1) == 0:
                        logging.info(
                            "epoch:{}/{}; batch:{}->{}/{}; average training loss:{:.6f}".format(
                                epoch + 1,
                                num_epochs,
                                i + 1,
                                min(i + 1 + num_batches // 10, num_batches),
                                num_batches,
                                training_loss / (i + 1),
                            )
                        )
        # empty cache
        del [x_batch, y_batch, mask_batch, token_type_ids_batch]
        torch.cuda.empty_cache()

    def predict(
        self,
        token_ids,
        input_mask,
        token_type_ids=None,
        num_gpus=None,
        batch_size=32,
        probabilities=False,
    ):
        """Scores the given dataset and returns the predicted classes.

        Args:
            token_ids (list): List of training token lists.
            input_mask (list): List of input mask lists.
            token_type_ids (list, optional): List of lists. Each sublist
                contains segment ids indicating if the token belongs to
                the first sentence(0) or second sentence(1). Only needed
                for two-sentence tasks.
            num_gpus (int, optional): The number of gpus to use.
                                      If None is specified, all available GPUs
                                      will be used. Defaults to None.
            batch_size (int, optional): Scoring batch size. Defaults to 32.
            probabilities (bool, optional):
                If True, the predicted probability distribution
                is also returned. Defaults to False.
        Returns:
            1darray, namedtuple(1darray, ndarray): Predicted classes or
                (classes, probabilities) if probabilities is True.
        """
        device, num_gpus = get_device(num_gpus)
        self.model = move_to_device(self.model, device, num_gpus)

        # score
        self.model.eval()

        token_ids_tensor = torch.tensor(token_ids, dtype=torch.long)
        input_mask_tensor = torch.tensor(input_mask, dtype=torch.long)

        if token_type_ids:
            token_type_ids_tensor = torch.tensor(token_type_ids, dtype=torch.long)
            test_dataset = TensorDataset(
                token_ids_tensor, input_mask_tensor, token_type_ids_tensor
            )
        else:
            test_dataset = TensorDataset(token_ids_tensor, input_mask_tensor)

        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(
            test_dataset, sampler=test_sampler, batch_size=batch_size
        )

        preds = []
        for i, batch in enumerate(tqdm(test_dataloader, desc="Iteration")):
            if token_type_ids:
                x_batch, mask_batch, token_type_ids_batch = tuple(
                    t.to(device) for t in batch
                )
            else:
                token_type_ids_batch = None
                x_batch, mask_batch = tuple(t.to(device) for t in batch)

            with torch.no_grad():
                p_batch = self.model(
                    input_ids=x_batch,
                    token_type_ids=token_type_ids_batch,
                    attention_mask=mask_batch,
                    labels=None,
                )
            preds.append(p_batch.cpu())

        preds = np.concatenate(preds)

        if probabilities:
            return namedtuple("Predictions", "classes probabilities")(
                preds.argmax(axis=1), nn.Softmax(dim=1)(torch.Tensor(preds)).numpy()
            )
        else:
            return preds.argmax(axis=1)
