import os
import logging

import numpy as np
import pandas as pd

from transformers import BertTokenizer
from interpret_text.common.dataset.utils_data_shared import download_and_unzip
import torch

from torch.autograd import Variable


def generate_data(batch, use_cuda):
    # sort for rnn happiness
    batch.sort_values("counts", inplace=True, ascending=False)

    x_mask = np.stack(batch["mask"], axis=0)
    # drop all zero columns
    zero_col_idxs = np.argwhere(np.all(x_mask[..., :] == 0, axis=0))
    x_mask = np.delete(x_mask, zero_col_idxs, axis=1)

    x_mat = np.stack(batch["tokens"], axis=0)
    # drop all zero columns
    x_mat = np.delete(x_mat, zero_col_idxs, axis=1)

    y_vec = np.stack(batch["labels"], axis=0)

    batch_x_ = Variable(torch.from_numpy(x_mat)).to(torch.int64)
    batch_m_ = Variable(torch.from_numpy(x_mask)).type(torch.FloatTensor)
    batch_y_ = Variable(torch.from_numpy(y_vec)).to(torch.int64)

    if use_cuda:
        batch_x_ = batch_x_.cuda()
        batch_m_ = batch_m_.cuda()
        batch_y_ = batch_y_.cuda()

    return {"x": batch_x_, "m": batch_m_, "y": batch_y_}


class ModelArguments:
    """Default parameters used to initialize the model (independent of module
       component types).
    """

    def __init__(self, cuda, pre_train_cls, batch_size, num_epochs,
                 save_best_model, model_save_dir=None, model_prefix=None):
        """Initialize model parameters

        :param cuda: Whether or not to use cuda
        :type cuda: bool
        :param pre_train_cls: Whether to pretrain the introspective
            generator's classifier
        :type pre_train_cls: bool
        :param batch_size: Batch size for training and testing
        :type batch_size: int
        :param num_epochs: Number of epochs to run in training
        :type num_epochs: int
        :param save_best_model: Whether to save the best model
        :type save_best_model: bool
        :param model_save_dir: Directory to save models and logs
        :type model_save_dir: string
        :param model_prefix: What to name saved models
        :type model_prefix: string
        """
        # to initialize model modules
        self.embedding_dim = 100
        self.hidden_dim = 200
        self.layer_num = 1
        # z indicates whether something is a rationale, dimension always 2
        self.z_dim = 2
        self.dropout_rate = 0.5
        self.label_embedding_dim = 400
        self.fixed_classifier = True

        # to initialize model
        self.fine_tuning = False
        self.lambda_sparsity = 1.0
        self.lambda_continuity = 0
        self.lambda_anti = 1.0
        self.exploration_rate = 0.05
        self.count_tokens = 8
        self.count_pieces = 4
        self.lambda_acc_gap = 1.2
        self.lr = 0.001

        # training parameters
        self.cuda = cuda
        self.pre_train_cls = pre_train_cls
        self.train_batch_size = batch_size
        self.test_batch_size = batch_size
        self.num_epochs = num_epochs
        self.save_best_model = save_best_model

        if save_best_model:
            assert model_prefix is not None,\
                "Please input a model name (prefix for saved files)"
            assert model_save_dir is not None,\
                "Please input a directory to save models in"
        self.model_prefix = model_prefix
        self.save_path = model_save_dir

        # for saving models and logging
        self.model_folder_path = os.path.join(
            self.save_path,
            self.model_prefix + "_training_")

        if not os.path.exists(self.model_folder_path):
            os.makedirs(self.model_folder_path)
        self.log_filepath = os.path.join(
            self.model_folder_path, "training_stats.txt"
        )
        logging.basicConfig(
            filename=self.log_filepath, filemode="a", level=logging.INFO
        )


class BertPreprocessor:
    """A class that tokenizes and otherwise processes text to be encoded
       by a BERT model.
    """

    def __init__(self, tokenizer=None, max_length=50, pad_to_max=True):
        """Initialize the BertPreprocessor.

        :param tokenizer: an initialized tokenizer with 'encode_plus' and
            'convert ids to tokens' methods; defaults to None
        :type tokenizer: tokenizer, optional
        :param max_length: the max number of tokens allowable in an example,
            defaults to 50
        :type max_length: int, optional
        :param pad_to_max: whether to pad all examples to the same (max)
            length, defaults to True
        :type pad_to_max: bool, optional
        """
        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        self.max_length = max_length
        self.pad_to_max = True

    def preprocess(self, data):
        """Converts a list of text into a dataframe containing padded token ids,
            masks distinguishing word tokens from pads, and word token counts
            for each text in the list.
        :param data: a list of strings (e.g. sentences)
        :type data: list
        :return tokens: a dataframe containing
            lists of word token ids, pad/word masks, and token counts
            for each string in the list
        :rtype: (pd.Dataframe)
        """
        input_ids = []
        attention_mask = []
        counts = []
        for text in data:
            d = self.tokenizer.encode_plus(
                text,
                max_length=self.max_length,
                pad_to_max_length=self.pad_to_max,
            )
            input_ids.append(d["input_ids"])
            attention_mask.append(d["attention_mask"])
            counts.append(sum(d["attention_mask"]))

        tokens = pd.DataFrame(
            {"tokens": input_ids, "mask": attention_mask, "counts": counts}
        )

        return tokens

    def decode_single(self, id_list):
        """Decodes a single list of token ids to tokens

        :param id_list: a list of token ids
        :type id_list: list
        :return: a list of tokens
        :rtype: list
        """
        return self.tokenizer.convert_ids_to_tokens(id_list)


class GlovePreprocessor:
    """Splits words into tokens that can be passed into glove embeddings
    """

    def __init__(self, text, count_thresh, token_cutoff):
        """Initialize the preprocessor.
        :param text: a list of sentences (strings)
        :type text: list
        :param count_thresh: the number of times a word has to appear in a
            sentence to be counted as part of the vocabulary
        :type count_thresh: int
        :param token_cutoff: the maximum number of tokens a sentence can have
        :type token_cutoff: int
        """
        (
            self.word_vocab,
            self.reverse_word_vocab,
            self.counts,
        ) = self.build_vocab(text)
        self.count_thresh = count_thresh
        self.token_cutoff = token_cutoff

    def build_vocab(self, text):
        """Build the vocabulary (all words that appear at least once in text)
        :param text: a list of sentences (strings)
        :type text: list
        :return:
            words_to_idxs (dict): a mapping of the set of unique words (keys)
                found in text (appearing more times than count_thresh) to
                unique indices (values)
            idxs_to_words (dict): a mapping from vocabulary indices (keys)
                to words (values)
            counts (dict): a mapping of the a word (key) to the number of
                times it appears in text (value)
        :rtype: tuple of dictionaries
        """
        words_to_idxs = {"<PAD>": 0, "<UNK>": 1}
        counts = {}
        for sentence in text:
            for word in sentence.split():
                if word not in words_to_idxs:
                    words_to_idxs[word] = len(words_to_idxs)
                    counts[word] = 1
                else:
                    counts[word] += 1
        idxs_to_words = {v: k for k, v in words_to_idxs.items()}
        return words_to_idxs, idxs_to_words, counts

    def generate_tokens(self, text):
        """Split text into padded lists of tokens that are part of the
           recognized vocabulary
        :param text: a piece of text (e.g. a sentence)
        :type text: string
        :return:
            indexed_text (np.array): the token/vocabulary indices of
                recognized words in text, padded to the maximum sentence length
            mask (np.array): a mask indicating which indices in indexed_text
                correspond to words (1s) and which correspond to pads (0s)
        :rtype: tuple (of np.arrays)
        """
        indexed_text = [
            self.word_vocab[word]
            if word in self.counts and
            self.counts[word] > self.count_thresh
            else self.word_vocab["<UNK>"]
            for word in text.split()
        ]
        pad_length = self.token_cutoff - len(indexed_text)
        mask = [1] * len(indexed_text) + [0] * pad_length

        indexed_text = indexed_text + [self.word_vocab["<PAD>"]] * pad_length

        return np.array(indexed_text), np.array(mask)

    def preprocess(self, data):
        """Convert a list of text into a dataframe containing padded token ids,
        masks distinguishing word tokens from pads, and word token counts for
        each text in the list.
        :param data: list of strings (e.g. sentences)
        :type data: list
        :return: tokens (pd.Dataframe): a dataframe containing
            lists of word token ids, pad/word masks, and token counts
            for each string in the list
        :rtype: pandas dataframe
        """
        token_lists = []
        masks = []
        counts = []
        for sentence in data:
            token_list, mask = self.generate_tokens(sentence)
            token_lists.append(token_list)
            masks.append(mask)
            counts.append(np.sum(mask))
        tokens = pd.DataFrame(
            {"tokens": token_lists, "mask": masks, "counts": counts}
        )
        return tokens

    def decode_single(self, id_list):
        """Decodes a single list of token ids to tokens

        :param id_list: a list of token ids
        :type id_list: list
        :return: a list of tokens
        :rtype: list
        """
        return [self.reverse_word_vocab[i.item()] for i in id_list]


def load_glove_embeddings(local_cache_path="."):
    """Download premade glove embeddings (if not already downloaded)
    and unzip the downloaded file

    :param local_cache_path: download destination directory, defaults to "."
    :type local_cache_path: str, optional
    :return: file path to downloaded/unzipped file
    :rtype: str
    """
    URL = "http://nlp.stanford.edu/data/glove.6B.zip"
    file_name = "glove.6B.100d.txt"
    # TODO: upload just the 6B.100d embedding to blob storage so
    # downloading the entire zip isn't necessary
    file_path = download_and_unzip(URL, file_name)
    return file_path
