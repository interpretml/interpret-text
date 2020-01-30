import os
import logging

import numpy as np
import pandas as pd

from transformers import BertTokenizer
from interpret_text.common.dataset.utils_data_shared import download_and_unzip
import torch

from torch.autograd import Variable


def generate_data(batch, use_cuda):
    """Create a formatted and ordered data batch to use in the
    three player model.

    :param batch: A pandas dataframe containing the tokens, masks, counts, and
        labels associated with a batch of data
    :type batch: pd.DataFrame
    :param use_cuda: whether to use CUDA
    :type use_cuda: bool
    :return: formatted and ordered tokens (x), masks (m), and
        labels (y) associated with a batch of data
    :rtype: dict
    """
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

    def __init__(self, cuda=True, pretrain_cls=True, batch_size=32,
                 num_epochs=200, num_pretrain_epochs = 10, save_best_model=False, model_save_dir=".",
                 model_prefix="3PlayerModel"):
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
        :param model_prefix: What to name saved models and logs
        :type model_prefix: string
        """
        # dimension of a token's embedding
        # ex: for glove 100d, embedding_dim is 100
        self.embedding_dim = None

        # dimension of the gen. classifier's last layer
        # ex: In BERT gen. classifier, hidden_dim is 768
        self.hidden_dim = None

        # dropout rate must be specified if RNN classifier modules are used
        # dropout rate only matters if an RNN with > 1 layer is provided
        self.dropout_rate = None

        # number of layers to use if the RNN module is used
        self.layer_num = None

        # only used if an RNN module is used
        self.embedding_path = None

        # not necessary to change, this dimension deals with encodes the labels 
        self.label_embedding_dim = 400

        # freezes entire gen. classifier if set to True
        self.fixed_classifier = False

        # lambas for loss function
        self.lambda_sparsity = 1.0
        self.lambda_continuity = 0
        self.lambda_anti = 1.0

        # target_sparsity is the desired sparsity ratio
        self.target_sparsity = 0.3

        # this is the target number of target continuous pieces
        # it has no effect now, because lambda_continuity is 0
        self.count_pieces = 4

        # rate at which the generator explores different rationales
        self.exploration_rate = 0.05

        # multiplier to reward/penalize an accuracy gap between the classifier and anti-classifier 
        self.lambda_acc_gap = 1.2

        # learning rate
        self.lr = 2e-4

        # whether to tune the weights of the embedding layer
        self.fine_tuning = False

        # training parameters
        self.cuda = cuda
        self.pretrain_cls = pretrain_cls
        self.num_pretrain_epochs = num_pretrain_epochs
        self.num_epochs = num_epochs
        self.train_batch_size = batch_size
        self.test_batch_size = batch_size

        # stop training if validation acc does not improve for more than
        # training_stop thresh
        self.training_stop_thresh = 5

        # the numerical labels for classification. ex: MNLI needs [0, 1, 2, 3, 4]
        self.labels = [0, 1]

        # for saving models and logging
        self.save_best_model = save_best_model
        self.model_prefix = model_prefix
        self.save_path = model_save_dir

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

    def __init__(self, build_word_dict=False, tokenizer=None, max_length=50,
                 pad_to_max=True, text=None):
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

        self.reverse_word_vocab = None
        if build_word_dict:
            assert (text is not None), "must include document sentences to\
            build word vocab"
            (
                self.word_vocab,
                self.reverse_word_vocab,
                self.counts,
            ) = self.build_vocab(text)

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
        words_to_idxs = {}
        counts = {}

        for special_token in self.tokenizer.all_special_tokens:
            words_to_idxs[special_token] =\
                self.tokenizer.convert_tokens_to_ids(special_token)
            counts[special_token] = 1000
        print(words_to_idxs)
        for sentence in text:
            tokens = self.tokenizer.tokenize(sentence)
            ids = self.tokenizer.encode(sentence, add_special_tokens=False)
            for (token, token_id) in zip(tokens, ids):
                if token not in words_to_idxs:
                    words_to_idxs[token] = token_id
                    counts[token] = 1
                else:
                    counts[token] += 1
        idxs_to_words = {v: k for k, v in words_to_idxs.items()}

        max_num = max(words_to_idxs.values())
        for i in range(max_num):
            if i not in idxs_to_words:
                token = self.tokenizer.convert_ids_to_tokens(i)
                idxs_to_words[i] = token
                words_to_idxs[token] = i
                counts[token] = 0
        print('max num:', max_num)
        print("vocab size:", len(words_to_idxs))
        return words_to_idxs, idxs_to_words, counts

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
        :param count_thresh: the minimum number of times a word has to appear 
            in a sentence to be counted as part of the vocabulary
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
            if ((word in self.counts) and (self.counts[word] > self.count_thresh))
            else self.word_vocab["<UNK>"]
            for word in text.split()
        ]
        pad_length = max((self.token_cutoff - len(indexed_text)), 0)
        mask = [1] * min(len(indexed_text), self.token_cutoff) + [0] * pad_length

        indexed_text = indexed_text[0:self.token_cutoff] + [self.word_vocab["<PAD>"]] * pad_length

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
        tokens = []
        for i in id_list:
            if i.item() in self.reverse_word_vocab:
                tokens.append(self.reverse_word_vocab[i.item()])
            else:
                tokens.append("<UNK>")
        return tokens

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
    file_path = download_and_unzip(URL, file_name)
    return file_path
