import numpy as np
import pandas as pd
from typing import Any, Iterable, List
from interpret_text.experimental.common.base_explainer import BaseTextPreprocessor

PAD = "<PAD>"
UNK = "<UNK>"


class GlovePreprocessor(BaseTextPreprocessor):
    """ Glove Preprocessor to split tokens that can be passed into glove embeddings """

    def __init__(self, count_threshold: int, token_cutoff: int):
        """ Initialize the GLovePreprocessor

        :param count_thresh: the minimum number of times a word has to appear
        in a sentence to be counted as part of the vocabulary
        :type int
        :param token_cutoff: the maximum number of tokens a sentence can have
        :type token_cutoff: int
        """
        self.count_threshold = count_threshold
        self.token_cutoff = token_cutoff

    def build_vocab(self, text):
        """ Build vocabulary
        """
        words_to_indexes = {PAD: 0, UNK: 1}
        counts = {}
        for sentence in text:
            for word in sentence.split():
                if word not in words_to_indexes:
                    words_to_indexes[word] = len(words_to_indexes)
                    counts[word] = 1
                else:
                    counts[word] += 1
        indexes_to_words = {v: k for k, v in words_to_indexes.items()}

        self.word_vocab = words_to_indexes
        self.reverse_word_vocab = indexes_to_words
        self.counts = counts

    def preprocess(self, data) -> pd.DataFrame:
        """ Convert a list of text into a dataframe containing padded token ids,
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

    def decode_single(self, id_list) -> Iterable[Any]:
        """ Decodes a single list of token ids to tokens

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
                tokens.append(UNK)
        return tokens

    def generate_tokens(self, text: str) -> List[Any]:
        """ Split text into padded lists of tokens that are part of the recognized vocabulary

        :param text: a piece of text (e.g. a sentence)
        :type text: str
        :return:
        indexed_text (np.array): the token/vocabulary indices of
                recognized words in text, padded to the maximum sentence length
        mask (np.array): a mask indicating which indices in indexed_text
                correspond to words (1s) and which correspond to pads (0s)
        :rtype: tuple (of np.arrays)
        """
        indexed_text = [
            self.word_vocab[word]
            if ((word in self.counts) and (self.counts[word]
                                           > self.count_threshold))
            else self.word_vocab[UNK]
            for word in text.split()
        ]
        pad_length = max((self.token_cutoff - len(indexed_text)), 0)
        mask = [1] * min(len(indexed_text), self.token_cutoff) + [0] * pad_length

        indexed_text = indexed_text[0:self.token_cutoff] + [self.word_vocab[PAD]] * pad_length

        return [np.array(indexed_text), np.array(mask)]

    def get_tokenizer(self) -> Any:
        return self.generate_tokens
