import numpy as np
import pandas as pd
from typing import Any, Iterable, Tuple, List
from interpret_text.common.base_explainer import BaseTextPreprocessor


class GlovePreprocessor(BaseTextPreprocessor):

    def __init__(self, count_threshold: int, token_cutoff: int):
        self.count_threshold = count_threshold
        self.token_cutoff = token_cutoff

    def build_vocab(self, text) -> Any:
        words_to_indexes = {"<PAD>": 0, "<UNK>": 1}
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

    def preprocess(self, data) -> Iterable[Any]:
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
        tokens = []
        for i in id_list:
            if i.item() in self.reverse_word_vocab:
                tokens.append(self.reverse_word_vocab[i.item()])
            else:
                tokens.append("<UNK>")
        return tokens

    def generate_tokens(self, text: str) -> List[Any]:
        indexed_text = [
            self.word_vocab[word]
            if ((word in self.counts) and (self.counts[word]
                                           > self.count_threshold))
            else self.word_vocab["<UNK>"]
            for word in text.split()
        ]
        pad_length = max((self.token_cutoff - len(indexed_text)), 0)
        mask = [1] * min(len(indexed_text), self.token_cutoff) + [0]\
            * pad_length

        indexed_text = indexed_text[0:self.token_cutoff]\
            + [self.word_vocab["<PAD>"]] * pad_length

        return [np.array(indexed_text), np.array(mask)]

    def get_tokenizer(self) -> Any:
        return self.generate_tokens
