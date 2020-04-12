import pandas as pd
from typing import List

from interpret_text.common.base_explainer import BaseTextPreprocessor
from transformers import BertTokenizer


class BertPreprocessor(BaseTextPreprocessor):
    """ A class that tokenizes and otherwise processes text to be encoded
       by a BERT model.
    """

    MAX_TOKEN_LENGTH = 50
    TOKEN_PAD_TO_MAX = True
    BUILD_WORD_DICT = True
    COUNT_SPECIAL_TOKENS = 1000

    def build_vocab(self, text: List[str]):
        """ Build vocabulary

        :param text: a list of text used to built vocabulary
        :type text: List
        """
        words_to_indexes = {}
        counts = {}
        tokenizer = self.get_tokenizer()

        for special_token in tokenizer.all_special_tokens:
            words_to_indexes[special_token] = tokenizer.convert_tokens_to_ids(special_token)
            counts[special_token] = self.COUNT_SPECIAL_TOKENS

        for sentence in text:
            tokens = self.generate_tokens(sentence)
            ids = tokenizer.encode(sentence, add_special_tokens=False)
            for (token, token_id) in zip(tokens, ids):
                if token not in words_to_indexes:
                    words_to_indexes[token] = token_id
                    counts[token] = 1
                else:
                    counts[token] += 1

        indexes_to_words = {v: k for k, v in words_to_indexes.items()}

        max_num = max(words_to_indexes.values())

        for i in range(max_num):
            if i not in indexes_to_words:
                token = tokenizer.convert_ids_to_tokens(i)
                indexes_to_words[i] = token
                words_to_indexes[token] = i
                counts[token] = 0

        self.word_vocab = words_to_indexes
        self.reverse_word_vocab = indexes_to_words
        self.counts = counts

    def preprocess(self, data: List[str]) -> pd.DataFrame:
        """ Converts a list of text into a dataframe containing padded token ids,
        masks distinguishing word tokens from pads, and word token counts for each text in the list.

        :param data: A list of strings
        :type data: List[str]
        :return: Pandas Dataframe of string
        :rtype  pd.DataFrame
        """
        input_ids = []
        attention_mask = []
        counts = []

        tokenizer = self.get_tokenizer()
        for text in data:
            d = tokenizer.encode_plus(
                text,
                max_length=self.MAX_TOKEN_LENGTH,
                pad_to_max_length=self.TOKEN_PAD_TO_MAX,
            )
            input_ids.append(d["input_ids"])
            attention_mask.append(d["attention_mask"])
            counts.append(sum(d["attention_mask"]))

        tokens = pd.DataFrame(
            {"tokens": input_ids, "mask": attention_mask, "counts": counts}
        )

        return tokens

    def decode_single(self, id_list: List) -> List:
        """ Decodes a single list of token ids to tokens

        :param id_list: a list of token ids
        :type id_list: List
        :return: a list of tokens
        :rtype: List
        """
        tokenizer = self.get_tokenizer()
        return tokenizer.convert_ids_to_tokens(id_list)

    def generate_tokens(self, sentence: str) -> List:
        """ Generate tokens for given sentence

        :param sentence: sentence to be tokenized
        :type sentence: str
        :return: A list of tokens
        :rtype List
        """
        return self.get_tokenizer().tokenize(sentence)

    def get_tokenizer(self) -> BertTokenizer:
        """ Return bert tokenizer

        :return: BertTokenizer
        :rtype BertTokenizer
        """
        return BertTokenizer.from_pretrained("bert-base-uncased")
