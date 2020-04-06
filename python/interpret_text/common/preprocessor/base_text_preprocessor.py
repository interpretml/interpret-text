from abc import abstractmethod, ABC
from typing import Any, Iterable


class BaseTextPreprocessor(ABC):
    """ Abstract class for base text explainer """

    @abstractmethod
    def build_vocab(self, text: str):
        """Build the vocabulary (all words that appear at least once in text)

        :param text: a list of sentences (strings)
        :type text: list
        """
        pass

    @abstractmethod
    def preprocess(self, data: Iterable[Any]) -> Iterable[Any]:
        """ Converts data into another iterable
        :param data text to preprocess
        :type data: Iterable[Any]
        :return: Iterable type of preprocessed text
        :rtype Iterable[Any]
        """
        pass

    @abstractmethod
    def decode_single(self, id_list: Iterable[Any]) -> Iterable[Any]:
        """ Decodes a single list of token ids to tokens
        :param id_list: list of token ids
        :type id_list: List
        :return: list of tokens
        :rtype list
        """
        pass

    @abstractmethod
    def generate_tokens(self, sentence: str) -> Iterable[Any]:
        """ Abstract mehtod to generate tokens for a given sentence

        :param sentence: Sentence to be tokenized
        :type sentence: str
        :return:  A list of tokens
        :rtype  Iterable[Any]
        """
        pass

    def get_tokenizer(self) -> Any:
        """ Abstract class to get tokenzier of this preprocessor

        :return  Tokenizer
        """
        pass
