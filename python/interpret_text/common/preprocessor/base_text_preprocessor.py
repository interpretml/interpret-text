from abc import abstractmethod
from typing import Any, Iterable


class BaseTextPreprocessor:

    @abstractmethod
    def build_vocab(self, text: str) -> Any:
        raise NotImplementedError

    @abstractmethod
    def preprocess(self, data: Iterable[Any]) -> Iterable[Any]:
        raise NotImplementedError

    @abstractmethod
    def decode_single(self, id_list: Iterable[Any]) -> Iterable[Any]:
        raise NotImplementedError

    @abstractmethod
    def generate_tokens(self, sentence: str) -> Iterable[Any]:
        raise NotImplementedError

    def get_tokenizer(self) -> Any:
        raise NotImplementedError
