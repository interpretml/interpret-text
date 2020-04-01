# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines the base text explainer API to create explanations."""
from abc import abstractmethod
from typing import Optional, Any, Callable, Iterable

from interpret_text.common.base_text_model import BaseTextModel
from interpret_text.common.preprocessor.base_text_preprocessor import BaseTextPreprocessor


class BaseTextExplainer:

    @abstractmethod
    def explain_local(self, text: str, **kwargs) -> Callable:
        raise NotImplementedError

    @abstractmethod
    def fit(self, X: Iterable[Any], y: Optional[Iterable[Any]]) -> BaseTextModel:
        raise NotImplementedError

    @abstractmethod
    def get_model(self, **kwargs) -> Callable:
        raise NotImplementedError

    @abstractmethod
    def get_preprocessor(self, classifier_type: str= "RNN", **kwargs) -> BaseTextPreprocessor:
        raise NotImplementedError
