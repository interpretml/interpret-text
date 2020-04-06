# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines the base text explainer API to create explanations."""
from abc import abstractmethod, ABC
from typing import Optional, Any, Iterable

from interpret_text.common.base_text_model import BaseTextModel
from interpret_text.common.preprocessor.base_text_preprocessor import BaseTextPreprocessor
from interpret_text.explanation.explanation import _create_local_explanation


class BaseTextExplainer(ABC):
    """ The base class for explainers to create explanataion """

    @abstractmethod
    def explain_local(self, text: str, **kwargs) -> _create_local_explanation:
        """Abstract method to explain local features

        :param text: text required for local feature importance
        :type str
        :param kwargs: additional parameters
        :type Any
        :return: A local explanation object
        :rtype DynamicLocalExplanation
        """
        pass

    @abstractmethod
    def fit(self, X: Iterable[Any], y: Optional[Iterable[Any]]) -> BaseTextModel:
        """ Fit model for a given explainer

        :param X: Iterable
        :type X: Iterable[Any]
        :param y
        :type y: Iterable[Any]
        :return: A base txt model object
        :rtype BaseTextModel
        """
        pass

    @abstractmethod
    def get_model(self) -> BaseTextModel:
        """ Abstract method to get explainer model

        :return: A model of type BaseTextModel
        :rtype: BaseTextModel
        """
        pass

    @abstractmethod
    def set_model(self, **kwargs):
        """ Abstract method to set model for the explainer

        :param kwargs: additional parameters
        :type Any
        """
        pass

    @abstractmethod
    def set_preprocessor(self, preprocessor: BaseTextPreprocessor):
        """ Abstract method to set processor for the explainer

        :param preprocessor: Provided preprocessor
        :type BaseTextPreprocessor
        """
        pass

    @abstractmethod
    def get_preprocessor(self)-> BaseTextPreprocessor:
        """ Abstract method to return processor

        :return: Preprocessor of the explainer
        :type BaseTextPreprocessor
        """
        pass
