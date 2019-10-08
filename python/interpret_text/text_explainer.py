# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines the text explainer for getting model explanations from text data."""

import numpy as np
import re

from azureml.explain.model.common.blackbox_explainer import BlackBoxExplainer
from azureml.explain.model.common.explanation_utils import _convert_to_list
from azureml.contrib.explain.model.explanation.explanation import _create_local_explanation
from .common.text_explainer_utils import _find_golden_doc
from azureml.explain.model.common.constants import Attributes, ExplainParams, ExplainType

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', 'Starting from version 2.2.1', UserWarning)
    import shap


class TextExplainer(BlackBoxExplainer):
    """Explain a model trained on a text dataset."""

    def __init__(self, model, is_function=False, **kwargs):
        """Initialize the Text Explainer.

        :param model: An object that represents a model. It is assumed that for the classification case
            it has a method of predict_proba() returning the prediction probabilities for each
            class and for the regression case a method of predict() returning the prediction value.
        :type model: object
        :param is_function: Default set to false, set to True if passing sklearn.predict or sklearn.predict_proba
            function instead of model.
        :type is_function: bool
        """
        super(TextExplainer, self).__init__(model, is_function=is_function, **kwargs)
        self._logger.debug('Initializing TextExplainer')
        self._method = 'text'

    def _explain_instance(self, row, function):
        """Explains the best example row with text data."""
        from sklearn.feature_extraction.text import CountVectorizer
        rowArr = [row]
        vectorizer = CountVectorizer(lowercase=False, min_df=0.0, binary=True)

        # convert input data to numeric data via term frequency
        numeric_vals = vectorizer.fit_transform(rowArr)
        zeros = np.zeros(numeric_vals.shape)
        ones = np.ones(numeric_vals.shape)
        features = np.array(vectorizer.get_feature_names())

        # convert from numeric back to text (inverse) and then predict it
        def from_numeric_predict(data):
            textData = vectorizer.inverse_transform(1 - data)
            # for given row of text data, we tokenize/preprocess it and then we filter out the
            # tokens that are not in the inverse-transformed text data generated from shap samples
            filteredData = []
            for textDataPermuted in textData:
                filteredData.append(''.join(filter(lambda x: x not in textDataPermuted, re.split('(\W+)', row))))
            prediction = function(filteredData)
            return prediction

        exp = shap.KernelExplainer(from_numeric_predict, zeros)
        shap_values = exp.shap_values(ones)
        return shap_values, exp, features

    def explain_global(self, evaluation_examples, classes=None):
        """Explain a model by explaining its predictions on the text document.

        Global explanations are currently not supported, we just return a local explanation
        for explain_global instead on a chosen golden document.
        If multiple documents are passed, we choose the one with the highest predicted probability
        or confidence from the given evaluation examples.

        :param evaluation_examples: A list of text documents.
        :type evaluation_examples: list
        :param classes: Class names, in any form that can be converted to an array of str. This includes
            lists, lists of tuples, tuples, tuples of tuples, tuples of lists and ndarrays. The order of
            the class names should match that of the model output.
        :type classes: array_like[str]
        :return: A local explanation of the text document containing the feature importance values,
            expected values and the chosen golden document with highest confidence score in model.
        :rtype: LocalExplanation
        """
        kwargs = {}
        if classes is not None:
            kwargs[ExplainParams.CLASSES] = classes
        return self.explain_local(evaluation_examples, **kwargs)

    def explain_local(self, evaluation_examples, classes=None):
        """Explain a model locally by explaining its predictions on the text document.

        If multiple documents are passed, we choose the one with the highest predicted probability
        or confidence from the given evaluation examples.

        :param evaluation_examples: A list of text documents.
        :type evaluation_examples: list
        :param classes: Class names, in any form that can be converted to an array of str. This includes
            lists, lists of tuples, tuples, tuples of tuples, tuples of lists and ndarrays. The order of
            the class names should match that of the model output.
        :type classes: array_like[str]
        :return: A local explanation of the text document containing the feature importance values,
            expected values and the chosen golden document with highest confidence score in model.
        :rtype: LocalExplanation
        """
        # find document with highest probability
        golden_doc = _find_golden_doc(self.function, evaluation_examples)
        # shap values for instance with highest prob
        shap_values, explainer, features = self._explain_instance(golden_doc, self.function)
        shap_values = np.asarray(shap_values)
        if explainer is not None and hasattr(explainer, Attributes.EXPECTED_VALUE):
            expected_values = explainer.expected_value
            if isinstance(expected_values, np.ndarray):
                expected_values = expected_values.tolist()
        local_importance_values = _convert_to_list(shap_values)
        features = features.tolist()
        kwargs = {ExplainParams.METHOD: ExplainType.SHAP}
        if classes is not None:
            kwargs[ExplainParams.CLASSES] = classes
        if self.predict_proba_flag:
            kwargs[ExplainType.MODEL_TASK] = ExplainType.CLASSIFICATION
        else:
            kwargs[ExplainType.MODEL_TASK] = ExplainType.REGRESSION
        if self.model is not None:
            kwargs[ExplainParams.MODEL_TYPE] = str(type(self.model))
        else:
            kwargs[ExplainParams.MODEL_TYPE] = ExplainType.FUNCTION
        return _create_local_explanation(local_importance_values=np.array(local_importance_values),
                                         expected_values=expected_values, text_explanation=True,
                                         features=features, **kwargs)
