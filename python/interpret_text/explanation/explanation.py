# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines the explanations that are returned from explaining models."""

import numpy as np
import uuid

from interpret_community.common.explanation_utils import _sort_values, _order_imp
from interpret_community.common.constants import Dynamic, ExplanationParams
from interpret_community.common.constants import ExplainParams
from interpret_community.explanation.explanation import (
    LocalExplanation,
    ExpectedValuesMixin,
    ClassesMixin,
)


class TextExplanation(LocalExplanation):
    """Defines the mixin for text explanations."""

    def __init__(self, predicted_label=None, true_label=None, **kwargs):
        """Create the text explanation.
        :param predicted_label: The label predicted by the classifier
        :type predicted_label: string
        :param true_label: The ground truth label for the sentense
        :type true_label: string
        """
        super(TextExplanation, self).__init__(**kwargs)
        order = _order_imp(np.abs(self.local_importance_values))
        self._local_importance_rank = _sort_values(self._features, order)
        self._predicted_label = predicted_label
        self._true_label = true_label
        self._logger.debug("Initializing TextExplanation")
        if len(order.shape) == 3:
            i = np.arange(order.shape[0])[:, np.newaxis]
            j = np.arange(order.shape[1])[:, np.newaxis]
            self._ordered_local_importance_values = np.array(
                self.local_importance_values
            )[i, j, order]
        else:
            self._ordered_local_importance_values = self.local_importance_values

    @property
    def predicted_label(self):
        """Get the task of the original model, i.e. classification or regression (others possibly in the future).

        :return: The task of the original model.
        :rtype: str
        """
        return self._predicted_label

    @property
    def local_importance_rank(self):
        """Feature names sorted by importance.

        This property exists for text explanations only and not for local because currently
        we are doing text explanations for a single document and it is more difficult to
        define order for multiple instances.  Note this is subject to change if we eventually
        add global explanations for text explainers.

        :return: The feature names sorted by importance.
        :rtype: list
        """
        return self._local_importance_rank.tolist()

    @property
    def ordered_local_importance_values(self):
        """Get the feature importance values ordered by importance.

        This property exists for text explanations only and not for local because currently
        we are doing text explanations for a single document and it is more difficult to
        define order for multiple instances.  Note this is subject to change if we eventually
        add global explanations for text explainers.

        :return: For a model with a single output such as regression, this
            returns a list of feature importance values. For models with vector outputs this function
            returns a list of such lists, one for each output. The dimension of this matrix
            is (# examples x # features).
        :rtype: list
        """
        return self._ordered_local_importance_values

    @classmethod
    def _does_quack(cls, explanation):
        """Validate that the explanation object passed in is a valid TextExplanation.

        :param explanation: The explanation to be validated.
        :type explanation: object
        :return: True if valid else False
        :rtype: bool
        """
        if not super()._does_quack(explanation):
            return False
        if (
            not hasattr(explanation, ExplainParams.LOCAL_IMPORTANCE_RANK)
            or explanation.local_importance_rank is None
        ):
            return False
        if (
            not hasattr(explanation, ExplainParams.ORDERED_LOCAL_IMPORTANCE_VALUES)
            or explanation.ordered_local_importance_values is None
        ):
            return False
        return True


def _create_local_explanation(
    expected_values=None,
    classification=True,
    text_explanation=False,
    image_explanation=False,
    explanation_id=None,
    **kwargs
):
    """Dynamically creates an explanation based on local type and specified data.

    :param expected_values: The expected values of the model.
    :type expected_values: list
    :param classification: Indicates if this is a classification or regression explanation.
    :type classification: bool
    :param text_explanation: Indicates if this is a text explanation.
    :type text_explanation: bool
    :param image_explanation: Indicates if this is an image explanation.
    :type image_explanation: bool
    :param explanation_id: If specified, puts the local explanation under a preexisting explanation object.
        If not, a new unique identifier will be created for the explanation.
    :type explanation_id: str
    :return: A model explanation object. It is guaranteed to be a LocalExplanation. If expected_values is not None, it
        will also have the properties of the ExpectedValuesMixin. If classification is set to True, it will have the
        properties of the ClassesMixin. If text_explanation is set to True, it will have the properties of
        TextExplanation.
    :rtype: DynamicLocalExplanation
    """
    exp_id = explanation_id or str(uuid.uuid4())
    if text_explanation:
        mixins = [TextExplanation]
    else:
        mixins = [LocalExplanation]
    if expected_values is not None:
        mixins.append(ExpectedValuesMixin)
        kwargs[ExplanationParams.EXPECTED_VALUES] = expected_values
    if classification:
        mixins.append(ClassesMixin)
    DynamicLocalExplanation = type(Dynamic.LOCAL_EXPLANATION, tuple(mixins), {})
    local_explanation = DynamicLocalExplanation(explanation_id=exp_id, **kwargs)
    return local_explanation
