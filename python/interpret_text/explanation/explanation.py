# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines the explanations that are returned from explaining models."""

import numpy as np
import uuid

from interpret_community.common.explanation_utils import _sort_values, _order_imp
from interpret_community.common.constants import Dynamic, ExplanationParams
from interpret_community.common.constants import ExplainParams
from interpret_community.explanation.explanation import _get_aggregate_kwargs, \
    _create_global_explanation_kwargs
from interpret_community.explanation.explanation import LocalExplanation, ExpectedValuesMixin, \
    ClassesMixin


class TextExplanation(LocalExplanation):
    """Defines the mixin for text explanations."""

    def __init__(self, **kwargs):
        """Create the text explanation."""
        super(TextExplanation, self).__init__(**kwargs)
        order = _order_imp(np.abs(self.local_importance_values))
        self._local_importance_rank = _sort_values(self._features, order)
        self._logger.debug('Initializing TextExplanation')
        if len(order.shape) == 3:
            i = np.arange(order.shape[0])[:, np.newaxis]
            j = np.arange(order.shape[1])[:, np.newaxis]
            self._ordered_local_importance_values = np.array(self.local_importance_values)[i, j, order]
        else:
            self._ordered_local_importance_values = self.local_importance_values

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
        if not hasattr(explanation, ExplainParams.LOCAL_IMPORTANCE_RANK) or explanation.local_importance_rank is None:
            return False
        if (not hasattr(explanation, ExplainParams.ORDERED_LOCAL_IMPORTANCE_VALUES) or
                explanation.ordered_local_importance_values is None):
            return False
        return True


class ImageExplanation(LocalExplanation):
    """Defines the mixin for image explanations."""

    def __init__(self, image_segments=None, **kwargs):
        """Create the image explanation.

        :param image_segments: List of image segmentations.
        :type image_segments: list[numpy.array]
        """
        super(ImageExplanation, self).__init__(**kwargs)
        self._logger.debug('Initializing ImageExplanation')
        self._image_segments = image_segments

    @property
    def image_segments(self):
        """Get a list of image segmentations.

        :return: A list of image segmentations.
        :rtype: list[numpy.array]
        """
        return self._image_segments

    @classmethod
    def _does_quack(cls, explanation):
        """Validate that the explanation object passed in is a valid ImageExplanation.

        :param explanation: The explanation to be validated.
        :type explanation: object
        :return: True if valid else False
        :rtype: bool
        """
        if not super()._does_quack(explanation):
            return False
        return True


def _create_local_explanation(expected_values=None, classification=True,
                              text_explanation=False, image_explanation=False,
                              explanation_id=None, **kwargs):
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
    elif image_explanation:
        mixins = [ImageExplanation]
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


def _create_global_explanation(local_explanation=None, expected_values=None,
                               classification=True,
                               text_explanation=False, explanation_id=None, **kwargs):
    """Dynamically creates an explanation based on global type and specified data.

    :param local_explanation: The local explanation information to include with global,
        can be done when the global explanation is a summary of local explanations.
    :type local_explanation: LocalExplanation
        :param expected_values: The expected values of the model.
        :type expected_values: list
    :param classification: Indicates if this is a classification or regression explanation.
    :type classification: bool
    :param text_explanation: Indicates if this is a text explanation.
    :type text_explanation: bool
    :param explanation_id: If specified, puts the global explanation under a preexisting explanation object.
        If not, a new unique identifier will be created for the explanation.
    :type explanation_id: str
    :return: A model explanation object. It is guaranteed to be a GlobalExplanation. If local_explanation is not None,
        it will also have the properties of a LocalExplanation. If expected_values is not None, it will have the
        properties of ExpectedValuesMixin. If classification is set to True, it will have the properties of
        ClassesMixin, and if a local explanation was passed in it will also have the properties of PerClassMixin. If
        text_explanation is set to True, it will have the properties of TextExplanation.
    :rtype: DynamicGlobalExplanation
    """
    kwargs, mixins = _create_global_explanation_kwargs(local_explanation, expected_values,
                                                       classification, explanation_id, **kwargs)
    DynamicGlobalExplanation = type(Dynamic.GLOBAL_EXPLANATION, tuple(mixins), {})
    global_explanation = DynamicGlobalExplanation(**kwargs)
    return global_explanation


def _aggregate_global_from_local_explanation(local_explanation=None, include_local=True,
                                             features=None, explanation_id=None, **kwargs):
    """Aggregate the local explanation information to global through averaging.

    :param local_explanation: The local explanation to summarize.
    :type local_explanation: LocalExplanation
    :param include_local: Whether the global explanation should also include local information.
    :type include_local: bool
    :param features: A list of feature names.
    :type features: list[str]
    :param explanation_id: If specified, puts the aggregated explanation under a preexisting explanation object.
        If not, a new unique identifier will be created for the explanation.
    :type explanation_id: str
    :return: A model explanation object. It is guaranteed to be a GlobalExplanation. If include_local is set to True,
        it will also have the properties of a LocalExplanation. If expected_values exists on local_explanation, it
        will have the properties of ExpectedValuesMixin. If local_explanation has ClassesMixin, it will have the
        properties of PerClassMixin.
    :rtype: DynamicGlobalExplanation
    """
    kwargs = _get_aggregate_kwargs(local_explanation, include_local, features, explanation_id, **kwargs)
    return _create_global_explanation(**kwargs)
