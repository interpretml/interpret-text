import torch
from torch import nn
from torch import optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from .common.structured_model_mixin import PureStructuredModelMixin
from interpret_community.common.base_explainer import LocalExplainer


class MSRAExplainer(PureStructuredModelMixin):
    """The MSRAExplainer for returning explanations for deep neural network models.

    :param model: The tree model to explain.
    :type model: bert, xlnet or pytorch NN model
    """
    def __init__(self, model):
        """Initialize the MSRAExplainer.

        :param model: The tree model to explain.
        :type model:  bert, xlnet or pytorch NN model
        """
        pass

    def explain_local(self, evaluation_examples):
        """Explain the model by using msra's interpretor

        :param evaluation_examples: A matrix of feature vector examples (# examples x # features) on which
            to explain the model's output.
        :type evaluation_examples: DatasetWrapper
        :return: A model explanation object. It is guaranteed to be a LocalExplanation
        """
        pass
