# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines the Explanation dashboard class."""

from .ExplanationWidget import ExplanationWidget
#from ._internal.constants import ExplanationDashboardInterface, WidgetRequestResponseConstants
from IPython.display import display
from scipy.sparse import issparse
import numpy as np
import pandas as pd


class ExplanationDashboard(object):
    """The dashboard class, wraps the dashboard component."""

    def __init__(self, explanation, model=None, sentence=None, trueY=None, classes=None, features=None):
        """Initialize the Explanation Dashboard for a single sentence.
        """
        self._widget_instance=ExplanationWidget()
        self._model=model
        self._sentence=sentence
        self.predicted_label=trueY
        self.classes=classes
        self.local_explanation=explanation

    def _show(self):
        display(self._widget_instance)