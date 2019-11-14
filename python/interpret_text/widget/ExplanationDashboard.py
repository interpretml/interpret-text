# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines the Explanation dashboard class."""

from .ExplanationWidget import ExplanationWidget
from IPython.display import display


class ExplanationDashboard(object):
    """The dashboard class, wraps the dashboard component."""

    def __init__(self, explanation, model=None, text=None, prediction=None, classNames=None,):
        """Initialize the Explanation Dashboard for a single sentence."""
        self._widget_instance = ExplanationWidget()
        self._model = model
        self._text = text
        self._prediction = prediction
        self._classNames = classNames
        self._local_explanation = explanation

    def _show(self):
        display(self._widget_instance)
