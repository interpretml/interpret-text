# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines the Explanation dashboard class."""

from .ExplanationWidget import ExplanationWidget
from IPython.display import display


class ExplanationDashboard(object):
    """The dashboard class, wraps the dashboard component."""

    def __init__(self, explanation):
        """Initialize the Explanation Dashboard for a single sentence."""
        classNames = explanation._classes.tolist()
        text = explanation._features
        explanation = explanation._local_importance_values.tolist()
        prediction = [i for i in range(len(classNames))]
        self._widget_instance = ExplanationWidget()
        self._widget_instance.value = {
            "text": text,
            "prediction": prediction,
            "classNames": classNames,
            "localExplanations": explanation,
        }
        display(self._widget_instance)

    def _show(self):
        display(self._widget_instance)
