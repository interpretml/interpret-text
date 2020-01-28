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
        local_importance_values = list(explanation._local_importance_values)
        classNames = list(explanation._classes)
        text = explanation._features
        prediction = [
            0 if name == explanation._predicted_label else 1 for name in classNames
        ]
        # ground_truth = explanation._true_label
        self._widget_instance = ExplanationWidget()
        self._widget_instance.value = {
            "text": text,
            "prediction": prediction,
            "classNames": classNames,
            "localExplanations": local_importance_values,
        }
        display(self._widget_instance)

    def _show(self):
        display(self._widget_instance)
