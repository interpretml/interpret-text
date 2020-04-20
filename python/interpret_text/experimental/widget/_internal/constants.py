# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines the Constant strings."""


class ExplanationDashboardInterface(object):
    """Dictonary properties shared between the python and javascript object."""
    LOCAL_EXPLANATIONS = "localExplanations"
    TRAINING_DATA = "trainingData"
    IS_CLASSIFIER = "isClassifier"
    CLASS_NAMES = "classNames"
    PROBABILITY_Y = "probabilityY"
    HAS_MODEL = "has_model"


class WidgetRequestResponseConstants(object):
    """Strings used to pass messages between python and javascript."""
    ID = "id"
    DATA = "data"
    ERROR = "error"
    REQUEST = "request"
