"""Defines constants for interpret community text."""


class ExplainerParams(object):
    """Provide constants for explainer parameters."""

    HYPERPARAM_RANGE = {
        "solver": ["saga"],
        "multi_class": ["multinomial"],
        "C": [10 ** 4],
    }
