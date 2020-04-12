# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines utilities common to the text explainers."""

import math


def _find_golden_doc(function, evaluation_examples):
    highest_prob_value = -math.inf
    highest_prob_index = -1
    # Find example with highest predicted prob in classification case
    # or highest prediction in regression case
    for index, row in enumerate(evaluation_examples):
        rowArr = [row]
        prediction = function(rowArr)
        if len(prediction.shape) == 2:
            prediction = prediction[0]
        # TODO: Change this to calculate multiple pred_max for each class prediction
        pred_max = max(prediction)
        if pred_max > highest_prob_value:
            highest_prob_value = pred_max
            highest_prob_index = index
    return evaluation_examples[highest_prob_index]
