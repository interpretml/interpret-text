# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import pytest

# Tests for model explainability SDK
import numpy as np
from scipy import stats
import shap
import logging
from sklearn.pipeline import Pipeline

# from azureml.explain.model.tabular_explainer import TabularExplainer
from common_utils import create_sklearn_random_forest_classifier, \
    create_sklearn_random_forest_regressor, create_sklearn_linear_regressor, \
    create_sklearn_logistic_regressor
from sklearn.model_selection import train_test_split

test_logger = logging.getLogger(__name__)
test_logger.setLevel(logging.INFO)

def tabular_explainer_imp(model, x_train, x_test, allow_eval_sampling=True):
    # Create local tabular explainer without run history
    exp = TabularExplainer(model, x_train, features=list(range(x_train.shape[1])))
    # Validate evaluation sampling
    policy = {ExplainParams.SAMPLING_POLICY: SamplingPolicy(allow_eval_sampling=allow_eval_sampling)}
    explanation = exp.explain_global(x_test, **policy)
    return explanation.global_importance_rank


# TODO: remove this and replace with current contrib method once azureml-contrib-explain-model moved to release
def dcg(true_order_relevance, validate_order, top_values=10):
    # retrieve relevance score for each value in validation order
    relevance = np.vectorize(lambda x: true_order_relevance.get(x, 0))(validate_order[:top_values])
    gain = 2 ** relevance - 1
    discount = np.log2(np.arange(1, len(gain) + 1) + 1)
    sum_dcg = np.sum(gain / discount)
    return sum_dcg


# TODO: remove this and replace with current contrib method once azureml-contrib-explain-model moved to release
def validate_correlation(true_order, validate_order, threshold, top_values=10):
    # Create map from true_order to "relevance" or reverse order index
    true_order_relevance = {}
    num_elems = len(true_order)
    for index, value in enumerate(true_order):
        # Set the range of the relevance scores to be between 0 and 10
        # This is to prevent very large values when computing 2 ** relevance - 1
        true_order_relevance[value] = ((num_elems - index) / float(num_elems)) * 10.0
    # See https://en.wikipedia.org/wiki/Discounted_cumulative_gain for reference
    dcg_p = dcg(true_order_relevance, validate_order, top_values)
    idcg_p = dcg(true_order_relevance, true_order, top_values)
    ndcg = dcg_p / idcg_p
    test_logger.info("ndcg: " + str(ndcg))
    assert(ndcg > threshold)


def validate_spearman_correlation(overall_imp, shap_overall_imp, threshold):
    # Calculate the spearman rank-order correlation
    rho, p_val = stats.spearmanr(overall_imp, shap_overall_imp)
    # Validate that the coefficients from the linear model are highly correlated with the results from shap
    test_logger.info("Calculated spearman correlation coefficient rho: " + str(rho) + " and p_val: " + str(p_val))
    assert(rho > threshold)


def get_shap_imp_classification(explanation):
    global_importance_values = np.mean(np.mean(np.absolute(explanation), axis=1), axis=0)
    return global_importance_values.argsort()[..., ::-1]


def get_shap_imp_regression(explanation):
    global_importance_values = np.mean(np.absolute(explanation), axis=0)
    return global_importance_values.argsort()[..., ::-1]
