# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

# import pytest

# Tests for model explainability SDK
# import shutil
import numpy as np
import logging
import shap

from interpret_text.text_explainer import TextExplainer

# from azureml.contrib.explain.model.explanation.explanation_client import ExplanationClient

# from azureml.contrib.explain.model.explanation.explanation_client import ExplanationClient

from common_utils import (
    create_binary_newsgroups_data,
    create_random_forest_tfidf,
    create_random_forest_vectorizer,
    create_linear_vectorizer,
    create_reviews_data,
)
from test_validate_explanations import validate_correlation

# from utilities.operations import sdk
# from utilities.constants import owner_email_tools_and_ux

test_logger = logging.getLogger(__name__)
test_logger.setLevel(logging.INFO)


# @pytest.mark.owner(email=owner_email_tools_and_ux)
# @pytest.mark.usefixtures("clean_dir")
class TestTextExplainer(object):
    def test_working(self):
        assert True

    # @pytest.mark.skip(reason='Using old upload/download')
    def test_explain_model_with_run(self):
        newsgroups_train, newsgroups_test, classes = create_binary_newsgroups_data()
        pipeline = create_random_forest_tfidf()
        pipeline.fit(newsgroups_train.data, newsgroups_train.target)
        # test_logger.info("Logging with experiment: {} and workspace {}".format(self.experiment_name,
        #                                                                       shared_workspace.name))
        # run = sdk.run.create(shared_workspace, self.experiment_name)
        # client = ExplanationClient.from_run(run)
        # run_id = run.id
        # test_logger.info("Created run id to pass to TextExplainer: {}".format(run_id))
        # exp = TextExplainer(pipeline)
        test_logger.info("Running explain model for test_explain_model_with_run_id")
        # The chosen example here comes from main LIME github page
        # note: we use 83:84 instead of just 83 to pass 2-D array to explain_local
        # explanation = exp.explain_local(newsgroups_test.data[83:84], classes=classes)
        # client.upload_model_explanation(explanation)
        # local_importance_values = explanation.local_importance_values
        # expected_values = explanation.expected_values

        test_logger.info("Getting feature importance values")
        # Get model exp data
        # dl_explanation = client.download_model_explanation()
        # Assert equal in local and remote locations
        # np.testing.assert_array_equal(local_importance_values, dl_explanation.local_importance_values)
        # np.testing.assert_array_equal(expected_values, dl_explanation.expected_values)
        # np.testing.assert_array_equal(classes, dl_explanation.classes)
        # cleanup_dirs = ["upload_explanation", "download_explanation"]
        # # Cleanup dirs used for explanation
        # for cdir in cleanup_dirs:
        #     shutil.rmtree(cdir, ignore_errors=True)

    def test_explain_model_local(self):
        newsgroups_train, newsgroups_test, classes = create_binary_newsgroups_data()
        pipeline = create_random_forest_tfidf()
        pipeline.fit(newsgroups_train.data, newsgroups_train.target)

        # Create local text explainer without run history
        exp = TextExplainer(pipeline)
        # The chosen example here comes from main LIME github page
        # note: we use 83:84 instead of just 83 to pass 2-D array to explain_local
        exp.explain_local(newsgroups_test.data[83:84], classes=classes)

    def test_explain_model_regression(self):
        # Verify that evaluation dataset can be downsampled and the scoring scenario can still be run
        x_train, x_test, y_train, _ = create_reviews_data(0.2)
        # Fit a linear regression model
        pipeline = create_linear_vectorizer()
        pipeline.fit(x_train, y_train)
        # Create TextExplainer without run history
        exp = TextExplainer(pipeline)
        test_logger.info("Running explain model for test_explain_model_regression")
        exp.explain_local(x_test)

    def test_validate_text_explanation(self):
        np.random.seed(seed=777)
        newsgroups_train, newsgroups_test, classes = create_binary_newsgroups_data()
        pipeline = create_random_forest_vectorizer()
        pipeline.fit(newsgroups_train.data, newsgroups_train.target)
        rf = pipeline.named_steps["rf"]
        vectorizer = pipeline.named_steps["vectorizer"]
        # The chosen example here comes from main LIME github page
        # note: we use 83:84 instead of just 83 to pass 2-D array to explain_local
        test_row = newsgroups_test.data[83:84]
        # Use tree explainer to get most accurate shap values
        exp = shap.TreeExplainer(rf)
        vec_row = vectorizer.transform(test_row)
        # get mapping from words to indexes
        tree_vocab = vectorizer.vocabulary_
        num_cols = vec_row.shape[1]
        ones = np.ones((1, num_cols))
        # This is explanation across all cols in training data so it is quite large
        tree_explanation = np.abs(exp.shap_values(ones))

        # Create local text explainer without run history
        exp = TextExplainer(pipeline)
        text_explanation = exp.explain_local(test_row, classes=classes)
        ordered_local_importance_values = (
            text_explanation.ordered_local_importance_values
        )
        imp = text_explanation.local_importance_rank
        # validate for each class in explanation
        for c in range(len(tree_explanation)):
            imp_pc = imp[c][0]
            # get the intersection of vocabularies
            vocab_intersect = np.array(
                list(set(tree_vocab.keys()).intersection(set(imp_pc)))
            )
            tree_vocab_indexes = [
                tree_vocab[vocab_word] for vocab_word in vocab_intersect
            ]
            text_vocab_indexes = [
                imp_pc.index(vocab_word) for vocab_word in vocab_intersect
            ]
            # reduce tree_explanation to be same as our TextExplainer explanation
            reduced_tree_explanation = tree_explanation[c][0, tree_vocab_indexes]
            # find the sorted indexes by shap values
            reduced_tree_imp = reduced_tree_explanation.argsort()[..., ::-1]
            text_imp = np.array(ordered_local_importance_values[c])[
                0, text_vocab_indexes
            ].argsort()[..., ::-1]
            # calculate correlation between text explanation and reduced tree explanation
            # Note: there are a lot of ties with 0 shap values from run to run that we can ignore
            validate_correlation(reduced_tree_imp, text_imp, 0.6)
