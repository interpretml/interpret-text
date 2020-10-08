# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

# Tests for classical explainer
from interpret_text.experimental.classical import ClassicalTextExplainer
from sklearn.preprocessing import LabelEncoder
from utils_test import setup_mnli_test_train_split
import pickle
import os

DOCUMENT = "rare bird has more than enough charm to make it memorable."


class TestClassicalExplainer(object):
    def test_working(self):
        assert True

    def test_explain_model_local_default(self):
        """
        Test for explain_local of classical explainer
        :return:
        """
        X_train, X_test, y_train, y_test = setup_mnli_test_train_split()
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        explainer = ClassicalTextExplainer()
        classifier, best_params = explainer.fit(X_train, y_train)
        explainer.preprocessor.labelEncoder = label_encoder

        local_explanation = explainer.explain_local(DOCUMENT)
        assert len(local_explanation.local_importance_values) == len(local_explanation.features)

    def test_explain_model_local_with_predicted_label(self):
        """
        Test for explain_local of classical explainer
        :return:
        """
        X_train, X_test, y_train, y_test = setup_mnli_test_train_split()

        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        explainer = ClassicalTextExplainer()
        classifier, best_params = explainer.fit(X_train, y_train)
        explainer.preprocessor.labelEncoder = label_encoder
        y = classifier.predict(DOCUMENT)
        predicted_label = label_encoder.inverse_transform(y)
        local_explanation = explainer.explain_local(DOCUMENT, predicted_label)
        assert len(local_explanation.local_importance_values) == len(local_explanation.features)

    def test_pickle(self, tmpdir):
        """
        Test for pickling of classical explainer
        :return:
        """
        X_train, X_test, y_train, y_test = setup_mnli_test_train_split()
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        explainer = ClassicalTextExplainer()
        classifier, best_params = explainer.fit(X_train, y_train)
        explainer_file_name = 'explainer.pkl'
        explainer_file_path = tmpdir.mkdir('explainers').join(explainer_file_name)
        with open(explainer_file_path, 'wb') as explainer_save_file:
            pickle.dump(explainer, explainer_save_file)
        with open(explainer_file_path, 'rb') as explainer_load_file:
            explainer = pickle.load(explainer_load_file)
        os.remove(explainer_file_path)
        explainer.preprocessor.labelEncoder = label_encoder
        local_explanation = explainer.explain_local(DOCUMENT)
        assert len(local_explanation.local_importance_values) == len(local_explanation.features)