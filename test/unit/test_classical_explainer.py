# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

# Tests for classical explainer
from interpret_text.experimental.classical import ClassicalTextExplainer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from utils_test import get_mnli_test_dataset

DOCUMENT = "rare bird has more than enough charm to make it memorable."


class TestClassicalExplainer(object):
    def test_working(self):
        assert True

    def test_explain_model_local_default(self):
        """
        Test for explain_local of classical explainer
        :return:
        """
        train_df = get_mnli_test_dataset('train')
        X_str = train_df['sentence1']
        ylabels = train_df['genre']
        X_train, X_test, y_train, y_test = train_test_split(X_str, ylabels, train_size=0.8, test_size=0.2)

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
        train_df = get_mnli_test_dataset('train')
        X_str = train_df['sentence1']
        ylabels = train_df['genre']
        X_train, X_test, y_train, y_test = train_test_split(X_str, ylabels, train_size=0.8, test_size=0.2)

        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        explainer = ClassicalTextExplainer()
        classifier, best_params = explainer.fit(X_train, y_train)
        explainer.preprocessor.labelEncoder = label_encoder

        y = classifier.predict(DOCUMENT)
        predicted_label = label_encoder.inverse_transform(y)
        
        local_explanation = explainer.explain_local(DOCUMENT, predicted_label)
        assert len(local_explanation.local_importance_values) == len(local_explanation.features)

