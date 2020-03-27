# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

# Tests for model explainability SDK
from notebooks.test_utils.utils_mnli import load_mnli_pandas_df
from interpret_text.classical.classical_text_explainer import ClassicalTextExplainer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from utils_test import get_mnli_test_dataset

DOCUMENT = "rare bird has more than enough charm to make it memorable."

class TestClassicalExplainer(object):
    def test_working(self):
        assert True

    def test_explain_model_local(self):
        # test data
        train_df = get_mnli_test_dataset('train')
        X_str = train_df['sentence1']
        ylabels = train_df['genre']

        #DATA_FOLDER = "./temp"
        #df = load_mnli_pandas_df(DATA_FOLDER, "train")
        #df = df[df["gold_label"] == "neutral"]  # get unique sentences

        # fetch documents and labels from data frame
        #X_str = df['sentence1'][:50]  # the document we want to analyze
        #ylabels = df['genre'][:50]  # the labels, or answers, we want to test against

        X_train, X_test, y_train, y_test = train_test_split(X_str, ylabels, train_size=0.8, test_size=0.2)

        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)

        # Create explainer
        explainer = ClassicalTextExplainer()

        classifier, best_params = explainer.fit(X_train, y_train)
        explainer.preprocessor.labelEncoder = label_encoder

        local_explanantion = explainer.explain_local(DOCUMENT)
        assert len(local_explanantion.local_importance_values) == len(local_explanantion.features)
        assrt len(local_explanantion.classes) == len(classifier.)










