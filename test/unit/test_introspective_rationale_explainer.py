# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

# Unit tests for model explanability SDK. Doesn't test visualization method

import os
import pandas as pd
import numpy as np
from utils_test import get_ssts_dataset
from interpret_text.introspective_rationale.introspective_rationale_explainer import IntrospectiveRationaleExplainer
from interpret_text.common.utils_introspective_rationale import ModelArguments, GlovePreprocessor, BertPreprocessor
from notebooks.test_utils.utils_data_shared import load_glove_embeddings

CUDA = False
MODEL_SAVE_DIR = os.path.join("..", "models")
model_prefix = "sst2rnpmodel"
DATA_FOLDER = "../../../data/sst2"
LABEL_COL = "labels"
TEXT_COL = "sentences"
SENTENCE = "This is a super amazing movie with bad acting"


class TestIntrospectiveRationaleExplainer(object):
    def test_working(self):
        assert True

    def test_explain_local_rnn(self):
        """
        Test explain_local with RNN model
        :return:
        """
        MODEL_TYPE = "RNN"

        token_count_thresh = 1
        max_sentence_token_count = 70

        args = ModelArguments(cuda=CUDA, model_save_dir=MODEL_SAVE_DIR, model_prefix=model_prefix)

        train_data = get_ssts_dataset('train')
        test_data = get_ssts_dataset('test')

        X_train = train_data[TEXT_COL]
        X_test = test_data[TEXT_COL]

        y_labels = test_data[LABEL_COL].unique()
        args.labels = np.array(sorted(y_labels))
        args.num_labels = len(y_labels)

        args.save_best_model = False
        args.pre_train_cls = False
        args.num_epochs = 1

        args.embedding_path = load_glove_embeddings(DATA_FOLDER)

        preprocessor = GlovePreprocessor(train_data[TEXT_COL], token_count_thresh, max_sentence_token_count)

        explainer = IntrospectiveRationaleExplainer(args, preprocessor, classifier_type=MODEL_TYPE)

        df_train = pd.concat([train_data[LABEL_COL], preprocessor.preprocess(X_train)], axis=1)
        df_test = pd.concat([test_data[LABEL_COL], preprocessor.preprocess(X_test)], axis=1)

        explainer.fit(df_train, df_test)

        local_explanation = explainer.explain_local(SENTENCE, preprocessor,
                                                    hard_importances=False)

        assert len(local_explanation.local_importance_values) == len(SENTENCE.split())

    def test_explain_local_bert(self):
        """
        Test explain_lodal with BERT
        :return:
        """
        MODEL_TYPE = "BERT"
        args = ModelArguments(cuda=CUDA, model_save_dir=MODEL_SAVE_DIR, model_prefix=model_prefix)

        train_data = get_ssts_dataset('train')
        test_data = get_ssts_dataset('test')

        X_train = train_data[TEXT_COL]
        X_test = test_data[TEXT_COL]

        y_labels = test_data[LABEL_COL].unique()
        args.labels = np.array(sorted(y_labels))
        args.num_labels = len(y_labels)

        args.embedding_path = load_glove_embeddings(DATA_FOLDER)

        preprocessor = BertPreprocessor()
        explainer = IntrospectiveRationaleExplainer(args, preprocessor, classifier_type=MODEL_TYPE)

        df_train = pd.concat([train_data[LABEL_COL], preprocessor.preprocess(X_train)], axis=1)
        df_test = pd.concat([test_data[LABEL_COL], preprocessor.preprocess(X_test)], axis=1)

        explainer.fit(df_train, df_test)
        local_explanation = explainer.explain_local(SENTENCE, preprocessor,
                                                    hard_importances=False)

        # BERT adds [CLS] at the beginning of a sentence and [SEP] at the end of each sentence.
        assert len(local_explanation.local_importance_values) == len(SENTENCE.split()) + 2



