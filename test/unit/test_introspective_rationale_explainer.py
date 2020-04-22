import pandas as pd
from interpret_text.experimental.common.preprocessor.bert_preprocessor import BertPreprocessor
from interpret_text.experimental.common.preprocessor.glove_preprocessor import GlovePreprocessor
from interpret_text.experimental.introspective_rationale.introspective_rationale_components import ClassifierModule
from interpret_text.experimental.introspective_rationale.introspective_rationale_explainer import IntrospectiveRationaleExplainer
from model_config_constants import RNN_MODEL_CONFIG, BERT_MODEL_CONFIG
from utils_test import get_ssts_dataset, setup_mock_rnn_introspective_rationale_explainer, setup_mock_bert_introspective_rationale_explainer

CLASSIFIER_TYPE_BERT = "BERT"
CLASSIFIER_TYPE_BERT_RNN = "BERT_RNN"
CLASSIFIER_TYPE_RNN = "RNN"
CLASSIFIER_TYPE_CUSTOM = "CUSTOM"
CUDA = False
TEXT_COL = "sentences"
LABEL_COL = "labels"
TOKEN_COUNT_THRESHOLD = 1
MAX_SENT_COUNT = 70
SENTENCE = "This is a super amazing movie with bad acting"


class TestIntrospectiveRationaleExplainer(object):
    def test_working(self):
        assert True

    def test_set_bert_classifier(self):
        model_config = BERT_MODEL_CONFIG
        explainer = setup_mock_bert_introspective_rationale_explainer(model_config=model_config)
        explainer.set_classifier(CLASSIFIER_TYPE_BERT)

    def test_set_rnn_classifier(self):
        all_data = get_ssts_dataset('train')
        all_data = all_data[TEXT_COL]
        model_config = RNN_MODEL_CONFIG
        explainer = setup_mock_rnn_introspective_rationale_explainer(model_config=model_config, data=all_data)
        explainer.set_classifier(CLASSIFIER_TYPE_BERT)

    def test_set_custom_classifier(self):
        all_data = get_ssts_dataset('train')
        model_config = RNN_MODEL_CONFIG
        preprocessor = GlovePreprocessor(TOKEN_COUNT_THRESHOLD, MAX_SENT_COUNT)
        preprocessor.build_vocab(all_data[TEXT_COL])
        explainer = IntrospectiveRationaleExplainer(classifier_type=CLASSIFIER_TYPE_CUSTOM,cuda=CUDA)
        explainer.build_model_config(model_config)
        # set custom classifier
        classifier = ClassifierModule(explainer.get_model_config(), preprocessor.word_vocab)
        explainer.set_preprocessor(preprocessor)
        explainer.set_classifier(classifier)
        assert explainer.classifier_type is CLASSIFIER_TYPE_CUSTOM

    def test_set_bert_rnn_classifier(self):
        pass

    def test_set_anti_classifier(self):
        all_data = get_ssts_dataset('train')
        all_data = all_data[TEXT_COL]
        model_config = RNN_MODEL_CONFIG
        explainer = setup_mock_rnn_introspective_rationale_explainer(model_config=model_config, data=all_data)
        explainer.set_anti_classifier(classifier_type=CLASSIFIER_TYPE_RNN)

    def test_set_custom_anti_classifier(self):
        model_config = BERT_MODEL_CONFIG
        explainer = setup_mock_bert_introspective_rationale_explainer(model_config=model_config)
        explainer.set_anti_classifier(classifier_type=CLASSIFIER_TYPE_BERT)

    def test_set_generator_classifier(self):
        all_data = get_ssts_dataset('train')
        all_data = all_data[TEXT_COL]
        model_config = RNN_MODEL_CONFIG
        explainer = setup_mock_rnn_introspective_rationale_explainer(model_config=model_config, data=all_data)
        explainer.set_generator_classifier(classifier_type=CLASSIFIER_TYPE_RNN)

    def test_set_custom_generator_classifier(self):
        model_config = BERT_MODEL_CONFIG
        explainer = setup_mock_bert_introspective_rationale_explainer(model_config=model_config)
        explainer.set_generator_classifier(classifier_type=CLASSIFIER_TYPE_CUSTOM)


    def test_load_explainer(self):
        model_config = BERT_MODEL_CONFIG
        explainer = setup_mock_bert_introspective_rationale_explainer(model_config=model_config)
        explainer.load()

    def test_bert_explain_local(self):
        train_data = get_ssts_dataset('train')
        test_data = get_ssts_dataset('test')
        X_train = train_data[TEXT_COL]
        X_test = test_data[TEXT_COL]
        preprocessor =BertPreprocessor()

        df_train = pd.concat([train_data[LABEL_COL], preprocessor.preprocess(X_train)], axis=1)
        df_test = pd.concat([test_data[LABEL_COL], preprocessor.preprocess(X_test)], axis=1)
        model_config = BERT_MODEL_CONFIG
        explainer = IntrospectiveRationaleExplainer(classifier_type=CLASSIFIER_TYPE_BERT, cuda=CUDA)
        explainer.build_model_config(model_config)
        explainer.set_preprocessor(preprocessor)
        explainer.load()
        explainer.fit(df_train, df_test)

        kwargs = {"preprocessor": preprocessor, "hard_importances": False}
        local_explanation = explainer.explain_local(SENTENCE, **kwargs)
        # BERT adds [CLS] at the beginning of a sentence and [SEP] at the end of each sentence  but we remove them.
        assert len(local_explanation.local_importance_values) == len(SENTENCE.split())

    def test_rnn_explain_local(self):
        train_data = get_ssts_dataset('train')
        test_data = get_ssts_dataset('test')
        all_data = pd.concat([train_data, test_data])
        X_train = train_data[TEXT_COL]
        X_test = test_data[TEXT_COL]
        preprocessor = GlovePreprocessor(count_threshold=TOKEN_COUNT_THRESHOLD, token_cutoff=MAX_SENT_COUNT)
        preprocessor.build_vocab(all_data[TEXT_COL])

        df_train = pd.concat([train_data[LABEL_COL], preprocessor.preprocess(X_train)], axis=1)
        df_test = pd.concat([test_data[LABEL_COL], preprocessor.preprocess(X_test)], axis=1)
        model_config = RNN_MODEL_CONFIG
        explainer = IntrospectiveRationaleExplainer(classifier_type=CLASSIFIER_TYPE_RNN, cuda=CUDA)
        explainer.build_model_config(model_config)
        explainer.set_preprocessor(preprocessor)
        explainer.load()
        explainer.fit(df_train, df_test)

        kwargs = {"preprocessor": preprocessor, "hard_importances": False}
        local_explanation = explainer.explain_local(SENTENCE, **kwargs)
        assert len(local_explanation.local_importance_values) == len(SENTENCE.split())
