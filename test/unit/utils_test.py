from interpret_text.experimental.common.preprocessor.bert_preprocessor import BertPreprocessor
from interpret_text.experimental.common.preprocessor.glove_preprocessor import GlovePreprocessor
from interpret_text.experimental.introspective_rationale import IntrospectiveRationaleExplainer

from notebooks.test_utils.utils_mnli import load_mnli_pandas_df
from notebooks.test_utils.utils_sst2 import load_sst2_pandas_df
from interpret_text.experimental.common.utils_bert import BERTSequenceClassifier

CLASSIFIER_TYPE_BERT = "BERT"
CLASSIFIER_TYPE_BERT_RNN = "BERT_RNN"
CLASSIFIER_TYPE_RNN = "RNN"
CLASSIFIER_TYPE_CUSTOM = "CUSTOM"
CUDA = False
TOKEN_COUNT_THRESHOLD = 1
MAX_SENT_COUNT = 70


def get_mnli_test_dataset(file_split):
    DATA_FOLDER = "./temp"
    df = load_mnli_pandas_df(DATA_FOLDER, file_split)
    df = df[df["gold_label"] == "neutral"]  # get unique sentences
    return df[:50]


def get_ssts_dataset(file_split):
    DATA_FOLDER = "./temp"
    df = load_sst2_pandas_df(file_split, DATA_FOLDER)
    return df[:50]


def get_bert_model():
    NUM_LABELS = 2
    cache_dir = "./temp"
    classifier = BERTSequenceClassifier(
        language="bert-base-uncased", num_labels=NUM_LABELS, cache_dir=cache_dir
    )
    return classifier.model


def setup_mock_bert_introspective_rationale_explainer(model_config):
    explainer = IntrospectiveRationaleExplainer(classifier_type=CLASSIFIER_TYPE_BERT, cuda=CUDA)
    preprocessor = BertPreprocessor()
    explainer.set_preprocessor(preprocessor)
    explainer.build_model_config(model_config)
    return explainer


def setup_mock_rnn_introspective_rationale_explainer(model_config, data):
    explainer = IntrospectiveRationaleExplainer(classifier_type=CLASSIFIER_TYPE_RNN, cuda=CUDA)
    preprocessor = GlovePreprocessor(TOKEN_COUNT_THRESHOLD, MAX_SENT_COUNT)
    preprocessor.build_vocab(data)
    explainer.set_preprocessor(preprocessor)
    explainer.build_model_config(model_config)
    return explainer


