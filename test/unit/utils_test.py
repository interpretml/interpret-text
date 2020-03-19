from notebooks.test_utils.utils_mnli import load_mnli_pandas_df
from notebooks.test_utils.utils_sst2 import load_sst2_pandas_df
from interpret_text.common.utils_bert import BERTSequenceClassifier


def get_mnli_test_dataset():
    DATA_FOLDER = "./temp"
    TEXT_COL = "sentence1"
    df = load_mnli_pandas_df(DATA_FOLDER, "train")
    df = df[df["gold_label"] == "neutral"]  # get unique sentences
    return df[TEXT_COL][:50]

def get_ssts_dataset(file_split):
    DATA_FOLDER = "./temp"
    df = load_sst2_pandas_df(DATA_FOLDER, file_split)
    return df[:50]

def get_bert_model():
    NUM_LABELS = 2
    cache_dir = "./temp"
    classifier = BERTSequenceClassifier(
        language="bert-base-uncased", num_labels=NUM_LABELS, cache_dir=cache_dir
    )
    return classifier.model
