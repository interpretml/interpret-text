import numpy as np
import pandas as pd
import os

from interpret_text.common.utils_unified import maybe_download

DATA_URLS = {
    "train": "https://github.com/AcademiaSinicaNLPLab/sentiment_dataset/raw/master/data/stsa.binary.train",
    "dev": "https://github.com/AcademiaSinicaNLPLab/sentiment_dataset/raw/master/data/stsa.binary.dev",
    "test": "https://github.com/AcademiaSinicaNLPLab/sentiment_dataset/raw/master/data/stsa.binary.test",
}

class ModelArguments():
    def __init__(self, **kwargs):
        # to initialize classifierModule and introspectionGeneratorModule
        self.embedding_dim = 100
        self.hidden_dim = 200
        self.layer_num = 1
        self.z_dim = 2
        self.dropout_rate = 0.5

        # to init only introspectionGeneratorModule
        self.label_embedding_dim = 400
        self.fixed_classifier = True

        # to init model
        self.fine_tuning = False
        self.lambda_sparsity = 1.0
        self.lambda_continuity = 1.0
        self.lambda_anti = 1.0
        self.exploration_rate = 0.05
        self.count_tokens = 8
        self.count_pieces = 4
        self.lambda_acc_gap = 1.2
        self.lr=0.001

def load_pandas_df(file_split, label_col, text_col, local_cache_path="."):
    """Loads extracted dataset into pandas
    Args:
        local_cache_path ([type], optional): [description]. Defaults to current working directory.
        file_split (str, optional): The subset to load.
            One of: {"train", "dev", "split"}
            Defaults to "train".
        label_col: the header of the column containing labels
        text_col: the header of the column containing text
    Returns:
        pd.DataFrame: pandas DataFrame containing the specified
            SST2 subset.
    """
    try:
        URL = DATA_URLS[file_split]
        file_name = URL.split("/")[-1]
        file_path = maybe_download(URL, file_name, local_cache_path)
    except Exception as e:
        raise e
    
    return load_data(file_path, label_col, text_col)

def load_data(fpath, label_col, text_col):
    df_dict = {label_col: [], text_col: []}
    with open(fpath, 'r') as f:
        label_start = 0
        sentence_start = 2
        for line in f:
            label = int(line[label_start])
            sentence = line[sentence_start:]
            df_dict[label_col].append(label)
            df_dict[text_col].append(sentence)
    return pd.DataFrame.from_dict(df_dict)

# Using glove embeddings (used in the original paper)
class GloveTokenizer:
    def __init__(self, text, count_thresh, token_cutoff):
        '''
        text: a list of sentences (strings)
        count_thresh: the number of times a word has to appear in a sentence to be
        counted as part of the vocabulary
        token_cutoff: the maximum number of tokens a sentence can have
        '''
        self.word_vocab, self.reverse_word_vocab, self.counts = self.build_vocab(text)
        self.count_thresh = count_thresh
        self.token_cutoff = token_cutoff
    
    # build the vocabulary (all words that appear at least once in the data)
    def build_vocab(self, text):
        d = {"<PAD>":0, "<UNK>":1}
        counts = {}
        for sentence in text:
            for word in sentence.split():
                if word not in d:
                    d[word] = len(d)
                    counts[word] = 1
                else:
                    counts[word] += 1
        reverse_d = {v: k for k, v in d.items()}
        return d, reverse_d, counts

    def generate_tokens_glove(self, word_vocab, text):
        indexed_text = [self.word_vocab[word] if (self.counts[word] > self.count_thresh) else self.word_vocab["<UNK>"] for word in text.split()]
        pad_length = self.token_cutoff - len(indexed_text)
        mask = [1] * len(indexed_text) + [0] * pad_length
        
        indexed_text = indexed_text + [self.word_vocab["<PAD>"]] * pad_length
        
        return np.array(indexed_text), np.array(mask)
    
    # tokenize using glove embeddings
    def tokenize(self, data):
        l = []
        m = []
        counts = []
        for sentence in data:
            token_list, mask = self.generate_tokens_glove(self.word_vocab, sentence)
            l.append(token_list)
            m.append(mask)
            counts.append(np.sum(mask))
        tokens = pd.DataFrame({"tokens": l, "mask": m, "counts": counts})
        return tokens
        