import numpy as np
import pandas as pd
import os

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
        