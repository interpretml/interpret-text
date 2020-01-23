import numpy as np
import pandas as pd

# default parameters used to initialize the model
class ModelArguments():
    def __init__(self, **kwargs):
        # to initialize model modules
        self.embedding_dim = 100
        self.hidden_dim = 200
        self.layer_num = 1
        self.z_dim = 2
        self.dropout_rate = 0.5
        self.label_embedding_dim = 400
        self.fixed_classifier = True

        # to initialize model
        self.fine_tuning = False
        self.lambda_sparsity = 1.0
        self.lambda_continuity = 1.0
        self.lambda_anti = 1.0
        self.exploration_rate = 0.05
        self.count_tokens = 8
        self.count_pieces = 4
        self.lambda_acc_gap = 1.2
        self.lr=0.001

# Splits words into tokens that are passed into glove embeddings
class GloveTokenizer:
    def __init__(self, text, count_thresh, token_cutoff):
        """
        text (list): a list of sentences (strings)
        count_thresh (int): the number of times a word has to appear in a sentence to be
        counted as part of the vocabulary
        token_cutoff (int): the maximum number of tokens a sentence can have
        """
        self.word_vocab, self.reverse_word_vocab, self.counts = self.build_vocab(text)
        self.count_thresh = count_thresh
        self.token_cutoff = token_cutoff
    
    
    def build_vocab(self, text):
        """
        Build the vocabulary (all words that appear at least once in text)
        Args:
            text (list): a list of sentences (strings)
        Returns:
            words_to_idxs (dict): a mapping of the set of unique words (keys) found in text 
                (appearing more times than count_thresh) to unique indices (values)
            idxs_to_words (dict): a mapping from vocabulary indices (keys) to words (values)
            counts (dict): a mapping of the a word (key) to the number of times it appears in text (value)
        """
        words_to_idxs = {"<PAD>":0, "<UNK>":1}
        counts = {}
        for sentence in text:
            for word in sentence.split():
                if word not in words_to_idxs:
                    words_to_idxs[word] = len(words_to_idxs)
                    counts[word] = 1
                else:
                    counts[word] += 1
        idxs_to_words = {v: k for k, v in words_to_idxs.items()}
        return words_to_idxs, idxs_to_words, counts

    def generate_tokens(self, text):
        """
        Split text into padded lists of tokens that are part of the recognized vocabulary
        Args:
            text (str): a piece of text (e.g. a sentence)
        Returns:
            indexed_text (np.array): the token/vocabulary indices of recognized words in text, 
                padded to the maximum sentence length
            mask (np.array): a mask indicating which indices in indexed_text correspond to words (1s)
                and which correspond to pads (0s)
        """
        indexed_text = [self.word_vocab[word] if (self.counts[word] > self.count_thresh) else self.word_vocab["<UNK>"] for word in text.split()]
        pad_length = self.token_cutoff - len(indexed_text)
        mask = [1] * len(indexed_text) + [0] * pad_length
        
        indexed_text = indexed_text + [self.word_vocab["<PAD>"]] * pad_length
        
        return np.array(indexed_text), np.array(mask)
    
    def tokenize(self, data):
        """
        Converts a list of text into a dataframe containing padded token ids,
        masks distinguishing word tokens from pads, and word token counts for
        each text in the list. 
        Args:
            data (list): a list of strings (e.g. sentences)
        Returns:
            tokens (pd.Dataframe): a dataframe containing
            lists of word token ids, pad/word masks, and token counts 
            for each string in the list
        """
        token_lists = []
        masks = []
        counts = []
        for sentence in data:
            token_list, mask = self.generate_tokens(sentence)
            token_lists.append(token_list)
            masks.append(mask)
            counts.append(np.sum(mask))
        tokens = pd.DataFrame({"tokens": token_lists, "mask": masks, "counts": counts})
        return tokens
        