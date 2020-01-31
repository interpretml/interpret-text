import sys
import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from interpret_text.common.dataset.utils_sst2 import load_sst2_pandas_df
from interpret_text.common.utils_unified import load_pandas_df
from interpret_text.three_player_introspective.three_player_introspective_explainer import ThreePlayerIntrospectiveExplainer
from interpret_text.common.utils_three_player import GlovePreprocessor, BertPreprocessor, ModelArguments, load_glove_embeddings
from interpret_text.widget import ExplanationDashboard
from sklearn.model_selection import train_test_split

'''
this is an example script for running the 3 Player Explainer
'''

MODEL_TYPE = "BERT"
QUICK_RUN = False
CUDA = True
LABEL_COL = "labels"
TEXT_COL = "sentences"

train_data = load_sst2_pandas_df('train')
test_data = load_sst2_pandas_df('test')
all_data = pd.concat([train_data, test_data])
if QUICK_RUN:
    train_data = train_data.head(50)
    test_data = test_data.head(50)

if MODEL_TYPE == "RNN":
    args = ModelArguments(CUDA, model_prefix="3PlayerModelRNN", save_best_model=False)
    preprocessor = GlovePreprocessor(all_data[TEXT_COL], token_count_thresh, max_sentence_token_count)
if MODEL_TYPE == "BERT":
    args = ModelArguments(CUDA, model_prefix="3PlayerModelBERT", save_best_model=False)
    preprocessor = BertPreprocessor()

args.labels = [0, 1]
args.num_labels = 2
# one thing we noticed is that if this is set too low (like .15), then
# the model will use [CLS] and [SEP] tokens to communicate 0 and 1 predictions, respectively
args.target_sparsity = .3

# append labels to tokenizer output
df_train = pd.concat([train_data[LABEL_COL], preprocessor.preprocess(train_data[TEXT_COL])], axis=1)
df_test = pd.concat([test_data[LABEL_COL], preprocessor.preprocess(test_data[TEXT_COL])], axis=1)

explainer = ThreePlayerIntrospectiveExplainer(args, preprocessor, classifier_type=MODEL_TYPE)

classifier = explainer.fit(df_train, df_test)