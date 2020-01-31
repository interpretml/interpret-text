import sys
import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from transformers import BertTokenizer
from interpret_text.common.dataset.utils_sst2 import load_sst2_pandas_df
from interpret_text.three_player_introspective.three_player_introspective_explainer import ThreePlayerIntrospectiveExplainer
from interpret_text.common.utils_three_player import GlovePreprocessor, BertPreprocessor, ModelArguments, load_glove_embeddings

'''
This is an example script for getting the predicted rationales and comparing it to the ground truth
rationales
'''

# load the training/testing data
train_data = load_sst2_pandas_df('train')
test_data = load_sst2_pandas_df('test')
all_data = pd.concat([train_data, test_data])

# prepare the ground truth data
df = pd.read_excel("./sst2_labels_100.xlsx", header=None)
sentences = df.loc[list(range(0, len(df), 2))]
labels = df.loc[list(range(1, len(df), 2))]

# assert that the test data and the labeled sentences correspond to the same text
print(test_data.head())
print(sentences.head())
print(labels.head())

all_masks = []
for row in range(0, len(sentences)):
    sentence_mask = []
    for col in range(0, len(sentences.columns)-1):
        sentence_mask.append(labels.iloc[row,col])

    all_masks.append(sentence_mask)

gt = np.array(all_masks)
ground_truth = pd.DataFrame({"sentences": test_data.head(100)["sentences"], "masks": all_masks})
print(ground_truth)
print("ground truth:")
print([list(l) for l in gt])

# set args for the model
MODEL_TYPE = "BERT"
QUICK_RUN = False
CUDA = True
LABEL_COL = "labels"
TEXT_COL = "sentences"

args = ModelArguments(CUDA, model_prefix="3PlayerModelBERT", save_best_model=False,
    num_epochs=30, num_pretrain_epochs=2)
preprocessor = BertPreprocessor()

args.labels = [0, 1]
args.num_labels = 2
args.target_sparsity = .25

# preprocess the training/testing data
if QUICK_RUN:
    train_data = train_data.head(50)
    test_data = test_data.head(100)
df_train = pd.concat([train_data[LABEL_COL], preprocessor.preprocess(train_data[TEXT_COL])], axis=1)
df_test = pd.concat([test_data[LABEL_COL], preprocessor.preprocess(test_data[TEXT_COL])], axis=1)

# run the model
explainer = ThreePlayerIntrospectiveExplainer(args, preprocessor, classifier_type=MODEL_TYPE)
classifier = explainer.fit(df_train, df_test)

# check results agains the ground truth
rationale_preds = explainer.predict(df_test.head(100))["rationale"]
pred = np.array(rationale_preds.cpu())

print("model predictions:")
print([list(l) for l in pred])