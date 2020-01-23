# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# *Copyright (c) Microsoft Corporation. All rights reserved.*
# 
# *Licensed under the MIT License.*
# 
# # Text Classification of SST-2 Sentences using a 3-Player Introspective Model

# %%
import sys
import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from interpret_text.three_player_introspective.three_player_introspective_explainer import ThreePlayerIntrospectiveExplainer
from interpret_text.common.utils_three_player import load_pandas_df, GloveTokenizer, ModelArguments
from interpret_text.widget import ExplanationDashboard

# %% [markdown]
# ## Introduction
# In this notebook, we train and evaluate a  [three-player explainer](http://people.csail.mit.edu/tommi/papers/YCZJ_EMNLP2019.pdf) model on a subset of the [SST-2](https://nlp.stanford.edu/sentiment/index.html/) dataset. To run this notebook, we used the SST-2 data files provided [here](https://github.com/AcademiaSinicaNLPLab/sentiment_dataset).
# %% [markdown]
# ### Set parameters
# Here we set some parameters that we use for our modeling task.

# %%
# data processing parameters
DATA_FOLDER = "../../../data/sst2"
LABEL_COL = "labels" 
TEXT_COL = "sentences"
token_count_thresh = 1
max_sentence_token_count = 70

# training procedure parameters
pre_trained_model_prefix = 'pre_trained_cls.model'
save_path = os.path.join("..", "models")
model_prefix = "sst2rnpmodel"
save_best_model = True
pre_train_cls = False

# parameters used internally in the model
args = ModelArguments() # intialize default model parameters
args.cuda = False
args.batch_size = 40
args.embedding_path = os.path.join(DATA_FOLDER, "glove.6B.100d.txt")

# %% [markdown]
# ## Read Dataset
# We start by loading a subset of the data for training and testing.

# %%
# TODO: load dataset to blob storage
train_data = load_pandas_df('train', LABEL_COL, TEXT_COL)
test_data = load_pandas_df('test', LABEL_COL, TEXT_COL)
all_data = pd.concat([train_data, test_data])
x_train = train_data[TEXT_COL]
x_test = test_data[TEXT_COL]


# %%
# get all unique labels
labels = all_data[LABEL_COL].unique()
args.labels = np.array(sorted(labels))
args.num_labels = len(labels)

# %% [markdown]
# ## Tokenization and embedding
# The data is then tokenized and embedded using glove embeddings.

# %%
tokenizer = GloveTokenizer(all_data[TEXT_COL], token_count_thresh, max_sentence_token_count)

# append labels to tokenizer output
df_train = pd.concat([train_data[LABEL_COL], tokenizer.tokenize(x_train)], axis=1)
df_test = pd.concat([test_data[LABEL_COL], tokenizer.tokenize(x_test)], axis=1)

print(df_train)

# %% [markdown]
# ## Explainer
# Then, we create and train the explainer.

# %%
explainer = ThreePlayerIntrospectiveExplainer(args, tokenizer.word_vocab) # word vocab is needed to initialize the model's embedding layer
classifier = explainer.fit(df_train, df_test, args.batch_size, num_iteration=1, pretrain_cls=pre_train_cls)

# %% [markdown]
# We can test the explainer and measure its performance:

# %%
accuracy, anti_accuracy, sparsity = explainer.score(df_test)
print("Test sparsity: ", sparsity)
print("Test accuracy: ", accuracy, "% Anti-accuracy: ", anti_accuracy)

# %% [markdown]
# ## Local importances
# We can display the found local importances (the most and least important words for a given sentence):

# %%
# Enter a sentence that needs to be interpreted
# sentence = "This great movie was really good"
# label = 1

# Initialize the tokenizer to use for the sentence
# local_explanantion = explainer.explain_local(sentence, label, tokenizer, hard_importances=False)

# %% [markdown]
# ## Visualize explanations
# We can visualize local feature importances as a heatmap over words in the document and view importance values of individual words.

# %%
# explainer.visualize(local_explanantion._local_importance_values, local_explanantion._features)

# ExplanationDashboard(local_explanantion)

