# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'notebooks\\explainers'))
	print(os.getcwd())
except:
	pass
# %%
from IPython import get_ipython

import pickle
import shutil
import numpy as np
import logging
import shap
import torch
import json
from numpy import dot
from numpy.linalg import norm
from urllib import request
from pytorch_pretrained_bert import BertModel, BertTokenizer

from interpret_text.msra.MSRAExplainer import MSRAExplainer


# %%
DATA_FOLDER = "./temp"
device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")

# %% [markdown]
# A function to generate embeddings for BERT Input

# %%
def embeddings_bert(text, device):
    # get the tokenized words.
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    words = ["[CLS]"] + tokenizer.tokenize(text) + ["[SEP]"]
    tokenized_ids = tokenizer.convert_tokens_to_ids(words)
    segment_ids = [0 for _ in range(len(words))]
    token_tensor = torch.tensor([tokenized_ids], device=device)
    segment_tensor = torch.tensor([segment_ids], device=device)
    x_bert = model.embeddings(token_tensor, segment_tensor)[0]
    return x_bert

# %% [markdown]
# Let's load the BERT base model with the saved finetuned parameters

# %%
#load the finetuned parameters
#model_state_dict = torch.load("models/model.pth")
#Load BERT base model with the finetuned parameters
# model = BertModel.from_pretrained("bert-base-uncased", state_dict=model_state_dict)
model = BertModel.from_pretrained("bert-base-uncased")
model.to(device)

for param in model.parameters():
    param.requires_grad = False
model.eval()

# %% [markdown]
# Now we generate the embeddings for the input text and initialize the interpreter. We also calculate the regularization parameter required by the MSR Asia Explainer using the function provided by the Explainer class.

# %%
text = "rare bird has more than enough charm to make it memorable."
embedded_input = embeddings_bert(text, device)
interpreter_msra = MSRAExplainer(device=device)
regularization = interpreter_msra.getRegularizationBERT(model=model)

# to calculate the regularization for the fine_tuned BERT
# tokens_train = torch.tensor(pickle.load( open( "dataset/tokens.p", "rb" ) ))
# embedding_x = model.embeddings(tokens_train)
# regularization = interpreter_msra.calculate_regularization(embedding_x, model, device, explain_layer=3, reduced_axes=None).tolist()[0]

# %% [markdown]
# We then call explain_local on the interpreter.

# %%
explanation_msra = interpreter_msra.explain_local(model=model, embedded_input=embedded_input, regularization=regularization)

# %% [markdown]
# Basic visualization until the visualization dashboard is fully integrated as a python widget

# %%
interpreter_msra.visualize(text)


# %%


