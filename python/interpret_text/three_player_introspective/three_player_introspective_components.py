import json
import logging
import os
import random
import sys
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm

# classes needed for Rationale3Player
class RnnModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_num, dropout_rate):
        """
        input_dim -- dimension of input
        hidden_dim -- dimension of filters
        layer_num -- number of RNN layers   
        """
        super(RnnModel, self).__init__()
        self.rnn_layer = nn.GRU(input_size=input_dim, 
                                hidden_size=hidden_dim//2, 
                                num_layers=layer_num,
                                bidirectional=True, dropout=dropout_rate)
    
    def forward(self, embeddings, mask=None, h0=None, c0=None):
        """
        Inputs:
            embeddings -- sequence of word embeddings, (batch_size, sequence_length, embedding_dim)
            mask -- a float tensor of masks, (batch_size, length)
            h0, c0 --  (num_layers * num_directions, batch, hidden_size)
        Outputs:
            hiddens -- sentence embedding tensor, (batch_size, hidden_dim, sequence_length)
        """
        embeddings_ = embeddings.transpose(0, 1) #(sequence_length, batch_size, embedding_dim)
        
        if mask is not None:
            seq_lengths = list(torch.sum(mask, dim=1).cpu().data.numpy())
            seq_lengths = list(map(int, seq_lengths))
            inputs_ = torch.nn.utils.rnn.pack_padded_sequence(embeddings_, seq_lengths)
        else:
            inputs_ = embeddings_
        
        if h0 is not None:
            hidden, _ = self.rnn_layer(inputs_, h0)
        else:
            hidden, _ = self.rnn_layer(inputs_) #(sequence_length, batch_size, hidden_dim (* 2 if bidirectional))

        if mask is not None:
            hidden, _ = torch.nn.utils.rnn.pad_packed_sequence(hidden) #(length, batch_size, hidden_dim)
        
        return hidden.permute(1, 2, 0) #(batch_size, hidden_dim, sequence_length)

class ClassifierModule(nn.Module):
    '''
    classifier for both E and E_anti models provided with RNP paper code
    '''
    def __init__(self, args, word_vocab):
        super(ClassifierModule, self).__init__()
        self.args = args
        self.encoder = RnnModel(self.args.embedding_dim, self.args.hidden_dim, self.args.layer_num, self.args.dropout_rate)
        self.predictor = nn.Linear(self.args.hidden_dim, self.args.num_labels)
        
        self.input_dim = args.embedding_dim
        self.embedding_path = args.embedding_path
        self.fine_tuning = args.fine_tuning

        self.init_embedding_layer(word_vocab)
        self.NEG_INF = -1.0e6
    
    def init_embedding_layer(self, word_vocab):
        # get initial vocab embeddings
        vocab_size = len(word_vocab)
        # initialize a numpy embedding matrix 
        embeddings = 0.1*np.random.randn(vocab_size, self.input_dim).astype(np.float32)

        # replace the <PAD> embedding by all zero
        embeddings[0, :] = np.zeros(self.input_dim, dtype=np.float32)

        if self.embedding_path and os.path.isfile(self.embedding_path):
            f = open(self.embedding_path, "r", encoding="utf8")
            counter = 0
            for line in f:
                data = line.strip().split(" ")
                word = data[0].strip()
                embedding = data[1::]
                embedding = list(map(np.float32, embedding))
                if word in word_vocab:
                    embeddings[word_vocab[word], :] = embedding
                    counter += 1
            f.close()
            print("%d words have been switched."%counter)
        else:
            print("embedding is initialized fully randomly.")

        # initialize embedding layer
        self.embed_layer = nn.Embedding(vocab_size, self.input_dim)
        self.embed_layer.weight.data = torch.from_numpy(embeddings)
        self.embed_layer.weight.requires_grad = self.fine_tuning

    def forward(self, X_tokens, attention_mask, z=None):
        """
        Inputs:
            word_embeddings -- torch Variable in shape of (batch_size, length, embed_dim)
            z -- rationale (batch_size, length)
            mask -- torch Variable in shape of (batch_size, length)
        Outputs:
            predict -- (batch_size, num_label)
        """        
        word_embeddings = self.embed_layer(X_tokens)
        if z is None:
            dtype = torch.cuda.float if torch.cuda.is_available() else torch.float
            z = torch.ones_like(X_tokens).type(dtype)

        masked_input = word_embeddings * z.unsqueeze(-1)
        hiddens = self.encoder(masked_input, attention_mask)

        max_hidden = torch.max(hiddens + (1 - attention_mask * z).unsqueeze(1) * self.NEG_INF, dim=2)[0]
        
        predict = self.predictor(max_hidden)
        # return cls_pred_logits, [hidden_states], _ b/c that is what bert does
        return predict, [word_embeddings, hiddens], None # the last one is for attention in the BERT model

# extra classes needed for introspective model 
class DepGenerator(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, layer_num, dropout_rate, z_dim):
        """     
        input_dim -- dimension of input   
        hidden_dim -- dimension of filters
        z_dim -- rationale or not, always 2    
        layer_num -- number of RNN layers   
        """
        super(DepGenerator, self).__init__()
        
        self.generator_model = RnnModel(input_dim, hidden_dim, layer_num, dropout_rate)
        self.output_layer = nn.Linear(hidden_dim, z_dim)
        
    def forward(self, X_embeddings, h0=None, c0=None, mask=None):
        """
        Given input x in shape of (batch_size, sequence_length) generate a 
        "binary" mask as the rationale
        Inputs:
            x -- input sequence of word embeddings, (batch_size, sequence_length, embedding_dim)
        Outputs:
            z -- output rationale, "binary" mask, (batch_size, sequence_length)
        """
        #(batch_size, sequence_length, hidden_dim)
        hiddens = self.generator_model(X_embeddings, mask, h0, c0).transpose(1, 2).contiguous() 
        scores = self.output_layer(hiddens) # (batch_size, sequence_length, 2)
        return scores
        
class IntrospectionGeneratorModule(nn.Module):
    '''
    classifier for both E and E_anti models
    RNN:
        """
        input_dim -- dimension of input
        hidden_dim -- dimension of filters
        layer_num -- number of RNN layers   
        """
    DepGenerator:
        """        
        input_dim -- dimension of input   
        hidden_dim -- dimension of filters
        z_dim -- rationale or not, always 2    
        layer_num -- number of RNN layers   
        """
    '''
    def __init__(self, args, classifier):
        super(IntrospectionGeneratorModule, self).__init__()
        self.args = args
        
        # for initializing RNN and DepGenerator
        self.input_dim = args.embedding_dim
        self.hidden_dim = args.hidden_dim
        self.layer_num = args.layer_num
        self.z_dim = args.z_dim
        self.dropout_rate = args.dropout_rate
        # for embedding labels
        self.num_labels = args.num_labels
        self.label_embedding_dim = args.label_embedding_dim
        
        # for training
        self.fixed_classifier = args.fixed_classifier
        
        self.NEG_INF = -1.0e6
        self.lab_embed_layer = self._create_label_embed_layer() # should be shared with the Classifier_pred weights
        
        # baseline classification model
        self.classifier = classifier

        self.Transformation = nn.Sequential()
        self.Transformation.add_module('linear_layer', nn.Linear(self.hidden_dim + self.label_embedding_dim, self.hidden_dim // 2))
        self.Transformation.add_module('tanh_layer', nn.Tanh())
        self.Generator = DepGenerator(self.input_dim, self.hidden_dim, self.layer_num, self.dropout_rate, self.z_dim)
        
    def _create_label_embed_layer(self):
        embed_layer = nn.Embedding(self.num_labels, self.label_embedding_dim)
        embed_layer.weight.data.normal_(mean=0, std=0.1)
        embed_layer.weight.requires_grad = True
        return embed_layer
        
    def forward(self, X_tokens, mask):
        cls_pred_logits, hidden_states, _ = self.classifier(X_tokens, attention_mask=mask)
        
        # hidden states must be in shape (batch_size, hidden_dim, length)
        # RNN returns (batch_size, hidden_dim, length)
        # BERT returns (batch_size, length, hidden_dim) #TODO replace with if BERT:
        if hidden_states[-1].shape[1] != self.hidden_dim:
            hidden_states[-1] = hidden_states[-1].transpose(1, 2)

        max_cls_hidden = torch.max(hidden_states[-1] + (1 - mask).unsqueeze(1) * self.NEG_INF, dim=2)[0] # (batch_size, hidden_dim)
        if self.fixed_classifier:
            max_cls_hidden = Variable(max_cls_hidden.data)

        word_embeddings = hidden_states[0]

        _, cls_pred = torch.max(cls_pred_logits, dim=1)

        cls_lab_embeddings = self.lab_embed_layer(cls_pred) # (batch_size, lab_emb_dim)
        
        init_h0 = self.Transformation(torch.cat([max_cls_hidden, cls_lab_embeddings], dim=1)) # (batch_size, hidden_dim / 2)
        init_h0 = init_h0.unsqueeze(0).expand(2, init_h0.size(0), init_h0.size(1)).contiguous() # (2, batch_size, hidden_dim / 2)
        
        z_scores_ = self.Generator(word_embeddings, mask=mask, h0=init_h0) #(batch_size, length, 2)
        z_scores_[:, :, 1] = z_scores_[:, :, 1] + (1 - mask) * self.NEG_INF
        
        return z_scores_, cls_pred_logits, word_embeddings
