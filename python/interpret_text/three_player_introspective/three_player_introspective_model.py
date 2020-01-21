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
    def __init__(self, args):
        super(ClassifierModule, self).__init__()
        self.args = args
        self.encoder = RnnModel(self.args.embedding_dim, self.args.hidden_dim, self.args.layer_num, self.args.dropout_rate)
        self.predictor = nn.Linear(self.args.hidden_dim, self.args.num_labels)
        
        self.NEG_INF = -1.0e6
        

    def forward(self, word_embeddings, z, mask):
        """
        Inputs:
            word_embeddings -- torch Variable in shape of (batch_size, length, embed_dim)
            z -- rationale (batch_size, length)
            mask -- torch Variable in shape of (batch_size, length)
        Outputs:
            predict -- (batch_size, num_label)
        """        

        masked_input = word_embeddings * z.unsqueeze(-1)
        hiddens = self.encoder(masked_input, mask)
        
        max_hidden = torch.max(hiddens + (1 - mask * z).unsqueeze(1) * self.NEG_INF, dim=2)[0]
        
        predict = self.predictor(max_hidden)
        return predict

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
        
        
    def forward(self, x, h0=None, c0=None, mask=None):
        """
        Given input x in shape of (batch_size, sequence_length) generate a 
        "binary" mask as the rationale
        Inputs:
            x -- input sequence of word embeddings, (batch_size, sequence_length, embedding_dim)
        Outputs:
            z -- output rationale, "binary" mask, (batch_size, sequence_length)
        """
        #(batch_size, sequence_length, hidden_dim)
        hiddens = self.generator_model(x, mask, h0, c0).transpose(1, 2).contiguous() 
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
    def __init__(self, args):
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
        self.Classifier_enc = RnnModel(self.input_dim, self.hidden_dim, self.layer_num, self.dropout_rate)
        self.Classifier_pred = nn.Linear(self.hidden_dim, self.num_labels)
        
        self.Transformation = nn.Sequential()
        self.Transformation.add_module('linear_layer', nn.Linear(self.hidden_dim + self.label_embedding_dim, self.hidden_dim // 2))
        self.Transformation.add_module('tanh_layer', nn.Tanh())
        self.Generator = DepGenerator(self.input_dim, self.hidden_dim, self.layer_num, self.dropout_rate, self.z_dim)
        
        
    def _create_label_embed_layer(self):
        embed_layer = nn.Embedding(self.num_labels, self.label_embedding_dim)
        embed_layer.weight.data.normal_(mean=0, std=0.1)
        embed_layer.weight.requires_grad = True
        return embed_layer
    
    
    def forward(self, word_embeddings, mask):
        cls_hiddens = self.Classifier_enc(word_embeddings, mask) # (batch_size, hidden_dim, sequence_length)
        max_cls_hidden = torch.max(cls_hiddens + (1 - mask).unsqueeze(1) * self.NEG_INF, dim=2)[0] # (batch_size, hidden_dim)
        
        if self.fixed_classifier:
            max_cls_hidden = Variable(max_cls_hidden.data)
        
        cls_pred_logits = self.Classifier_pred(max_cls_hidden) # (batch_size, num_labels)
        
        _, cls_pred = torch.max(cls_pred_logits, dim=1) # (batch_size,)
        
        cls_lab_embeddings = self.lab_embed_layer(cls_pred) # (batch_size, lab_emb_dim)
        
        init_h0 = self.Transformation(torch.cat([max_cls_hidden, cls_lab_embeddings], dim=1)) # (batch_size, hidden_dim / 2)
        init_h0 = init_h0.unsqueeze(0).expand(2, init_h0.size(0), init_h0.size(1)).contiguous() # (2, batch_size, hidden_dim / 2)
        z_scores_ = self.Generator(word_embeddings, h0=init_h0, mask=mask) #(batch_size, length, 2)
        z_scores_[:, :, 1] = z_scores_[:, :, 1] + (1 - mask) * self.NEG_INF
        
        return z_scores_, cls_pred_logits


class ThreePlayerIntrospectiveModel(nn.Module):
    """flattening the HardIntrospectionRationale3PlayerClassificationModel -> HardRationale3PlayerClassificationModel -> 
       Rationale3PlayerClassificationModel dependency structure from original paper code"""

    def __init__(self, args, word_vocab, explainer, anti_explainer, generator, classifier):
        """Initializes the model, including the explainer, anti-rationale explainer
        Args:
            args: 
            embeddings:
            classificationModule: type of classifier
            
        """
        super(ThreePlayerIntrospectiveModel, self).__init__()
        self.args = args
        # from Rationale3PlayerClassificationModel initialization
        self.lambda_sparsity = args.lambda_sparsity
        self.lambda_continuity = args.lambda_continuity
        self.lambda_anti = args.lambda_anti
        self.hidden_dim = args.hidden_dim
        self.input_dim = args.embedding_dim
        self.embedding_path = args.embedding_path
        self.fine_tuning = args.fine_tuning
        # from Hardrationale3PlayerClassificationModel initialization
        self.exploration_rate = args.exploration_rate
        # from HardIntrospection3PlayerClassificationModel initialization:
        self.lambda_acc_gap = args.lambda_acc_gap
        self.fixed_classifier = args.fixed_classifier
        self.count_tokens = args.count_tokens # used to calc sparsity loss
        self.count_pieces = args.count_pieces # used to calc sparsity loss
        self.use_cuda = args.cuda
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.num_labels = args.num_labels 

        # initialize model components
        self.E_model = explainer(args)
        self.E_anti_model = anti_explainer(args)
        self.C_model = classifier(args)
        self.generator = generator(args)
        self.init_embedding_layer(word_vocab)
        self.word_vocab = word_vocab
        self.reverse_word_vocab = {v: k for k, v in word_vocab.items()}
                    
        # no internal code dependencies
        self.NEG_INF = -1.0e6
        self.loss_func = nn.CrossEntropyLoss(reduce=False)
        self.z_history_rewards = deque([0], maxlen=200)
        self.train_accs = []

        # initialize optimizers
        self.init_optimizers()
        self.init_rl_optimizers()


    # methods from Hardrationale3PlayerClassificationModel
    def init_optimizers(self): # not sure if this can be merged with initializer
        self.opt_E = torch.optim.Adam(filter(lambda x: x.requires_grad, self.E_model.parameters()), lr=self.lr)
        self.opt_E_anti = torch.optim.Adam(filter(lambda x: x.requires_grad, self.E_anti_model.parameters()), lr=self.lr)
    
    def init_rl_optimizers(self):
        self.opt_G_sup = torch.optim.Adam(filter(lambda x: x.requires_grad, self.generator.parameters()), lr=self.lr)
        self.opt_G_rl = torch.optim.Adam(filter(lambda x: x.requires_grad, self.generator.parameters()), lr=self.lr * 0.1)
    
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

    def _generate_rationales(self, z_prob_):
        '''
        Input:
            z_prob_ -- (num_rows, length, 2)
        Output:
            z -- (num_rows, length)
        '''        
        z_prob__ = z_prob_.view(-1, 2) # (num_rows * length, 2)
        
        # sample actions
        sampler = torch.distributions.Categorical(z_prob__)
        if self.training:
            z_ = sampler.sample() # (num_rows * p_length,)
        else:
            z_ = torch.max(z_prob__, dim=-1)[1]
        
        #(num_rows, length)
        z = z_.view(z_prob_.size(0), z_prob_.size(1))
        
        if self.use_cuda:
            z = z.type(torch.cuda.FloatTensor)
        else:
            z = z.type(torch.FloatTensor)
            
        # (num_rows * length,)
        neg_log_probs_ = -sampler.log_prob(z_)
        # (num_rows, length)
        neg_log_probs = neg_log_probs_.view(z_prob_.size(0), z_prob_.size(1))
        
        return z, neg_log_probs
        
    # methods from emnlp model
    def count_regularization_baos_for_both(self, z, count_tokens, count_pieces, mask=None):
        """
        Compute regularization loss, based on a given rationale sequence
        Use Yujia's formulation

        Inputs:
            z -- torch variable, "binary" rationale, (batch_size, sequence_length)
            percentage -- the percentage of words to keep
        Outputs:
            a loss value that contains two parts:
            continuity_loss --  \sum_{i} | z_{i-1} - z_{i} | 
            sparsity_loss -- |mean(z_{i}) - percent|
        """

        # (batch_size,)
        if mask is not None:
            mask_z = z * mask
            seq_lengths = torch.sum(mask, dim=1)
        else:
            mask_z = z
            seq_lengths = torch.sum(z - z + 1.0, dim=1)
        
        mask_z_ = torch.cat([mask_z[:, 1:], mask_z[:, -1:]], dim=-1)
            
        continuity_ratio = torch.sum(torch.abs(mask_z - mask_z_), dim=-1) / seq_lengths #(batch_size,) 
        percentage = count_pieces * 2 / seq_lengths
        continuity_loss = torch.abs(continuity_ratio - percentage)
        
        sparsity_ratio = torch.sum(mask_z, dim=-1) / seq_lengths #(batch_size,)
        percentage = count_tokens / seq_lengths #(batch_size,)
        sparsity_loss = torch.abs(sparsity_ratio - percentage)

        return continuity_loss, sparsity_loss

    def train_one_step(self, x, label, baseline, mask):
        # TODO: try to see whether removing the follows makes any differences
        self.opt_E_anti.zero_grad()
        self.opt_E.zero_grad()
        self.opt_G_sup.zero_grad()
        self.opt_G_rl.zero_grad()
        
        predict, anti_predict, cls_predict, z, neg_log_probs = self.forward(x, mask)
        
        e_loss_anti = torch.mean(self.loss_func(anti_predict, label))
        
#         e_loss = torch.mean(self.loss_func(predict, label))
        _, cls_pred = torch.max(cls_predict, dim=1) # (batch_size,)
#         e_loss = torch.mean(self.loss_func(predict, cls_pred)) # e_loss comes from only consistency
        e_loss = (torch.mean(self.loss_func(predict, label)) + torch.mean(self.loss_func(predict, cls_pred))) / 2
        
        # g_sup_loss comes from only cls pred loss
        g_sup_loss, g_rl_loss, rewards, consistency_loss, continuity_loss, sparsity_loss = self.get_loss(predict, 
                                                                         anti_predict, 
                                                                         cls_predict, label, z, 
                                                                         neg_log_probs, baseline, mask)
        
        losses = {'e_loss':e_loss.cpu().data, 'e_loss_anti':e_loss_anti.cpu().data,
                 'g_sup_loss':g_sup_loss.cpu().data, 'g_rl_loss':g_rl_loss.cpu().data}
        
        e_loss_anti.backward()
        self.opt_E_anti.step()
        self.opt_E_anti.zero_grad()
        
        e_loss.backward()
        self.opt_E.step()
        self.opt_E.zero_grad()
        
        if not self.fixed_classifier:
            g_sup_loss.backward(retain_graph=True)
            self.opt_G_sup.step()
            self.opt_G_sup.zero_grad()

        g_rl_loss.backward()
        self.opt_G_rl.step()
        self.opt_G_rl.zero_grad()
        
        return losses, predict, anti_predict, cls_predict, z, rewards, consistency_loss, continuity_loss, sparsity_loss
        
    def forward(self, x, mask):
        """
        Inputs:
            x -- torch Variable in shape of (batch_size, length)
            mask -- torch Variable in shape of (batch_size, length)
        Outputs:
            predict -- (batch_size, num_label)
            z -- rationale (batch_size, length)
        """        
        word_embeddings = self.embed_layer(x) #(batch_size, length, embedding_dim)

        z_scores_, cls_predict = self.generator(word_embeddings, mask)
        
        z_probs_ = F.softmax(z_scores_, dim=-1)
        
        z_probs_ = (mask.unsqueeze(-1) * ( (1 - self.exploration_rate) * z_probs_ + self.exploration_rate / z_probs_.size(-1) ) ) + ((1 - mask.unsqueeze(-1)) * z_probs_)
        
        z, neg_log_probs = self._generate_rationales(z_probs_) #(batch_size, length)
        
        predict = self.E_model(word_embeddings, z, mask)
        
        anti_predict = self.E_anti_model(word_embeddings, 1 - z, mask)

        return predict, anti_predict, cls_predict, z, neg_log_probs
    
    def get_z_scores(self, df_test):
        """
        Inputs:
            x -- torch Variable in shape of (batch_size, length)
            mask -- torch Variable in shape of (batch_size, length)
        Outputs:
            z_scores -- non-softmaxed rationale, (batch_size, length)
            cls_predict -- prediction of generator's classifier, (batch_size, num_label)
        """        
        x, mask, _ = self.generate_data(df_test)
        word_embeddings = self.embed_layer(x) #(batch_size, length, embedding_dim)
        z_scores, cls_predict = self.generator(word_embeddings, mask)
        z_scores = F.softmax(z_scores, dim=-1)


        return z_scores, cls_predict

    def get_advantages(self, pred_logits, anti_pred_logits, cls_pred_logits, label, z, neg_log_probs, baseline, mask):
        '''
        Input:
            z -- (batch_size, length)
        '''
        
        # supervised loss
        prediction_loss = self.loss_func(cls_pred_logits, label) # (batch_size, )
        sup_loss = torch.mean(prediction_loss)
        
        # total loss of accuracy (not batchwise)
        _, cls_pred = torch.max(cls_pred_logits, dim=1) # (batch_size,)
        _, ver_pred = torch.max(pred_logits, dim=1) # (batch_size,)
        consistency_loss = self.loss_func(pred_logits, cls_pred)
        
        prediction = (ver_pred == label).type(torch.FloatTensor)
        pred_consistency = (ver_pred == cls_pred).type(torch.FloatTensor)
        
        _, anti_pred = torch.max(anti_pred_logits, dim=1)
        prediction_anti = (anti_pred == label).type(torch.FloatTensor) * self.lambda_anti
        
        if self.use_cuda:
            prediction = prediction.cuda()  #(batch_size,)
            pred_consistency = pred_consistency.cuda()  #(batch_size,)
            prediction_anti = prediction_anti.cuda()

        continuity_loss, sparsity_loss = self.count_regularization_baos_for_both(z, self.count_tokens, self.count_pieces, mask)
        
        continuity_loss = continuity_loss * self.lambda_continuity
        sparsity_loss = sparsity_loss * self.lambda_sparsity

        # batch RL reward 
        # rewards = (prediction + pred_consistency) * self.args.lambda_pos_reward - prediction_anti - sparsity_loss - continuity_loss
        rewards = 0.1 * prediction + self.lambda_acc_gap * (prediction - prediction_anti) - sparsity_loss - continuity_loss
        
        advantages = rewards - baseline # (batch_size,)
        advantages = Variable(advantages.data, requires_grad=False)
        if self.use_cuda:
            advantages = advantages.cuda()
        
        return sup_loss, advantages, rewards, pred_consistency, continuity_loss, sparsity_loss
    
    def get_loss(self, pred_logits, anti_pred_logits, cls_pred_logits, label, z, neg_log_probs, baseline, mask):
        reward_tuple = self.get_advantages(pred_logits, anti_pred_logits, cls_pred_logits,
                                           label, z, neg_log_probs, baseline, mask)
        sup_loss, advantages, rewards, consistency_loss, continuity_loss, sparsity_loss = reward_tuple
        
        # (batch_size, q_length)
        advantages_expand_ = advantages.unsqueeze(-1).expand_as(neg_log_probs)
        rl_loss = torch.sum(neg_log_probs * advantages_expand_ * mask)
        
        return sup_loss, rl_loss, rewards, consistency_loss, continuity_loss, sparsity_loss


    def train_cls_one_step(self, x, label, mask):
 
        self.opt_G_sup.zero_grad()

        word_embeddings = self.embed_layer(x) #(batch_size, length, embedding_dim)
        
        cls_hiddens = self.generator.Classifier_enc(word_embeddings, mask) # (batch_size, hidden_dim, sequence_length)
        max_cls_hidden = torch.max(cls_hiddens + (1 - mask).unsqueeze(1) * self.NEG_INF, dim=2)[0] # (batch_size, hidden_dim)
        cls_predict = self.generator.Classifier_pred(max_cls_hidden)
        
        sup_loss = torch.mean(self.loss_func(cls_predict, label))
        
        losses = {'g_sup_loss':sup_loss.cpu().data}
        
        sup_loss.backward()
        self.opt_G_sup.step()
        
        return losses, cls_predict

    def train_gen_one_step(self, x, label, mask):
        z_baseline = Variable(torch.FloatTensor([float(np.mean(self.z_history_rewards))]))
        if self.use_cuda:
            z_baseline = z_baseline.cuda()
        
        self.opt_G_rl.zero_grad()
        
        predict, anti_predict, cls_predict, z, neg_log_probs = self.forward(x, mask)
        
#         e_loss = torch.mean(self.loss_func(predict, label))
        _, cls_pred = torch.max(cls_predict, dim=1) # (batch_size,)
#         e_loss = torch.mean(self.loss_func(predict, cls_pred)) # e_loss comes from only consistency
        e_loss = (torch.mean(self.loss_func(predict, label)) + torch.mean(self.loss_func(predict, cls_pred))) / 2
        
        # g_sup_loss comes from only cls pred loss
        _, g_rl_loss, z_rewards, consistency_loss, continuity_loss, sparsity_loss = self.get_loss(predict, 
                                                                         anti_predict, 
                                                                         cls_predict, label, z, 
                                                                         neg_log_probs, z_baseline, mask)
        
        losses = {'g_rl_loss':g_rl_loss.cpu().data}

        g_rl_loss.backward()
        self.opt_G_rl.step()
        self.opt_G_rl.zero_grad()
    
        z_batch_reward = np.mean(z_rewards.cpu().data.numpy())
        self.z_history_rewards.append(z_batch_reward)
        
        return losses, predict, anti_predict, cls_predict, z, z_rewards, consistency_loss, continuity_loss, sparsity_loss
    
    def forward_cls(self, x, mask):
        """
        Inputs:
            x -- torch Variable in shape of (batch_size, length)
            mask -- torch Variable in shape of (batch_size, length)
        Outputs:
            predict -- (batch_size, num_label)
            z -- rationale (batch_size, length)
        """        
        word_embeddings = self.embed_layer(x) #(batch_size, length, embedding_dim)
        
        z = torch.ones_like(x).type(torch.cuda.FloatTensor)
        
        predict = self.E_model(word_embeddings, z, mask)

        return predict

    def generate_data(self, batch):
        # sort for rnn happiness
        batch.sort_values("counts", inplace=True, ascending=False)
        
        x_mask = np.stack(batch["mask"], axis=0)
        # drop all zero columns
        zero_col_idxs = np.argwhere(np.all(x_mask[...,:] == 0, axis=0))
        x_mask = np.delete(x_mask, zero_col_idxs, axis=1)

        x_mat = np.stack(batch["tokens"], axis=0)
        # drop all zero columns
        x_mat = np.delete(x_mat, zero_col_idxs, axis=1)

        y_vec = np.stack(batch["labels"], axis=0)
        
        batch_x_ = Variable(torch.from_numpy(x_mat)).to(torch.int64)
        batch_m_ = Variable(torch.from_numpy(x_mask)).type(torch.FloatTensor)
        batch_y_ = Variable(torch.from_numpy(y_vec)).to(torch.int64)

        if self.use_cuda:
            batch_x_ = batch_x_.cuda()
            batch_m_ = batch_m_.cuda()
            batch_y_ = batch_y_.cuda()

        return batch_x_, batch_m_, batch_y_

    def _get_sparsity(self, z, mask):
        mask_z = z * mask
        seq_lengths = torch.sum(mask, dim=1)

        sparsity_ratio = torch.sum(mask_z, dim=-1) / seq_lengths #(batch_size,)
        return sparsity_ratio

    def _get_continuity(self, z, mask):
        mask_z = z * mask
        seq_lengths = torch.sum(mask, dim=1)
        
        mask_z_ = torch.cat([mask_z[:, 1:], mask_z[:, -1:]], dim=-1)
            
        continuity_ratio = torch.sum(torch.abs(mask_z - mask_z_), dim=-1) / seq_lengths #(batch_size,) 
        
        return continuity_ratio

    def display_example(self, x, m, z):
        seq_len = int(m.sum().item())
        ids = x[:seq_len]
        tokens = [self.reverse_word_vocab[i.item()] for i in ids]

        final = ""
        for i in range(len(tokens)):
            if z[i]:
                final += "[" + tokens[i] + "]"
            else:
                final += tokens[i]
            final += " "
        print(final)

    def test(self, df_test, test_batch_size):
        self.eval()

        accuracy = 0
        anti_accuracy = 0
        sparsity_total = 0

        for i in range(len(df_test)//test_batch_size):
            test_batch = df_test.iloc[i*test_batch_size:(i+1)*test_batch_size]
            batch_x_, batch_m_, batch_y_ = self.generate_data(test_batch)
            predict, anti_predict, cls_predict, z, neg_log_probs = self.forward(batch_x_, batch_m_)
            
            # do a softmax on the predicted class probabilities
            _, y_pred = torch.max(predict, dim=1)
            _, anti_y_pred = torch.max(anti_predict, dim=1)
            
            accuracy += (y_pred == batch_y_).sum().item()
            anti_accuracy += (anti_y_pred == batch_y_).sum().item()

            # calculate sparsity
            sparsity_ratios = self._get_sparsity(z, batch_m_)
            sparsity_total += sparsity_ratios.sum().item()

        accuracy = accuracy / len(df_test)
        anti_accuracy = anti_accuracy / len(df_test)
        sparsity = sparsity_total / len(df_test)

        rand_idx = random.randint(0, test_batch_size-1)
        # display an example
        print("Gold Label: ", batch_y_[rand_idx].item(), " Pred label: ", y_pred[rand_idx].item())
        self.display_example(batch_x_[rand_idx], batch_m_[rand_idx], z[rand_idx])

        return accuracy, anti_accuracy, sparsity

    def pretrain_classifier(self, df_train, df_test, batch_size, num_iteration=5, test_iteration=5):
        train_accs = []
        test_accs = []
        best_train_acc = 0.0
        best_test_acc = 0.0
        self.init_optimizers()
        self.init_rl_optimizers()

        for i in tqdm(range(num_iteration)):
            self.train() # pytorch fn; sets module to train mode

            # sample a batch of data
            batch = df_train.sample(batch_size, replace=True)
            batch_x_, batch_m_, batch_y_ = self.generate_data(batch)

            losses, predict = self.train_cls_one_step(batch_x_, batch_y_, batch_m_)

            # calculate classification accuarcy
            _, y_pred = torch.max(predict, dim=1)

            acc = np.float((y_pred == batch_y_).sum().cpu().data.item()) / batch_size
            train_accs.append(acc)

            if acc > best_train_acc:
                best_train_acc = acc

            if (i+1) % test_iteration == 0:
                self.eval() # set module to eval mode
                test_correct = 0.0
                test_total = 0.0
                test_count = 0

                for test_iter in range(len(df_test)//batch_size):
                    test_batch = df_test.sample(batch_size) # TODO: originally used dev batch here?
                    batch_x_, batch_m_, batch_y_ = self.generate_data(test_batch)
                    embeddings = self.embed_layer(batch_x_)
                    _, predict = self.generator(embeddings, batch_m_)

                    _, y_pred = torch.max(predict, dim=1)

                    test_correct += np.float((y_pred == batch_y_).sum().cpu().data.item())
                    test_total += batch_size

                    test_count += batch_size

                    test_accs.append(test_correct / test_total)
                
                if test_correct / test_total > best_test_acc:
                    best_test_acc = test_correct / test_total

                avg_train_accs = sum(train_accs[len(train_accs) - 10:len(train_accs)])/10
                print('train:', avg_train_accs, 'best train acc:', best_train_acc)
                print('test:', test_accs[-1], 'best test:', best_test_acc)

    def fit(self, df_train, batch_size, num_iteration):
        for i in tqdm(range(num_iteration)):
            self.train()
            # sample a batch of data
            batch = df_train.sample(batch_size, replace=True)
            batch_x_, batch_m_, batch_y_ = self.generate_data(batch)

            z_baseline = Variable(torch.FloatTensor([float(np.mean(self.z_history_rewards))]))
            if self.use_cuda:
                z_baseline = z_baseline.cuda()

            losses, predict, anti_predict, cls_predict, z, z_rewards, consistency_loss, continuity_loss, sparsity_loss = self.train_one_step(\
            batch_x_, batch_y_, z_baseline, batch_m_)

            z_batch_reward = np.mean(z_rewards.cpu().data.numpy())
            self.z_history_rewards.append(z_batch_reward)

            # calculate classification accuarcy
            _, y_pred = torch.max(predict, dim=1)

            acc = np.float((y_pred == batch_y_).sum().cpu().data.item()) / self.batch_size
            self.train_accs.append(acc)
        return losses, predict, anti_predict, cls_predict, z, z_rewards, consistency_loss, continuity_loss, sparsity_loss

    def fit_test(self, df_train, df_test, batch_size, pretrain_cls=True, num_iteration=40000, test_iteration=200):
        '''
            Training pipeline -- iteratively trains and tests the model, saving "best models" as necessary
        '''
        # configure folder for saving models and stats
        current_datetime = datetime.now().strftime("%m_%d_%y_%H_%M_%S")
        if self.args.save_best_model:
            model_folder_path = os.path.join(self.args.save_path, self.args.model_prefix + current_datetime + "training_run")
            os.mkdir(model_folder_path)
            log_filepath = os.path.join(model_folder_path, "training_stats.txt")
            logging.basicConfig(filename=log_filepath, filemode='a', level=logging.INFO)
        
        if pretrain_cls:
            print('pre-training the classifier')
            self.pretrain_classifier(df_train, df_test, batch_size)

        best_test_acc = 0.0
        for _ in range(num_iteration//test_iteration):
            losses, predict, anti_predict, cls_predict, z, z_rewards, consistency_loss, continuity_loss, sparsity_loss = self.fit(df_train, batch_size, test_iteration)
            avg_train_acc = sum(self.train_accs[len(self.train_accs) - 20: len(self.train_accs)]) / 20
            print("\nAvg. train accuracy: ", avg_train_acc)

            test_acc, test_anti_acc, test_sparsity = self.test(df_test, batch_size)
            if test_acc > best_test_acc:
                if self.args.save_best_model:
                    print("saving best model and model stats")
                    current_datetime = datetime.now().strftime("%m_%d_%y_%H_%M_%S")
                    # save model
                    torch.save(self.state_dict(), os.path.join(model_folder_path, self.args.model_prefix + current_datetime + ".pth"))
                    # save stats
                    logging.info('best model at time ' + current_datetime)
                    logging.info('sparsity lambda: %.4f'%(self.lambda_sparsity))
                    logging.info('last train acc: %.4f, avg train acc: %.4f'%(self.train_accs[-1], avg_train_acc))
                    logging.info('last test acc: %.4f, previous best test acc: %.4f, last anti test acc: %.4f'%(test_acc,  best_test_acc, test_anti_acc))
                    logging.info('last test sparsity: %.4f'%test_sparsity)
                    logging.info('supervised_loss: %.4f, sparsity_loss: %.4f, continuity_loss: %.4f'%(losses['e_loss'], torch.mean(sparsity_loss).cpu().data, torch.mean(continuity_loss).cpu().data))
                best_test_acc = test_acc