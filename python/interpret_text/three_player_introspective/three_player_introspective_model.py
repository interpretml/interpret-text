import logging
import os
import random
from collections import deque
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm

from interpret_text.common.utils_three_player import generate_data

class ThreePlayerIntrospectiveModel(nn.Module):
    """flattening the HardIntrospectionRationale3PlayerClassificationModel ->
       HardRationale3PlayerClassificationModel ->
       Rationale3PlayerClassificationModel dependency structure
       from original paper code"""

    def __init__(
        self,
        args,
        preprocessor,
        explainer,
        anti_explainer,
        generator,
        classifier,
    ):
        """Initializes the model, including the explainer, anti-rationale explainer
        """
        super(ThreePlayerIntrospectiveModel, self).__init__()
        self.args = args
        self.lambda_sparsity = args.lambda_sparsity
        self.lambda_continuity = args.lambda_continuity
        self.lambda_anti = args.lambda_anti
        self.hidden_dim = args.hidden_dim
        self.input_dim = args.embedding_dim
        self.embedding_path = args.embedding_path
        self.fine_tuning = args.fine_tuning
        self.exploration_rate = args.exploration_rate
        self.lambda_acc_gap = args.lambda_acc_gap
        self.fixed_classifier = args.fixed_classifier
        self.count_tokens = args.count_tokens  # used to calc sparsity loss
        self.count_pieces = args.count_pieces  # used to calc sparsity loss
        self.use_cuda = args.cuda
        self.train_batch_size = args.train_batch_size
        self.test_batch_size = args.test_batch_size
        self.num_epochs = args.num_epochs
        self.lr = args.lr
        self.num_labels = args.num_labels

        # initialize model components
        self.E_model = explainer
        self.E_anti_model = anti_explainer
        self.gen_C_model = classifier
        self.generator = generator

        self.preprocessor = preprocessor

        self.NEG_INF = -1.0e6
        self.loss_func = nn.CrossEntropyLoss(reduce=False)
        self.z_history_rewards = deque([0], maxlen=200)
        self.train_accs = []

        # initialize optimizers
        self.init_optimizers()
        self.init_rl_optimizers()

        # training_stop_thresh epochs: max number of epochs
        # allowed to train since improvement before fit() stops
        self.training_stop_thresh = 5
        self.epochs_since_improv = 0

        self.train_accs = []
        self.test_accs = []
        self.test_losses = []
        self.best_test_acc = 0

    def init_optimizers(self):
        self.opt_E = torch.optim.Adam(
            filter(lambda x: x.requires_grad, self.E_model.parameters()),
            lr=self.lr,
        )
        self.opt_E_anti = torch.optim.Adam(
            filter(lambda x: x.requires_grad, self.E_anti_model.parameters()),
            lr=self.lr,
        )

    def init_rl_optimizers(self):
        self.opt_G_sup = torch.optim.Adam(
            filter(lambda x: x.requires_grad, self.generator.parameters()),
            lr=self.lr,
        )
        self.opt_G_rl = torch.optim.Adam(
            filter(lambda x: x.requires_grad, self.generator.parameters()),
            lr=self.lr * 0.1,
        )

    def _generate_rationales(self, z_prob_):
        """
        Input:
            z_prob_ -- (num_rows, length, 2)
        Output:
            z -- (num_rows, length)
        """
        z_prob__ = z_prob_.view(-1, 2)  # (num_rows * length, 2)

        # sample actions
        sampler = torch.distributions.Categorical(z_prob__)
        if self.training:
            z_ = sampler.sample()  # (num_rows * p_length,)
        else:
            z_ = torch.max(z_prob__, dim=-1)[1]

        # (num_rows, length)
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

    def _count_regularization_baos_for_both(
        self, z, count_tokens, count_pieces, mask=None
    ):
        """
        Compute regularization loss, based on a given rationale sequence
        Use Yujia's formulation
        Inputs:
            z -- torch variable, "binary" rationale, (batch_size,
                sequence_length)
            percentage -- the percentage of words to keep
        Outputs:
            a loss value that contains two parts:
            continuity_loss -- sum_{i} | z_{i-1} - z_{i} |
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

        continuity_ratio = (
            torch.sum(torch.abs(mask_z - mask_z_), dim=-1) / seq_lengths
        )  # (batch_size,)
        percentage = count_pieces * 2 / seq_lengths
        continuity_loss = torch.abs(continuity_ratio - percentage)

        sparsity_ratio = (
            torch.sum(mask_z, dim=-1) / seq_lengths
        )  # (batch_size,)
        percentage = count_tokens / seq_lengths  # (batch_size,)
        sparsity_loss = torch.abs(sparsity_ratio - percentage)

        return continuity_loss, sparsity_loss

    def _train_one_step(self, X_tokens, label, baseline, mask):
        self.opt_E_anti.zero_grad()
        self.opt_E.zero_grad()
        self.opt_G_sup.zero_grad()
        self.opt_G_rl.zero_grad()

        forward_dict = self.forward(
            X_tokens, mask
        )
        predict = forward_dict["predict"]
        anti_predict = forward_dict["anti_predict"]
        cls_predict = forward_dict["cls_predict"]
        z = forward_dict["z"]
        neg_log_probs = forward_dict["neg_log_probs"]

        e_loss_anti = torch.mean(self.loss_func(anti_predict, label))
        _, cls_pred = torch.max(cls_predict, dim=1)  # (batch_size,)
        e_loss = (
            torch.mean(self.loss_func(predict, label))
            + torch.mean(self.loss_func(predict, cls_pred))
        ) / 2

        # g_sup_loss comes from only cls pred loss
        (
            g_sup_loss,
            g_rl_loss,
            rewards,
            consistency_loss,
            continuity_loss,
            sparsity_loss,
        ) = self._get_loss(
            predict,
            anti_predict,
            cls_predict,
            label,
            z,
            neg_log_probs,
            baseline,
            mask,
        )

        losses = {
            "e_loss": e_loss.cpu().data,
            "e_loss_anti": e_loss_anti.cpu().data,
            "g_sup_loss": g_sup_loss.cpu().data,
            "g_rl_loss": g_rl_loss.cpu().data,
            "consistency_loss": consistency_loss,
            "continuity_loss": continuity_loss,
            "sparsity_loss": sparsity_loss
        }

        e_loss_anti.backward(retain_graph=True)
        self.opt_E_anti.step()
        self.opt_E_anti.zero_grad()

        e_loss.backward(retain_graph=True)
        self.opt_E.step()
        self.opt_E.zero_grad()

        if not self.fixed_classifier:
            g_sup_loss.backward(retain_graph=True)
            self.opt_G_sup.step()
            self.opt_G_sup.zero_grad()

        g_rl_loss.backward(retain_graph=True)
        self.opt_G_rl.step()
        self.opt_G_rl.zero_grad()

        return (
            losses,
            predict,
            anti_predict,
            cls_predict,
            z,
            rewards
        )

    def forward(self, X_tokens, X_mask):
        """
        Inputs:
            x -- torch Variable in shape of (batch_size, length)
            mask -- torch Variable in shape of (batch_size, length)
        Outputs:
            predict -- (batch_size, num_label)
            z -- rationale (batch_size, length)
        """
        z_scores_, cls_predict, word_embeddings = self.generator(
            X_tokens, X_mask
        )

        z_probs_ = F.softmax(z_scores_, dim=-1)

        z_probs_ = (
            X_mask.unsqueeze(-1)
            * (
                (1 - self.exploration_rate) * z_probs_
                + self.exploration_rate / z_probs_.size(-1)
            )
        ) + ((1 - X_mask.unsqueeze(-1)) * z_probs_)

        z, neg_log_probs = self._generate_rationales(
            z_probs_
        )  # (batch_size, length)

        if not self.args.BERT:
            predict = self.E_model(X_tokens, X_mask, z)[
                0
            ]  # the first output are the logits
            anti_predict = self.E_anti_model(X_tokens, X_mask, (1 - z))[0]
        else:
            predict = self.E_model(X_tokens, attention_mask=z)[
                0
            ]  # the first output are the logits
            anti_predict = self.E_anti_model(X_tokens, attention_mask=(1 - z))[
                0
            ]
        return {"predict": predict,
                "anti_predict": anti_predict,
                "cls_predict": cls_predict,
                "z": z,
                "neg_log_probs": neg_log_probs}

    def get_z_scores(self, df_test):
        """
        Inputs:
            x -- torch Variable in shape of (batch_size, length)
            mask -- torch Variable in shape of (batch_size, length)
        Outputs:
            z_scores -- non-softmaxed rationale, (batch_size, length)
            cls_predict -- prediction of generator's classifier,
                (batch_size, num_label)
        """
        batch_dict = generate_data(df_test, self.use_cuda)
        x_tokens = batch_dict["x"]
        mask = batch_dict["m"]
        z_scores, _, _ = self.generator(x_tokens, mask)
        z_scores = F.softmax(z_scores, dim=-1)

        return z_scores

    def _get_advantages(
        self,
        pred_logits,
        anti_pred_logits,
        cls_pred_logits,
        label,
        z,
        neg_log_probs,
        baseline,
        mask,
    ):
        """
        Input:
            z -- (batch_size, length)
        """

        # supervised loss
        prediction_loss = self.loss_func(
            cls_pred_logits, label
        )  # (batch_size, )
        sup_loss = torch.mean(prediction_loss)

        # total loss of accuracy (not batchwise)
        _, cls_pred = torch.max(cls_pred_logits, dim=1)  # (batch_size,)
        _, ver_pred = torch.max(pred_logits, dim=1)  # (batch_size,)

        prediction = (ver_pred == label).type(torch.FloatTensor)
        pred_consistency = (ver_pred == cls_pred).type(torch.FloatTensor)

        _, anti_pred = torch.max(anti_pred_logits, dim=1)
        prediction_anti = (anti_pred == label).type(
            torch.FloatTensor
        ) * self.lambda_anti

        if self.use_cuda:
            prediction = prediction.cuda()  # (batch_size,)
            pred_consistency = pred_consistency.cuda()  # (batch_size,)
            prediction_anti = prediction_anti.cuda()

        (
            continuity_loss,
            sparsity_loss,
        ) = self._count_regularization_baos_for_both(
            z, self.count_tokens, self.count_pieces, mask
        )

        continuity_loss = continuity_loss * self.lambda_continuity
        sparsity_loss = sparsity_loss * self.lambda_sparsity

        # batch RL reward
        # rewards = (prediction + pred_consistency) *
        #           self.args.lambda_pos_reward -
        #           prediction_anti - sparsity_loss - continuity_loss
        rewards = (
            0.1 * prediction
            + self.lambda_acc_gap * (prediction - prediction_anti)
            - sparsity_loss
            - continuity_loss
        )

        advantages = rewards - baseline  # (batch_size,)
        advantages = Variable(advantages.data, requires_grad=False)
        if self.use_cuda:
            advantages = advantages.cuda()

        return (
            sup_loss,
            advantages,
            rewards,
            pred_consistency,
            continuity_loss,
            sparsity_loss,
        )

    def _get_loss(
        self,
        pred_logits,
        anti_pred_logits,
        cls_pred_logits,
        label,
        z,
        neg_log_probs,
        baseline,
        mask,
    ):
        reward_tuple = self._get_advantages(
            pred_logits,
            anti_pred_logits,
            cls_pred_logits,
            label,
            z,
            neg_log_probs,
            baseline,
            mask,
        )
        (
            sup_loss,
            advantages,
            rewards,
            consistency_loss,
            continuity_loss,
            sparsity_loss,
        ) = reward_tuple

        # (batch_size, q_length)
        advantages_expand_ = advantages.unsqueeze(-1).expand_as(neg_log_probs)
        rl_loss = torch.sum(neg_log_probs * advantages_expand_ * mask)

        return (
            sup_loss,
            rl_loss,
            rewards,
            consistency_loss,
            continuity_loss,
            sparsity_loss,
        )

    def _get_sparsity(self, z, mask):
        mask_z = z * mask
        seq_lengths = torch.sum(mask, dim=1)

        sparsity_ratio = (
            torch.sum(mask_z, dim=-1) / seq_lengths
        )  # (batch_size,)
        return sparsity_ratio

    def _get_continuity(self, z, mask):
        mask_z = z * mask
        seq_lengths = torch.sum(mask, dim=1)

        mask_z_ = torch.cat([mask_z[:, 1:], mask_z[:, -1:]], dim=-1)

        continuity_ratio = (
            torch.sum(torch.abs(mask_z - mask_z_), dim=-1) / seq_lengths
        )  # (batch_size,)

        return continuity_ratio

    def display_example(self, x, m, z):
        seq_len = int(m.sum().item())
        ids = x[:seq_len]
        tokens = self.preprocessor.decode_single(ids)

        final = ""
        for i in range(len(tokens)):
            if z[i]:
                final += "[" + tokens[i] + "]"
            else:
                final += tokens[i]
            final += " "
        return final

    def test(self, df_test, verbosity=2):
        """Calculate and store as model attributes:
        Average classification accuracy using rationales (self.avg_accuracy),
        Average classification accuracy rationale complements
            (self.anti_accuracy)
        Average sparsity of rationales (self.avg_sparsity)

        :param df_test: dataframe containing test data labels, tokens, masks,
            and counts
        :type df_test: pandas dataframe
        :param verbosity: {0, 1, 2}, default 2
            If 0, does not log any output
            If 1, logs accuracy, anti-rationale accuracy, sparsity, and
            continuity scores
            If 2, displays a random test example with rationale and
            classification
        :type verbosity: int, optional
        """
        self.eval()

        accuracy = 0
        anti_accuracy = 0
        sparsity_total = 0
        cont_total = 0

        for i in range(len(df_test) // self.test_batch_size):
            test_batch = df_test.iloc[
                i * self.test_batch_size: (i + 1) * self.test_batch_size
            ]
            batch_dict = generate_data(test_batch, self.use_cuda)
            batch_x_ = batch_dict["x"]
            batch_m_ = batch_dict["m"]
            batch_y_ = batch_dict["y"]
            forward_dict = self.forward(batch_x_, batch_m_)
            predict = forward_dict["predict"]
            anti_predict = forward_dict["anti_predict"]
            z = forward_dict["z"]

            # do a softmax on the predicted class probabilities
            _, y_pred = torch.max(predict, dim=1)
            _, anti_y_pred = torch.max(anti_predict, dim=1)

            accuracy += (y_pred == batch_y_).sum().item()
            anti_accuracy += (anti_y_pred == batch_y_).sum().item()

            # calculate sparsity
            sparsity_ratios = self._get_sparsity(z, batch_m_)
            sparsity_total += sparsity_ratios.sum().item()

            cont_ratios = self._get_continuity(z, batch_m_)
            cont_total += cont_ratios.sum().item()

        self.avg_accuracy = accuracy / len(df_test)
        self.test_accs.append(self.avg_accuracy)
        self.avg_anti_accuracy = anti_accuracy / len(df_test)
        self.avg_sparsity = sparsity_total / len(df_test)
        self.avg_continuity = cont_total / len(df_test)

        if verbosity > 0:
            logging.info("test acc: %.4f test anti acc: %.4f" %
                         (self.avg_accuracy, self.avg_anti_accuracy))
            logging.info("test sparsity: %.4f test continuity: %.4f" %
                         (self.avg_sparsity, self.avg_continuity))

        if verbosity > 1:
            rand_idx = random.randint(0, self.test_batch_size - 1)
            # display a random example
            logging.info(
                "Gold Label: " + str(batch_y_[rand_idx].item()) +
                " Pred label: " + str(y_pred[rand_idx].item()))
            logging.info(self.display_example(
                batch_x_[rand_idx], batch_m_[rand_idx], z[rand_idx]
            ))

        if self.args.save_best_model:
            if self.avg_accuracy > self.best_test_acc:
                logging.info("saving best model and model stats")
                # save model
                torch.save(
                    self.state_dict(),
                    os.path.join(
                        self.args.model_folder_path,
                        self.args.model_prefix + ".pth",
                    ),
                )

        if self.best_test_acc > self.avg_accuracy:
            self.best_test_acc = self.avg_accuracy
            self.epochs_since_improv = 0
        else:
            self.epochs_since_improv += 1

    def fit(self, df_train, df_test):
        self.init_optimizers()
        self.init_rl_optimizers()

        total_train = len(df_train)
        indices = np.array(list(range(0, total_train)))

        for i in tqdm(range(self.num_epochs)):
            self.train()  # pytorch fn; sets module to train mode

            # shuffle the data in this epoch
            np.random.shuffle(indices)

            total_train_acc = 0
            for i in range(total_train//self.train_batch_size):
                # sample a batch of data
                start = i*self.train_batch_size
                end = min((i+1)*self.train_batch_size, total_train)
                batch = df_train.loc[indices[start:end]]
                batch_dict = generate_data(batch, self.use_cuda)
                batch_x_ = batch_dict["x"]
                batch_m_ = batch_dict["m"]
                batch_y_ = batch_dict["y"]

                z_baseline = Variable(
                    torch.FloatTensor([float(np.mean(self.z_history_rewards))])
                )
                if self.use_cuda:
                    z_baseline = z_baseline.cuda()

                losses, predict, anti_predict, cls_predict, z, z_rewards =\
                    self._train_one_step(batch_x_,
                                         batch_y_,
                                         z_baseline,
                                         batch_m_)

                z_batch_reward = np.mean(z_rewards.cpu().data.numpy())
                self.z_history_rewards.append(z_batch_reward)

                # calculate classification accuarcy
                _, y_pred = torch.max(predict, dim=1)

                acc = np.float((y_pred == batch_y_).sum().cpu().data.item())
                total_train_acc += acc

            total_acc_percent = total_train_acc / total_train
            self.train_accs.append(total_acc_percent)

            self.test(df_test)
