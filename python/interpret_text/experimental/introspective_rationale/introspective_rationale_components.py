import os
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm

from interpret_text.experimental.common.utils_introspective_rationale import generate_data


class ClassifierWrapper():
    """Wrapper to provide a common interfaces among different classifier modules
    """

    def __init__(self, args, model):
        """Initialize an instance of the wrapper

        :param args: arguments containing training and structure parameters
        :type args: ModelArguments
        :param model: A classifier module, ex. BERT or RNN classifier module
        :type model: BertForSequenceClassification or ClassifierModule
        """
        self.args = args
        self.model = model
        self.opt = None

        self.num_epochs = args.num_pretrain_epochs
        self.epochs_since_improv = 0
        self.best_test_acc = 0
        self.avg_accuracy = 0
        self.test_accs = []
        self.train_accs = []

        self.loss_func = nn.CrossEntropyLoss(reduction="none")

    def init_optimizer(self):
        """Initialize the classifier's optimizer
        """
        self.opt = torch.optim.Adam(filter(lambda x: x.requires_grad,
                                           self.model.parameters()),
                                    lr=self.args.lr)

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
        self.model.eval()
        accuracy = 0
        for i in range(len(df_test) // self.args.test_batch_size):
            test_batch = df_test.iloc[
                i * self.args.test_batch_size: (i + 1)
                * self.args.test_batch_size
            ]
            batch_dict = generate_data(test_batch, self.args.cuda)
            batch_x_ = batch_dict["x"]
            batch_m_ = batch_dict["m"]
            batch_y_ = batch_dict["y"]
            predict, _, _ = self.model(batch_x_, batch_m_)

            # do a softmax on the predicted class probabilities
            _, y_pred = torch.max(predict, dim=1)

            accuracy += (y_pred == batch_y_).sum().item()

        self.avg_accuracy = accuracy / len(df_test)
        self.test_accs.append(self.avg_accuracy)

        if verbosity > 0:
            logging.info("train acc: %.4f, test acc: %.4f" %
                         (self.train_accs[-1], self.avg_accuracy))

        if self.args.save_best_model:
            if self.avg_accuracy > self.best_test_acc:
                logging.info("saving best classifier model and model stats")
                # save model
                torch.save(
                    self.model.state_dict(),
                    os.path.join(
                        self.args.model_folder_path,
                        self.args.model_prefix + "gen_classifier.pth",
                    ),
                )

        if self.avg_accuracy > self.best_test_acc:
            self.best_test_acc = self.avg_accuracy
            self.epochs_since_improv = 0
        else:
            self.epochs_since_improv += 1

    def _train_one_step(self, X_tokens, label, X_mask):
        """Train the classifier for one optimization step.

        :param X_tokens: Tokenized and embedded training example
        :type X_tokens: torch.int64
        :param label: Label of the training example
        :type label: torch.int64
        :param X_mask: Mask differentiating tokens vs not tokens
        :type X_mask: torch.FloatTensor
        :return: losses, classifier prediction logits
        :rtype: tuple
        """
        self.opt.zero_grad()
        self.model.zero_grad()

        cls_predict_logits, _, _ = self.model(
            X_tokens, attention_mask=X_mask
        )  # dimensions: (batch_size, hidden_dim, sequence_length)

        sup_loss = torch.mean(self.loss_func(cls_predict_logits, label))
        losses = {"g_sup_loss": sup_loss.cpu().data}
        sup_loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        self.opt.step()
        return losses, cls_predict_logits

    def fit(self, df_train, df_test):
        """Train the classifier on the training data, with testing
        at the end of every epoch.

        :param df_train: training data containing labels, lists of word token
        ids, pad/word masks, and token counts for each training example
        :type df_train: pd.DataFrame
        :param df_test: testing data containing labels, lists of word token
        ids, pad/word masks, and token counts for each testing example
        :type df_test: pd.DataFrame
        """
        self.init_optimizer()

        total_train = len(df_train)
        indices = np.array(list(range(0, total_train)))

        for i in tqdm(range(self.num_epochs)):
            self.model.train()  # pytorch fn; sets module to train mode

            # shuffle the epoch
            np.random.shuffle(indices)

            total_train_acc = 0
            for i in range(total_train // self.args.train_batch_size):
                # sample a batch of data
                start = i * self.args.train_batch_size
                end = min((i + 1) * self.args.train_batch_size, total_train)
                batch = df_train.loc[indices[start:end]]
                batch_dict = generate_data(batch, self.args.cuda)
                batch_x_ = batch_dict["x"]
                batch_m_ = batch_dict["m"]
                batch_y_ = batch_dict["y"]

                losses, predict = self._train_one_step(
                    batch_x_, batch_y_, batch_m_
                )

                # calculate classification accuarcy
                _, y_pred = torch.max(predict, dim=1)

                acc = np.float((y_pred == batch_y_).sum().cpu().data.item())
                total_train_acc += acc

            total_acc_percent = total_train_acc / total_train
            self.train_accs.append(total_acc_percent)

            self.test(df_test)
            # stop training if there have been no improvements
            if self.epochs_since_improv > self.args.training_stop_thresh:
                break


# Modules that can be used in the three player introspective model
class RnnModel(nn.Module):
    """RNN Module
    """

    def __init__(self, input_dim, hidden_dim, layer_num, dropout_rate):
        """Initialize an RNN.
        :param input_dim: dimension of input
        :type input_dim: int
        :param hidden_dim: dimension of filters
        :type hidden_dim: int
        :param layer_num: number of RNN layers
        :type layer_num: int
        :param dropout_rate: dropout rate
        :type dropout_rate: float
        """
        super(RnnModel, self).__init__()
        self.rnn_layer = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim // 2,
            num_layers=layer_num,
            bidirectional=True,
            dropout=dropout_rate,
        )

    def forward(self, embeddings, mask=None, h0=None):
        """Forward pass in the RNN.
        :param embeddings: sequence of word embeddings with dimension
            (batch_size, sequence_length, embedding_dim)
        :type embeddings: torch.FloatTensor
        :param mask: a float tensor of masks with dimension
            (batch_size, length), defaults to None
        :type mask: torch.FloatTensor, optional
        :param h0: initial RNN weights with dimension
            (num_layers * num_directions, batch, hidden_size), defaults to None
        :type h0: torch.FloatTensor, optional
        :return: hiddens, a sentence embedding tensor with dimension
            (batch_size, hidden_dim, sequence_length)
        :rtype: torch.FloatTensor
        """
        # dimensions: (sequence_length, batch_size, embedding_dim)
        embeddings_ = embeddings.transpose(0, 1)

        if mask is not None:
            seq_lengths = list(torch.sum(mask, dim=1).cpu().data.numpy())
            seq_lengths = list(map(int, seq_lengths))
            inputs_ = torch.nn.utils.rnn.pack_padded_sequence(
                embeddings_, seq_lengths
            )
        else:
            inputs_ = embeddings_

        if h0 is not None:
            hidden, _ = self.rnn_layer(inputs_, h0)
        else:
            # hidden's dimensions:
            # (sequence_length, batch_size, hidden_dim (* 2 if bidirectional))
            hidden, _ = self.rnn_layer(inputs_)

        if mask is not None:
            # hidden's dimensions: (length, batch_size, hidden_dim)
            hidden, _ = torch.nn.utils.rnn.pad_packed_sequence(hidden)

        # output dimensions: (batch_size, hidden_dim, sequence_length)
        return hidden.permute(1, 2, 0)


class ClassifierModule(nn.Module):
    """Module for classifying text used in original paper code.
    """

    def __init__(self, args, word_vocab):
        """Initialize a ClassifierModule.

        :param args: model structure parameters and hyperparameters
        :type args: ModelArguments
        :param word_vocab: a mapping of a set of words (keys) to
            indices (values)
        :type word_vocab: dict
        """
        super(ClassifierModule, self).__init__()
        self.args = args
        self.encoder = RnnModel(
            self.args.embedding_dim,
            self.args.hidden_dim,
            self.args.layer_num,
            self.args.dropout_rate,
        )
        self.predictor = nn.Linear(self.args.hidden_dim, self.args.num_labels)

        self.input_dim = args.embedding_dim
        self.embedding_path = args.embedding_path
        self.fine_tuning = args.fine_tuning

        self.init_embedding_layer(word_vocab)
        self.NEG_INF = -1.0e6

    def init_embedding_layer(self, word_vocab):
        """Initialize the layer that embeds tokens according to a provided embedding

        :param word_vocab: a mapping of a set of words (keys) to
            indices (values)
        :type word_vocab: dict
        """
        # get initial vocab embeddings
        vocab_size = len(word_vocab)
        # initialize a numpy embedding matrix
        embeddings = 0.1 * np.random.randn(vocab_size, self.input_dim).astype(
            np.float32
        )

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
            logging.info("%d words have been switched." % counter)
        else:
            logging.info("embedding is initialized fully randomly.")

        # initialize embedding layer
        self.embed_layer = nn.Embedding(vocab_size, self.input_dim)
        self.embed_layer.weight.data = torch.from_numpy(embeddings)
        self.embed_layer.weight.requires_grad = self.fine_tuning

    def forward(self, X_tokens, attention_mask, z=None):
        """Forward pass in the classifier module
        :param X_tokens: tokenized and embedded text with shape
            (batch_size, length, embed_dim)
        :type X_tokens: torch Variable
        :param attention_mask: mask indicating word tokens (1) and padding (0)
            with shape (batch_size, length)
        :type attention_mask: torch.FloatTensor
        :param z: chosen rationales for sentence tokens (whether a given token
            is important for classification)
            with shape (batch_size, length), defaults to None
        :type z: torch.FloatTensor, optional
        :return: prediction (batch_size, num_label), word_embeddings, encoded
            input, None
        :rtype: tuple
        """
        word_embeddings = self.embed_layer(X_tokens)
        if z is None:
            z = torch.ones_like(X_tokens)
            if torch.cuda.is_available():
                z = z.type(torch.cuda.FloatTensor)
            else:
                z = z.type(torch.FloatTensor)

        masked_input = word_embeddings * z.unsqueeze(-1)
        hiddens = self.encoder(masked_input, attention_mask)

        max_hidden = torch.max(
            hiddens + (1 - attention_mask * z).unsqueeze(1) * self.NEG_INF,
            dim=2,
        )[0]

        predict = self.predictor(max_hidden)
        # the last one is for attention in the BERT model
        return predict, [word_embeddings, hiddens], None


# extra classes needed to make model introspective
class DepGenerator(nn.Module):
    """Rationale generator module
    """

    def __init__(self, input_dim, hidden_dim, layer_num, dropout_rate):
        """
        :param input_dim: dimension of input
        :type input_dim: int
        :param hidden_dim: dimension of filters
        :type hidden_dim: int
        :param layer_num: number of RNN layers
        :type layer_num: int
        :param dropout_rate: dropout rate of RNN
        :type dropout_rate: float
        """
        super(DepGenerator, self).__init__()

        self.generator_model = RnnModel(
            input_dim, hidden_dim, layer_num, dropout_rate
        )
        # rationale has dimension (num_tokens, 2)
        self.output_layer = nn.Linear(hidden_dim, 2)

    def forward(self, X_embeddings, h0=None, mask=None):
        """Forward pass in the DepGenerator

        :param X_embeddings: input sequence of word embeddings
        :type X_embeddings: (batch_size, sequence_length, embedding_dim)
        :param h0: initial RNN weights, defaults to None
        :type h0: torch.FloatTensor, optional
        :param mask: mask indicating word tokens (1) and padding (0)
            with shape (batch_size, length), defaults to None
        :type mask: torch.FloatTensor, optional
        :return: scores of the importance of each word in X_embeddings
        :rtype: torch.FloatTensor
        """
        """
        Outputs:
            z -- output rationale, "binary" mask, (batch_size, sequence_length)
        """
        # hiddens' dimensions: (batch_size, sequence_length, hidden_dim)
        hiddens = (
            self.generator_model(X_embeddings, mask, h0)
            .transpose(1, 2)
            .contiguous()
        )
        # scores' dimensions: (batch_size, sequence_length, 2)
        scores = self.output_layer(hiddens)
        return scores


class IntrospectionGeneratorModule(nn.Module):
    """Introspective rationale generator used in paper
    """

    def __init__(self, args, classifier):
        """Initialize the IntrospectionGeneratorModule

        :param args: model structure parameters and hyperparameters
        :type args: ModelArguments
        :param classifier: an instantiated classifier module with an embedding
            layer and forward method
        :type classifier: an instantiated classifier module
            e.g. ClassifierModule
        """
        super(IntrospectionGeneratorModule, self).__init__()
        self.args = args

        # for initializing RNN and DepGenerator
        self.input_dim = args.gen_embedding_dim
        self.hidden_dim = args.hidden_dim
        self.layer_num = args.layer_num
        self.dropout_rate = args.dropout_rate
        # for embedding labels
        self.num_labels = args.num_labels
        self.label_embedding_dim = args.label_embedding_dim

        # for training
        self.fixed_classifier = args.fixed_classifier

        self.NEG_INF = -1.0e6
        # should be shared with the Classifier_pred weights
        self.lab_embed_layer = self._create_label_embed_layer()

        # baseline classification model
        self.classifier = classifier

        self.Transformation = nn.Sequential()
        self.Transformation.add_module(
            "linear_layer",
            nn.Linear(
                self.hidden_dim + self.label_embedding_dim,
                self.hidden_dim // 2,
            ),
        )
        self.Transformation.add_module("tanh_layer", nn.Tanh())
        self.Generator = DepGenerator(
            self.input_dim,
            self.hidden_dim,
            self.layer_num,
            self.dropout_rate,
        )

    def _create_label_embed_layer(self):
        embed_layer = nn.Embedding(self.num_labels, self.label_embedding_dim)
        embed_layer.weight.data.normal_(mean=0, std=0.1)
        embed_layer.weight.requires_grad = True
        return embed_layer

    def forward(self, X_tokens, mask):
        """Forward pass of the introspection generator module

        :param X_tokens: tokenized and embedded text with shape
            (batch_size, length, embed_dim)
        :type X_tokens: torch Variable
        :param mask: mask indicating word tokens (1) and padding (0)
            with shape (batch_size, length)
        :type mask: torch.FloatTensor
        :return: z_scores_ (scores of token importances),
        cls_pred_logits (internal classifier predictions),
        word_embeddings (embedded tokenized text input)
        :rtype: tuple()
        """
        cls_pred_logits, hidden_states, _ = self.classifier(
            X_tokens, attention_mask=mask
        )

        # hidden states must be in shape (batch_size, hidden_dim, length)
        # RNN returns (batch_size, hidden_dim, length)
        # BERT returns (batch_size, length, hidden_dim)
        last_hidden_state = hidden_states[-1]
        if last_hidden_state.shape[1] != self.hidden_dim:
            last_hidden_state = hidden_states[-1].transpose(1, 2)

        # max_cls_hidden dimensions: (batch_size, hidden_dim)
        max_cls_hidden = torch.max(
            last_hidden_state + (1 - mask).unsqueeze(1) * self.NEG_INF, dim=2
        )[0]
        if self.fixed_classifier:
            max_cls_hidden = Variable(max_cls_hidden.data)

        word_embeddings = hidden_states[0]

        _, cls_pred = torch.max(cls_pred_logits, dim=1)

        # classifier label embedding dimensions: (batch_size, lab_emb_dim)
        cls_lab_embeddings = self.lab_embed_layer(cls_pred)

        # initial h0 dimensions: (batch_size, hidden_dim / 2)
        init_h0 = self.Transformation(
            torch.cat([max_cls_hidden, cls_lab_embeddings], dim=1)
        )
        # initial h0 dimensions: (2, batch_size, hidden_dim / 2)
        init_h0 = (
            init_h0.unsqueeze(0)
            .expand(2, init_h0.size(0), init_h0.size(1))
            .contiguous()
        )

        # z_scores' dimensions: (batch_size, length, 2)
        z_scores_ = self.Generator(word_embeddings, mask=mask, h0=init_h0)
        z_scores_[:, :, 1] = z_scores_[:, :, 1] + (1 - mask) * self.NEG_INF

        return z_scores_, cls_pred_logits, word_embeddings
