import numpy as np
import torch
from torch.autograd import Variable

CLASSIFIER_TYPE_BERT = "BERT"
CLASSIFIER_TYPE_BERT_RNN = "BERT_RNN"
CLASSIFIER_TYPE_RNN = "RNN"


def generate_data(batch, use_cuda):
    """Create a formatted and ordered data batch to use in the
    three player model.

    :param batch: A pandas dataframe containing the tokens, masks, counts, and
        labels associated with a batch of data
    :type batch: DataFrame
    :param use_cuda: whether to use CUDA
    :type use_cuda: bool
    :return: formatted and ordered tokens (x), masks (m), and
        labels (y) associated with a batch of data
    :rtype: dict
    """
    # sort for rnn happiness
    batch.sort_values("counts", inplace=True, ascending=False)

    x_mask = np.stack(batch["mask"], axis=0)
    # drop all zero columns
    zero_col_idxs = np.argwhere(np.all(x_mask[..., :] == 0, axis=0))
    x_mask = np.delete(x_mask, zero_col_idxs, axis=1)

    x_mat = np.stack(batch["tokens"], axis=0)
    # drop all zero columns
    x_mat = np.delete(x_mat, zero_col_idxs, axis=1)

    y_vec = np.stack(batch["labels"], axis=0)

    batch_x_ = Variable(torch.from_numpy(x_mat)).to(torch.int64)
    batch_m_ = Variable(torch.from_numpy(x_mask)).type(torch.FloatTensor)
    batch_y_ = Variable(torch.from_numpy(y_vec)).to(torch.int64)

    if use_cuda:
        batch_x_ = batch_x_.cuda()
        batch_m_ = batch_m_.cuda()
        batch_y_ = batch_y_.cuda()

    return {"x": batch_x_, "m": batch_m_, "y": batch_y_}
