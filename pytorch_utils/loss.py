"""
Utilities for calculating common loss
"""
import torch
import sys, os
import numpy as np
from torch.autograd import Variable

from ml_toolkit.pytorch_utils.preprocess import convert_to_onehot
from ml_toolkit.pytorch_utils.misc import autocuda


def normalize_by_rows(m):
    "divide each element by its row's L2 norm"
    qn = torch.norm(m, p=2, dim=1).detach()
    return m.div(qn.expand_as(m))


def get_pairwise_sim_loss(feats,labels,num_classes=31,normalize=True,feat_scale=16,sigmoid_alpha=1):
    labels_onehot = autocuda(Variable(convert_to_onehot(labels=labels,num_class=num_classes).float(),requires_grad=False))

    # do inner product of features
    vec_len = feats.data.size(1)
    A_square = torch.mm(feats, feats.t()) / vec_len * feat_scale # we need to divide the vector length, otherwise the loss will be affected by the length of hash code
    TINY = 10e-8
    A_square_sigmod = (torch.sigmoid(A_square * sigmoid_alpha) - 0.5) * (1 - TINY) + 0.5
    is_same_lbl = torch.mm(labels_onehot, labels_onehot.t())

    # calc log probability loss
    log_prob = torch.mul(torch.log(A_square_sigmod), is_same_lbl)
    log_prob += torch.mul(torch.log(1 - A_square_sigmod), 1 - is_same_lbl)
    sum_log_prob = (log_prob.sum() - log_prob.diag().sum()) / 2.0

    # divide the sum to get average loss
    num_pairs = len(labels) * (len(labels) - 1) / 2
    return torch.sum(-sum_log_prob) / num_pairs if normalize else torch.sum(-sum_log_prob)


def get_crossdom_pairwise_sim_loss(src_feats,tgt_feats,src_labels,tgt_labels,num_classes=31,normalize=True,feat_scale=16,sigmoid_alpha=1):
    "loss: sum(logp(hi,hj)), where p is defined by sigmoid function"
    src_labels_onehot = autocuda(Variable(convert_to_onehot(labels=src_labels, num_class=num_classes), requires_grad=False))
    tgt_labels_onehot = autocuda(Variable(convert_to_onehot(labels=tgt_labels, num_class=num_classes),
                                 requires_grad=False))
    assert src_feats.data.size(1) == tgt_feats.data.size(1)
    vec_len = src_feats.data.size(1)

    # get inner product
    A_square = torch.mm(src_feats, tgt_feats.t()) / vec_len * feat_scale  # we need to divide the vector length, otherwise the loss will be affected by the length of hash code
    TINY = 10e-8
    A_square_sigmod = (torch.sigmoid(A_square * sigmoid_alpha) - 0.5) * (1 - TINY) + 0.5

    # calc log probability
    is_same_lbl = torch.mm(src_labels_onehot, tgt_labels_onehot.t())
    log_prob = torch.mul(torch.log(A_square_sigmod), is_same_lbl)
    log_prob += torch.mul(torch.log(1 - A_square_sigmod), 1 - is_same_lbl)
    sum_log_prob = (log_prob.sum() - log_prob.diag().sum()) / 2.0

    num_pairs = len(src_labels) * len(tgt_labels)
    return torch.sum(-sum_log_prob) / num_pairs if normalize else torch.sum(-sum_log_prob)

def _linear_mmd_loss(x,y):
    "MMD loss with linear kernel"
    x = x.view(x.size(0), -1)
    y = y.view(y.size(0), -1)
    vec_len = x.data.size(1)
    x_bar = x.mean(0) #TODO: which axis will it sum over?
    y_bar = y.mean(0)
    z_bar = x_bar - y_bar
    return torch.dot(z_bar,z_bar) / vec_len

def _rbf_mmd_loss(x,y,alpha=1.0):
    "MMD loss with RBF kernel"
    assert len(x) == len(y)
    B = len(x)  # batch size
    x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
    y = y.view(y.size(0), y.size(1) * y.size(2) * y.size(3))

    x = x / x.size(1)
    y = y / y.size(1)

    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    K = torch.exp(- alpha * (rx.t() + rx - 2 * xx))
    L = torch.exp(- alpha * (ry.t() + ry - 2 * yy))
    P = torch.exp(- alpha * (rx.t() + ry - 2 * zz))

    beta = (1. / (B * (B - 1)))
    gamma = (2. / (B * B))

    return beta * (torch.sum(K) + torch.sum(L)) - gamma * torch.sum(P)

def get_mmd_loss(x,y, kernel="rbf",alpha=1.0):
    """
    this will first normalize each row before computing MMD
    :param x: source domain features
    :param y: target domain features
    :param kernel: "rbf" or "linear"
    :param alpha: kernel parameter
    :return: RBF Kernel MMD
    """
    if (kernel == "rbf"):
        return _rbf_mmd_loss(x=x,y=y,alpha=alpha)
    elif (kernel == "linear"):
        return _linear_mmd_loss(x=x,y=y)
    else:
        raise Exception("{} is not a valid kernel".format(kernel))

def get_L2_norm(x):
    "return ||x||2"
    pass

def get_L1_norm(x):
    "return |x|1, sum along each row"
    return torch.sum(torch.abs(x),dim=1)


def get_rms_diff(x,y):
    "return the root mean square of (x-y)"
    p = torch.pow(
        torch.add(x,-y),
        2
    )
    vector_len = Variable(torch.FloatTensor([x.data.size(0) for _ in range(x.data.size(0))]), requires_grad=False)
    return torch.sqrt(torch.sum(p / vector_len))

def get_quantization_loss(continuous_code):
    "continuous code is output of `torch.tanh`, which should range between [-1,1]"
    discrete_code = torch.sign(continuous_code)
    quantization_loss = torch.sum(
        torch.pow(
            torch.add(discrete_code, torch.neg(continuous_code)), 2
        )
    )
    return quantization_loss / (continuous_code.data.size(0) * continuous_code.data.size(1)) # normalize by batch_size * code_length


def get_euclidean_distance(H):
    """
    The euclidean distance |hi-hj|^2 can be decomposed as |hi|^2 + |hj|^2 - 2 * <hi,hj>
    :param H: H is a matrix, with each row being a vector
    :return: a matrix containing euclidean distance of each pair of vectors
    """
    num_vectors = H.size(0)

    # compute |hi|^2 + |hj|^2
    squared = torch.pow(H,2)
    square_sum = torch.sum(squared,dim=1)
    square_sum_matrix = square_sum.repeat(1,num_vectors)

    # compute <hi,hj>
    inner_products = torch.mm(H,H.t())

    # |hi|^2 + |hj|^2 - 2 * <hi,hj>
    result = torch.clamp(square_sum_matrix + square_sum_matrix.t() - 2 * inner_products,min=1e-8)

    return result

def get_tdist_pairwise_similarity_loss(feats,labels,tanh_alpha=None):
    """
    this is an implementation of t-distribution pairwise similarity loss,
    as proposed in the paper "Transfer Adversarial Hashing for Hamming Space Retrieval"
    :param feats: a matrix, with each row being a vector
    :param labels: array of one-hot vectors
    compute - sum of log of p(Sij|hi,hj)
    p(Sij|hi,hj) = tanh(tanh_alpha * sim(hi,hj)) if Sij = 1
                 = 1 - tanh(tanh_alpha * sim(hi,hj)) if Sij = 0
    sim(hi,hj) = feat_length / (1 + euclidean_distance(hi,hj) )
    """
    assert feats.size(0) == labels.size(0)
    feat_len = feats.size(1)
    tanh_alpha = 2.00 / feat_len if tanh_alpha is None else tanh_alpha

    # compute sim(hi,hj)
    distance_matrix = get_euclidean_distance(H=feats)
    sim = feat_len / (1 + distance_matrix)

    # compute tanh(tanh_alpha * sim(hi,hj))
    sim_tanh = torch.tanh(tanh_alpha * sim)

    # compute sum of log[ p(Sij|hi,hj) ]
    is_same_lbl = torch.mm(labels, labels.t())

    log_prob = torch.mul(torch.log(sim_tanh), is_same_lbl) + torch.mul(torch.log(1 - sim_tanh), 1 - is_same_lbl)
    sum_log_prob = (log_prob.sum() - log_prob.diag().sum()) / 2.0

    # normalize the sum by # of pairs of feat vectors
    num_pairs = feats.size(0) * (feats.size(0) - 1) / 2
    return - sum_log_prob / num_pairs