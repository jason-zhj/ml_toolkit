"""
Utilities for calculating common loss
"""
import torch
import sys, os
from torch.autograd import Variable
sys.path.append(os.path.dirname(__file__))
from preprocess import convert_to_onehot
from misc import autocuda

def get_pairwise_sim_loss(feats,labels,num_classes=31,normalize=True,feat_scale=16):
    labels_onehot = autocuda(Variable(convert_to_onehot(labels=labels,num_class=num_classes),requires_grad=False))

    # do inner product of features
    vec_len = feats.data.size(1)
    A_square = torch.mm(feats, feats.t()) / vec_len * feat_scale # we need to divide the vector length, otherwise the loss will be affected by the length of hash code
    TINY = 10e-8
    A_square_sigmod = (torch.sigmoid(A_square) - 0.5) * (1 - TINY) + 0.5
    is_same_lbl = torch.mm(labels_onehot, labels_onehot.t())

    # calc log probability loss
    log_prob = torch.mul(torch.log(A_square_sigmod), is_same_lbl)
    log_prob += torch.mul(torch.log(1 - A_square_sigmod), 1 - is_same_lbl)
    sum_log_prob = (log_prob.sum() - log_prob.diag().sum()) / 2.0

    # divide the sum to get average loss
    num_pairs = len(labels) * (len(labels) - 1) / 2
    return torch.sum(-sum_log_prob) / num_pairs if normalize else torch.sum(-sum_log_prob)


def get_crossdom_pairwise_sim_loss(src_feats,tgt_feats,src_labels,tgt_labels,num_classes=31,normalize=True,feat_scale=16):
    "loss: sum(logp(hi,hj)), where p is defined by sigmoid function"
    src_labels_onehot = autocuda(Variable(convert_to_onehot(labels=src_labels, num_class=num_classes), requires_grad=False))
    tgt_labels_onehot = autocuda(Variable(convert_to_onehot(labels=tgt_labels, num_class=num_classes),
                                 requires_grad=False))
    assert src_feats.data.size(1) == tgt_feats.data.size(1)
    vec_len = src_feats.data.size(1)

    # get inner product
    A_square = torch.mm(src_feats, tgt_feats.t()) / vec_len * feat_scale  # we need to divide the vector length, otherwise the loss will be affected by the length of hash code
    TINY = 10e-8
    A_square_sigmod = (torch.sigmoid(A_square) - 0.5) * (1 - TINY) + 0.5

    # calc log probability
    is_same_lbl = torch.mm(src_labels_onehot, tgt_labels_onehot.t())
    log_prob = torch.mul(torch.log(A_square_sigmod), is_same_lbl)
    log_prob += torch.mul(torch.log(1 - A_square_sigmod), 1 - is_same_lbl)
    sum_log_prob = (log_prob.sum() - log_prob.diag().sum()) / 2.0

    num_pairs = len(src_labels) * len(tgt_labels)
    return torch.sum(-sum_log_prob) / num_pairs if normalize else torch.sum(-sum_log_prob)

#TODO: edit this
def get_mmd_loss(x,y, alpha=1.1, B=1.1):
    "Kernel MMD loss"
    x = x.view(x.size(0), x.size(2) * x.size(3))
    y = y.view(y.size(0), y.size(2) * y.size(3))

    xx, yy, zz = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x,y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    K = torch.exp(- alpha * (rx.t() + rx - 2*xx))
    L = torch.exp(- alpha * (ry.t() + ry - 2*yy))
    P = torch.exp(- alpha * (rx.t() + ry - 2*zz))

    beta = (1./(B*(B-1)))
    gamma = (2./(B*B))

    return beta * (torch.sum(K)+torch.sum(L)) - gamma * torch.sum(P)

def get_L2_norm(x):
    "return ||x||2"
    pass

def get_L1_norm(x):
    "return |x|1"
    return torch.sum(torch.abs(x))


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
    return quantization_loss / len(continuous_code)