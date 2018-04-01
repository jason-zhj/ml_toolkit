import torch

def convert_to_onehot(labels,num_class):
    "labels is a one-dimensional Int Tensor"
    y_onehot = torch.LongTensor(len(labels), num_class)
    y_onehot.zero_()
    y_onehot.scatter_(1, torch.unsqueeze(labels.cpu(),1), 1)
    return y_onehot