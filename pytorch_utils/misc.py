import torch
import numpy as np
from torch.autograd import Variable

def calc_output_dim(models,input_dim):
    """
    :param models: a list of model objects
    :param input_dim: like [3,100,100]
    :return:  the output dimension (in one number) if an image of `input_dim` is passed into `models`
    """
    input_tensor = torch.from_numpy(np.zeros(input_dim))
    input_tensor.unsqueeze_(0)
    img = Variable(input_tensor).float()
    output = img
    for model in models:
        output = model(output)
    return output.data.view(output.data.size(0), -1).size(1)