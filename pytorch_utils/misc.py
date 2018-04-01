import torch
import numpy as np
import torchvision
from torch.autograd import Variable
from torchvision import transforms

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


def autocuda(var):
    if torch.cuda.is_available():
        return var.cuda()
    else:
        return var

def str_param_module(params):
    "return the string representation of the params module"
    filtered_items = {key: value for key, value in vars(params).items() if key.find("__") != 0}
    ls = []
    for key, value in filtered_items.items():
        ls.append("{}={}".format(key, value))

    return "\n".join(ls)

def save_models(models,save_model_to,save_obj=True,save_params=True):
    "models is a dict {name:model_obj}"
    for name,model in models.items():
        if (save_obj):
            torch.save(model, os.path.join(save_model_to, "{}.model".format(name)))
        if (save_params):
            torch.save(model.state_dict(), os.path.join(save_model_to, "{}.params".format(name)))


def get_data_loader(data_path,image_scale,dataset_mean,dataset_std,batch_size,shuffle_batch):
    "return a torch data loader"
    # image loading
    preprocess = transforms.Compose([
        transforms.Scale(image_scale),
        transforms.ToTensor(),
        transforms.Normalize(mean=dataset_mean, std=dataset_std)
    ])

    # create dataloader
    dataset = torchvision.datasets.ImageFolder(root=data_path, transform=preprocess)
    return  torch.utils.data.DataLoader(dataset,
                                                batch_size=batch_size, shuffle=shuffle_batch)

# if you have a hash function that returns list of digits -1, 1, 0(inactive bit) as a hash code
# this wrapper converts that to a string of 0 ,1, and `-` (inactive bit)
def hash_func_wrapper(hash_func):

    def wrapped(images):
        outputs = hash_func(images)
        outputs = outputs.data.numpy().astype(np.int8).astype(str)
        outputs[outputs == '0'] = '-'
        outputs[outputs == '-1'] = '0'
        return ["".join(o) for o in outputs]

    return wrapped
	
	
def init_model(net, restore):
    """Init models with cuda and weights."""

    # restore model weights
    if restore is not None and os.path.exists(restore):
        net.load_state_dict(torch.load(restore))
        net.restored = True
        print("Restore model from: {}".format(os.path.abspath(restore)))

    # check if cuda is available
    if torch.cuda.is_available():
        cudnn.benchmark = True
        net.cuda()

    return net

def make_variable(tensor, volatile=False,requires_grad=False):
    """Convert Tensor to Variable."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor, volatile=volatile, requires_grad = requires_grad)