import torch
import os
from ml_toolkit.hash_toolkit.metrics.precision_recall import calculate_precision_recall,get_precision_recall_curve
from ml_toolkit.pytorch_utils.misc import make_variable

def load_models(path,model_names,test_mode=True):
    "load .model files, return a dict {name:model_obj}"
    models = {}
    for name in model_names:
        m = torch.load(os.path.join(path,"{}.model".format(name)))
        if (test_mode): m.eval()
        models[name] = m

    return models

def _hash_data(data_loader,hash_model):
    "hash all the data from data_loader, and return a dict {label,hash}"
    hash_ls = []
    label_ls = []
    for i,(images,labels) in enumerate(data_loader):
        hash_ls += hash_model(images)
        label_ls += labels.numpy().tolist()

    return [
        {"label": label_ls[i], "hash": hash_ls[i]}
        for i in range(len(hash_ls))
    ]


def run_test(query_loader,db_loader,query_hash_model,db_hash_model,radius):
    """
    :param hash_model: a function that takes in image tensor and output hash code
    :param radius: radius within which to calculate precision, recall
    :return: result dict
    """
    # hash data
    query_set = _hash_data(data_loader=query_loader,hash_model=query_hash_model)
    db_set = _hash_data(data_loader=db_loader,hash_model=db_hash_model)

    # measure performance
    precision_recall_results = calculate_precision_recall(radius=radius, db_set=db_set, test_set=query_set)
    pr_curve = get_precision_recall_curve(query_set=query_set,db_set=db_set)

    return {
        "precision-recall-results": precision_recall_results,
        "precision-recall-curve": pr_curve
    }


def run_classification_test(data_loader,clf_model):
    "run classification test, return accuracy"
    total = 0
    correct = 0

    for images, labels in data_loader:
        outputs = clf_model(make_variable(images.float()))
        _, predicted = torch.max(outputs.data, 1)
        total += predicted.size(0)
        correct += (predicted == labels).sum()

    # calculate accuracy
    return {
        "accuracy": correct * 100.0 / total
    }

	
def save_test_results(test_results,save_to):
    # create folder
    if (not os.path.exists(save_to)): os.makedirs(save_to)
    # save
    test_results["precision-recall-curve"].save(os.path.join(save_to, "precision-recall.png"))
    with open(os.path.join(save_to, "metrics.txt"), "w") as f:
        f.write(str(test_results["precision-recall-results"]))