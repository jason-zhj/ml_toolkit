"""
this includes the computation for precision, recall, MAP
This supports inactive bit, '-' represents inactive
"""
import operator

import matplotlib.pyplot as plt
from ml_toolkit.hash_toolkit.metrics.utils import _fig2img, _compute_hash_with_dist, _retrieve_items_using_hash, get_hdist, _retrieve_items_all

def _calc_precision_recall(radius,item,db_set):
    """
    :param radius:
    :param item: {"hash":...,"label":..}
    :param db_set:
    :return: precisions, recalls, MAP for each distance within `radius`
    """
    query_hash = item["hash"]
    query_label = item["label"]
    total_correct = sum([1 for item in db_set if item["label"]==query_label])  # total number of correct items to be retrieved
    correct_retrieved = [0 for _ in range(radius+1)]  # record # correctly retrieved for each distance
    total_retrieved = [0 for _ in range(radius+1)] # record # retrieved for each distance
    # record correct_retrieved and total_retrieved for each distance within radius
    for db_item in db_set:
        dist = get_hdist(item["hash"],db_item["hash"])
        if (dist <= radius):
            total_retrieved[dist] += 1
            if (item["label"]==db_item["label"]):
                correct_retrieved[dist] += 1

    # calc precision recall
    dist_precisions = [correct_retrieved[i] * 1.0 / total_retrieved[i] if total_retrieved[i] > 0 else 0
                       for i in range(len(total_retrieved))]
    radius_precisions = [sum(correct_retrieved[:i+1]) * 1.0 / sum(total_retrieved[:i+1]) if sum(total_retrieved[:i+1]) > 0 else 0
                       for i in range(len(total_retrieved))]
    dist_recalls = [correct_retrieved[i] * 1.0 / total_correct if total_correct > 0 else 0
                       for i in range(len(correct_retrieved))]
    radius_recalls = [sum(correct_retrieved[:i+1]) * 1.0 / total_correct if total_correct > 0 else 0
                       for i in range(len(correct_retrieved))]
    # calc MAP
    index_to_use = [i for i in range(len(correct_retrieved)) if correct_retrieved[i] > 0]

    avg_precision = sum([radius_precisions[i] for i in index_to_use]) / len(index_to_use) \
        if len(index_to_use) > 0 else 0

    return {
        "precision-dist": dist_precisions,
        "precision-radius": radius_precisions,
        "recall-dist": dist_recalls,
        "recall-radius": radius_recalls,
        "avg-precision": avg_precision,
        "retrieved-dist": total_retrieved,
        "retrieved-ratio-dist": [i * 1.0 / len(db_set) for i in total_retrieved] # percentage of db data retrieved
    }

def calculate_precision_recall(radius, db_set, test_set, get_label_specific_details = False):
    """
    :param dist_or_radius: either dist or radius
    :param use_dist: if True, `dist_or_radius` will be used as distance, otherwise as radius
    :return: result_dict, see below
    """
    assert radius >=0
    result_dict = {
        "precision-dist":[0 for _ in range(radius+1)],
        "precision-radius":[0 for _ in range(radius+1)],
        "recall-dist":[0 for _ in range(radius+1)],
        "recall-radius":[0 for _ in range(radius+1)],
        "retrieved-dist": [0 for _ in range(radius+1)],
        "retrieved-ratio-dist": [0 for _ in range(radius + 1)],
        "label-specific-details":{
            # record the average precision, recall for each class (i.e. label) of images
        }
    }
    label_detail_dict = result_dict["label-specific-details"]
    mean_avg_precision = 0

    for item in test_set:
        item_result = _calc_precision_recall(radius=radius,item=item, db_set=db_set)

        # recall overall results
        for key in result_dict.keys():
            if (key != "label-specific-details"):
                result_dict[key] = list(map(operator.add,result_dict[key],item_result[key]))
        mean_avg_precision += item_result["avg-precision"]

        # recall results specific for each label
        if (get_label_specific_details):
            item_label = item["label"]

            if (item_label not in label_detail_dict.keys()):
                label_detail_dict[item_label] = {
                    "count":1,
                    "precision-dist": item_result["precision-dist"],
                    "precision-radius": item_result["precision-radius"],
                    "recall-dist": item_result["recall-dist"],
                    "recall-radius": item_result["recall-radius"],
                }
            else:
                label_detail_dict[item_label]["count"] += 1
                for key in ["precision-dist","precision-radius","recall-dist","recall-radius"]:
                    label_detail_dict[item_label][key] = list(map(operator.add,label_detail_dict[item_label][key],item_result[key]))


    # divide by number of test items (overall result)
    for key in result_dict.keys():
        if (key!="label-specific-details"):
            result_dict[key] = [item / len(test_set) for item in result_dict[key]]
    result_dict["mean-avg-precision"] = mean_avg_precision / len(test_set)

    # divide by number of items (label-specific result)
    for label in label_detail_dict.keys():
        detail_dict = label_detail_dict[label]
        count = detail_dict["count"]
        for key in ["precision-dist", "precision-radius", "recall-dist", "recall-radius"]:
            detail_dict[key] = list(map(operator.truediv,detail_dict[key],[count for _ in range(len(detail_dict[key]))]))

    return result_dict


def get_mean_avg_precision(test_set,db_set,maxdist):
    """
    refer to https://i.ytimg.com/vi/pM6DJ0ZZee0/maxresdefault.jpg for formula
    :param test_set:list of dict {"label":..,"hash":""}
    :param db_set:list of dict {"label":..,"hash":""}
    :param maxdist:
    :return: mean average precision
    """
    m_a_p = 0
    for query in test_set:
        m_a_p += _get_avg_precision_for_query(query=query, db_set=db_set, max_hdist=maxdist)
    return m_a_p / len(test_set)


def _get_avg_precision_for_query(query, db_set, max_hdist=None):
    query_hash = query["hash"]
    query_label = query["label"]
    retrieve_results = _retrieve_items_all(db_set=db_set, hashcode=query_hash, max_hdist=max_hdist)
    max_key = max(retrieve_results.keys())
    total_retrieved = [len(retrieve_results[i]) for i in range(max_key+1)]
    correct_retrieved = [
        len([item for item in retrieve_results[i] if item["label"]==query_label])
        for i in range(max_key + 1)
    ]
    radius_precisions = [
        sum(correct_retrieved[:i + 1]) * 1.0 / sum(total_retrieved[:i + 1]) if sum(total_retrieved[:i + 1]) > 0 else 0
        for i in range(len(total_retrieved))]

    avg_precision = 0
    num_added = 0
    for i, correct_num in enumerate(correct_retrieved):
        if (correct_num > 0):
            avg_precision += radius_precisions[i]
            num_added += 1

    return avg_precision / num_added if num_added > 0 else 0



def get_precision_vs_recall(test_set,db_set,max_hdist):
    """
    :param test_set: list of dict {"label":..,"hash":""}
    :param db_set: the data acting as the database
    :param max_hdist: maximum hamming distance two hash code can have
    :return: a list of {"precisions":[precision@r=0,precision@r=1,...],"recalls":[recall@r=0,recall@r=1...]}
    """
    results = []
    # for each test item, there is a precision list and recall list
    for i in range(len(test_set)):
        item = test_set[i]
        query_hash = item["hash"]
        query_label = item["label"]
        record_dict = {i:{"correct":0,"wrong":0} for i in range(max_hdist+1)}
        # loop through the db_set
        total_class_num = 0 # how many items in db have the same label as query_label
        for db_item in db_set:
            db_item_hash = db_item["hash"]
            hdist = get_hdist(query_hash, db_item_hash)
            if (db_item["label"] == query_label):
                record_dict[hdist]["correct"] += 1
                total_class_num += 1
            else:
                record_dict[hdist]["wrong"] += 1

        # compute precisions and recalls
        total_correct = 0
        total_retrieved = 0
        precisions = []
        recalls = []
        for key,value in record_dict.items():
            total_correct += value["correct"]
            total_retrieved += value["correct"] + value["wrong"]
            if (total_retrieved>0):
                precisions.append(total_correct / total_retrieved)
            else:
                precisions.append(0)
            recalls.append(total_correct/total_class_num)

        print("finish item {}/{}".format(i,len(test_set)))
        # add to results
        results.append({"precisions":precisions,"recalls":recalls})
    return  results


def plot_avg_precision_vs_recall(pr_list,popup=True):
    """
    :param pr_list: list of {"precisions":[],"recalls":[]}, output of `get_precision_vs_recall`
    :param popup: the figure will pop up if this is True
    :return: {"avg_precisions":[],"avg_recalls":[],"plot":Image object}
    """
    for item in pr_list:
        assert len(item["precisions"]) ==len(item["recalls"])

    valuelen = len(pr_list[0]["precisions"])
    avg_precisions = []
    avg_recalls = []
    for i in range(valuelen):
        # calculate avg precision, recall at hamming distance = i
        avg_p = 0
        avg_r = 0
        for item in pr_list:
            avg_p += item["precisions"][i] / len(pr_list)
            avg_r += item["recalls"][i] / len(pr_list)
        avg_precisions.append(avg_p)
        avg_recalls.append(avg_r)

    fig = plt.figure()
    figplt = fig.add_subplot(111)
    figplt.plot(avg_recalls,avg_precisions)
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.title("Precision vs. Recall")
    if (popup):
        plt.show()
    image = _fig2img(fig=fig)
    return {"avg_precisions":avg_precisions,"avg_recalls":avg_recalls,"plot":image}

def get_precision_recall_curve(query_set,db_set):
    "return PIL Image object, of P-R curve"
    max_hdist = len(query_set[0]["hash"])
    pr_dict = get_precision_vs_recall(test_set=query_set, db_set=db_set, max_hdist=max_hdist)
    precision_recall_dict = plot_avg_precision_vs_recall(pr_list=pr_dict, popup=False)
    return precision_recall_dict["plot"]