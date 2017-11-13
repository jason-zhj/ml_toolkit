"""
this includes the computation for precision, recall, MAP
"""
import operator

import matplotlib.pyplot as plt
from ml_toolkit.hash_toolkit.metrics.utils import _fig2img, _compute_hash_with_dist, _retrieve_items_using_hash, _get_hdist


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
    for dist in range(radius+1):
        hashes_to_retr = _compute_hash_with_dist(hashcode=query_hash, dist=dist)
        retrieved_items = []
        for hashcode in hashes_to_retr:
            retrieved_items += _retrieve_items_using_hash(db_set=db_set, hashcode=hashcode)

        total_retrieved[dist] = len(retrieved_items)
        correct_retrieved[dist] = len([item for item in retrieved_items if item["label"] == query_label])

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

def calculate_precision_recall(radius, db_set, test_set):
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
    }
    mean_avg_precision = 0
    for item in test_set:
        item_result = _calc_precision_recall(radius=radius,item=item, db_set=db_set)
        for key in result_dict.keys():
            result_dict[key] = list(map(operator.add,result_dict[key],item_result[key]))
        mean_avg_precision += item_result["avg-precision"]

    # divide by number of test items
    for key in result_dict.keys():
        result_dict[key] = [item / len(test_set) for item in result_dict[key]]
    result_dict["mean-avg-precision"] = mean_avg_precision / len(test_set)

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
        m_a_p += _get_avg_precision_for_query(query=query,db_set=db_set,maxdist=maxdist)
    return m_a_p / len(test_set)

def _get_avg_precision_for_query(query, db_set, maxdist):
    """
    refer to https://i.ytimg.com/vi/pM6DJ0ZZee0/maxresdefault.jpg for formula
    :param query: list of {hash:"101",label:".."}
    :param db_set: list of {hash:"101",label:".."}
    :param maxdist: maximum distance to retrieve
    :return: average precision for this single query
    """
    query_code = query["hash"]
    query_label = query["label"]
    avg_precision = 0
    num_added = 0
    total_retrieved = 0
    correct_retrieved = 0
    for d in range(maxdist):
        hash_ls = _compute_hash_with_dist(hashcode=query_code, dist=d)
        retrieved_items = []
        include = False # include precision at this distance when there is correct item retrieved
        for hash in hash_ls:
            retrieved_items = _retrieve_items_using_hash(db_set=db_set, hashcode=hash)
            total_retrieved += len(retrieved_items)
            new_correct = len([t for t in retrieved_items if t["label"]==query_label])
            if (new_correct > 0):
                correct_retrieved +=new_correct
                include = True

        if (include):
            num_added += 1
            avg_precision += float(correct_retrieved) / total_retrieved

    return avg_precision / num_added


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
            hdist = _get_hdist(query_hash, db_item_hash)
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

