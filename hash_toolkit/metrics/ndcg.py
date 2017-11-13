import math
from ml_toolkit.hash_toolkit.metrics.utils import _compute_hash_with_dist, _retrieve_items_using_hash

#TODO: need more comprehensive testing

def _calc_ndcg_from_rel_ls(rel_ls):
    "apply NDCG formula to relevancy list"
    return sum(
        [(2**rel_ls[i]-1)/math.log(2+i) for i in range(len(rel_ls))]
    )

def _calc_ndcg_with_topk(item,db_set,k,rel_score):
    "calculate NDCG for first `k` retrieved items"
    assert rel_score > 0

    query_hash = item["hash"]
    query_label = item["label"]
    retr_rel_ls = []
    for dist in range(len(query_hash)):
        hashes_to_retr = _compute_hash_with_dist(hashcode=query_hash, dist=dist)
        retrieved_items = []
        for hashcode in hashes_to_retr:
            retrieved_items += _retrieve_items_using_hash(db_set=db_set, hashcode=hashcode)
        # relevancy score, sorted from low to high
        rel_ls = sorted([rel_score if i["label"]==query_label else 0 for i in retrieved_items])
        if (len(retr_rel_ls) + len(rel_ls) < k):
            retr_rel_ls += rel_ls
        else:
            retr_rel_ls += rel_ls[:k-len(retr_rel_ls)]

    return _calc_ndcg_from_rel_ls(rel_ls=retr_rel_ls)

def _calc_ndcg_with_radius(item,db_set,radius,rel_score):
    "calculate NDCG for items within `radius`"
    query_hash = item["hash"]
    query_label = item["label"]
    retr_rel_ls = []
    for dist in range(radius + 1):
        hashes_to_retr = _compute_hash_with_dist(hashcode=query_hash, dist=dist)
        retrieved_items = []
        for hashcode in hashes_to_retr:
            retrieved_items += _retrieve_items_using_hash(db_set=db_set, hashcode=hashcode)
        # relevancy score, sorted from low to high
        rel_ls = sorted([rel_score if i["label"] == query_label else 0 for i in retrieved_items])
        retr_rel_ls += rel_ls

    return _calc_ndcg_from_rel_ls(rel_ls=retr_rel_ls)

def _calc_ndcg(item,db_set,radius_or_topk,use_topk,rel_score=2):
    """
    Implementation of Normalized Discounted Cumulative Gain
    Here we consider the worst case, for the db items with the same hamming distance, we assume wrong items are ranked higher
    :param radius_or_topk:
    :param use_topk: if this is True,  `radius_or_topk` will be used as K
    :param rel_score: relevancy score for correct item
    """

    if (use_topk):
        K = radius_or_topk
        return _calc_ndcg_with_topk(item=item,db_set=db_set,k=K,rel_score=rel_score)
    else:
        R = radius_or_topk
        return _calc_ndcg_with_radius(item=item,db_set=db_set,radius=R,rel_score=rel_score)

def calculate_NDCG(query_set,db_set,radius_or_topk,use_topk,rel_score=2):
    "return the average NDCG for items in `query_set`, refer to `_calc_ndcg` for the use of parameters"
    return sum(
        [_calc_ndcg(item=item,db_set=db_set,radius_or_topk=radius_or_topk,use_topk=use_topk,rel_score=rel_score) for item in query_set]
    ) / len(query_set)