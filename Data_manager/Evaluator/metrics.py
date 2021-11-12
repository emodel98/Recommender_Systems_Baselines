#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import scipy.sparse as sps


class _Metrics_Object(object):
    """
    Abstract class that should be used as superclass of all metrics requiring an object, therefore a state, to be computed
    """
    def __init__(self):
        pass

    def __str__(self):
        return "{:.4f}".format(self.get_metric_value())

    def add_recommendations(self, recommended_items_ids):
        raise NotImplementedError()

    def get_metric_value(self):
        raise NotImplementedError()




def AUROC(is_relevant):
    ranks = np.arange(len(is_relevant))
    pos_ranks = ranks[is_relevant]
    neg_ranks = ranks[~is_relevant]
    auc_score = 0.0
    
    if len(neg_ranks) == 0:
        return 1.0
    if len(pos_ranks) > 0:
        for pos_pred in pos_ranks:
            auc_score += np.sum(pos_pred < neg_ranks, dtype=np.float32)
        auc_score /= (pos_ranks.shape[0] * neg_ranks.shape[0])
        
    assert 0 <= auc_score <= 1, auc_score
    return auc_score


def ndcg(ranked_list, pos_items, relevance=None, at=None):

    if relevance is None:
        relevance = np.ones_like(pos_items)
    assert len(relevance) == pos_items.shape[0]

    # Create a dictionary associating item_id to its relevance
    # it2rel[item] -> relevance[item]
    it2rel = {it: r for it, r in zip(pos_items, relevance)}

    # Creates array of length "at" with the relevance associated to the item in that position
    rank_scores = np.asarray([it2rel.get(it, 0.0) for it in ranked_list[:at]], dtype=np.float32)

    # IDCG has all relevances to 1, up to the number of items in the test set
    ideal_dcg = dcg(np.sort(relevance)[::-1])

    # DCG uses the relevance of the recommended items
    rank_dcg = dcg(rank_scores)

    if rank_dcg == 0.0:
        return 0.0

    ndcg_ = rank_dcg / ideal_dcg

    return ndcg_

def dcg(scores):
    return np.sum(np.divide(np.power(2, scores) - 1, np.log(np.arange(scores.shape[0], dtype=np.float32) + 2)),
                  dtype=np.float32)