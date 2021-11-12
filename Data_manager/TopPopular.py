#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from BaseRecommender import BaseRecommender
import scipy.sparse as sps
from check_matrix import check_matrix



class TopPopular(BaseRecommender):
    RECOMMENDER_NAME = "TopPopularRecommender"
    
    def __init__(self, URM_train):
        super(TopPopular, self).__init__(URM_train)
           
    def fit(self):
        self.item_pop = np.ediff1d(self.URM_train.tocsc().indptr)
        self.n_items = self.URM_train.shape[1]
    def _compute_item_score(self, user_id_array, items_to_compute = None):

        # Create a single (n_items, ) array with the item score, then copy it for every user

        if items_to_compute is not None:
            item_pop_to_copy = - np.ones(self.n_items, dtype=np.float32)*np.inf
            item_pop_to_copy[items_to_compute] = self.item_pop[items_to_compute].copy()
        else:
            item_pop_to_copy = self.item_pop.copy()

        item_scores = np.array(item_pop_to_copy, dtype=np.float32).reshape((1, -1))
        item_scores = np.repeat(item_scores, len(user_id_array), axis = 0)

        return item_scores

