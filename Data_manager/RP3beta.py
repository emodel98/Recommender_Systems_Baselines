#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import scipy.sparse as sps

from sklearn.preprocessing import normalize
from check_matrix import check_matrix

import time, sys



from BaseRecommender import BaseRecommender
    
    
class BaseSimilarityMatrixRecommender(BaseRecommender):

    def __init__(self, URM_train, verbose=True):
        super(BaseSimilarityMatrixRecommender, self).__init__(URM_train, verbose = verbose)

        self._URM_train_format_checked = False
        self._W_sparse_format_checked = False



    def _check_format(self):

        if not self._URM_train_format_checked:

            if self.URM_train.getformat() != "csr":
                self._print("PERFORMANCE ALERT compute_item_score: {} is not {}, this will significantly slow down the computation.".format("URM_train", "csr"))

            self._URM_train_format_checked = True

        if not self._W_sparse_format_checked:

            if self.W_sparse.getformat() != "csr":
                self._print("PERFORMANCE ALERT compute_item_score: {} is not {}, this will significantly slow down the computation.".format("W_sparse", "csr"))

            self._W_sparse_format_checked = True

    
class BaseItemSimilarityMatrixRecommender(BaseSimilarityMatrixRecommender):

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        """
        URM_train and W_sparse must have the same format, CSR
        :param user_id_array:
        :param items_to_compute:
        :return:
        """

        self._check_format()

        user_profile_array = self.URM_train[user_id_array]

        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.URM_train.shape[1]), dtype=np.float32)*np.inf
            item_scores_all = user_profile_array.dot(self.W_sparse).toarray()
            item_scores[:, items_to_compute] = item_scores_all[:, items_to_compute]
        else:
            item_scores = user_profile_array.dot(self.W_sparse).toarray()

        return item_scores


        
    
def similarityMatrixTopK(item_weights, k=100, verbose = False):

    assert (item_weights.shape[0] == item_weights.shape[1]), "selectTopK: ItemWeights is not a square matrix"

    start_time = time.time()

    if verbose:
        print("Generating topK matrix")

    nitems = item_weights.shape[1]
    k = min(k, nitems)

    # for each column, keep only the top-k scored items
    sparse_weights = not isinstance(item_weights, np.ndarray)

    # iterate over each column and keep only the top-k similar items
    data, rows_indices, cols_indptr = [], [], []

    if sparse_weights:
        item_weights = check_matrix(item_weights, format='csc', dtype=np.float32)
    else:
        column_row_index = np.arange(nitems, dtype=np.int32)



    for item_idx in range(nitems):

        cols_indptr.append(len(data))

        if sparse_weights: #arr
            start_position = item_weights.indptr[item_idx]
            end_position = item_weights.indptr[item_idx+1]

            column_data = item_weights.data[start_position:end_position]
            column_row_index = item_weights.indices[start_position:end_position]

        else: #mat
            column_data = item_weights[:,item_idx]


        non_zero_data = column_data!=0

        idx_sorted = np.argsort(column_data[non_zero_data])  # sort by column
        top_k_idx = idx_sorted[-k:]

        data.extend(column_data[non_zero_data][top_k_idx])
        rows_indices.extend(column_row_index[non_zero_data][top_k_idx])


    cols_indptr.append(len(data))

    # During testing CSR is faster
    W_sparse = sps.csc_matrix((data, rows_indices, cols_indptr), shape=(nitems, nitems), dtype=np.float32)

    if verbose:
        print("Sparse TopK matrix generated in {:.2f} seconds".format(time.time() - start_time))

    return W_sparse


class RP3beta(BaseItemSimilarityMatrixRecommender):
    """ RP3beta recommender """

    RECOMMENDER_NAME = "RP3beta"

    def __init__(self, URM_train, verbose = True):
        super(RP3beta, self).__init__(URM_train, verbose = verbose)


    def __str__(self):
        return "RP3beta(alpha={}, beta={}, min_rating={}, topk={}, implicit={}, normalize_similarity={})".format(self.alpha,
                                                                                        self.beta, self.min_rating, self.topK,
                                                                                        self.implicit, self.normalize_similarity)
    def fit(self, alpha=1., beta=0.6, min_rating=0, topK=100, implicit=False, normalize_similarity=True):

        self.alpha = alpha
        self.beta = beta
        self.min_rating = min_rating
        self.topK = topK
#         self.implicit = implicit
#         self.normalize_similarity = normalize_similarity

        

        if self.min_rating > 0:
            self.URM_train.data[self.URM_train.data < self.min_rating] = 0
            self.URM_train.eliminate_zeros()
            if self.implicit:
                self.URM_train.data = np.ones(self.URM_train.data.size, dtype=np.float32)

        #Pui is the row-normalized urm
        Pui = normalize(self.URM_train, norm='l1', axis=1)

        #Piu is the column-normalized, "boolean" urm transposed
        X_bool = self.URM_train.transpose(copy=True)
        X_bool.data = np.ones(X_bool.data.size, np.float32)

        # Taking the degree of each item to penalize top popular
        # Some rows might be zero, make sure their degree remains zero
        X_bool_sum = np.array(X_bool.sum(axis=1)).ravel()

        degree = np.zeros(self.URM_train.shape[1])

        nonZeroMask = X_bool_sum!=0.0

        degree[nonZeroMask] = np.power(X_bool_sum[nonZeroMask], -self.beta)

        #ATTENTION: axis is still 1 because i transposed before the normalization
        Piu = normalize(X_bool, norm='l1', axis=1)
        del(X_bool)

        # Alfa power
        if self.alpha != 1.:
            Pui = Pui.power(self.alpha)
            Piu = Piu.power(self.alpha)

        # Final matrix is computed as Pui * Piu * Pui
        # Multiplication unpacked for memory usage reasons
        block_dim = 200
        d_t = Piu


        # Use array as it reduces memory requirements compared to lists
        dataBlock = 10000000

        rows = np.zeros(dataBlock, dtype=np.int32)
        cols = np.zeros(dataBlock, dtype=np.int32)
        values = np.zeros(dataBlock, dtype=np.float32)

        numCells = 0


        start_time = time.time()
        start_time_printBatch = start_time

        for current_block_start_row in range(0, Pui.shape[1], block_dim):

            if current_block_start_row + block_dim > Pui.shape[1]:
                block_dim = Pui.shape[1] - current_block_start_row

            similarity_block = d_t[current_block_start_row:current_block_start_row + block_dim, :] * Pui
            similarity_block = similarity_block.toarray()

            for row_in_block in range(block_dim):
                row_data = np.multiply(similarity_block[row_in_block, :], degree)
                row_data[current_block_start_row + row_in_block] = 0

                best = row_data.argsort()[::-1][:self.topK]

                notZerosMask = row_data[best] != 0.0

                values_to_add = row_data[best][notZerosMask]
                cols_to_add = best[notZerosMask]

                for index in range(len(values_to_add)):

                    if numCells == len(rows):
                        rows = np.concatenate((rows, np.zeros(dataBlock, dtype=np.int32)))
                        cols = np.concatenate((cols, np.zeros(dataBlock, dtype=np.int32)))
                        values = np.concatenate((values, np.zeros(dataBlock, dtype=np.float32)))


                    rows[numCells] = current_block_start_row + row_in_block
                    cols[numCells] = cols_to_add[index]
                    values[numCells] = values_to_add[index]

                    numCells += 1


            if time.time() - start_time_printBatch > 60:
                self._print("Processed {} ( {:.2f}% ) in {:.2f} minutes. Rows per second: {:.0f}".format(
                    current_block_start_row,
                    100.0 * float(current_block_start_row) / Pui.shape[1],
                    (time.time() - start_time) / 60,
                    float(current_block_start_row) / (time.time() - start_time)))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_printBatch = time.time()


        self.W_sparse = sps.csr_matrix((values[:numCells], (rows[:numCells], cols[:numCells])), shape=(Pui.shape[1], Pui.shape[1]))

#         if self.normalize_similarity:
#             self.W_sparse = normalize(self.W_sparse, norm='l1', axis=1)


        if self.topK != False:
            self.W_sparse = similarityMatrixTopK(self.W_sparse, k=self.topK)


        self.W_sparse = check_matrix(self.W_sparse, format='csr')

