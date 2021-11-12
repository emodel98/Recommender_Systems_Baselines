#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy.sparse as sps
import time, sys, copy

from enum import Enum
from Evaluator.metrics import AUROC, ndcg,_Metrics_Object



def seconds_to_biggest_unit(time_in_seconds, data_array = None):

    conversion_factor = [
        ("sec", 60),
        ("min", 60),
        ("hour", 24),
        ("day", 365),
    ]

    terminate = False
    unit_index = 0

    new_time_value = time_in_seconds
    new_time_unit = "sec"

    while not terminate:

        next_time = new_time_value/conversion_factor[unit_index][1]

        if next_time >= 1.0:
            new_time_value = next_time

            if data_array is not None:
                data_array /= conversion_factor[unit_index][1]

            unit_index += 1
            new_time_unit = conversion_factor[unit_index][0]

        else:
            terminate = True


    if data_array is not None:
        return new_time_value, new_time_unit, data_array

    else:
        return new_time_value, new_time_unit



def get_result_string(results_run, n_decimals=7):

    output_str = ""

    for cutoff in results_run.keys():

        results_run_current_cutoff = results_run[cutoff]

        output_str += "CUTOFF: {} - ".format(cutoff)

        for metric in results_run_current_cutoff.keys():
            output_str += "{}: {:.{n_decimals}f}, ".format(metric, results_run_current_cutoff[metric], n_decimals = n_decimals)

        output_str += "\n"

    return output_str

# In[4]:


class EvaluatorMetrics(Enum):
    AUROC = 'AUROC'
    NDCG  = 'NDCG'
    HR    = 'HR'

def _create_empty_metrics_dict(cutoff_list, n_items, n_users, URM_train, URM_test):
    empty_dict = {}
    
    for cutoff in cutoff_list:
        cutoff_dict ={}
        
        for metric in EvaluatorMetrics:
            cutoff_dict[metric.value] = 0.0
            
        empty_dict[cutoff] = cutoff_dict
    return empty_dict

def _remove_item_interactions(URM, item_list):

    URM = sps.csc_matrix(URM.copy())

    for item_index in item_list:

        start_pos = URM.indptr[item_index]
        end_pos = URM.indptr[item_index+1]

        URM.data[start_pos:end_pos] = np.zeros_like(URM.data[start_pos:end_pos])

    URM.eliminate_zeros()
    URM = sps.csr_matrix(URM)

    return URM

class Evaluator(object):
    EVALUATOR_NAME='EVALUATOR_BASE_CLASS'
    
    def __init__(self, URM_test_list, cutoff_list, exclude_seen=True,
                 verbose=True):
        
        self.verbose = verbose
        
        self.cutoff_list = cutoff_list.copy()
        self.max_cutoff = max(self.cutoff_list)

        self.exclude_seen = exclude_seen

        if not isinstance(URM_test_list, list):
            self.URM_test = URM_test_list.copy()
            URM_test_list = [URM_test_list]
        else:
            raise ValueError("List of URM_test not supported")
            

           
        self.n_users, self.n_items = URM_test_list[0].shape
        
        users_to_evaluate_mask = np.ones(self.n_users, dtype=np.bool)

        self.users_to_evaluate = np.arange(self.n_users)[users_to_evaluate_mask]
        
        self.users_to_evaluate = list(self.users_to_evaluate)

        # Those will be set at each new evaluation
        self._start_time = np.nan
        self._start_time_print = np.nan
        self._n_users_evaluated = np.nan
        
    def _print(self, string):

        if self.verbose:
            print("{}: {}".format(self.EVALUATOR_NAME, string))
            
    
    

    
    def evaluateRecommender(self, recommender_object):
        self._start_time = time.time()
        self._start_time_print = time.time()
        self._n_users_evaluated = 0

        
        results_dict = self._run_evaluation_on_selected_users(recommender_object, self.users_to_evaluate)
        
        if self._n_users_evaluated >0:
            for cutoff in self.cutoff_list:
                results_current_cutoff = results_dict[cutoff]
                
                for key in results_current_cutoff.keys():
                    value =results_current_cutoff[key]
                    
                    if isinstance(value, _Metrics_Object):
                        results_current_cutoff[key] = value.get_metric_value()
                    else:
                        results_current_cutoff[key] = value/self._n_users_evaluated
        else:
            self._print("WARNING: No users had a sufficient number of relevant items")
            

        results_run_string = get_result_string(results_dict)            
            
        return results_run_string
    
    
    def get_user_relevant_items(self, user_id):

        assert self.URM_test.getformat() == "csr", "Evaluator_Base_Class: URM_test is not CSR, this will cause errors in getting relevant items"

        return self.URM_test.indices[self.URM_test.indptr[user_id]:self.URM_test.indptr[user_id+1]]


    def get_user_test_ratings(self, user_id):

        assert self.URM_test.getformat() == "csr", "Evaluator_Base_Class: URM_test is not CSR, this will cause errors in relevant items ratings"

        return self.URM_test.data[self.URM_test.indptr[user_id]:self.URM_test.indptr[user_id+1]]


    def _compute_metrics_on_recommendation_list(self, test_user_batch_array, recommended_items_batch_list, scores_batch, results_dict):

        assert len(recommended_items_batch_list) == len(test_user_batch_array), "{}: recommended_items_batch_list contained recommendations for {} users, expected was {}".format(
            self.EVALUATOR_NAME, len(recommended_items_batch_list), len(test_user_batch_array))

        assert scores_batch.shape[0] == len(test_user_batch_array), "{}: scores_batch contained scores for {} users, expected was {}".format(
            self.EVALUATOR_NAME, scores_batch.shape[0], len(test_user_batch_array))

        assert scores_batch.shape[1] == self.n_items, "{}: scores_batch contained scores for {} items, expected was {}".format(
            self.EVALUATOR_NAME, scores_batch.shape[1], self.n_items)


        # Compute recommendation quality for each user in batch
        for batch_user_index in range(len(recommended_items_batch_list)):

            test_user = test_user_batch_array[batch_user_index]

            relevant_items = self.get_user_relevant_items(test_user)

            # Being the URM CSR, the indices are the non-zero column indexes
            recommended_items = recommended_items_batch_list[batch_user_index]
            is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

            self._n_users_evaluated += 1

            for cutoff in self.cutoff_list:

                results_current_cutoff = results_dict[cutoff]

                is_relevant_current_cutoff = is_relevant[0:cutoff]
                recommended_items_current_cutoff = recommended_items[0:cutoff]

                results_current_cutoff[EvaluatorMetrics.AUROC.value]              += AUROC(is_relevant_current_cutoff)
                results_current_cutoff[EvaluatorMetrics.NDCG.value]                 += ndcg(recommended_items_current_cutoff, relevant_items, relevance=self.get_user_test_ratings(test_user), at=cutoff)
                results_current_cutoff[EvaluatorMetrics.HR.value]             += is_relevant_current_cutoff.sum()


        if time.time() - self._start_time_print > 30 or self._n_users_evaluated==len(self.users_to_evaluate):

            elapsed_time = time.time()-self._start_time
            new_time_value, new_time_unit = seconds_to_biggest_unit(elapsed_time)

            self._print("Processed {} ( {:.2f}% ) in {:.2f} {}. Users per second: {:.0f}".format(
                          self._n_users_evaluated,
                          100.0* float(self._n_users_evaluated)/len(self.users_to_evaluate),
                          new_time_value, new_time_unit,
                          float(self._n_users_evaluated)/elapsed_time))

            sys.stdout.flush()
            sys.stderr.flush()

            self._start_time_print = time.time()

        return results_dict



class EvaluatorHoldout(Evaluator):

    EVALUATOR_NAME = "EvaluatorHoldout"


    def __init__(self, URM_test_list, cutoff_list, exclude_seen=True,

                 verbose=True):


        super(EvaluatorHoldout, self).__init__(URM_test_list, cutoff_list, exclude_seen=exclude_seen,
                                               verbose = verbose)




    def _run_evaluation_on_selected_users(self, recommender_object, users_to_evaluate, block_size = None):

        if block_size is None:
            block_size = min(1000, int(1e8/self.n_items))
            block_size = min(block_size, len(users_to_evaluate))


        results_dict = _create_empty_metrics_dict(self.cutoff_list,
                                                  self.n_items, self.n_users,
                                                  recommender_object.get_URM_train(),
                                                  self.URM_test)


        user_batch_start = 0
        user_batch_end = 0

        while user_batch_start < len(users_to_evaluate):

            user_batch_end = user_batch_start + block_size
            user_batch_end = min(user_batch_end, len(users_to_evaluate))

            test_user_batch_array = np.array(users_to_evaluate[user_batch_start:user_batch_end])
            user_batch_start = user_batch_end

            recommended_items_batch_list, scores_batch = recommender_object.recommend(test_user_batch_array,
                                                                      remove_seen_flag=self.exclude_seen,
                                                                      cutoff = self.max_cutoff,
                                                                      return_scores = True
                                                                     )

            results_dict = self._compute_metrics_on_recommendation_list(test_user_batch_array = test_user_batch_array,
                                                         recommended_items_batch_list = recommended_items_batch_list,
                                                         scores_batch = scores_batch,
                                                         results_dict = results_dict)

        return results_dict




