import pickle
import pickle5 as p
import pandas as pd
from matplotlib import *
from matplotlib import pyplot as plt
import numpy as np
import mlflow

def loadDictionaryFromPickleFile(dictionaryPath):
    ''' Load the pickle file as a dictionary
    Args:
        dictionaryPath: path to the pickle file
    Return: dictionary from the pickle file
    '''
    filePointer=open(dictionaryPath, 'rb')
    dictionary = p.load(filePointer)
    filePointer.close()
    return dictionary

def saveDictionaryAsPickleFile(dictionary, dictionaryPath):
    ''' Save dictionary as a pickle file
    Args:
        dictionary to be saved
        dictionaryPath: filepath to which the dictionary will be saved
    '''
    filePointer=open(dictionaryPath, 'wb')
    pickle.dump(dictionary,filePointer, protocol=pickle.HIGHEST_PROTOCOL)
    filePointer.close()


def calcMetrics(max_k, k_range, resultFile, gtPath=None, resPath=None, record=True, verbose=False):
    '''Calculate and log the performance metrics: MAP@k, Precision@k, Recall@k, F1@k
    Args:
        max_k: maximum K value (10 for SANTOS, 60 for TUS)
        k_range: step size for K values
        resultFile: dictionary containing search results
        gtPath: path to groundtruth pickle file
        resPath: (deprecated) path to results file
        record: whether to log to MLFlow
        verbose: whether to print intermediate results
    Returns:
        Dictionary containing arrays of metrics at each k:
        {
            'precision': [P@1, P@2, ..., P@max_k],
            'recall': [R@1, R@2, ..., R@max_k],
            'map': [MAP@1, MAP@2, ..., MAP@max_k],
            'f1': [F1@1, F1@2, ..., F1@max_k],
            'used_k': [k values used for evaluation],
            'metrics_at_k': {
                k1: {'precision': P@k1, 'recall': R@k1, 'map': MAP@k1, 'f1': F1@k1},
                k2: {'precision': P@k2, 'recall': R@k2, 'map': MAP@k2, 'f1': F1@k2},
                ...
            }
        }
    '''
    groundtruth = loadDictionaryFromPickleFile(gtPath)
    
    # resultFile is already a dictionary, no need to convert
    precision_array = []
    recall_array = []
    f1_array = []
    
    for k in range(1, max_k+1):
        true_positive = 0
        false_positive = 0
        false_negative = 0
        rec = 0
        ideal_recall = []
        
        # Iterate over queries in the result dictionary
        for query_id, results in resultFile.items():
            if query_id in groundtruth:
                groundtruth_set = set(groundtruth[query_id])
                result_set = set(results[:k])  # Get first k results
                
                # Calculate intersection
                find_intersection = result_set.intersection(groundtruth_set)
                tp = len(find_intersection)
                fp = k - tp
                fn = len(groundtruth_set) - tp
                
                if len(groundtruth_set) >= k:
                    true_positive += tp
                    false_positive += fp
                    false_negative += fn
                rec += tp / len(groundtruth_set)  # Changed from tp/(tp+fn)
                ideal_recall.append(k/len(groundtruth[query_id]))
        
        # Calculate metrics for this k
        num_queries = len(resultFile)
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = rec / num_queries if num_queries > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        precision_array.append(precision)
        recall_array.append(recall)
        f1_array.append(f1)
        
        if verbose:
            print(f"k={k}:")
            print(f"  TP: {true_positive}, FP: {false_positive}, FN: {false_negative}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1: {f1:.4f}")
    
    # Calculate MAP for each k
    map_array = []
    for k in range(1, max_k + 1):
        map_sum = sum(precision_array[:k])
        map_array.append(map_sum/k)
    
    # Store k values used for evaluation
    used_k = [k_range]
    if max_k > k_range:
        for i in range(k_range * 2, max_k+1, k_range):
            used_k.append(i)
    
    # Store metrics at each k point
    metrics_at_k = {}
    for k in used_k:
        metrics_at_k[k] = {
            'precision': precision_array[k-1],
            'recall': recall_array[k-1],
            'map': map_array[k-1],
            'f1': f1_array[k-1]
        }
    
    if verbose:
        print("--------------------------")
        for k in used_k:
            print("Precision at k = ",k,"=", precision_array[k-1])
            print("Recall at k = ",k,"=", recall_array[k-1])
            print("MAP at k = ",k,"=", map_array[k-1])
            print("F1 at k = ",k,"=", f1_array[k-1])
            print("--------------------------")

    # logging to mlflow
    if record:
        mlflow.log_metric("mean_avg_precision", map_array[-1])
        mlflow.log_metric("prec_k", precision_array[-1])
        mlflow.log_metric("recall_k", recall_array[-1])
        mlflow.log_metric("f1_k", f1_array[-1])

    return {
        'precision': precision_array,    # P@k for k=1 to max_k
        'recall': recall_array,          # R@k for k=1 to max_k
        'map': map_array,                # MAP@k for k=1 to max_k
        'f1': f1_array,                  # F1@k for k=1 to max_k
        'used_k': used_k,                # K values used for evaluation
        'metrics_at_k': metrics_at_k     # Metrics at specific k points
    } 