#!/bin/bash python
import numpy as np
def safe_div(x,y):
    if y == 0:
        return 0.0
    return float(x)/y

def evaluation_metrics(gold_list, pred_list, 
                       measures=['macro_f1', 'acc', 'avgrecall'], 
                       idx_to_label=['negative', 'neutral', 'positive']):
    label_mapper = {}
    for i, label in enumerate(idx_to_label):
        label_mapper[label] = i
    positive_label = label_mapper['positive']
    negative_label = label_mapper['negative']
    
    count_matrix = np.zeros([len(idx_to_label), len(idx_to_label)], dtype=np.int64)
    for gold_label, pred_label in zip(gold_list, pred_list):
        count_matrix[gold_label][pred_label] += 1
    
    result = {}
    if 'macro_f1' in measures:
        acc_pos = safe_div(count_matrix[positive_label, positive_label], np.sum(count_matrix[:, positive_label]))
        recall_pos = safe_div(count_matrix[positive_label, positive_label], np.sum(count_matrix[positive_label, :]))
        F1_pos = safe_div(2*recall_pos*acc_pos,recall_pos+acc_pos)
        result['F1_pos'] = F1_pos

        acc_neg = safe_div(count_matrix[negative_label, negative_label], np.sum(count_matrix[:, negative_label]))
        recall_neg = safe_div(count_matrix[negative_label, negative_label], np.sum(count_matrix[negative_label, :]))
        F1_neg = safe_div(2*recall_neg*acc_neg,recall_neg+acc_neg)
        result['F1_neg'] = F1_neg
        result['macro_f1'] = safe_div(F1_pos + F1_neg, 2)
    
    if 'acc' in measures:
        correct_num = 0
        for i in range(len(idx_to_label)):
            correct_num += count_matrix[i][i]
        result['acc'] = safe_div(correct_num, len(gold_list))
    
    if 'avgrecall' in measures:
        result['avgrecall'] = 0.0
        for i in range(len(idx_to_label)):
            result['avgrecall'] += safe_div(count_matrix[i][i], np.sum(count_matrix[i][:])) / len(idx_to_label)
    result['matrix'] = count_matrix
    return result