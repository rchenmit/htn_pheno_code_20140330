## adding the other diagnoses as features:
## use mlpy to run classification

import numpy as np
import matplotlib.pyplot as plt
import mlpy
import sys
import random
import math
from pylab import * # for plotting stuff

## feature matrix for other diagnosis:
l_other_diag = d_other_diag_clinician_binary.keys()
num_diag = len(d_other_diag_clinician_binary)
feature_matrix_diag = np.zeros((num_subjects, num_diag))
for dz in d_other_diag_clinician_binary:
    vector_this_dz = []
    for i in range(len(y_ruid)):
        ptid_index= i
        ptid = y_ruid[i]
        if ptid in d_other_diag_clinician_binary[dz]:
            index_of_this_diag_in_l_other_diag = l_other_diag.index(dz)
            feature_matrix_diag [ptid_index, index_of_this_diag_in_l_other_diag] = d_other_diag_clinician_binary[dz][ptid]
feature_vector_DM = feature_matrix_diag[:,0].reshape([num_subjects, 1])
feature_vector_CHF = feature_matrix_diag[:,1].reshape([num_subjects, 1])
y_DM = feature_matrix_diag[:,0]
y_DM = np.array([int(y) for y in y_DM])
y_CHF = feature_matrix_diag[:,1]
y_CHF = np.array([int(y) for y in y_CHF])
## feature matrix for eGFR
feature_matrix_egfr = np.zeros((num_subjects, 1))
for i in range(len(y_ruid)):
    ptid_index= i
    ptid = y_ruid[i]
    if ptid in d_egfr_pt_level:
        feature_matrix_egfr [ptid_index] = d_egfr_pt_level[ptid]

## run mlpy for CV (ONE ITERATION) : CLINICIAN ~ COUNTS OF MED + OTHER DIAG + EGFR + ITEMSETS
from operator import itemgetter
from sklearn import metrics

#data
#x = feature_matrix_counts_ALLMEDS
#x = feature_vector_CHF
#x = feature_matrix_egfr
#x = feature_matrix_diag
#x = feature_matrix_itemsets
#x = feature_matrix_counts
#x = np.concatenate((feature_matrix_counts_ALLMEDS, feature_matrix_egfr), axis = 1) #combine the MED feature matrix and the FREQITEMSET feature matrix
#x = np.concatenate((feature_matrix_counts_ALLMEDS, feature_matrix_itemsets), axis = 1) #combine the MED feature matrix and the FREQITEMSET feature matrix
#x = np.concatenate((feature_matrix_counts_ALLMEDS, feature_matrix_itemsets,feature_matrix_diag ), axis = 1) #combine the MED feature matrix and the FREQITEMSET feature matrix
#x = np.concatenate((feature_matrix_counts, feature_matrix_itemsets,feature_matrix_diag ), axis = 1) #combine the MED feature matrix and the FREQITEMSET feature matrix
x = np.concatenate((feature_matrix_counts, feature_matrix_itemsets ), axis = 1) #combine the MED feature matrix and the FREQITEMSET feature matrix
#x = np.concatenate((feature_matrix_counts, feature_matrix_itemsets, feature_matrix_labs), axis = 1) #combine the MED feature matrix and the FREQITEMSET feature matrix
#x = np.concatenate((feature_matrix_counts, feature_matrix_itemsets, feature_matrix_icd), axis = 1) #combine the MED feature matrix and the FREQITEMSET feature matrix
#x = np.concatenate(( feature_matrix_labs, feature_matrix_icd), axis = 1) #combine the MED feature matrix and the FREQITEMSET feature matrix
#x = np.concatenate(( feature_matrix_counts, feature_matrix_labs), axis = 1) #combine the MED feature matrix and the FREQITEMSET feature matrix
#x = np.concatenate(( feature_matrix_counts, feature_matrix_labs,feature_matrix_icd), axis = 1) #combine the MED feature matrix and the FREQITEMSET feature matrix
#x = np.concatenate(( feature_matrix_counts, feature_matrix_icd), axis = 1) #combine the MED feature matrix and the FREQITEMSET feature matrix
#x = feature_matrix_icd
y = np.array(l_status_for_pt_with_htn_meds_recorded_CLINICIAN) #vector with classes (HTN control status); cast it as numpy array
#y = y_DM
#y = y_CHF
#cv PARAMETERS
numsamples = len(y)
numfolds = 10
idx = mlpy.cv_kfold(n=numsamples, k=10)
#for tr, ts in idx: print(tr, ts) #print out the indexes for CV

#do a k-fold CV
d_results = {'ACCURACY': None, 'AUC': None, 'SENSITIVITY': None, 'SPECIFICITY' :None,  'CORRECT_PER_FOLD' : [],
             'NUM_FOLD': numfolds,'AUC_PER_FOLD': [], 'FPR_PER_FOLD':[],
             'TPR_PER_FOLD':[], 'ACC_PER_FOLD':[],
             'SENS_PER_FOLD': [] ,'SPEC_PER_FOLD':[],
             'REPORT_PER_FOLD': [],
             'PPV_PER_FOLD':[], 'NPV_PER_FOLD':[],
             'PPV': None, 'NPV': None}
for tr, ts in idx:
    trainset_samples = itemgetter(*tr)(y)
    testset_samples = itemgetter(*ts)(y)
    trainset_features = itemgetter(*tr)(x)
    testset_features = itemgetter(*ts)(x)
    #build the regression model ###################################
    model_logistic = mlpy.LibLinear(solver_type='l2r_lr') #default: mlpy.LibLinear(solver_type='l2r_lr', C=1, eps=0.01, weight={})
    model_logistic.learn(trainset_features, trainset_samples)
    #do the prediction on testing set
    y_predicted = np.array(model_logistic.pred(testset_features))
    #record results:
    np_bool_correct_or_not  = (y_predicted ==  testset_samples) ##test if predicted correctly or not!
    numcorrect_this_fold = sum(np_bool_correct_or_not)
    d_results['CORRECT_PER_FOLD'].append(numcorrect_this_fold)
    #sklearn metrics
    report = metrics.classification_report(testset_samples, y_predicted)
    fpr, tpr, thresholds = metrics.roc_curve(testset_samples, y_predicted, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    acc = metrics.accuracy_score(testset_samples, y_predicted)
    confusion_matrix = metrics.confusion_matrix(testset_samples, y_predicted)
    TN = confusion_matrix[0,0]
    FN = confusion_matrix[1,0]
    FP = confusion_matrix[0,1]
    TP = confusion_matrix[1,1]
    PPV = TP / float(TP + FP)
    NPV = TN / float(TN + FN)
    d_results['AUC_PER_FOLD'].append(auc)
    d_results['FPR_PER_FOLD'].append(fpr)
    d_results['TPR_PER_FOLD'].append(tpr) #same as sens
    d_results['ACC_PER_FOLD'].append(acc)
    d_results['SENS_PER_FOLD'].append(tpr[1])
    d_results['SPEC_PER_FOLD'].append(1-fpr[1])
    d_results['REPORT_PER_FOLD'].append(report)
    if PPV >= 0:
        d_results['PPV_PER_FOLD'].append(PPV)
    if NPV >=0:
        d_results['NPV_PER_FOLD'].append(NPV)
    
#Determine stats:
d_results['ACCURACY'] = sum(d_results['ACC_PER_FOLD']) / float(numfolds)
d_results['AUC'] = sum(d_results['AUC_PER_FOLD']) / float(numfolds)
d_results['SENSITIVITY'] = sum(d_results['SENS_PER_FOLD']) / float(numfolds)
d_results['SPECIFICITY'] = sum(d_results['SPEC_PER_FOLD']) / float(numfolds)
d_results['PPV'] = sum(d_results['PPV_PER_FOLD']) / float(numfolds)
d_results['NPV'] = sum(d_results['NPV_PER_FOLD']) / float(numfolds)

#print:
print("CV Results for CLINICIAN_BP_STATUS ~ COUNTS OF MED + Frequent Itemsets")
print("ACC: " + str(d_results['ACCURACY']))
print("AUC: " + str(d_results['AUC']))
print("PPV: " + str(d_results['PPV']))
print("NPV: " + str(d_results['NPV']))
print("SENS: " + str(d_results['SENSITIVITY']))
print("SPEC: " + str(d_results['SPECIFICITY']))

d_all_classifiers['LOGISTIC']['CLINICIANBP_MED_OTHERDIAG_EGFR'] = d_results
