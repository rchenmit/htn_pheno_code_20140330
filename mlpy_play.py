## use mlpy to run classification

import numpy as np
import matplotlib.pyplot as plt
import mlpy
import sys
import random
import math
from pylab import * # for plotting stuff

## prepare data
l_pt_ruid_with_htn_meds_recorded = d_dfdata_per_pt_unique_MHT.keys()#grab the list of patient ID's that overlap with MHT

l_status_for_pt_with_htn_meds_recorded_CLINICIAN = [] #clinician determined numbers in MHT_strategy.txt
l_status_for_pt_with_htn_meds_recorded_RECORD_NUMBER = [] #recorded numbers in BP.txt
l_pt_ruid_CLINICIAN = []
l_pt_ruid_RECORD_NUMBER = []
l_pt_ruid_CLINICIAN_IN_CONTROL = []
l_pt_ruid_CLINICIAN_OUT_CONTROL = []

for i in l_pt_ruid_with_htn_meds_recorded:
    if i in d_bp_status_pt_level_clinician and d_bp_status_pt_level_clinician[i] != 0: #if clinician determined BP, and if status is not mixed
        l_status_for_pt_with_htn_meds_recorded_CLINICIAN.append(d_bp_status_pt_level_clinician[i])
        l_pt_ruid_CLINICIAN.append(i)
    if i in d_bp_status_pt_level and d_bp_status_pt_level[i] != 0: #if BP record was made, and if status is not mixed
        l_status_for_pt_with_htn_meds_recorded_RECORD_NUMBER.append(d_bp_status_pt_level[i])
        l_pt_ruid_RECORD_NUMBER.append(i)
    if i in d_bp_status_pt_level_clinician and d_bp_status_pt_level_clinician[i] == 1:
        l_pt_ruid_CLINICIAN_IN_CONTROL.append(i)
    if i in d_bp_status_pt_level_clinician and d_bp_status_pt_level_clinician[i] == -1:
        l_pt_ruid_CLINICIAN_OUT_CONTROL.append(i)
## pick response variable
y = l_status_for_pt_with_htn_meds_recorded_CLINICIAN
y_ruid = l_pt_ruid_CLINICIAN
num_subjects = len(y)

## feature matrix for med PERCENTAGES:
l_all_drug_classes = []
for key in d_drug_classes:
    l_all_drug_classes.append(key)
l_ALLMEDS = []
for key in d_dfdata_per_pt_unique_ALLMEDS_MHT:
    for j in list(d_dfdata_per_pt_unique_ALLMEDS_MHT[key]['DRUG_NAME']):
        l_ALLMEDS.append(j)
l_ALLMEDS = list(set(l_ALLMEDS))

num_classes = len(d_drug_classes.keys())
feature_matrix_percentage = np.zeros((num_subjects, num_classes)) #initialize feature matrix to all 0's
for i in range(len(y_ruid)):
    ptid_index= i
    ptid = y_ruid[i]
    for drug_class in d_class_percent_per_pt[ptid].index:
        index_of_this_drug_class_in_l_all_drug_classes = l_all_drug_classes.index(drug_class)
        feature_matrix_percentage [ptid_index, index_of_this_drug_class_in_l_all_drug_classes] = d_class_percent_per_pt[ptid][drug_class]

## feature matrix for htn med COUNTS:
l_all_drug_classes = []
for key in d_drug_classes:
    l_all_drug_classes.append(key)

num_classes = len(d_drug_classes.keys())
feature_matrix_counts = np.zeros((num_subjects, num_classes)) #initialize feature matrix to all 0's
for i in range(len(y_ruid)):
    ptid_index= i
    ptid = y_ruid[i]
    for drug_class in d_class_counts_per_pt[ptid].index:
        index_of_this_drug_class_in_l_all_drug_classes = l_all_drug_classes.index(drug_class)
        feature_matrix_counts [ptid_index, index_of_this_drug_class_in_l_all_drug_classes] = d_class_counts_per_pt[ptid][drug_class]
feature_matrix_counts_IN_CONTROL = np.zeros((len(l_pt_ruid_CLINICIAN_IN_CONTROL), num_classes))
for i in range(len(l_pt_ruid_CLINICIAN_IN_CONTROL)):
    ptid_index= i
    ptid = l_pt_ruid_CLINICIAN_IN_CONTROL[i]
    for drug_class in d_class_counts_per_pt[ptid].index:
        index_of_this_drug_class_in_l_all_drug_classes = l_all_drug_classes.index(drug_class)
        feature_matrix_counts_IN_CONTROL [ptid_index, index_of_this_drug_class_in_l_all_drug_classes] = d_class_counts_per_pt[ptid][drug_class]
feature_matrix_counts_OUT_CONTROL = np.zeros((len(l_pt_ruid_CLINICIAN_OUT_CONTROL), num_classes))
for i in range(len(l_pt_ruid_CLINICIAN_OUT_CONTROL)):
    ptid_index= i
    ptid = l_pt_ruid_CLINICIAN_OUT_CONTROL[i]
    for drug_class in d_class_counts_per_pt[ptid].index:
        index_of_this_drug_class_in_l_all_drug_classes = l_all_drug_classes.index(drug_class)
        feature_matrix_counts_OUT_CONTROL [ptid_index, index_of_this_drug_class_in_l_all_drug_classes] = d_class_counts_per_pt[ptid][drug_class]
       
## feature matrix for ALL MED COUNTS:
num_meds_ALL = len(l_ALLMEDS)
feature_matrix_counts_ALLMEDS = np.zeros((num_subjects, num_meds_ALL))
for i in range(len(y_ruid)):
    ptid_index = i
    ptid = y_ruid[i]
    for drug in d_alldrugs_counts_per_pt[ptid].index:
        index_of_this_drug_name_in_l_ALLMEDS = l_ALLMEDS.index(drug)
        feature_matrix_counts_ALLMEDS[ptid_index, index_of_this_drug_name_in_l_ALLMEDS] = d_alldrugs_counts_per_pt[ptid][drug]

## convert dict of dicts for FREQUENT ITEMSETS: pymining.itemmining result into a matrix
#first, generate list of all itemsets
l_all_itemsets = []
for key in d_d_freqitemsets_by_pt:
    for k in d_d_freqitemsets_by_pt[key]:
        if not(k in l_all_itemsets):
            l_all_itemsets.append(k)

#next, we will generate the feature matrix
num_features = len(l_all_itemsets)
feature_matrix_itemsets = np.zeros((num_subjects, num_features)) #initialize feature matrix to all 0's
##for ptid in d_d_freqitemsets_by_pt:
##    ptid_index = int(ptid)-1;
##    for k in d_d_freqitemsets_by_pt[ptid]:
##        index_of_this_k_in_l_all_itemsets = l_all_itemsets.index(k)
##        feature_matrix[ ptid_index, index_of_this_k_in_l_all_itemsets] = d_d_freqitemsets_by_pt[ptid][k] #put the NUMBER in the right part of the feature matrix
for i in range(len(y_ruid)): #THIS ALLOWS US TO ONLY USE THE PATIENTS (y_ruid) FOR WHICH MEDS ARE RECORDED AND CLINICIAN DETERMINED BP (OR NUMBER BP) ARE ALSO RECORDED!
    ptid_index = i
    ptid = y_ruid[i]
    for k in d_d_freqitemsets_by_pt[ptid]:
        index_of_this_k_in_l_all_itemsets = l_all_itemsets.index(k)
        feature_matrix_itemsets[ ptid_index, index_of_this_k_in_l_all_itemsets] = d_d_freqitemsets_by_pt[ptid][k]

#plot the feature matrix
##plt.clf()
##figure(1)
##plt.imshow(feature_matrix, interpolation='nearest')
##plt.show()

## make dict() for all types of models
d_all_classifiers = dict()
d_all_classifiers['LOGISTIC'] = dict()

## run mlpy for CV (ONE ITERATION) : CLINICIAN ~ COUNTS OF MED
from operator import itemgetter
from sklearn import metrics

#data
x = feature_matrix_counts #set x (feature matrix)
y = np.array(l_status_for_pt_with_htn_meds_recorded_CLINICIAN) #vector with classes (HTN control status); cast it as numpy array

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
    model_logistic = mlpy.LibLinear(solver_type='l2r_lr', C=1) #default: mlpy.LibLinear(solver_type='l2r_lr', C=1, eps=0.01, weight={})
    model_logistic.learn(trainset_features, trainset_samples)
    #do the prediction on testing set
    y_predicted = np.array(model_logistic.pred(testset_features))
    #record results:
    np_bool_correct_or_not  = (y_predicted ==  testset_samples) ##test if predicted correctly or not!
    numcorrect_this_fold = sum(np_bool_correct_or_not)
    d_results['CORRECT_PER_FOLD'].append(numcorrect_this_fold)
    #sklearn metrics
    report = metrics.classification_report(testset_samples, y_predicted)
    fpr, tpr, thresholds = metrics.roc_curve(testset_samples, y_predicted, pos_label=-1)
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
    d_results['PPV_PER_FOLD'].append(PPV)
    d_results['NPV_PER_FOLD'].append(NPV)
    
#Determine stats:
d_results['ACCURACY'] = sum(d_results['ACC_PER_FOLD']) / float(numfolds)
d_results['AUC'] = sum(d_results['AUC_PER_FOLD']) / float(numfolds)
d_results['SENSITIVITY'] = sum(d_results['SENS_PER_FOLD']) / float(numfolds)
d_results['SPECIFICITY'] = sum(d_results['SPEC_PER_FOLD']) / float(numfolds)
d_results['PPV'] = sum(d_results['PPV_PER_FOLD']) / float(numfolds)
d_results['NPV'] = sum(d_results['NPV_PER_FOLD']) / float(numfolds)

#print:
print("CV Results for CLINICIAN_BP_STATUS ~ Counts of Med")
print("ACC: " + str(d_results['ACCURACY']))
print("AUC: " + str(d_results['AUC']))
print("PPV: " + str(d_results['PPV']))
print("NPV: " + str(d_results['NPV']))
print("SENS: " + str(d_results['SENSITIVITY']))
print("SPEC: " + str(d_results['SPECIFICITY']))

d_all_classifiers['LOGISTIC']['CLINICIANBP_MED'] = d_results

## run mlpy for CV (ONE ITERATION) : CLINICIAN ~ FREQ ITEMSETS
from operator import itemgetter
from sklearn import metrics

#data
x = feature_matrix_itemsets #set x (feature matrix)
y = np.array(l_status_for_pt_with_htn_meds_recorded_CLINICIAN) #vector with classes (HTN control status); cast it as numpy array

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
    fpr, tpr, thresholds = metrics.roc_curve(testset_samples, y_predicted, pos_label=-1)
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
    d_results['PPV_PER_FOLD'].append(PPV)
    d_results['NPV_PER_FOLD'].append(NPV)
    
#Determine stats:
d_results['ACCURACY'] = sum(d_results['ACC_PER_FOLD']) / float(numfolds)
d_results['AUC'] = sum(d_results['AUC_PER_FOLD']) / float(numfolds)
d_results['SENSITIVITY'] = sum(d_results['SENS_PER_FOLD']) / float(numfolds)
d_results['SPECIFICITY'] = sum(d_results['SPEC_PER_FOLD']) / float(numfolds)
d_results['PPV'] = sum(d_results['PPV_PER_FOLD']) / float(numfolds)
d_results['NPV'] = sum(d_results['NPV_PER_FOLD']) / float(numfolds)

#print:
print("CV Results for CLINICIAN_BP_STATUS ~ Frequent Itemsets")
print("ACC: " + str(d_results['ACCURACY']))
print("AUC: " + str(d_results['AUC']))
print("PPV: " + str(d_results['PPV']))
print("NPV: " + str(d_results['NPV']))
print("SENS: " + str(d_results['SENSITIVITY']))
print("SPEC: " + str(d_results['SPECIFICITY']))

d_all_classifiers['LOGISTIC']['CLINICIANBP_FREQITEM'] = d_results


## run mlpy for CV (ONE ITERATION) : CLINICIAN ~ COUNTS OF MED + FREQ ITEMSETS
from operator import itemgetter
from sklearn import metrics

#data
x = np.concatenate((feature_matrix_counts, feature_matrix_itemsets), axis = 1) #combine the MED feature matrix and the FREQITEMSET feature matrix
y = np.array(l_status_for_pt_with_htn_meds_recorded_CLINICIAN) #vector with classes (HTN control status); cast it as numpy array

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
    fpr, tpr, thresholds = metrics.roc_curve(testset_samples, y_predicted, pos_label=-1)
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
    d_results['PPV_PER_FOLD'].append(PPV)
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

d_all_classifiers['LOGISTIC']['CLINICIANBP_MED_FREQITEM'] = d_results

## run mlpy for CV (ONE ITERATION) : CLINICIAN ~ ALL MEDS
from operator import itemgetter
from sklearn import metrics

#data
x = feature_matrix_counts_ALLMEDS
y = np.array(l_status_for_pt_with_htn_meds_recorded_CLINICIAN) #vector with classes (HTN control status); cast it as numpy array

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
    fpr, tpr, thresholds = metrics.roc_curve(testset_samples, y_predicted, pos_label=-1)
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
    d_results['PPV_PER_FOLD'].append(PPV)
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

d_all_classifiers['LOGISTIC']['CLINICIANBP_ALLMED'] = d_results


## boxplot of AUC's for all three 
from pylab import *

auc1 = d_all_classifiers['LOGISTIC']['CLINICIANBP_MED']['AUC_PER_FOLD']
auc2 = d_all_classifiers['LOGISTIC']['CLINICIANBP_FREQITEM']['AUC_PER_FOLD']
auc3 = d_all_classifiers['LOGISTIC']['CLINICIANBP_MED_FREQITEM']['AUC_PER_FOLD']
auc4 = d_all_classifiers['LOGISTIC']['CLINICIANBP_ALLMED']['AUC_PER_FOLD']

data = [auc1,auc2,auc3, auc4]

figure()
boxplot(data)
show()

















            
