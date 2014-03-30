## adding the other diagnoses as features:
## use sklearn to run classification

import numpy as np
import matplotlib.pyplot as plt
import mlpy
import sys
import random
import math
from pylab import * # for plotting stuff

##other way to do CV##############################################################################
def run_cv(x,y,classifier, str_classifier, str_featurespace):
    cv_strats = cross_validation.StratifiedKFold(y, n_folds=10)
    mean_tpr=0.0
    mean_fpr = np.linspace(0,1,100)
    all_tpr = []
    f1 = cross_validation.cross_val_score(classifier, x, y , cv=10, scoring='f1').mean()
    acc = cross_validation.cross_val_score(classifier, x,y, cv=10, scoring='accuracy').mean()
    auc = cross_validation.cross_val_score(classifier, x, y , cv=10, scoring='roc_auc').mean()
    ppv = cross_validation.cross_val_score(classifier, x,y, cv=10, scoring='precision').mean()
    tpr_per_fold = []
    fpr_per_fold = []
    npv_per_fold = []
    auc_per_fold = []
    for i, (train,test) in enumerate(cv_strats):
        probas_ = classifier.fit(x[train], y[train]).predict_proba(x[test])
        #compute ROC curve and AUC
        fpr, tpr, thresholds = metrics.roc_curve(y[test], probas_[:,1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = metrics.auc(fpr, tpr)
        plot(fpr, tpr, lw=1, label='ROC fold %d (area= %0.2f)' % (i, roc_auc))
        tpr_per_fold.append(mean(tpr))
        fpr_per_fold.append(mean(fpr))
        poslabels_from_probas = probas_[:,1].copy()
        neglabels_from_probas = probas_[:,0].copy()
        poslabels_from_probas[poslabels_from_probas >= 0.5] = 1
        poslabels_from_probas[poslabels_from_probas < 0.5] = 0
        neglabels_from_probas[neglabels_from_probas >= 0.5] = 0
        neglabels_from_probas[neglabels_from_probas < 0.5] = 1
        correct_pred_neglabels = neglabels_from_probas.copy()
        num_labeled_neg = sum(neglabels_from_probas)
        for i in range(len(neglabels_from_probas)):
            if neglabels_from_probas[i]==1 and y[test][i] != 1:
                correct_pred_neglabels[i] = 0
        npv = sum(correct_pred_neglabels) / float(num_labeled_neg)
        npv_per_fold.append(npv)
        auc_per_fold.append(roc_auc)
        
    plot([0,1],[0,1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    mean_tpr /= len(cv_strats)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    plot(mean_fpr, mean_tpr, 'k--', label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
    xlim([-0.05, 1.05])
    ylim([-0.05, 1.05])
    xlabel('False Positive Rate')
    ylabel('True Positive Rate')
    title_str = 'Receiver operating characteristic: ' + str_classifier + str_featurespace
    title(title_str )
    legend(loc="lower right")
    show()

    #Determine stats:
    d_results = dict()
    sensitivity = mean(tpr_per_fold)
    specificity = 1 - mean(fpr_per_fold)
    npv = mean(npv_per_fold)
    d_results['ACCURACY'] = acc
    d_results['AUC'] = mean_auc
    d_results['SENSITIVITY'] = sensitivity
    d_results['SPECIFICITY'] = specificity
    d_results['PPV'] = ppv
    d_results['NPV'] = npv

    #print:
    print("CV Results for CLINICIAN_BP_STATUS ~ COUNTS OF MED + Frequent Itemsets")
    print("ACC: " + str(d_results['ACCURACY']))
    print("AUC: " + str(d_results['AUC']))
    print("PPV: " + str(d_results['PPV']))
    print("NPV: " + str(d_results['NPV']))
    print("SENS: " + str(d_results['SENSITIVITY']))
    print("SPEC: " + str(d_results['SPECIFICITY']))

    #d_l_auc_all_featurespaces['HTNMED'] = auc_per_fold
    return auc_per_fold

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
## dictionary for all feature spaces
d_l_auc_all_featurespaces = dict()

        
## run mlpy for CV (ONE ITERATION) : CLINICIAN ~ COUNTS OF MED + OTHER DIAG + EGFR + ITEMSETS
from operator import itemgetter
from sklearn import metrics
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model


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
#x = np.concatenate((feature_matrix_counts, feature_matrix_itemsets ), axis = 1) #combine the MED feature matrix and the FREQITEMSET feature matrix
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

#
x = feature_matrix_counts
str_featurespace = 'HTN meds'
str_classifier = 'Logistic'
classifier = linear_model.LogisticRegression()
d_l_auc_all_featurespaces['HTNMED'] = run_cv(x,y,classifier, str_classifier, str_featurespace)
#
x = feature_matrix_itemsets
str_featurespace = 'HTN Freq'
str_classifier = 'Logistic'
classifier = linear_model.LogisticRegression()
d_l_auc_all_featurespaces['HTNFREQ'] = run_cv(x,y,classifier, str_classifier, str_featurespace)
#
x = np.concatenate((feature_matrix_counts, feature_matrix_itemsets ), axis = 1)
str_featurespace = 'HTN med + HTN Freq'
str_classifier = 'Logistic'
classifier = linear_model.LogisticRegression()
d_l_auc_all_featurespaces['HTNMED_HTNFREQ'] = run_cv(x,y,classifier, str_classifier, str_featurespace)
#
x = feature_matrix_counts_ALLMEDS
str_featurespace = 'ALL Med'
str_classifier = 'Logistic'
classifier = linear_model.LogisticRegression()
d_l_auc_all_featurespaces['ALLMED'] = run_cv(x,y,classifier, str_classifier, str_featurespace)

## plot AUC boxplot
data = [d_l_auc_all_featurespaces['HTNMED'], d_l_auc_all_featurespaces['HTNFREQ'],d_l_auc_all_featurespaces['HTNMED_HTNFREQ'],d_l_auc_all_featurespaces['ALLMED']]
figure()
boxplot(data)
show()
