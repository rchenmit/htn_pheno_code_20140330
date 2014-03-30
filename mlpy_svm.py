## SVM in mlpy package
d_all_classifiers = dict()

## run mlpy for CV (ONE ITERATION) : CLINICIAN ~ PERCENTAGE OF MED
from operator import itemgetter
from sklearn import metrics

#data
x = feature_matrix_percentage #set x (feature matrix)
y = np.array(l_status_for_pt_with_htn_meds_recorded_CLINICIAN) #vector with classes (HTN control status); cast it as numpy array

#cv PARAMETERS
numsamples = len(y)
numfolds = 10
idx = mlpy.cv_kfold(n=numsamples, k=10)
for tr, ts in idx: print(tr, ts) #print out the indexes for CV

#do a k-fold CV
d_results = {'ACCURACY': None, 'AUC': None, 'SENSITIVITY': None, 'SPECIFICITY' :None,  'CORRECT_PER_FOLD' : [],
             'NUM_FOLD': numfolds,'AUC_PER_FOLD': [], 'FPR_PER_FOLD':[],
             'TPR_PER_FOLD':[], 'ACC_PER_FOLD':[],
             'SENS_PER_FOLD': [] ,'SPEC_PER_FOLD':[],
             'REPORT_PER_FOLD': []}
for tr, ts in idx:
    trainset_samples = itemgetter(*tr)(y)
    testset_samples = itemgetter(*ts)(y)
    trainset_features = itemgetter(*tr)(x)
    testset_features = itemgetter(*ts)(x)
    #build the regression model ###################################
    model_svc = mlpy.LibLinear(solver_type='l2r_l2loss_svc_dual') #default: mlpy.LibLinear(solver_type='l2r_lr', C=1, eps=0.01, weight={})
    model_svc.learn(trainset_features, trainset_samples)
    #do the prediction on testing set
    y_predicted = np.array(model_svc.pred(testset_features))
    #record results:
    np_bool_correct_or_not  = (y_predicted ==  testset_samples) ##test if predicted correctly or not!
    numcorrect_this_fold = sum(np_bool_correct_or_not)
    d_results['CORRECT_PER_FOLD'].append(numcorrect_this_fold)
    #sklearn metrics
    report = metrics.classification_report(testset_samples, y_predicted)
    fpr, tpr, thresholds = metrics.roc_curve(testset_samples, y_predicted, pos_label=-1)
    auc = metrics.auc(fpr, tpr)
    acc = metrics.accuracy_score(testset_samples, y_predicted)
    d_results['AUC_PER_FOLD'].append(auc)
    d_results['FPR_PER_FOLD'].append(fpr)
    d_results['TPR_PER_FOLD'].append(tpr) #same as sens
    d_results['ACC_PER_FOLD'].append(acc)
    d_results['SENS_PER_FOLD'].append(tpr)
    d_results['SPEC_PER_FOLD'].append(1-fpr)
    d_results['REPORT_PER_FOLD'].append(report)
#Determine stats:
d_results['ACCURACY'] = sum(d_results['ACC_PER_FOLD']) / float(numfolds)
d_results['AUC'] = sum(d_results['AUC_PER_FOLD']) / float(numfolds)
d_results['SENSITIVITY'] = sum(d_results['SENS_PER_FOLD']) / float(numfolds)
d_results['SPECIFICITY'] = sum(d_results['SPEC_PER_FOLD']) / float(numfolds)

#print:
print("CV Results for CLINICIAN_BP_STATUS ~ Percentage of Med")
print("ACC: " + str(d_results['ACCURACY']))
print("AUC: " + str(d_results['AUC']))
print("SENS: " + str(d_results['SENSITIVITY']))
print("SPEC: " + str(d_results['SPECIFICITY']))

d_all_classifiers['SVC_CLINICIANBP_MED'] = d_results

