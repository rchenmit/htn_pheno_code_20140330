## for CV multiple iterations

## defining function for cross validation
from sklearn import metrics
def cv_logistic(numfold, numiter, x,y):
    numsamples = len(y)
    numfolds = numfold
    d_results_eachiter = {}
    for cnt_iter in range(numiter):
        idx = mlpy.cv_kfold(n = numsamples, k=numfolds, seed=random.randint(0, sys.maxint)) #make sure you set the random seed!
        d_results = {'ACCURACY': nan , 'SENSITIVITY': nan, 'SPECIFICITY': nan,
                     'FPR': nan, 'PPV': nan,
                     'NPV': nan, 'AUC': nan, 'CORRECT_PER_FOLD' : [],
                     'NUM_FOLD': numfolds, 'TP_PER_FOLD': [], 
                     'FP_PER_FOLD': [], 'TN_PER_FOLD': [], 'FN_PER_FOLD': []}
        cnt_P = 0
        cnt_N = 0
        cnt_TP= 0
        cnt_TN =0
        cnt_FP= 0
        cnt_FN =0
        ##
        all_acc_thisiter = []
        all_sens_thisiter = []
        all_spec_thisiter = []
        all_fpr_thisiter = []
        all_ppv_thisiter = []
        all_npv_thisiter = []
        all_auc_thisiter = []
        for tr, ts in idx:
            cnt_P_thisfold = 0
            cnt_N_thisfold  = 0
            cnt_TP_thisfold = 0
            cnt_TN_thisfold  =0
            cnt_FP_thisfold = 0
            cnt_FN_thisfold  =0
            trainset_samples = itemgetter(*tr)(y)
            testset_samples = itemgetter(*ts)(y)
            trainset_features = itemgetter(*tr)(x)
            testset_features = itemgetter(*ts)(x)
            ## build the regression model
            da = mlpy.DLDA(delta=0.1)
            da.learn(trainset_features, trainset_samples)
            #do the prediction on testing set
            y_predicted = np.array(da.pred(testset_features))
            #record results:
            np_bool_correct_or_not  = (y_predicted ==  testset_samples) ##test if predicted correctly or not!
            numcorrect_this_fold = sum(np_bool_correct_or_not)
            d_results['CORRECT_PER_FOLD'].append(numcorrect_this_fold)
            #determine stats:
            for i in range(len(y_predicted)):
                if testset_samples[i] == -1:
                    cnt_P_thisfold  += 1
                    if y_predicted[i] == testset_samples[i]:
                        cnt_TP_thisfold  += 1
                    else:
                        cnt_FN_thisfold  += 1
                elif testset_samples[i] == 1:
                    cnt_N_thisfold  += 1
                    if y_predicted[i] == testset_samples[i]:
                        cnt_TN_thisfold  += 1
                    else:
                        cnt_FP_thisfold  += 1
                elif testset_samples[i] == 0: #for now, we'll count "mixed HTN status" as a negative!
                    cnt_N_thisfold  += 1
                    if y_predicted[i] == testset_samples[i]:
                        cnt_TN_thisfold  += 1
                    else:
                        cnt_FP_thisfold  +=1
            all_acc_thisiter.append(numcorrect_this_fold / float(len(testset_samples)))
            all_sens_thisiter.append(cnt_TP_thisfold  / float(cnt_P_thisfold))
            all_spec_thisiter.append(cnt_TN_thisfold / float(cnt_N_thisfold))
            #all_fpr_thisiter
            #all_ppv_thisiter
            #all_npv_thisiter
            ## find AUC for this fold
            #fpr, tpr, thresholds = metrics.roc_curve(np.array(testset_samples), y_predicted, pos_label = -1) #-1 is the POSITIVE (the AFFECTED)
            #auc_this_fold = metrics.auc(fpr, tpr)
            #all_auc_thisiter.append(auc_this_fold)
            
        ## results for this CV - fix these!
        d_results['ACCURACY'] = float(sum(d_results['CORRECT_PER_FOLD'])) / float(numfolds*len(testset_samples))
        d_results['SENSITIVITY'] = float(cnt_TP) / float(cnt_P)
        d_results['SPECIFICITY'] = float(cnt_TN) / float(cnt_N)
        d_results['FPR'] = float(cnt_FP) / float(cnt_N)
        d_results['PPV'] = float(cnt_TP) / float(cnt_TP + cnt_FP)
        d_results['NPV'] = float(cnt_TN) / float(cnt_TN + cnt_FN)
        #auc curve
        fpr, tpr, thresholds = metrics.roc_curve(np.array(testset_samples), y_predicted, pos_label = -1) #-1 is the POSITIVE (the AFFECTED)
        print(testset_samples)
        print(y_predicted)
        print(fpr)
        print(tpr)
        print(thresholds)
        d_results['AUC'] = metrics.auc(fpr, tpr)
        #add reults for this iter!
        d_results_eachiter[cnt_iter] = d_results


    ##compile statistics across all iterations
    all_acc = []
    all_sens = []
    all_spec = []
    all_fpr = []
    all_ppv = []
    all_npv = []
    all_auc = []
    for key_iter in d_results_eachiter:
        all_acc.append( d_results_eachiter[key_iter]['ACCURACY'])
        all_sens.append( d_results_eachiter[key_iter]['SENSITIVITY']) #note: sensitivity = TPR
        all_spec.append( d_results_eachiter[key_iter]['SPECIFICITY'])
        all_fpr.append( d_results_eachiter[key_iter]['FPR'])
        all_ppv.append( d_results_eachiter[key_iter]['PPV'])
        all_npv.append( d_results_eachiter[key_iter]['NPV'])
        all_auc.append( d_results_eachiter[key_iter]['AUC'])
    avg_acc = sum(all_acc) / float(len(all_acc))
    avg_sens = sum(all_sens) / float(len(all_sens))
    avg_spec = sum(all_spec) / float(len(all_spec))
    avg_fpr = sum(all_fpr) / float(len(all_fpr))
    avg_ppv = sum(all_ppv) / float(len(all_ppv))
    avg_npv = sum(all_npv) / float(len(all_npv))
    avg_auc = sum(all_auc) / float(len(all_auc))
    ##

    dict_avg_stats_all_iter = {'ACCURACY': avg_acc, 'SENSITIVITY': avg_sens,
                               'SPECIFICITY': avg_spec, 'FPR': avg_fpr,
                               'PPV': avg_ppv, 'NPV': avg_npv,
                               'AUC': avg_auc }
    return dict_avg_stats_all_iter

## do MANY ITERATIONS of CV for logistic fitting:
numfold = 10
numiter = 1

#with freq pattern
input_x = feature_matrix
input_y = np.array(l_control_status)
dict_cv_logistic_results = cv_logistic(numfold, numiter, input_x,input_y)
print(dict_cv_logistic_results)
