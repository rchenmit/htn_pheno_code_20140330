## frequent pattern using pymining

import pymining
from pymining import itemmining
import operator

import matplotlib
matplotlib.use('Agg')

bool_plot = False #don't plot for full dataset - there's 4.5K patients!
## find raw occurrences / percentage of each med class / labs / ICD codes
#htn drugs COUNTS and PERCENT
d_ts_per_pt = dict()
d_class_counts_per_pt = dict()
d_class_percent_per_pt = dict()
for i in range(len(list(d_dfdata_per_pt_unique_MHT.keys()))):
    key = list(d_dfdata_per_pt_unique_MHT.keys())[i]
    d_ts_per_pt[key] = d_dfdata_per_pt_unique_MHT[key].unstack() #this converts from df to pd.Series
    d_class_counts_per_pt[key] = d_ts_per_pt[key].value_counts()
    d_class_percent_per_pt[key] = d_ts_per_pt[key].value_counts() / sum(d_ts_per_pt[key].value_counts())
#alldrugs COUNTS
d_ts_per_pt = dict()
d_alldrugs_counts_per_pt = dict()
for i in range(len(list(d_dfdata_per_pt_unique_ALLMEDS_MHT.keys()))):
    key = list(d_dfdata_per_pt_unique_ALLMEDS_MHT.keys())[i]
    d_ts_per_pt[key] = d_dfdata_per_pt_unique_ALLMEDS_MHT[key].unstack() #this converts from df to pd.Series
    d_alldrugs_counts_per_pt[key] = d_ts_per_pt[key].value_counts() #the value will be a dict() with k=med name, v=#occurrences
#ICD code COUNTS
d_ts_per_pt = dict()
d_icd_counts_per_pt = dict()
for key in d_df_icd_per_pt_MHT:
    d_ts_per_pt[key] = d_df_icd_per_pt_MHT[key].unstack()
    d_icd_counts_per_pt[key] = d_ts_per_pt[key].value_counts()
#labs AVERAGES
d_ts_per_pt = dict()
d_labs_averages_per_pt = dict()
cnt = 0
for key in d_df_labs_per_pt_MHT:
    d_labs_averages_per_pt[key] = dict()
    d_ts_per_pt[key] = d_df_labs_per_pt_MHT[key].unstack()
    series_labs_occurrences_per_pt = d_ts_per_pt[key]['LAB_NAME'].value_counts()
    for i in range(len(series_labs_occurrences_per_pt)):
        this_lab_name = series_labs_occurrences_per_pt.index[i]
        this_lab_avg = np.mean(d_df_labs_per_pt_MHT[key]['LAB_VALUE'][d_df_labs_per_pt_MHT[key]['LAB_NAME']==this_lab_name])
        d_labs_averages_per_pt[key][this_lab_name] = this_lab_avg
    if cnt % 100 == 0:
        print cnt
    cnt = cnt + 1

        
    
## detect frequent patterns in the aggregate of all itemsets across ALL patients, IN control, OUT of control
## purpose of this is to be able to list the most frequent number of occurrences for each itemset
    ## CHANGE THE VAR NAMES!!! THESE ARE NOT DICTIONARIES!!
#all pts#####
#set of all the patterns (occur more than 1 time) found in across all patients
l_all_itemsets_across_all_pts_AS_FROZENSETS = [frozenset(x) for x in l_all_itemsets_across_all_pts]
d_freq_itemsets_across_all_pts = dict( (fs, l_all_itemsets_across_all_pts_AS_FROZENSETS.count(fs)) for fs in l_all_itemsets_across_all_pts_AS_FROZENSETS)
#now, remove all keys (frozensets) in d_freq_itemsets_across_all_pts if the frozenset only occurred once across all pts
temp_copy = d_freq_itemsets_across_all_pts.copy()
for fs in temp_copy:
    if temp_copy[fs] < 2:
        d_freq_itemsets_across_all_pts.pop(fs, None)    
#in control pts#####
l_all_itemsets_in_control_AS_FROZENSETS = [frozenset(x) for x in l_all_itemsets_in_control]
d_freq_itemsets_in_control = dict( (fs, l_all_itemsets_in_control_AS_FROZENSETS.count(fs)) for fs in l_all_itemsets_in_control_AS_FROZENSETS)
temp_copy = d_freq_itemsets_in_control.copy()
for fs in temp_copy:
    if temp_copy[fs] < 2:
        d_freq_itemsets_in_control.pop(fs, None)   
#out of control pts#####
l_all_itemsets_out_control_AS_FROZENSETS = [frozenset(x) for x in l_all_itemsets_out_control]
d_freq_itemsets_out_control = dict( (fs, l_all_itemsets_out_control_AS_FROZENSETS.count(fs)) for fs in l_all_itemsets_in_control_AS_FROZENSETS)
temp_copy = d_freq_itemsets_out_control.copy()
for fs in temp_copy:
    if temp_copy[fs] < 2:
        d_freq_itemsets_out_control.pop(fs, None)   


## loop thru dict of [list of itemsets] for each patient
        ### use the PYMINING module!
#key = pt id
#value = [list of itemsets]
d_d_freqitemsets_by_pt = dict()
for key in d_itemsets_per_pt:   
    transactions = d_itemsets_per_pt[key]
    relim_input = itemmining.get_relim_input(transactions)
    report  = itemmining.relim(relim_input, min_support=2)
    d_freq_itemsets_this_pt = report
    report_keys = report.keys()
    for fs_itemset in report_keys: #loop thru dict of counts
        if (fs_itemset not in d_freq_itemsets_across_all_pts):#if itemset is a not a freq pattern across all patients
            d_freq_itemsets_this_pt.pop(fs_itemset, None) #then remove the key (itemset) from the dict and dont return an error
    d_d_freqitemsets_by_pt[key] = d_freq_itemsets_this_pt

## plot!
if bool_plot:
    for key in d_d_freqitemsets_by_pt:
        report = d_d_freqitemsets_by_pt[key]
        xlabels = []
        plot_dict = {}
        for k in report:
            s = str(list(k))
            plot_dict[s] = report[k]
        plt.clf()
        fig = plt.figure()
        plt.bar(range(len(plot_dict)), plot_dict.values(), align='center')
        plt.xticks(range(len(plot_dict)), plot_dict.keys())
        locs, labels = plt.xticks() #grab the labels handle
        plt.setp(labels, rotation=90, y=1, fontsize = 9) #set labels to be rotated 90 degrees
        pt_bp_control_status = d_bp_status_pt_level[key]
        savestr = 'bargraph_freqItemSet_pt' + str(int(key)) + '_status_' + str(pt_bp_control_status) #name of the save file
        plt.savefig(savestr)
        fig.clear()
