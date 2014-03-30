## feature generation for transition point
days_look_back = 300

## analyze recorded BP status: using MHT_strategy.txt (physician reported)######################################################################################
#make dictionary of BP Control Status (at the patient level, ie mostly in control or out of control)
print("calculating intervals of in control vs out of control from recorded numbers: \n")
#d_bp_status_pt_level_clinician = dict()
d_d_transitions_dfdata_per_pt = dict()
d_d_transitions_FROMstatus_per_pt = dict()
for key in d_bp_clinician:
    if key in d_dfdata_per_pt_unique_ALLMEDS:
        d_days_in_out = {-1: 0, 1:0}
        ts_status_this_pt = d_bp_clinician[key]['STATUS'].sort_index()
        last_status = ts_status_this_pt[0]
        last_transition_timestamp = ts_status_this_pt.index[0]
        last_timestamp = ts_status_this_pt.index[0]

        d_transition_dates_FROM_status = dict() #status that you are changing FROM
        d_transition_dates_meds = dict()
        if len(ts_status_this_pt) > 1 and (max(ts_status_this_pt.index) - min(ts_status_this_pt.index)).days > 1: #if there are more than 1 entry, and more than 1 day's worth (if theres more than one entry and they're not all on the same day)
            #loop thru the timeSeries of status for this patient
            for timestamp in ts_status_this_pt.index:
                time_delta = (timestamp - last_timestamp).days
                d_days_in_out[last_status] += time_delta #add the time that has passed
                if ts_status_this_pt[timestamp].size > 1: #multiple clinician assessments on this date
                    status_at_this_timestamp = ts_status_this_pt[timestamp][-1] #pick the last recorded status for this timestamp
                    if status_at_this_timestamp != last_status: #if the status changed############
                        d_transition_dates_FROM_status[timestamp.date()] = last_status #mark the transition date, and the status going backward
                        #d_transition_dates_meds[timestamp.date()] = d_dfdata_per_pt_unique_ALLMEDS[key][(d_dfdata_per_pt_unique_ALLMEDS[key].index >= last_timestamp.date()) & (d_dfdata_per_pt_unique_ALLMEDS[key].index <= timestamp.date())]
                        d_transition_dates_meds[timestamp.date()] = d_dfdata_per_pt_unique_ALLMEDS[key][d_dfdata_per_pt_unique_ALLMEDS[key].index >= last_timestamp.date()-datetime.timedelta(days=days_look_back)]
                        last_status = status_at_this_timestamp
                        last_timestamp = timestamp
                else: #only one clinician assessment on this date
                    status_at_this_timestamp = ts_status_this_pt[timestamp]
                    if status_at_this_timestamp != last_status: #if the status changed############
                        d_transition_dates_FROM_status[timestamp.date()] = last_status
                        #d_transition_dates_meds[timestamp.date()] = d_dfdata_per_pt_unique_ALLMEDS[key][(d_dfdata_per_pt_unique_ALLMEDS[key].index >= last_timestamp.date()) & (d_dfdata_per_pt_unique_ALLMEDS[key].index <= timestamp.date())]
                        d_transition_dates_meds[timestamp.date()] = d_dfdata_per_pt_unique_ALLMEDS[key][d_dfdata_per_pt_unique_ALLMEDS[key].index >= last_timestamp.date()-datetime.timedelta(days=days_look_back)]

                        last_status = status_at_this_timestamp #then change last_status to reflect this so that you add to the right status for the next timestamp
                        last_timestamp = timestamp
            #add transition to master dictionary of transitions / associated meds
            if len(d_transition_dates_FROM_status) >= 1:# if there are transitions for this patient
                d_d_transitions_dfdata_per_pt[key] = d_transition_dates_meds
                d_d_transitions_FROMstatus_per_pt[key] = d_transition_dates_FROM_status
            
## sort into diff dictionaries by BP status
print("sort into diff dicitonaries by BP status")
d_d_df_meds_assoc_transition_per_pt_IN_CONTROL = dict()
d_d_df_meds_assoc_transition_per_pt_OUT_CONTROL = dict()

for key in d_d_transitions_dfdata_per_pt:
    if d_bp_status_pt_level_clinician[key] == 1: #if in control
        d_d_df_meds_assoc_transition_per_pt_IN_CONTROL[key] = d_d_transitions_dfdata_per_pt[key]
    elif d_bp_status_pt_level_clinician[key] == -1:
        d_d_df_meds_assoc_transition_per_pt_OUT_CONTROL[key] = d_d_transitions_dfdata_per_pt[key]

## convert [dict of df] into [dict of [list of lists]]
# key = pt id
# value = list of lists        
print("convert [dict of df] into [dict of [list of lists]]")
l_all_meds_occurred = []
d_transition_med_counts_IN_CONTROL = dict()
d_transition_med_counts_OUT_CONTROL = dict()
d_transition_med_counts_IN_CONTROL[1] = []
d_transition_med_counts_IN_CONTROL[-1] = []
d_transition_med_counts_OUT_CONTROL[1] =[]
d_transition_med_counts_OUT_CONTROL[-1] =[]
for key in d_d_df_meds_assoc_transition_per_pt_IN_CONTROL:
    d_transitions_this_pt = d_d_df_meds_assoc_transition_per_pt_IN_CONTROL[key]
    for k in d_transitions_this_pt:
        temp = d_transitions_this_pt[k].unstack()
        s_counts_this_transition = temp.value_counts()
        fromtype_this_transition = d_d_transitions_FROMstatus_per_pt[key][k]
        d_transition_med_counts_IN_CONTROL[fromtype_this_transition].append(s_counts_this_transition)
for key in d_d_df_meds_assoc_transition_per_pt_OUT_CONTROL:
    d_transitions_this_pt = d_d_df_meds_assoc_transition_per_pt_OUT_CONTROL[key]
    for k in d_transitions_this_pt:
        temp = d_transitions_this_pt[k].unstack()
        s_counts_this_transition = temp.value_counts()
        fromtype_this_transition = d_d_transitions_FROMstatus_per_pt[key][k]
        d_transition_med_counts_OUT_CONTROL[fromtype_this_transition].append(s_counts_this_transition)

## make master feature vector
#all transitions for IN CONTROL
y_IN_CONTROL_TRANSITION_FROM_IN = [1]*len(d_transition_med_counts_IN_CONTROL[1])
y_IN_CONTROL_TRANSITION_FROM_OUT = [-1]*len(d_transition_med_counts_IN_CONTROL[-1])
y_IN_CONTROL_TRANSITION_FROM_ALL = y_IN_CONTROL_TRANSITION_FROM_IN + y_IN_CONTROL_TRANSITION_FROM_OUT
y_OUT_CONTROL_TRANSITION_FROM_IN = [1]*len(d_transition_med_counts_OUT_CONTROL[1])
y_OUT_CONTROL_TRANSITION_FROM_OUT = [-1]*len(d_transition_med_counts_OUT_CONTROL[-1])
y_OUT_CONTROL_TRANSITION_FROM_ALL = y_OUT_CONTROL_TRANSITION_FROM_IN + y_OUT_CONTROL_TRANSITION_FROM_OUT
#get a list of all med names for IN CONTROL patients
l_ALLMEDS_IN_CONTROL = []
for key in d_transition_med_counts_IN_CONTROL:
    for s_meds in d_transition_med_counts_IN_CONTROL[key]:
        if len(s_meds)>=1:
            for j in s_meds.index:
                l_ALLMEDS_IN_CONTROL.append(j)
l_ALLMEDS_IN_CONTROL = list(set(l_ALLMEDS_IN_CONTROL))

#get a list of all med names for OUT OF CONTROL patients
l_ALLMEDS_OUT_CONTROL = []
for key in d_transition_med_counts_OUT_CONTROL:
    for s_meds in d_transition_med_counts_OUT_CONTROL[key]:
        if len(s_meds) >= 1:
            for j in s_meds.index:
                l_ALLMEDS_OUT_CONTROL.append(j)
l_ALLMEDS_OUT_CONTROL = list(set(l_ALLMEDS_OUT_CONTROL))

#build feature vectors for IN CONTROL
num_features = len(l_ALLMEDS_IN_CONTROL)
feature_matrix_med_counts_FROM_IN = np.zeros((len(y_IN_CONTROL_TRANSITION_FROM_IN) , num_features)) #initialize feature matrix to all 0's
for i in range(len(y_IN_CONTROL_TRANSITION_FROM_IN)):
    for drug in d_transition_med_counts_IN_CONTROL[1][i].index:
        index_of_this_drug_in_l_ALLMEDS_IN_CONTROL = l_ALLMEDS_IN_CONTROL.index(drug)
        feature_matrix_med_counts_FROM_IN [i, index_of_this_drug_in_l_ALLMEDS_IN_CONTROL] = d_transition_med_counts_IN_CONTROL[1][i].loc[drug]
feature_matrix_med_counts_FROM_OUT = np.zeros((len(y_IN_CONTROL_TRANSITION_FROM_OUT) , num_features))
for i in range(len(y_IN_CONTROL_TRANSITION_FROM_OUT)):
    for drug in d_transition_med_counts_IN_CONTROL[-1][i].index:
        index_of_this_drug_in_l_ALLMEDS_IN_CONTROL = l_ALLMEDS_IN_CONTROL.index(drug)
        feature_matrix_med_counts_FROM_OUT [i, index_of_this_drug_in_l_ALLMEDS_IN_CONTROL] = d_transition_med_counts_IN_CONTROL[-1][i].loc[drug]
feature_matrix_med_counts_IN_CONTROL_FROM_ALL = np.concatenate((feature_matrix_med_counts_FROM_IN, feature_matrix_med_counts_FROM_OUT))

#build feature vectors for OUT OF CONTROL
num_features = len(l_ALLMEDS_OUT_CONTROL)
feature_matrix_med_counts_FROM_IN = np.zeros((len(y_OUT_CONTROL_TRANSITION_FROM_IN) , num_features)) #initialize feature matrix to all 0's
for i in range(len(y_OUT_CONTROL_TRANSITION_FROM_IN)):
    for drug in d_transition_med_counts_OUT_CONTROL[1][i].index:
        index_of_this_drug_in_l_ALLMEDS_OUT_CONTROL = l_ALLMEDS_OUT_CONTROL.index(drug)
        feature_matrix_med_counts_FROM_IN [i, index_of_this_drug_in_l_ALLMEDS_OUT_CONTROL] = d_transition_med_counts_OUT_CONTROL[1][i].loc[drug]
feature_matrix_med_counts_FROM_OUT = np.zeros((len(y_OUT_CONTROL_TRANSITION_FROM_OUT) , num_features))
for i in range(len(y_OUT_CONTROL_TRANSITION_FROM_OUT)):
    for drug in d_transition_med_counts_OUT_CONTROL[-1][i].index:
        index_of_this_drug_in_l_ALLMEDS_OUT_CONTROL = l_ALLMEDS_OUT_CONTROL.index(drug)
        feature_matrix_med_counts_FROM_OUT [i, index_of_this_drug_in_l_ALLMEDS_OUT_CONTROL] = d_transition_med_counts_OUT_CONTROL[-1][i].loc[drug]
feature_matrix_med_counts_OUT_CONTROL_FROM_ALL = np.concatenate((feature_matrix_med_counts_FROM_IN, feature_matrix_med_counts_FROM_OUT))




