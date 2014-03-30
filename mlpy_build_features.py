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
