## feature matrix for labs
l_labs = []
for key in d_df_labs_per_pt_MHT:
    for j in list(d_df_labs_per_pt_MHT[key]['LAB_NAME']):
        l_labs.append(j)
l_labs = list(set(l_labs))

num_labs = len(l_labs)
feature_matrix_labs = np.zeros((num_subjects, num_labs))
cnt = 0
for i in range(len(y_ruid)):
    ptid_index = i
    ptid = y_ruid[i]
    if ptid in d_labs_averages_per_pt:
        for lab in d_labs_averages_per_pt[ptid]:
            index_of_this_lab_name_in_l_labs = l_labs.index(lab)
            feature_matrix_labs[ptid_index, index_of_this_lab_name_in_l_labs] = d_labs_averages_per_pt[ptid][lab]
    else:
        cnt = cnt + 1
print("number pts without labs recorded: " + str(cnt))
print("number of labs: " + str(num_labs))

## feature matrix for ICD codes
l_icd = []
for key in d_df_icd_per_pt_MHT:
    for j in list(d_df_icd_per_pt_MHT[key]['ICD 9 CODE']):
        l_icd.append(j)
l_icd = list(set(l_icd))

num_icd = len(l_icd)
feature_matrix_icd = np.zeros((num_subjects, num_icd))
cnt = 0
for i in range(len(y_ruid)):
    ptid_index = i
    ptid = y_ruid[i]
    if ptid in d_icd_counts_per_pt:
        for icd in d_icd_counts_per_pt[ptid].index:
            index_of_this_icd_name_in_l_icd = l_icd.index(icd)
            feature_matrix_icd[ptid_index, index_of_this_icd_name_in_l_icd] = d_icd_counts_per_pt[ptid][icd]
    else:
        cnt = cnt + 1
print("number pts without ICD recorded: " + str(cnt))
print("number of icd: " + str(num_icd))
