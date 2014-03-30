## make weka ARFF file for input

#input matrices (make sure they are loaded already)
l_feature_labels = l_all_drug_classes
mat_feature_mat = feature_matrix_counts
l_subject_vector = l_status_for_pt_with_htn_meds_recorded_CLINICIAN
#output filename
filename = './bp_htn_med_MHT.arff'


f = open(filename,'w')
f.write('@RELATION' + '\t' + 'bpstatus' + '\n')
for item in l_feature_labels:
    attribute_str = "\"" + str(item) + "\""
    writestr = '@ATTRIBUTE' + '\t' +  attribute_str + '\tNUMERIC' + '\n'
    f.write(writestr)
f.write('@ATTRIBUTE' + '\t' + 'class' + '\t' + '{-1,1}' + '\n')
f.write('@DATA' + '\n')
#add in the numpy array with feature vector and response

data = np.concatenate((mat_feature_mat, np.array(l_subject_vector).reshape(len(l_subject_vector),1)), axis=1)
np.savetxt(f, data, fmt='%i', delimiter=',')

f.close()





