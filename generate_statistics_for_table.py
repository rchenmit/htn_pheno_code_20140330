## generate statistics

## number counts
num_out = 0
num_in = 0
num_mix = 0
for k in d_dfdata_per_pt_unique_MHT:
    if d_bp_status_pt_level_clinician[k] == -1:
        num_out += 1
    elif d_bp_status_pt_level_clinician[k] == 1:
        num_in += 1
    else:
        num_mix +=1

print "num_out: ", num_out, "num_in: ", num_in, "num_mix: ", num_mix
        

## average BP - ONLY FOR PATIENTS USED IN ANALYSIS!!
l_avgSBP_incontrol = []
l_avgSBP_outcontrol = []
l_avgDBP_incontrol = []
l_avgDBP_outcontrol = []
        
for i in l_pt_ruid_with_htn_meds_recorded:
    if i in d_bp_status_pt_level_clinician and i in d_bp_record:
        avg_SBP = np.mean(d_bp_record[i]['SYSTOLIC'])
        avg_DBP = np.mean(d_bp_record[i]['DIASTOLIC'])
        if d_bp_status_pt_level_clinician[i] == -1: #if clinician determined BP, and if status is not mixed
            l_avgSBP_outcontrol.append(avg_SBP)
            l_avgDBP_outcontrol.append(avg_DBP)
        elif d_bp_status_pt_level_clinician[i] == 1:
            l_avgSBP_incontrol.append(avg_SBP)
            l_avgDBP_incontrol.append(avg_DBP)

mean_SBP_incontrol = np.mean(l_avgSBP_incontrol)
mean_DBP_incontrol = np.mean(l_avgDBP_incontrol)
mean_SBP_outcontrol = np.mean(l_avgSBP_outcontrol)
mean_DBP_outcontrol = np.mean(l_avgDBP_outcontrol)

print "mean_SBP_incontrol: ", mean_SBP_incontrol, "std", np.std(l_avgSBP_incontrol), ";mean_DBP_incontrol: ", mean_DBP_incontrol, "std",np.std(l_avgDBP_incontrol), ";mean_SBP_outcontrol: ", mean_SBP_outcontrol,"std",np.std(l_avgSBP_outcontrol), ";mean_DBP_outcontrol: ", mean_DBP_outcontrol, "std",np.std(l_avgDBP_outcontrol)


## print most common frequent patterns:
#for ALL pts
l_t_sorted_itemsets_all_pts = sorted(d_freq_itemsets_across_all_pts.items(), key=lambda x: x[1])
l_t_sorted_itemsets_in_control = sorted(d_freq_itemsets_in_control.items(), key=lambda x: x[1])
l_t_sorted_itemsets_out_control = sorted(d_freq_itemsets_out_control.items(), key=lambda x: x[1])

num_to_display = 30

#for ALL patients
print "number itemsets (all patients): ", len(l_t_sorted_itemsets_all_pts)
for i in range(len(l_t_sorted_itemsets_all_pts),len(l_t_sorted_itemsets_all_pts)-num_to_display , -1):
    print(str(list(l_t_sorted_itemsets_all_pts[i-1][0])) + ':' + str(l_t_sorted_itemsets_all_pts[i-1][1]))
#for IN control
print "number itemsets (IN control): ", len(l_t_sorted_itemsets_in_control)
for i in range(len(l_t_sorted_itemsets_in_control), len(l_t_sorted_itemsets_in_control)-num_to_display, -1):
    print(str(list(l_t_sorted_itemsets_in_control[i-1][0])) + ':' + str(l_t_sorted_itemsets_in_control[i-1][1]))
#for OUT of control
print "number itemsets (OUT of control): ", len(l_t_sorted_itemsets_out_control)
for i in range(len(l_t_sorted_itemsets_out_control),len(l_t_sorted_itemsets_out_control)-num_to_display, -1):
    print(str(list(l_t_sorted_itemsets_out_control[i-1][0])) + ':' + str(l_t_sorted_itemsets_out_control[i-1][1]))    
