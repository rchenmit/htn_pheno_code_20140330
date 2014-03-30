## same as mlpy_play.py except, we use number recorded BPs for in/out of control
num_out = 0
num_in = 0
num_mix = 0
for k in range(len(l_status_for_pt_with_htn_meds_recorded_CLINICIAN)):
    if l_status_for_pt_with_htn_meds_recorded_CLINICIAN[k] == -1:
        num_out += 1
    elif l_status_for_pt_with_htn_meds_recorded_CLINICIAN[k] == 1:
        num_in += 1
    else:
        num_mix +=1

print "num_out: ", num_out, "num_in: ", num_in, "num_mix: ", num_mix
        

