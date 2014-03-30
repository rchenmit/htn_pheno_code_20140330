## PURGE THIS!!
## DONT NEED ANYMORE


## parsing dataframe for labs ------- PURGE THIS!!!
list_ruid = list(set(df_labs_record.index.values)) #list of floats
d_labs_record = dict()
cnt = 0
print("bulding dictionary of clinician determined BP statuses (20K lines total in input file)-----------------\n")
#note: not all lines are htn, some lines are for Diabetes (dm) control status!
for i in range(len(df_bp_clinician)):
    cnt+=1
    if (cnt % 10000 == 0):
        print(cnt) 
    key = df_bp_clinician.index[i]
    indexes_for_df = np.array([])
    data = [] #this only has data for THIS LINE!
    this_date = parser.parse(df_bp_clinician.iloc[i]['STRATEGY_DATE']) ##PARSE THE DATE OUT!
    this_lab_name = df_bp_clinician.iloc[i]['DISEASE']
    if this_disease in d_other_diag_clinician:
        d_other_diag_clinician_binary[this_disease][key] = 1
        indexes_for_df = np.append(indexes_for_df, this_date)
        if df_bp_clinician.iloc[i]['CONTROL_LEVEL'] == 'In Control':
            data.append(1) #1 = in control
            if key in d_other_diag_clinician[this_disease]: #then append
                d_other_diag_clinician[this_disease][key] = d_other_diag_clinician[this_disease][key].append(pd.DataFrame(data, index = indexes_for_df, columns = ['STATUS']))
            else: #then initialize
                d_other_diag_clinician[this_disease][key] = pd.DataFrame(data, index = indexes_for_df, columns = ['STATUS'])
        elif df_bp_clinician.iloc[i]['CONTROL_LEVEL'] == 'Out of Control':
            data.append(-1) #-1 = out of control
            if key in d_other_diag_clinician[this_disease]: #then append
                d_other_diag_clinician[this_disease][key] = d_other_diag_clinician[this_disease][key].append(pd.DataFrame(data, index = indexes_for_df, columns = ['STATUS']))
            else: #then initialize
                d_other_diag_clinician[this_disease][key] = pd.DataFrame(data, index = indexes_for_df, columns = ['STATUS'])    
    else:
        d_other_diag_clinician_binary[this_disease] = dict()
        d_other_diag_clinician_binary[this_disease][key] = 1
        d_other_diag_clinician[this_disease] = dict()
        indexes_for_df = np.append(indexes_for_df, this_date)
        if df_bp_clinician.iloc[i]['CONTROL_LEVEL'] == 'In Control':
            data.append(1) #1 = in control
            if key in d_other_diag_clinician[this_disease]: #then append
                d_other_diag_clinician[this_disease][key] = d_other_diag_clinician[this_disease][key].append(pd.DataFrame(data, index = indexes_for_df, columns = ['STATUS']))
            else: #then initialize
                d_other_diag_clinician[this_disease][key] = pd.DataFrame(data, index = indexes_for_df, columns = ['STATUS'])
        elif df_bp_clinician.iloc[i]['CONTROL_LEVEL'] == 'Out of Control':
            data.append(-1) #-1 = out of control
            if key in d_other_diag_clinician[this_disease]: #then append
                d_other_diag_clinician[this_disease][key] = d_other_diag_clinician[this_disease][key].append(pd.DataFrame(data, index = indexes_for_df, columns = ['STATUS']))
            else: #then initialize
                d_other_diag_clinician[this_disease][key] = pd.DataFrame(data, index = indexes_for_df, columns = ['STATUS'])    
 

