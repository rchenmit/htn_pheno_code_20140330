## write dictionaries to CSV files for Kunal / Yu-ying
outputDir = './CSV_FILES_PROCESSED/'

f = open(outputDir + 'd_bp_status_pt_level_clinician.csv','w')
for key in d_bp_status_pt_level_clinician:
    f.write(str(key) + ',' + str(d_bp_status_pt_level_clinician[key]) + '\n') # python will convert \n to os.linesep
f.close() 

df_htn_med_data_MHT = pd.DataFrame()
for key in d_dfdata_per_pt_unique_MHT:
    df = d_dfdata_per_pt_unique_MHT[key]
    numentries = len(df)
    ruid_col = [key] * numentries
    df['RUID'] = ruid_col
    df_htn_med_data_MHT = pd.concat([df_htn_med_data_MHT, df])
df_htn_med_data_MHT.to_csv(outputDir + 'df_htn_med_data_MHT.csv')


##
##f = open(outputDir + 'd_dfdata_per_pt_unique_MHT.csv', 'w')
##    f.write(
