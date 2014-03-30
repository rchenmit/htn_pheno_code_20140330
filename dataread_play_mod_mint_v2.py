## Robert Chen
## Monday 3/12/2014
##
## trying to parse this in python
##
import os
import sys
if os.name == 'nt': #'nt' = windows
    sys.path.append('C:\\anaconda\\lib\\site-packages') #in windows, alot of modules were installed with Anaconda
import pandas as pd
import numpy as np
import math
import copy
import csv
import scipy as s
import openpyxl
from openpyxl import load_workbook
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
from dateutil import parser
from collections import Counter
#import pickle
if sys.version_info.major == 3:
    import pickle
else:
    import cPickle as pickle


## ENTER import files ##########################################################################################################
datadir = '../data/all_mhav_20130301_copied_from_pace/'
filename = datadir + 'Meds.txt'
file_classes = datadir + 'MedClasses.xlsx'
file_BP_clinician = datadir + 'MHT_strategy.txt'
file_BP_record = datadir + 'BP.txt'
file_eGFR_record = datadir + 'eGFR.txt'
###file_labs_record = datadir + 'labs.txt'
l_file_labs_record = [datadir + 'labs_chunkaa.txt', datadir + 'labs_chunkab.txt', datadir + 'labs_chunkac.txt',
                      datadir + 'labs_chunkad.txt', datadir + 'labs_chunkae.txt', datadir + 'labs_chunkaf.txt',
                      datadir + 'labs_chunkag.txt', datadir + 'labs_chunkah.txt', datadir + 'labs_chunkai.txt',
                      datadir + 'labs_chunkaj.txt', datadir + 'labs_chunkak.txt', datadir + 'labs_chunkal.txt',
                      datadir + 'labs_chunkam.txt', datadir + 'labs_chunkan.txt', datadir + 'labs_chunkao.txt',
                      datadir + 'labs_chunkap.txt', datadir + 'labs_chunkaq.txt', datadir + 'labs_chunkar.txt',
                      datadir + 'labs_chunkas.txt', datadir + 'labs_chunkat.txt', datadir + 'labs_chunkau.txt',
                      datadir + 'labs_chunkav.txt', datadir + 'labs_chunkaw.txt', datadir + 'labs_chunkax.txt']
l_file_icd9_codes = [datadir + 'icd9_codes_chunkaa.txt', datadir + 'icd9_codes_chunkab.txt', datadir + 'icd9_codes_chunkac.txt',
                     datadir + 'icd9_codes_chunkad.txt', datadir + 'icd9_codes_chunkae.txt', datadir + 'icd9_codes_chunkaf.txt']

## Functions  ############################################################################################################
def read_csv_to_df(filename):
    reader = csv.reader(open(filename, 'rU'), delimiter='\t')
    l_headers = next(reader)
    print(l_headers)
    num_cols = len(l_headers)
    indexes_for_df = np.array([])
    data = []
    print("reading in the raw data: ---------------------\n")
    cnt = 0
    for row in reader:
        cnt +=1
        if (cnt % 10000 == 0):
            print(str(cnt)+ " lines read")
        index_ruid_raw = row[0]
        if index_ruid_raw != "":
            index_ruid = int(index_ruid_raw)
            indexes_for_df = np.append(indexes_for_df, index_ruid)
            vals_for_row = []
            for i in range(1, num_cols):
                vals_for_row.append(row[i])
            data.append(vals_for_row)
    df_data = pd.DataFrame(data, index = indexes_for_df, columns = l_headers[1:])
    return df_data

    
def read_csv_to_df_med(filename):
    print("start reading in raw data: ----------------\n")
    reader = csv.reader(open(filename, 'rU'), delimiter='\t')
    l_headers = next(reader)
    
    index_ruid = np.array([]) #all the patient RUID's
    index_drug = np.array([])
    data_for_pt = [] #all the associated data; row by row
    data_for_drug = []
    
    print("reading in the raw data: ---------------------- \n")
    cnt = 0
    for row in reader:
        cnt +=1
        if (cnt % 10000 == 0):
            print(str(cnt) +  " lines read so far")
        ruid = int(row[0])
        entry_date = pd.to_datetime(row[1])
        drug_name = row[2]
        drug_form = row[3]
        drug_strength = row[4]
        route = row[5]
        dose_amt = row[6]
        drug_freq = row[7]
        duration = row[8]

        index_ruid = np.append(index_ruid, ruid) #add RUID
        index_drug = np.append(index_drug, drug_name)
        data_for_pt.append([entry_date, drug_name,drug_form,drug_strength,dose_amt,route, drug_freq,duration])
        data_for_drug.append([ruid, entry_date, drug_form,drug_strength,dose_amt,route, drug_freq,duration])
    
    #create pandas dataframe with patient RUID as index
    df_data_pt = pd.DataFrame(data_for_pt, index=index_ruid, columns=l_headers[1:9])
    df_data_drug = pd.DataFrame(data_for_drug, index = index_drug, columns = np.concatenate([l_headers[0:2],l_headers[3:9]]))
    return df_data_pt, df_data_drug


## Read in med data ############################################################################################################
df_data_by_pt, df_data_by_drug = read_csv_to_df_med(filename)
#pickle
with open(r"df_data_by_pt.pickle", "wb") as output_file:
    pickle.dump(df_data_by_pt, output_file)
output_file.close()
with open(r"df_data_by_drug.pickle", "wb") as output_file:
    pickle.dump(df_data_by_drug, output_file)
output_file.close()


## Read in med classes  ############################################################################################################
#Read in med classes - using pandas
xls = pd.ExcelFile(file_classes)
df = xls.parse(xls.sheet_names[0])
allclasses = str(df['Hypertension_Med_Classes'])
alldrugnames = str(df['Drug_Name'])
allbrandnames = str(df['Brand_Name'])
#put into a dictionary
d_drug_classes = dict()
d_drug_classes_by_name = dict()
for ind in range(len(df['Hypertension_Med_Classes'])):
    key = str(df['Hypertension_Med_Classes'][ind]).upper()
    val_drug = str(df['Drug_Name'][ind]).upper()  
    val_brand = str(df['Brand_Name'][ind]).upper() 
    if key in d_drug_classes.keys():
        d_drug_classes[key].append(val_drug)
        d_drug_classes[key].append(val_brand)
    else:
        d_drug_classes[key] = list()
        d_drug_classes[key].append(val_drug)
        d_drug_classes[key].append(val_brand)
#do it by drug name/brand name
for ind in range(len(df['Drug_Name'])):
    key1 = str(df['Drug_Name'][ind]).upper()
    key2 = str(df['Brand_Name'][ind]).upper()
    value = str(df['Hypertension_Med_Classes'][ind]).upper()
    if key1 not in d_drug_classes_by_name.keys():
        d_drug_classes_by_name[key1] = value
    if key2 not in d_drug_classes_by_name.keys():
        d_drug_classes_by_name[key2] = value

with open(r"d_drug_classes.pickle", "wb") as output_file:
    pickle.dump(d_drug_classes, output_file)
output_file.close()

with open(r"d_drug_classes_by_name.pickle", "wb") as output_file:
    pickle.dump(d_drug_classes_by_name, output_file)
output_file.close()
    

## Read in blood pressures; eGFR; labs.txt convert to time series  ################################################################################
df_bp_clinician = read_csv_to_df(file_BP_clinician)
df_bp_record = read_csv_to_df(file_BP_record)
df_egfr_record = read_csv_to_df(file_eGFR_record)
#df_labs_record = read_csv_to_df(file_labs_record)
#read in labs separately
d_df_labs_record_chunk_readin = dict()
df_labs_record = pd.DataFrame() #initialize df
d_df_icd_chunk_readin = dict()
df_icd_codes = pd.DataFrame()
#for the labs
for labchunk in l_file_labs_record:
    d_df_labs_record_chunk_readin[labchunk] = read_csv_to_df(labchunk)
#for the ICD9 codes
for chunk in l_file_icd9_codes:
    d_df_icd_chunk_readin[chunk]  = read_csv_to_df(chunk)
    

#pickle
with open(r"df_bp_clinician.pickle", "wb") as output_file:
    pickle.dump(df_bp_clinician, output_file)
output_file.close()

with open(r"df_bp_record.pickle", "wb") as output_file:
    pickle.dump(df_bp_record, output_file)
output_file.close()

with open(r"df_egfr_record.pickle", "wb") as output_file:
    pickle.dump(df_egfr_record, output_file)
output_file.close()

##with open(r"df_labs_record.pickle", "wb") as output_file:
##    pickle.dump(df_labs_record, output_file)
##output_file.close()
##
##with open(r"df_icd_codes.pickle", "wb") as output_file:
##    pickle.dump(df_icd_codes, output_file)
##output_file.close()

p1 = pickle.Pickler(open("d_df_labs_record_chunk_readin.pickle", "wb"))
p1.fast = True
p1.dump(d_df_labs_record_chunk_readin)

p2 = pickle.Pickler(open("d_df_icd_chunk_readin.pickle", "wb"))
p2.fast = True
p2.dump(d_df_icd_chunk_readin)



## analyze recorded EGFR: using GFR.txt (reported numbers) ###############################################################################
list_ruid = list(set(df_egfr_record.index.values)) #list of floats
early_date = datetime(1990,1,1)
late_date = datetime.today()
#make dictionary of eGFR's key'd by RUID
d_egfr_record = dict()
cnt = 0
print ("building dictionary of recorded BP's  (118K lines total) ------------\n")
for i in range(len(df_egfr_record)):
    cnt +=1
    if (cnt % 10000 == 0):
        print(cnt)     
    key = df_egfr_record.index[i]
    indexes_for_df = np.array([])
    data = []
    this_date = parser.parse(df_egfr_record.iloc[i]['LAB_DATE']) ##PARSE THE DATE OUT!
    bool_this_date_good = this_date > early_date and this_date < late_date
    indexes_for_df = np.append(indexes_for_df, this_date)
    if df_egfr_record.iloc[i]['EGFR']!="" and bool_this_date_good:
        data.append([float(df_egfr_record.iloc[i]['EGFR'])]) #CAST ELEMENTS AS FLOAT!!!!
        if key in d_egfr_record: #then append
            d_egfr_record[key] = d_egfr_record[key].append(pd.DataFrame(data, index = indexes_for_df, columns = ['EGFR']))
        else: #then initialize
            d_egfr_record[key] = pd.DataFrame(data, index = indexes_for_df, columns = ['EGFR'])
#eGFR on patient level
d_egfr_pt_level = dict()
for key in d_egfr_record:
    d_egfr_pt_level[key] = np.mean(d_egfr_record[key])
#pickle
with open(r"d_egfr_pt_level.pickle", "wb") as output_file:
    pickle.dump(d_egfr_pt_level, output_file)
output_file.close()


## analyze recorded BP's: using BP.txt (reported numbers)#################################################################################
list_ruid = list(set(df_data_by_pt.index.values)) #list of floats
#earliest and latest possible date : for throwing out bad data
early_date = datetime(1990,1,1)
late_date = datetime.today()

#make dictionary of BP's key'd by RUID
d_bp_record = dict()
cnt = 0
print("bulding dictionary of recorded BP's (346K lines total)-----------------\n")
for i in range(len(df_bp_record)):
    cnt+=1
    if (cnt % 10000 == 0):
        print(cnt) 
    key = df_bp_record.index[i]
    indexes_for_df = np.array([])
    data = []
    this_date = parser.parse(df_bp_record.iloc[i]['MEASURE_DATE']) ##PARSE THE DATE OUT!
    bool_this_date_good = this_date > early_date and this_date < late_date
    indexes_for_df = np.append(indexes_for_df, this_date)
    if df_bp_record.iloc[i]['SYSTOLIC'].isdigit() and df_bp_record.iloc[i]['DIASTOLIC'].isdigit() and bool_this_date_good:
        data.append([int(df_bp_record.iloc[i]['SYSTOLIC']), int(df_bp_record.iloc[i]['DIASTOLIC'])]) #CAST ELEMENTS AS INTEGERS!!!!
        if key in d_bp_record: #then append
            d_bp_record[key] = d_bp_record[key].append(pd.DataFrame(data, index = indexes_for_df, columns = ['SYSTOLIC', 'DIASTOLIC']))
        else: #then initialize
            d_bp_record[key] = pd.DataFrame(data, index = indexes_for_df, columns = ['SYSTOLIC', 'DIASTOLIC'])

#add in status at each time point
print("calculating BP control status from recorded numbers: \n")
for key in d_bp_record: #loop thru the keys in dictionary
    d_bp_record[key]['STATUS'] = 0
    bool_condition_systolic = d_bp_record[key]['SYSTOLIC'] < 140
    bool_condition_diastolic = d_bp_record[key]['DIASTOLIC'] < 90
    bool_condition_INCONTROL = bool_condition_systolic & bool_condition_diastolic
    d_bp_record[key].loc[bool_condition_INCONTROL, 'STATUS'] = 1 #-1 => IN CONTROL
    d_bp_record[key].loc[~bool_condition_INCONTROL, 'STATUS'] = -1 #1 => OUT OF CONTROL
            
#make dictionary of BP Control Status (at the patient level, ie mostly in control or out of control)
print("calculating intervals of in control vs out of control from recorded numbers: \n")
d_bp_status_pt_level = dict()
for key in d_bp_record:
    d_days_in_out = {-1: 0, 1:0}
    ts_status_this_pt = d_bp_record[key]['STATUS'].sort_index()
    last_status = ts_status_this_pt[0]
    last_timestamp = ts_status_this_pt.index[0]

    if len(ts_status_this_pt) > 1 and (max(ts_status_this_pt.index) - min(ts_status_this_pt.index)).days > 1: #if there are more than 1 entry, and more than 1 day's worth (if theres more than one entry and they're not all on the same day)
        #loop thru the timeSeries of status for this patient
        for timestamp in ts_status_this_pt.index:
            time_delta = (timestamp - last_timestamp).days
            d_days_in_out[last_status] += time_delta #add the time that has passed
            if ts_status_this_pt[timestamp].size > 1:
                status_at_this_timestamp = ts_status_this_pt[timestamp][-1] #pick the last recorded status for this timestamp
                if status_at_this_timestamp != last_status: #if the status changed
                    last_status = status_at_this_timestamp           
            else:
                status_at_this_timestamp = ts_status_this_pt[timestamp]
                if status_at_this_timestamp != last_status: #if the status changed
                    last_status = status_at_this_timestamp #then change last_status to reflect this so that you add to the right status for the next timestamp
            last_timestamp = timestamp
                
        #now count how many days in /out and detemrine if mostly in or mostly out or mixed
        num_in = d_days_in_out[1]
        num_out = d_days_in_out[-1]
    else: #if only one BP measurement was taken for the patient
        if last_status == 1:
            num_in = 1
            num_out = 0
        else:
            num_in = 0
            num_out = 1  
    
    if num_in == 0 and num_out == 0:
        print("ERROR 0: no days in or out!  " + str(key))
        d_bp_status_pt_level[key] = 0
    elif num_out == 0:
        if num_in > num_out:
            d_bp_status_pt_level[key] = 1
        else:
            print("ERROR1 - check!")
    elif num_in == 0:
        if num_out > num_in:
            d_bp_status_pt_level[key] = -1
        else:
            print("ERROR2 - check!")
    elif num_in > num_out and num_out == 0:
        d_bp_status_pt_level[key] = 1
    elif num_out > num_in and num_in == 0:
        d_bp_status_pt_level[key] = -1
    elif num_in / float(num_out) > 1.5:
        d_bp_status_pt_level[key] = 1
    elif num_out / float(num_in) > 1.5:
        d_bp_status_pt_level[key] = -1
    else:
        d_bp_status_pt_level[key] = 0
        
#print counts
print("number patients with each control class (from numbers: ")
counter_control_status = Counter(val for val in d_bp_status_pt_level.values())
print(counter_control_status)

#pickle:
with open(r"d_bp_record.pickle", "wb") as output_file:
    pickle.dump(d_bp_record, output_file)
output_file.close()

with open(r"d_bp_status_pt_level.pickle", "wb") as output_file:
    pickle.dump(d_bp_status_pt_level, output_file)
output_file.close()

## analyze recorded BP status: using MHT_strategy.txt (physician reported)######################################################################################
list_ruid = list(set(df_data_by_pt.index.values)) #list of floats
d_bp_clinician = dict()
d_other_diag_clinician_binary = dict()
d_other_diag_clinician = dict()
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
    this_disease = df_bp_clinician.iloc[i]['DISEASE']
    if this_disease == "htn": ##only analyze the lines in dataframe where the disease is HTN!!
        indexes_for_df = np.append(indexes_for_df, this_date)
        if df_bp_clinician.iloc[i]['CONTROL_LEVEL'] == 'In Control':
            data.append(1) #1 = in control
            if key in d_bp_clinician: #then append
                d_bp_clinician[key] = d_bp_clinician[key].append(pd.DataFrame(data, index = indexes_for_df, columns = ['STATUS']))
            else: #then initialize
                d_bp_clinician[key] = pd.DataFrame(data, index = indexes_for_df, columns = ['STATUS'])
        elif df_bp_clinician.iloc[i]['CONTROL_LEVEL'] == 'Out of Control':
            data.append(-1) #-1 = out of control
            if key in d_bp_clinician: #then append
                d_bp_clinician[key] = d_bp_clinician[key].append(pd.DataFrame(data, index = indexes_for_df, columns = ['STATUS']))
            else: #then initialize
                d_bp_clinician[key] = pd.DataFrame(data, index = indexes_for_df, columns = ['STATUS'])
    else: #if its another disease, mark the existence of the other disease
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
     
    
        
#make dictionary of BP Control Status (at the patient level, ie mostly in control or out of control)
print("calculating intervals of in control vs out of control from recorded numbers: \n")
d_bp_status_pt_level_clinician = dict()
for key in d_bp_clinician:
    d_days_in_out = {-1: 0, 1:0}
    ts_status_this_pt = d_bp_clinician[key]['STATUS'].sort_index()
    last_status = ts_status_this_pt[0]
    last_timestamp = ts_status_this_pt.index[0]

    if len(ts_status_this_pt) > 1 and (max(ts_status_this_pt.index) - min(ts_status_this_pt.index)).days > 1: #if there are more than 1 entry, and more than 1 day's worth (if theres more than one entry and they're not all on the same day)
        #loop thru the timeSeries of status for this patient
        for timestamp in ts_status_this_pt.index:
            time_delta = (timestamp - last_timestamp).days
            d_days_in_out[last_status] += time_delta #add the time that has passed
            if ts_status_this_pt[timestamp].size > 1:
                status_at_this_timestamp = ts_status_this_pt[timestamp][-1] #pick the last recorded status for this timestamp
                if status_at_this_timestamp != last_status: #if the status changed
                    last_status = status_at_this_timestamp           
            else:
                status_at_this_timestamp = ts_status_this_pt[timestamp]
                if status_at_this_timestamp != last_status: #if the status changed
                    last_status = status_at_this_timestamp #then change last_status to reflect this so that you add to the right status for the next timestamp
            last_timestamp = timestamp
                
        #now count how many days in /out and detemrine if mostly in or mostly out or mixed
        num_in = d_days_in_out[1]
        num_out = d_days_in_out[-1]
    else: #if only one BP measurement was taken for the patient
        if last_status == 1:
            num_in = 1
            num_out = 0
        else:
            num_in = 0
            num_out = 1  
    
    if num_in == 0 and num_out == 0:
        print("ERROR 0: no days in or out!  " + str(key))
        d_bp_status_pt_level_clinician[key] = 0
    elif num_out == 0:
        if num_in > num_out:
            d_bp_status_pt_level_clinician[key] = 1
        else:
            print("ERROR1 - check!")
    elif num_in == 0:
        if num_out > num_in:
            d_bp_status_pt_level_clinician[key] = -1
        else:
            print("ERROR2 - check!")
    elif num_in > num_out and num_out == 0:
        d_bp_status_pt_level_clinician[key] = 1
    elif num_out > num_in and num_in == 0:
        d_bp_status_pt_level_clinician[key] = -1
    elif num_in / float(num_out) > 1.5:
        d_bp_status_pt_level_clinician[key] = 1
    elif num_out / float(num_in) > 1.5:
        d_bp_status_pt_level_clinician[key] = -1
    else:
        d_bp_status_pt_level_clinician[key] = 0


#for other diseases, determine at the patient level if in control or out of control
print("calculating intervals of in control vs out of control from recorded numbers [OTHER DIAGNOSIS]: \n")
d_other_diag_pt_level_clinician = dict()
for disease in d_other_diag_clinician:
    d_other_diag_pt_level_clinician[disease] = dict()
    for key in d_other_diag_clinician[disease]:
        d_days_in_out = {-1: 0, 1:0}
        ts_status_this_pt = d_other_diag_clinician[disease][key]['STATUS'].sort_index()
        last_status = ts_status_this_pt[0]
        last_timestamp = ts_status_this_pt.index[0]

        if len(ts_status_this_pt) > 1 and (max(ts_status_this_pt.index) - min(ts_status_this_pt.index)).days > 1: #if there are more than 1 entry, and more than 1 day's worth (if theres more than one entry and they're not all on the same day)
            #loop thru the timeSeries of status for this patient
            for timestamp in ts_status_this_pt.index:
                time_delta = (timestamp - last_timestamp).days
                d_days_in_out[last_status] += time_delta #add the time that has passed
                if ts_status_this_pt[timestamp].size > 1:
                    status_at_this_timestamp = ts_status_this_pt[timestamp][-1] #pick the last recorded status for this timestamp
                    if status_at_this_timestamp != last_status: #if the status changed
                        last_status = status_at_this_timestamp           
                else:
                    status_at_this_timestamp = ts_status_this_pt[timestamp]
                    if status_at_this_timestamp != last_status: #if the status changed
                        last_status = status_at_this_timestamp #then change last_status to reflect this so that you add to the right status for the next timestamp
                last_timestamp = timestamp
                    
            #now count how many days in /out and detemrine if mostly in or mostly out or mixed
            num_in = d_days_in_out[1]
            num_out = d_days_in_out[-1]
        else: #if only one BP measurement was taken for the patient
            if last_status == 1:
                num_in = 1
                num_out = 0
            else:
                num_in = 0
                num_out = 1  
        
        if num_in == 0 and num_out == 0:
            print("ERROR 0: no days in or out!  " + str(key))
            d_other_diag_pt_level_clinician[disease][key] = 0
        elif num_out == 0:
            if num_in > num_out:
                d_other_diag_pt_level_clinician[disease][key] = 1
            else:
                print("ERROR1 - check!")
        elif num_in == 0:
            if num_out > num_in:
                d_other_diag_pt_level_clinician[disease][key] = -1
            else:
                print("ERROR2 - check!")
        elif num_in > num_out and num_out == 0:
            d_other_diag_pt_level_clinician[disease][key] = 1
        elif num_out > num_in and num_in == 0:
            d_other_diag_pt_level_clinician[disease][key] = -1
        elif num_in / float(num_out) > 1.5:
            d_other_diag_pt_level_clinician[disease][key] = 1
        elif num_out / float(num_in) > 1.5:
            d_other_diag_pt_level_clinician[disease][key] = -1
        else:
            d_other_diag_pt_level_clinician[disease][key] = 0

        
#print counts
print("number patients with each control class (from clinician assessment): ")
counter_control_status = Counter(val for val in d_bp_status_pt_level_clinician.values())
print(counter_control_status)
#print counts for OTHER diagnosis
print("number patients with each control status for OTHER disease (from clinician assessment): ")
for dz in d_other_diag_pt_level_clinician:
    print(dz)
    counter_control_status = Counter(val for val in d_other_diag_pt_level_clinician[dz].values())
    print(counter_control_status)

#pickle
with open(r"d_bp_clinician.pickle", "wb") as output_file:
    pickle.dump(d_bp_clinician, output_file)
output_file.close()

with open(r"d_bp_status_pt_level_clinician.pickle", "wb") as output_file:
    pickle.dump(d_bp_status_pt_level_clinician, output_file)
output_file.close()

with open(r"d_other_diag_clinician_binary.pickle", "wb") as output_file:
    pickle.dump(d_other_diag_clinician_binary, output_file)
output_file.close()

with open(r"d_other_diag_clinician.pickle", "wb") as output_file:
    pickle.dump(d_other_diag_clinician, output_file)
output_file.close()

with open(r"d_other_diag_pt_level_clinician.pickle", "wb") as output_file:
    pickle.dump(d_other_diag_pt_level_clinician, output_file)
output_file.close()

## put med data into dict of DataFrames
#dataframe for each patient
#index = Dates for each patient from day 1 to n
#values = med classes (x16), 0 or 1; initialize all to 0
print("put medication data into a dict of DataFrames")
d_dfdata_per_pt = dict()
d_dfdata_per_pt_ALLMEDS = dict()
#loop thru data, put separate for each pt in dict()
cnt = 0
for i in range(len(df_data_by_pt)):
    key = df_data_by_pt.index[i]
    indexes_for_df = np.array([])
    data = []
    this_date = df_data_by_pt.iloc[i]['ENTRY_DATE'].date() #EXTRACT THE DATE FROM THE TIMESTAMP!!!!
    indexes_for_df = np.append(indexes_for_df, this_date)
    temp = str(df_data_by_pt.iloc[i]['DRUG_NAME']) #the name of the drug recorded on this line
    s_parse_colon = temp.split(':') # because some drugs are listsed as "GENERIC NAME : BRANDNAME", ex. 'ASPIRIN ENTERIC COATED:ECOTRIN'
    if len(s_parse_colon)>1:
        beforecolon = s_parse_colon[0].upper().strip()
        aftercolon = s_parse_colon[1].upper().strip()
        if beforecolon in d_drug_classes_by_name:
            s_this_drug_name = beforecolon
        else:
            s_this_drug_name = aftercolon
    else:
        s_this_drug_name = s_parse_colon[0].strip()
    if s_this_drug_name in d_drug_classes_by_name:
        s_this_drug_class = d_drug_classes_by_name[s_this_drug_name] #the CLASS of the drug thats assoc'd with the name on this line
        data.append([s_this_drug_class])
        if key in d_dfdata_per_pt and (s_this_drug_class): #then append
            #check if its been recorded for the same date as well!
            if this_date in d_dfdata_per_pt[key].index: # if this_date is already recorded in the dataframe for this pt
                rows_for_this_date = d_dfdata_per_pt[key].ix[this_date] #rows for this date
                #if not (rows_for_this_date.DRUG_CLASS.str.contains(s_this_drug_class)): #use function str.contains to see if the same drug class has not been recorded so far for this date
                if not s_this_drug_class in rows_for_this_date: #if this med hasn't already been recorded for this day
                    d_dfdata_per_pt[key] = d_dfdata_per_pt[key].append(pd.DataFrame(data, index = indexes_for_df, columns = ['DRUG_CLASS']))
            else:
                d_dfdata_per_pt[key] = d_dfdata_per_pt[key].append(pd.DataFrame(data, index = indexes_for_df, columns = ['DRUG_CLASS']))

        else:
            d_dfdata_per_pt[key] = pd.DataFrame(data, index = indexes_for_df, columns = ['DRUG_CLASS'])

    ## dictionary for ALL meds (not just HTN meds)
    if isinstance(s_this_drug_name, str): ## if the med is not actually a string
        data = []
        indexes_for_df = np.array([])
        indexes_for_df = np.append(indexes_for_df, this_date)
        data.append([s_this_drug_name])
        if key in d_dfdata_per_pt_ALLMEDS:
            if this_date in d_dfdata_per_pt_ALLMEDS[key].index: #if no meds for this date have been recorded
                rows_for_this_date = d_dfdata_per_pt_ALLMEDS[key].ix[this_date]
                if not s_this_drug_name in rows_for_this_date: #if this med hasn't already been recorded for this day
                    d_dfdata_per_pt_ALLMEDS[key] = d_dfdata_per_pt_ALLMEDS[key].append(pd.DataFrame(data, index = indexes_for_df, columns = ['DRUG_NAME']))
            else:
                d_dfdata_per_pt_ALLMEDS[key] = d_dfdata_per_pt_ALLMEDS[key].append(pd.DataFrame(data, index = indexes_for_df, columns = ['DRUG_NAME']))
        else: #if patient doesn't have a key in the dicitonary, create a new entry
            d_dfdata_per_pt_ALLMEDS[key] = pd.DataFrame(data, index = indexes_for_df, columns = ['DRUG_NAME'])

    ## count progress
    cnt += 1
    if cnt % 10000 == 0:
        print(cnt)
    
#pickle
with open(r"d_dfdata_per_pt.pickle", "wb") as output_file:
    pickle.dump(d_dfdata_per_pt, output_file)
output_file.close()
with open(r"d_dfdata_per_pt_ALLMEDS.pickle", "wb") as output_file:
    pickle.dump(d_dfdata_per_pt_ALLMEDS, output_file)
output_file.close()

## put LAB data into dict of dataframes 
d_df_labs_per_pt = dict()
d_df_icd_per_pt = dict()
#for labs
cnt = 0
#for chunk in d_df_labs_record_chunk_readin:

#to finish parsing from where error happened (line 8670 in chunk ai)
chunk = '../data/all_mhav_20130301_copied_from_pace/labs_chunkai.txt'
i_start = 8670
df_chunk = d_df_labs_record_chunk_readin[chunk]
i = i_start
while i<len(df_chunk):
    key = df_chunk.index[i]
    indexes_for_df = np.array([])
    data = []
    this_date = parser.parse(df_chunk.iloc[i]['LAB_DATE']) #EXTRACT THE DATE FROM THE TIMESTAMP!!!!
    s_this_lab_name = df_chunk.iloc[i]['LAB_NAME']
    f_this_lab_value = float(df_chunk.iloc[i]['LAB_VALUE'])
    indexes_for_df = np.append(indexes_for_df, this_date)
    cols_for_df = ['LAB_NAME', 'LAB_VALUE']
    data.append([s_this_lab_name, f_this_lab_value])
    if key in d_df_labs_per_pt: #then append
        d_df_labs_per_pt[key]= d_df_labs_per_pt[key].append(pd.DataFrame(data, index = indexes_for_df, columns = cols_for_df))
    else:
        d_df_labs_per_pt[key] = pd.DataFrame(data, index = indexes_for_df, columns = cols_for_df)
    ## count progress
    cnt += 1
    if cnt % 10000 == 0:
        print(cnt)    
    i = i + 1

l_remaining_labs_to_parse = ['../data/all_mhav_20130301_copied_from_pace/labs_chunkaw.txt', '../data/all_mhav_20130301_copied_from_pace/labs_chunkad.txt', '../data/all_mhav_20130301_copied_from_pace/labs_chunkak.txt', '../data/all_mhav_20130301_copied_from_pace/labs_chunkaf.txt', '../data/all_mhav_20130301_copied_from_pace/labs_chunkar.txt', '../data/all_mhav_20130301_copied_from_pace/labs_chunkao.txt', '../data/all_mhav_20130301_copied_from_pace/labs_chunkan.txt', '../data/all_mhav_20130301_copied_from_pace/labs_chunkau.txt', '../data/all_mhav_20130301_copied_from_pace/labs_chunkap.txt', '../data/all_mhav_20130301_copied_from_pace/labs_chunkal.txt', '../data/all_mhav_20130301_copied_from_pace/labs_chunkax.txt', '../data/all_mhav_20130301_copied_from_pace/labs_chunkag.txt']
for chunk in l_remaining_labs_to_parse:
    df_chunk = d_df_labs_record_chunk_readin[chunk]
    for i in range(len(df_chunk)):
        key = df_chunk.index[i]
        indexes_for_df = np.array([])
        data = []
        this_date = parser.parse(df_chunk.iloc[i]['LAB_DATE']) #EXTRACT THE DATE FROM THE TIMESTAMP!!!!
        s_this_lab_name = df_chunk.iloc[i]['LAB_NAME']
        f_this_lab_value = float(df_chunk.iloc[i]['LAB_VALUE'])
        indexes_for_df = np.append(indexes_for_df, this_date)
        cols_for_df = ['LAB_NAME', 'LAB_VALUE']
        data.append([s_this_lab_name, f_this_lab_value])
        if key in d_df_labs_per_pt: #then append
            d_df_labs_per_pt[key]= d_df_labs_per_pt[key].append(pd.DataFrame(data, index = indexes_for_df, columns = cols_for_df))
        else:
            d_df_labs_per_pt[key] = pd.DataFrame(data, index = indexes_for_df, columns = cols_for_df)
        ## count progress
        cnt += 1
        if cnt % 10000 == 0:
            print(cnt)

d_df_labs_per_pt_part2 = dict()
i_start = 130302
df_chunk = d_df_labs_record_chunk_readin[chunk]
i = i_start
while i<len(df_chunk):
    key = df_chunk.index[i]
    indexes_for_df = np.array([])
    data = []
    this_date = parser.parse(df_chunk.iloc[i]['LAB_DATE']) #EXTRACT THE DATE FROM THE TIMESTAMP!!!!
    s_this_lab_name = df_chunk.iloc[i]['LAB_NAME']
    f_this_lab_value = float(df_chunk.iloc[i]['LAB_VALUE'])
    indexes_for_df = np.append(indexes_for_df, this_date)
    cols_for_df = ['LAB_NAME', 'LAB_VALUE']
    data.append([s_this_lab_name, f_this_lab_value])
    if key in d_df_labs_per_pt_part2: #then append
        d_df_labs_per_pt_part2[key]= d_df_labs_per_pt_part2[key].append(pd.DataFrame(data, index = indexes_for_df, columns = cols_for_df))
    else:
        d_df_labs_per_pt_part2[key] = pd.DataFrame(data, index = indexes_for_df, columns = cols_for_df)
    ## count progress
    cnt += 1
    if cnt % 10000 == 0:
        print(cnt)    
    i = i + 1

#for ICD codes
cnt = 0
for chunk in d_df_icd_chunk_readin:
    df_chunk = d_df_icd_chunk_readin[chunk]
    for i in range(len(df_chunk)):
        key = df_chunk.index[i]
        indexes_for_df = np.array([])
        data = []
        this_date = parser.parse(df_chunk.iloc[i]['LAB_DATE']) #EXTRACT THE DATE FROM THE TIMESTAMP!!!!
        f_this_lab_value = float(df_chunk.iloc[i]['LAB_VALUE'])
        indexes_for_df = np.append(indexes_for_df, this_date)
        cols_for_df = ['ICD 9 CODE']
        data.append([f_this_lab_value])
        if key in d_df_labs_per_pt: #then append
            d_df_icd_per_pt[key]= d_df_icd_per_pt[key].append(data, index = indexes_for_df, columns = cols_for_df)
        else:
            d_df_icd_per_pt[key] = pd.DataFrame(data, index = indexes_for_df, columns = cols_for_df)
        ## count progress
        cnt += 1
        if cnt % 10000 == 0:
            print(cnt)
with open(r"d_df_labs_per_pt_part1.pickle", "wb") as output_file:
    pickle.dump(d_df_labs_per_pt, output_file)
output_file.close()
with open(r"d_df_icd_per_pt.pickle", "wb") as output_file:
    pickle.dump(d_df_icd_per_pt, output_file)
output_file.close()
## drop duplicate rows for patient data
print("drop duplicate rows for patient medication data")
d_dfdata_per_pt_unique = dict()
for key in d_dfdata_per_pt:
    value_dfdata_this_key = d_dfdata_per_pt[key].copy()
    value_dfdata_this_key["index"] = value_dfdata_this_key.index
    value_dfdata_this_key = value_dfdata_this_key.drop_duplicates(cols=['DRUG_CLASS', 'index'], take_last=True)
    del value_dfdata_this_key["index"]
    d_dfdata_per_pt_unique[key]= value_dfdata_this_key
print("drop duplicate rows for patient medication data")
d_dfdata_per_pt_unique_ALLMEDS = dict()
for key in d_dfdata_per_pt_ALLMEDS:
    value_dfdata_this_key = d_dfdata_per_pt_ALLMEDS[key].copy()
    value_dfdata_this_key["index"] = value_dfdata_this_key.index
    value_dfdata_this_key = value_dfdata_this_key.drop_duplicates(cols=['DRUG_NAME', 'index'], take_last=True)
    del value_dfdata_this_key["index"]
    d_dfdata_per_pt_unique_ALLMEDS[key]= value_dfdata_this_key
## only use MHT intervals!
d_dfdata_per_pt_unique_MHT = dict()
d_dfdata_per_pt_unique_ALLMEDS_MHT = dict()
d_df_labs_per_pt_MHT = dict()
for key in d_dfdata_per_pt_unique:
    if key in d_bp_clinician: ##put it in the dict representing overlap of MHT
        df_this_pt = d_dfdata_per_pt_unique[key].copy()
        index_this_pt = df_this_pt.index
        mht_start_date = d_bp_clinician[key].sort_index().index[0].date()
        mht_end_date = d_bp_clinician[key].sort_index().index[-1].date()
        new_df_this_pt = d_dfdata_per_pt_unique[key][(d_dfdata_per_pt_unique[key].index >= mht_start_date) & (d_dfdata_per_pt_unique[key].index <= mht_end_date)]
        d_dfdata_per_pt_unique_MHT[key] = new_df_this_pt 
for key in d_dfdata_per_pt_unique_ALLMEDS:
    if key in d_bp_clinician: ##put it in the dict representing overlap of MHT
        df_this_pt = d_dfdata_per_pt_unique_ALLMEDS[key].copy()
        index_this_pt = df_this_pt.index
        mht_start_date = d_bp_clinician[key].sort_index().index[0].date()
        mht_end_date = d_bp_clinician[key].sort_index().index[-1].date()
        new_df_this_pt = d_dfdata_per_pt_unique_ALLMEDS[key][(d_dfdata_per_pt_unique_ALLMEDS[key].index >= mht_start_date) & (d_dfdata_per_pt_unique_ALLMEDS[key].index <= mht_end_date)]
        d_dfdata_per_pt_unique_ALLMEDS_MHT[key] = new_df_this_pt
for key in d_df_labs_per_pt_part1:#part 1
    if key in d_bp_clinician:
        df_this_pt = d_df_labs_per_pt_part1[key].copy()
        index_this_pt = df_this_pt.index
        mht_start_date = d_bp_clinician[key].sort_index().index[0]
        mht_end_date = d_bp_clinician[key].sort_index().index[-1]
        new_df_this_pt = d_df_labs_per_pt_part1[key][(d_df_labs_per_pt_part1[key].index >= mht_start_date) & (d_df_labs_per_pt_part1[key].index <= mht_end_date)]
        d_df_labs_per_pt_MHT[key] = new_df_this_pt
for key in d_df_labs_per_pt_part2:#part 2
    if key in d_bp_clinician:
        df_this_pt = d_df_labs_per_pt_part2[key].copy()
        index_this_pt = df_this_pt.index
        mht_start_date = d_bp_clinician[key].sort_index().index[0]
        mht_end_date = d_bp_clinician[key].sort_index().index[-1]
        new_df_this_pt = d_df_labs_per_pt_part2[key][(d_df_labs_per_pt_part2[key].index >= mht_start_date) & (d_df_labs_per_pt_part2[key].index <= mht_end_date)]
        d_df_labs_per_pt_MHT[key] = new_df_this_pt
for key in d_df_labs_per_pt_part3:#part 3
    if key in d_bp_clinician:
        df_this_pt = d_df_labs_per_pt_part3[key].copy()
        index_this_pt = df_this_pt.index
        mht_start_date = d_bp_clinician[key].sort_index().index[0]
        mht_end_date = d_bp_clinician[key].sort_index().index[-1]
        new_df_this_pt = d_df_labs_per_pt_part3[key][(d_df_labs_per_pt_part3[key].index >= mht_start_date) & (d_df_labs_per_pt_part3[key].index <= mht_end_date)]
        d_df_labs_per_pt_MHT[key] = new_df_this_pt
d_df_icd_per_pt_MHT = dict()
for key in d_df_icd_per_pt:
    if key in d_bp_clinician:
        df_this_pt = d_df_icd_per_pt[key].copy()
        index_this_pt = df_this_pt.index
        mht_start_date = d_bp_clinician[key].sort_index().index[0]
        mht_end_date = d_bp_clinician[key].sort_index().index[-1]
        new_df_this_pt = d_df_icd_per_pt[key][(d_df_icd_per_pt[key].index >= mht_start_date) & (d_df_icd_per_pt[key].index <= mht_end_date)]
        d_df_icd_per_pt_MHT[key] = new_df_this_pt        
#delete keys where theres NO entries that overlap with MHT dates (ie, the len of the value(a df) is 0):
l_keys = d_dfdata_per_pt_unique_MHT.keys()
for k in l_keys:
    if len(d_dfdata_per_pt_unique_MHT[k]) == 0:
        del d_dfdata_per_pt_unique_MHT[k]
l_keys = d_dfdata_per_pt_unique_ALLMEDS_MHT.keys()
for k in l_keys:
    if len(d_dfdata_per_pt_unique_ALLMEDS_MHT[k]) == 0:
        del d_dfdata_per_pt_unique_ALLMEDS_MHT[k]
l_keys = d_df_labs_per_pt_MHT.keys()
for k in l_keys:
    if len(d_df_labs_per_pt_MHT[k]) == 0:
        del d_df_labs_per_pt_MHT[k]
l_keys = d_df_icd_per_pt_MHT.keys()
for k in l_keys:
    if len(d_df_icd_per_pt_MHT[k]) == 0:
        del d_df_icd_per_pt_MHT[k]
 
#pickle
with open(r"d_dfdata_per_pt_unique.pickle", "wb") as output_file:
    pickle.dump(d_dfdata_per_pt_unique, output_file)
output_file.close()
with open(r"d_dfdata_per_pt_unique_ALLMEDS.pickle", "wb") as output_file:
    pickle.dump(d_dfdata_per_pt_unique_ALLMEDS, output_file)
output_file.close()
with open(r"d_dfdata_per_pt_unique_MHT.pickle", "wb") as output_file:
    pickle.dump(d_dfdata_per_pt_unique_MHT, output_file)
output_file.close()
with open(r"d_dfdata_per_pt_unique_ALLMEDS_MHT.pickle", "wb") as output_file:
    pickle.dump(d_dfdata_per_pt_unique_ALLMEDS_MHT, output_file)
output_file.close()
#### new adds on 3/20/2014
with open(r"d_df_labs_per_pt_MHT.pickle", "wb") as output_file:
    pickle.dump(d_df_labs_per_pt_MHT, output_file)
output_file.close()
with open(r"d_df_icd_per_pt_MHT.pickle", "wb") as output_file:
    pickle.dump(d_df_icd_per_pt_MHT, output_file)
output_file.close()
## create dict of list of lists for frequent itemsets
# key = pt id
# value = list of lists
print("convert dictionary of medication data to list of lists")
d_itemsets_per_pt = dict()
l_all_itemsets_across_all_pts = [] #use this for mining itemsets ACROSS ALL PATIENTS
l_all_itemsets_in_control = []
l_all_itemsets_out_control = []
for key in d_dfdata_per_pt_unique_MHT:
    df_this_pt = d_dfdata_per_pt_unique_MHT[key].copy()
    index_this_pt = df_this_pt.index
    unique_index_this_pt = set(index_this_pt)
    l_itemsets = [] #initialize list
    for date in unique_index_this_pt:
        df_this_date = df_this_pt.loc[date]
        if len(df_this_date) == 1:
            l_itemsets.append([df_this_date['DRUG_CLASS']])
        else:
            l_itemsets.append(list(df_this_date['DRUG_CLASS']))
    d_itemsets_per_pt[key] = l_itemsets
    #add all itemsets to the master list of all itemsets
    for s in l_itemsets:
        l_all_itemsets_across_all_pts.append(s)
        if d_bp_status_pt_level_clinician[key] == -1: #if out of control
            l_all_itemsets_out_control.append(s)
        elif d_bp_status_pt_level_clinician[key] == 1: #if in control
            l_all_itemsets_in_control.append(s)


#pickle
with open(r"d_itemsets_per_pt.pickle", "wb") as output_file:
    pickle.dump(d_itemsets_per_pt, output_file)
output_file.close()
with open(r"l_all_itemsets_across_all_pts.pickle", "wb") as output_file:
    pickle.dump(l_all_itemsets_across_all_pts, output_file)
output_file.close()
with open(r"l_all_itemsets_in_control.pickle", "wb") as output_file:
    pickle.dump(l_all_itemsets_in_control, output_file)
output_file.close()
with open(r"l_all_itemsets_out_control.pickle", "wb") as output_file:
    pickle.dump(l_all_itemsets_out_control, output_file)
output_file.close()


    

        
                               




    
    
    
        
        





