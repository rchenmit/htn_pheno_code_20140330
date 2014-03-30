## Robert Chen
## Monday 2/17/2014
##
## trying to parse this in python
## 
if os.name == 'nt': #'nt' = windows
    sys.path.append('C:\\anaconda\\lib\\site-packages') #in windows, alot of modules were installed with Anaconda
import os
import sys
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
        index_ruid = int(row[0])
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
allclasses = df['Hypertension_Med_Classes']
alldrugnames = df['Drug_Name']
allbrandnames = df['Brand_Name']
#put into a dictionary
d_drug_classes = dict()
d_drug_classes_by_name = dict()
for ind in range(len(df['Hypertension_Med_Classes'])):
    key = df['Hypertension_Med_Classes'][ind]
    val_drug = df['Drug_Name'][ind]
    val_brand = df['Brand_Name'][ind]
    if key in d_drug_classes.keys():
        d_drug_classes[key].append(val_drug)
        d_drug_classes[key].append(val_brand)
    else:
        d_drug_classes[key] = list()
        d_drug_classes[key].append(val_drug)
        d_drug_classes[key].append(val_brand)
#do it by drug name/brand name
for ind in range(len(df['Drug_Name'])):
    key1 = df['Drug_Name'][ind]
    key2 = df['Brand_Name'][ind]
    value = df['Hypertension_Med_Classes'][ind]
    if key1 not in d_drug_classes_by_name.keys():
        d_drug_classes_by_name[key1] = value
    if key2 not in d_drug_classes_by_name.keys():
        d_drug_classes_by_name[key2] = value

with open(r"d_drug_classes.pickle", "wb") as output_file:
    pickle.dump(d_drug_classes, output_file)
output_file.close()

    

## Read in blood pressures; convert to time series  ###############################################################################
df_bp_clinician = read_csv_to_df(file_BP_clinician)
df_bp_record = read_csv_to_df(file_BP_record)
#pickle
with open(r"df_bp_clinician.pickle", "wb") as output_file:
    pickle.dump(df_bp_clinician, output_file)
output_file.close()

with open(r"df_bp_record.pickle", "wb") as output_file:
    pickle.dump(df_bp_record, output_file)
output_file.close()


## analyze recorded BP's: using BP.txt (reported numbers)######################################################################################
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
cnt = 0
print("bulding dictionary of clinician determined BP statuses (20K lines total in input file)-----------------\n")
#note: not all lines are htn, some lines are for Diabetes (dm) control status!
for i in range(len(df_bp_clinician)):
    cnt+=1
    if (cnt % 10000 == 0):
        print(cnt) 
    key = df_bp_clinician.index[i]
    indexes_for_df = np.array([])
    data = []
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
        
#print counts
print("number patients with each control class (from numbers: ")
counter_control_status = Counter(val for val in d_bp_status_pt_level_clinician.values())
print(counter_control_status)

#pickle
with open(r"d_bp_clinician.pickle", "wb") as output_file:
    pickle.dump(d_bp_clinician, output_file)
output_file.close()

with open(r"d_bp_status_pt_level_clinician.pickle", "wb") as output_file:
    pickle.dump(d_bp_status_pt_level_clinician, output_file)
output_file.close()

## put med data into dict of DataFrames
#dataframe for each patient
#index = Dates for each patient from day 1 to n
#values = med classes (x16), 0 or 1; initialize all to 0
print("put medication data into a dict of DataFrames")
d_dfdata_per_pt = dict()
#loop thru data, put separate for each pt in dict()
cnt = 0
for i in range(len(df_data_by_pt)):
    key = df_data_by_pt.index[i]
    indexes_for_df = np.array([])
    data = []
    this_date = df_data_by_pt.iloc[i]['ENTRY_DATE'].date() #EXTRACT THE DATE FROM THE TIMESTAMP!!!!
    indexes_for_df = np.append(indexes_for_df, this_date)
    s_this_drug_name = df_data_by_pt.iloc[i]['DRUG_NAME'] #the name of the drug recorded on this line
    if s_this_drug_name in d_drug_classes_by_name:
        s_this_drug_class = d_drug_classes_by_name[s_this_drug_name] #the CLASS of the drug thats assoc'd with the name on this line
        data.append([s_this_drug_class])
        if key in d_dfdata_per_pt and (s_this_drug_class): #then append
            #check if its been recorded for the same date as well!
            if this_date in d_dfdata_per_pt[key].index: # if this_date is already recorded in the dataframe for this pt
                rows_for_this_date = d_dfdata_per_pt[key].ix[this_date] #rows for this date
                #if not (rows_for_this_date.DRUG_CLASS.str.contains(s_this_drug_class)): #use function str.contains to see if the same drug class has not been recorded so far for this date
                if not s_this_drug_class in rows_for_this_date:
                    d_dfdata_per_pt[key] = d_dfdata_per_pt[key].append(pd.DataFrame(data, index = indexes_for_df, columns = ['DRUG_CLASS']))
            else:
                d_dfdata_per_pt[key] = d_dfdata_per_pt[key].append(pd.DataFrame(data, index = indexes_for_df, columns = ['DRUG_CLASS']))

        else:
            d_dfdata_per_pt[key] = pd.DataFrame(data, index = indexes_for_df, columns = ['DRUG_CLASS'])
    cnt += 1
    if cnt % 10000 == 0:
        print(cnt)
    
#pickle
with open(r"d_dfdata_per_pt.pickle", "wb") as output_file:
    pickle.dump(d_dfdata_per_pt, output_file)
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

#pickle
with open(r"d_dfdata_per_pt_unique.pickle", "wb") as output_file:
    pickle.dump(d_dfdata_per_pt_unique, output_file)
output_file.close()

## convert to dict of list of lists
# key = pt id
# value = list of lists
print("convert dictionary of medication data to list of lists")
d_itemsets_per_pt = dict()
for key in d_dfdata_per_pt_unique:
    df_this_pt = d_dfdata_per_pt_unique[key].copy()
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
    

#pickle
with open(r"d_itemsets_per_pt.pickle", "wb") as output_file:
    pickle.dump(d_itemsets_per_pt, output_file)
output_file.close()

    




        
                               




    
    
    
        
        





