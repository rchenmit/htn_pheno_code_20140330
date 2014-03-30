## Robert Chen
## Monday 2/17/2014
##
## trying to parse this in python
## 

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
from analysis_functions import *

#### check for OS
####isWindows = sys.platform.startswith('win')
####if isWindows:
####    workDir = "C:\\Users\\Thinkpad\\Dropbox\\GT\\GT_Sunlab\\med_status\\ANALYSIS_FULL_DATASET\\code"#data\\samples"
####else:
####    workDir = "/home/x230/Dropbox/GT/GT_Sunlab/med_status/ANALYSIS_FULL_DATASET/code"

## import files ##########################################################################################################
filename = '../data/all_mhav_20130301/Meds.txt'
file_classes = '../data/all_mhav_20130301/MedClasses.xlsx'
file_BP_clinician = '../data/all_mhav_20130301/mht_strategy.txt'
file_BP_record = '../data/all_mhav_20130301/BP.txt'

## Functions  ############################################################################################################
def read_csv_to_df(filename):
    reader = csv.reader(open(filename, 'rU'), delimiter='\t')
    l_headers = next(reader)
    num_cols = len(l_headers)
    indexes_for_df = np.array([])
    data = []
    print("reading in the raw data: ---------------------\n")
    for row in reader:
        index_ruid = int(row[0])
        indexes_for_df = np.append(indexes_for_df, index_ruid)
        vals_for_row = []
        for i in range(1, num_cols):
            vals_for_row.append(row[i])
        data.append(vals_for_row)
    df_data = pd.DataFrame(data, index = indexes_for_df, columns = l_headers[1:])
    return df_data

    
def read_csv_to_df_med(filename):
    reader = csv.reader(open(filename, 'rU'), delimiter='\t')
    l_headers = next(reader)
    
    index_ruid = np.array([]) #all the patient RUID's
    index_drug = np.array([])
    data_for_pt = [] #all the associated data; row by row
    data_for_drug = []
    
    print("reading in the raw data: ---------------------- \n")
    for row in reader:
        ruid = int(row[0])
        entry_date = pd.to_datetime(row[1])
        drug_name = row[2]
        drug_form = row[3]
        drug_strength = row[4]
        dose_amt = row[5]
        route = row[6]
        drug_freq = row[7]
        duration = row[8]

        index_ruid = np.append(index_ruid, ruid) #add RUID
        index_drug = np.append(index_drug, drug_name)
        data_for_pt.append([entry_date, drug_name,drug_form,drug_strength,dose_amt,route, drug_freq,duration])
        data_for_drug.append([ruid, entry_date, drug_form,drug_strength,dose_amt,route, drug_freq,duration])
    
    #create pandas dataframe with patient RUID as index
    df_data_pt = pd.DataFrame(data_for_pt, index=index_ruid, columns=l_headers[1:])
    df_data_drug = pd.DataFrame(data_for_drug, index = index_drug, columns = np.concatenate([l_headers[0:2],l_headers[3:]]))
    return df_data_pt, df_data_drug


## Read in med data ############################################################################################################
df_data_by_pt, df_data_by_drug = read_csv_to_df_med(filename)

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

        

    
    
#take out unique elements

## Read in blood pressures; convert to time series  ###############################################################################
df_bp_clinician = read_csv_to_df(file_BP_clinician)
df_bp_record = read_csv_to_df(file_BP_record)


##analyze recorded BP's ###########################################################################################################
list_ruid = list(set(df_data_by_pt.index.values)) #list of floats

#make dictionary of BP's key'd by RUID
d_bp_record = dict()
for i in range(len(df_bp_record)):
    key = df_bp_record.index[i]
    indexes_for_df = np.array([])
    data = []
    this_date = parser.parse(df_bp_record.iloc[i]['MEASURE_DATE']) ##PARSE THE DATE OUT!
    indexes_for_df = np.append(indexes_for_df, this_date)
    data.append([int(df_bp_record.iloc[i]['SYSTOLIC']), int(df_bp_record.iloc[i]['DIASTOLIC'])]) #CAST ELEMENTS AS INTEGERS!!!!
    if key in d_bp_record: #then append
        d_bp_record[key] = d_bp_record[key].append(pd.DataFrame(data, index = indexes_for_df, columns = ['SYSTOLIC', 'DIASTOLIC']))
    else: #then initialize
        d_bp_record[key] = pd.DataFrame(data, index = indexes_for_df, columns = ['SYSTOLIC', 'DIASTOLIC'])
#add in status at each time point
for key in d_bp_record: #loop thru the keys in dictionary
    #df_this_ruid = d_bp_record[key]
    d_bp_record[key]['STATUS'] = 0
##    for i in range(len(d_bp_record[key])):
##        if d_bp_record[key].iloc[i]['SYSTOLIC'] < 140 and d_bp_record[key].iloc[i]['DIASTOLIC'] < 90:
##            d_bp_record[key].iloc[i]['STATUS'] = 1
##        else:
##            d_bp_record[key].iloc[i]['STATUS'] = -1
    #####tried vectorizing with following: but it didnt work
    bool_condition_systolic = d_bp_record[key]['SYSTOLIC'] < 140
    bool_condition_diastolic = d_bp_record[key]['DIASTOLIC'] < 90
    bool_condition_INCONTROL = bool_condition_systolic & bool_condition_diastolic
    d_bp_record[key].loc[bool_condition_INCONTROL, 'STATUS'] = 1 #-1 => IN CONTROL
    d_bp_record[key].loc[~bool_condition_INCONTROL, 'STATUS'] = -1 #1 => OUT OF CONTROL
            
#make dictionary of BP Control Status (at the patient level, ie mostly in control or out of control)
d_bp_status_pt_level = dict()
for key in d_bp_record:
    if sum(d_bp_record[key]['STATUS']) > 0:
        d_bp_status_pt_level[key] = 1
    elif sum(d_bp_record[key]['STATUS']) < 0:
        d_bp_status_pt_level[key] = -1
    else:
        d_bp_status_pt_level[key] = 0

#dataframe for each patient
#index = Dates for each patient from day 1 to n
#values = med classes (x16), 0 or 1; initialize all to 0
d_dfdata_per_pt = dict()
#loop thru data, put separate for each pt in dict()
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

## drop duplicate rows
d_dfdata_per_pt_unique = dict()
for key in d_dfdata_per_pt:
    value_dfdata_this_key = d_dfdata_per_pt[key].copy()
    value_dfdata_this_key["index"] = value_dfdata_this_key.index
    value_dfdata_this_key = value_dfdata_this_key.drop_duplicates(cols=['DRUG_CLASS', 'index'], take_last=True)
    del value_dfdata_this_key["index"]
    d_dfdata_per_pt_unique[key]= value_dfdata_this_key


## convert to dict of list of lists
# key = pt id
# value = list of lists
d_itemsets_per_pt = dict()
for key in d_dfdata_per_pt_unique:
    df_this_pt = d_dfdata_per_pt_unique[key].copy()
    index_this_pt = df_this_pt.index
    unique_index_this_pt = set(index_this_pt)
    l_itemsets = [] #initialize list
    for date in unique_index_this_pt:
        df_this_date = df_this_pt.loc[date]
        if len(df_this_date) == 1:
            l_itemsets.append([df_this_date.DRUG_CLASS])
        else:
            l_itemsets.append(list(df_this_date.DRUG_CLASS))
    d_itemsets_per_pt[key] = l_itemsets
    


        



        
                               




    
    
    
        
        





