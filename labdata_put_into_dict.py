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
import gc




## define dir
workingDir = 'C:\\Users\\Thinkpad\\Dropbox\\GT\\GT_Sunlab\\med_status\\ANALYSIS_FULL_DATASET\\code'
pickleDir = 'workingDir\\pickle_20140306_7pm'
import os
import sys
os.chdir(workingDir)

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

## read in pickels
##with open('./d_df_labs_record_chunk_readin.pickle', 'rb') as fhandle20:
##    d_df_labs_record_chunk_readin = pickle.load(fhandle20)
##fhandle20.close()

with open('./d_df_icd_chunk_readin.pickle', 'rb') as fhandle21:
    d_df_icd_chunk_readin = pickle.load(fhandle21)
fhandle21.close()

## read in labs.txt again:
d_df_labs_record_chunk_readin = dict()
df_labs_record = pd.DataFrame() #initialize df
for labchunk in l_file_labs_record:
    d_df_labs_record_chunk_readin[labchunk] = read_csv_to_df(labchunk)
with open(r"d_df_labs_record_chunk_readin.pickle", "wb") as output_file:
    pickle.dump(d_df_labs_record_chunk_readin, output_file)
output_file.close()
## output data structs
d_df_labs_per_pt_part1 = dict()
d_df_labs_per_pt_part2 = dict()
d_df_labs_per_pt_part3 = dict()

#for labs
cnt = 0
#for chunk in d_df_labs_record_chunk_readin:
##
l_remaining_labs_to_parse = [datadir + 'labs_chunkaa.txt', datadir + 'labs_chunkab.txt', datadir + 'labs_chunkac.txt',
                      datadir + 'labs_chunkad.txt', datadir + 'labs_chunkae.txt', datadir + 'labs_chunkaf.txt',
                      datadir + 'labs_chunkag.txt', datadir + 'labs_chunkah.txt', datadir + 'labs_chunkai.txt']
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
        if key in d_df_labs_per_pt_part1: #then append
            d_df_labs_per_pt_part1[key]= d_df_labs_per_pt_part1[key].append(pd.DataFrame(data, index = indexes_for_df, columns = cols_for_df))
        else:
            d_df_labs_per_pt_part1[key] = pd.DataFrame(data, index = indexes_for_df, columns = cols_for_df)
        ## count progress
        cnt += 1
        if cnt % 10000 == 0:
            print(cnt)
with open(r"d_df_labs_per_pt_part1.pickle", "wb") as output_file:
    pickle.dump(d_df_labs_per_pt_part1, output_file)
output_file.close()
del d_df_labs_per_pt_part1
gc.collect()
##
l_remaining_labs_to_parse = [datadir + 'labs_chunkaj.txt', datadir + 'labs_chunkak.txt', datadir + 'labs_chunkal.txt',
                      datadir + 'labs_chunkam.txt', datadir + 'labs_chunkan.txt', datadir + 'labs_chunkao.txt',
                      datadir + 'labs_chunkap.txt', datadir + 'labs_chunkaq.txt', datadir + 'labs_chunkar.txt']
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
        if key in d_df_labs_per_pt_part2: #then append
            d_df_labs_per_pt_part2[key]= d_df_labs_per_pt_part2[key].append(pd.DataFrame(data, index = indexes_for_df, columns = cols_for_df))
        else:
            d_df_labs_per_pt_part2[key] = pd.DataFrame(data, index = indexes_for_df, columns = cols_for_df)
        ## count progress
        cnt += 1
        if cnt % 10000 == 0:
            print(cnt)
with open(r"d_df_labs_per_pt_part2.pickle", "wb") as output_file:
    pickle.dump(d_df_labs_per_pt_part2, output_file)
output_file.close()
del d_df_labs_per_pt_part2
gc.collect()
##
l_remaining_labs_to_parse = [datadir + 'labs_chunkas.txt', datadir + 'labs_chunkat.txt', datadir + 'labs_chunkau.txt',
                      datadir + 'labs_chunkav.txt', datadir + 'labs_chunkaw.txt', datadir + 'labs_chunkax.txt']

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
        if key in d_df_labs_per_pt_part3: #then append
            d_df_labs_per_pt_part3[key]= d_df_labs_per_pt_part3[key].append(pd.DataFrame(data, index = indexes_for_df, columns = cols_for_df))
        else:
            d_df_labs_per_pt_part3[key] = pd.DataFrame(data, index = indexes_for_df, columns = cols_for_df)
        ## count progress
        cnt += 1
        if cnt % 10000 == 0:
            print(cnt)
with open(r"d_df_labs_per_pt_part3.pickle", "wb") as output_file:
    pickle.dump(d_df_labs_per_pt_part3, output_file)
output_file.close()
del d_df_labs_per_pt_part3
gc.collect()
##
#for ICD codes
d_df_icd_per_pt = dict()
cnt = 0
for chunk in d_df_icd_chunk_readin:
    df_chunk = d_df_icd_chunk_readin[chunk]
    for i in range(len(df_chunk)):
        key = df_chunk.index[i]
        indexes_for_df = np.array([])
        data = []
        this_date = parser.parse(df_chunk.iloc[i]['EVENT_DATE']) #EXTRACT THE DATE FROM THE TIMESTAMP!!!!
        f_this_icd_code = df_chunk.iloc[i]['ICD 9 CODE']
        indexes_for_df = np.append(indexes_for_df, this_date)
        cols_for_df = ['ICD 9 CODE']
        data.append([f_this_icd_code])
        if key in d_df_icd_per_pt: #then append
            d_df_icd_per_pt[key]= d_df_icd_per_pt[key].append(pd.DataFrame(data, index = indexes_for_df, columns = cols_for_df))
        else:
            d_df_icd_per_pt[key] = pd.DataFrame(data, index = indexes_for_df, columns = cols_for_df)
        ## count progress
        cnt += 1
        if cnt % 10000 == 0:
            print(cnt)
with open(r"d_df_icd_per_pt.pickle", "wb") as output_file:
    pickle.dump(d_df_icd_per_pt, output_file)
output_file.close()
gc.collect()
