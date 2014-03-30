## load saved pickles for mlpy
import os
import sys

if os.name == 'nt': #'nt' = windows
    sys.path.append('C:\\anaconda\\lib\\site-packages')
    
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


#################################################################
    
with open('./d_dfdata_per_pt.pickle', 'rb') as fhandle:
    d_dfdata_per_pt = pickle.load(fhandle)
fhandle.close()

with open('./d_itemsets_per_pt.pickle', 'rb') as fhandle2:
    d_itemsets_per_pt = pickle.load(fhandle2)
fhandle2.close()
    
with open('./d_dfdata_per_pt_unique.pickle', 'rb') as fhandle3:
    d_dfdata_per_pt_unique = pickle.load(fhandle3)
fhandle3.close()

with open('./d_bp_status_pt_level_clinician.pickle', 'rb') as fhandle4:
    d_bp_status_pt_level_clinician = pickle.load(fhandle4)
fhandle4.close()

with open('./d_bp_status_pt_level.pickle', 'rb') as fhandle5:
    d_bp_status_pt_level = pickle.load(fhandle5)
fhandle5.close()

with open('./d_drug_classes.pickle', 'rb') as fhandle6:
    d_drug_classes = pickle.load(fhandle6)
fhandle6.close()

with open('./d_bp_record.pickle', 'rb') as fhandle7:
    d_bp_record = pickle.load(fhandle7)
fhandle7.close()

with open('./df_data_by_pt.pickle', 'rb') as fhandle8:
    df_data_by_pt = pickle.load(fhandle8)
fhandle8.close()

with open('./df_data_by_drug.pickle', 'rb') as fhandle9:
    df_data_by_drug = pickle.load(fhandle9)
fhandle9.close()

with open('./d_dfdata_per_pt_ALLMEDS.pickle', 'rb') as fhandle10:
    d_dfdata_per_pt_ALLMEDS = pickle.load(fhandle10)
fhandle10.close()

with open('./d_dfdata_per_pt_unique_ALLMEDS.pickle', 'rb') as fhandle11:
    d_dfdata_per_pt_unique_ALLMEDS = pickle.load(fhandle11)
fhandle11.close()

with open('./d_drug_classes_by_name.pickle', 'rb') as fhandle12:
    d_drug_classes_by_name = pickle.load(fhandle12)
fhandle12.close()

with open('./d_dfdata_per_pt_unique_MHT.pickle', 'rb') as fhandle13:
    d_dfdata_per_pt_unique_MHT = pickle.load(fhandle13)
fhandle13.close()

with open('./d_dfdata_per_pt_unique_ALLMEDS_MHT.pickle', 'rb') as fhandle14:
    d_dfdata_per_pt_unique_ALLMEDS_MHT = pickle.load(fhandle14)
fhandle14.close()

with open('./d_bp_clinician.pickle', 'rb') as fhandle15:
    d_bp_clinician = pickle.load(fhandle15)
fhandle15.close()

with open('./l_all_itemsets_across_all_pts.pickle', 'rb') as fhandle16:
    l_all_itemsets_across_all_pts = pickle.load(fhandle16)
fhandle16.close()

with open('./df_bp_clinician.pickle', 'rb') as fhandle17:
    df_bp_clinician = pickle.load(fhandle17)
fhandle17.close()

with open('./l_all_itemsets_in_control.pickle', 'rb') as fhandle18:
    l_all_itemsets_in_control = pickle.load(fhandle18)
fhandle18.close()

with open('./l_all_itemsets_out_control.pickle', 'rb') as fhandle19:
    l_all_itemsets_out_control = pickle.load(fhandle19)
fhandle19.close()

with open('./d_df_labs_record_chunk_readin.pickle', 'rb') as fhandle20:
    d_df_labs_record_chunk_readin = pickle.load(fhandle20)
fhandle20.close()

with open('./d_df_icd_chunk_readin.pickle', 'rb') as fhandle21:
    d_df_icd_chunk_readin = pickle.load(fhandle21)
fhandle21.close()

with open('./d_df_labs_per_pt_part1.pickle', 'rb') as fhandle22:
    d_df_labs_per_pt_part1 = pickle.load(fhandle22)
fhandle22.close()

with open('./d_df_labs_per_pt_part2.pickle', 'rb') as fhandle23:
    d_df_labs_per_pt_part2 = pickle.load(fhandle23)
fhandle23.close()

with open('./d_df_labs_per_pt_part3.pickle', 'rb') as fhandle24:
    d_df_labs_per_pt_part3 = pickle.load(fhandle24)
fhandle24.close()

with open('./d_df_icd_per_pt.pickle', 'rb') as fhandle25:
    d_df_icd_per_pt = pickle.load(fhandle25)
fhandle25.close()

with open('./d_other_diag_clinician_binary.pickle', 'rb') as fhandle26:
    d_other_diag_clinician_binary = pickle.load(fhandle26)
fhandle26.close()

with open('./d_other_diag_clinician.pickle', 'rb') as fhandle27:
    d_other_diag_clinician = pickle.load(fhandle27)
fhandle27.close()

with open('./d_other_diag_pt_level_clinician.pickle', 'rb') as fhandle28:
    d_other_diag_pt_level_clinician = pickle.load(fhandle28)
fhandle28.close()

with open('./d_egfr_pt_level.pickle', 'rb') as fhandle29:
    d_egfr_pt_level = pickle.load(fhandle29)
fhandle29.close()

with open('./d_df_labs_per_pt_MHT.pickle', 'rb') as fhandle30:
    d_df_labs_per_pt_MHT = pickle.load(fhandle30)
fhandle30.close()

with open('./d_df_icd_per_pt_MHT.pickle', 'rb') as fhandle31:
    d_df_icd_per_pt_MHT = pickle.load(fhandle31)
fhandle31.close()

