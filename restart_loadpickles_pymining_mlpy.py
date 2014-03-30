## wrapper

## path
#workingDir= '/home/orbit/Dropbox/GT/GT_Sunlab/med_status/ANALYSIS_FULL_DATASET/code/'
#sys.path.append('C:\\anaconda\\lib\\site-packages')
workingDir = 'C:\\Users\\Thinkpad\\Dropbox\\GT\\GT_Sunlab\\med_status\\ANALYSIS_FULL_DATASET\\code'
pickleDir = 'workingDir\\pickle_20140306_7pm'

## import these
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


os.chdir(workingDir)
sys.path.append('./')


## load pickles
with open('./d_drug_classes.pickle', 'rb') as fhandle6:
    d_drug_classes = pickle.load(fhandle6)
fhandle6.close()
with open('./d_bp_status_pt_level_clinician.pickle', 'rb') as fhandle4:
    d_bp_status_pt_level_clinician = pickle.load(fhandle4)
fhandle4.close()

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
with open('./l_all_itemsets_across_all_pts.pickle', 'rb') as fhandle16:
    l_all_itemsets_across_all_pts = pickle.load(fhandle16)
fhandle16.close()
with open('./l_all_itemsets_in_control.pickle', 'rb') as fhandle18:
    l_all_itemsets_in_control = pickle.load(fhandle18)
fhandle18.close()

with open('./l_all_itemsets_out_control.pickle', 'rb') as fhandle19:
    l_all_itemsets_out_control = pickle.load(fhandle19)
fhandle19.close()
with open('./d_itemsets_per_pt.pickle', 'rb') as fhandle2:
    d_itemsets_per_pt = pickle.load(fhandle2)
fhandle2.close()
with open('./d_dfdata_per_pt_unique_MHT.pickle', 'rb') as fhandle:
    d_dfdata_per_pt_unique_MHT = pickle.load(fhandle)
fhandle.close()
with open('./d_dfdata_per_pt_unique_ALLMEDS_MHT.pickle', 'rb') as fhandle:
    d_dfdata_per_pt_unique_ALLMEDS_MHT = pickle.load(fhandle)
fhandle.close()
with open('./d_bp_status_pt_level.pickle', 'rb') as fhandle:
    d_bp_status_pt_level = pickle.load(fhandle)
fhandle.close()

## run pymining
execfile('pymining_play.py')
execfile('mlpy_build_features.py')

execfile('generate_statistics_for_table.py')



