## wrapper

## path
#workingDir= '/home/orbit/Dropbox/GT/GT_Sunlab/med_status/ANALYSIS_FULL_DATASET/code/'
#sys.path.append('C:\\anaconda\\lib\\site-packages')
workingDir = 'C:\\Users\\Thinkpad\\Dropbox\\GT\\GT_Sunlab\\med_status\\ANALYSIS_FULL_DATASET\\code'
pickleDir = 'workingDir\\pickle_20140306_7pm'

## import these
import os
import sys

os.chdir(workingDir)
sys.path.append('./')


## load pickles
#for i in 

## execute files
execfile('dataread_play_mod_mint.py')
execfile('pymining_play.py')
execfile('mlpy_play.py')


