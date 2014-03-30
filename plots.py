## plots

import pandas as pd
import scipy as s
import numpy as np
import matplotlib
matplotlib.use('Agg')

## plot the occurences for each class
###plot the number of occurences of each drug
#first, count the number of occurences for each class
d_ts_per_pt = dict()
d_class_counts_per_pt = dict()
d_class_percent_per_pt = dict()
for i in range(len(list(d_dfdata_per_pt.keys()))):
    key = list(d_dfdata_per_pt.keys())[i]
    d_ts_per_pt[key] = d_dfdata_per_pt[key].unstack() #this converts from df to pd.Series
    d_class_counts_per_pt[key] = d_ts_per_pt[key].value_counts()
    d_class_percent_per_pt[key] = d_ts_per_pt[key].value_counts() / sum(d_ts_per_pt[key].value_counts())

## for making the plot and saving it! ####################################

## plot the raw number of occurence
plt.clf()
fig = plt.figure()

df_class_counts_per_pt = pd.DataFrame(d_class_counts_per_pt)

ax = plt.subplot(111)
df_class_counts_per_pt.transpose().plot(kind='bar')
#shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
#put legend to right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1,0.5))

plt.savefig('class_counts_by_pt')
fig.clear()

## plot the percentage
plt.clf()
fig = plt.figure()

df_class_percent_per_pt = pd.DataFrame(d_class_percent_per_pt)

ax = plt.subplot(111)
df_class_percent_per_pt.transpose().plot(kind='bar')
#shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
#put legend to right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1,0.5))

plt.savefig('class_percent_by_pt')
fig.clear()
## 


