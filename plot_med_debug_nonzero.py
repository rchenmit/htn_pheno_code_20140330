## go thru each med, and only plot the patients where there is >0 occurrences of the med:

## boxplot ofo HTN Meds COUNTS (side-by-side IN/OUT) for each Med (32 total bars); ONLY PLOT PATIENTS WITH NONZERO OCCURRENCES!
#feature_matrix_counts_IN_CONTROL
#feature_matrix_counts_OUT_CONTROL
num_med = len(l_all_drug_classes)
data = []
num_subjects_per_med = []
for i in range(num_med):
    nonzero_IN = feature_matrix_counts_IN_CONTROL[:,i][feature_matrix_counts_IN_CONTROL[:,i] > 0]
    nonzero_OUT = feature_matrix_counts_OUT_CONTROL[:,i][feature_matrix_counts_OUT_CONTROL[:,i] > 0]
    if len(nonzero_IN)>0:
        data.append(nonzero_IN)
        num_subjects_per_med.append(len(nonzero_IN))
    else:
        data.append([0])
        num_subjects_per_med.append(0)
    if len(nonzero_OUT)>0:
        data.append(nonzero_OUT)
        num_subjects_per_med.append(len(nonzero_OUT))
    else:
        data.append([0])
        num_subjects_per_med.append(0)
labels = []
for i in range(num_med):
    labels.append(l_all_drug_classes[i] + ': IN')
    labels.append(l_all_drug_classes[i] + ': OUT')
    
numBoxes = num_med

fig, ax1 = plt.subplots(figsize=(numBoxes*2,6))
fig.canvas.set_window_title('A Boxplot Example')
plt.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)
bp = plt.boxplot(data, notch=0, sym='+', vert=1, whis=1.5)
plt.setp(bp['boxes'], color='black')
plt.setp(bp['whiskers'], color='black')
plt.setp(bp['fliers'], color='red', marker='+')
# Add a horizontal grid to the plot, but make it very light in color
# so we can use it for reading data values but not be distracting
ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
              alpha=0.5)

# Hide these grid behind plot objects
ax1.set_axisbelow(True)
ax1.set_title('Comparison of Medication Occurrences in Patients IN Control vs. OUT of Control')
ax1.set_xlabel('Medication / Control Status')
ax1.set_ylabel('# Occurrences')
# Now fill the boxes with desired colors
boxColors = ['darkkhaki','royalblue']
numBoxes = numBoxes*2
medians = range(numBoxes)
for i in range(numBoxes):
  box = bp['boxes'][i]
  boxX = []
  boxY = []
  for j in range(5):
      boxX.append(box.get_xdata()[j])
      boxY.append(box.get_ydata()[j])
  boxCoords = zip(boxX,boxY)
  # Alternate between Dark Khaki and Royal Blue
  k = i % 2
  boxPolygon = Polygon(boxCoords, facecolor=boxColors[k])
  ax1.add_patch(boxPolygon)
  # Now draw the median lines back over what we just filled in
  med = bp['medians'][i]
  medianX = []
  medianY = []
  for j in range(2):
      medianX.append(med.get_xdata()[j])
      medianY.append(med.get_ydata()[j])
      plt.plot(medianX, medianY, 'k')
      medians[i] = medianY[0]
  # Finally, overplot the sample averages, with horizontal alignment
  # in the center of each box
  plt.plot([np.average(med.get_xdata())], [np.average(data[i])],
           color='w', marker='*', markeredgecolor='k')
# Set the axes ranges and axes labels
ax1.set_xlim(0.5, numBoxes+0.5)
top = 40
bottom = -5
ax1.set_ylim(bottom, top)
xtickNames = plt.setp(ax1, xticklabels=labels)
plt.setp(xtickNames, rotation=45, fontsize=8)
# Due to the Y-axis scale being different across samples, it can be
# hard to compare differences in medians across the samples. Add upper
# X-axis tick labels with the sample medians to aid in comparison
# (just use two decimal places of precision)
# ALSO: ADD the NUMBER of SAMPLES used in each one (because we took out all subjects where the count for the med was 0 ########## THIS IS NEW ADD TO THE CODE
pos = np.arange(numBoxes)+1
upperLabels = [str(np.round(s, 2)) for s in medians]
upperLabels_numSubjects = [str(np.round(s, 2)) for s in num_subjects_per_med]
weights = ['bold', 'semibold']
for tick,label in zip(range(numBoxes),ax1.get_xticklabels()):
   k = tick % 2
   ax1.text(pos[tick], top-(top*0.05), upperLabels[tick],
        horizontalalignment='center', size='x-small', weight=weights[k],
        color=boxColors[k])
   ax1.text(pos[tick], top-(2*top*0.05), upperLabels_numSubjects[tick],
        horizontalalignment='center', size='x-small', weight=weights[k],
        color=boxColors[k]) #right under the label for the median
# Finally, add a basic legend
plt.figtext(0.80, 0.08,  str(len(feature_matrix_counts_IN_CONTROL)) + ' IN Control' ,
           backgroundcolor=boxColors[0], color='black', weight='roman',
           size='x-small')
plt.figtext(0.80, 0.045, str(len(feature_matrix_counts_OUT_CONTROL)) + ' OUT Control' ,
backgroundcolor=boxColors[1],
           color='white', weight='roman', size='x-small')
plt.figtext(0.80, 0.015, '*', color='white', backgroundcolor='silver',
           weight='roman', size='medium')
plt.figtext(0.815, 0.013, ' Average Value', color='black', weight='roman',
           size='x-small')

##show the plot
plt.show()
