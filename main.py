# main.py

import matplotlib.pyplot as plt
import numpy as np
import settings
from factory import Helpers, Filter

'''Validating data and create plots'''

settings.init()
Filter.removeMovementTutorials()
Filter.removeShieldTutorials()

dataframes = settings.dataframes
conc_dataframes = settings.conc_dataframes


'''Plot: Playing time per user 
'''
plt.ylabel('Playing time [s]')
plt.title('Playing time per user per round')
colors = Helpers.getColors()

time_df = conc_dataframes.groupby(['userID', 'round'])['Time'].max()

time_df.plot.bar(color=Helpers.getColors())
plt.tight_layout()
plt.savefig(settings.svn_base_path + '/Plots/Playing_time_per_user.pdf')


'''Plot: Heartrate 
'''
plt.figure()
plt.ylabel('Heartrate [bpm]')
plt.xlabel('Playing time [s]')
plt.title('Heartrate of all users')

for idx, df in enumerate(dataframes):
    if not (df['Heartrate']==-1).all():  #Filter out dataframes without HR measurements
        df['Heartrate'].plot( title='Heartrate')

plt.savefig(settings.svn_base_path + '/Plots/Heartrate_series.pdf')


'''Plot: Mean and std bpm per user in a box-chart
'''
df2 = conc_dataframes.pivot(columns=conc_dataframes.columns[1], index=conc_dataframes.index)
df2.columns = df2.columns.droplevel()
conc_dataframes[['Heartrate','userID']].boxplot(by='userID', grid=False, sym='r+')
plt.ylabel('Heartrate [bpm]')
plt.title('')
plt.savefig(settings.svn_base_path + '/Plots/Mean_heartrate.pdf')

''' Plot Heartrate change
    TODO: Average the two rounds per user!
'''
bpm_changes_max = []
bpm_changes_rel = []
X = []
for idx, df in enumerate(dataframes):
    if not (df['Heartrate'] == -1).all():
        X.append(idx)
        percentage_change = np.diff(df['Heartrate']) / df['Heartrate'][:-1] * 100.
        bpm_changes_max.append(percentage_change.max())
        bpm_changes_rel.append(percentage_change)
plt.figure()
plt.title('Heartrate change')
plt.ylabel('#Times HR changed by x%')
plt.xlabel('Change in Heartrate [%]')
plt.hist(bpm_changes_rel[0]) # Only plot for 1 user
plt.savefig(settings.svn_base_path + '/Plots/heartrate_change_percentage.pdf')

plt.figure()
plt.title('Maximal heartrate change')
plt.ylabel('Max heartrate change [%]')
plt.xlabel('User id')
plt.bar([x for x in X], bpm_changes_max, color='r', width=0.25)
plt.savefig(settings.svn_base_path + '/Plots/heartrate_change_abs.pdf')

plt.show()
