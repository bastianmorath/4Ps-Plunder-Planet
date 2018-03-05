# main.py

import glob
import re

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pandas import DataFrame, read_csv

import settings
from factory import Helpers, Filter

'''Validating data and create plots'''

settings.init()
Filter.removeMovementTutorials()
Filter.removeShieldTutorials()

dataframes = settings.dataframes
conc_dataframes = settings.conc_dataframes


'''Plot: Playing time per usser 
'''
plt.ylabel('Playing time [s]')
plt.title('Playing time per user per round')
colors = Helpers.getColors()

time_df = conc_dataframes.groupby(['userID', 'round'])['Time'].max()

time_df.plot.bar(color=Helpers.getColors())
plt.tight_layout()
plt.savefig('Playing_time_per_user.pdf')


'''Plot: Heartrate 
'''
plt.figure()
plt.ylabel('Heartrate [bpm]')
plt.xlabel('Playing time [s]')

plt.title('Heartrate of all users')

for idx, df in enumerate(dataframes):
    if not (df['Heartrate']==-1).all():# Filter out dataframes without HR measurements
        df['Heartrate'].plot( title='Heartrate')

plt.savefig('Heartrate_series.pdf')

'''Plot: Heartrate correlated with Difficulty-Level
'''
plt.figure()
plt.ylabel('Heartrate [bpm]')
plt.xlabel('Playing time [s]')
plt.title('Heartrate correlated with Difficulty-Level')


df = next( (x for x in dataframes if not (x['Heartrate']==-1).all())) #get first dataframe that has HB measurements


plt.savefig('Heartrate_With_Difficulty.pdf')


'''Plot: Mean and std bpm per user in a box-chart
'''
df2 = conc_dataframes.pivot(columns=conc_dataframes.columns[1], index=conc_dataframes.index)
df2.columns = df2.columns.droplevel()
conc_dataframes[['Heartrate','userID']].boxplot(by='userID', grid=False)
plt.ylabel('Heartrate [bpm]')
plt.title('')
plt.savefig('Mean_heartrate.pdf')


plt.show()
