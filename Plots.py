# Deprecated -> old version
# 
import sys
import os
import re
import glob

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pandas import DataFrame, read_csv

from Factory import Helpers

'''Validating data and create plots'''

dir_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'Python/Logs/text_logs'))

'''
    I differentiate between log data that:
     - is from FBMC
     - is from Kinect 

    file_expressions: 0 -> All files, 
                      1-> FBMC
                      4-> Kinect
''' 
file_expressions = [r'.{0,}.log',
                    r'.{0,}Flitz.{0,}.log',
                    r'.{0,}Kinect.{0,}.log',
                    ]


rel_files = [f for f in sorted(os.listdir(dir_path)) if re.search(file_expressions[1], f)]
logs = [dir_path + "/" +  s for s in rel_files]
column_names = ['Time','Logtype','Gamemode','Points','Heartrate','physDifficulty','psyStress','psyDifficulty','obstacle']
dataframes = list(pd.read_csv(log, sep=';', skiprows=5, index_col=False, names=column_names) for log in logs)

''' Add user_id and round (1 or 2) as extra column
'''
for idx, df in enumerate(dataframes):
    df.insert(9, 'userID', np.full( (len(df.index),1), int(np.floor(idx/2))))
    m = re.search(r'(\d)_[0,1]', rel_files[idx], re.IGNORECASE)
    match = re.search('(\d)_(\d+)', rel_files[idx]) # add if round 0 or 1
    if match:
        df.insert(10, 'round', np.full( (len(df.index),1), match.group(1)))
    else:
        df.insert(10, 'round', np.full( (len(df.index),1), 0))



conc_dataframes = pd.concat(dataframes, ignore_index=True)

''' Since the MOVEMENTTUTORIAL at the beginning is different for each player/logfile, 
    we need to align them, i.e. remove tutorial-log entries and then set timer to 0
'''
for idx, df in enumerate(dataframes):
    if 'MOVEMENTTUTORIAL' in df['Gamemode'].values:

        tutorial_mask = df['Gamemode']=='MOVEMENTTUTORIAL'
        tutorial_entries = df[tutorial_mask]
        tutorial_endtime = tutorial_entries['Time'].max()

        df['Time'] = df['Time'].apply(lambda x: x - tutorial_endtime)
        dataframes[idx] = df[~tutorial_mask].reset_index(drop=True)

''' Since the SHIELDTUTORIAL at the beginning is different for each player/logfile, 
    we need to remove them and adjust timer
'''
for idx, df in enumerate(dataframes):
    if 'SHIELDTUTORIAL' in df['Gamemode'].values: 
        tutorial_mask = df['Gamemode']=='SHIELDTUTORIAL'
        tutorial_entries = df[tutorial_mask]
        start_time = tutorial_entries['Time'].min()
        end_time = tutorial_entries['Time'].max()
        tutorial_endtime =  end_time - start_time

        df['Time'] = df['Time'][df['Time']>start_time].apply(lambda x: x - tutorial_endtime)
        dataframes[idx] = df[~tutorial_mask].reset_index(drop=True)



'''Plot: Playing time per user 
'''


plt.ylabel('Playing time [s]')
plt.title('Playing time per user per round')
colors = Helpers.getColor(conc_dataframes)

time_df = conc_dataframes.groupby(['userID', 'round'])['Time'].max()

time_df.plot.bar(color=colors)
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
