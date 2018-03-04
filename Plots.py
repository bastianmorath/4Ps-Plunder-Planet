import sys
import os
import re
import glob

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pandas import DataFrame, read_csv

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



rel_files = [f for f in os.listdir(dir_path) if re.search(file_expressions[1], f)]
logs = [dir_path + "/" +  s for s in rel_files]
column_names = ['Time','Logtype','Gamemode','Points','Heartrate','physDifficulty','psyStress','psyDifficulty','obstacle']
dataframes = list(pd.read_csv(log, sep=';', skiprows=5, index_col=False, names=column_names) for log in logs)

''' Add user_id as column for entire dataframe
'''
for idx, df in enumerate(dataframes):
    df.insert(9, 'userID', np.full( (len(df.index),1), idx))


conc_dataframes = pd.concat(dataframes, ignore_index=True)

''' Since the MOVEMENTTUTORIAL at the beginning is different for each player/logfile, 
    we need to align them, i.e. remove tutorial-log entries and then set timer to 0
'''
avg_time = 0

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
avg_time = 0 
for idx, df in enumerate(dataframes):
    if 'SHIELDTUTORIAL' in df['Gamemode'].values: 
        tutorial_mask = df['Gamemode']=='SHIELDTUTORIAL'
        tutorial_entries = df[tutorial_mask]
        start_time = tutorial_entries['Time'].min()
        end_time = tutorial_entries['Time'].max()
        tutorial_endtime =  end_time - start_time

        df['Time'] = df['Time'][df['Time']>start_time].apply(lambda x: x - tutorial_endtime)
        dataframes[idx] = df[~tutorial_mask].reset_index(drop=True)


'''Plot Playing time per user 
'''
time_df = conc_dataframes.groupby(['userID'])['Time'].max()
time_df.plot.bar()

'''Plot Heartrate 
'''

for idx, df in enumerate(dataframes):
    if not (df['Heartrate']==-1).all(): # Filter out dataframes without HR measurements
        df['Heartrate'].plot()





plt.show()