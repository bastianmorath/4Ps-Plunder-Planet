'''Parsing the LogFiles to Python with pandas
'''


import sys
import os
import re
import glob

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pandas import DataFrame, read_csv

dir_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'Python/Logs/text_logs'))
file_expressions = [r'.{0,}.log',
                    r'.{0,}Flitz.{0,}.log',
                    r'(EK|FM|MH|MK|NK|PG|TW).{0,}Flitz.{0,}.log', 
                    r'(EK|FM|MH|MK|NK|PG|TW).{0,}Flitz.{0,}.log', # wrong yet
                    r'.{0,}Kinect.{0,}.log',
                    r'(EK|FM|MH|MK|NK|PG|TW).{0,}Kinect.{0,}.log',
                    r'(EK|FM|MH|MK|NK|PG|TW).{0,}Kinect.{0,}.log' # wrong yet
                    ]

'''
    I differentiate between log-data that:
     - is from FBMC, Kinect or both (F, K, FK)
     - has or has no Heartrate data (H, nH, all)

    file_expressions: 0 -> All files, 
                   1-> F/all, 2-> F/H, 3-> F/nH,  
                   4-> K/all, 5-> F/H, 6-> F/nH
''' 

rel_files = [f for f in os.listdir(dir_path) if re.search(file_expressions[0], f)]
print(rel_files)

abs_files = [dir_path + "/" +  s for s in rel_files]

column_names = ['Time','Logtype','Gamemode','Points','Heartrate','physDifficulty','psyStress','psyDifficulty','obstacle']
df_from_each_file = (pd.read_csv(f, sep=';', skiprows=5, index_col=False, names=column_names) for f in abs_files)
concatenated_df = pd.concat(df_from_each_file, ignore_index=True)

print(concatenated_df)

filtered = concatenated_df[concatenated_df['Heartrate']>50][concatenated_df['Time']<30] #to plot nicer results

mean_heartreate_per_time = filtered.groupby('Time')['Heartrate'].mean()

plt.plot(mean_heartreate_per_time)
plt.show()
