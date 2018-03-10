# globals.py

import os
import re

import pandas as pd
import numpy as np

file_expressions = [r'.{0,}.log',
                    r'.{0,}Flitz.{0,}.log',
                    r'.{0,}Kinect.{0,}.log',
                    ]
local_base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
svn_base_path = local_base_path + '/Code/plunder planet'
logs_path = local_base_path + '/Logs/text_logs'
rel_files= []
dataframes = []
conc_dataframes = []


def init():
    global conc_dataframes
    init_rel_files()
    init_dataframes()
    add_user_and_round()
    conc_dataframes = pd.concat(dataframes, ignore_index=True)


''' I differentiate between log data that:
        - is from FBMC
        - is from Kinect 

        file_expressions: 0 -> All files, 
                        1-> FBMC
                        4-> Kinect
'''


def init_rel_files():
    global rel_files
    rel_files = [f for f in sorted(os.listdir(logs_path)) if re.search(file_expressions[1], f)]


def init_dataframes():
    global dataframes
    logs = [logs_path + "/" +  s for s in rel_files]
    column_names = ['Time','Logtype','Gamemode','Points','Heartrate','physDifficulty','psyStress','psyDifficulty','obstacle']
    dataframes = list(pd.read_csv(log, sep=';', skiprows=5, index_col=False, names=column_names) for log in logs)


''' Add user_id and round (1 or 2) as extra column
'''


def add_user_and_round():
        for idx, df in enumerate(dataframes):
            df.insert(9, 'userID', np.full( (len(df.index),1), int(np.floor(idx/2))))
            # m = re.search(r'(\d)_[0,1]',rel_files[idx], re.IGNORECASE)
            match = re.search('(\d)_(\d+)', rel_files[idx])  # add if round 0 or 1
            if match:
                df.insert(10, 'round', np.full( (len(df.index),1), match.group(1)))
            else:
                df.insert(10, 'round', np.full( (len(df.index),1), 0))
            dataframes[idx] = df