#globals_model.py

import os
import re

import pandas as pd
import numpy as np
import datetime

file_expressions = [r'.{0,}.log',
                    r'.{0,}Flitz.{0,}.log',
                    r'.{0,}Kinect.{0,}.log',
                    ]

                    
local_base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
svn_base_path = local_base_path + '/Code/plunder planet/ML Model'
logs_path = local_base_path + '/Logs/text_logs'
rel_files= []
dataframes = []
main_df = []


def init():
    global main_df
    init_rel_files()
    init_dataframes()
    main_df = pd.concat(dataframes, ignore_index=True)

def init_rel_files():
    global rel_files
    rel_files = [f for f in sorted(os.listdir(logs_path)) if re.search(file_expressions[1], f)]


def init_dataframes():
    global dataframes
    logs = [logs_path + "/" +  s for s in rel_files]
    column_names = ['Time','Logtype','Gamemode','Points','Heartrate','physDifficulty','psyStress','psyDifficulty','obstacle']
    dataframes = list(pd.read_csv(log, sep=';', skiprows=5, index_col=False, names=column_names) for log in logs)