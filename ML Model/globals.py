"""This file contains variables that are used across different modules"""

import os

cw = 0  # Stores the size of the crash_window
hw = 0  # stores the size of the heart_rate window

testing = False  # If Testing==True, only  a small sample of dataframes is used  to accelerate everything
use_cache = True  # If use_cache==True, use cached data (accelerates testing on same data)

working_directory_path = os.path.abspath(os.path.dirname(__file__))
project_path = os.path.abspath(os.path.join(working_directory_path, '../../..'))

abs_path_logfiles = project_path + '/Logs/text_logs'
names_logfiles = []  # Name of the logfiles

df_list = []  # List with all dataframes; 1 dataframe per logfile

df_without_features = []  # All dataframes, concatanated to one single dataframe (without features as columns)
df = []  # Resampled dataframe with feature-columns, concatanated to one single dataframe

obstacle_df = []  # obstacle_df contains timestamp of each obstacle and whether or not the user crashed
