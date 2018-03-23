
import os
import re
import time

import pandas as pd
import numpy as np
from datetime import timedelta

import factory_model as factory


file_expressions = [r'.{0,}.log',
                    r'.{0,}Flitz.{0,}.log',
                    r'.{0,}Kinect.{0,}.log',
                    ]


working_directory_path = os.path.abspath(os.path.dirname(__file__))
project_path = os.path.abspath(os.path.join(working_directory_path, '../../..'))

abs_path_logfiles = project_path + '/Logs/text_logs'
names_logfiles = []  # Name of the logfiles

df_list = []  # List with all dataframes; 1 dataframe per logfile
df_total = []  # All dataframes, concatanated to one single dataframe
df = []  # Resampled dataframe with feature-columns, concatanated to one single dataframe


def init(crash_window, heartrate_window):
    global df, df_total
    init_dataframes()

    if os.path.isfile(working_directory_path + '/df.csv'):
        print('Dataframe already cached. Used this file to improve performance')
        df = pd.read_csv(working_directory_path + '/df.csv', index_col=0)
        df_total = pd.concat(df_list, ignore_index=True)
    else:
        print('Dataframe not cached. Creating dataframe...')
        df_total = pd.concat(df_list, ignore_index=True)
        df = add_mean_hr_and_crashes_columns_resampled(crash_window, heartrate_window)

        # Save to .csv for caching
        df.to_csv('df.csv', index=True, header=True)
        print('Dataframe created')

    df.reset_index(inplace=True)
    df_total.reset_index(inplace=True)
    return df


def init_names_logfiles():
    global names_logfiles
    names_logfiles = [f for f in sorted(os.listdir(abs_path_logfiles)) if re.search(file_expressions[1], f)]


def init_dataframes():
    global df_list
    init_names_logfiles()
    logs = [abs_path_logfiles + "/" + s for s in names_logfiles]
    column_names = ['Time', 'Logtype', 'Gamemode', 'Points', 'Heartrate', 'physDifficulty', 'psyStress', 'psyDifficulty', 'obstacle']
    df_list = list(pd.read_csv(log, sep=';', skiprows=5, index_col=False, names=column_names) for log in logs)
    df_list = cut_frames(df_list)  # Cut frames to same length
    add_log_column(df_list)
    add_timedelta_column(df_list)


'''Cuts dataframes to the same length, namely to the shortest of the dataframes in the list
'''


def cut_frames(dataframe_list):
    cutted_df_list = [] 
    min_time = min(dataframe['Time'].max() for dataframe in dataframe_list)
    for dataframe in dataframe_list:
        cutted_df_list.append(dataframe[dataframe['Time'] < min_time])
    return cutted_df_list


'''For a lot of queries, it is useful to have the ['Time'] as a timedeltaIndex object
'''


def add_timedelta_column(dataframe_list):
    for idx, dataframe in enumerate(dataframe_list):
        new = dataframe['Time'].apply(lambda x: timedelta(seconds=x))
        dataframe_list[idx] = dataframe_list[idx].assign(timedelta=new)


''' Add user_id and round (1 or 2) as extra column
'''


def add_log_column(dataframe_list):
    for idx, dataframe in enumerate(dataframe_list):
        new = np.full((len(dataframe.index),1), int(np.floor(idx/2)))
        dataframe_list[idx] = dataframe_list[idx].assign(userID=new)


def add_mean_hr_and_crashes_columns_resampled(crash_window, heartrate_window):
    dataframe = df_total

    # Add timedelta=0.0 row s.t. resampling starts at 0 seconds
    []

    time1 = time.time()

    # Compute mean_hr over last 'heart_window' seconds
    df_with_hr = dataframe[dataframe['Heartrate'] != -1]

    df_with_hr['mean_hr'] = factory.get_mean_heartrate_column(df_with_hr, heartrate_window)
    mean_hr_resampled_column = factory.resample_dataframe(df_with_hr, 1)['mean_hr']

    time2 = time.time()
    print("Time to get mean_hr: " + str(time2 - time1))

    # Compute %crashes over last 'crash_window' seconds
    dataframe['%crashes'] = factory.get_crashes_column(dataframe, crash_window)
    df_resampled = factory.resample_dataframe(dataframe, 1)
    time3 = time.time()
    print("Time to get %crashes: " + str(time3 - time2))

    df_resampled['mean_hr'] = mean_hr_resampled_column
    return df_resampled
