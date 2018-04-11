
import os
import re

import pandas as pd
import numpy as np
from datetime import timedelta

import globals as gl
import factory
import features_factory as f_factory


def setup():

    read_and_prepare_logs()

    # Store computed dataframe in pickle file for faster processing
    if gl.use_cache & os.path.isfile(gl.working_directory_path + '/Pickle/df.pickle'):
        print('Dataframe already cached. Used this file to improve performance')
        gl.df = pd.read_pickle(gl.working_directory_path + '/Pickle/df.pickle')
        gl.obstacle_df = pd.read_pickle(gl.working_directory_path + '/Pickle//obstacle_df.pickle')

    else:
        print('Dataframe not cached. Creating dataframe...')
        gl.df = f_factory.get_df_with_feature_columns()
        gl.obstacle_df = factory.get_obstacle_times_with_success()

        # Save to .pickle for caching
        gl.obstacle_df.to_pickle(gl.working_directory_path + '/Pickle/obstacle_df.pickle')
        gl.df.to_pickle(gl.working_directory_path + '/Pickle/df.pickle')
        print('Dataframe created')


'''Reads the logfiles and parses them into Pandas dataframes. 
    Also adds additional log&timedelta column, cuts them to the same length and normalizes heartrate
'''


def read_and_prepare_logs():
    gl.names_logfiles = [f for f in sorted(os.listdir(gl.abs_path_logfiles)) if re.search(r'.{0,}Flitz.{0,}.log', f)]
    logs = [gl.abs_path_logfiles + "/" + s for s in gl.names_logfiles]
    column_names = ['Time', 'Logtype', 'Gamemode', 'Points', 'Heartrate', 'physDifficulty', 'psyStress', 'psyDifficulty', 'obstacle']
    gl.df_list = list(pd.read_csv(log, sep=';', skiprows=5, index_col=False, names=column_names) for log in logs)
    if gl.testing:
        # 2 dataframes with/ 1 dataframe without heartrate
        gl.df_list = gl.df_list[19:22]
    cut_frames()  # Cut frames to same length
    normalize_heartrate()
    add_log_column()
    add_timedelta_column()
    gl.df_without_features = pd.concat(gl.df_list, ignore_index=True)


"""The following methods add additional columns to all dataframes in the gl.df_list"""


'''Cuts dataframes to the same length, namely to the shortest of the dataframes in the list'''


def cut_frames():
    cutted_df_list = [] 
    min_time = min(dataframe['Time'].max() for dataframe in gl.df_list)
    for dataframe in gl.df_list:
        cutted_df_list.append(dataframe[dataframe['Time'] < min_time])
    gl.df_list = cutted_df_list


'''Normalize heartrate of each dataframe/user by dividing by mean of first 60 seconds'''


def normalize_heartrate():
    if gl.normalize_heartrate:
        normalized_df_list = []
        for dataframe in gl.df_list:
            if not (dataframe['Heartrate'] == -1).all():
                baseline = dataframe[dataframe['Time'] < 60]['Heartrate'].mean()
                # dataframe['Heartrate'] = dataframe['Heartrate'] - baseline + 123.93  # add mean over all heartrates
                dataframe['Heartrate'] = dataframe['Heartrate'] / baseline
                normalized_df_list.append(dataframe)
            else:
                normalized_df_list.append(dataframe)

        gl.df_list = normalized_df_list


'''For a lot of queries, it is useful to have the ['Time'] as a timedeltaIndex object'''


def add_timedelta_column():
    for idx, dataframe in enumerate(gl.df_list):
        new = dataframe['Time'].apply(lambda x: timedelta(seconds=x))
        gl.df_list[idx] = gl.df_list[idx].assign(timedelta=new)


'''Add log_number column'''


def add_log_column():

    for idx, dataframe in enumerate(gl.df_list):
        new = np.full((len(dataframe.index), 1), int(np.floor(idx/2)))
        gl.df_list[idx] = gl.df_list[idx].assign(userID=new)


