
import os
import re

import pandas as pd
import numpy as np
from datetime import timedelta

import features_factory as f_factory
import plots

file_expressions = [r'.{0,}.log',
                    r'.{0,}Flitz.{0,}.log',
                    r'.{0,}Kinect.{0,}.log',
                    ]


# TODO: Look that all methods have a return value instead of them modifying arguments
# TODO: Add print logs to log what the program is currently doing

working_directory_path = os.path.abspath(os.path.dirname(__file__))
project_path = os.path.abspath(os.path.join(working_directory_path, '../../..'))

abs_path_logfiles = project_path + '/Logs/text_logs'
names_logfiles = []  # Name of the logfiles

df_list = []  # List with all dataframes; 1 dataframe per logfile

df_total = []  # All dataframes, concatanated to one single dataframe (without features as columns)
df = []  # Resampled dataframe with feature-columns, concatanated to one single dataframe

cw = 0  # Stores the size of the crash_window
hw = 0  # stores the size of the heart_rate window

testing = True  # If Testing=True, only  a small sample of dataframes is used  to accelerate everything


def init(cache, crash_window, heartrate_window):
    global cw, hw, df, df_total
    cw = crash_window
    hw = heartrate_window

    init_dataframes()

    # Store computed dataframe in pickle file for faster processing
    if cache & os.path.isfile(working_directory_path + '/Pickle/df.pickle'):
        print('Dataframe already cached. Used this file to improve performance')
        df = pd.read_pickle(working_directory_path + '/Pickle/df.pickle')
        df_total = pd.concat(df_list, ignore_index=True)
    else:
        print('Dataframe not cached. Creating dataframe...')
        df_total = pd.concat(df_list, ignore_index=True)
        df = df_total
        f_factory.add_mean_hr_to_df(heartrate_window)
        f_factory.add_crashes_to_df(crash_window)
        # TODO: window
        f_factory.add_max_over_min_hr_to_df(30)
        # Save to .pickle for caching
        df.to_pickle(working_directory_path + '/Pickle/df.pickle')
        print('Dataframe created')


''' Inits the dataframes not from the logfiles, but with synthesized data
'''
'''
def init_with_testdata(crash_window, heartrate_window):
    global cw, hw, df, df_total
    cw = crash_window
    hw = heartrate_window
    num_dataframes = 10 # How many dataframes should be created?
    mean_hr = 120 # Mean of normal distribution of heartrate
    for i in  range(1, num_dataframes):
        times = [1,2,3,4,5,6]
        logtypes =  ['EVENT_OBSTACLE', 'EVENT_CRASH',  'EVENT_CRASH', 'EVENT_OBSTACLE', 'EVENT_CRASH', 'EVENT_OBSTACLE']
        heartrates =  [378, 155, 77, 973, 973, 973]
        timedeltas = [timedelta(seconds =1), timedelta(seconds =2), timedelta(seconds =3), timedelta(seconds =4), timedelta(seconds =5), timedelta(seconds =6)]
        dataframe = pd.DataFrame(data = {'Time': times, 'Logtype' : logtypes, 'Heartrate' : heartrates, 'timedelta': timedeltas})
        df_list.append(dataframe)

    df_total = pd.concat(df_list, ignore_index=True)
    df = df_total

    f_factory.add_mean_hr_to_df(heartrate_window)
    f_factory.add_crashes_to_df(crash_window)
    # TODO: window
    f_factory.add_max_over_min_hr_to_df(30)
    print(df)

'''
'''Reads the logfiles and parses them into Pandas dataframes. 
    Also adds additional log&timedelta column and cuts them to the same length
'''


def init_dataframes():
    global df_list, names_logfiles
    names_logfiles = [f for f in sorted(os.listdir(abs_path_logfiles)) if re.search(file_expressions[1], f)]
    logs = [abs_path_logfiles + "/" + s for s in names_logfiles]
    column_names = ['Time', 'Logtype', 'Gamemode', 'Points', 'Heartrate', 'physDifficulty', 'psyStress', 'psyDifficulty', 'obstacle']
    df_list = list(pd.read_csv(log, sep=';', skiprows=5, index_col=False, names=column_names) for log in logs)
    if testing:
        df_list = df_list[5:6]
    df_list = cut_frames(df_list)  # Cut frames to same length
    df_list = normalize_heartrate(df_list)  # Normalize heartrate

    add_log_column(df_list)
    add_timedelta_column(df_list)
    plots.plot_hr_of_dataframes()



'''Cuts dataframes to the same length, namely to the shortest of the dataframes in the list
'''


def cut_frames(dataframe_list):
    cutted_df_list = [] 
    min_time = min(dataframe['Time'].max() for dataframe in dataframe_list)
    for dataframe in dataframe_list:
        cutted_df_list.append(dataframe[dataframe['Time'] < min_time])
    return cutted_df_list


'''Normalize heartrate of each dataframe/user by subtracting baseline (first 10 seconds of log)
'''

def normalize_heartrate(dataframe_list):
    normalized_df_list = []
    for dataframe in dataframe_list:
        mean = dataframe['Heartrate'].mean()
        dataframe['Heartrate'] = dataframe['Heartrate'] / mean
        normalized_df_list.append(dataframe)
    return normalized_df_list


'''For a lot of queries, it is useful to have the ['Time'] as a timedeltaIndex object
'''


def add_timedelta_column(dataframe_list):
    for idx, dataframe in enumerate(dataframe_list):
        new = dataframe['Time'].apply(lambda x: timedelta(seconds=x))
        dataframe_list[idx] = dataframe_list[idx].assign(timedelta=new)


''' Add log_number column
'''


def add_log_column(dataframe_list):
    for idx, dataframe in enumerate(dataframe_list):
        new = np.full((len(dataframe.index), 1), int(np.floor(idx/2)))
        dataframe_list[idx] = dataframe_list[idx].assign(userID=new)


