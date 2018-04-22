
import os
import re

import pandas as pd
import pickle
from datetime import timedelta

import globals as gl
import factory


def setup():

    read_and_prepare_logs()

    # Store computed dataframe in pickle file for faster processing
    if gl.use_cache & os.path.isfile(gl.working_directory_path + '/Pickle/df_list.pickle'):
        print('Dataframe already cached. Used this file to improve performance')
        gl.obstacle_df_list = pickle.load(open(gl.working_directory_path + '/Pickle/obstacle_df.pickle', "rb"))
        gl.df_list = pickle.load(open(gl.working_directory_path + '/Pickle/df_list.pickle', "rb"))
    else:
        print('Dataframe not cached. Creating dataframe...')

        gl.obstacle_df_list = factory.get_obstacle_times_with_success()

        # Save to .pickle for caching
        pickle.dump(gl.obstacle_df_list, open(gl.working_directory_path + '/Pickle/obstacle_df.pickle', "wb"))
        pickle.dump(gl.df_list, open(gl.working_directory_path + '/Pickle/df_list.pickle', "wb"))

        print('Dataframe created')

    # factory.print_keynumbers_logfiles()


'''Reads the logfiles and parses them into Pandas dataframes. 
    Also adds additional log&timedelta column, cuts them to the same length and normalizes heartrate
'''


def read_and_prepare_logs():
    kinect_names = [f for f in sorted(os.listdir(gl.abs_path_logfiles)) if re.search(r'.{0,}Kinect.{0,}.log', f)]
    all_names = [f for f in sorted(os.listdir(gl.abs_path_logfiles)) if re.search(r'.log', f)]
    fbmc_names = [item for item in all_names if item not in kinect_names]

    gl.names_logfiles = fbmc_names

    logs = [gl.abs_path_logfiles + "/" + s for s in gl.names_logfiles]
    column_names = ['Time', 'Logtype', 'Gamemode', 'Points', 'Heartrate', 'physDifficulty', 'psyStress', 'psyDifficulty', 'obstacle']
    # This read_csv is used when using the original logfiles
    # gl.df_list = list(pd.read_csv(log, sep=';', skiprows=5, index_col=False, names=column_names) for log in logs)

    # This read_csv is used when using the refactored logs (Crash/Obstacle cleaned up)
    gl.df_list = list(pd.read_csv(log, sep=';', index_col=False, names=column_names) for log in logs)
    if gl.testing:
        # 2 dataframes with and 1 dataframe without heartrate
        gl.df_list = gl.df_list[19:21]
    # NOTE: Has only ever been called once to refactore logs
    # refactor_crashes()

    # cut_frames()  # Cut frames to same length
    remove_logs_without_heartrates()
    normalize_heartrate()
    add_log_column()
    add_timedelta_column()


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
                baseline = dataframe[dataframe['Time'] < 20]['Heartrate'].min()
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
        # TODO: Now each logfile has a new userID. Maybe I want the same for each user?
        # new = np.full((len(dataframe.index), 1), int(np.floor(idx/2)))
        gl.df_list[idx] = gl.df_list[idx].assign(logID=idx)


'''At the moment, there is always a EVENT_CRASH and a EVENT_OBSTACLE inc case of a crash, which makes it more difficult to analyze the data. 
    Thus, in case of a crash, I remove the EVENT_OBSTACLE and move its obstacle inforamtion to the EVENT_CRASH log
    
    Input: Original files
    Output: New logfiles, without headers or anything
    
    Done ONE TIME only and saved in new folder 'text_logs_refactored_crashes'. From now on, always those logs are used'''


def refactor_crashes():

    ''' If there was a crash, then there would be an 'EVENT_CRASH' in the preceding around 1 seconds of the event'''
    print('Refactoring crashes...')

    def get_next_obstacle_row(index, df):
        cnt = 1
        while True:
            logtype = df.loc[index + cnt]['Logtype']
            if logtype == 'EVENT_OBSTACLE':
                return index+cnt, df.loc[index + cnt]
            cnt += 1

    for df_idx, dataframe in enumerate(gl.df_list):
        new_df = pd.DataFrame()
        count = 0
        obst_indices = []
        for idx, row in dataframe.iterrows():
            if not (idx in obst_indices):
                if row['Logtype'] == 'EVENT_CRASH':
                    if count == dataframe.index[-1]:
                        row['obstacle'] = ''
                    else:
                        obst_idx, obstacle_row = get_next_obstacle_row(idx, dataframe)
                        obst_indices.append(obst_idx)
                        row['obstacle'] = obstacle_row['obstacle']
                        # dataframe.drop(obst_idx, inplace=True)
                new_df = new_df.append(row)

                count += 1
        new_df.reset_index(inplace=True, drop=True)
        column_names = ['Time', 'Logtype', 'Gamemode', 'Points', 'Heartrate', 'physDifficulty', 'psyStress',
                        'psyDifficulty', 'obstacle']

        new_df = new_df.reindex(column_names, axis=1)
        gl.df_list[df_idx] = new_df
        print('next')
        new_df.to_csv(gl.abs_path_logfiles + "/" + gl.names_logfiles[df_idx], header=False, index=False, sep=';')


'''Remove all logfiles that do not have any heartrate data )since we then can't calculate our features'''


def remove_logs_without_heartrates():
    gl.df_list = [df for df in gl.df_list if not (df['Heartrate'] == -1).all()]
