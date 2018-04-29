"""This file does some refactoring to the logfiles, mostly at the very beginning,
saving the changes as new logfiles, such that we don't have to apply it again"""


import pandas as pd
from datetime import timedelta

import globals as gl

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
        print('Normalize hr...')
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


def add_log_and_user_column():
    """
     Add log_number and user_id

    """

    names = [gl.names_logfiles[i][0:2] for i in range(0, len(gl.df_list))] # E.g. AK, LK, MR

    last_name = names[0]
    user_id = 0

    for idx, dataframe in enumerate(gl.df_list):
        if not names[idx] == last_name:
            user_id += 1

        log_id = gl.names_logfiles[idx][-5]
        df = dataframe.assign(userID=user_id)
        df = df.assign(logID=log_id)
        gl.df_list[idx] = df
        last_name = names[idx]


'''At the moment, there is always a EVENT_CRASH and a EVENT_OBSTACLE inc case of a crash, which makes it more difficult 
    to analyze the data. 
    Thus, in case of a crash, I remove the EVENT_OBSTACLE and move its obstacle inforamtion to the EVENT_CRASH log
    Additionaly, I add a column with the userID and whether it's the first or second logfile of the user
    I also add a timedelta-column

    Input: Original files
    Output: New logfiles, without headers or anything

    Done ONE TIME only and saved in new folder 'text_logs_refactored_crashes'. From now on, always those logs are used'''


def refactor_crashes():
    # If there was a crash, then there would be an 'EVENT_CRASH' in the preceding around 1 seconds of the event
    add_log_and_user_column()
    add_timedelta_column()

    print('Refactoring crashes...')

    def get_next_obstacle_row(index, df):
        cnt = 1
        while True:
            logtype = df.loc[index + cnt]['Logtype']
            if logtype == 'EVENT_OBSTACLE':
                return index + cnt, df.loc[index + cnt]
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


'''Remove all logfiles that do not have any heartrate data or points (since we then can't calculate our features)'''


def remove_logs_without_heartrates_or_points():
    gl.df_list = [df for df in gl.df_list if not (df['Heartrate'] == -1).all()]
    gl.df_list = [df for df in gl.df_list if not (df['Points'] == 0).all()]