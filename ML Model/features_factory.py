import pandas as pd
import os

from scipy import stats
import numpy as np
import itertools

import factory as factory
import globals as gl

'''Current features:
    1. Mean_hr over last x seconds
    2. % of crashes in the last x seconds
    3. Max over min heartrate in last x seconds
'''


''' Returns a matrix containing the features, and the labels
    There is one feature-row for each obstacle
 '''


def get_feature_matrix_and_label():

    matrix = pd.DataFrame()

    if gl.use_cache & (not gl.test_data) & os.path.isfile(gl.working_directory_path + '/Pickle/feature_matrix.pickle'):
        matrix = pd.read_pickle(gl.working_directory_path + '/Pickle/feature_matrix.pickle')
    else:
        matrix['mean_hr'] = get_mean_hr_feature()
        matrix['%crashes'] = get_percentage_crashes_feature()
        matrix['max_over_min'] = get_max_over_min_feature()
        matrix['last_obstacle_crash'] = get_last_obstacle_crash_feature()
        matrix.to_pickle(gl.working_directory_path + '/Pickle/feature_matrix.pickle')
    labels = []
    for df in gl.obstacle_df_list:
        # remove ~ first heartrate_window rows (they have < hw seconds to compute features, and are thus not accurate)
        labels.append(df[df['Time'] > max(gl.cw, gl.hw)]['crash'].copy())
    labels = list(itertools.chain.from_iterable(labels))

    # Boxcox transformation
    if gl.use_boxcox:
        # TODO: Values cant be <=0 -> Shift by epsilon (boxcox doesn't include shift parameter)
        matrix['mean_hr'] = stats.boxcox(matrix['mean_hr'])[0]
        matrix['%crashes'] = stats.boxcox(matrix['%crashes']+0.01)[0] # Add shift parameter
        matrix['max_over_min'] = stats.boxcox(matrix['max_over_min'])[0]

    return matrix.as_matrix(), labels


"""The following methods append a column to the feature matrix (after resampling it)"""


def get_mean_hr_feature():
    mean_hr_list = []  # list that contains a list of mean_hrs for each logfile/df
    for list_idx, df in enumerate(gl.df_list):
        df['mean_hr'] = get_mean_heartrate_column(df)
        df = df[df['Time'] > max(gl.cw, gl.hw)]  # remove first window-seconds bc. not accurate data
        mean_hr_df = []
        for _, row in gl.obstacle_df_list[list_idx].iterrows():
            corresp_row = df[df['Time'] <= row['Time']].iloc[-1]
            mean_hr_df.append(corresp_row['mean_hr'])
        mean_hr_list.append(mean_hr_df)
    return pd.DataFrame(list(itertools.chain.from_iterable(mean_hr_list)), columns=['mean_hr'])


# TODO: Normalize crashes depending on size/assembly of the obstacle


def get_percentage_crashes_feature():
    crashes_list = []  # list that contains a list of %crashes for each logfile/df
    for list_idx, df in enumerate(gl.df_list):
        df['%crashes'] = get_percentage_crashes_column(df)
        df = df[df['Time'] > max(gl.cw, gl.hw)]  # remove first window-seconds bc. not accurate data

        df_obstacles = df[(df['Logtype'] == 'EVENT_CRASH') | (df['Logtype'] == 'EVENT_OBSTACLE')]
        crashes_list.append(df_obstacles['%crashes'])
    return pd.DataFrame(list(itertools.chain.from_iterable(crashes_list)), columns=['%crashes'])


def get_max_over_min_feature():
    max_over_min_list = []  # list that contains a list of max_over_min  for each logfile/df

    for list_idx, df in enumerate(gl.df_list):
        df['max_over_min'] = get_max_over_min_column(df)
        df = df[df['Time'] > max(gl.cw, gl.hw)]  # remove first window-seconds bc. not accurate data

        # max_over_min_resampled = factory.resample_dataframe(df[['timedelta', 'Time', 'max_over_min']], 1)
        max_over_min = []
        for _, row in gl.obstacle_df_list[list_idx].iterrows():
            corresp_row = df[df['Time'] <= row['Time']].iloc[-1]
            max_over_min.append(corresp_row['max_over_min'])
        max_over_min_list.append(max_over_min)

    return pd.DataFrame(list(itertools.chain.from_iterable(max_over_min_list)), columns=['max_over_min'])


def get_last_obstacle_crash_feature():
    crashes_list = []  # list that contains a list of whether crash or not for each logfile/df
    for df in gl.df_list:
        df['last_obstacle_crash'] = get_last_obstacle_crash_column(df)
        df = df[df['Time'] > max(gl.cw, gl.hw)]  # remove first window-seconds bc. not accurate data

        df_obstacles = df[(df['Logtype'] == 'EVENT_CRASH') | (df['Logtype'] == 'EVENT_OBSTACLE')]
        crashes_list.append(df_obstacles['last_obstacle_crash'])
    return pd.DataFrame(list(itertools.chain.from_iterable(crashes_list)), columns=['last_obstacle_crash'])


"""The following methods calculate the features as a new dataframe column"""


'''Adds a column at each timestamp that indicates whether or not the user crashed
    into the last obstacle
'''


def get_percentage_crashes_column(df):
    def df_from_to(_from, _to):
        mask = (_from < df['Time']) & (df['Time'] < _to)
        return df[mask]

    def compute_crashes(row):
        last_x_seconds_df = df_from_to(max(0, row['Time'] - gl.cw), row['Time'])
        num_obstacles = len(last_x_seconds_df[(last_x_seconds_df['Logtype'] == 'EVENT_OBSTACLE')
                                              | (last_x_seconds_df['Logtype'] == 'EVENT_CRASH')].index)
        num_crashes = len(last_x_seconds_df[last_x_seconds_df['Logtype'] == 'EVENT_CRASH'].index)
        return (num_crashes/num_obstacles * 100 if num_crashes < num_obstacles else 100) if num_obstacles != 0 else 0

    return df[['Time', 'Logtype']].apply(compute_crashes, axis=1)


''' Adds a column at each timestamp that indicates the mean
    heartrate over the last 'heartrate_window' seconds
'''


def get_mean_heartrate_column(df):
    def df_from_to(_from, _to):
        mask = (_from < df['Time']) & (df['Time'] <= _to)
        return df[mask]

    def compute_mean_hr(row):
        last_x_seconds_df = df_from_to(max(0, row['Time'] - gl.hw), row['Time'])
        return last_x_seconds_df[last_x_seconds_df['Heartrate'] != -1]['Heartrate'].mean()

    return df[['Time', 'Heartrate']].apply(compute_mean_hr, axis=1)


'''Adds a column at each timestamp that indicates the difference between max and min 
    hearrate in the last x seconds
'''


def get_max_over_min_column(df):

    def df_from_to(_from, _to):
        mask = (_from < df['Time']) & (df['Time'] <= _to)
        return df[mask]

    def compute_mean_hr(row):
        last_x_seconds_df = df_from_to(max(0, row['Time'] - gl.hw), row['Time'])
        max_hr = last_x_seconds_df[last_x_seconds_df['Heartrate'] != -1]['Heartrate'].max()
        min_hr = last_x_seconds_df[last_x_seconds_df['Heartrate'] != -1]['Heartrate'].min()

        return max_hr / min_hr

    return df[['Time', 'Heartrate']].apply(compute_mean_hr, axis=1)


'''Adds a column at each timestamp that indicates whether the user crashed into the last obstacle or not
'''


def get_last_obstacle_crash_column(df):
    def compute_crashes(row):
        last = df[(df['Time'] < row['Time']) & ((df['Logtype'] == 'EVENT_OBSTACLE') | (df['Logtype'] == 'EVENT_CRASH'))]
        if last.empty:
            return 0
        return 1 if last.iloc[-1]['Logtype'] == 'EVENT_CRASH' else 0

    return df[['Time', 'Logtype']].apply(compute_crashes, axis=1)