import pandas as pd
import os
import math

from scipy import stats
import itertools

import globals as gl

'''Current features:
    1. Mean_hr over last x seconds
    2. % of crashes in the last x seconds
    3. Max over min heartrate in last x seconds
'''

feature_names = ['mean_hr', 'max_hr', 'min_hr', 'std_hr', '%crashes', 'max_over_min', 'last_obstacle_crash']

''' Returns a matrix containing the features, and the labels
    There is one feature-row for each obstacle
 '''


def get_feature_matrix_and_label():

    matrix = pd.DataFrame()

    if gl.use_cache & (not gl.test_data) & os.path.isfile(gl.working_directory_path + '/Pickle/feature_matrix.pickle'):
        matrix = pd.read_pickle(gl.working_directory_path + '/Pickle/feature_matrix.pickle')
    else:
        matrix['mean_hr'] = get_hr_feature('mean_hr')
        matrix['max_hr'] = get_hr_feature('max_hr')
        matrix['min_hr'] = get_hr_feature('min_hr')
        matrix['std_hr'] = get_hr_feature('std_hr')
        matrix['max_over_min'] = get_hr_feature('max_over_min')
        matrix['%crashes'] = get_percentage_crashes_feature()
        matrix['last_obstacle_crash'] = get_last_obstacle_crash_feature()

        matrix.to_pickle(gl.working_directory_path + '/Pickle/feature_matrix.pickle')

    # remove ~ first heartrate_window rows (they have < hw seconds to compute features, and are thus not accurate)
    labels = []
    for df in gl.obstacle_df_list:
        labels.append(df[df['Time'] > max(gl.cw, gl.hw)]['crash'].copy())
    labels = list(itertools.chain.from_iterable(labels))

    # Boxcox transformation
    if gl.use_boxcox:
        # Values must be positive. If not, shift it
        for feature in feature_names:
            if not feature == 'last_obstacle_crash':  # Doesn't makes ense to do boxcox here
                if matrix[feature].min() <= 0:
                    matrix[feature] = stats.boxcox(matrix[feature] - matrix[feature].min() + 0.01)[0]
                else:
                    matrix[feature] = stats.boxcox(matrix[feature])[0]

    return matrix.as_matrix(), labels


"""The following methods append a column to the feature matrix"""


def get_hr_feature(hr_applier):
    print('Creating ' + hr_applier + ' feature...')
    hr_df_list = []  # list that contains a list of mean_hrs for each logfile/df
    for list_idx, df in enumerate(gl.df_list):
        if not (df['Heartrate'] == -1).all(): # NOTE: Can be omitted if logfiles without heartrate data is removed in setup.py

            hr_df = get_heartrate_column(list_idx, df, hr_applier)
            hr_df_list.append(hr_df)

    return pd.DataFrame(list(itertools.chain.from_iterable(hr_df_list)), columns=[hr_applier])


# TODO: Normalize crashes depending on size/assembly of the obstacle


def get_percentage_crashes_feature():
    print('Creating %crashes feature...')

    crashes_list = []  # list that contains a list of %crashes for each logfile/df
    for list_idx, df in enumerate(gl.df_list):
        crashes = get_percentage_crashes_column(list_idx, df)
        # df = df[df['Time'] > max(gl.cw, gl.hw)]  # remove first window-seconds bc. not accurate data
        crashes_list.append(crashes)
    return pd.DataFrame(list(itertools.chain.from_iterable(crashes_list)), columns=['%crashes'])


def get_last_obstacle_crash_feature():
    print('Creating last_obstacle_crash feature...')
    crashes_list = []  # list that contains a list of whether crash or not for each logfile/df
    for list_idx, df in enumerate(gl.df_list):
        df_obstacles = get_last_obstacle_crash_column(list_idx, df)
        # df = df[df['Time'] > max(gl.cw, gl.hw)]  # remove first window-seconds bc. not accurate data
        crashes_list.append(df_obstacles)

    return pd.DataFrame(list(itertools.chain.from_iterable(crashes_list)), columns=['last_obstacle_crash'])


"""The following methods calculate the features as a new dataframe column"""


'''Returns the dataframe where time is between _from and _to'''


def df_from_to(_from, _to, df):
    mask = (_from <= df['Time']) & (df['Time'] < _to)
    return df[mask]


''' Returns a dataframe column that indicates at each timestamp the 
    heartrate over the last 'heartrate_window' seconds, after applying 'applyer' (e.g. mean, min, max, std)
'''


def get_heartrate_column(idx, df, hr_applier):

    def compute_hr(row):
        last_x_seconds_df = df_from_to(max(0, row['Time'] - gl.hw), row['Time'], df)
        res = 0
        if hr_applier == 'mean_hr':
            res = last_x_seconds_df[last_x_seconds_df['Heartrate'] != -1]['Heartrate'].mean()
        elif hr_applier == 'min_hr':
            res = last_x_seconds_df[last_x_seconds_df['Heartrate'] != -1]['Heartrate'].min()
        elif hr_applier == 'max_hr':
            res = last_x_seconds_df[last_x_seconds_df['Heartrate'] != -1]['Heartrate'].max()
        elif hr_applier == 'std_hr':
            res = last_x_seconds_df[last_x_seconds_df['Heartrate'] != -1]['Heartrate'].std()
        elif hr_applier == 'max_over_min':
            last_x_seconds_df = df_from_to(max(0, row['Time'] - gl.max_over_min_hw), row['Time'], df)
            max_hr = last_x_seconds_df[last_x_seconds_df['Heartrate'] != -1]['Heartrate'].max()
            min_hr = last_x_seconds_df[last_x_seconds_df['Heartrate'] != -1]['Heartrate'].min()
            res = max_hr / min_hr
        if res == 0:
            print('error')

        # first mean will be nan, so replace it with second row instead
        return res if not math.isnan(res) else compute_hr(df.iloc[1])

    return gl.obstacle_df_list[idx].apply(compute_hr, axis=1)


''' Returns a dataframe column that indicates at each timestamp how many percentage of the last obstacles in the 
    last crash-window-seconds the user crashed into
'''


def get_percentage_crashes_column(idx, df):

    def compute_crashes(row):
        last_x_seconds_df = df_from_to(max(0, row['Time'] - gl.cw), row['Time'], df)
        num_obstacles = len(last_x_seconds_df[(last_x_seconds_df['Logtype'] == 'EVENT_OBSTACLE')
                                              | (last_x_seconds_df['Logtype'] == 'EVENT_CRASH')].index)
        num_crashes = len(last_x_seconds_df[last_x_seconds_df['Logtype'] == 'EVENT_CRASH'].index)
        return (num_crashes/num_obstacles * 100 if num_crashes < num_obstacles else 100) if num_obstacles != 0 else 0

    return gl.obstacle_df_list[idx].apply(compute_crashes, axis=1)


'''Returns a dataframe column that indicates at each timestamp whether the user crashed into the last obstacle or not
'''


def get_last_obstacle_crash_column(idx, df):
    def compute_crashes(row):
        last = df[(df['Time'] < row['Time']) & ((df['Logtype'] == 'EVENT_OBSTACLE') | (df['Logtype'] == 'EVENT_CRASH'))]
        if last.empty:
            return 0
        return 1 if last.iloc[-1]['Logtype'] == 'EVENT_CRASH' else 0

    return gl.obstacle_df_list[idx].apply(compute_crashes, axis=1)