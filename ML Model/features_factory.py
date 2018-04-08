import pandas as pd
import os

import factory as factory
import globals as gl

'''Current features:
    1. Mean_hr over last x seconds
    2. % of crashes in the last x seconds
    3. Max over min heartrate in last x seconds
'''

"""Computes features, stores it as additional columns in gl.df_without_features and returns it"""


def get_df_with_feature_columns():
    df = gl.df_without_features
    df['mean_hr'] = get_mean_heartrate_column()
    df['%crashes'] = get_crashes_column()
    df['max_over_min'] = get_max_over_min_column()
    return df


''' Returns a matrix containing the features, and the labels
    There is one feature-row for each obstacle
 '''


def get_feature_matrix_and_label():
    # remove ~ first heartrate_window rows (they have < hw seconds to compute features, and are thus not accurate)
    gl.df_without_features = gl.df_without_features[gl.df_without_features['Time'] > gl.hw]
    gl.obstacle_df = gl.obstacle_df[gl.obstacle_df['Time'] > gl.hw]

    matrix = pd.DataFrame()
    if gl.use_cache & os.path.isfile(gl.working_directory_path + '/Pickle/feature_matrix.pickle'):
        matrix = pd.read_pickle(gl.working_directory_path + '/Pickle/feature_matrix.pickle')
    else:
        add_mean_hr_to_dataframe(matrix)
        add_crashes_to_dataframe(matrix)
        add_max_over_min_hr_to_dataframe(matrix)
        matrix.to_pickle(gl.working_directory_path + '/Pickle/feature_matrix.pickle')
    labels = gl.obstacle_df['crash'].copy()
    return matrix.as_matrix(), labels.tolist()


"""The following methods append a column to the feature matrix (after resampling it)"""


def add_mean_hr_to_dataframe(matrix):
    mean_hr_resampled = factory.resample_dataframe(gl.df[['timedelta', 'Time', 'mean_hr']], 1)
    mean_hr_df = []
    for idx, row in gl.obstacle_df.iterrows():
        corresp_row = mean_hr_resampled[mean_hr_resampled['Time'] <= row['Time']].iloc[-1]
        mean_hr_df.append(corresp_row['mean_hr'])

    matrix['mean_hr'] = mean_hr_df


# TODO: Normalize crashes depending on size/assembly of the obstacle


def add_crashes_to_dataframe(matrix):
    crashes_df = []
    crashes_resampled = factory.resample_dataframe(gl.df[['timedelta', 'Time', '%crashes']], 1)
    for idx, row in gl.obstacle_df.iterrows():
        corresp_row = crashes_resampled[crashes_resampled['Time'] <= row['Time']].iloc[-1]
        crashes_df.append(corresp_row['%crashes'])
    matrix['%crashes'] = crashes_df


def add_max_over_min_hr_to_dataframe(matrix):
    max_over_min_resampled = factory.resample_dataframe(gl.df[['timedelta', 'Time', 'max_over_min']], 1)
    max_over_min = []
    for idx, row in gl.obstacle_df.iterrows():
        corresp_row = max_over_min_resampled[max_over_min_resampled['Time'] <= row['Time']].iloc[-1]
        max_over_min.append(corresp_row['max_over_min'])

    matrix['max_over_min'] = max_over_min


"""The following methods calculate the features as a new dataframe column"""


'''Adds a column at each timestamp that indicates the %Crashes 
    the user did in the last 'window_size' seconds
'''


def get_crashes_column():
    df = gl.df_without_features

    def df_from_to(_from, _to):
        mask = (_from < df['Time']) & (df['Time'] <= _to)
        return df[mask]

    def compute_crashes(row):
        last_x_seconds_df = df_from_to(max(0, row['Time'] - gl.cw), row['Time'])
        num_obstacles = len(last_x_seconds_df[last_x_seconds_df['Logtype'] == 'EVENT_OBSTACLE'].index)
        num_crashes = len(last_x_seconds_df[last_x_seconds_df['Logtype'] == 'EVENT_CRASH'].index)
        return (num_crashes/num_obstacles * 100 if num_crashes < num_obstacles else 100) if num_obstacles != 0 else 0

    return df[['Time', 'Logtype']].apply(compute_crashes, axis=1)


''' Adds a column at each timestamp that indicates the mean
    heartrate over the last 'heartrate_window' seconds
'''


def get_mean_heartrate_column():
    df = gl.df_without_features

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


def get_max_over_min_column():
    df = gl.df_without_features

    def df_from_to(_from, _to):
        mask = (_from < df['Time']) & (df['Time'] <= _to)
        return df[mask]

    def compute_mean_hr(row):
        last_x_seconds_df = df_from_to(max(0, row['Time'] - gl.hw), row['Time'])
        max_hr = last_x_seconds_df[last_x_seconds_df['Heartrate'] != -1]['Heartrate'].max()
        min_hr = last_x_seconds_df[last_x_seconds_df['Heartrate'] != -1]['Heartrate'].min()

        return max_hr / min_hr

    return df[['Time', 'Heartrate']].apply(compute_mean_hr, axis=1)