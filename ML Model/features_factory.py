import pandas as pd
import os

import factory as factory
import globals as gl
''' For each obstacle, add mean_hr, %crashes in the past x seconds

'''

# obstacle_df contains timestamp of each obstacle and whether or not the user crashed
obstacle_df = []


def get_feature_matrix_and_label():
    global obstacle_df
    matrix = pd.DataFrame()
    obstacle_df = []

    if os.path.isfile(gl.working_directory_path + '/Pickle/feature_matrix.pickle'):
        matrix = pd.read_pickle(gl.working_directory_path + '/Pickle/feature_matrix.pickle')
        obstacle_df = pd.read_pickle(gl.working_directory_path + '/Pickle//obstacle_df.pickle')
    else:
        obstacle_df = factory.get_obstacle_times_with_success()
        obstacle_df.to_pickle(gl.working_directory_path + '/Pickle/obstacle_df.pickle')
        add_mean_hr_to_dataframe(matrix)
        add_crashes_to_dataframe(matrix)
        add_max_over_min_hr_to_dataframe(matrix)
        matrix.to_pickle(gl.working_directory_path + '/Pickle/feature_matrix.pickle')

    labels = obstacle_df['crash'].copy()
    return matrix.as_matrix(), labels.tolist()


'''The following methods append a column to the featurematrix
'''


def add_mean_hr_to_dataframe(matrix):
    obst_df = obstacle_df.copy()
    mean_hr_resampled = factory.resample_dataframe(gl.df[['timedelta', 'Time', 'mean_hr']], 1)
    mean_hr_df = []
    for idx, row in obst_df.iterrows():
        corresp_row = mean_hr_resampled[mean_hr_resampled['Time'] <= row['Time']].iloc[-1]
        mean_hr_df.append(corresp_row['mean_hr'])

    matrix['mean_hr'] = mean_hr_df

# TODO: Normalize crashes depending on size/assembly of the obstacle


def add_crashes_to_dataframe(matrix):
    obst_df = obstacle_df.copy()
    crashes_df = []
    crashes_resampled = factory.resample_dataframe(gl.df[['timedelta', 'Time', '%crashes']], 1)
    for idx, row in obst_df.iterrows():
        corresp_row = crashes_resampled[crashes_resampled['Time'] <= row['Time']].iloc[-1]
        crashes_df.append(corresp_row['%crashes'])

    matrix['%crashes'] = crashes_df


def add_max_over_min_hr_to_dataframe(matrix):
    obst_df = obstacle_df.copy()
    max_over_min_resampled = factory.resample_dataframe(gl.df[['timedelta', 'Time', 'max_over_min']], 1)
    max_over_min = []
    for idx, row in obst_df.iterrows():
        corresp_row = max_over_min_resampled[max_over_min_resampled['Time'] <= row['Time']].iloc[-1]
        max_over_min.append(corresp_row['max_over_min'])

    matrix['max_over_min'] = max_over_min


'''The following methods append a column to the global dataframe with the
corresponding values
'''


def add_mean_hr_to_df(heartrate_window):
    gl.df['mean_hr'] = factory.get_mean_heartrate_column(gl.df, heartrate_window)


def add_crashes_to_df(crash_window):
    # Compute %crashes over last 'crash_window' seconds
    gl.df['%crashes'] = factory.get_crashes_column(gl.df, crash_window)


def add_max_over_min_hr_to_df(max_over_min_window):
    gl.df['max_over_min'] = factory.get_max_over_min_column(gl.df, max_over_min_window)
