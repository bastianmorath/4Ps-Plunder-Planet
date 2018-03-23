# factory_model.py

from __future__ import division  # s.t. division uses float result
import matplotlib.pyplot as plt

import globals_model as gl
import pandas as pd
import numpy as np


green_color = '#AEBD38'
blue_color = '#68829E'
red_color = '#A62A2A'

'''
    Adds a column at each timestamp that indicates the %Crashes 
    the user did in the last 'window_size' seconds
'''


def get_crashes_column(df, crash_window):
    def df_from_to(_from, _to):
        mask = (_from < df['Time']) & (df['Time'] <= _to)
        return df[mask]
        
    def compute_crashes(row):
        last_x_seconds_df = df_from_to(max(0, row['Time']-crash_window), row['Time'])
        num_obstacles = len(last_x_seconds_df[last_x_seconds_df['Logtype'] == 'EVENT_OBSTACLE'].index)
        num_crashes = len(last_x_seconds_df[last_x_seconds_df['Logtype'] == 'EVENT_CRASH'].index)
        return (num_crashes/num_obstacles * 100 if num_crashes < num_obstacles else 100) if num_obstacles != 0 else 0

    return df[['Time', 'Logtype']].apply(compute_crashes, axis=1)


'''
    Adds a column at each timestamp that indicates the mean
    heartrate over the last 'heartrate_window' seconds
'''


def get_mean_heartrate_column(df, heartrate_window):

    def df_from_to(_from, _to):
        mask = (_from < df['Time']) & (df['Time'] <= _to)
        return df[mask]

    def compute_mean_hr(row):
        last_x_seconds_df = df_from_to(max(0, row['Time'] - heartrate_window), row['Time'])
        return last_x_seconds_df[last_x_seconds_df['Heartrate'] != -1]['Heartrate'].mean()

    return df[['Time', 'Heartrate']].apply(compute_mean_hr, axis=1)


'''Resamples a dataframe with a sampling frquency of 'resolution'
    -> Smoothes the plots
'''


def resample_dataframe(df, resolution):
    df.set_index('timedelta', inplace=True)  # set timedelta as new index
    return df.resample(str(resolution)+'S').mean()  # Resample series'


'''Plots the mean_hr and %crashes that were calulated for the last x seconds for each each second
'''


def plot(df):
    fig, ax1 = plt.subplots()
    fig.suptitle('%Crashes and mean_hr over last x seconds')

    # Plot mean_hr
    ax1.plot(df['Time'], df['mean_hr'], blue_color)
    ax1.set_xlabel('Playing time [s]')
    ax1.set_ylabel('Heartrate', color=blue_color)
    ax1.tick_params('y', colors=blue_color)

    # Plot %crashes
    ax2 = ax1.twinx()
    ax2.plot(df['Time'], df['%crashes'], red_color)
    ax2.set_ylabel('Crashes [%]', color=red_color)
    ax2.tick_params('y', colors=red_color)

    plt.savefig(gl.working_directory_path + '/crashes_and_mean_hr.pdf')


''' Returns a dataframe with the time of each obstacle and whether or not
    the user crashed into it or not
'''


def get_obstacle_times_with_success():
    df = gl.df_total

    obstacle_time_crash = []

    ''' If there was a crash, then there would be a 'EVENT_CRASH' in the preceding around 1 seconds of the event
    '''
    def is_a_crash(index):
        count = 1
        while True:
            logtype = df.iloc[index-count]['Logtype']
            if logtype == 'EVENT_OBSTACLE':
                return 0
            if logtype == 'EVENT_CRASH':
                return 1
            count += 1

    for idx, row in df.iterrows():
        if row['Logtype'] == 'EVENT_OBSTACLE':
            obstacle_time_crash.append((row['Time'], is_a_crash(idx)))

    times = np.asarray([a for (a, b) in obstacle_time_crash])
    crashes = np.asarray([b for (a, b) in obstacle_time_crash])
    return pd.DataFrame({'Time': times, 'crash': crashes})


''' For each obstacle, add mean_hr, %crashes in the past x seconds

'''


def get_feature_matrix_and_label(obstacle_df, dataframe):
    # For each timestamp, add already calculated mean_hr and %crashes
    mean_hr_df = []
    crashes_df = []
    df = obstacle_df.copy()
    for idx, row in df.iterrows():
        corresp_row = dataframe[dataframe['Time'] <= row['Time']].iloc[-1]
        mean_hr_df.append(corresp_row['mean_hr'])
        crashes_df.append(corresp_row['%crashes'])

    df.drop(['crash', 'Time'], axis=1, inplace=True)
    df['mean_hr'] = mean_hr_df
    df['crashes'] = crashes_df
    return df, obstacle_df['crash'].copy()
