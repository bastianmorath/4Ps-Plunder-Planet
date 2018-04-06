# factory_model.py

from __future__ import division  # s.t. division uses float result
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

import globals as gl
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
    df = df.set_index('timedelta', drop=True)  # set timedelta as new index
    return df.resample(str(resolution)+'S').mean()  # Resample series'


'''Plots the mean_hr and %crashes that were calulated for the last x seconds for each each second
'''


def plot_features(gamma, c, auroc, percentage):
    fig, ax1 = plt.subplots()
    fig.suptitle('%Crashes and mean_hr over last x seconds')

    # Plot mean_hr
    df = gl.df.sort_values('Time')
    ax1.plot(df['Time'], df['mean_hr'], blue_color)
    ax1.set_xlabel('Playing time [s]')
    ax1.set_ylabel('Heartrate', color=blue_color)
    ax1.tick_params('y', colors=blue_color)

    '''
    # Plot max_over_min_hr
    df = gl.df.sort_values('Time')
    ax1.plot(df['Time'], df['max_over_min'], blue_color)
    ax1.set_xlabel('Playing time [s]')
    ax1.set_ylabel('max_over_min_hr', color=blue_color)
    ax1.tick_params('y', colors=blue_color)
    '''
    # Plot %crashes
    ax2 = ax1.twinx()
    ax2.plot(df['Time'], df['%crashes'], red_color)
    ax2.set_ylabel('Crashes [%]', color=red_color)
    ax2.tick_params('y', colors=red_color)


    ax2.text(0.5, 0.35, 'Crash_window: ' + str(gl.cw),
         transform=ax2.transAxes, fontsize=10)
    ax2.text(0.5, 0.3, 'Max_over_min window: ' + str(gl.hw),
             transform=ax2.transAxes, fontsize=10)
    ax2.text(0.5, 0.25, 'Best gamma: 10e' + str(round(gamma, 3)),
             transform=ax2.transAxes, fontsize=10)
    ax2.text(0.5, 0.2, 'Best c: 10e' + str(round(c, 3)),
             transform=ax2.transAxes, fontsize=10)
    ax2.text(0.5, 0.15, 'Auroc: ' + str(round(auroc, 3)),
             transform=ax2.transAxes, fontsize=10)
    ax2.text(0.5, 0.1, 'Correctly predicted: ' + str(round(percentage, 2)),
             transform=ax2.transAxes, fontsize=10)

    plt.savefig(gl.working_directory_path + '/features_plot_'+str(gl.cw) + '_'+str(gl.hw) + '.pdf')


'''Plot features and corresponding labels to (hopefully) see patterns
'''


def plot_features_with_labels(X, y):
    fig, ax = plt.subplots()
    # ax = Axes3D(fig)
    x1 = X[:, 0]  # mean_hr
    x2 = X[:, 1]  # %crashes
    x3 = X[:, 2]  # max_over_min_hr
    color = ['red' if x else 'green' for x in y]
    ax.scatter(x2, x3, color=color)
    ax.set_xlabel('crashes [%]')
    ax.set_ylabel('max_over_min')
    # ax.set_zlabel('max_hr / min_hr')
    # plt.show()
    plt.savefig(gl.working_directory_path + '/Plots//features_label_crashes__max_over_min.pdf')


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


'''
    Adds a column at each timestamp that indicates the difference between max and min 
    hearrate in the last x seconds
'''


def get_max_over_min_column(df, heartrate_window):
    def df_from_to(_from, _to):
        mask = (_from < df['Time']) & (df['Time'] <= _to)
        return df[mask]

    def compute_mean_hr(row):
        last_x_seconds_df = df_from_to(max(0, row['Time'] - heartrate_window), row['Time'])
        max_hr = last_x_seconds_df[last_x_seconds_df['Heartrate'] != -1]['Heartrate'].max()
        min_hr = last_x_seconds_df[last_x_seconds_df['Heartrate'] != -1]['Heartrate'].min()

        return max_hr / min_hr

    return df[['Time', 'Heartrate']].apply(compute_mean_hr, axis=1)
