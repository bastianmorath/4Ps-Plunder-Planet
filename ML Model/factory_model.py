# factory_model.py

from __future__ import division  # s.t. division uses float result
import matplotlib.pyplot as plt

import globals_model as gl

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


def get_obstacle_times_with_success():
    df = gl.df_total
    obstacle_time_crash = []

    def df_from_to(_from, _to):
        mask = (_from < df['Time']) & (df['Time'] <= _to)
        return df[mask]

    for index, row in df.iterrows():
        if row['Logtype'] == 'EVENT_OBSTACLE':
            last_second_df = df_from_to(row['Time'] - 1, row['Time'])
            crash = False
            if (last_second_df['Logtype'] == 'EVENT_CRASH').any():
                crash = True
            obstacle_time_crash.append([row['Time'], crash])
    print(obstacle_time_crash)
