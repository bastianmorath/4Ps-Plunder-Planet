import pandas as pd
import time

import factory as factory
import globals as gl
''' For each obstacle, add mean_hr, %crashes in the past x seconds

'''

# obstacle_df contains timestamp of each obstacle and whether or not the user crashed
obstacle_df = []

def get_feature_matrix_and_label():
    global obstacle_df
    factory.plot_features()

    obstacle_df = factory.get_obstacle_times_with_success()

    feature_df = pd.DataFrame()
    add_mean_hr_to_dataframe(feature_df)
    add_crashes_to_dataframe(feature_df)

    labels = obstacle_df['crash'].copy()

    return feature_df.as_matrix(), labels.tolist()


def add_mean_hr_to_dataframe(dataframe):
    obst_df = obstacle_df.copy()
    mean_hr_resampled = factory.resample_dataframe(gl.df[['timedelta', 'Time', 'mean_hr']], 1)
    mean_hr_df = []
    for idx, row in obst_df.iterrows():
        corresp_row = mean_hr_resampled[mean_hr_resampled['Time'] <= row['Time']].iloc[-1]
        mean_hr_df.append(corresp_row['mean_hr'])

    dataframe['mean_hr'] = mean_hr_df

# TODO: Normalize crashes depending on size/assembly of the obstacle
def add_crashes_to_dataframe(dataframe):
    obst_df = obstacle_df.copy()
    crashes_df = []
    crashes_resampled = factory.resample_dataframe(gl.df[['timedelta', 'Time', '%crashes']], 1)
    for idx, row in obst_df.iterrows():
        corresp_row = crashes_resampled[crashes_resampled['Time'] <= row['Time']].iloc[-1]
        crashes_df.append(corresp_row['%crashes'])
    dataframe['%crashes'] = crashes_df


def add_mean_hr_to_df(heartrate_window):
    time1 = time.time()
    # Compute mean_hr over last 'heart_window' seconds
    df_with_hr = gl.df[gl.df['Heartrate'] != -1]
    if len(df_with_hr.index) == 0:
        print('ERROR: Data has no heartrate! ')
        return

    gl.df['mean_hr'] = factory.get_mean_heartrate_column(gl.df, heartrate_window)
    time2 = time.time()
    print("Time to get mean_hr: " + str(time2 - time1))


def add_crashes_to_df(crash_window):
    time1 = time.time()

    # Compute %crashes over last 'crash_window' seconds
    gl.df['%crashes'] = factory.get_crashes_column(gl.df, crash_window)

    time2 = time.time()
    print("Time to get %crashes: " + str(time2 - time1))
