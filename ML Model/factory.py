# factory_model.py

from __future__ import division  # s.t. division uses float result
# from mpl_toolkits.mplot3d import Axes3D

import globals as gl
import pandas as pd
import numpy as np


'''Resamples a dataframe with a sampling frquency of 'resolution'
    -> Smoothes the plots
'''


def resample_dataframe(df, resolution):
    df = df.set_index('timedelta', drop=True)  # set timedelta as new index
    resampled = df.resample(str(resolution)+'S').mean()
    resampled.reset_index(inplace=True)
    # timedelta was resampled, so we need to do the same with the Time-column
    resampled['Time'] = resampled['timedelta'].apply(lambda time: time.total_seconds())
    return resampled


''' Returns a dataframe with the time of each obstacle and whether or not
    the user crashed into it or not
'''


def get_obstacle_times_with_success():

    obstacle_time_crash = []

    ''' If there was a crash, then there would be a 'EVENT_CRASH' in the preceding around 1 seconds of the event'''
    def is_a_crash(index):
        count = 1
        while True:
            logtype = gl.df_without_features.iloc[index-count]['Logtype']
            if logtype == 'EVENT_OBSTACLE':
                return 0
            if logtype == 'EVENT_CRASH':
                return 1
            count += 1
    for idx, row in gl.df_without_features.iterrows():
        if row['Logtype'] == 'EVENT_OBSTACLE':
            obstacle_time_crash.append((row['Time'], is_a_crash(idx)))

    times = np.asarray([a for (a, b) in obstacle_time_crash])
    crashes = np.asarray([b for (a, b) in obstacle_time_crash])
    return pd.DataFrame({'Time': times, 'crash': crashes})




