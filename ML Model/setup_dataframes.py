"""This module reads the logfiles and transforms them into dataframes.

- One can choose whether one wants only the fbmc, only the kinect, or all files.

- The heartrates get normalized and a timedelta column is appended to the dataframes

- For each logfile, two dataframes are created:
    - one dataframe that just contains all the information from the logfile, stored in globals.df_list

    - one dataframe that contains one row for each obstacle occuring in the logfile,
        indicating the time, whether the user crashed into it or not, the user_id and the log_id

- At the very first time, using the original logfiles, refactor_crashes() has to be called. This accelerates
    all the computations later

"""

import os
import re
import numpy as np
import pandas as pd

import refactoring_logfiles as refactoring

import features_factory as f_factory


use_fewer_data = False  # Can be used for debugging (fewer data is used)

working_directory_path = os.path.abspath(os.path.dirname(__file__))
project_path = os.path.abspath(os.path.join(working_directory_path, '../../..'))

abs_path_logfiles = project_path + '/Logs/text_logs_refactored_crashes'  # Logfiles to use
names_logfiles = []  # Name of the logfiles

df_list = []  # List with all dataframes; 1 dataframe per logfile

# list of dataframes, each dataframe has time of each obstacle and whether crash or not (1 df per logfile)
# First max(cw, hw) seconds removed
obstacle_df_list = []

print_key_numbers = False


def setup(use_fewer_data=False):
    """

    :param use_fewer_data:

    """

    all_names = [f for f in sorted(os.listdir(abs_path_logfiles)) if re.search(r'.log', f)]

    kinect_names_all = [f for f in sorted(os.listdir(abs_path_logfiles)) if re.search(r'.{0,}Kinect.{0,}.log', f)]
    kinect_names_hr = [f for f in sorted(os.listdir(abs_path_logfiles)) if re.search(r'.{0,}Kinect_hr.{0,}.log', f)]

    fbmc_names_all = [f for f in sorted(os.listdir(abs_path_logfiles)) if re.search(r'.{0,}FBMC.{0,}.log', f)]
    fbmc_names_hr_points = [f for f in sorted(os.listdir(abs_path_logfiles)) if
                            re.search(r'.{0,}FBMC_hr_(1|2).{0,}.log', f)]

    globals()['names_logfiles'] = fbmc_names_hr_points
    globals()['use_fewer_data'] = use_fewer_data

    if use_fewer_data:
        # names_logfiles = ['ISI_FBMC_hr_1.log', 'LZ_FBMC_hr_2.log', 'MH_FBMC_hr_1.log']
        globals()['names_logfiles'] = ['ISI_FBMC_hr_1.log']

    print('Creating dataframes...')
    read_and_prepare_logs()

    globals()['obstacle_df_list'] = get_obstacle_times_with_success()

    if print_key_numbers:
        print_keynumbers_logfiles()


'''Reads the logfiles and parses them into Pandas dataframes. 
    Also adds additional log&timedelta column, cuts them to the same length and normalizes heartrate
'''


def read_and_prepare_logs():

    logs = [abs_path_logfiles + "/" + s for s in names_logfiles]

    # This read_csv is used when using the original logfiles
    # column_names = ['Time', 'Logtype', 'Gamemode', 'Points', 'Heartrate', 'physDifficulty',
    #                 'psyStress', 'psyDifficulty', 'obstacle']
    #  globals()['df_list'] = list(pd.read_csv(log, sep=';', skiprows=5, index_col=False,
    #                                  names=column_names) for log in logs)

    # This read_csv is used when using the refactored logs (Crash/Obstacle cleaned up)
    column_names = ['Time', 'Logtype', 'Gamemode', 'Points', 'Heartrate', 'physDifficulty',
                    'psyStress', 'psyDifficulty', 'obstacle', 'userID', 'logID']
    globals()['df_list'] = list(pd.read_csv(log, sep=';', index_col=False, names=column_names) for log in logs)

    refactoring.add_timedelta_column()  # Is added after refactoring, since timedelta format gets lost otherwise
    # POSSIBLY_DANGEROUS
    refactoring.remove_logs_without_heartrates_or_points()  # Now done by using corresponding logs directly

    refactoring.normalize_heartrate()

    refactoring.add_log_and_user_column()


def refactor_crashes():
    """
        In the original logfiles, there are always two event happening in case of an obstacle:
        an EVENT_CRASH (Which doesn't have any information about its obstacle) and an EVENT_OBSTACLE, which makes it
        more difficult to use the logs later.
        Thus, in case of a crash, I remove the EVENT_OBSTACLE and move its obstacle information to the EVENT_CRASH log
        Additionaly, I add a column with the userID and whether it's the first or second logfile of the user

    :return: New dataframe that either contains an EVENT_CRASH (obstacle with crash) or
                EVENT_OBSTACLE (obstacle without a crash)

    """
    refactoring.refactor_crashes()


def print_keynumbers_logfiles():
    """ Prints important numbers of the logfiles:
        - number of logs, events, features (=obstacles)
        - number of files that contain heartrate vs no heartrate
    """

    df_lengths = []
    for d in df_list:
        df_lengths.append(d['Time'].max())
    print('average:' + str(np.mean(df_lengths)) + ', std: ' + str(np.std(df_lengths)) +
          ', max: ' + str(np.max(df_lengths)) + ', min: ' + str(np.min(df_lengths)))

    print('#files: ' + str(len(df_list)))
    print('#files with heartrate: ' + str(len([a for a in df_list if not (a['Heartrate'] == -1).all()])))
    print('#datapoints: ' + str(sum([len(a.index) for a in df_list])))
    print('#obstacles: ' + str(sum([len(df.index) for df in obstacle_df_list])))
    print('#crashes: ' + str(sum([len(df[df['crash'] == 1]) for df in obstacle_df_list])))


def get_obstacle_times_with_success():
    """Returns a list of dataframes, each dataframe has time of each obstacle and whether crash or not (1 df per logfile)

    userID/logID is used such that we can later (after creating feature matrix), still differentiate logfiles

    :return: List which contains a dataframe df per log. Each df contains the time, userID, logID and whether the user
                crashed (of each obstacle)

    """
    obstacle_time_crash = []

    for dataframe in df_list:
        obstacle_times_current_df = []
        for idx, row in dataframe.iterrows():
            if row['Time'] > max(f_factory.cw, f_factory.hw):
                if row['Logtype'] == 'EVENT_OBSTACLE':
                    obstacle_times_current_df.append((row['Time'], 0, row['userID'], row['logID']))
                if row['Logtype'] == 'EVENT_CRASH':
                    obstacle_times_current_df.append((row['Time'], 1, row['userID'], row['logID']))
        times = np.asarray([a for (a, b, c, d) in obstacle_times_current_df])
        crashes = np.asarray([b for (a, b, c, d) in obstacle_times_current_df])
        userIDs = np.asarray([c for (a, b, c, d) in obstacle_times_current_df])
        logIDs = np.asarray([d for (a, b, c, d) in obstacle_times_current_df])

        obstacle_time_crash.append(pd.DataFrame({'Time': times, 'crash': crashes,
                                                 'userID': userIDs,
                                                 'logID': logIDs}))

    return obstacle_time_crash
