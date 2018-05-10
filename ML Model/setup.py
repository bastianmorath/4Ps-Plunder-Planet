"""This module reads the logfiles and transofrms them into dataframes.

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

import globals as gl

# TODO: Do all those setup (e.g. add timeldelta, userid, etc. at the very beginning, then save it as new..
# logfiles/pickle and only ever use those


print_key_numbers = False


def print_keynumbers_logfiles():
    """ Prints important numbers of the logfiles:
        - number of logs, events, features (=obstacles)
        - number of files that contain heartrate vs no heartrate
    """

    df_lengths = []
    for d in gl.df_list:
        df_lengths.append(d['Time'].max())
    print('average:' + str(np.mean(df_lengths)) + ', std: ' + str(np.std(df_lengths)) +
          ', max: ' + str(np.max(df_lengths)) + ', min: ' + str(np.min(df_lengths)))

    print('#files: ' + str(len(gl.df_list)))
    print('#files with heartrate: ' + str(len([a for a in gl.df_list if not (a['Heartrate'] == -1).all()])))
    print('#datapoints: ' + str(sum([len(a.index) for a in gl.df_list])))
    print('#obstacles: ' + str(sum([len(df.index) for df in gl.obstacle_df_list])))
    print('#crashes: ' + str(sum([len(df[df['crash'] == 1]) for df in gl.obstacle_df_list ])))


def get_obstacle_times_with_success():
    """Returns a list of dataframes, each dataframe has time of each obstacle and whether crash or not (1 df per logfile)

    userID/logID is used such that we can later (after creating feature matrix), still differentiate logfiles

    :return: List which contains a dataframe df per log. Each df contains the time, userID, logID and whether the user
                crashed of each obstacle

    """

    obstacle_time_crash = []

    for dataframe in gl.df_list:
        obstacle_times_current_df = []
        for idx, row in dataframe.iterrows():
            if row['Time'] > max(gl.cw, gl.hw):
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


def setup():
    all_names = [f for f in sorted(os.listdir(gl.abs_path_logfiles)) if re.search(r'.log', f)]

    kinect_names_all = [f for f in sorted(os.listdir(gl.abs_path_logfiles)) if re.search(r'.{0,}Kinect.{0,}.log', f)]
    kinect_names_hr = [f for f in sorted(os.listdir(gl.abs_path_logfiles)) if re.search(r'.{0,}Kinect_hr.{0,}.log', f)]

    fbmc_names_all = [f for f in sorted(os.listdir(gl.abs_path_logfiles)) if re.search(r'.{0,}FBMC.{0,}.log', f)]
    fbmc_names_hr_points = [f for f in sorted(os.listdir(gl.abs_path_logfiles)) if
                            re.search(r'.{0,}FBMC_hr_(1|2).{0,}.log', f)]

    gl.names_logfiles = fbmc_names_hr_points

    if gl.testing:
        # gl.names_logfiles = ['ISI_FBMC_hr_1.log', 'LZ_FBMC_hr_2.log', 'MH_FBMC_hr_1.log']
        gl.names_logfiles = ['ISI_FBMC_hr_1.log']

    # Store computed dataframe in pickle file for faster processing
    if gl.use_cache:
        print('Dataframe already cached. Used this file to improve performance')
        if gl.use_boxcox and gl.reduced_features:
            gl.obstacle_df_list = pd.read_pickle(gl.working_directory_path + '/Pickle/reduced_features_boxcox/obstacle_df.pickle')
            gl.df_list = pd.read_pickle(gl.working_directory_path + '/Pickle/reduced_features_boxcox/df_list.pickle')
        elif gl.use_boxcox and not gl.reduced_features:
            gl.obstacle_df_list = pd.read_pickle(gl.working_directory_path + '/Pickle/all_features_boxcox/obstacle_df.pickle')
            gl.df_list = pd.read_pickle(gl.working_directory_path + '/Pickle/all_features_boxcox/df_list.pickle')
        elif not gl.use_boxcox and gl.reduced_features:
            gl.obstacle_df_list = pd.read_pickle(gl.working_directory_path + '/Pickle/reduced_features/obstacle_df.pickle')
            gl.df_list = pd.read_pickle(gl.working_directory_path + '/Pickle/reduced_features/df_list.pickle')
        elif not gl.use_boxcox and not gl.reduced_features:
            gl.obstacle_df_list = pd.read_pickle(gl.working_directory_path + '/Pickle/all_features/obstacle_df.pickle')
            gl.df_list = pd.read_pickle(gl.working_directory_path + '/Pickle/all_features/df_list.pickle')
    else:
        print('Dataframe not cached. Creating dataframe...')
        read_and_prepare_logs()

        gl.obstacle_df_list = get_obstacle_times_with_success()

        # Save to .pickle for caching
        if gl.use_boxcox and gl.reduced_features:
            pd.to_pickle(gl.obstacle_df_list, gl.working_directory_path + '/Pickle/reduced_features_boxcox/obstacle_df.pickle')
            pd.to_pickle(gl.df_list, gl.working_directory_path + '/Pickle/reduced_features_boxcox/df_list.pickle')
        elif gl.use_boxcox and not gl.reduced_features:
            pd.to_pickle(gl.obstacle_df_list, gl.working_directory_path + '/Pickle/all_features_boxcox/obstacle_df.pickle')
            pd.to_pickle(gl.df_list, gl.working_directory_path + '/Pickle/all_features_boxcox/df_list.pickle')
        elif not gl.use_boxcox and gl.reduced_features:
            pd.to_pickle(gl.obstacle_df_list, gl.working_directory_path + '/Pickle/reduced_features/obstacle_df.pickle')
            pd.to_pickle(gl.df_list, gl.working_directory_path + '/Pickle/reduced_features/df_list.pickle')
        elif not gl.use_boxcox and not gl.reduced_features:
            pd.to_pickle(gl.obstacle_df_list, gl.working_directory_path + '/Pickle/all_features/obstacle_df.pickle')
            pd.to_pickle(gl.df_list, gl.working_directory_path + '/Pickle/all_features/df_list.pickle')

        print('Dataframe created')
    if print_key_numbers:
        print_keynumbers_logfiles()


'''Reads the logfiles and parses them into Pandas dataframes. 
    Also adds additional log&timedelta column, cuts them to the same length and normalizes heartrate
'''


def read_and_prepare_logs():

    logs = [gl.abs_path_logfiles + "/" + s for s in gl.names_logfiles]

    # This read_csv is used when using the original logfiles
    # column_names = ['Time', 'Logtype', 'Gamemode', 'Points', 'Heartrate', 'physDifficulty',
    #                 'psyStress', 'psyDifficulty', 'obstacle']
    # gl.df_list = list(pd.read_csv(log, sep=';', skiprows=5, index_col=False, names=column_names) for log in logs)

    # This read_csv is used when using the refactored logs (Crash/Obstacle cleaned up)
    column_names = ['Time', 'Logtype', 'Gamemode', 'Points', 'Heartrate', 'physDifficulty',
                    'psyStress', 'psyDifficulty', 'obstacle', 'userID', 'logID']
    gl.df_list = list(pd.read_csv(log, sep=';', index_col=False, names=column_names) for log in logs)

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


