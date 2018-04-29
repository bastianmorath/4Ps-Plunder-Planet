
import os
import re

import pandas as pd
import pickle
import refactoring_logfiles as refactoring

import globals as gl
import factory


def setup():

    read_and_prepare_logs()

    # Store computed dataframe in pickle file for faster processing
    if gl.use_cache and os.path.isfile(gl.working_directory_path + '/Pickle/df_list.pickle'):
        print('Dataframe already cached. Used this file to improve performance')
        gl.obstacle_df_list = pickle.load(open(gl.working_directory_path + '/Pickle/obstacle_df.pickle', "rb"))
        gl.df_list = pickle.load(open(gl.working_directory_path + '/Pickle/df_list.pickle', "rb"))
    else:
        print('Dataframe not cached. Creating dataframe...')
        gl.obstacle_df_list = factory.get_obstacle_times_with_success()

        # Save to .pickle for caching
        pickle.dump(gl.obstacle_df_list, open(gl.working_directory_path + '/Pickle/obstacle_df.pickle', "wb"))
        pickle.dump(gl.df_list, open(gl.working_directory_path + '/Pickle/df_list.pickle', "wb"))

        print('Dataframe created')

    # factory.print_keynumbers_logfiles()


'''Reads the logfiles and parses them into Pandas dataframes. 
    Also adds additional log&timedelta column, cuts them to the same length and normalizes heartrate
'''


def read_and_prepare_logs():
    kinect_names = [f for f in sorted(os.listdir(gl.abs_path_logfiles)) if re.search(r'.{0,}Kinect.{0,}.log', f)]
    all_names = [f for f in sorted(os.listdir(gl.abs_path_logfiles)) if re.search(r'.log', f)]
    fbmc_names = [f for f in sorted(os.listdir(gl.abs_path_logfiles)) if re.search(r'.{0,}FBMC_hr.{0,}.log', f)]

    gl.names_logfiles = all_names
    print(len(gl.names_logfiles))

    logs = [gl.abs_path_logfiles + "/" + s for s in gl.names_logfiles]
    column_names = ['Time', 'Logtype', 'Gamemode', 'Points', 'Heartrate', 'physDifficulty', 'psyStress', 'psyDifficulty', 'obstacle']
    # This read_csv is used when using the original logfiles
    gl.df_list = list(pd.read_csv(log, sep=';', skiprows=5, index_col=False, names=column_names) for log in logs)

    # This read_csv is used when using the refactored logs (Crash/Obstacle cleaned up)
    # gl.df_list = list(pd.read_csv(log, sep=';', index_col=False, names=column_names) for log in logs)
    if gl.testing:
        gl.df_list = gl.df_list[4:6]

    # NOTE: Has only ever to be called once to refactore logs
    refactoring.refactor_crashes()
    refactoring.remove_logs_without_heartrates_or_points()
    refactoring.normalize_heartrate()



