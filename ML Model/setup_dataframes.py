"""
This module reads the logfiles and transforms them into dataframes.

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
from pathlib import Path

import numpy as np
import pandas as pd

import refactoring_logfiles as refactoring

use_fewer_data = False  # Can be used for debugging (fewer data is used)

working_directory_path = os.path.abspath(os.path.dirname(__file__))
project_path = os.path.abspath(os.path.join(working_directory_path, "../../.."))

abs_path_logfiles = project_path + "/Logs/text_logs_refactored"  # Logfiles to use
names_logfiles = []  # Name of the logfiles

df_list = []  # List with all dataframes; 1 dataframe per logfile

# list of dataframes, each dataframe has time of each obstacle and whether crash or not (1 df per logfile)
# First max(cw, hw) seconds removed
obstacle_df_list = []

print_key_numbers = False


def setup(fewer_data=False, normalize_heartrate=True):
    """ Sets up the logfile datastructures/lists and does some basic refactoring
    
    Note: No machine learning specific things are done yet!

    :param use_fewer_data: Whether we should only use a little part of the data or not (helps for faster debugging)
    :param normalize_heartrate: Whether we should normalize heartrate or not
    """

    print("Loading dataframes...")

    if not Path(abs_path_logfiles).exists():
        # The very first time, we need to refactor the original logfiles to speed up everything afterwards
        refactoring.refactor_crashes()

    all_names = [
        f for f in sorted(os.listdir(abs_path_logfiles)) if re.search(r".log", f)
    ]

    kinect_names_hr = [
        f
        for f in sorted(os.listdir(abs_path_logfiles))
        if re.search(r".{0,}Kinect_hr.{0,}.log", f)
    ]
    
    fbmc_names_hr_points = [
        f
        for f in sorted(os.listdir(abs_path_logfiles))
        if re.search(r".{0,}FBMC_hr_(1|2).{0,}.log", f)  # TODO: Stefan is not considered..
    ]

    sorted_names = sorted(fbmc_names_hr_points)

    globals()["names_logfiles"] = sorted_names
    globals()["use_fewer_data"] = fewer_data

    if fewer_data:
        # globals()["names_logfiles"] = ['Is_FBMC_hr_1.log', 'Lo_FBMC_hr_1.log', 'MH_FBMC_hr_1.log']
        globals()["names_logfiles"] = ['TW_FBMC_hr_1.log']

    column_names = [
        "Time",
        "Logtype",
        "Gamemode",
        "Points",
        "Heartrate",
        "physDifficulty",
        "psyStress",
        "psyDifficulty",
        "obstacle",
        "userID",
        "logID",
    ]

    logs = [abs_path_logfiles + "/" + s for s in names_logfiles]

    globals()["df_list"] = list(
        pd.read_csv(log, sep=";", index_col=False, names=column_names) for log in logs
    )

    if normalize_heartrate:
        normalize_heartrate_of_logfiles()

    remove_movement_tutorials()

    refactoring.add_timedelta_column()

    '''
    len_before = len(pd.concat(df_list, ignore_index=True))
    refactoring.cut_frames()
    len_after = len(pd.concat(df_list, ignore_index=True))
    print('Dataframes cutted to same length: Lost ' + str(round((1.0 - len_after/len_before) * 100, 3))
          + ' percent of data')
    '''


def print_keynumbers_logfiles():
    """ Prints important numbers of the logfiles:
        - number of logs, events, features (=obstacles)
        - number of files that contain heartrate vs no heartrate
    """

    print("\nImportant numbers about the logfiles:")
    print("\t#files: " + str(len(df_list)))
    print(
        "\t#files with heartrate: "
        + str(len([a for a in df_list if not (a["Heartrate"] == -1).all()]))
    )
    print("\t#datapoints: " + str(sum([len(a.index) for a in df_list])))
    print("\t#obstacles: " + str(sum([len(df.index) for df in obstacle_df_list])))
    print(
        "\t#crashes: " + str(sum([len(df[df["crash"] == 1]) for df in obstacle_df_list]))
    )

    df_lengths = []
    for d in df_list:
        df_lengths.append(d["Time"].max())
    print(
        "\taverage length: "
        + str(round(np.mean(df_lengths), 2))
        + ", std: "
        + str(round(np.std(df_lengths), 2))
        + ", max: "
        + str(round(np.max(df_lengths), 2))
        + ", min: "
        + str(round(np.min(df_lengths), 2))
    )


def get_obstacle_times_with_success():
    """Returns a list of dataframes, each dataframe has time of each obstacle and whether crash or not (1 df per logfile)
    First max(f_factory.cw, f_factory.hw) seconds are removed

    userID/logID is used such that we can later (after creating feature matrix), still differentiate logfiles

    :return: List which contains a dataframe df per log. Each df contains the time, userID, logID and whether the user
                crashed (of each obstacle)

    """

    import features_factory as f_factory  # To avoid circular dependency

    obstacle_time_crash = []

    for dataframe in df_list:
        obstacle_times_current_df = []
        for idx, row in dataframe.iterrows():
            if row["Time"] > max(f_factory.cw, f_factory.hw, f_factory.gradient_w):
                if row["Logtype"] == "EVENT_OBSTACLE":
                    obstacle_times_current_df.append(
                        (row["Time"], 0, row["userID"], row["logID"])
                    )
                if row["Logtype"] == "EVENT_CRASH":
                    obstacle_times_current_df.append(
                        (row["Time"], 1, row["userID"], row["logID"])
                    )
        times = np.asarray([a for (a, b, c, d) in obstacle_times_current_df])
        crashes = np.asarray([b for (a, b, c, d) in obstacle_times_current_df])
        userIDs = np.asarray([c for (a, b, c, d) in obstacle_times_current_df])
        logIDs = np.asarray([d for (a, b, c, d) in obstacle_times_current_df])

        obstacle_time_crash.append(
            pd.DataFrame(
                {"Time": times, "crash": crashes, "userID": userIDs, "logID": logIDs}
            )
        )

    return obstacle_time_crash


def normalize_heartrate_of_logfiles():
    """Normalizes heartrate of each dataframe/user by dividing by min of the movementtutorial.
        Saves changes directly to globals.df_list

        Note: I didn't do this on the refactoring-step on purpose, since the user might want to
        do some plots, which might be more convenient to do with non-normalized heartrate data
    """

    normalized_df_list = []
    for dataframe in globals()["df_list"]:
        if 'MOVEMENTTUTORIAL' in dataframe['Gamemode'].values:
            # Remove movement tutorial
            tutorial_mask = dataframe['Gamemode'] == 'MOVEMENTTUTORIAL'
            tutorial_entries = dataframe[tutorial_mask]
            tutorial_endtime = tutorial_entries['Time'].max()

            baseline = dataframe[dataframe["Time"] < tutorial_endtime]["Heartrate"].min() # Use MINIMUM of tutorial 
            if baseline == -1:
                print('ERROR: No Heartrate data!!!')

            dataframe["Heartrate"] = dataframe["Heartrate"] / baseline

            normalized_df_list.append(dataframe)
        else:
            print('ERROR: No Movement tutorial')

    globals()["df_list"] = normalized_df_list


def remove_movement_tutorials():
    """Since the MOVEMENTTUTORIAL at the beginning is different for each player/logfile,
        we need to align them, i.e. remove tutorial-log entries and then set timer to 0

    """

    dataframe_list_with_tutorials_removed = []
    for dataframe in globals()["df_list"]:
        if 'MOVEMENTTUTORIAL' in dataframe['Gamemode'].values:
            # Remove movement tutorial
            tutorial_mask = dataframe['Gamemode'] == 'MOVEMENTTUTORIAL'
            tutorial_entries = dataframe[tutorial_mask]
            tutorial_endtime = tutorial_entries['Time'].max()
            dataframe['Time'] = dataframe['Time'].apply(lambda x: x - tutorial_endtime) # Adjust time by removing time of tutorial

            dataframe_list_with_tutorials_removed.append(dataframe[~tutorial_mask].reset_index(drop=True))
        else:
            print('ERROR: No Movement tutorial')

    globals()["df_list"] = dataframe_list_with_tutorials_removed
   

def remove_logs_without_heartrates_or_points():
    """Removes all logfiles that do not have any heartrate data or points (since we then can't calculate our features)
        Saves changes directly to globals.df_list

    """

    globals()["df_list"] = [df for df in df_list if not (df["Heartrate"] == -1).all()]
    globals()["df_list"] = [df for df in df_list if not (df["Points"] == 0).all()]
