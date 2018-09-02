"""
This module is responsible for refactoring the logfiles, which is mostly only done the very first time the
project is running
"""

import os
import re
from pathlib import Path
from datetime import timedelta

import pandas as pd

import setup_dataframes as sd

_names_logfiles = []
_dataframes = []


def _sanity_check():
    """
    The original FK0410 have a 'm' or 'w' after its logfilename. remove it to have consistency among naming of the
    files and don't run in trouble later.

    In original text_logs, there are two users with MK initials (but different user_id).Rename it to MR

    In original text_logs, there is a username with non-capital kinect

    """

    path_logfiles_original = sd.project_path + '/Logs/text_logs_original/'

    nl = sorted([f for f in sorted(os.listdir(sd.project_path + '/Logs/text_logs_original'))
                if re.search(r'.log', f)])
    for idx, name in enumerate(nl):
        path = path_logfiles_original + name
        if 'FK0410' in name:
            new_name = name.replace('w', '')
            new_name = new_name.replace('m', '')
            new_name = new_name.replace('1_2', '2_1')
            os.rename(path, path_logfiles_original + new_name)
        if 'MK0902' in name:
            new_name = name.replace('MK', 'MR')
            os.rename(path, path_logfiles_original + new_name)
        if 'kinect' in name:
            new_name = name.replace('kinect', 'Kinect')
            os.rename(path, path_logfiles_original + new_name)


def cut_frames():
    """
    Cuts dataframes to the same length, namely to the shortest of the dataframes in the list
    Saves changes directly to globals.df_list

    """

    cutted_df_list = []
    min_time = min(dataframe['Time'].max() for dataframe in sd.df_list)

    for dataframe in sd.df_list:
        cutted_df_list.append(dataframe[dataframe['Time'] < min_time])

    sd.df_list = cutted_df_list


def add_timedelta_column():
    """
    For a lot of queries, it is useful to have the ['Time'] as a timedeltaIndex object
    Saves changes directly to globals.df_list

    """

    for idx, dataframe in enumerate(sd.df_list):
        new = dataframe['Time'].apply(lambda x: timedelta(seconds=x))
        sd.df_list[idx] = sd.df_list[idx].assign(timedelta=new)


def add_log_and_user_column():
    """
    Add log_number and user_id
    Saves changes directly to globals.df_list

    """

    global _dataframes, _names_logfiles

    last_name_abbr = _names_logfiles[0][:2]
    user_abbreviations = [f[:2] for f in _names_logfiles]
    user_id = 0

    for idx, dataframe in enumerate(_dataframes):
        name = _names_logfiles[idx]
        if not name[:2] == last_name_abbr:
            user_id += 1
        if not ('_' in name):  # No logfile number or sth.
            if not name[:2] in user_abbreviations[:idx]:
                log_id = '1'
            else:
                log_id = '2'
        else:
            log_id = name[-7]
        df = dataframe.assign(userID=user_id)
        df = df.assign(logID=log_id)
        _dataframes[idx] = df
        last_name_abbr = name[:2]


def refactor_crashes():
    """
    In the original logfiles, there is always an EVENT_CRASH and an EVENT_OBSTACLE in case of a crash, which makes it
    hard to analyze and use the data.
    Thus, in case of a crash, I remove the EVENT_OBSTACLE and move its obstacle information to the EVENT_CRASH log
    Additionally, I add a column with the userID and whether it's the first or second logfile of the user

    Input: Original logfiles

    Output: Refactored logfiles saved as new csv files in /text_logs_refactored

    """

    global _names_logfiles, _dataframes

    # If there was a crash, then there would be an 'EVENT_CRASH' in the preceding around 1 seconds of the event

    print('Refactoring crashes...')

    _sanity_check()

    _names_logfiles = sorted([f for f in sorted(os.listdir(sd.project_path + '/Logs/text_logs_original'))
                             if re.search(r'.log', f)])

    paths_logfiles = [sd.project_path + '/Logs/text_logs_original/' + name for name in _names_logfiles]

    # This read_csv is used when using the original logfiles
    column_names = ['Time', 'Logtype', 'Gamemode', 'Points', 'Heartrate', 'physDifficulty',
                    'psyStress', 'psyDifficulty', 'obstacle']

    _dataframes = list(pd.read_csv(log, sep=';', skiprows=5, index_col=False,
                                   names=column_names) for log in paths_logfiles)

    if not Path(sd.abs_path_logfiles).exists():
        os.mkdir(sd.abs_path_logfiles)

    add_log_and_user_column()

    def get_next_obstacle_row(index, df):
        cnt = 1
        while True:
            logtype = df.loc[index + cnt]['Logtype']
            if logtype == 'EVENT_OBSTACLE':
                return index + cnt, df.loc[index + cnt]
            cnt += 1

    for df_idx, dataframe in enumerate(_dataframes):
        old_name = _names_logfiles[df_idx]
        new_df = pd.DataFrame()
        count = 0
        obst_indices = []
        for idx, row in dataframe.iterrows():
            if not (idx in obst_indices):
                if row['Logtype'] == 'EVENT_CRASH':
                    if count == dataframe.index[-1]:
                        row['obstacle'] = ''
                    else:
                        obst_idx, obstacle_row = get_next_obstacle_row(idx, dataframe)
                        obst_indices.append(obst_idx)
                        row['obstacle'] = obstacle_row['obstacle']
                        row['Time'] = obstacle_row['Time']
                        dataframe.drop(obst_idx, inplace=True)
                new_df = new_df.append(row)

                count += 1

        new_df.reset_index(inplace=True, drop=True)
        column_names = ['Time', 'Logtype', 'Gamemode', 'Points', 'Heartrate', 'physDifficulty', 'psyStress',
                        'psyDifficulty', 'obstacle', 'userID', 'logID']

        new_df = new_df.reindex(column_names, axis=1)

        # Create new name
        new_name: str = old_name[:2]

        if 'Kinect' in old_name:
            new_name += '_Kinect'
        else:
            new_name += '_FBMC'
        hr_list = new_df['Heartrate']

        if not (((hr_list == -1).all()) or (len(set(hr_list)) == 1)):  # If -1 or const
            new_name += '_hr'

        if (new_df['Points'] == 0).all():
            new_name += '_np'
        new_name += '_' + new_df['logID'][0]

        import sys
        sys.stdout.write("\r{0}>".format("=" * df_idx))
        sys.stdout.flush()

        new_df.to_csv(sd.abs_path_logfiles + '/../text_logs_refactored/' + new_name + '.log', header=False,
                      index=False, sep=';')
