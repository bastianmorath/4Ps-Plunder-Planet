"""
This module is responsible to generate features from the data/logfiles

"""

import os
import math
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import setup_dataframes as sd
import synthesized_data

feature_names = []  # Set below

_verbose = True

hw = 20  # Over how many preceeding seconds should most of features such as min, max, mean of hr and points be averaged?
cw = 10  # Over how many preceeding seconds should %crashes be calculated?
gradient_w = 30  # Over how many preceeding seconds should hr features be calculated that have sth. do to with change?


_path_reduced_features = sd.working_directory_path + '/Pickle/reduced_features/'
_path_all_features = sd.working_directory_path + '/Pickle/all_features/'
_path_reduced_features_boxcox = sd.working_directory_path + '/Pickle/reduced_features_boxcox/'
_path_all_features_boxcox = sd.working_directory_path + '/Pickle/all_features_boxcox/'


def get_feature_matrix_and_label(verbose=True, use_cached_feature_matrix=True, save_as_pickle_file=False,
                                 use_boxcox=False, reduced_features=False,
                                 h_window=hw, c_window=cw, gradient_window=gradient_w):

    """
    Computes the feature matrix and the corresponding labels and creates a correlation_matrix

    :param verbose:                      Whether to print messages
    :param use_cached_feature_matrix:    Use already cached matrix; 'all' (use all features), 'selected'
                                            (do feature selection first), None (don't use cache)
    :param save_as_pickle_file:          If use_use_cached_feature_matrix=False, then store newly computed
                                            matrix in a pickle file (IMPORTANT: Usually only used the very first time to
                                            store feature matrix with e.g. default windows)
    :param use_boxcox:                   Whether boxcox transformation should be done (e.g. when Naive Bayes
                                            classifier is used)
    :param reduced_features:             Whether to do feature selection or not
    :param h_window:                     Size of heartrate window
    :param c_window:                     Size of crash window
    :param gradient_window:              Size of gradient window

    :return: Feature matrix, labels

    """

    for df in sd.df_list:
        assert (max(h_window, c_window, gradient_window) < max(df['Time'])),\
            'Window sizes must be smaller than maximal logfile length'

    globals()['hw'] = h_window
    globals()['cw'] = c_window
    globals()['gradient_w'] = gradient_window

    globals()['use_cached_feature_matrix'] = use_cached_feature_matrix
    globals()['_verbose'] = verbose

    matrix = pd.DataFrame()

    should_read_from_pickle_file, path = _should_read_from_cache(use_cached_feature_matrix, use_boxcox,
                                                                 reduced_features)

    sd.obstacle_df_list = sd.get_obstacle_times_with_success()

    if should_read_from_pickle_file:
        if _verbose:
            print('Feature matrix already cached!')

        matrix = pd.read_pickle(path)

    else:
        if _verbose:
            print('Creating feature matrix...')

        matrix['mean_hr'] = _get_standard_feature('mean', 'Heartrate')  # hw
        matrix['std_hr'] = _get_standard_feature('std', 'Heartrate')  # hw
        matrix['max_minus_min_hr'] = _get_standard_feature('max_minus_min', 'Heartrate')  # hw
        matrix['hr_gradient_changes'] = _get_number_of_gradient_changes('Heartrate')  # gradient_w
        matrix['lin_regression_hr_slope'] = _get_lin_regression_hr_slope_feature()  # gradient_w
        matrix['mean_score'] = _get_standard_feature('mean', 'Points')  # hw
        matrix['std_score'] = _get_standard_feature('std', 'Points')  # hw
        matrix['max_minus_min_score'] = _get_standard_feature('max_minus_min', 'Points')  # hw
        matrix['%crashes'] = _get_percentage_crashes_feature()  # cw
        matrix['last_obstacle_crash'] = _get_last_obstacle_crash_feature()  # cw
        matrix['timedelta_to_last_obst'] = _get_timedelta_to_last_obst_feature(do_normalize=False)

        if not reduced_features:

            matrix['max_hr'] = _get_standard_feature('max', 'Heartrate')  # hw
            matrix['min_hr'] = _get_standard_feature('min', 'Heartrate')  # hw
            matrix['max_over_min_hr'] = _get_standard_feature('max_over_min', 'Heartrate')  # hw
            matrix['max_score'] = _get_standard_feature('max', 'Points')  # hw
            matrix['min_score'] = _get_standard_feature('min', 'Points')  # hw
            matrix['score_gradient_changes'] = _get_number_of_gradient_changes('Points')  # gradient_w

        # Boxcox transformation
        if use_boxcox:
            # Values must be positive. If not, shift it
            non_boxcox = ['last_obstacle_crash']
            for feature in feature_names:
                if feature not in non_boxcox:  # Doesn't makes sense to do boxcox here
                    if matrix[feature].min() <= 0:
                        matrix[feature] = stats.boxcox(matrix[feature] - matrix[feature].min() + 0.01)[0]
                    else:
                        matrix[feature] = stats.boxcox(matrix[feature])[0]

        if save_as_pickle_file and (not sd.use_fewer_data):
            matrix.to_pickle(path)

    # remove ~ first couple of seconds (they have < window seconds to compute features, and are thus not accurate)
    labels = []
    for df in sd.obstacle_df_list:
        labels.append(df[df['Time'] > max(cw, hw, gradient_w)]['crash'].copy())
    y = list(itertools.chain.from_iterable(labels))
    np.set_printoptions(suppress=True)

    matrix.dropna(inplace=True)  # First max(hw, cw, gradient_w) seconds did not get computed since inaccurate -> Delete

    globals()['feature_names'] = list(matrix)

    # Create feature matrix from df
    X = matrix.values

    if verbose:
        print('Feature matrix and labels created!')

    return X, y


def _get_timedelta_to_last_obst_feature(do_normalize=False):
    """
    Returns the timedelta to the previous obstacle

    :param do_normalize: Normalize the timedelta with previous timedelta (bc. it varies slightly within and across
                         logfiles)

    """

    timedeltas_df_list = []  # list that contains a dataframe with feature for each logfile
    computed_timedeltas = []
    if _verbose:
        print('Creating timedelta_to_last_obst feature...')

    def compute(row):
        if row['Time'] > max(cw, hw, gradient_w):
            last_obstacles = df[(df['Time'] < row['Time']) & ((df['Logtype'] == 'EVENT_OBSTACLE') |
                                                              (df['Logtype'] == 'EVENT_CRASH'))]
            if last_obstacles.empty:
                computed_timedeltas.append(2.2)
                return 1

            timedelta = row['Time'] - last_obstacles.iloc[-1]['Time']
            # Clamp outliers (e.g. because of tutorials etc.). If timedelta >3, it's most likely e.g 33 seconds, so I
            # clamp to c.a. the average/last timedelta
            if timedelta > 3 or timedelta < 1:
                if len(computed_timedeltas) > 0:
                    timedelta = computed_timedeltas[-1]
                else:
                    timedelta = 2

            # AdaBoost: 2 or 3 is best
            # Random Forest: 1 is best
            last_n_obst = min(len(computed_timedeltas), 1)
            if len(computed_timedeltas) > 0:
                normalized = timedelta / np.mean(computed_timedeltas[-last_n_obst:])
            else:
                normalized = 1

            computed_timedeltas.append(timedelta)

            return normalized if do_normalize else timedelta

    for list_idx, df in enumerate(sd.df_list):
        timedeltas_df_list.append(sd.obstacle_df_list[list_idx].apply(compute, axis=1))
        computed_timedeltas = []

    return pd.DataFrame(list(itertools.chain.from_iterable(timedeltas_df_list)), columns=['timedelta_to_last_obst'])


def _get_standard_feature(feature, data_name):
    """
    This is a wrapper to compute common features such as min, max, mean for either Points or Heartrate

    :param feature: min, max, mean, std
    :param data_name: Either Heartrate or Points

    :return: Dataframe column containing the feature

    """

    if _verbose:
        print('Creating ' + feature + '_' + data_name + ' feature...')

    hr_df_list = []  # list that contains a dataframe with feature for each logfile
    for list_idx, df in enumerate(sd.df_list):
        hr_df = _get_column(list_idx, feature, data_name)
        hr_df_list.append(hr_df)

    return pd.DataFrame(list(itertools.chain.from_iterable(hr_df_list)), columns=[feature])


def _get_percentage_crashes_feature():
    if _verbose:
        print('Creating %crashes feature...')

    crashes_list = []  # list that contains one dataframe with %crashes for each point in time for each logfile
    for list_idx, df in enumerate(sd.df_list):
        crashes = _get_percentage_crashes_column(list_idx)
        crashes_list.append(crashes)

    return pd.DataFrame(list(itertools.chain.from_iterable(crashes_list)), columns=['%crashes'])


def _get_last_obstacle_crash_feature():
    if _verbose:
        print('Creating last_obstacle_crash feature...')
    # list that contains one dataframe with whether last obstacle was a crash or not
    # for each point in time for each logfile
    crashes_list = []

    for list_idx, df in enumerate(sd.df_list):
        df_obstacles = _get_last_obstacle_crash_column(list_idx)
        crashes_list.append(df_obstacles)

    return pd.DataFrame(list(itertools.chain.from_iterable(crashes_list)), columns=['last_obstacle_crash'])


def _get_lin_regression_hr_slope_feature():
    if _verbose:
        print('Creating lin_regression_hr_slope feature...')
    slopes = []  # list that contains (for each logfile) a dataframe with the slope of the heartrate

    for list_idx, df in enumerate(sd.df_list):
        slope = _get_hr_slope_column(list_idx)
        slopes.append(slope)

    return pd.DataFrame(list(itertools.chain.from_iterable(slopes)), columns=['lin_regression_hr_slope'])


def _get_number_of_gradient_changes(data_name):
    if _verbose:
        print('Creating %s_gradient_changes feature...' % data_name)
    changes_list = []  # list that contains (for each logfile) a dataframe with the number of slope changes

    for list_idx, df in enumerate(sd.df_list):
        changes = _get_gradient_changes_column(list_idx, data_name)
        changes_list.append(changes)

    if data_name == 'Points':
        return pd.DataFrame(list(itertools.chain.from_iterable(changes_list)), columns=['score_gradient_changes'])
    else:
        return pd.DataFrame(list(itertools.chain.from_iterable(changes_list)), columns=['hr_gradient_changes'])


"""
The following methods calculate the features of one single dataframe and return it as a new dataframe column

"""


def _df_from_to(_from, _to, df):
    """
    Returns the part of the dataframe where time is between _from and _to

    :param _from: Start of dataframe ['Time']
    :param _to: End of dataframe ['Time']
    :param df: Dataframe

    :return: new Dataframe where row['Time'] between _from and _to

    """

    mask = (_from <= df['Time']) & (df['Time'] < _to)
    return df[mask]


def _get_column(idx, applier, data_name):
    """
    This is a wrapper which returns a dataframe column that indicates at each timestamp the
    heartrate or points over the last 'window' seconds, after applying 'applyer' (e.g. mean, max, min)

    :param idx: Index of dataframe in gl.df_list
    :param applier: mean, min, max, std
    :param data_name: Heartrate or Points

    :return: Dataframe column with feature

    """

    df = sd.df_list[idx]

    window = hw

    def compute(row):
        if row['Time'] > max(cw, hw, gradient_w):
            last_x_seconds_df = _df_from_to(max(0, row['Time'] - window), row['Time'], df)
            res = -1
            if applier == 'mean':
                res = last_x_seconds_df[data_name].mean()
            elif applier == 'min':
                res = last_x_seconds_df[data_name].min()
            elif applier == 'max':
                res = last_x_seconds_df[data_name].max()
            elif applier == 'std':
                res = last_x_seconds_df[data_name].std()
            elif applier == 'max_minus_min':
                last_x_seconds_df = _df_from_to(max(0, row['Time'] - gradient_w), row['Time'], df)
                max_v = last_x_seconds_df[data_name].max()
                min_v = last_x_seconds_df[data_name].min()
                res = max_v - min_v
            elif applier == 'max_over_min':
                last_x_seconds_df = _df_from_to(max(0, row['Time'] - gradient_w), row['Time'], df)
                max_v = last_x_seconds_df[data_name].max()
                min_v = last_x_seconds_df[data_name].min()
                res = max_v / min_v
            if res == -1:
                print('error in applying ' + data_name + '_' + applier)

            # first mean will be nan, so replace it with second row instead
            return res if not math.isnan(res) else compute(df.iloc[1])

    return sd.obstacle_df_list[idx].apply(compute, axis=1)


def _get_percentage_crashes_column(idx):
    """
    Returns a dataframe column that indicates at each timestamp how many percentage of the last obstacles in the
    last crash-window-seconds the user crashed into

    :param idx: Index into gl.df_list (indicates the dataframe)

    :return: Percentage feature column

    """

    df = sd.df_list[idx]

    '''
    # Scale feature depending on timedelta (the shorter, the more difficult...
    def get_factor(timedelta):
        if timedelta < 2:
            return 0.8
        if 2 <= timedelta < 3:
            return 1.2
        else:
            return 1
    '''
    def compute_crashes(row):
        if row['Time'] > max(cw, hw, gradient_w):
            last_x_seconds_df = _df_from_to(max(0, row['Time'] - cw), row['Time'], df)
            num_obstacles = len(last_x_seconds_df[(last_x_seconds_df['Logtype'] == 'EVENT_OBSTACLE')
                                                  | (last_x_seconds_df['Logtype'] == 'EVENT_CRASH')].index)
            num_crashes = len(last_x_seconds_df[last_x_seconds_df['Logtype'] == 'EVENT_CRASH'].index)

            return (num_crashes/num_obstacles * 100 if num_crashes < num_obstacles else 100) if num_obstacles != 0 \
                else 0

    return sd.obstacle_df_list[idx].apply(compute_crashes, axis=1)


def _get_last_obstacle_crash_column(idx):
    """
    Returns a dataframe column that indicates at each timestamp whether the user crashed into the last obstacle or not

    :param idx: Index into gl.df_list (indicates the dataframe)

    :return: last_obstacle_crash feature column

    """

    df = sd.df_list[idx]

    def compute_crashes(row):
        if row['Time'] > max(cw, hw, gradient_w):

            last_obstacles = df[(df['Time'] < row['Time']) & ((df['Logtype'] == 'EVENT_OBSTACLE') |
                                                              (df['Logtype'] == 'EVENT_CRASH'))]
            if last_obstacles.empty:
                return 0
            return 1 if last_obstacles.iloc[-1]['Logtype'] == 'EVENT_CRASH' else 0

    return sd.obstacle_df_list[idx].apply(compute_crashes, axis=1)


def _get_hr_slope_column(idx):
    """
    Returns a dataframe column that indicates at each timestamp the slope of the fitting lin/ regression
    line over the heartrate in the last hw seconds

    :param idx: Index into gl.df_list (indicates the dataframe)

    :return: hr_slope feature column

    """

    df = sd.df_list[idx]

    # noinspection PyTupleAssignmentBalance
    def compute_slope(row):
        if row['Time'] > max(cw, hw, gradient_w):
            last_x_seconds_df = _df_from_to(max(0, row['Time'] - gradient_w), row['Time'], df)

            slope, _ = np.polyfit(last_x_seconds_df['Time'], last_x_seconds_df['Heartrate'], 1)
            return slope if not math.isnan(slope) else compute_slope(df.iloc[1])

    return sd.obstacle_df_list[idx].apply(compute_slope, axis=1)


def _get_gradient_changes_column(idx, data_name):
    """
    Returns a dataframe column that indicates at each timestamp the number of times 'data_name' (points or Heartrate)
    have changed from increasing to decreasing and the other way around

    :param idx: Index into gl.df_list (indicates the dataframe)
    :param data_name: Points or Heartrate

    :return: gradient_changes feature column for either points or heartrate

    """

    df = sd.df_list[idx]

    def compute_gradient_changes(row):
        if row['Time'] > max(cw, hw, gradient_w):

            last_x_seconds_df = _df_from_to(max(0, row['Time'] - cw), row['Time'], df)
            data = last_x_seconds_df[data_name].tolist()
            gradx = np.gradient(data)
            asign = np.sign(gradx)

            num_sign_changes = len(list(itertools.groupby(asign, lambda x: x >= 0))) - 1
            if num_sign_changes == 0:
                num_sign_changes = 1
            return num_sign_changes if not math.isnan(num_sign_changes) else compute_gradient_changes(df.iloc[1])

    return sd.obstacle_df_list[idx].apply(compute_gradient_changes, axis=1)


"""
Helper functions

"""


def _should_read_from_cache(use_cached_feature_matrix, use_boxcox, reduced_features):
    """
    If the user wants to use an already saved feature matrix ('all' or 'reduced'), then check if those
    pickle files really exist. If not, new files have to be created


    :param use_cached_feature_matrix: Use already cached matrix; 'all' (use all features), 'selected'
                                (do feature selection first), None (don't use cache)
    :param use_boxcox: Whether boxcox transofrmation should be done (e.g. when Naive Bayes classifier is used)
    :param reduced_features: Whether to do feature selection or not

    :return: Whether reading from cache is okey and  path where to read from/write to new pickel file (if necessary)

    """

    err_string = 'ERROR: Pickle file of Feature matrix not yet created. Creating new one...'
    path = ''
    file_name = 'feature_matrix_%s_%s_%s.pickle' % (hw, cw, gradient_w)
    if not reduced_features:
        if use_boxcox:
            path = _path_all_features_boxcox
        else:
            path = _path_all_features

    elif reduced_features:
        if use_boxcox:
            path = _path_reduced_features_boxcox
        else:
            path = _path_reduced_features

    file_path = path + file_name

    if not use_cached_feature_matrix or sd.use_fewer_data or synthesized_data.synthesized_data_enabled:
        return False, file_path
    else:
        if not Path(file_path).exists():
            print(err_string)
            if not Path(path).exists():  # Check if at least the folder exists
                os.makedirs(path)
            return False, file_path
        else:
            return True, file_path
