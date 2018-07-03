"""
This module is responsible to generate all the features from the data/logfiles

"""

import pandas as pd
import math
import os
from pathlib import Path

from scipy import stats
import numpy as np
import itertools
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import setup_dataframes as sd
import plots
import synthesized_data

"""INITIALIZATION"""

feature_names = []  # Set below
use_reduced_features = True

_verbose = True

# TODO: Explicitly write down which features use which window
# TODO: Onlt if not_reduced, add if-clauses. But we can always add the shared features...
# TODO: Simplify feature_selection: Store is at variable in this class and use this always (without argument passing)


hw = 3  # Over how many preceeding seconds should most of features such as min, max, mean of hr and points be averaged?
cw = 3  # Over how many preceeding seconds should %crashes be calculated?
gradient_w = 3  # Over how many preceeding seconds should hr features be calculated that have sth. do to with change?


path_reduced_features = sd.working_directory_path + '/Pickle/reduced_features/'
path_all_features = sd.working_directory_path + '/Pickle/all_features/'
path_reduced_features_boxcox = sd.working_directory_path + '/Pickle/reduced_features_boxcox/'
path_all_features_boxcox = sd.working_directory_path + '/Pickle/all_features_boxcox/'


def should_read_from_cache(use_cached_feature_matrix, use_boxcox, feature_selection):
    """ If the user wants to use an already saved feature matrix ('all' or 'reduced'), then check if those
    pickle files really exist. If not, new files have to be created


    :param use_cached_feature_matrix: Use already cached matrix; 'all' (use all features), 'selected'
                                (do feature selection first), None (don't use cache)
    :param use_boxcox: Whether boxcox transofrmation should be done (e.g. when Naive Bayes classifier is used)
    :param feature_selection: Whether to do feature selection or not

    :return: Whether reading from cache is okey and  path where to read from/write to new pickel file (if necessary)

    """
    err_string = 'ERROR: Pickle file of Feature matrix not yet created. Creating new one...'
    path = ''
    file_name = 'feature_matrix_%s_%s_%s.pickle' % (hw, cw, gradient_w)
    if not feature_selection:
        if use_boxcox:
            path = path_all_features_boxcox
        else:
            path = path_all_features

    elif feature_selection:
        if use_boxcox:
            path = path_reduced_features_boxcox
        else:
            path = path_reduced_features

    file_path = path + file_name

    if not use_cached_feature_matrix or sd.use_fewer_data or synthesized_data.test_data_enabled:
        return False, file_path
    else:
        if not Path(file_path).exists():
            print(err_string)
            if not Path(path).exists():  # Check if at least the folder exists
                os.makedirs(path)
            return False, file_path
        else:
            return True, file_path


def get_feature_matrix_and_label(verbose=True, use_cached_feature_matrix=True, save_as_pickle_file=False,
                                 use_boxcox=False, feature_selection=True,
                                 h_window=hw, c_window=cw, gradient_window=gradient_w):

    """ Computes the feature matrix and the corresponding labels and creates a correlation_matrix

    :param verbose:                      Whether to print messages
    :param use_cached_feature_matrix:    Use already cached matrix; 'all' (use all features), 'selected'
                                            (do feature selection first), None (don't use cache)
    :param save_as_pickle_file:          if use_use_cached_feature_matrix=False, then store newly computed
                                            matrix in a pickle file (IMPORTANT: Usually only used the very first time to
                                            store feature matrix with e.g. default windows)
    :param use_boxcox:                   Whether boxcox transformation should be done (e.g. when Naive Bayes
                                            classifier is used)
    :param feature_selection:            Whether to do feature selection or not
    :param h_window:                     Size of heartrate window
    :param c_window:                     Size of crash window
    :param gradient_window:              Size of gradient window

    :return: Feature matrix and labels

    """
    for df in sd.df_list:
        assert (max(h_window, c_window, gradient_window) < max(df['Time'])),\
            'Window sizes must be smaller than maximal logfile length'

    globals()['hw'] = h_window
    globals()['cw'] = c_window
    globals()['gradient_w'] = gradient_window

    globals()['use_cached_feature_matrix'] = use_cached_feature_matrix
    globals()['use_reduced_features'] = feature_selection
    globals()['_verbose'] = verbose

    if feature_selection:
        globals()['feature_names'] = ['last_obstacle_crash', 'timedelta_last_obst']
    else:
        globals()['feature_names'] = ['last_obstacle_crash', 'timedelta_last_obst', 'mean_hr', 'std_hr',
                                      'max_minus_min_hr', 'lin_regression_hr_slope', 'hr_gradient_changes', '%crashes',
                                      'points_gradient_changes', 'mean_points', 'std_points', 'max_hr', 'min_hr',
                                      'max_over_min_hr', 'max_points', 'min_points', 'max_minus_min_points']
    matrix = pd.DataFrame()

    should_read_from_pickle_file, path = should_read_from_cache(use_cached_feature_matrix, use_boxcox, feature_selection)

    sd.obstacle_df_list = sd.get_obstacle_times_with_success()

    if should_read_from_pickle_file:
        if _verbose:
            print('Feature matrix already cached!')
        matrix = pd.read_pickle(path)
    else:
        if _verbose:
            print('Creating feature matrix...')

        matrix['last_obstacle_crash'] = get_last_obstacle_crash_feature()
        matrix['timedelta_last_obst'] = get_timedelta_last_obst_feature(do_normalize=False)

        if not use_reduced_features:
            matrix['mean_hr'] = get_standard_feature('mean', 'Heartrate')
            matrix['std_hr'] = get_standard_feature('std', 'Heartrate')
            matrix['max_minus_min_hr'] = get_standard_feature('max_minus_min', 'Heartrate')
            matrix['lin_regression_hr_slope'] = get_lin_regression_hr_slope_feature()
            matrix['hr_gradient_changes'] = get_number_of_gradient_changes('Heartrate')

            matrix['%crashes'] = get_percentage_crashes_feature()

            matrix['points_gradient_changes'] = get_number_of_gradient_changes('Points')
            matrix['mean_points'] = get_standard_feature('mean', 'Points')
            matrix['std_points'] = get_standard_feature('std', 'Points')
            matrix['max_hr'] = get_standard_feature('max', 'Heartrate')
            matrix['min_hr'] = get_standard_feature('min', 'Heartrate')
            matrix['max_over_min_hr'] = get_standard_feature('max_over_min', 'Heartrate')
            matrix['last_obstacle_crash'] = get_last_obstacle_crash_feature()
            matrix['max_points'] = get_standard_feature('max', 'Points')
            matrix['min_points'] = get_standard_feature('min', 'Points')
            matrix['max_minus_min_points'] = get_standard_feature('max_minus_min', 'Points')

        # Boxcox transformation
        if use_boxcox:
            # Values must be positive. If not, shift it
            for feature in feature_names:
                if not feature == 'last_obstacle_crash':  # Doesn't makes sense to do boxcox here
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

    # Create feature matrix from df
    X = matrix.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)  # Rescale between 0 and 1



    plots.plot_correlation_matrix(matrix)
    if verbose:
        print('Feature matrix and labels created!')

    return X, y


def get_timedelta_last_obst_feature(do_normalize=False):
    """ Returns the timedelta to the previous obstacle

    NOTE: Significantly improves SVM/Linear SVM (from ca. auc=0.6 to 0.9) and Ada Boost (from auc=0.73 to 0.87
    :param do_normalize: Normalize the timedelta with previous timedelta (bc. it varies slightly within and across
                         logfiles)



    """
    timedeltas_df_list = []  # list that contains a dataframe with feature for each logfile
    computed_timedeltas = []

    def compute(row):
        if row['Time'] > max(cw, hw, gradient_w):
            last_obstacles = df[(df['Time'] < row['Time']) & ((df['Logtype'] == 'EVENT_OBSTACLE') |
                                                                 (df['Logtype'] == 'EVENT_CRASH'))]
            if last_obstacles.empty:
                computed_timedeltas.append(2.2)
                return 1

            timedelta = row['Time'] - last_obstacles.iloc[-1]['Time']
            # Clamp outliers (e.g. because of tutorials etc.). If timedelta >3, it's most likely e.g 33 seconds, so I
            # clamp to c.a. the average
            if timedelta > 3 or timedelta < 1:
                timedelta = 2

            # AdaBoost: 2 or 3 is best
            # Random Forest: 1 is best
            last_n_obst = min(len(computed_timedeltas), 1)
            if len(computed_timedeltas) > 0:
                normalized = timedelta / np.mean(computed_timedeltas[-last_n_obst:])
            else:
                normalized = 1

            # print(normalized, np.mean(computed_timedeltas[-last_n_obst:]), timedelta)

            computed_timedeltas.append(timedelta)

            return normalized if do_normalize else timedelta

    for list_idx, df in enumerate(sd.df_list):
        timedeltas_df_list.append(sd.obstacle_df_list[list_idx].apply(compute, axis=1))
        computed_timedeltas = []

    return pd.DataFrame(list(itertools.chain.from_iterable(timedeltas_df_list)), columns=['timedelta_last_obst'])


def get_standard_feature(feature, data_name):
    """This is a wrapper to compute common features such as min, max, mean for either Points or Heartrate

    :param feature: min, max, mean, std
    :param data_name: Either Heartrate or Points

    :return: Dataframe column containing the feature

    """
    if _verbose:
        print('Creating ' + feature + '_' + data_name + ' feature...')

    hr_df_list = []  # list that contains a dataframe with feature for each logfile
    for list_idx, df in enumerate(sd.df_list):
        hr_df = get_column(list_idx, feature, data_name)
        hr_df_list.append(hr_df)

    return pd.DataFrame(list(itertools.chain.from_iterable(hr_df_list)), columns=[feature])


def get_percentage_crashes_feature():
    # TODO: Normalize crashes depending on size/assembly of the obstacle

    if _verbose:
        print('Creating %crashes feature...')

    crashes_list = []  # list that contains one dataframe with %crashes for each point in time for each logfile
    for list_idx, df in enumerate(sd.df_list):
        crashes = get_percentage_crashes_column(list_idx)
        crashes_list.append(crashes)

    return pd.DataFrame(list(itertools.chain.from_iterable(crashes_list)), columns=['%crashes'])


def get_last_obstacle_crash_feature():
    if _verbose:
        print('Creating last_obstacle_crash feature...')
    # list that contains one dataframe with whether last obstacle was a crash or not
    # for each point in time for each logfile
    crashes_list = []

    for list_idx, df in enumerate(sd.df_list):
        df_obstacles = get_last_obstacle_crash_column(list_idx)
        # df = df[df['Time'] > max(cw, hw, gradient_w)]  # remove first window-seconds bc. not accurate data
        crashes_list.append(df_obstacles)

    return pd.DataFrame(list(itertools.chain.from_iterable(crashes_list)), columns=['last_obstacle_crash'])


def get_lin_regression_hr_slope_feature():
    if _verbose:
        print('Creating lin_regression_hr_slope feature...')
    slopes = []  # list that contains (for each logfile) a dataframe with the slope of the heartrate

    for list_idx, df in enumerate(sd.df_list):
        slope = get_hr_slope_column(list_idx)
        slopes.append(slope)

    return pd.DataFrame(list(itertools.chain.from_iterable(slopes)), columns=['lin_regression_hr_slope'])


def get_number_of_gradient_changes(data_name):
    if _verbose:
        print('Creating %s_gradient_changes feature...' % data_name)
    changes_list = []  # list that contains (for each logfile) a dataframe with the number of slope changes

    for list_idx, df in enumerate(sd.df_list):
        changes = get_gradient_changes_column(list_idx, data_name)
        changes_list.append(changes)

    if data_name == 'Points':
        return pd.DataFrame(list(itertools.chain.from_iterable(changes_list)), columns=['points_gradient_changes'])
    else:
        return pd.DataFrame(list(itertools.chain.from_iterable(changes_list)), columns=['hr_gradient_changes'])


"""The following methods calculate the features as a new dataframe column"""


def df_from_to(_from, _to, df):
    """Returns the part of the dataframe where time is between _from and _to

    :param _from: Start of dataframe ['Time']
    :param _to: End of dataframe ['Time']
    :param df: Dataframe

    :return: new Dataframe where row['Time'] between _from and _to

    """
    mask = (_from <= df['Time']) & (df['Time'] < _to)
    return df[mask]


def get_column(idx, applier, data_name):
    """This is a wrapper which returns a dataframe column that indicates at each timestamp the
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
            last_x_seconds_df = df_from_to(max(0, row['Time'] - window), row['Time'], df)
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
                last_x_seconds_df = df_from_to(max(0, row['Time'] - gradient_w), row['Time'], df)
                max_v = last_x_seconds_df[data_name].max()
                min_v = last_x_seconds_df[data_name].min()
                res = max_v - min_v
            elif applier == 'max_over_min':
                last_x_seconds_df = df_from_to(max(0, row['Time'] - gradient_w), row['Time'], df)
                max_v = last_x_seconds_df[data_name].max()
                min_v = last_x_seconds_df[data_name].min()
                res = max_v / min_v
            if res == -1:
                print('error in applying ' + data_name + '_' + applier)

            # first mean will be nan, so replace it with second row instead
            return res if not math.isnan(res) else compute(df.iloc[1])

    return sd.obstacle_df_list[idx].apply(compute, axis=1)


def get_percentage_crashes_column(idx):
    """Returns a dataframe column that indicates at each timestamp how many percentage of the last obstacles in the
        last crash-window-seconds the user crashed into

    :param idx: Index into gl.df_list (indicated the dataframe)

    :return: Percentage feature column

    """

    df = sd.df_list[idx]

    def compute_crashes(row):
        if row['Time'] > max(cw, hw, gradient_w):
            last_x_seconds_df = df_from_to(max(0, row['Time'] - cw), row['Time'], df)
            num_obstacles = len(last_x_seconds_df[(last_x_seconds_df['Logtype'] == 'EVENT_OBSTACLE')
                                                  | (last_x_seconds_df['Logtype'] == 'EVENT_CRASH')].index)
            num_crashes = len(last_x_seconds_df[last_x_seconds_df['Logtype'] == 'EVENT_CRASH'].index)
            return (num_crashes/num_obstacles * 100 if num_crashes < num_obstacles else 100) if num_obstacles != 0 else 0

    return sd.obstacle_df_list[idx].apply(compute_crashes, axis=1)


def get_last_obstacle_crash_column(idx):
    """Returns a dataframe column that indicates at each timestamp whether the user crashed into the last obstacle or not

      :param idx: Index into gl.df_list (indicated the dataframe)

      :return: last_obstacle_crash feature column

    """

    df = sd.df_list[idx]

    def compute_crashes(row):
        if row['Time'] > max(cw, hw, gradient_w):

            last_obstacles = df[(df['Time'] < row['Time']) & ((df['Logtype'] == 'EVENT_OBSTACLE') | (df['Logtype'] == 'EVENT_CRASH'))]
            if last_obstacles.empty:
                return 0
            return 1 if last_obstacles.iloc[-1]['Logtype'] == 'EVENT_CRASH' else 0

    return sd.obstacle_df_list[idx].apply(compute_crashes, axis=1)


def get_hr_slope_column(idx):
    """Returns a dataframe column that indicates at each timestamp the slope of the fitting lin/ regression
        line over the heartrate in the last hw seconds

          :param idx: Index into gl.df_list (indicated the dataframe)

          :return: hr_slope feature column

          """

    df = sd.df_list[idx]

    def compute_slope(row):
        if row['Time'] > max(cw, hw, gradient_w):
            last_x_seconds_df = df_from_to(max(0, row['Time'] - gradient_w), row['Time'], df)

            slope, _ = np.polyfit(last_x_seconds_df['Time'], last_x_seconds_df['Heartrate'], 1)

            return slope if not math.isnan(slope) else compute_slope(df.iloc[1])

    return sd.obstacle_df_list[idx].apply(compute_slope, axis=1)


def get_gradient_changes_column(idx, data_name):
    """Returns a dataframe column that indicates at each timestamp the number of times 'data_name' (points or Heartrate)
        have changed from increasing to decreasing and the other way around

        :param idx: Index into gl.df_list (indicated the dataframe)
        :param data_name: Points or Heartrate

        :return: gradient_changes feature column for either points or heartrate

    """

    df = sd.df_list[idx]

    def compute_gradient_changes(row):
        if row['Time'] > max(cw, hw, gradient_w):

            last_x_seconds_df = df_from_to(max(0, row['Time'] - cw), row['Time'], df)
            data = last_x_seconds_df[data_name].tolist()
            gradx = np.gradient(data)
            asign = np.sign(gradx)

            num_sign_changes = len(list(itertools.groupby(asign, lambda x: x >= 0))) - 1
            if num_sign_changes == 0:
                num_sign_changes = 1
            return num_sign_changes if not math.isnan(num_sign_changes) else compute_gradient_changes(df.iloc[1])

    return sd.obstacle_df_list[idx].apply(compute_gradient_changes, axis=1)

