"""This module is responsible to generate all the features from the data/logfiles

"""

import pandas as pd
import math
import os

from scipy import stats
import numpy as np
import itertools
from sklearn.preprocessing import MinMaxScaler

import setup_dataframes as sd
import plots

"""INITIALIZATION"""
plot_corr_matrix = False

feature_names = []  # Set below
use_reduced_features = True

_verbose = True


cw = 30  # Over how many preceeding seconds should %crashes be calculated?
hw = 30  # Over how many preceeding seconds should heartrate features such as min, max, mean be averaged?
gradient_w = 10  # Over how many preceeding seconds should hr features be calculated that have sth. do to with change?

path_reduced_features = sd.working_directory_path + '/Pickle/reduced_features/feature_matrix.pickle'
path_all_features = sd.working_directory_path + '/Pickle/all_features/feature_matrix.pickle'
path_reduced_features_boxcox = sd.working_directory_path + '/Pickle/reduced_features_boxcox/feature_matrix.pickle'
path_all_features_boxcox = sd.working_directory_path + '/Pickle/all_features_boxcox/feature_matrix.pickle'


def check_if_reading_from_cache_is_okey(use_feature_matrix, use_boxcox):
    """ If the user wants to use an already saved feature amtrix ('all' or 'reduced'), then check if those
    pickle files really exist. If not, new files have to be created


    :param use_feature_matrix: Use already cached matrix; 'all' (use all features), 'selected'
                                (do feature selection first), None (don't use cache)
    :param use_boxcox: Whether boxcox transofrmation should be done (e.g. when Naive Bayes classifier is used)

    :return: Whether reading from cache is okey and  path where to read from/write to new pickel file (if necessary)

    """
    err_string = 'ERROR: Pickle file of Feature matrix not yet created. Creating new one...'
    path = ''

    if not sd.use_fewer_data:
        return False, path

    if use_feature_matrix == 'all':
        if use_boxcox:
            path = path_all_features_boxcox

        else:
            path = path_all_features

    elif use_feature_matrix == 'selected':
        if use_boxcox:
            path = path_reduced_features_boxcox
        else:
            path = path_reduced_features

    else:  # None, i.e. don't use cache
        return False, path

    if not path.exists():
        print(err_string)
        os.makedirs(path.rsplit('/', 1)[0])
        return False, path
    else:
        return True, path


def get_feature_matrix_and_label(verbose=True, use_feature_matrix='all', save_as_pickle_file=True, use_boxcox=False):

    """ Computes the feature matrix and the corresponding labels

    :argument verbose:
    :argument use_feature_matrix: Use already cached matrix; 'all' (use all features), 'selected'
                                    (do feature selection first), None (don't use cache)
    :argument save_as_pickle_file: if use_cached_feature_matrix=False, then store newly computed
                                    matrix in a pickle file
    :argument use_boxcox: Whether boxcox transofrmation should be done (e.g. when Naive Bayes classifier is used)

    :return: Feature matrix and labels

    """

    globals()['use_reduced_features'] = use_feature_matrix == 'selected'

    globals()['_verbose'] = verbose

    if use_feature_matrix == 'reduced':
        globals()['feature_names'] = ['mean_hr', 'std_hr', 'max_minus_min_hr', 'lin_regression_hr_slope', 'hr_gradient_changes',
                                      '%crashes',
                                      'points_gradient_changes', 'mean_points', 'std_points']
    else:
        globals()['feature_names'] = ['mean_hr', 'max_hr', 'min_hr', 'std_hr', 'max_minus_min_hr', 'max_over_min_hr',
                                      'lin_regression_hr_slope', 'hr_gradient_changes',

                                      '%crashes', 'last_obstacle_crash',

                                      'points_gradient_changes', 'mean_points', 'max_points', 'min_points', 'std_points',
                                      'max_minus_min_points']

    matrix = pd.DataFrame()

    reading_from_cache_okey, path = check_if_reading_from_cache_is_okey(use_feature_matrix, use_boxcox)

    if reading_from_cache_okey:
        if _verbose:
            print('Feature matrix already cached!')
            matrix = pd.read_pickle(path)
    else:
        if _verbose:
            print('Creating feature matrix...\n')

        # TODO: Ugly....
        if 'mean_hr' in feature_names:
            matrix['mean_hr'] = get_standard_feature('mean', 'Heartrate')
        if 'max_hr' in feature_names:
            matrix['max_hr'] = get_standard_feature('max', 'Heartrate')
        if 'min_hr' in feature_names:
            matrix['min_hr'] = get_standard_feature('min', 'Heartrate')
        if 'std_hr' in feature_names:
            matrix['std_hr'] = get_standard_feature('std', 'Heartrate')
        if 'max_minus_min_hr' in feature_names:
            matrix['max_minus_min_hr'] = get_standard_feature('max_minus_min', 'Heartrate')
        if 'max_over_min_hr' in feature_names:
            matrix['max_over_min_hr'] = get_standard_feature('max_over_min', 'Heartrate')
        if 'lin_regression_hr_slope' in feature_names:
            matrix['lin_regression_hr_slope'] = get_lin_regression_hr_slope_feature()
        if 'hr_gradient_changes' in feature_names:
            matrix['hr_gradient_changes'] = get_number_of_gradient_changes('Heartrate')
        if '%crashes' in feature_names:
            matrix['%crashes'] = get_percentage_crashes_feature()
        if 'last_obstacle_crash' in feature_names:
            matrix['last_obstacle_crash'] = get_last_obstacle_crash_feature()

        if 'points_gradient_changes' in feature_names:
            matrix['points_gradient_changes'] = get_number_of_gradient_changes('Points')
        if 'mean_points' in feature_names:
            matrix['mean_points'] = get_standard_feature('mean', 'Points')
        if 'max_points' in feature_names:
            matrix['max_points'] = get_standard_feature('max', 'Points')
        if 'min_points' in feature_names:
            matrix['min_points'] = get_standard_feature('min', 'Points')
        if 'std_points' in feature_names:
            matrix['std_points'] = get_standard_feature('std', 'Points')
        if 'max_minus_min_points' in feature_names:
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

    # remove ~ first heartrate_window rows (they have < hw seconds to compute features, and are thus not accurate)
    labels = []
    for df in sd.obstacle_df_list:
        labels.append(df[df['Time'] > max(cw, hw)]['crash'].copy())

    y = list(itertools.chain.from_iterable(labels))


    X = matrix.as_matrix()
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)  # Rescale between 0 and 1

    if plot_corr_matrix:
        plots.plot_correlation_matrix(matrix)
        globals()['plot_corr_matrix'] = False

    if verbose:
        print('\nFeature matrix and labels created!')

    return X, y


def get_standard_feature(feature, data_name):
    """This is a wrapper to compute common features such as min, max, mean for either Points or Heartrate

    :param feature: min, max, mean, std
    :param data_name: Either Heartrate or Points

    :return: Dataframe column containing the feature

    """
    if _verbose:
        print('Creating ' + feature + '_' + data_name + ' feature...')

    hr_df_list = []  # list that contains a dataframe with mean_hrs for each logfile
    for list_idx, df in enumerate(sd.df_list):
        if not (df['Heartrate'] == -1).all():  # NOTE: Can be omitted if logfiles without heartrate data is removed in prepare_dataframes.py
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
        # df = df[df['Time'] > max(cw, hw)]  # remove first window-seconds bc. not accurate data
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
    if data_name == 'Points':
        window = cw

    def compute(row):
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
        last = df[(df['Time'] < row['Time']) & ((df['Logtype'] == 'EVENT_OBSTACLE') | (df['Logtype'] == 'EVENT_CRASH'))]
        if last.empty:
            return 0
        return 1 if last.iloc[-1]['Logtype'] == 'EVENT_CRASH' else 0

    return sd.obstacle_df_list[idx].apply(compute_crashes, axis=1)


def get_hr_slope_column(idx):
    """Returns a dataframe column that indicates at each timestamp the slope of the fitting lin/ regression
        line over the heartrate in the last hw seconds

          :param idx: Index into gl.df_list (indicated the dataframe)

          :return: hr_slope feature column

          """

    df = sd.df_list[idx]

    def compute_slope(row):

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

        last_x_seconds_df = df_from_to(max(0, row['Time'] - cw), row['Time'], df)
        data = last_x_seconds_df[data_name].tolist()
        gradx = np.gradient(data)
        asign = np.sign(gradx)

        num_sign_changes = len(list(itertools.groupby(asign, lambda x: x >= 0))) - 1
        if num_sign_changes == 0:
            num_sign_changes = 1
        return num_sign_changes if not math.isnan(num_sign_changes) else compute_gradient_changes(df.iloc[1])

    return sd.obstacle_df_list[idx].apply(compute_gradient_changes, axis=1)

