import pandas as pd
import os
import math

from scipy import stats
import numpy as np
import itertools

import globals as gl

'''Current features:
    1. Mean_hr over last x seconds
    2. % of crashes in the last x seconds
    3. Max over min heartrate in last x seconds
'''

# NOTE: Have to be the same order as below... (matrix[...]=...)
feature_names = ['mean_hr', 'max_hr', 'min_hr', 'std_hr', 'max_minus_min_hr', 'max_over_min_hr', 'lin_regression_hr_slope', 'hr_gradient_changes',
                '%crashes', 'last_obstacle_crash',
                 'points_gradient_changes', 'mean_points', 'max_points', 'min_points', 'std_points', 'max_minus_min_points']


# feature_names = ['mean_hr', 'std_hr', 'max_minus_min_hr', 'max_over_min_hr', 'lin_regression_hr_slope', 'hr_gradient_changes',
#                 '%crashes',
#                 'mean_points', 'std_points']


''' Returns a matrix containing the features, and the labels
    There is one feature-row for each obstacle
 '''


def get_feature_matrix_and_label():

    matrix = pd.DataFrame()

    if gl.use_cache and (not gl.test_data) and os.path.isfile(gl.working_directory_path + '/Pickle/feature_matrix.pickle'):
        matrix = pd.read_pickle(gl.working_directory_path + '/Pickle/feature_matrix.pickle')
    else:
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

        matrix.to_pickle(gl.working_directory_path + '/Pickle/feature_matrix.pickle')

        # matrix.to_csv(gl.working_directory_path + '/Pickle/feature_matrix.csv')
        # df = pd.concat(gl.obstacle_df_list)
        # df.to_csv(gl.working_directory_path + '/Pickle/obstacle_df.csv')
    # remove ~ first heartrate_window rows (they have < hw seconds to compute features, and are thus not accurate)
    labels = []
    for df in gl.obstacle_df_list:
        labels.append(df[df['Time'] > max(gl.cw, gl.hw)]['crash'].copy())
    labels = list(itertools.chain.from_iterable(labels))

    # Boxcox transformation
    if gl.use_boxcox:
        # Values must be positive. If not, shift it
        for feature in feature_names:
            if not feature == 'last_obstacle_crash':  # Doesn't makes sense to do boxcox here
                if matrix[feature].min() <= 0:
                    matrix[feature] = stats.boxcox(matrix[feature] - matrix[feature].min() + 0.01)[0]
                else:
                    matrix[feature] = stats.boxcox(matrix[feature])[0]


    return matrix.as_matrix(), labels


"""The following methods append a column to the feature matrix"""


def get_standard_feature(feature, data_name):
    print('Creating ' + feature + '_' + data_name + ' feature...')

    hr_df_list = []  # list that contains a list of mean_hrs for each logfile/df
    for list_idx, df in enumerate(gl.df_list):
        if not (df['Heartrate'] == -1).all(): # NOTE: Can be omitted if logfiles without heartrate data is removed in setup.py
            hr_df = get_column(list_idx, df, feature, data_name)
            hr_df_list.append(hr_df)

    return pd.DataFrame(list(itertools.chain.from_iterable(hr_df_list)), columns=[feature])


# TODO: Normalize crashes depending on size/assembly of the obstacle


def get_percentage_crashes_feature():
    print('Creating %crashes feature...')

    crashes_list = []  # list that contains a list of %crashes for each logfile/df
    for list_idx, df in enumerate(gl.df_list):
        crashes = get_percentage_crashes_column(list_idx, df)
        # df = df[df['Time'] > max(gl.cw, gl.hw)]  # remove first window-seconds bc. not accurate data
        crashes_list.append(crashes)
    return pd.DataFrame(list(itertools.chain.from_iterable(crashes_list)), columns=['%crashes'])


def get_last_obstacle_crash_feature():
    print('Creating last_obstacle_crash feature...')
    crashes_list = []  # list that contains a list of whether crash or not for each logfile/df
    for list_idx, df in enumerate(gl.df_list):
        df_obstacles = get_last_obstacle_crash_column(list_idx, df)
        # df = df[df['Time'] > max(gl.cw, gl.hw)]  # remove first window-seconds bc. not accurate data
        crashes_list.append(df_obstacles)

    return pd.DataFrame(list(itertools.chain.from_iterable(crashes_list)), columns=['last_obstacle_crash'])


def get_lin_regression_hr_slope_feature():
    print('Creating lin_regression_hr_slope feature...')
    slopes = []  # list that contains a list of the slope
    for list_idx, df in enumerate(gl.df_list):
        slope = get_hr_slope_column(list_idx, df)
        slopes.append(slope)

    return pd.DataFrame(list(itertools.chain.from_iterable(slopes)), columns=['lin_regression_hr_slope'])


def get_number_of_gradient_changes(data_name):
    print('Creating points_gradient_changes feature...')
    changes_list = []  # list that contains a list of the slope
    for list_idx, df in enumerate(gl.df_list):
        changes = get_gradient_changes_column(list_idx, df, data_name)
        changes_list.append(changes)
    if data_name == 'Points' :
        return pd.DataFrame(list(itertools.chain.from_iterable(changes_list)), columns=['points_gradient_changes'])
    else:
        return pd.DataFrame(list(itertools.chain.from_iterable(changes_list)), columns=['hr_gradient_changes'])



"""The following methods calculate the features as a new dataframe column"""


'''Returns the dataframe where time is between _from and _to'''


def df_from_to(_from, _to, df):
    mask = (_from <= df['Time']) & (df['Time'] < _to)
    return df[mask]


''' Returns a dataframe column that indicates at each timestamp the 
    heartrate or points over the last 'window' seconds, after applying 'applyer' (e.g. mean, min, max, std)
    
    data_name = Points or Heartrate
'''


def get_column(idx, df, applier, data_name):
    window = gl.hw
    if data_name == 'Points':
        window = gl.cw

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
            last_x_seconds_df = df_from_to(max(0, row['Time'] - gl.hw_change), row['Time'], df)
            max_v = last_x_seconds_df[data_name].max()
            min_v = last_x_seconds_df[data_name].min()
            res = max_v - min_v
        elif applier == 'max_over_min':
            last_x_seconds_df = df_from_to(max(0, row['Time'] - gl.hw_change), row['Time'], df)
            max_v = last_x_seconds_df[data_name].max()
            min_v = last_x_seconds_df[data_name].min()
            res = max_v / min_v
        if res == -1:
            print('error in applying ' + data_name + '_' + applier)

        # first mean will be nan, so replace it with second row instead
        return res if not math.isnan(res) else compute(df.iloc[1])

    return gl.obstacle_df_list[idx].apply(compute, axis=1)


''' Returns a dataframe column that indicates at each timestamp how many percentage of the last obstacles in the 
    last crash-window-seconds the user crashed into
'''


def get_percentage_crashes_column(idx, df):

    def compute_crashes(row):
        last_x_seconds_df = df_from_to(max(0, row['Time'] - gl.cw), row['Time'], df)
        num_obstacles = len(last_x_seconds_df[(last_x_seconds_df['Logtype'] == 'EVENT_OBSTACLE')
                                              | (last_x_seconds_df['Logtype'] == 'EVENT_CRASH')].index)
        num_crashes = len(last_x_seconds_df[last_x_seconds_df['Logtype'] == 'EVENT_CRASH'].index)
        return (num_crashes/num_obstacles * 100 if num_crashes < num_obstacles else 100) if num_obstacles != 0 else 0

    return gl.obstacle_df_list[idx].apply(compute_crashes, axis=1)


'''Returns a dataframe column that indicates at each timestamp whether the user crashed into the last obstacle or not
'''


def get_last_obstacle_crash_column(idx, df):
    def compute_crashes(row):
        last = df[(df['Time'] < row['Time']) & ((df['Logtype'] == 'EVENT_OBSTACLE') | (df['Logtype'] == 'EVENT_CRASH'))]
        if last.empty:
            return 0
        return 1 if last.iloc[-1]['Logtype'] == 'EVENT_CRASH' else 0

    return gl.obstacle_df_list[idx].apply(compute_crashes, axis=1)


'''Returns a dataframe column that indicates at each timestamp the slope of the fitting lin/ regression 
        line over the heartrate in the last hw seconds
'''

# TODO: maybe also add intercept, r_value, p_value, std_err as features?


def get_hr_slope_column(idx, df):

    def compute_slope(row):

        last_x_seconds_df = df_from_to(max(0, row['Time'] - gl.hw_change), row['Time'], df)

        slope, _ = np.polyfit(last_x_seconds_df['Time'], last_x_seconds_df['Heartrate'], 1)

        '''# Plot slopes
        fit = np.polyfit(last_x_seconds_df['Time'], last_x_seconds_df['Heartrate'], 1)
        fit_fn = np.poly1d(fit)
        plt.plot(last_x_seconds_df['Time'], last_x_seconds_df['Heartrate'], '', last_x_seconds_df['Time'], fit_fn(last_x_seconds_df['Time']))

        plt.savefig(gl.working_directory_path + '/Plots/lin_regression.pdf')
        '''
        return slope if not math.isnan(slope) else compute_slope(df.iloc[1])

    return gl.obstacle_df_list[idx].apply(compute_slope, axis=1)


'''Returns a dataframe column that indicates at each timestamp the number of times 'data_name' (points or Heartrate) 
have changed from increasing to decreasing and the other way around
'''


def get_gradient_changes_column(idx, df, data_name):

    def compute_gradient_changes(row):

        last_x_seconds_df = df_from_to(max(0, row['Time'] - gl.cw), row['Time'], df)
        data = last_x_seconds_df[data_name].tolist()
        gradx = np.gradient(data)
        asign = np.sign(gradx)

        num_sign_changes = len(list(itertools.groupby(asign, lambda x: x >= 0))) - 1
        if num_sign_changes == 0:
            num_sign_changes = 1
        return num_sign_changes if not math.isnan(num_sign_changes) else compute_gradient_changes(df.iloc[1])

    return gl.obstacle_df_list[idx].apply(compute_gradient_changes, axis=1)


