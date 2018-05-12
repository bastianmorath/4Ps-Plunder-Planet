"""This module is the main file when one wants to ...

"""

from __future__ import division, print_function  # s.t. division uses float result

import time
import argparse

import setup
import plots
import test_data
import globals as gl
import features_factory as f_factory
import window_optimization
import hyperparameter_optimization
import model_factory
import leave_one_out_cv

"""INITIALIZATION"""
plot_heartrate_of_each_logfile = False
plot_feature_distributions = False
plot_heartrate_histogram = False
plot_mean_value_of_feature_at_crash = False

# Add user-friendly command-line interface to enter windows and RandomSearchCV parameters etc.

parser = argparse.ArgumentParser()
parser.add_argument('--feature_selection', help='Do feature selection with cross_correlation matrix')
parser.add_argument('--test_windows', type=int, nargs='3', default=[30, 30, 10],
                    help='Provide the windowsizes of the heartrate, crashwindow and the gradient_change window [s]')
parser.add_argument('--grid_search', type=[str, int], nargs='2', default=['all', 20],
                    help='Provide the classifier name and the n_iter for RandomSearchCV.'
                         'clf_id=\'all\' if you want to test all classifiers')
parser.add_argument('--leave_one_out', help='Plot performance when leaving out a logfile vs leaving out a whole user in'
                                            'crossvalidation')
args = parser.parse_args()


print('Init dataframes...')

start = time.time()

if args.feature_selection:
    gl.reduced_features = True

if gl.test_data:
    test_data.init_with_testdata_events_random_hr_const()
    # test_data.init_with_testdata_events_const_hr_const()
    # test_data.init_with_testdata_events_random_hr_continuous()
else:
    setup.setup()
    if plot_heartrate_of_each_logfile:
        plots.plot_hr_of_dataframes()

X, y = f_factory.get_feature_matrix_and_label()

if plot_feature_distributions:
    plots.plot_feature_distributions(X)

if plot_heartrate_histogram:
    plots.plot_heartrate_histogram()

if plot_mean_value_of_feature_at_crash:
    plots.plot_mean_value_of_feature_at_crash(X, y)


if args.test_windows:
    window_optimization.write_window_scores_to_file(X, y, args.test_windows[0],
                                                    args.test_windows[1], args.test_windows[2])

if args.grid_search:
    if args.grid_search[0] == 'all':
        names, scores, optimal_params, conf_mats = \
            hyperparameter_optimization.get_performance_of_all_clf_with_optimized_hyperparameters(X, y, 20)

        model_factory.plot_barchart_scores(names, scores)
        model_factory.write_scores_to_file(names, scores, optimal_params, conf_mats)
    else:
        args = hyperparameter_optimization.get_clf_with_optimized_hyperparameters(X, y, args.grid_search[0], 20)

if args.leave_one_out:
    leave_one_out_cv.clf_performance_with_user_left_out_vs_normal(X, y, True)

end = time.time()
print('Time elapsed: ' + str(end - start))
