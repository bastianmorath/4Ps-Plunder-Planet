"""
This module is the main file when one wants to ...

"""

from __future__ import division, print_function  # s.t. division uses float result

import time
import argparse


import setup_dataframes
import plots
import synthesized_data
import features_factory as f_factory
import window_optimization
import hyperparameter_optimization
import model_factory
import leave_one_group_out_cv
import LSTM
import classifiers

# TODO: Put underscore in front of private functions
# TODO: Store X, y somewhere s.t. we don't have to pass it to method calls everytime
# TODO: Add :type in docstrings where necessary

_num_iter = 1


def main(args):
    start = time.time()
    f_factory.use_reduced_features = not args.no_feature_selection

    assert (not (args.use_synthesized_data and args.leave_one_group_out)), \
        'Can\'t do leave_one_group_out with synthesized data'

    if args.use_synthesized_data:

        print('Creating synthesized data...')
        synthesized_data.synthesized_data_enabled = True
        # synthesized_data.init_with_testdata_events_random_hr_const()
        synthesized_data.init_with_testdata_events_const_hr_const()
        # synthesized_data.init_with_testdata_events_random_hr_continuous()
        X, y = f_factory.get_feature_matrix_and_label(
            verbose=args.verbose, use_cached_feature_matrix=False, save_as_pickle_file=False,
            feature_selection=f_factory.use_reduced_features
        )

    else:
        setup_dataframes.setup(
            fewer_data=args.reduced_data,  # Specify if we want fewer data (for debugging purposes...)
            normalize_heartrate=(not args.do_not_normalize_heartrate),
        )

        if not args.test_windows:  # We most likely have to calculate new feature matrix anyways
            X, y = f_factory.get_feature_matrix_and_label(
                verbose=True,
                use_cached_feature_matrix=True,
                save_as_pickle_file=True,
                # TODO: Remove f_selection argument as it is stored as local variable anyways
                feature_selection=f_factory.use_reduced_features,
                use_boxcox=False,
            )

    # model_factory.test_clf_with_timedelta_only()

    if args.print_keynumbers_logfiles:
        print("\n################# Printing keynumbers #################\n")

        setup_dataframes.print_keynumbers_logfiles()

    if args.test_windows:
        print("\n################# Window optimization #################\n")
        window_optimization.performance_score_for_windows(
            args.test_windows[0],
            args.test_windows[1],
            args.test_windows[2],
            verbose=args.verbose,
            write_to_file=True,
        )

    if args.performance_without_tuning or args.performance_with_tuning:
        if args.performance_with_tuning:
            print("\n################# Calculating performance with hyperparameter tuning #################\n")
        else:
            print("\n################# Calculating performance without hyperparameter tuning #################\n")

        if args.performance_without_tuning == 'all' or args.performance_with_tuning == 'all':
            model_factory. \
                calculate_performance_of_classifiers(X, y, tune_hyperparameters=args.performance_with_tuning,
                                                     reduced_clfs=False, num_iter=_num_iter)
        else:
            if args.performance_with_tuning:
                clf, tuned_params = hyperparameter_optimization.get_tuned_clf_and_tuned_hyperparameters(
                    X, y, clf_name=args.performance_with_tuning, num_iter=_num_iter
                )
                print('(n_iter in RandomizedSearchCV=' + str(_num_iter) + ')')
                model_factory.get_performance(clf, args.performance_with_tuning, X, y, tuned_params,
                                              verbose=True, do_write_to_file=False)
            else:
                model = classifiers.get_cclassifier_with_name(args.performance_without_tuning, X, y).clf
                model_factory.get_performance(model, args.performance_without_tuning, X, y,
                                              verbose=True, do_write_to_file=False)

    if args.leave_one_group_out:
        # TODO: Add old plot (where logfile_left_out is used) into report
        print("\n################# Leave one out #################\n")
        leave_one_group_out_cv.clf_performance_with_user_left_out_vs_normal(
            X, y, True
        )

    if args.get_trained_lstm:
        print("\n################# Get trained LSTM #################\n")
        LSTM.get_trained_lstm_classifier(X, y, n_epochs=args.get_trained_lstm[0])

    if args.generate_plots_about_features:
        print("\n################# Generate plots about features #################\n")
        plot_features(X, y)

    if args.generate_plots_about_logfiles:
        print("\n################# Generate plots about logfiles #################\n")
        plot_logfiles()

    end = time.time()
    print("Time elapsed: " + str(end - start))


def plot_features(X, y):
    plots.plot_corr_knn_distr(X, y)
    plots.plot_timedeltas_and_crash_per_logfile(do_normalize=True)
    plots.plot_feature_distributions(X)
    plots.plot_heartrate_histogram()
    plots.plot_mean_value_of_feature_at_crash(X, y)

    for i in range(0, len(f_factory.feature_names)):
        plots.plot_feature(X, i)


def plot_logfiles():
    if not args.do_not_normalize_heartrate:
        print('Attention: Heartrates are normalized. Maybe call module with --do_not_normalize_heartrate')
    plots.crashes_per_obstacle_arrangement()
    # plots.plot_crashes_vs_size_of_obstacle()
    plots.plot_hr_vs_difficulty_scatter_plot()
    plots.print_obstacle_information()
    plots.plot_difficulty_vs_size_obstacle_scatter_plot()
    plots.plot_hr_or_points_corr_with_difficulty('Heartrate')
    plots.plot_hr_or_points_corr_with_difficulty('Points')
    plots.plot_heartrate_change()
    plots.plot_mean_and_std_hr_boxplot()
    plots.plot_hr_of_dataframes()
    plots.plot_average_hr_over_all_logfiles()


if __name__ == "__main__":
    # Add user-friendly command-line interface to enter windows and RandomSearchCV parameters etc.

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-w",
        "--performance_without_tuning",
        type=str,
        help="Outputs detailed scores of the given classifier without doing hyperparameter tuning."
             " Set clf_name='all' if you want to test all classifiers",
        metavar='clf_name',
    )

    parser.add_argument(
        "-o",
        "--performance_with_tuning",
        type=str,
        help="Optimizes the given classifier with RandomSearchCV and outputs detailed scores."
             " Set clf_name='all' if you want to test all classifiers",
        metavar='clf_name',
    )

    parser.add_argument(
        "-t",
        "--test_windows",
        type=int,
        nargs=3,
        help="Trains and tests a SVM with the given window sizes. Stores roc_auc score in "
             "a file in /Evaluation/Performance/Windows. "
             "Note: Provide the windows in seconds",
        metavar=('hw_window', 'crash_window', 'gc_window'),
    )

    parser.add_argument(
        "-l",
        "--leave_one_group_out",
        action="store_true",
        help="Plot performance when leaving out a logfile "
             "vs leaving out a whole user in crossvalidation",
    )

    parser.add_argument(
        "-m",
        "--get_trained_lstm",
        type=int,
        nargs=1,
        help="Train an LSTM newtwork with n_epochs",
        metavar='n_epochs',
    )

    parser.add_argument(
        "-k",
        "--print_keynumbers_logfiles",
        action="store_true",
        help="Print important numbers and stats about the logfiles ",
    )

    parser.add_argument(
        "-f",
        "--generate_plots_about_features",
        action="store_true",
        help="Generates different plots from the feature matrix (Look at main.py for details) and stores it "
             "in folder /Evaluation/Features",
    )

    parser.add_argument(
        "-p",
        "--generate_plots_about_logfiles",
        action="store_true",
        help="Generates different plots from the logfiles (Look at main.py for details) and stores it "
             "in folder /Evaluation/Logfiles (Note: Probably use in combination with -n, i.e. without "
             "normalizing heartrate)",
    )

    parser.add_argument(
        "-s",
        "--no_feature_selection",
        action='store_true',
        help="Do not do feature selection with cross_correlation matrix"
    )

    parser.add_argument(
        "-d",
        "--use_synthesized_data",
        action="store_true",
        help="Use synthesized data. Might not work with everything."  # TODO
    )

    parser.add_argument(
        "-n",
        "--do_not_normalize_heartrate",
        action="store_true",
        help="Do not normalize heartrate (e.g. if you want plots or values with real heartrate)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Prints various information while computing",
    )

    parser.add_argument(
        "-r",
        "--reduced_data",
        action="store_true",
        help="Use only a small part of the data. Mostly for debugging purposes",
    )

    args = parser.parse_args()

    main(args)



