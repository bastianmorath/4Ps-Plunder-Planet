"""This module is the main file when one wants to ...

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
import leave_one_out_cv

plot_correlation_matrix = False


def main(args):
    start = time.time()

    if args.use_test_data:
        synthesized_data.init_with_testdata_events_random_hr_const()
        # test_data.init_with_testdata_events_const_hr_const()
        # test_data.init_with_testdata_events_random_hr_continuous()
        X, y = f_factory.get_feature_matrix_and_label(
            verbose=args.verbose, use_cached_feature_matrix=True, save_as_pickle_file=False
        )
    else:
        if plot_correlation_matrix:
            f_factory.plot_corr_matrix = True

        setup_dataframes.setup(
            use_fewer_data=args.reduced_data,
            normalize_heartrate=(not args.do_not_normalize_heartrate),
        )  # Specify if we want fewer data (for debugging purposes...)
        X, y = f_factory.get_feature_matrix_and_label(
            verbose=args.verbose,
            use_cached_feature_matrix=True,
            save_as_pickle_file=True,
            feature_selection=args.do_feature_selection,
            use_boxcox=False,
        )

    if args.print_keynumbers_logfiles:
        setup_dataframes.print_keynumbers_logfiles()

    if args.test_windows:
        print("\n'window_optimization' called: ")
        window_optimization.performance_score_for_windows(
            args.test_windows[0],
            args.test_windows[1],
            args.test_windows[2],
            verbose=args.v,
            write_to_file=True,
        )

    if args.grid_search:
        print("\n'hyperparameter_optimization' called")

        if args.grid_search == "all":
            names, scores, optimal_params, conf_mats = hyperparameter_optimization.\
                get_performance_of_all_clf_with_optimized_hyperparameters(X, y, 20)

            model_factory.plot_barchart_scores(names, scores)
            model_factory.write_scores_to_file(names, scores, optimal_params, conf_mats)
        else:
             hyperparameter_optimization.get_clf_with_optimized_hyperparameters(
                X, y, args.grid_search, 20, verbose=True  # TODO: args.verbose
            )

    if args.leave_one_out:
        print("'leave_one_out' called")
        leave_one_out_cv.clf_performance_with_user_left_out_vs_normal(
            X, y, True
        )

    if args.generate_plots_about_features:
        print("Generating plots about features")
        plot_features(X, y)

    if args.generate_plots_about_logfiles:
        print("Generating plots about logfiles")
        plot_logfiles(X, y)

    end = time.time()
    print("Time elapsed: " + str(end - start))


def plot_features(X, y):

    plots.plot_feature_distributions(X)

    plots.plot_heartrate_histogram()

    plots.plot_mean_value_of_feature_at_crash(X, y)
    for i in range(0, len(f_factory.feature_names)):
        plots.plot_feature(X, i)


def plot_logfiles():

    plots.plot_hr_of_dataframes()


if __name__ == "__main__":
    # Add user-friendly command-line interface to enter windows and RandomSearchCV parameters etc.

    parser = argparse.ArgumentParser()

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
        "-g",
        "--grid_search",
        type=str,
        help="Optimizes the given classifier with RAndomSearchCV. Set"
        "clf_name='all' if you want to test all classifiers",
        metavar='clf_name',
    )

    parser.add_argument(
        "-l",
        "--leave_one_out",
        action="store_true",
        help="Plot performance when leaving out a logfile "
             "vs leaving out a whole user in crossvalidation",
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
             "in folder /Evaluation/Logfiles",
    )

    parser.add_argument(
        "-s",
        "--do_feature_selection",
        action='store_true',
        help="Do feature selection with cross_correlation matrix"
    )

    parser.add_argument(
        "-d",
        "--use_test_data",
        action="store_true",
        help="Plot performance when leaving out a logfile "
        "vs leaving out a whole user in crossvalidation",
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
        help="Use only a small aprt of the data. Mostly for debugging purposes",
    )

    args = parser.parse_args()

    main(args)
