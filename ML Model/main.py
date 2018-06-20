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


"""INITIALIZATION"""
plot_all = False
plot_heartrate_of_each_logfile = False
plot_feature_distributions = False
plot_heartrate_histogram = False
plot_mean_value_of_feature_at_crash = False
plot_correlation_matrix = False
plot_feature = (False, 8)


def main(args):
    start = time.time()

    if args.use_test_data:
        synthesized_data.init_with_testdata_events_random_hr_const()
        # test_data.init_with_testdata_events_const_hr_const()
        # test_data.init_with_testdata_events_random_hr_continuous()
        X, y = f_factory.get_feature_matrix_and_label(
            verbose=True, use_cached_feature_matrix=True, save_as_pickle_file=False
        )
    else:
        if plot_correlation_matrix:
            f_factory.plot_corr_matrix = True

        setup_dataframes.setup(
            use_fewer_data=False
        )  # Specify if we want fewer data (for debugging purposes...)

        X, y = f_factory.get_feature_matrix_and_label(
            verbose=True,
            use_cached_feature_matrix=True,
            save_as_pickle_file=True,
            feature_selection=args.do_feature_selection,
            use_boxcox=False,
        )

    if args.print_keynumbers_logfiles:
        setup_dataframes.print_keynumbers_logfiles()

    plot(X, y)

    if args.test_windows:
        print("\n'window_optimization' called: ")
        window_optimization.performance_score_for_windows(
            args.test_windows[0],
            args.test_windows[1],
            args.test_windows[2],
            verbose=False,
            write_to_file=True,
        )

    if args.grid_search:
        print("\n'hyperparameter_optimization' called")

        if args.grid_search[0] == "all":
            names, scores, optimal_params, conf_mats = hyperparameter_optimization.get_performance_of_all_clf_with_optimized_hyperparameters(
                X, y, 20
            )

            model_factory.plot_barchart_scores(names, scores)
            model_factory.write_scores_to_file(names, scores, optimal_params, conf_mats)
        else:
            args = hyperparameter_optimization.get_clf_with_optimized_hyperparameters(
                X, y, args.grid_search[0], 20
            )

    if args.leave_one_out:
        print("'leave_one_out' called")
        leave_one_out_cv.clf_performance_with_user_left_out_vs_normal(
            X, y, True
        )  # Works

    end = time.time()
    print("Time elapsed: " + str(end - start))


def plot(X, y):
    if (
        plot_heartrate_of_each_logfile
        or plot_feature_distributions
        or plot_heartrate_histogram
        or plot_mean_value_of_feature_at_crash
    ):
        print("Plotting...")

    if plot_heartrate_of_each_logfile or plot_all:
        plots.plot_hr_of_dataframes()  # DONE

    if plot_feature_distributions or plot_all:

        plots.plot_feature_distributions(X)  # DONE

    if plot_heartrate_histogram or plot_all:

        plots.plot_heartrate_histogram()  # DONE

    if plot_mean_value_of_feature_at_crash or plot_all:

        plots.plot_mean_value_of_feature_at_crash(X, y)  # DONE

    if plot_feature[0] or plot_all:
        plots.plot_feature(X, plot_feature[1])


if __name__ == "__main__":
    # Add user-friendly command-line interface to enter windows and RandomSearchCV parameters etc.

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-test_windows",
        type=int,
        nargs=3,
        help="Provide the windowsizes of the heartrate, crashwindow and the gradient_change window [s]",
        metavar=('hw_window', 'crash_window', 'gc_window'),
    )
    parser.add_argument(
        "-grid_search",
        type=str,
        help="Provide the classifier name for RandomSearchCV. Set"
        "clf_name='all' if you want to test all classifiers",
        metavar='clf_name',
    )

    parser.add_argument(
        "-leave_one_out",
        action="store_true",
        help="Plot performance when leaving out a logfile "
             "vs leaving out a whole user in crossvalidation",
    )

    parser.add_argument(
        "-print_keynumbers_logfiles",
        action="store_true",
        help="Print important numbers and stats about the logfiles ",
    )

    parser.add_argument(
        "--do_feature_selection",
        action='store_true',
        help="Do feature selection with cross_correlation matrix"
    )

    parser.add_argument(
        "--use_test_data",
        action="store_true",
        help="Plot performance when leaving out a logfile "
        "vs leaving out a whole user in crossvalidation",
    )
    args = parser.parse_args()

    main(args)
