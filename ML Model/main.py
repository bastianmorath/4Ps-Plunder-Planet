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


# TODO: Put underscore in front of private functions
# TODO: Store X, y somewhere s.t. we don't have to pass it to method calls everytime
# TODO: Add :type in docstrings where necessary
# TODO: In RandomSearchCV, also try out standard parameters!

def main(args):
    start = time.time()
    f_factory.use_reduced_features = not args.no_feature_selection

    assert (not (args.use_test_data and args.leave_one_out)), 'Can\'t do leave_one_out with synthesized data'

    if args.use_test_data:

        print('Creating synthesized data...')
        synthesized_data.test_data_enabled = True
        synthesized_data.init_with_testdata_events_random_hr_const()
        # test_data.init_with_testdata_events_const_hr_const()
        # test_data.init_with_testdata_events_random_hr_continuous()
        X, y = f_factory.get_feature_matrix_and_label(
            verbose=args.verbose, use_cached_feature_matrix=False, save_as_pickle_file=False,
            feature_selection=f_factory.use_reduced_features
        )

    else:

        setup_dataframes.setup(
            use_fewer_data=args.reduced_data,  # Specify if we want fewer data (for debugging purposes...)
            normalize_heartrate=(not args.do_not_normalize_heartrate),
        )
        if not args.test_windows:  # We most likely have to calculate new feature matrix anyways
            X, y = f_factory.get_feature_matrix_and_label(
                verbose=True,
                use_cached_feature_matrix=True,
                save_as_pickle_file=True,
                feature_selection=f_factory.use_reduced_features, # TODO: Remove f_selection argument as it is stored as local variable anyways
                use_boxcox=False,
            )

    if args.print_keynumbers_logfiles:
        print("\n################# Printing keynumbers #################\n")

        setup_dataframes.print_keynumbers_logfiles()

    if args.scores_without_tuning:
        print("\n################# Calculating performance without hyperparameter tuning #################\n")
        model_factory.calculate_performance_of_classifiers(X, y, tune_hyperparameters=False, reduced_clfs=True)

    if args.test_windows:
        print("\n################# Window optimization #################\n")
        window_optimization.performance_score_for_windows(
            args.test_windows[0],
            args.test_windows[1],
            args.test_windows[2],
            verbose=args.verbose,
            write_to_file=True,
        )

    if args.optimize_clf:
        print("\n################# Hyperparameter optimization #################\n")
        if args.optimize_clf == "all":
            model_factory. \
                calculate_performance_of_classifiers(X, y, tune_hyperparameters=True, reduced_clfs=False, num_iter=2)

        else:
            clf, tuned_params = hyperparameter_optimization.get_tuned_clf_and_tuned_hyperparameters(
                X, y, clf_name=args.optimize_clf, num_iter=2  # TODO: Increase num_iter
            )
            _, _, _, _, _, rep = model_factory.get_performance(clf, args.optimize_clf, X, y, tuned_params)
            print(rep)

    if args.leave_one_out:
        # TODO: Add old plot where logfile_left_out is used into report
        print("\n################# Leave one out #################\n")
        leave_one_out_cv.clf_performance_with_user_left_out_vs_normal(
            X, y, True
        )

    if args.generate_plots_about_features:
        print("\n################# Generate plots about features #################\n")
        plot_features(X, y)

    if args.generate_plots_about_logfiles:
        print("\n################# Generate plots about logfiles #################\n")
        plot_logfiles()

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
        "-w",
        "--scores_without_tuning",
        action="store_true",
        help="Calculates the performance of SVM, LinearSVM, NearestNeighbor, DecisionTree and Naive Bayes"
             "and plots it in a barchart. Also creates ROC curves",
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
        "-o",
        "--optimize_clf",  # TODO SVM works really slow?
        type=str,
        help="Optimizes the given classifier with RandomSearchCV and outputs detailed scores."
             " Set clf_name='all' if you want to test all classifiers",
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
             "in folder /Evaluation/Logfiles (Note: Probably use with -n, i.e. without normalizing heartrate)",
    )

    parser.add_argument(
        "-s",
        "--no_feature_selection",
        action='store_true',
        help="Do not do feature selection with cross_correlation matrix"
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
        help="Use only a small part of the data. Mostly for debugging purposes",
    )

    args = parser.parse_args()

    main(args)
