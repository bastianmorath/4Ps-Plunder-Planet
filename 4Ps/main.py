"""
Entry point of this project.
Type 'python main.py -h' to see how to use this module

"""

from __future__ import division  # s.t. division uses float result
from __future__ import print_function

import time

import matplotlib

matplotlib.use('Agg')

import LSTM
import classifiers
import model_factory
import argparse_setup
import plots_report
import plots_features as fp
import plots_logfiles as lp
import features_factory as f_factory
import setup_dataframes
import synthesized_data
import window_optimization
import leave_one_group_out_cv
import hyperparameter_optimization


def main(args):
    """
    Call '$ python main.py -h' to see how to use this module
    :param args: ArgumentParser

    """

    start = time.time()

    f_factory.use_reduced_features = not args.use_all_features

    assert (not (args.use_synthesized_data and args.leave_one_group_out)), \
        'Can\'t do leave_one_group_out with synthesized data'

    if args.use_synthesized_data:
        print('Creating synthesized data...')

        synthesized_data.init_with_testdata_events_const_hr_const()

        X, y = f_factory.get_feature_matrix_and_label(
            verbose=True, use_cached_feature_matrix=False, save_as_pickle_file=False,
            reduced_features=f_factory.use_reduced_features
        )

    else:
        setup_dataframes.setup(
            fewer_data=args.debugging,  # Specify if we want fewer data (for debugging purposes...)
            normalize_heartrate=(not args.do_not_normalize_heartrate),
            remove_tutorials=False
        )

        X, y = f_factory.get_feature_matrix_and_label(
                verbose=True,
                use_cached_feature_matrix=True,
                save_as_pickle_file=True,
                reduced_features=f_factory.use_reduced_features,
                use_boxcox=False
        )

    if args.print_keynumbers_logfiles:
        print("\n################# Printing keynumbers #################\n")

        setup_dataframes.print_keynumbers_logfiles()

    if args.test_windows:
        print("\n################# Window optimization #################\n")
        window_optimization.performance_score_for_windows(
            args.test_windows[0],
            args.test_windows[1],
            args.test_windows[2],
            verbose=True,
            write_to_file=True,
        )

    if args.performance_without_tuning or args.performance_with_tuning:
        # IMPORTANT: pre-set == True takes already tuned parameters if possible. Change if you always want to do tuning!
        pre_set = True

        if args.performance_with_tuning:
            print("\n################# Calculating performance with hyperparameter tuning #################\n")
        else:
            print("\n################# Calculating performance without hyperparameter tuning #################\n")

        if args.performance_without_tuning == 'all' or args.performance_with_tuning == 'all':
            model_factory. \
                calculate_performance_of_classifiers(X, y, tune_hyperparameters=args.performance_with_tuning,
                                                     reduced_clfs=True, pre_set=pre_set)
        else:
            X_old = X
            y_old = y
            if (args.performance_with_tuning == 'Naive Bayes') or (args.performance_without_tuning == 'Naive Bayes'):
                X, y = f_factory.get_feature_matrix_and_label(verbose=False, use_cached_feature_matrix=True,
                                                              save_as_pickle_file=True, use_boxcox=True)

            if args.performance_with_tuning:
                clf, tuned_params = hyperparameter_optimization.get_tuned_clf_and_tuned_hyperparameters(
                    X, y, clf_name=args.performance_with_tuning, pre_set=pre_set,
                )
                _, _, _, _, _, _, _, _, _, _, _, report = model_factory.get_performance(clf, args.performance_with_tuning, X,
                                                                                  y, tuned_params, verbose=True,
                                                                                  do_write_to_file=False)
            else:
                model = classifiers.get_cclassifier_with_name(args.performance_without_tuning, X, y)

                _, _, _, _, _, _, _, _, _, _, _, report = model_factory.get_performance(model.clf,
                                                                                  args.performance_without_tuning,
                                                                                  X, y, verbose=True,
                                                                                  do_write_to_file=False)
            X = X_old
            y = y_old

            print(report)

    if args.leave_one_group_out:
        print("\n################# Leave one out #################\n")
        leave_one_group_out_cv.clf_performance_with_user_left_out_vs_normal(
            X, y, True, reduced_features=f_factory.use_reduced_features, reduced_classifiers=True
        )

    if args.evaluate_lstm:
        print("\n################# Get trained LSTM #################\n")
        # TODO: Separate those maybe
        # LSTM.get_performance_of_lstm_classifier(X, y, n_epochs=args.evaluate_lstm[0])
        LSTM.get_finalscore(X, y, n_epochs=args.evaluate_lstm[0])

    if args.generate_plots_about_features:
        print("\n################# Generate plots about features #################\n")
        plot_features(X, y)

    if args.generate_plots_about_logfiles:
        print("\n################# Generate plots about logfiles #################\n")
        plot_logfiles(args)

    if args.generate_plots_for_report:
        print("\n################# Generate plots for report #################\n")
        plots_report.generate_plots_for_report()

    end = time.time()
    print("Time elapsed: " + str(end - start))


def plot_features(X, y):
    fp.generate_plots_about_features(X, y)


def plot_logfiles(args):
    if not args.do_not_normalize_heartrate:
        print('Note: Heartrates are normalized. Maybe call module with --do_not_normalize_heartrate')
    lp.generate_plots_about_logfiles()


if __name__ == "__main__":
    # Add user-friendly command-line interface to enter windows and RandomizedSearchCV parameters etc.

    _args = argparse_setup.get_argparse()
    main(_args)
