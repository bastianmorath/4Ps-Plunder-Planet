"""
This module is responsible for setting up argparse

"""
import argparse


def get_argparse():
    """
    Generates an ArgumentParser to parse commands in the main.py file

    :return: ArgumentParser object
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-r",
        "--generate_plots_for_report",
        action="store_true",
        help="Generates plots that are used in the Bachelor Thesis report and stores it "
             "in folder /Plots/Report",
    )

    parser.add_argument(
        "-p",
        "--performance_without_tuning",
        type=str,
        help="Outputs detailed scores of the given classifier without doing hyperparameter tuning. "
             " Set clf_name='all' if you want to test all classifiers (file is saved in "
             "Evaluation/Performance/clf_performance_without_hp_tuning_{window_sizes}.txt)",
        metavar='clf_name',
    )

    parser.add_argument(
        "-t",
        "--performance_with_tuning",
        type=str,
        help="Optimizes the given classifier with RandomizedSearchCV and outputs detailed scores."
             " Set clf_name='all' if you want to test all classifiers (file is saved in "
             "Evaluation/Performance/clf_performance_with_hp_tuning_{window_sizes}.txt)",
        metavar='clf_name',
    )

    parser.add_argument(
        "-w",
        "--test_windows",
        type=int,
        nargs=3,
        help="Trains and tests all classifiers with the given window sizes. Provide the windows in seconds. "
             "Stores roc_auc score under /Evaluation/Performance/Windows/",
        metavar=('hw_window', 'crash_window', 'gc_window'),
    )

    parser.add_argument(
        "-g",
        "--leave_one_group_out",
        action="store_true",
        help="Plot performance when leaving out a logfile "
             "vs leaving out a whole user in crossvalidation under Plots/Performance/LeaveOneGroupOut",
    )

    parser.add_argument(
        "-m",
        "--evaluate_lstm",
        type=int,
        nargs=1,
        help="Compile, train and evaluate an LSTM network with n_epochs epochs",
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
             "in folder /Plots/Features",
    )

    parser.add_argument(
        "-l",
        "--generate_plots_about_logfiles",
        action="store_true",
        help="Generates different plots from the logfiles (Look at main.py for details) and stores it "
             "in folder /Plots/Logfiles (Note: Probably use in combination with -n, i.e. without "
             "normalizing heartrate)",
    )

    parser.add_argument(
        "-u",
        "--do_not_use_pre_tuned_hyperparameters",
        action='store_true',
        help="There are some hyperparameters that were tuned on Euler and are used per default. If you want to tune "
             "them manually/on your computer, use this flag"
    )

    parser.add_argument(
        "-s",
        "--use_synthesized_data",
        action="store_true",
        help="Use synthesized data. Might not work with everything."
    )

    parser.add_argument(
        "-n",
        "--do_not_normalize_heartrate",
        action="store_true",
        help="Do not normalize heartrate (e.g. if you want plots or values with real heartrate). Note that for "
             "the feature matrices, we always enforce normalization.",
    )

    parser.add_argument(
        "-d",
        "--debugging",
        action="store_true",
        help="Use only a small part of the data. Mostly for debugging purposes",
    )

    return parser.parse_args()
