
"""
This is the main file to find the best window-sizes. For this, we do RandomizedSearchCV for the SVM
classifier and find the roc_auc for different windowsizes.

"""

from __future__ import division  # s.t. division uses float result
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

import features_factory as f_factory
import model_factory
import plots_helpers


def performance_score_for_windows(hw, cw, gradient_w, verbose=True, write_to_file=True):
    """
    Prints and writes to a file scores of all classifiers with the given window sizes

    :param hw: Heartrate window size
    :param cw: Crash window size
    :param gradient_w: gradient window size
    :param verbose: Whether scores should be printed out
    :param write_to_file: Whehter scores should be written to a file

    """

    print('Testing window size ' + str(hw) + ', ' + str(cw) + ', ' + str(gradient_w) + '...')

    X, y = f_factory.get_feature_matrix_and_label(verbose=True, use_cached_feature_matrix=True,
                                                  save_as_pickle_file=True, h_window=hw, c_window=cw,
                                                  gradient_window=gradient_w)
    auc_mean_scores, auc_std_scores, s = model_factory.\
        calculate_performance_of_classifiers(X, y, tune_hyperparameters=False,
                                             reduced_clfs=False, do_write_to_file=False)

    if verbose:
        print(s)

    if write_to_file:
        # Write result to a file
        filename = 'window_test_' + str(hw) + '_' + str(cw) + '_' + str(gradient_w) + '.txt'
        model_factory.write_to_file(s, 'Performance/Windows/', filename, 'w+')


# Note: Not used in the main program
def test_all_windows():
    """
    Keeps one window fixed and changes the other two. At the end, it plots the mean value over all
    classifier roc_auc scores with plt.imshow.

    """
    print("\n################# Testing all window sizes #################\n")

    const_window = 'gradient_w'

    const_w = 10
    list_1 = [5, 10, 20, 30, 50, 60, 80, 100, 120]
    list_2 = list_1[::-1]

    if const_window == 'hw':
        name1 = 'Crash window (s)'
        name2 = 'Gradient window (s)'
        filename = 'windows_const_hw.pdf'
    elif const_window == 'cw':
        name1 = 'Heartrate window (s)'
        name2 = 'Gradient window (s)'
        filename = 'windows_const_cw.pdf'
    else:
        name1 = 'Crash window'
        name2 = 'Heartrate window'
        filename = 'windows_const_gradient_w.pdf'

    mean_scores = np.zeros((len(list_1), len(list_2)))

    for idx_w1, w1 in enumerate(list_1):
        for idx_w2, w2 in enumerate(list_2):
            if const_window == 'hw':
                X, y = f_factory.get_feature_matrix_and_label(verbose=True, use_cached_feature_matrix=True,
                                                              save_as_pickle_file=True, h_window=const_w, c_window=w1,
                                                              gradient_window=w2)

                auc_mean_scores, auc_std_scores, _ = model_factory. \
                    calculate_performance_of_classifiers(X, y, tune_hyperparameters=False,
                                                         reduced_clfs=False, do_write_to_file=False)

                print(np.max(auc_mean_scores))
                print(np.mean(auc_mean_scores))
                print(np.min(auc_mean_scores))
                print(auc_mean_scores[0])
                mean_scores[idx_w1][idx_w2] = auc_mean_scores[0]  # np.mean(auc_mean_scores)
            elif const_window == 'cw':
                X, y = f_factory.get_feature_matrix_and_label(verbose=True, use_cached_feature_matrix=True,
                                                              save_as_pickle_file=True, h_window=w1, c_window=const_w,
                                                              gradient_window=w2)

                auc_mean_scores, auc_std_scores, _ = model_factory. \
                    calculate_performance_of_classifiers(X, y, tune_hyperparameters=False,
                                                         reduced_clfs=False, do_write_to_file=False)

                print(np.max(auc_mean_scores))
                print(np.mean(auc_mean_scores))
                print(np.min(auc_mean_scores))
                print(auc_mean_scores[0])
                mean_scores[idx_w1][idx_w2] = auc_mean_scores[0]  # np.mean(auc_mean_scores)
            else:
                X, y = f_factory.get_feature_matrix_and_label(verbose=True, use_cached_feature_matrix=True,
                                                              save_as_pickle_file=True, h_window=w1, c_window=w2,
                                                              gradient_window=const_w)

                auc_mean_scores, auc_std_scores, _ = model_factory. \
                    calculate_performance_of_classifiers(X, y, tune_hyperparameters=False,
                                                         reduced_clfs=False, do_write_to_file=False)

                print(np.max(auc_mean_scores))
                print(np.mean(auc_mean_scores))
                print(np.min(auc_mean_scores))
                mean_scores[idx_w1][idx_w2] = auc_mean_scores[0]

    mean_scores = np.fliplr(np.flipud(mean_scores))  # Flip to plot it correctly

    # Plot elements
    plt.subplot()
    plt.imshow(mean_scores, cmap='RdYlGn')
    plt.title('Average classifier performance when using constant ' + const_window)
    ax = plt.gca()
    ax.set_xticks(np.arange(0, len(list_1), 1))
    ax.set_yticks(np.arange(0, len(list_2), 1))
    ax.set_xticklabels(list_1)
    ax.set_yticklabels(list_2)
    ax.set_ylabel(name1)
    ax.set_xlabel(name2)
    plt.colorbar()
    plots_helpers.save_plot(plt, 'Performance/Windows/', filename)
