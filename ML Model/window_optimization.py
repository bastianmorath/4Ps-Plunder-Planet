
"""This is the main file to find the best window-sizes. For this, we do RandomizedSearchCV for the SVM
    classifier and find the roc_auc for different windowsizes.

    ...

"""

from __future__ import division, print_function  # s.t. division uses float result

import numpy as np
import model_factory
import features_factory as f_factory
import matplotlib.pyplot as plt

import plots_helpers


def performance_score_for_windows(hw, cw, gradient_w, verbose=True, write_to_file=True):
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


def test_all_windows():
    """ Keeps one window fixed and changes the other two. At the end, it plots the mean value over all
    classifier roc_auc scores with plt.imshow.

    """
    hw = 10

    cw_list = [3, 5, 10, 20, 30, 60]
    gradient_w = [3, 4, 5, 10, 20, 30, 60]
    mean_scores = np.zeros((len(cw_list), len(gradient_w)))
    scores_std = np.zeros((len(cw_list), len(gradient_w)))

    for idx_hw, cw in enumerate(cw_list):
        for idx_cw, gradient_w in enumerate(gradient_w):
            X, y = f_factory.get_feature_matrix_and_label(verbose=True, use_cached_feature_matrix=True,
                save_as_pickle_file=True, h_window=hw, c_window=cw,
                gradient_window=gradient_w)

            auc_mean_scores, auc_std_scores, _ = model_factory. \
                calculate_performance_of_classifiers(X, y, tune_hyperparameters=False,
                reduced_clfs=False, do_write_to_file=False)
            print(np.mean(auc_mean_scores), np.mean(auc_std_scores), idx_hw, idx_cw)
            mean_scores[idx_hw][idx_cw] = np.mean(auc_mean_scores)
            scores_std[idx_hw][idx_cw] = np.mean(auc_std_scores)

    # Plot elements

    plt.imshow(mean_scores, cmap='RdYlGn')
    plt.title('Average classifier performance when using constant gradient_window')
    ax = plt.gca()
    ax.set_xticks(np.arange(0, len(gradient_w), 1))
    ax.set_yticks(np.arange(0, len(cw_list), 1))
    ax.set_xticklabels(gradient_w)
    ax.set_yticklabels(cw_list)
    ax.set_ylabel('Crash window')
    ax.set_xlabel('Gradient window')
    plt.colorbar()
    plots_helpers.save_plot(plt, 'Performance/Windows/', 'windows_const_hw.pdf')

