
"""This is the main file to find the best window-sizes. For this, we do RandomSearchCV for the SVM
    classifier and find the roc_auc for different windowsizes.

    ...

"""

from __future__ import division, print_function  # s.t. division uses float result

from sklearn import svm

import model_factory
import features_factory as f_factory


def performance_score_for_windows(hw, cw, gradient_w, verbose=True, write_to_file=True):
    print('\tCalculating performance with standard svm for windows %i, %i, %i...' % (hw, cw, gradient_w))

    clf = svm.SVC(class_weight='balanced')

    X, y = f_factory.get_feature_matrix_and_label(verbose=False, cached_feature_matrix=None,
                                                  save_as_pickle_file=False)

    _, _, _, _, _, s = model_factory.get_performance(clf, "SVM (w/ rfb kernel)", X, y,
                                                     hw, cw, gradient_w, False, False)

    if verbose:
        print(s)

    if write_to_file:
        # Write result to a file
        filename = 'window_test_' + str(hw) + '_' + str(cw) + '_' + str(gradient_w) + '.txt'
        model_factory.write_to_file(s, 'Performance/Windows/', filename, 'w+')



