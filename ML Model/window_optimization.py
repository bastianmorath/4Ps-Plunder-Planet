
"""This is the main file to find the best window-sizes. For this, we do RandomSearchCV for the SVM
    classifier and find the roc_auc for different windowsizes.

    ...

"""

from __future__ import division, print_function  # s.t. division uses float result

from sklearn import svm
import numpy as np
from sklearn.utils import class_weight

import model_factory
import globals as gl


def write_window_scores_to_file(X, y, hw, cw, gradient_w):
    gl.hw = hw
    gl.cw = cw
    gl.gradient_w = gradient_w

    class_w = class_weight.compute_class_weight('balanced', np.unique(y), y)
    class_weight_dict = dict(enumerate(class_w))

    clf = svm.SVC(class_weight=class_weight_dict)
    clf.fit(X, y)
    roc_auc, recall, specificity, precision, conf_mat = model_factory.get_performance(clf, "Naive Bayes", X, y)

    s = 'Scores for %s (Windows:  %i, %i, %i): \n\n' \
        '\troc_auc: %.3f, ' \
        'recall: %.3f, ' \
        'specificity: %.3f, ' \
        'precision: %.3f \n\n' \
        '\tConfusion matrix: \t %s \n\t\t\t\t %s\n\n\n'  \
        % ("std. SVM", gl.hw, gl.cw, gl.gradient_w, roc_auc, recall, specificity, precision, conf_mat[0], conf_mat[1])

    file = open(gl.working_directory_path + '/window_test_' + str(gl.hw) + '_' + str(gl.cw) + '_' +
                str(gl.gradient_w) + '.txt', 'w+')
    file.write(s)
