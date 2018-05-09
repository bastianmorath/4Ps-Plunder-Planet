
"""This is the main file to find the best window-sizes. For this, we do RandomSearchCV for the SVM
    classifier and find the roc_auc for different windowsizes.
    You can run this file with the following arguments:

    python hyperparameter_optimization.py hw cw gradient_w
    with
        - hw: The size of the heartrate window
        - cw: The size of the crash window
        - gradient_w:  The size of the window used for all features where we want to compute some sort of gradients
    ...

"""

from __future__ import division, print_function  # s.t. division uses float result

from sklearn import naive_bayes, svm
from sklearn.preprocessing import MinMaxScaler

import time
import numpy as np
from sklearn.utils import class_weight

import ml_model
import setup

import features_factory as f_factory
import globals as gl
import argparse

# Add user-friendly command-line interface to enter windows and RandomSearchCV parameters etc.

parser = argparse.ArgumentParser(description='RandomSearchCV with given arguments')

parser.add_argument('hw', type=int, nargs='?', default=30,
                    help='The size of the heartrate window (Features: TODO')
parser.add_argument('cw', type=int, nargs='?', default=30,
                    help='The size of the crash window (Features: TODO')
parser.add_argument('gradient_w', type=int, nargs='?', default=10,
                    help='The size of the window used for all features where we want to compute some sort of gradients')
args = parser.parse_args()

# Change windows to the values given by the command-line arguments
gl.hw = args.hw
gl.cw = args.cw
gl.gradient_w = args.gradient_w

print('Init dataframes...')

start = time.time()
setup.setup()

X, y = f_factory.get_feature_matrix_and_label()

class_w = class_weight.compute_class_weight('balanced', np.unique(y), y)
class_weight_dict = dict(enumerate(class_w))

clf = svm.SVC(class_weight=class_weight_dict)
clf.fit(X, y)
roc_auc, recall, specificity, precision, conf_mat = ml_model.get_performance(clf, "Naive Bayes", X, y)

s = 'Scores for %s (Windows:  %i, %i, %i): \n\n' \
     '\troc_auc: %.3f, ' \
    'recall: %.3f, ' \
    'specificity: %.3f, ' \
    'precision: %.3f \n\n' \
    '\tConfusion matrix: \t %s \n\t\t\t\t %s\n\n\n'  \
     % ("std. SVM", gl.hw, gl.cw, gl.gradient_w, roc_auc, recall, specificity, precision, conf_mat[0], conf_mat[1])
print(gl.hw, gl.cw, gl.gradient_w)
file = open(gl.working_directory_path + '/window_test_' + str(gl.hw) + '_' + str(gl.cw) + '_' +
            str(gl.gradient_w) + '.txt', 'w+')
file.write(s)

end = time.time()
print('Time elapsed: ' + str(end - start))