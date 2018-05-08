
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


from sklearn.preprocessing import MinMaxScaler

import time

import setup

import features_factory as f_factory
import grid_search
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


print('Creating feature matrix...\n')

X, y = f_factory.get_feature_matrix_and_label()


'''Preprocess data'''

scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)  # Rescale between 0 and 1


'''Do RandomSearchCV'''
grid_search.do_grid_search_for_classifiers(X, y, 0, 1)


end = time.time()
print('Time elapsed: ' + str(end - start))