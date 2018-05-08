
"""This is the main file to do Hyperparameter tuning over the classifiers. Uses standard windows(40,40,10)
    You can run this file with the following arguments:

    python hyperparameter_optimization.py clf_id n_iter hw cw gradient_w
    with
        - clf_id which classifier to test (-1 for all)
        - n_iter: #iterations RandomSearchCV should do
    ...

"""

from __future__ import division, print_function  # s.t. division uses float result


from sklearn.preprocessing import MinMaxScaler

import time
import argparse

import setup

import features_factory as f_factory
import grid_search


# Add user-friendly command-line interface to enter windows and RandomSearchCV parameters etc.

parser = argparse.ArgumentParser(description='RandomSearchCV with given arguments')
parser.add_argument('clf_id', type=int, nargs='?', default=-1,
                    help='id of classifier to test. -1 if you want to test all')
parser.add_argument('n_iter', type=int, nargs='?', default=20,
                    help='#iterations RandomSearchCV should do')
args = parser.parse_args()

print('Init dataframes...')

start = time.time()
setup.setup()


print('Creating feature matrix...\n')

X, y = f_factory.get_feature_matrix_and_label()


'''Preprocess data'''

scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)  # Rescale between 0 and 1


'''Do RandomSearchCV'''

grid_search.do_grid_search_for_classifiers(X, y, int(args.clf_id), int(args.n_iter))


end = time.time()
print('Time elapsed: ' + str(end - start))