
"""This is an implementation of a ML model to predict the performance
    of the player.
    In particular, the model will predict whether or not the user is going to
    crash into the next obstacle.

    As features, we use:
    ...

"""

from __future__ import division, print_function  # s.t. division uses float result

import sklearn as sk
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight

from sklearn import naive_bayes
from sklearn import svm
from sklearn import neighbors
from sklearn import discriminant_analysis


import time
import numpy as np

import setup
import plots
import test_data
import globals as gl
import features_factory as f_factory
import ml_model


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# NOTE: heartrate is normalized, i.e. on a scale around ~ 1

''' Get data and create feature matrix and labels
    Column 0: Id/Time
    Column 1: %Crashes in last x seconds
    Column 2: mean heartrate over last y seconds
'''

print('Params: \n\t testing: ' + str(gl.testing) + ', \n\t use_cache: ' + str(gl.use_cache) + ', \n\t test_data: ' +
      str(gl.test_data) + ', \n\t use_boxcox: ' + str(gl.use_boxcox) + ', \n\t plots_enabled: ' + str(gl.plots_enabled))

print('Init dataframes...')

start = time.time()
if gl.test_data:
    test_data.init_with_testdata()
else:
    setup.setup()
    # if gl.plots_enabled:
        # plots.plot_hr_of_dataframes()

# factory.print_keynumbers_logfiles()
# plots.plot_hr_of_dataframes()

print('Creating feature matrix...\n')


X, y = f_factory.get_feature_matrix_and_label()
# print('Feature matrix X: \n' + str(X))
# print('labels y:\n' + str(y) + '\n')

'''Preprocess data'''
# scaler = StandardScaler().fit(X)  # Because we likely have a Gaussian distribution
# X = scaler.transform(X)
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)  # Rescale between 0 and 1

if gl.plots_enabled:
    print('Plotting...')
    # plots.plot_heartrate_histogram()
    plots.plot_feature_distributions(X)
    plots.print_mean_features_crash(X, y)


''' Apply Model with Cross-Validation'''


print('Model fitting...')

cw = class_weight.compute_class_weight('balanced', np.unique(y), y)
class_weight_dict = dict(enumerate(cw))

feature_selection = False
if feature_selection:
    print('Feature selection with LinearSVC l1-loss: \n')
    clf = svm.LinearSVC(class_weight=class_weight_dict, penalty="l1", dual=False).fit(X, y)
    model = SelectFromModel(clf, prefit=True)
    X = model.transform(X)
    features = [f_factory.feature_names[i] for i in range(0, len(f_factory.feature_names)) if not model.get_support()[i]]
    print('Features not selected:: ' + str(features))


print('\nLinearSVC with class_weights: ')
clf = svm.LinearSVC(class_weight=class_weight_dict).fit(X, y)
ml_model.apply_cv_model(clf, X, y)


print('\nSVC with class_weights: ')
clf = svm.SVC(class_weight=class_weight_dict)
ml_model.apply_cv_groups_model(clf, X, y)


# Set the parameters by cross-validation
# tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
#                     'C': [0.1, 1, 10, 100, 1000]}]
# clf = ml_model.param_estimation_grid_cv(X, y, clf, tuned_parameters)
# print('\nSVC with class_weights and hypertuning: ')

# ml_model.apply_cv_model(clf, X, y)


print('\nKNearestNeighbors: ')
clf = neighbors.KNeighborsClassifier()
ml_model.apply_cv_model(clf, X, y)


print('\nGaussian: ')
clf = naive_bayes.GaussianNB()
ml_model.apply_cv_model(clf, X, y)


print('\nQuadratic Discriminant analysis: ')
clf = discriminant_analysis.QuadraticDiscriminantAnalysis()
ml_model.apply_cv_model(clf, X, y)


# print('\nExtraTreesClassifier with feature selection and class_weights: ')

# ml_model.feature_selection(X, y)


# ml_model.test_classifiers(X, y)


end = time.time()
print('Time elapsed: ' + str(end - start))