
"""This is a first implementation of a ML model to predict the performance
    of the player.
    In particular, the model will predict whether or not the user is going to
    crash into the next obstacle.

    As features, we use:
        - %Crashes in last x seconds
        - mean HR
        - Max/Min HR ratio
        - Crystals (later...)
        - %Points change

"""

from __future__ import division  # s.t. division uses float result

from sklearn.model_selection import cross_val_predict, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight
import numpy as np
import pandas as pd

from sklearn import naive_bayes, metrics
from sklearn import svm
from sklearn import neighbors


import setup
import plots
import test_data
import globals as gl
import features_factory as f_factory
import ml_model as model

# NOTE: heartrate is normalized, i.e. on a scale around ~ 1


''' Get data and create feature matrix and labels
    Column 0: Id/Time
    Column 1: %Crashes in last x seconds
    Column 2: mean heartrate over last y seconds
'''

print('Params: \n\t testing: ' + str(gl.testing) + ', \n\t use_cache: ' + str(gl.use_cache) + ', \n\t test_data: ' +
      str(gl.test_data) + ', \n\t use_boxcox: ' + str(gl.use_boxcox) + ', \n\t plots_enabled: ' + str(gl.plots_enabled))

print('Init dataframes...')


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
    # plots.plot_feature_correlations(X, y)
    # plots.plot_heartrate_histogram()
    plots.plot_feature_distributions(X)
    plots.print_mean_features_crash(X, y)


'''Feature selection'''

# forest = factory.feature_selection(X, y)
# model = SelectFromModel(forest, prefit=True)
# X = model.transform(X)
# print('\n# features after feature-selection: ' + str(X.shape[1]) + '\n')


''' Apply Model with Cross-Validation'''


print('Model fitting...')



print('\nSVM with class_weights: ')
class_weight = class_weight.compute_class_weight('balanced', np.unique(y), y)
class_weight_dict = dict(enumerate(class_weight))
clf = svm.SVC(class_weight=class_weight_dict)
model.apply_cv_groups_model(clf, X, y)


print('\nKNearestNeighbors: ')
clf = neighbors.KNeighborsClassifier()
model.apply_cv_groups_model(clf, X, y)


print('\nGaussian: ')
clf = naive_bayes.GaussianNB()
model.apply_cv_groups_model(clf, X, y)






