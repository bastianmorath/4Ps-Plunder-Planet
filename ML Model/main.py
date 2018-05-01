
"""This is an implementation of a ML model to predict the performance
    of the player.
    In particular, the model will predict whether or not the user is going to
    crash into the next obstacle.

    As features, we use:
    ...

"""

from __future__ import division, print_function  # s.t. division uses float result

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight

from sklearn import naive_bayes
from sklearn import svm
from sklearn import neighbors

import time
import numpy as np

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

print('\nSVM with class_weights: ')
cw = class_weight.compute_class_weight('balanced', np.unique(y), y)
class_weight_dict = dict(enumerate(cw))
clf = svm.SVC(class_weight=class_weight_dict, C=1000, kernel='rbf')
# model.apply_cv_model(clf, X, y)


print('\nKNearestNeighbors: ')
clf = neighbors.KNeighborsClassifier()
# model.apply_cv_model(clf, X, y)


print('\nGaussian: ')
clf = naive_bayes.GaussianNB()
# model.apply_cv_model(clf, X, y)

print('\nExtraTreesClassifier with feature selection and class_weights: ')

# model.feature_selection(X, y)


from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC


# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.7, random_state=0)

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()
    cw = class_weight.compute_class_weight('balanced', np.unique(y), y)
    class_weight_dict = dict(enumerate(cw))
    clf = GridSearchCV(SVC(class_weight=class_weight_dict), tuned_parameters, cv=5,
                       scoring='recall')
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred, target_names=['No Crash: ', 'Crash: ']))
    print()