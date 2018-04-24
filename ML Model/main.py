
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

from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.model_selection import train_test_split  # IMPORTANT: use sklearn.cross_val for of Euler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn import metrics

import numpy as np

from sklearn import naive_bayes
from sklearn import svm
from sklearn import neighbors


import setup
import plots
import test_data
import globals as gl
import features_factory as f_factory


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
    if gl.plots_enabled:
        plots.plot_hr_of_dataframes()

print('Creating feature matrix...\n')


X, y = f_factory.get_feature_matrix_and_label()
print('Feature matrix X: \n' + str(X))
print('labels y:\n' + str(y) + '\n')


if gl.plots_enabled:
    print('Plotting...')
    # plots.plot_feature_correlations(X, y)
    # plots.plot_heartrate_histogram()
    plots.plot_feature_distributions(X)
    # plots.print_mean_features_crash(X, y)

'''Preprocess data'''
# scaler = StandardScaler().fit(X)  # Because we likely have a Gaussian distribution
# X = scaler.transform(X)
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)  # Rescale between 0 and 1


''' Apply Model with Cross-Validation'''

# factory.test_windows()
print('Model fitting...\n')


model = svm.SVC()
model = neighbors.KNeighborsClassifier()
model = naive_bayes.GaussianNB()


y_pred = cross_val_predict(model, X, y, cv=10)
accuracy = round(metrics.accuracy_score(y, y_pred) * 100, 2)
null_accuracy = round( max(np.mean(y), 1 - np.mean(y)) * 100, 2)

print('Accuracy: ' + str(accuracy) + '% (vs. Null accuracy of ' + str(null_accuracy) + '%\n')



conf_mat = confusion_matrix(y, y_pred)
print('Confusion matrix: \n' + str(conf_mat))

# factory.print_confidentiality_scores(X_train, X_test, y_train, y_test)



