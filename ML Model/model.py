
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

from sklearn.model_selection import train_test_split  # IMPORTANT: use sklearn.cross_val for of Euler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import metrics
from collections import Counter

import numpy as np

import setup
import plots
import SVM_model
import test_data
import globals as gl
import features_factory as f_factory


# NOTE: heartrate is normalized, i.e. on a scale around ~ 1

''' Get data and create feature matrix and labels
    Column 0: Id/Time
    Column 1: %Crashes in last x seconds
    Column 2: mean heartrate over last y seconds
'''

print('Init dataframes...')

if gl.test_data:
    test_data.init_with_testdata()
else:
    setup.setup()
    # plots.plot_hr_of_dataframes()



print('Creating feature matrix...')

(X, y) = f_factory.get_feature_matrix_and_label()
# plots.plot_features_with_labels(X, y) # WARNING: Only works with non_testdata (since we don't have windows otherwise)
# plots.plot_heartrate_histogram()
plots.plot_feature_distributions(X, y)

'''Preprocess data'''

print('Preprocessing data...')

scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)  # Rescale between 0 and 1
scaler = StandardScaler().fit(X)  # Because we likely have a Gaussian distribution
X = scaler.transform(X)


''' Apply Model with Cross-Validation'''

print('Cross Validation and hyperparameter tuning...')


X_train, X_test, y_train, y_test = train_test_split(
             X, y, test_size=0.3, random_state=42)

model = SVM_model.SVMModel(X_train, y_train)

# Predict values on test data
y_test_predicted = model.predict(X_test)

# Print result as %correctly predicted labels
print('Uniquely predicted values: ' + str(Counter(y_test_predicted).most_common(2)))

print('roc-auc-score: ' + str(metrics.roc_auc_score(y_test, y_test_predicted)))

percentage = metrics.accuracy_score(y_test, y_test_predicted)
print('Percentage of correctly classified data: ' + str(percentage))





