
'''This is a first implementation of a ML model to predict the performance
    of the player.
    In particular, the model will predict whether or not the user is going to
    crash into the next obstacle.

    As features, we use:
        - %Crashes in last x seconds
        - mean HR
        - Max/Min HR ratio
        - Crystals (later...)
        - %Points change


'''

from __future__ import division  # s.t. division uses float result

from sklearn.model_selection import train_test_split  # IMPORTANT: use sklearn.cross_val for of Euler
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn import metrics
from collections import Counter



import globals as gl
import features_factory as f_factory
import factory

import SVM_model

test_data = False


crash_window = 30  # Over how many preceeding seconds should %crashes be calculated?
heartrate_window = 60  # Over how many preceeding seconds should the heartrate be averaged?


''' Get data and create feature matrix and labels
    Column 0: Id/Time
    Column 1: %Crashes in last x seconds
    Column 2: mean heartrate over last y seconds
'''

print('Init dataframes...')
print('Crash_window: ' + str(crash_window) + ', Heartrate_window: ' + str(heartrate_window))

if test_data:
    gl.init_with_testdata(crash_window, heartrate_window)
else:
    gl.init(True, crash_window, heartrate_window)  # Entire dataframe with features-column

print('Creating feature matrix...')

(X, y) = f_factory.get_feature_matrix_and_label()
factory.plot_features_with_labels(X, y) # WARNING: Only works with non_testdata (since we don't have windows otherwise...)


'''Preprocess data
'''

print('Preprocessing data...')

scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)  # Rescale between 0 and 1
scaler = StandardScaler().fit(X)  # Because we likely have a Gaussian distribution
X = scaler.transform(X)


''' Apply SVM Model with Cross-Validation
'''


print('Cross Validation and hyperparameter tuning...')


X_train, X_test, y_train, y_test = train_test_split(
             X, y, test_size=0.3, random_state=0)

model = SVM_model.SVM_Model(X_train, y_train)

# Predict values on test data
y_test_predicted = model.predict(X_test)

# Print result as %correctly predicted labels
print('Unique prediction values: ' + str(Counter(y_test_predicted).keys())) # equals to list(set(words))

percentage = metrics.accuracy_score(y_test, y_test_predicted)
print('Percentage of correctly classified data: ' + str(percentage))





