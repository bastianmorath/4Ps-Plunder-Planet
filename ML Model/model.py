
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

import setup
import plots
import test_data
import globals as gl
import factory
import features_factory as f_factory


# NOTE: heartrate is normalized, i.e. on a scale around ~ 1


''' Get data and create feature matrix and labels
    Column 0: Id/Time
    Column 1: %Crashes in last x seconds
    Column 2: mean heartrate over last y seconds
'''

print('Params: \n\t testing: ' + str(gl.testing) + ', \n\t use_cache: ' + str(gl.use_cache) + ', \n\t test_data: ' +
      str(gl.test_data) + ', \n\t use_boxcox: ' + str(gl.use_boxcox))

print('Init dataframes...')


if gl.test_data:
    test_data.init_with_testdata()
else:
    setup.setup()
    # plots.plot_hr_of_dataframes()

print('Creating feature matrix...')


X, y = f_factory.get_feature_matrix_and_label()

print('Feature matrix X: \n' + str(X))
print('labels y:\n' + str(y))


# plots.plot_feature_correlations(X, y)
# plots.plot_heartrate_histogram()
# plots.plot_feature_distributions(X)
# plots.print_mean_features_crash(X, y)

'''Preprocess data'''
# scaler = StandardScaler().fit(X)  # Because we likely have a Gaussian distribution
# X = scaler.transform(X)
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)  # Rescale between 0 and 1


''' Apply Model with Cross-Validation'''

factory.test_windows()
print('Model fitting with ' + str(gl.model))

X_train, X_test, y_train, y_test = train_test_split(
             X, y, test_size=0.3)


model = gl.model(X_train, y_train)

# Predict values on test data
y_test_predicted = model.predict(X_test)

model.print_score(y_test, y_test_predicted)

# factory.print_confidentiality_scores(X_train, X_test, y_train, y_test)



