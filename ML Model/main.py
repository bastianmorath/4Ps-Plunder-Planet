
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

from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight
from sklearn.svm import LinearSVC
import numpy as np

from sklearn import naive_bayes, metrics
from sklearn import svm
from sklearn import neighbors


import setup
import plots
import factory
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


def apply_model(model, x_new):
    y_pred = cross_val_predict(model, x_new, y, cv=10)
    conf_mat = confusion_matrix(y, y_pred)

    precision = metrics.precision_score(y, y_pred)
    recall = metrics.recall_score(y, y_pred)
    specificity = conf_mat[0, 0]/(conf_mat[0, 0]+conf_mat[0, 1])
    print('Note: \n\t Recall = %0.3f = Probability of, given a crash, a crash is correctly predicted; '
          '\n\t Specificity = %0.3f = Probability of, given no crash, no crash is correctly predicted;'
          '\n\t Precision = %.3f = Probability that, given a crash is predicted, a crash really happened; [n'
          % (recall, specificity, precision))

    print('\tConfusion matrix: \n\t\t' + str(conf_mat).replace('\n', '\n\t\t'))

    predicted_accuracy = round(metrics.accuracy_score(y, y_pred) * 100, 2)
    null_accuracy = round(max(np.mean(y), 1 - np.mean(y)) * 100, 2)

    print('Correctly classified data: ' + str(predicted_accuracy) + '% (vs. null accuracy: ' + str(null_accuracy) + '%)')


print('\nSVM with class_weights: ')
class_weight = class_weight.compute_class_weight('balanced', np.unique(y), y)
class_weight_dict = dict(enumerate(class_weight))
clf = svm.SVC(class_weight=class_weight_dict)
apply_model(clf, X)


print('\nKNearestNeighbors with class_weights: ')
clf = neighbors.KNeighborsClassifier()
apply_model(clf, X)


print('\nGaussian with class_weights: ')
clf = naive_bayes.GaussianNB()
apply_model(clf, X)






