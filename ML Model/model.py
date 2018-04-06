
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

    SVM as the binary classifier and 10-fold Cross-Validation is used
'''

from __future__ import division  # s.t. division uses float result

from sklearn import svm
from sklearn.model_selection import train_test_split  # IMPORTANT: use sklearn.cross_val for of Euler
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn import metrics
from collections import Counter

import optunity
import optunity.metrics

import globals as gl
import features_factory as f_factory
import factory

crash_window = 30  # Over how many preceeding seconds should %crashes be calculated?
heartrate_window = 60  # Over how many preceeding seconds should the heartrate be averaged?


''' Get data and create feature matrix and labels
    Column 0: Id/Time
    Column 1: %Crashes in last x seconds
    Column 2: mean heartrate over last y seconds
'''

print('Init dataframes...')
print('Crash_window: ' + str(crash_window) + ', Heartrate_window: ' + str(heartrate_window))

gl.init(True, crash_window, heartrate_window)  # Entire dataframe with features-column

print('Creating feature matrix...')

(X, y) = f_factory.get_feature_matrix_and_label()
factory.plot_features_with_labels(X, y)


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


# TODO: Other metrics, e.g. precision
@optunity.cross_validated(x=X_train, y=y_train, num_folds=10, num_iter=1)
def svm_auc(x_train, y_train, x_test, y_test, log_c, log_gamma):
    model = svm.SVC(C=10 ** log_c, gamma=10 ** log_gamma).fit(x_train, y_train)
    decision_values = model.decision_function(x_test)
    return optunity.metrics.roc_auc(y_test, decision_values)


# perform tuning
optimal_rbf_pars, info, _ = optunity.maximize(svm_auc, num_evals=20, log_c=[-20, 0], log_gamma=[-15, 0])

# train model on the full training set with tuned hyperparameters
optimal_model = svm.SVC(C=10 ** optimal_rbf_pars['log_c'], gamma=10 ** optimal_rbf_pars['log_gamma'],
                            class_weight={0: 1, 1: 3}).fit(X_train, y_train)

optimal_model.fit(X_train, y_train)
print("Optimal parameters (10e): " + str(optimal_rbf_pars))
print("AUROC of tuned SVM with RBF kernel: %1.3f" % info.optimum)


# Predict values on test data
y_test_predicted = optimal_model.predict(X_test)

# Print result as %correctly predicted labels
print('Unique prediction values: ' + str(Counter(y_test_predicted).keys())) # equals to list(set(words))

percentage = metrics.accuracy_score(y_test, y_test_predicted)
print('Percentage of correctly classified data: ' + str(percentage))


'''Plot features with infos
'''


factory.plot_features(optimal_rbf_pars['log_gamma'], optimal_rbf_pars['log_c'], info.optimum, percentage )




