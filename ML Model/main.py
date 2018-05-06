
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
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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


names = ["Linear SVM", "RBF SVM", "Nearest Neighbor", "Naive Bayes", "QDA"]

classifiers = [
        svm.LinearSVC(class_weight=class_weight_dict),
        svm.SVC(class_weight=class_weight_dict),
        neighbors.KNeighborsClassifier(),
        naive_bayes.GaussianNB(),
        discriminant_analysis.QuadraticDiscriminantAnalysis()
        ]

auc_scores = []

for name, clf in zip(names, classifiers):
    print(name+'...')
    clf.fit(X, y)
    auc_scores.append(ml_model.apply_cv_per_user_model(clf, name, X, y, per_logfile=False)[0])
    # ml_model.plot_roc_curve(clf, X, y, name)

plt = plots.plot_barchart(title='roc_auc w/out hyperparameter tuning',
                          x_axis_name='',
                          y_axis_name='roc_auc',
                          x_labels=names,
                          values=auc_scores,
                          lbl=None,
                          )

plt.savefig(gl.working_directory_path + '/Plots/roc_auc_per_classifier.pdf')

grid_search = False
if grid_search:
    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [10, 100, 1000]}]
    clf = ml_model.param_estimation_grid_cv(X, y, classifiers[1], tuned_parameters)
    print('\nSVC with class_weights and hypertuning: ')

    auc, _, _, _ = ml_model.get_performance(clf, names[1], X, y)
    print(auc)

# print('\nExtraTreesClassifier with feature selection and class_weights: ')

# ml_model.feature_selection(X, y)


end = time.time()
print('Time elapsed: ' + str(end - start))