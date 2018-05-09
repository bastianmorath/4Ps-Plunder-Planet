
"""This is an implementation of a ML model to predict the performance
    of the player.
    In particular, the model will predict whether or not the user is going to
    crash into the next obstacle.

    As features, we use:
    ...

"""

from __future__ import division, print_function  # s.t. division uses float result


from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight

from sklearn import naive_bayes
from sklearn import svm
from sklearn import neighbors
from sklearn import discriminant_analysis


import time
import numpy as np
import sys

import setup
import plots
import test_data
import globals as gl
import features_factory as f_factory
import ml_model

import warnings
import factory


def warn(*args, **kwargs):
    pass


warnings.warn = warn

print('Number of arguments:', len(sys.argv), 'arguments.')


print('Params: \n\t testing: ' + str(gl.testing) + ', \n\t use_cache: ' + str(gl.use_cache) + ', \n\t test_data: ' +
      str(gl.test_data) + ', \n\t use_boxcox: ' + str(gl.use_boxcox) + ', \n\t plots_enabled: ' + str(gl.plots_enabled)
      + ', \n\t reduced_features: ' + str(gl.reduced_features))

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


X, y = f_factory.get_feature_matrix_and_label()

if gl.plots_enabled:
    print('Plotting...')
    # plots.plot_heartrate_histogram()
    plots.plot_feature_distributions(X)
    plots.print_mean_features_crash(X, y)


''' Apply Model with Cross-Validation'''


print('Model fitting...')

cw = class_weight.compute_class_weight('balanced', np.unique(y), y)
class_weight_dict = dict(enumerate(cw))

# Does feature selection with SVM l1 loss
feature_selection = False
if feature_selection:
    print('Feature selection with LinearSVC l1-loss: \n')
    clf = svm.LinearSVC(class_weight=class_weight_dict, penalty="l1", dual=False).fit(X, y)
    model = SelectFromModel(clf, prefit=True)
    X = model.transform(X)
    features = [f_factory.feature_names[i] for i in range(0, len(f_factory.feature_names)) if not model.get_support()[i]]
    print('Features not selected:: ' + str(features))


# Plots performance of the given classifiers in a barchart for comparison
plot_classifier_scores = True
if plot_classifier_scores:
    names = ["Linear SVM", "RBF SVM", "Nearest Neighbor", "Naive Bayes", "QDA"]

    classifiers = [
        svm.LinearSVC(class_weight=class_weight_dict),
        svm.SVC(class_weight=class_weight_dict),
        neighbors.KNeighborsClassifier(),
        naive_bayes.GaussianNB(),
        discriminant_analysis.QuadraticDiscriminantAnalysis()
    ]
    auc_scores = []
    auc_std = []
    for name, clf in zip(names, classifiers):
        print(name+'...')
        # If NaiveBayes classifier is used, then use Boxcox since features must be gaussian distributed
        if name == 'Naive Bayes':
            old_bx = gl.use_boxcox
            gl.use_boxcox = True
            X, _ = f_factory.get_feature_matrix_and_label()
            gl.use_boxcox = old_bx

        clf.fit(X, y)
        auc_scores.append(ml_model.get_performance(clf, name, X, y)[0])

        ml_model.plot_roc_curve(clf, X, y, name)

    # Plots roc_auc for the different classifiers
    plt = plots.plot_barchart(title='roc_auc w/out hyperparameter tuning',
                              x_axis_name='',
                              y_axis_name='roc_auc',
                              x_labels=names,
                              values=auc_scores,
                              lbl=None,
                              )
    plt.savefig(gl.working_directory_path + '/Classifier Performance/roc_auc_per_classifier.pdf')

end = time.time()
print('Time elapsed: ' + str(end - start))