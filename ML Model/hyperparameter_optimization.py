"""This module takes a classifier name and n_iter and does RandomizedSearchCV to find the best hyperparameters of the

classifier with this name.
"""

from __future__ import division, print_function  # s.t. division uses float result

import numpy as np
import pandas as pd

from sklearn.model_selection import (RandomizedSearchCV)

import matplotlib.pyplot as plt

import seaborn as sns

import model_factory
import classifiers
import plots_helpers
import features_factory as f_factory
import synthesized_data


def report(results, n_top=3):
    """Prints a  report with the scores from the n_top hyperparameter configurations with the best score

    :param results: cv_results of GridSearch/RandomizedSearchCV
    :param n_top: Best n_top hyperparameter configurations should be displayed

    """
    s = '******** Scores of best ' + str(n_top) + ' hyperparameter configurations ********\n'
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            s += "\tModel with rank: {0}".format(i)
            s += "\n\t\tMean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate])
            s += "\n\t\tParameters: {0}".format(results['params'][candidate])
            s += "\n"

    print(s)


def get_tuned_clf_and_tuned_hyperparameters(X, y, clf_name='svm', verbose=True):
    """This method optimizes hyperparameters with cross-validation using RandomizedSearchCV, optionally creates a ROC curve
        and returns this optimized classifier and the tuned parameters

    :param X: Feature matrix
    :param y: labels
    :param clf_name:  Name of the classifier as given in classifiers.py
    :param verbose: Whether scores of top hyperparameter configurations should be printed out

    :return: optimized classifier, dictionary of tuned_params

    """

    c_classifier = classifiers.get_cclassifier_with_name(clf_name, X, y)

    print('Doing RandomizedSearchCV with n_iter=' + str(c_classifier.num_iter) + ' for ' + clf_name + '...')

    if clf_name == 'Naive Bayes':  # Naive Bayes doesn't have any hyperparameters to tune
        if synthesized_data.synthesized_data_enabled:
            print('in')
            X_n, y_n = f_factory.get_feature_matrix_and_label(False, False, True, True, f_factory.use_reduced_features)  # Use features with boxcox
        else:
            X_n, y_n = f_factory.get_feature_matrix_and_label(True, True, True, True, f_factory.use_reduced_features)  # Use features with boxcox

        c_classifier.clf.fit(X_n, y_n)

        return c_classifier.clf, []

    else:
        clf = RandomizedSearchCV(c_classifier.clf, c_classifier.tuned_params, cv=5,
                                 scoring='roc_auc', n_iter=c_classifier.num_iter)
        clf.fit(X, y)

        if verbose:
            report(clf.cv_results_)

    return clf.best_estimator_, model_factory.get_tuned_params_dict(clf.best_estimator_, list(c_classifier.tuned_params.keys()))


def plot_heat_map_of_grid_search(cv_results, Classifier):
    """Plots a heatmap over the hyperparameters, showing the corresponding roc_auc score
        Problem: We can only show 2 hyperparameters
    :param cv_results: cv_results of RandomizedSearchCV
    :param Classifier: the classfier
    :return:
    """

    params = ([list(set(v.compressed())) for k, v in cv_results.items() if k.startswith('param_')])
    print(params)
    plt.figure()
    results_df = pd.DataFrame(cv_results)
    print(results_df)
    scores = np.array(results_df.mean_test_score).reshape(len(params[0]), len(params[1]))
    sns.heatmap(scores, annot=True,
                xticklabels=params[0], yticklabels=params[1], cmap=plt.cm.RdYlGn)
    plt.title('Grid Search roc_auc Score')
    plots_helpers.save_plot(plt, 'Gridsearch/', Classifier.name + '.pdf')