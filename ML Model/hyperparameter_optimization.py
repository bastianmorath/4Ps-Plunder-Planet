"""This module takes a classifier name and n_iter and does RandomSearchCV to find the best hyperparameters of the

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
import plots
import features_factory as f_factory


def report(results, n_top=3):
    """Prints a  report with the scores from the n_top hyperparameter configurations with the best score

    :param results: cv_results of GridSearch/RandomSearchCV
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


def calculate_performance_of_all_classifiers_with_optimized_hyperparameters(X, y, num_iter=20):
    """Does Hyperparameter optimization for all classifiers and plots roc_aucs in a abarchart and writes a detailed
    report into a file

    :param X: Feature matrix
    :param y: labels
    :param num_iter: number of iterations RandomSearchCv should do

    """

    clf_list = []
    clf_names = []

    for name in classifiers.names:
        clf_list.append(get_clf_with_optimized_hyperparameters(X, y, name, num_iter, verbose=False))
        clf_names.append(name)

    model_factory.print_and_plot_performance_of_classifiers(clf_list, clf_names, X, y, True)


def get_clf_with_optimized_hyperparameters(X, y, clf_name='svm', num_iter=20, verbose=True):
    """This method optimizes hyperparameters with cross-validation using RandomSearchCV, optionally creates a ROC curve
        and returns this optimized classifier and a report

    :param X: Feature matrix
    :param y: labels
    :param clf_name:  Name of the classifier as given in classifiers.py
    :param num_iter: Number of iterations the RandomSearchCV should perform
    :param verbose: Whether scores of top hyperparameter configurations should be printed out
    :param create_roc: Create roc_curve of optimized classifier

    :return: optimized classifier

    """

    c_classifier = classifiers.get_clf_with_name(clf_name, X, y)

    print('Doing RandomSearchCV for ' + clf_name + '...')

    # RandomSearchCV

    if clf_name == 'Naive Bayes':  # Naive Bayes doesn't have any hyperparameters to tune

        clf = c_classifier.clf
        X, y = f_factory.get_feature_matrix_and_label(True, True, True, True, True)
        clf.fit(X, y)

    else:
        clf = RandomizedSearchCV(c_classifier.clf, c_classifier.tuned_params, cv=5,
                                 scoring='roc_auc', n_iter=num_iter)
        clf.fit(X, y)

        if verbose:
            report(clf.cv_results_)

    return clf


def plot_heat_map_of_grid_search(cv_results, Classifier):
    """Plots a heatmap over the hyperparameters, showing the corresponding roc_auc score
        Problem: We can only show 2 hyperparameters
    :param cv_results: cv_results of RandomSearchCV
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
    plots.save_plot(plt, 'Gridsearch/', Classifier.name + '.pdf')