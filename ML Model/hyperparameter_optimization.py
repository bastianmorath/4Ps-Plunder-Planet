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


def get_performance_of_all_clf_with_optimized_hyperparameters(X, y, num_iter=20):
    """Does Hyperparameter optimization for all classifiers and returns for the classifier names,
     roc_auc, precision, specificity, recall and the conf_matrix a list each

    :param X: Feature matrix
    :param y: labels
    :param num_iter: number of iterations RandomSearchCv should do

    :return: lists for names, scores, optimal_params and conf_mats

    """

    scores = []
    optimal_params = []
    conf_mats = []

    for name in classifiers.names:
        optimal_clf, rep = get_clf_with_optimized_hyperparameters(X, y, name, num_iter)
        # plot_heat_map_of_grid_search(optimal_clf.cv_results_, Classifier)
        optimal_params.append(optimal_clf.best_params_)

        roc_auc, recall, specificity, precision, conf_mat = \
            model_factory.get_performance(optimal_clf, name, X, y, verbose=False)

        scores.append([roc_auc, recall, specificity, precision])
        conf_mats.append(conf_mat)

    return classifiers.names, scores, optimal_params, conf_mats


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


def get_clf_with_optimized_hyperparameters(X, y, clf_name='svm', num_iter=20):
    """This method optimizes hyperparameters with cross-validation, which is done using RandomSearchCV and
    returns this optimized classifier and a report

    :param X: Feature matrix
    :param y: labels
    :param clf_name:  Name of the classifier as given in classifiers.py
    :param num_iter: Number of iterations the RandomSearchCV should perform

    :return: optimized classifier and report (

    """
    c_classifier = classifiers.get_clf_with_name(clf_name, X, y)

    print('Doing RandomSearchCV for ' + clf_name + '...\n')


    # Naive Bayes doesn't have any hyperparameters to tune
    if clf_name == 'Naive Bayes':
        clf = c_classifier.clf
        clf.fit(X, y)

        # print('Best parameters: ' + str(model_factory.get_tuned_params_dict(clf, c_classifier.tuned_params)) + '\n')
    else:
        clf = RandomizedSearchCV(c_classifier.clf, c_classifier.tuned_params, cv=2,
                                 scoring='roc_auc', n_iter=3)  # TODO: Change to num_iter and 10
        clf.fit(X, y)

        # print('Best parameters: ' + str(model_factory.get_tuned_params_dict(clf.best_estimator_,
        #                                                                     c_classifier.tuned_params)) + '\n')
        report(clf.cv_results_)

    # Naive Bayes doesn't have any hyperparameters to tune
    if clf_name == 'Naive Bayes':
        rep  = model_factory.get_performance(clf, clf_name, X, y)[5]
    else:
        rep = model_factory.get_performance(clf.best_estimator_, clf_name, X, y, list(c_classifier.tuned_params.keys()))[5]

    return clf, rep


def evaluate_model(results, n_top=3):
    """Returns the n_top hyperparameter configurations with the best score and its scores

        :param results: cv_results of GridSearch/RandomSearchCV
        :param n_top: Best n_top hyperparameter configurations should be displayed

        :return report
        """
    s = ''
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            s += "Model with rank: {0}".format(i)
            s += "\n\tMean validation score: {0:.3f} (std: {1:.3f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate])
            s += "\n\tParameters: {0}".format(results['params'][candidate])
            s += "\n"

    return s
def plot_heat_map_of_grid_search(cv_results, Classifier):
    """Plots a heatmap over the hyperparameters, showing the corresponding roc_auc score

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