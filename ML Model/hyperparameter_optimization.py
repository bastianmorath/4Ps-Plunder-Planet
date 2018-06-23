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


def get_performance_of_all_clf_with_optimized_hyperparameters(X, y, num_iter=20):
    """Does Hyperparameter optimization for all classifiers and returns for the classifier names,
     roc_auc, precision, specificity, recall and the conf_matrix a list each

    :param X: Feature matrix
    :param y: labels
    :param num_iter: number of iterations RandomSearchCv should do

    :return: lists of names, scores, optimal_params and conf_mats

    """

    scores = []
    optimal_params = []
    conf_mats = []

    for name in classifiers.names:
        optimal_clf, roc_auc, recall, specificity, precision, conf_mat, rep = get_clf_with_optimized_hyperparameters(X, y, name, num_iter, verbose=False)
        # plot_heat_map_of_grid_search(optimal_clf.cv_results_, Classifier) # TODO maybe add
        if name == 'Naive Bayes':
            optimal_params.append([])
        else:
            optimal_params.append(optimal_clf.best_params_)

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


def get_clf_with_optimized_hyperparameters(X, y, clf_name='svm', num_iter=20, verbose=True, create_roc=True):
    """This method optimizes hyperparameters with cross-validation using RandomSearchCV, optionally creates a ROC curve
        and returns this optimized classifier and a report

    :param X: Feature matrix
    :param y: labels
    :param clf_name:  Name of the classifier as given in classifiers.py
    :param num_iter: Number of iterations the RandomSearchCV should perform
    :param verbose: Whether a detailed score should be printed out (optional)

    :return: optimized classifier, roc_auc, recall, specificity, precision, conf_mat and report of those as a string

    """

    c_classifier = classifiers.get_clf_with_name(clf_name, X, y)

    print('Doing RandomSearchCV for ' + clf_name + '...')

    # Naive Bayes doesn't have any hyperparameters to tune
    if clf_name == 'Naive Bayes':
        clf = c_classifier.clf
        X, y = f_factory.get_feature_matrix_and_label(True, True, True, True, True)
        clf.fit(X, y)

        # print('Best parameters: ' + str(model_factory.get_tuned_params_dict(clf, c_classifier.tuned_params)) + '\n')
    else:
        clf = RandomizedSearchCV(c_classifier.clf, c_classifier.tuned_params, cv=10,
                                 scoring='roc_auc', n_iter=num_iter)  # TODO: Change num_iter
        clf.fit(X, y)

        # print('Best parameters: ' + str(model_factory.get_tuned_params_dict(clf.best_estimator_,
        #                                                                     c_classifier.tuned_params)) + '\n')
        if verbose:
            report(clf.cv_results_)

    # Naive Bayes doesn't have any hyperparameters to tune
    if clf_name == 'Naive Bayes':
        roc_auc, recall, specificity, precision, conf_mat, rep = model_factory.get_performance(clf, clf_name, X, y)
    else:
        roc_auc, recall, specificity, precision, conf_mat, rep = \
            model_factory.get_performance(clf.best_estimator_, clf_name, X, y,
                                          list(c_classifier.tuned_params.keys()))

    if create_roc:
        filename = 'roc_with_hp_tuning' + clf_name + '.pdf'
        model_factory.plot_roc_curve(clf, X, y, filename, 'ROC for ' + clf_name +
                                     'with hyperparameter tuning')

    return clf, roc_auc, recall, specificity, precision, conf_mat, rep


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