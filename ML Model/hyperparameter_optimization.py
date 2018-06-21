"""This module takes a classifier name and n_iter and does RandomSearchCV to find the best hyperparameters of the

classifier with this name.
"""

from __future__ import division, print_function  # s.t. division uses float result

import numpy as np
import pandas as pd

from sklearn.metrics import (classification_report, confusion_matrix)
from sklearn.model_selection import ( RandomizedSearchCV, cross_val_predict)

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

    names = ['SVM', 'Linear SVM', 'Nearest Neighbor', 'QDA', 'Gradient Boosting', 'Decision Tree',
             'Random Forest', 'Ada Boost', 'Naive Bayes']

    scores = []
    optimal_params = []
    conf_mats = []

    for name in names:
        optimal_clf = get_clf_with_optimized_hyperparameters(X, y, name, num_iter)
        # plot_heat_map_of_grid_search(optimal_clf.cv_results_, Classifier)
        optimal_params.append(optimal_clf.best_params_)

        roc_auc, recall, specificity, precision, conf_mat = \
            model_factory.get_performance(optimal_clf, name, X, y, verbose=False)

        scores.append([roc_auc, recall, specificity, precision])
        conf_mats.append(conf_mat)

    return names, scores, optimal_params, conf_mats


def report(results, n_top=3):
    """Displays the n_top hyperparameter configurations with the best score and reports
    performance scores

    :param results: cv_results of GridSearch/RandomSearchCV
    :param n_top: First n_top hyperparameter configurations should be displayed
    """

    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("\tMean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("\tParameters: {0}".format(results['params'][candidate]))
            print("")


def get_clf_with_optimized_hyperparameters(X, y, clf_name='svm', num_iter=20, verbose=False):
    """This method optimizes hyperparameters with cross-validation, which is done using RandomSearchCV and
    returns this optimized classifier

    :param X: Feature matrix
    :param y: labels
    :param clf_name:  Name of the classifier as given in classifiers.py
    :param num_iter: Number of iterations the RandomSearchCV should perform
    :param verbose: Whether a detailed report should be printed

    :return: roc_auc, recall, specificity, precision, conf_mat

    """
    c_classifier = classifiers.get_clf_with_name(clf_name, X, y)

    if verbose:
        print('Doing RandomSearchCV...\n')

    clf = RandomizedSearchCV(c_classifier.clf, c_classifier.tuned_params, cv=10,
                             scoring='roc_auc', n_iter=num_iter)

    clf.fit(X, y)

    if verbose:
        model_factory.get_performance(clf, clf_name, X, y)
        '''
        print()
        print("roc-auc grid scores on development set:")
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']

        report(clf.cv_results_)
        print("Detailed classification report: \n")

        y_pred = cross_val_predict(clf, X, y, cv=10)
        conf_mat = confusion_matrix(y, y_pred)

        print('\t Confusion matrix: \n\t\t' + str(conf_mat).replace('\n', '\n\t\t'))
        print(classification_report(y, y_pred, target_names=['No Crash: ', 'Crash: ']))
        print()
        '''

    return clf


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