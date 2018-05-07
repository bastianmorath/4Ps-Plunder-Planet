"""
This module is responsible for doing GridSearchCV  over the hyperparameters of the different classifiers
"""

import numpy as np
import pandas as pd

from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV, cross_val_predict)

import matplotlib.pyplot as plt

import seaborn as sns

import ml_model
import plots
import globals as gl
import classifiers


def get_optimal_clf(classifier, X, y, tuned_params, num_iter, verbose=False):
    """This method optimizes hyperparameters of svm with cross-validation, which is done using GridSearchCV on a
        training set
        The performance of the selected hyper-parameters and trained model is then measured on a dedicated test
        set that was not used during the model selection step.

    :return: classifier with optimal tuned hyperparameters

    """
    if verbose:
        print('# Tuning hyper-parameters for roc_auc \n')

    clf = RandomizedSearchCV(classifier, tuned_params, cv=5,
                             scoring='roc_auc', n_iter=num_iter, n_jobs=4)

    clf.fit(X, y)

    if verbose:
        print()
        print("roc-auc grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report: \n")

        y_pred = cross_val_predict(clf, X, y, cv=5)
        conf_mat = confusion_matrix(y, y_pred)

        print('\t Confusion matrix: \n\t\t' + str(conf_mat).replace('\n', '\n\t\t'))
        print(classification_report(y, y_pred, target_names=['No Crash: ', 'Crash: ']))
        print()

    return clf


def do_grid_search_for_classifiers(X, y, idx, num_iter):
    """Test different classifiers wihtout hyperparameter optimization, and prints its auc scores in a barplot

      Arguments:
          X {matrix} -- Feature matrix
          y {list/ndarray} -- labels
      """


    clfs = [
        classifiers.CSVM(X, y),
        classifiers.CNearestNeighbors(X, y),
        classifiers.CQuadraticDiscriminantAnalysis(X, y),
        classifiers.CGradientBoostingClassifier(X, y),
        classifiers.CDecisionTreeClassifier(X, y),
        classifiers.CRandomForest(X, y),
        classifiers.CAdaBoost(X, y)
    ]

    names = [clf.name for clf in clfs]

    scores = []
    optimal_params = []
    if idx > 0:
        clfs = [clfs[idx]]
        names = [names[idx]]

    for idx, (Classifier, name) in enumerate(zip(clfs, names)):
        optimal_clf = Classifier.optimal_clf(X, y, num_iter)
        # plot_heat_map_of_grid_search(optimal_clf.cv_results_, Classifier)
        optimal_params.append(optimal_clf.best_params_)

        roc_auc, recall, specificity, precision = ml_model.get_performance(optimal_clf, name, X, y, verbose=False)
        scores.append([roc_auc, recall, specificity, precision])


    plt = plots.plot_barchart(title='Scores by classifier with hyperparameter tuning',
                              x_axis_name='Classifier',
                              y_axis_name='Performance',
                              x_labels=names,
                              values=[a[0] for a in scores],
                              lbl='auc_score'
                              )

    plt.savefig(gl.working_directory_path + '/Performance_scores/performance_per_clf_after_grid_search.pdf')

    s = ''
    for idx, sc in enumerate(scores):
        s += 'Scores for %s: \n' \
            '\troc_auc: %.3f, ' \
            'recall: %.3f, ' \
            'specificity: %.3f, ' \
            'precision: %.3f \n' \
            '\tOptimal params: %s \n\n\n' % (names[idx], sc[0], sc[1], sc[2], sc[3], optimal_params[idx])

    file = open(gl.working_directory_path + '/performance_' + str(idx) + '_' + str(num_iter) + '.txt', 'w+')
    file.write(s)


def plot_heat_map_of_grid_search(cv_results, Classifier):
    params = ([list(set(v.compressed())) for k, v in cv_results.items() if k.startswith('param_')])
    print(params)
    plt.figure()
    results_df = pd.DataFrame(cv_results)
    print(results_df)
    scores = np.array(results_df.mean_test_score).reshape(len(params[0]), len(params[1]))
    sns.heatmap(scores, annot=True,
                xticklabels=params[0], yticklabels=params[1], cmap=plt.cm.RdYlGn)
    plt.title('Grid Search roc_auc Score')
    plt.savefig(gl.working_directory_path + '/Plots/GridSearch_heatmaps/' + Classifier.name + '.pdf')



