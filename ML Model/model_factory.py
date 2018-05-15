"""This module mainly serves as a factory with various methods used by the machine learning modules

"""
from __future__ import division  # s.t. division uses float result

# matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (auc, confusion_matrix, roc_curve)
from sklearn.model_selection import (cross_val_predict, train_test_split)

from sklearn.calibration import CalibratedClassifierCV

import features_factory as f_factory
import setup_dataframes as sd
import plots
import classifiers
import setup_dataframes


def get_performance(model, clf_name, X, y, hw=30, cw=30, gradient_w=10, verbose=True, write_to_file=False):
    """Computes performance of the model by doing cross validation with 10 folds, using
        cross_val_predict, and returns roc_auc, recall, specificity, precision, confusion matrix and summary of those
        as a string

    :param model: the classifier that should be applied
    :param clf_name: Name of the classifier (used to print scores)
    :param hw: size of heartrate_window (default=30)
    :param cw: size of crash_window (default=30)
    :param gradient_w: size of gradient_window (default=10)
    :param use_cache: Use the cached featurematrix if existent
    :param verbose: Whether a detailed score should be printed out
    :param write_to_file: Write summary of performance in a file

    :return: roc_auc, recall, specificity, precicion, confusion_matrix and summary of those as a string
    """
    if verbose:
        print('Calculating performance of %s...' % clf_name)

    f_factory.hw = hw
    f_factory.cw = cw
    f_factory.gradient_w = gradient_w

    sd.obstacle_df_list = setup_dataframes.get_obstacle_times_with_success()

    # Compute performance scores
    y_pred = cross_val_predict(model, X, y, cv=5)
    conf_mat = confusion_matrix(y, y_pred)

    precision = metrics.precision_score(y, y_pred)
    recall = metrics.recall_score(y, y_pred)
    specificity = conf_mat[0, 0] / (conf_mat[0, 0] + conf_mat[0, 1])
    roc_auc = metrics.roc_auc_score(y, y_pred)

    s = 'Scores for %s (Windows:  %i, %i, %i): \n\n' \
        '\troc_auc: %.3f, ' \
        'recall: %.3f, ' \
        'specificity: %.3f, ' \
        'precision: %.3f \n\n' \
        '\tConfusion matrix: \t %s \n\t\t\t\t\t %s\n\n\n' \
        % (clf_name, hw, cw, gradient_w, roc_auc, recall, specificity, precision, conf_mat[0], conf_mat[1])

    if verbose:
        print(s)

    if write_to_file:
        # Write result to a file
        filename = 'performance_' + clf_name + '_windows_' + str(hw) + '_' + str(cw) + '_' + str(gradient_w) + '.txt'
        write_to_file(s, 'Performance/', filename, 'w+')

    return roc_auc, recall, specificity, precision, conf_mat, s


def feature_selection(X, y, verbose=False):
    """Feature Selection with ExtraTreesClassifier. Prints and plots the importance of the features


    Source: http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html

    :param X:  Feature matrix
    :param y: labels
    :param verbose: Whether a detailed report should be printed out

    :return new feature matrix with selected features

    """

    clf = ExtraTreesClassifier(n_estimators=250, class_weight='balanced')

    forest = clf.fit(X, y)

    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    X_new = SelectFromModel(clf).fit_transform(X, y)

    # Print the feature ranking
    if verbose:
        print("Feature ranking:")
        print('\n# features after feature-selection: ' + str(X_new.shape[1]))
    x_ticks = []
    for f in range(X.shape[1]):
        x_ticks.append(f_factory.feature_names[indices[f]])
        if verbose:
            print("%d. feature %s (%.3f)" % (f + 1, f_factory.feature_names[indices[f]], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), x_ticks, rotation='vertical')
    plt.xlim([-1, X.shape[1]])
    plt.tight_layout()

    plots.save_plot(plt, 'Features/', 'feature_importance.pdf')

    return X_new, y


def plot_roc_curve(classifier, X, y, classifier_name):
    """Plots roc_curve for a given classifier

    :param classifier:  Classifier
    :param X: Feature matrix
    :param y: labels
    :param classifier_name: Name of the classifier

    """

    # allows to add probability output to classifiers which implement decision_function()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    classifier.fit(X_train, y_train)
    clf = CalibratedClassifierCV(classifier)
    predicted_probas = clf.predict_proba(X_test)  # returns class probabilities for each class

    fpr, tpr, _ = roc_curve(y_test, predicted_probas[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    filename = 'roc_curve_'+classifier_name+'.pdf'
    plots.save_plot(plt, 'Performance/Roc Curves/', filename)


def print_confidentiality_scores(X_train, X_test, y_train, y_test):
    """Prints all wrongly classifed datapoints and with which confidentiality the classifier classified them

    :param X_train: Training data (Feature matrix)
    :param X_test:  Test data (Feature matrix)
    :param y_train: labels of training data
    :param y_test:  labels of test data

    """

    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    probas = model.predict_proba(X_test)
    y_predicted = model.predict(X_test)
    for idx, [a, b] in enumerate(probas):
        if y_test[idx] != y_predicted[idx]:
            print('True/Predicted: (' + str(y_test[idx]) + ', ' + str(y_predicted[idx]) + '), Confidentiality: '
                  + str(max(a,b)*100) + '%')


def plot_performance_of_classifiers_without_hyperparameter_tuning(X, y):
    # Plots performance of the given classifiers in a barchart for comparison without hyperparameter tuning

    clf_list = [
        classifiers.CLinearSVM(X, y),
        classifiers.CSVM(X, y),
        classifiers.CNearestNeighbors(X, y),
        classifiers.CNaiveBayes(X, y),
        classifiers.CQuadraticDiscriminantAnalysis(X, y),
    ]

    auc_scores = []
    for classifier in clf_list:
        print('Name: ' + classifier.name)
        # If NaiveBayes classifier is used, then use Boxcox since features must be gaussian distributed
        if classifier.name == 'Naive Bayes':
            X_nb, y_nb = f_factory.get_feature_matrix_and_label(verbose=False, cached_feature_matrix='all',
                                                                use_boxcox=True)

            classifier.clf.fit(X_nb, y_nb)
            auc_scores.append(get_performance(classifier.clf, classifier.name, X_nb, y_nb)[0])

            plot_roc_curve(classifier.clf, X_nb, y_nb, classifier.name)
        else:
            classifier.clf.fit(X, y)
            auc_scores.append(get_performance(classifier.clf, classifier.name, X, y)[0])

            plot_roc_curve(classifier.clf, X, y, classifier.name)

    # Plots roc_auc for the different classifiers
    plots.plot_barchart(title='performance_per_clf_without_grid_search',
                        xlabel='',
                        ylabel='roc_auc',
                        x_tick_labels=[clf.name for clf in clf_list],
                        values=auc_scores,
                        lbl='roc_auc',
                        filename='performance_per_clf_without_grid_search.pdf'
                        )


def plot_barchart_scores(names, scores):
    plots.plot_barchart(title='Scores by classifier with hyperparameter tuning',
                        xlabel='Classifier',
                        ylabel='Performance',
                        x_tick_labels=names,
                        values=[a[0] for a in scores],
                        lbl='auc_score',
                        filename='performance_per_clf_after_grid_search.pdf'
                        )


def write_scores_to_file(names, scores, optimal_params, conf_mat):
    s = ''
    for i, sc in enumerate(scores):
        s += 'Scores for %s: \n\n' \
             '\troc_auc: %.3f, ' \
             'recall: %.3f, ' \
             'specificity: %.3f, ' \
             'precision: %.3f \n\n' \
             '\tOptimal params: %s \n\n' \
             '\tConfusion matrix: \t %s \n\t\t\t %s\n\n\n' \
             % (names[i], sc[0], sc[1], sc[2], sc[3], optimal_params[i], conf_mat[2 * i], conf_mat[2 * i + 1])

    write_to_file(s, 'Performance/', 'classifier_performances_randomsearch_cv.txt', 'w+')


def write_to_file(string, folder, filename, mode, verbose=True):
    """Writes a string to a file while checking that the path already exists and creating it if not

        :param string:  Strin to be written to the file
        :param folder: Folder to be saved to
        :param filename: The name (.pdf) under which the plot should be saved\
        :param mode: w+, a+, etc..

    """
    path = sd.working_directory_path + '/Evaluation/' + folder + '/'

    if not os.path.exists(path):
        os.makedirs(path)

    savepath = path + '/' + filename
    if verbose:
        print('Scores written to file...')
    file = open(savepath, mode)
    file.write(string)