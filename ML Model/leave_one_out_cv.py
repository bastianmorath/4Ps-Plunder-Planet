
from __future__ import division  # s.t. division uses float result
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import (LeaveOneGroupOut, cross_val_predict)


import plots
import globals as gl
import features_factory as f_factory
import classifiers
import model_factory


def clf_performance_with_user_left_out_vs_normal(X, y, plot_auc_score_per_user=True):
    """Plots a barchart with the mean roc_auc score for each classfier in two scenarios:
        1. Do normal crossvalidation to get roc_auc (There can thus be part of a users
            logfile in the training set AND in the testset. This could influence the performance on the testset as
            the model has already seen part of the users data/behavior in the training set)
        2. For the training_data, use all but one user, and then predict score on the last user that was NOT
            used in the training phase!


    :param X: Feature matrix
    :param y: labels
    :param plot_auc_score_per_user: Whether or not we should create a plot for each user left out with the auc_score of
                                    each classifier when using LeaveOneOut cross validation

    """
    names = ['SVM', 'Linear SVM', 'Nearest Neighbor', 'QDA', 'Naive Bayes']

    clfs = [classifiers.CSVM(X, y),
            classifiers.CLinearSVM(X, y),
            classifiers.CNearestNeighbors(X, y),
            classifiers.CQuadraticDiscriminantAnalysis(X, y),
            classifiers.CGradientBoostingClassifier(X, y),
            classifiers.CNaiveBayes(X, y)
            ]

    # Get scores for scenario 1 (normal crossvalidation)
    auc_scores_scenario_1 = []
    auc_stds_scenario_1 = []

    for name, classifier in zip(names, clfs):
        # If NaiveBayes classifier is used, then use Boxcox since features must be gaussian distributed
        if name == 'Naive Bayes':
            old_bx = gl.use_boxcox
            gl.use_boxcox = True
            X, _ = f_factory.get_feature_matrix_and_label()
            gl.use_boxcox = old_bx
        classifier.clf.fit(X, y)
        auc_scores_scenario_1.append(model_factory.get_performance(classifier.clf, name, X, y))
        auc_stds_scenario_1.append(apply_cv_per_user_model(classifier.clf, name, X, y))

    # Get scores for scenario 2 (Leave one user out in training phase)
    auc_scores_scenario_2 = []
    auc_stds_scenario_2 = []
    for name, classifier in zip(names, clfs):
        # If NaiveBayes classifier is used, then use Boxcox since features must be gaussian distributed
        if name == 'Naive Bayes':
            old_bx = gl.use_boxcox
            gl.use_boxcox = True
            X, _ = f_factory.get_feature_matrix_and_label()
            gl.use_boxcox = old_bx
        classifier.clf.fit(X, y)
        auc_mean, auc_std = apply_cv_per_user_model(classifier.clf, name, X, y, plot_auc_score_per_user)
        auc_scores_scenario_2.append(auc_mean)
        auc_stds_scenario_2.append(auc_std)

    _plot_scores_normal_cv_vs_leaveoneout_cv(names, auc_scores_scenario_1, auc_stds_scenario_1,
                                             auc_scores_scenario_2, auc_stds_scenario_2)


def apply_cv_per_user_model(model, clf_name, X, y, plot_auc_score_per_user=False, verbose=False):

    """Takes one entire user (i.e. two logfiles most of the time) out of training phase and does prediction
    on left out user. Result can be used as an indication which users are hard to predict

    :param model: the model that should be applied
    :param clf_name: String name of the classifier
    :param X: the feature matrix
    :param y: True labels

    :return: auc_mean and auc_std

    """

    y = np.asarray(y)  # Used for .split() function

    # Each user should be a separate group, s.t. we can always leaveout one user
    groups_ids = pd.concat(gl.obstacle_df_list)['userID'].map(str).tolist()

    logo = LeaveOneGroupOut()
    scores_and_ids = []  # tuples of (auc, recall, specificity, precision, user_id)
    df_obstacles_concatenated = pd.concat(gl.obstacle_df_list, ignore_index=True)
    for train_index, test_index in logo.split(X, y, groups_ids):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        conf_mat = confusion_matrix(y_test, y_pred)

        recall = metrics.recall_score(y_test, y_pred)
        specificity = conf_mat[0, 0] / (conf_mat[0, 0] + conf_mat[0, 1])
        precision = metrics.precision_score(y_test, y_pred)
        auc = metrics.roc_auc_score(y_test, y_pred)

        # I calculate the indices that were left out, and map them back to one row of the data,
        # then taking its userid and logID
        left_out_group_indices = sorted(set(range(0, len(df_obstacles_concatenated))) - set(train_index))
        group = df_obstacles_concatenated.loc[[left_out_group_indices[0]]]['userID']

        user_id = group['userID'].item()
        scores_and_ids.append((auc, recall, specificity, precision, user_id))

    # Get a list with the user names (in the order that LeaveOneGroupOut left the users out in training phase)
    names = []
    for _, (auc, rec, spec, prec, user_id, log_id) in enumerate(scores_and_ids):
        for df_idx, df in enumerate(gl.df_list):
            # Get the username out of userID
            if df.iloc[0]['userID'] == user_id:
                name = gl.names_logfiles[df_idx][:2]
                names.append(name)

    if verbose:
        # Print scores for each individual user left out in cross_validation
        print('roc_auc score for each user that was left out in training set and predicted on in test_set')
        for i, (auc, rec, spec, prec, user_id, log_id) in enumerate(scores_and_ids):
            name = names[i]
            print(name + ':\t\t Auc= %.3f, Recall = %.3f, Specificity= %.3f, '
                         'Precision = %.3f' % (auc, rec, spec, prec))

    names = list(dict.fromkeys(names))  # Filter duplicates while preserving order
    aucs = [a[0] for a in scores_and_ids]

    auc_mean = np.mean(aucs)
    auc_std = np.std(aucs)
    if plot_auc_score_per_user:
        title = r'Auc scores per user with %s  ($\mu$=%.3f, $\sigma$=%.3f))' % (clf_name, auc_mean, auc_std)
        filename = 'LeaveOneOut/performance_per_user_' + clf_name + '.pdf'
        plots.plot_barchart(title, 'Users', 'Auc score', names, aucs, 'auc_score', filename)

    if verbose:
        # Print mean score
        auc_mean = np.mean([a[0] for a in scores_and_ids])
        recall_mean = np.mean([a[1] for a in scores_and_ids])
        specificity_mean = np.mean([a[2] for a in scores_and_ids])
        precision_mean = np.mean([a[3] for a in scores_and_ids])

        auc_std = np.std([a[0] for a in scores_and_ids])
        recall_std = np.std([a[1] for a in scores_and_ids])
        specificity_std = np.std([a[2] for a in scores_and_ids])
        precision_std = np.std([a[3] for a in scores_and_ids])

        auc_max = np.max([a[0] for a in scores_and_ids])
        recall_max = np.max([a[1] for a in scores_and_ids])
        specificity_max = np.max([a[2] for a in scores_and_ids])
        precision_max = np.max([a[3] for a in scores_and_ids])
        print('Performance score when doing LeaveOneGroupOut with logfiles: ')

        print('\n Performance: '
              'Auc = %.3f (+-%.3f, max: %.3f), '
              'Recall = %.3f (+-%.3f, max: %.3f), '
              'Specificity= %.3f (+-%.3f, max: %.3f), '
              'Precision = %.3f (+-%.3f, max: %.3f)'
               % (auc_mean, auc_std, auc_max,
                recall_mean, recall_std, recall_max,
                specificity_mean, specificity_std, specificity_max,
                precision_mean, precision_std, precision_max))

        y_pred = cross_val_predict(model, X, y, cv=logo.split(X, y, groups_ids))
        conf_mat = confusion_matrix(y, y_pred)
        print('\t Confusion matrix: \n\t\t' + str(conf_mat).replace('\n', '\n\t\t'))

        predicted_accuracy = round(metrics.accuracy_score(y, y_pred) * 100, 2)
        null_accuracy = round(max(np.mean(y), 1 - np.mean(y)) * 100, 2)

        print('\tCorrectly classified data: ' + str(predicted_accuracy) + '% (vs. null accuracy: ' + str
        (null_accuracy) + '%)')

    return auc_mean, auc_std


def _plot_scores_normal_cv_vs_leaveoneout_cv(names, auc_scores_scenario_1, auc_stds_scenario_1,
                                             auc_scores_scenario_2, auc_stds_scenario_2):
    fix, ax = plt.subplots()
    bar_width = 0.3
    opacity = 0.4
    index = np.arange(len(auc_scores_scenario_1))

    r1 = plt.bar(index, auc_scores_scenario_1, bar_width,
                 alpha=opacity,
                 color=plots.green_color,
                 label='roc_auc normal CV',
                 yerr=auc_stds_scenario_1)

    r2 = plt.bar(index, auc_scores_scenario_2, bar_width,
                 alpha=opacity,
                 color=plots.red_color,
                 label='roc_auc LeaveOneGroupOut CV',
                 yerr=auc_stds_scenario_2)

    plt.ylabel('roc_auc')
    plt.title('clf_performance_with_user_left_out_vs_normal')
    plt.xticks(index, names, rotation='vertical')
    plt.legend()

    def autolabel(rects):
        """
           Attach a text label above each bar displaying its height
           """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                    '%d' % int(height),
                    ha='center', va='bottom', size=5)

    autolabel(r1)
    autolabel(r2)

    plt.tight_layout()

    plots.save_plot(plt, 'Performance', 'clf_performance_with_user_left_out_vs_normal.pdf')
