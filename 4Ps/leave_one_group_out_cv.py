"""
This module is responsible for testin the classifier performances when doing normal K-fold CrossValidation
vs. LeaveOneGroupOut-Crossvalidation, i.e. training on all but one logfile, and then test on the last one.

"""

from __future__ import division  # s.t. division uses float result

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict

import classifiers
import features_factory as f_factory
import model_factory
import plots_helpers
import setup_dataframes as sd


def clf_performance_with_user_left_out_vs_normal(X, y, plot_auc_score_per_user=True, reduced_features=False):
    """
    Plots a barchart with the mean roc_auc score for each classfier in two scenarios:
    1. Do normal crossvalidation to get roc_auc (There can thus be part of a users
        logfile in the training set AND in the testset. This could influence the performance on the testset as
        the model has already seen part of the users data/behavior in the training set)
    2. For the training_data, use all but one user, and then predict score on the last user that was NOT
        used in the training phase!


    :param X: Feature matrix
    :param y: labels
    :param plot_auc_score_per_user: Whether or not we should create a plot for each user left out with the auc_score of
                                    each classifier when using LeaveOneGroupOut cross validation
    :param reduced_features: Whether we should use all features or do feature selection first

    """

    clf_names = classifiers.names

    clf_list = [classifiers.get_cclassifier_with_name(name, X, y).clf for name in clf_names]

    # Get scores for scenario 1 (normal crossvalidation)
    print('\n***** Scenario 1 (normal crossvalidation) *****\n')
    auc_scores_scenario_1, auc_stds_scenario_1, s = model_factory. \
        calculate_performance_of_classifiers(X, y, tune_hyperparameters=False,
                                             reduced_clfs=False, create_curves=False, do_write_to_file=False)

    # Get scores for scenario 2 (Leave one user out in training phase)
    print('\n***** Scenario 2  (Leave one user out in training phase) ***** \n')
    auc_scores_scenario_2 = []
    auc_stds_scenario_2 = []
    for name, classifier in zip(clf_names, clf_list):
        print('Calculating performance of %s with doing LeaveOneGroupOut ...' % name)

        # If NaiveBayes classifier is used, then use Boxcox since features must be gaussian distributed
        if name == 'Naive Bayes':
            feature_selection = 'selected' if reduced_features else 'all'
            X_nb, y_nb = f_factory.get_feature_matrix_and_label(verbose=False,
                                                                use_cached_feature_matrix=feature_selection,
                                                                save_as_pickle_file=True, use_boxcox=True)
            classifier.fit(X_nb, y_nb)

            auc_mean, auc_std = _apply_cv_per_user_model(classifier, name,
                                                         X_nb, y_nb, plot_auc_score_per_user)
        else:
            classifier.fit(X, y)

            auc_mean, auc_std = _apply_cv_per_user_model(classifier, name,
                                                         X, y, plot_auc_score_per_user)

        auc_scores_scenario_2.append(auc_mean)
        auc_stds_scenario_2.append(auc_std)

    _plot_scores_normal_cv_vs_leaveone_group_out_cv(clf_names, auc_scores_scenario_1, auc_stds_scenario_1,
                                                    auc_scores_scenario_2, auc_stds_scenario_2)


def _apply_cv_per_user_model(model, clf_name, X, y, plot_auc_score_per_user=True):
    """
    Takes one entire user (i.e. two logfiles most of the time) out of training phase and does prediction
    on left out user. Result can be used as an indication which users are hard to predict

    :param model: the model that should be applied
    :param clf_name: String name of the classifier
    :param X: the feature matrix
    :param y: True labels
    :param plot_auc_score_per_user: Generates one plot per user with scores for all classifiers

    :return: MACRO auc_mean and auc_std

    """

    y = np.asarray(y)  # Used for .split() function

    # Each user should be a separate group, s.t. we can always leaveout one user
    groups_ids = pd.concat(sd.obstacle_df_list)['userID'].map(str).tolist()
    logo = LeaveOneGroupOut()
    scores_and_ids = []  # tuples of (auc, recall, specificity, precision, user_id)
    df_obstacles_concatenated = pd.concat(sd.obstacle_df_list, ignore_index=True)
    for train_index, test_index in logo.split(X, y, groups_ids):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)

        # I do MACRO averaging such that I can compute std!
        roc_auc, _, recall, _, specificity, precision, _, conf_mat, _ = \
            model_factory.get_performance(model, "", X_test, y_test, verbose=False, create_curves=False)

        # I calculate the indices that were left out, and map them back to one row of the data,
        # then taking its userid and logID
        left_out_group_indices = sorted(set(range(0, len(df_obstacles_concatenated))) - set(train_index))
        group = df_obstacles_concatenated.loc[[left_out_group_indices[0]]]
        user_id = group['userID'].item()
        # print(auc, recall, precision, specificity)
        scores_and_ids.append((roc_auc, recall, specificity, precision, user_id))

    # Get a list with the user names (in the order that LeaveOneGroupOut left the users out in training phase)
    names = []
    for _, (auc, rec, spec, prec, user_id) in enumerate(scores_and_ids):
        for df_idx, df in enumerate(sd.df_list):
            # Get the username out of userID
            if df.iloc[0]['userID'] == user_id:
                name = sd.names_logfiles[df_idx][:2]
                names.append(name)

    names = list(dict.fromkeys(names))  # Filter duplicates while preserving order
    aucs = [a[0] for a in scores_and_ids]

    auc_mean: float = np.mean(aucs)  # Calculating MACRO-average
    auc_std: float = np.std(aucs)

    if plot_auc_score_per_user:
        title = r'Auc scores per user with %s  ($\mu$=%.3f, $\sigma$=%.3f))' % (clf_name, auc_mean, auc_std)
        filename = 'LeaveOneGroupOut/performance_per_user_' + clf_name + '.pdf'
        plots_helpers.plot_barchart(title, 'Users', 'Auc score', names, aucs, 'auc_score', filename, verbose=False)

    y_pred = cross_val_predict(model, X, y, cv=logo.split(X, y, groups_ids))

    _write_detailed_report_to_file(scores_and_ids, y, y_pred, clf_name, names)

    return auc_mean, auc_std


def _plot_scores_normal_cv_vs_leaveone_group_out_cv(names, auc_scores_scenario_1, auc_stds_scenario_1,
                                                    auc_scores_scenario_2, auc_stds_scenario_2):
    """
    Plots the roc_auc score and the standard deviation for each classifier for both scenarios next to each other

    :param names: names of the logfiles
    :param auc_scores_scenario_1: list of roc_auc scores when doing normal cv
    :param auc_stds_scenario_1: list of roc_auc_std scores when doing normal cv
    :param auc_scores_scenario_2: list of roc_auc scores when doing leave_one_user_out cv
    :param auc_stds_scenario_2: list of roc_auc_std scores when doing leave_one_user_out cv

    """

    fix, ax = plt.subplots()
    bar_width = 0.3
    line_width = 0.3

    index = np.arange(len(auc_scores_scenario_1))
    ax.yaxis.grid(True, zorder=0, color='grey',  linewidth=0.3)
    ax.set_axisbelow(True)
    [i.set_linewidth(line_width) for i in ax.spines.values()]

    plt.bar(index, auc_scores_scenario_1, bar_width,
            color=plots_helpers.blue_color,
            label='roc_auc normal CV',
            yerr=auc_stds_scenario_1,
            error_kw={'elinewidth': line_width,
                      'capsize': 1.4,
                      'markeredgewidth': line_width},
            )

    plt.bar(index + bar_width, auc_scores_scenario_2, bar_width,
            color=plots_helpers.red_color,
            label='roc_auc LeaveOneGroupOut CV',
            yerr=auc_stds_scenario_2,
            error_kw={'elinewidth': line_width,
                      'capsize': 1.4,
                      'markeredgewidth': line_width},
            )

    plt.ylabel('roc_auc')
    plt.title('Performance when leaving one user out in training phase')
    plt.xticks(index + bar_width/2, names, rotation='vertical')
    ax.set_ylim([0, 1.2])
    plt.legend(prop={'size': 6})

    '''
    def autolabel(rects):
        """
           Attach a text label above each bar displaying its height
           """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 1.1 * height,
                    '%0.2f' % height,
                    ha='center', va='bottom', size=5)

    # autolabel(r1)
    # autolabel(r2)
    '''

    plt.tight_layout()

    plots_helpers.save_plot(plt, 'Performance/LeaveOneGroupOut/', 'clf_performance_with_user_left_out_vs_normal.pdf')


def _write_detailed_report_to_file(scores, y, y_pred, clf_name, names):
    """
    Appends a detailed report to a file, i.e. for each classifier it appends
    the performance for each fold (i.e. in each round of CV, there was one user left out for test-set. Write down
    auc, recall, specificity and precision with std for each such user.) Also write down overall conf_matrix
    and mean-auc

    :param scores: array for each cv fold: auc, recall, specificity, precision, user_id
    :param y: true labels
    :param y_pred: y_predicted with scross_val_predict
    :param clf_name: Name of classifier
    :param names: Names of the users

    """

    # Print mean score
    auc_mean = np.mean([a[0] for a in scores])
    recall_mean = np.mean([a[1] for a in scores])
    specificity_mean = np.mean([a[2] for a in scores])
    precision_mean = np.mean([a[3] for a in scores])

    auc_std = np.std([a[0] for a in scores])
    recall_std = np.std([a[1] for a in scores])
    precision_std = np.std([a[3] for a in scores])

    conf_mat = confusion_matrix(y, y_pred)

    s = model_factory.create_string_from_scores(clf_name, auc_mean, auc_std, recall_mean, recall_std,  specificity_mean,
                                                precision_mean, precision_std, conf_mat)

    # scores for each individual user left out in cross_validation
    s += '\n\nroc_auc score for each user that was left out in training set and predicted on in test_set:'
    for i, (auc, rec, spec, prec, user_id) in enumerate(scores):
        name = names[i]
        s += '\n' + name + ':\t\t Auc= %.3f, Recall = %.3f, Specificity= %.3f, Precision = %.3f' \
             % (auc, rec, spec, prec)

    # Write log to file
    if clf_name == classifiers.names[0]:  # First classifier -> New file
        model_factory.write_to_file(s, 'Performance/LeaveOneGroupOut',
                                    'clf_performance_with_user_left_out_vs_normal_detailed.txt', 'w+', verbose=False)
    else:  # Else: Append to already created file
        model_factory.write_to_file(s, 'Performance/LeaveOneGroupOut',
                                    'clf_performance_with_user_left_out_vs_normal_detailed.txt', 'a+', verbose=False)
