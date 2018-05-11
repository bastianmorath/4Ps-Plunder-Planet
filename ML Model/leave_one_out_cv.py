
from __future__ import division  # s.t. division uses float result
import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
import pandas as pd

from sklearn import metrics

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import (LeaveOneGroupOut, cross_val_predict)


from sklearn.utils import class_weight

from sklearn import naive_bayes
from sklearn import svm
from sklearn import neighbors
from sklearn import discriminant_analysis


import time
import numpy as np

import setup
import plots

import globals as gl
import features_factory as f_factory


def apply_cv_per_user_model(model, clf_name, X, y, per_logfile=False, verbose=False):

    """Take one entire user (i.e. two logfiles most of the time) as test-data, the rest as training-data. Result can be used
        as an indication which users are hard to predict

    :param model: the model that should be applied
    :param clf_name: String name of the classifier
    :param X: the feature matrix
    :param y: True labels
    :param per_logfile: Whether we should leave out one logfile or one user in cross validation
    :return: auc_mean and auc_std

    """

    y = np.asarray(y)  # Used for .split() function

    # Each file should be a separate group, s.t. we can always leaveout one logfile (NOT one user)
    if per_logfile:
        groups_ids = (pd.concat(gl.obstacle_df_list)['userID'].map(str) + pd.concat(gl.obstacle_df_list)['logID'].map(str)).tolist()
    # Each user should be a separate group, s.t. we can always leaveout one user (NOT one logfile)
    else:
        groups_ids = pd.concat(gl.obstacle_df_list)['userID'].map(str).tolist()

    logo = LeaveOneGroupOut()
    scores_and_ids = []
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
        group = df_obstacles_concatenated.loc[[left_out_group_indices[0]]][['userID', 'logID']]

        if per_logfile:
            user_id, log_id = group['userID'].item(), group['logID'].item()
            scores_and_ids.append((auc, recall, specificity, precision, user_id, log_id))
        else:
            user_id = group['userID'].item()
            scores_and_ids.append((auc, recall, specificity, precision, user_id, -1))

    names = []

    # Print scores for each individual logfile/user left out in cross_validation
    for _, (auc, rec, spec, prec, user_id, log_id) in enumerate(scores_and_ids):
        for df_idx, df in enumerate(gl.df_list):
            # Get the logname out of userID and log_id (1 or 2)
            if per_logfile and df.iloc[0]['userID'] == user_id and df.iloc[0]['logID'] == log_id:
                name = gl.names_logfiles[df_idx][:2] + '_' + gl.names_logfiles[df_idx][-5]
                names.append(name)
                if verbose:
                    print(name + ':\t\t Auc= %.3f, Recall = %.3f, Specificity= %.3f,'
                                 'Precision = %.3f' % (auc, rec, spec, prec))
            # Get the username out of userID
            elif (not per_logfile) and df.iloc[0]['userID'] == user_id:
                name = gl.names_logfiles[df_idx][:2]
                names.append(name)
                if verbose:
                    print(name + ':\t\t Auc= %.3f, Recall = %.3f, Specificity= %.3f, '
                                 'Precision = %.3f' % (auc, rec, spec, prec))

    names = list(dict.fromkeys(names))  # Filter duplicates while preserving order

    index = np.arange(len(names))
    aucs = [a[0] for a in scores_and_ids]
    recalls = [a[1] for a in scores_and_ids]
    specificities = [a[2] for a in scores_and_ids]
    precisions = [a[3] for a in scores_and_ids]


    _, ax = plt.subplots()
    bar_width = 0.3
    opacity = 0.4
    r = plt.bar(index, aucs, bar_width,
                    alpha=opacity,
                    color=plots.blue_color,
                    label='Auc')

    mean = np.mean(aucs)
    std = np.std(aucs)

    if per_logfile:
        plt.xlabel('Logfile')
        plt.title(r'Auc scores by logfile with %s ($\mu$=%.3f, $\sigma$=%.3f))' % (clf_name, mean, std))
    else:
        plt.xlabel('User')
        plt.title(r'Auc scores by user with %s  ($\mu$=%.3f, $\sigma$=%.3f))' % (clf_name, mean, std))

    plt.ylabel('Auc score')
    plt.xticks(index, names, rotation='vertical')
    plt.legend()

    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for i, rect in enumerate(rects):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 1.02 * height,
                    '%0.3f' % aucs[i],
                    ha='center', va='bottom', size=5)

    autolabel(r)

    plt.tight_layout()
    if per_logfile:
        plots.save_plot(plt, 'Performance', 'performance_per_logfile_' + clf_name+'.pdf')
    else:
        plots.save_plot(plt, 'Performance', 'performance_per_user_' + clf_name+'.pdf')

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

    if verbose:
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


print('Params: \n\t testing: ' + str(gl.testing) + ', \n\t use_cache: ' + str(gl.use_cache) + ', \n\t test_data: ' +
      str(gl.test_data) + ', \n\t use_boxcox: ' + str(gl.use_boxcox) + ', \n\t plots_enabled: ' + str(gl.plots_enabled)
      + ', \n\t reduced_features: ' + str(gl.reduced_features))

print('Init dataframes...')

start = time.time()

setup.setup()

print('Creating feature matrix...\n')

X, y = f_factory.get_feature_matrix_and_label()

cw = class_weight.compute_class_weight('balanced', np.unique(y), y)
class_weight_dict = dict(enumerate(cw))

names = ["Linear SVM", "RBF SVM", "Nearest Neighbor", "Naive Bayes", "QDA"]

classifiers = [
        svm.LinearSVC(class_weight=class_weight_dict),
        svm.SVC(class_weight=class_weight_dict),
        neighbors.KNeighborsClassifier(),
        naive_bayes.GaussianNB(),
        discriminant_analysis.QuadraticDiscriminantAnalysis()
        ]

plot_classifier_scores = True
if plot_classifier_scores:
    auc_scores = []
    auc_stds = []
    for name, clf in zip(names, classifiers):
        print(name+'...')
        # If NaiveBayes classifier is used, then use Boxcox since features must be gaussian distributed
        if name == 'Naive Bayes':
            old_bx = gl.use_boxcox
            gl.use_boxcox = True
            X, _ = f_factory.get_feature_matrix_and_label()
            gl.use_boxcox = old_bx
        clf.fit(X, y)
        auc_scores.append(apply_cv_per_user_model(clf, name, X, y, per_logfile=False)[0])
        auc_stds.append(apply_cv_per_user_model(clf, name, X, y, per_logfile=False)[1])

        # ml_model.plot_roc_curve(clf, X, y, name)

    plt = plots.plot_barchart(title='roc_auc w/out hyperparameter tuning',
                              x_axis_name='',
                              y_axis_name='roc_auc',
                              x_labels=names,
                              values=auc_scores,
                              lbl=None,
                              std_err=auc_stds,
                              )

    plots.save_plot(plt, 'Performance/', 'roc_auc_per_classifier_leave_one_out.pdf')

end = time.time()
print('Time elapsed: ' + str(end - start))


