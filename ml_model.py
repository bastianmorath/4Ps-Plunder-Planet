
from __future__ import division  # s.t. division uses float result

import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_predict, LeaveOneGroupOut
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from sklearn import metrics

import globals as gl
import features_factory as f_factory


def apply_cv_model(model, X, y):
    """
        Build feature matrix, then do cross-validation over the entire data

      :param model: the model that should be applied
      :param X: the feature matrix
      :param y: True labels
      :return:
      """
    y_pred = cross_val_predict(model, X, y, cv=10)
    conf_mat = confusion_matrix(y, y_pred)

    precision = metrics.precision_score(y, y_pred)
    recall = metrics.recall_score(y, y_pred)
    specificity = conf_mat[0, 0 ] /(conf_mat[0, 0 ] +conf_mat[0, 1])
    print('\n\t Recall = %0.3f = Probability of, given a crash, a crash is correctly predicted; '
          '\n\t Specificity = %0.3f = Probability of, given no crash, no crash is correctly predicted;'
          '\n\t Precision = %.3f = Probability that, given a crash is predicted, a crash really happened; \n'
          % (recall, specificity, precision))

    print('\t Confusion matrix: \n\t\t' + str(conf_mat).replace('\n', '\n\t\t'))

    predicted_accuracy = round(metrics.accuracy_score(y, y_pred) * 100, 2)
    null_accuracy = round(max(np.mean(y), 1 - np.mean(y)) * 100, 2)

    print('\tCorrectly classified data: ' + str(predicted_accuracy) + '% (vs. null accuracy: ' + str
        (null_accuracy) + '%)')


def apply_cv_groups_model(model, X, y):
    """
        Take one entire logfile as test-data, the rest as training-data. Result can be used
        as an indication which files are hard to predict

    :param model: the model that should be applied
    :param X: the feature matrix
    :param y: True labels
    :return:
    """
    y = np.asarray(y)  # Used for .split() function

    # Each file should be a separate group, s.t. we can aloways leaveout one logfile (NOT one user)
    groups_ids = (pd.concat(gl.obstacle_df_list)['userID'].map(str) + pd.concat(gl.obstacle_df_list)['logID'].map(str)).tolist()
    logo = LeaveOneGroupOut()
    scores_and_logid = []
    df_obstacles_concatenated = pd.concat(gl.obstacle_df_list, ignore_index=True)
    for train_index, test_index in logo.split(X, y, groups_ids, ):
        # I calculate the indices that were left out, and map them back to one row of the data,
        # then taking its userid and logID
        left_out_group_indices = sorted(set(range(0, len(df_obstacles_concatenated))) - set(train_index))
        group = df_obstacles_concatenated.loc[[left_out_group_indices[0]]][['userID', 'logID']]
        user_id, log_id = group['userID'].item(), group['logID'].item()
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        conf_mat = confusion_matrix(y_test, y_pred)

        recall = metrics.recall_score(y_test, y_pred)
        specificity = conf_mat[0, 0] / (conf_mat[0, 0] + conf_mat[0, 1])
        precision = metrics.precision_score(y_test, y_pred)

        scores_and_logid.append((recall, specificity, precision, user_id, log_id))

    # Print scores for each individual logfile left out in cross_validation
    for idx, (rec, spec, prec, user_id, log_id) in enumerate(scores_and_logid):
        # Get the logname out of userID and log_id (1 or 2)
        logname = ''
        for df_idx, df in enumerate(gl.df_list):
            if df.iloc[0]['userID'] == user_id and df.iloc[0]['logID'] == log_id:
                logname = gl.names_logfiles[df_idx]
        print(logname + ':\t\t Recall = %.3f, Specificity= %.3f,'
                                       'Precision = %.3f' % (rec, spec, prec))


    # Print mean score
    recall_mean = np.mean([a for (a, _, _, _, _) in scores_and_logid])
    specificity_mean = np.mean([b for (_, b,_, _, _) in scores_and_logid])
    precision_mean = np.mean([c for (_, _, c, _, _) in scores_and_logid])

    recall_std = np.std([a for (a, _, _, _, _) in scores_and_logid])
    specificity_std = np.std([b for (_, b, _, _, _) in scores_and_logid])
    precision_std = np.std([c for (_, _, c, _, _) in scores_and_logid])

    recall_max = np.max([a for (a, _, _, _, _) in scores_and_logid])
    specificity_max = np.max([b for (_, b, _, _, _) in scores_and_logid])
    precision_max = np.max([c for (_, _, c, _, _) in scores_and_logid])

    print('\n Performance:' + ': Recall = %.3f (+-%.3f, max: %.3f), Specificity= %.3f (+-%.3f, max: %.3f),'
                          ' Precision = %.3f (+-%.3f, max: %.3f)'
          % (recall_mean, recall_std, recall_max,
             specificity_mean, specificity_std, specificity_max,
             precision_mean, precision_std, precision_max))

    y_pred = cross_val_predict(model, X, y, cv=logo.split(X, y, groups_ids))
    conf_mat = confusion_matrix(y, y_pred)
    print('\t Confusion matrix: \n\t\t' + str(conf_mat).replace('\n', '\n\t\t'))

    predicted_accuracy = round(metrics.accuracy_score(y, y_pred) * 100, 2)
    null_accuracy = round(max(np.mean(y), 1 - np.mean(y)) * 100, 2)

    print('\tCorrectly classified data: ' + str(predicted_accuracy) + '% (vs. null accuracy: ' + str
    (null_accuracy) + '%)')





''' Feature Selection. Prints and plots the importance of the features'''


def feature_selection(X, y):

    forest = ExtraTreesClassifier(n_estimators=250,
                                  random_state=0)

    forest.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")
    x_ticks = []
    for f in range(X.shape[1]):
        x_ticks.append(f_factory.feature_names[indices[f]])
        print("%d. feature %s (%f)" % (f + 1, f_factory.feature_names[indices[f]], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), x_ticks, rotation='vertical')
    plt.xlim([-1, X.shape[1]])
    plt.tight_layout()
    # plt.savefig(gl.working_directory_path + '/Plots/feature_importance.pdf')

    return forest
