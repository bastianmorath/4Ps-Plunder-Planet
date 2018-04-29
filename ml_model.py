
from __future__ import division  # s.t. division uses float result

from sklearn.model_selection import cross_val_predict, LeaveOneGroupOut, GroupKFold
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import re
from sklearn import metrics

import globals as gl


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
          '\n\t Precision = %.3f = Probability that, given a crash is predicted, a crash really happened; [n'
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
    for train_index, test_index in logo.split(X, y, groups_ids):
        # I calculate the indices that wereleft out, and map them back to one row of the data,
        # then taking its userid and logID
        left_out_group_indices = sorted(set(range(0, 6120)) - set(train_index))
        group = df_obstacles_concatenated.loc[[left_out_group_indices[10]]][['userID', 'logID']]
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

    for idx, (rec, spec, prec, user_id, log_id) in enumerate(scores_and_logid):
        # Get the logname out of userID and log_id (1 or 2)
        logname = ''
        for idx, df in enumerate(gl.df_list):
            if df.iloc[0]['userID'] == user_id and df.iloc[0]['logID'] == log_id:
                logname = gl.names_logfiles[idx]
        print(logname + ':\t\t Recall = %.3f, Specificity= %.3f,'
                                       'Precision = %.3f' % (rec, spec, prec))
