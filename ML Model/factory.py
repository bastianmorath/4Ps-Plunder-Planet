# factory_model.py

from __future__ import division  # s.t. division uses float result
# from mpl_toolkits.mplot3d import Axes3D

import pandas as pd


from sklearn import metrics, svm
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle

from sklearn.utils import class_weight

import features_factory as f_factory

import globals as gl


def resample_dataframe(df, resolution):
    """Resamples a dataframe with a sampling frquency of 'resolution'
    -> Smoothes the plots

    :param df: Dataframe to be resampled. Must contain numbers only
    :param resolution:  Resolution of the newly sampled plot
    :return: Resampled dataframe

    """
    df = df.set_index('timedelta', drop=True)  # set timedelta as new index
    resampled = df.resample(str(resolution)+'S').mean()
    resampled.reset_index(inplace=True)
    # timedelta was resampled, so we need to do the same with the Time-column
    resampled['Time'] = resampled['timedelta'].apply(lambda time: time.total_seconds())
    return resampled


def get_obstacle_times_with_success():
    """Returns a list of dataframes, each dataframe has time of each obstacle and whether crash or not (1 df per logfile)

    userID/logID is used such that we cal later (after creating featurematrix), still differentiate logfiles

    :return: List which contains a dataframe df per log. Each df contains the time, userID, logID and whether the user
                crashed of each obstacle

    """
    print('Compute crashes...')

    obstacle_time_crash = []

    for dataframe in gl.df_list:
        obstacle_times_current_df = []
        for idx, row in dataframe.iterrows():
            if row['Time'] > max(gl.cw, gl.hw):
                if row['Logtype'] == 'EVENT_OBSTACLE':
                    obstacle_times_current_df.append((row['Time'], 0, row['userID'], row['logID']))
                if row['Logtype'] == 'EVENT_CRASH':
                    obstacle_times_current_df.append((row['Time'], 1, row['userID'], row['logID']))
        times = np.asarray([a for (a, b, c, d) in obstacle_times_current_df])
        crashes = np.asarray([b for (a, b, c, d) in obstacle_times_current_df])
        userIDs = np.asarray([c for (a, b, c, d) in obstacle_times_current_df])
        logIDs = np.asarray([d for (a, b, c, d) in obstacle_times_current_df])

        obstacle_time_crash.append(pd.DataFrame({'Time': times, 'crash': crashes,
                                                 'userID': userIDs,
                                                 'logID': logIDs}))

    return obstacle_time_crash


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


def test_windows():
    # CHECK performance depending on window sizes
    hws = [2, 5, 10, 30, 60]
    cws = [5, 10, 30, 60]
    gradient_ws = [5, 10, 30]
    results = []  # hw, cw, null_accuracy, predicted_accuracy
    for hw in hws:
        for cw in cws:
            gl.hw = hw
            gl.cw = cw
            gl.use_boxcox = False
            X, y = f_factory.get_feature_matrix_and_label()
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3)

            class_w = class_weight.compute_class_weight('balanced', np.unique(y), y)
            class_weight_dict = dict(enumerate(class_w))

            clf = svm.SVC(class_weight=class_weight_dict)
            clf.fit(X_train, y_train)

            # Predict values on test data
            y_test_predicted = clf.predict(X_test)
            conf_mat = confusion_matrix(y_test, y_test_predicted)
            precision = metrics.precision_score(y_test, y_test_predicted)
            recall = metrics.recall_score(y_test, y_test_predicted)
            specificity = conf_mat[0, 0] / (conf_mat[0, 0] + conf_mat[0, 1])
            auc = metrics.roc_auc_score(y_test, y_test_predicted)
            results.append([hw, cw, auc, recall, precision, specificity])
            print([hw, cw, auc, recall, precision, specificity])

    results.sort(key=lambda x: x[3])
    pickle.dump(results, open(gl.working_directory_path + '/Pickle/window_results.pickle', "wb"))


def print_keynumbers_logfiles():
    """ Print all important keynumbers, such as number of logs, number of features (=obstacles) etc.

    """

    df_lengths = []
    for d in gl.df_list:
        df_lengths.append(d['Time'].max())
    print('average:' + str(np.mean(df_lengths)) + ', std: ' + str(np.std(df_lengths)) +
          ', max: ' + str(np.max(df_lengths)) + ', min: ' + str(np.min(df_lengths)))

    print('#files: ' + str(len(gl.df_list)))
    print('#files with heartrate: ' + str(len([a for a in gl.df_list if not (a['Heartrate'] == -1).all()])))
    print('#datapoints: ' + str(sum([len(a.index) for a in gl.df_list])))
    print('#obstacles: ' + str(sum([len(df.index) for df in gl.obstacle_df_list])))
    print('#crashes: ' + str(sum([len(df[df['crash'] == 1]) for df in gl.obstacle_df_list ])))


