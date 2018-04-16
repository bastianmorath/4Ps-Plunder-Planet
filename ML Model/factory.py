# factory_model.py

from __future__ import division  # s.t. division uses float result
# from mpl_toolkits.mplot3d import Axes3D

import globals as gl
import pandas as pd
import numpy as np


'''Resamples a dataframe with a sampling frquency of 'resolution'
    -> Smoothes the plots
'''


def resample_dataframe(df, resolution):
    df = df.set_index('timedelta', drop=True)  # set timedelta as new index
    resampled = df.resample(str(resolution)+'S').mean()
    resampled.reset_index(inplace=True)
    # timedelta was resampled, so we need to do the same with the Time-column
    resampled['Time'] = resampled['timedelta'].apply(lambda time: time.total_seconds())
    return resampled


''' Returns a list of dataframes, each dataframe has time of each obstacle and whether crash or not (1 df per logfile)
'''


def get_obstacle_times_with_success():

    obstacle_time_crash = []

    for dataframe in gl.df_list:
        obstacle_times_current_df = []
        for idx, row in dataframe.iterrows():
            if row['Time'] > max(gl.cw, gl.hw):
                if row['Logtype'] == 'EVENT_OBSTACLE':
                    obstacle_times_current_df.append((row['Time'], 0))
                if row['Logtype'] == 'EVENT_CRASH':
                    obstacle_times_current_df.append((row['Time'], 1))
        times = np.asarray([a for (a, b) in obstacle_times_current_df])
        crashes = np.asarray([b for (a, b) in obstacle_times_current_df])

        obstacle_time_crash.append(pd.DataFrame({'Time': times, 'crash': crashes}))

    return obstacle_time_crash


'''Prints all wrongly classifed datapoints and with which confidentiality the classifier classified them'''


def print_confidentiality_scores(X_train, X_test, y_train, y_test):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    probas = model.predict_proba(X_test)
    y_predicted = model.predict(X_test)
    for idx, [a, b] in enumerate(probas):
        if y_test[idx] != y_predicted[idx]:
            print('True/Predicted: (' + str(y_test[idx]) + ', ' + str(y_predicted[idx]) + '), Confidentiality: '
                  + str(max(a,b)*100) + '%')


from sklearn import metrics

from sklearn.cross_validation import train_test_split  # IMPORTANT: use sklearn.cross_val for of Euler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import pickle
import features_factory as f_factory

def test_windows():
    # CHECK performance depending on window sizes
    hws = [5, 10, 20, 30, 40, 50]
    cws = [1, 2, 5, 10, 20, 30, 40, 50]
    results = []  # hw, cw, null_accuracy, predicted_accuracy
    for hw in hws:
        for cw in cws:
            gl.hw = hw
            gl.cw = cw
            X, y = f_factory.get_feature_matrix_and_label()
            scaler = MinMaxScaler(feature_range=(0, 1))
            X = scaler.fit_transform(X)  # Rescale between 0 and 1
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3)

            model = gl.model(X_train, y_train)

            # Predict values on test data
            y_test_predicted = model.predict(X_test)

            null_accuracy = max(np.mean(y_test), 1 - np.mean(y_test)) * 100
            predicted_accuracy = metrics.accuracy_score(y_test, y_test_predicted) * 100
            results.append([hw, cw, null_accuracy, predicted_accuracy])

    results.sort(key=lambda x: x[3])
    print(results)
    pickle.dump(results, open(gl.working_directory_path + '/Pickle/window_results.pickle', "wb"))

