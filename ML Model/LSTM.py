"""
This module is responsible to for the LSTM network

"""
from collections import Counter

from keras import Sequential
from keras.layers import LSTM, Dense, K, Flatten, TimeDistributed
from keras.preprocessing import sequence
import tensorflow as tf
from numpy import array
import numpy as np
from sklearn import metrics

import setup_dataframes as sd
import model_factory


def get_trained_lstm_classifier(X, y):
    """
    Reshapes feature matrix X, applies LSTM and returns the performance of the neural network

    :param X: List of non-reshaped/original feature matrices (one per logfile)
    :param y: labels

    :return: neural network
    """
    # TODO: Feature matrix is sorted along time!!!!!????
    X_splitted, y_splitted = get_splitted_up_feature_matrix_and_labels(X, y)

    X_reshaped, y_reshaped = get_reshaped_feature_matrix(X_splitted, y_splitted)

    apply_lstm(X_reshaped, y_reshaped)


"""
1. Reshape feature matrix

We first need to get one feature matrix per logfile (Split up generated big feature matrix). We also need to resample it
 (datapoints must be uniformly sampled!) We can also encode time as a new feature maybe...

Samples: Number of logfiles
Time Steps: Number of data points/obstacles
Features: 9 or 16 (depending on feature_reduction or not)
"""


def get_reshaped_feature_matrix(X_splitted, y_splitted):
    """ Returns the reshaped feature matrix needed for applying LSTM. Also does zero-padding

    :param X_splitted: Non-reshaped/original feature matrix

    :return: Reshaped feature matrix (3D)
    """

    maxlen = max(len(fm) for fm in X_splitted)
    minlen = min(len(fm) for fm in X_splitted)

    print('Maxlen (=Max. #obstacles of logfiles) is ' + str(maxlen) + ', minlen is ' + str(minlen))
    # Since values of feature matrix are between -1 and +1, I pad not with 0, but with e.g. -99
    padded_X = sequence.pad_sequences(X_splitted, maxlen=maxlen, padding='post', dtype='float64', value=-99)
    padded_y = sequence.pad_sequences(y_splitted, maxlen=maxlen, padding='post', dtype='float64', value=-99)
    X_reshaped = array(padded_X).reshape(len(padded_X), maxlen, padded_X[0].shape[1])
    y_reshaped = array(padded_y).reshape(len(padded_X), maxlen, 1)

    return X_reshaped, y_reshaped


"""
2. Apply LSTM
"""


def apply_lstm(X_reshaped, y_reshaped):
    """

    :param X_reshaped: Reshaped feature matrix
    :param y: labels

    :return: trained classifier
    """

    class_weights = {'0': 1, '1': 5}

    model = Sequential()
    model.add(LSTM(340, return_sequences=True, input_shape=(X_reshaped.shape[1], X_reshaped.shape[2])))
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))

    metrics = [sensitivity, specificity, 'accuracy']
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print('Shape X: ' + str(X_reshaped.shape))
    print('Shape y: ' + str(array(y_reshaped).shape))

    model.fit(X_reshaped, array(y_reshaped), epochs=20, batch_size=200,
              verbose=1, shuffle=False)
    result = model.predict(X_reshaped, batch_size=200, verbose=0)

    c = Counter(result.ravel())
    pred = []
    for v, _ in c.items():
        if v < 0.5:
            pred.append(0)
        else:
            pred.append(1)

    c2 = Counter(pred)
    print(c2)


"""
3. Calculate performance
"""

def calculate_performance(y, y_pred):
    conf_mat = tf.confusion_matrix(y, y_pred)

    precision = metrics.precision_score(y, y_pred)
    recall = metrics.recall_score(y, y_pred)
    specificity = conf_mat[0, 0] / (conf_mat[0, 0] + conf_mat[0, 1])
    roc_auc = metrics.roc_auc_score(y, y_pred)
    print(model_factory.create_string_from_scores('LSTM', roc_auc, recall, specificity, precision, conf_mat)
)
"""
Helper methods
"""


def get_splitted_up_feature_matrix_and_labels(X, y):
    """
    Feature matrix X is the concatenation of all feature matrices per logfile. We need to split it up such that we
    have one feature matrix per logfile

    :param X: Non-reshaped, original feature matrix

    :return: List of feature matrices, non-reshaped
    """

    feature_matrices = []
    label_lists = []
    obstacles_so_far = 0
    for df in sd.obstacle_df_list:
        num_obstacles = len(df.index)
        feature_matrices.append(X.take(range(obstacles_so_far, obstacles_so_far + num_obstacles), axis=0))
        label_lists.append(y[obstacles_so_far:obstacles_so_far + num_obstacles])
        obstacles_so_far += num_obstacles

    return feature_matrices, label_lists


#-----------------------------------------------------------------------------------------------------------------------------------------------------
# AUC for a binary classifier
def auc(y_true, y_pred):
    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
    binSizes = -(pfas[1:]-pfas[:-1])
    s = ptas*binSizes
    return K.sum(s, axis=0)

#-----------------------------------------------------------------------------------------------------------------------------------------------------
# PFA, prob false alert for binary classifier
def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # N = total number of negative labels
    N = K.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = K.sum(y_pred - y_pred * y_true)
    return FP/N
#-----------------------------------------------------------------------------------------------------------------------------------------------------
# P_TA prob true alerts for binary classifier
def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # P = total number of positive labels
    P = K.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class labels
    TP = K.sum(y_pred * y_true)
    return TP/P


def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())
