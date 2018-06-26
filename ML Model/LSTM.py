"""
This module is responsible to for the LSTM network

"""
from collections import Counter

from keras import Sequential
from keras.layers import LSTM, K, Dense, TimeDistributed, Masking
import tensorflow as tf
from keras.preprocessing import sequence
from sklearn.metrics import confusion_matrix
import numpy as np
from numpy import array
from sklearn import metrics
import itertools

import model_factory
import setup_dataframes as sd

_nepochs = 30
_maxlen = 0


# TODO: Split into train, test, validation set
# TODO: Save model in pickle file for later use
# TODO: Try softmax pred_classes


def get_trained_lstm_classifier(X, y, padding=True):
    """
    Reshapes feature matrix X, applies LSTM and returns the performance of the neural network

    :param X: List of non-reshaped/original feature matrices (one per logfile)
    :param y: labels

    :return: neural network
    """

    X_splitted, y_splitted = get_splitted_up_feature_matrix_and_labels(X, y)

    globals()["_maxlen"] = max(len(fm) for fm in X_splitted)

    X_reshaped, y_reshaped = get_reshaped_feature_matrix(X_splitted, y_splitted, padding)

    X_lstm = array(X_reshaped)
    y_lstm = array(y_reshaped)

    model = generate_lstm_classifier(X_lstm, y_lstm, padding=padding)

    calculate_performance(X_splitted, y_splitted, model)


"""
1. Reshape feature matrix

We first need to get one feature matrix per logfile (Split up generated big feature matrix). We also need to resample it
 (datapoints must be uniformly sampled!) We can also encode time as a new feature maybe...

Samples: Number of logfiles
Time Steps: Number of data points/obstacles
Features: 9 or 16 (depending on feature_reduction or not)
"""


def get_reshaped_feature_matrix(new_X, new_y, padding=True):
    """ Returns the reshaped feature matrix needed for applying LSTM. Also does zero-padding

    :param new_X: Non-reshaped/original feature matrix (list)

    :return: Reshaped feature matrix (3D)
    """

    minlen = min(len(fm) for fm in new_X)

    print('Maxlen (=Max. #obstacles of logfiles) is ' + str(_maxlen) + ', minlen is ' + str(minlen))
    # Since values of feature matrix are between -1 and +1, I pad not with 0, but with e.g. -99
    if padding:
        new_X = sequence.pad_sequences(new_X, maxlen=_maxlen, padding='post', dtype='float64', value=-99)
        new_y = sequence.pad_sequences(new_y, maxlen=_maxlen, padding='post', dtype='float64', value=-99)

        X_reshaped = array(new_X).reshape(len(new_X), _maxlen, new_X[0].shape[1])
        y_reshaped = array(new_y).reshape(len(new_y), _maxlen, 1)
    else:
        X_reshaped = array(new_X).reshape(len(new_X), None, new_X[0].shape[1])
        y_reshaped = array(new_y).reshape(len(new_y), 1)

    return X_reshaped, y_reshaped


"""
2. Generate LSTM
"""


def generate_lstm_classifier(X_reshaped, y_reshaped, padding=True):
    """

    :param X_reshaped: Reshaped feature matrix
    :param y_reshaped: Reshaped labels

    :return: trained classifier
    """

    class_weights = {'0': 1, '1': 5}

    print('Shape X: ' + str(X_reshaped.shape))
    print('Shape y: ' + str(array(y_reshaped).shape) + '\n')

    _metrics = [auc, sensitivity, 'accuracy']
    print('Compiling lstm network...')
    model = Sequential()

    if padding:
        model.add(Masking(mask_value=-99, input_shape=(X_reshaped.shape[1], X_reshaped.shape[2])))
        model.add(LSTM(256, return_sequences=True))
        model.add(Dense(64, activation='sigmoid'))

        model.add(TimeDistributed(Dense(1, activation='sigmoid')))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=_metrics)
        model.fit(X_reshaped, y_reshaped, epochs=_nepochs, batch_size=50,
                  verbose=1, shuffle=False)
    else:
        model.add(LSTM(120, return_sequences=True, input_shape=(None, X_reshaped.shape[2])))
        model.add(TimeDistributed(Dense(1, activation='sigmoid')))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=_metrics)
        model.fit(X_reshaped, y_reshaped, epochs=_nepochs, batch_size=1,
                  verbose=1, shuffle=False)
    print(model.summary())

    return model


"""
3. Calculate performance
"""


def calculate_performance(X_splitted, y_true_list, lstm_model):
    """
    Iterates over all featurematrices (one per logfile), pads it to max_len (since lstm got trained on that length,
    then opredicts labels, and discards the padded ones
    :param X_splitted:
    :param lstm_model:
    :return:
    """
    y_pred_list = []
    for X in X_splitted:
        # We need to predict on maxlen, and can then take the first len(X_original) values
        length_old = len(X)
        X = X.reshape(1, X.shape[0], X.shape[1])
        X_padded = sequence.pad_sequences(X, maxlen=_maxlen, padding='post', dtype='float64', value=-99)
        predictions = lstm_model.predict_proba(X_padded, batch_size=1, verbose=0).ravel()[:length_old]
        predictions_truncated = predictions  # Remove predictions for padded values

        y_pred_list.append([int(round(x)) for x in predictions_truncated])


        """
        c = Counter(result)
        pred = []
        for v, times in c.items():
            if v < 0.5:
                pred.extend([0] * times)
            else:
                pred.extend([1] * times)
        c2 = Counter(pred)

        print(c2)
        """

    y_pred = list(itertools.chain.from_iterable(y_pred_list))
    y_true = list(itertools.chain.from_iterable(y_true_list))

    conf_mat = confusion_matrix(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    specificity = conf_mat[0, 0] / (conf_mat[0, 0] + conf_mat[0, 1])
    roc_auc = metrics.roc_auc_score(y_true, y_pred)

    print(model_factory.create_string_from_scores('LSTM', roc_auc, recall, specificity, precision, conf_mat))


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


