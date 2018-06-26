"""
This module is responsible to for the LSTM network

"""
from collections import Counter

from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import LSTM, K, Dense, TimeDistributed, Masking
import tensorflow as tf
from keras.preprocessing import sequence
from sklearn.metrics import confusion_matrix
import numpy as np
from numpy import array
from sklearn import metrics
import itertools
import tflearn # Used for roc_auc optimization
import model_factory
import setup_dataframes as sd

_nepochs = 1000
_maxlen = 0


# TODO: Split into train, test, validation set
# TODO: Save model in pickle file for later use
# TODO: Try softmax pred_classes
# TODO: https://towardsdatascience.com/hyperparameter-optimization-with-keras-b82e6364ca53

def get_trained_lstm_classifier(X, y, padding=True):
    """
    Reshapes feature matrix X, applies LSTM and returns the performance of the neural network

    :param X: List of non-reshaped/original feature matrices (one per logfile)
    :param y: labels

    :return: neural network
    """

    X_list, y_list = get_splitted_up_feature_matrix_and_labels(X, y)
    X_train_list, y_train_list, X_test_list, y_test_list = split_into_train_and_test_data(X_list, y_list, leave_out=2)

    globals()["_maxlen"] = max(len(fm) for fm in X_list)

    X_train_reshaped, y_train_reshaped = get_reshaped_feature_matrix(X_train_list, y_train_list, padding)

    X_lstm = array(X_train_reshaped)
    y_lstm = array(y_train_reshaped)

    model = generate_lstm_classifier(X_lstm, y_lstm)

    calculate_performance(X_test_list, y_test_list, model)


"""
1. Reshape feature matrix

We first need to get one feature matrix per logfile (Split up generated big feature matrix). Since datapoints are not 
sampled uniformly, the timedelta was added as a feature

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


def generate_lstm_classifier(X_reshaped, y_reshaped):
    """

    :param X_reshaped: Reshaped feature matrix
    :param y_reshaped: Reshaped labels

    :return: trained classifier
    """

    class_weights = {'0': 1, '1': 5}

    print('Shape X: ' + str(X_reshaped.shape))
    print('Shape y: ' + str(array(y_reshaped).shape) + '\n')

    # Metrics are NOT used in training phase (only loss function is tried to minimized)
    _metrics = [auc_roc,  recall, specificity, precision]
    my_callbacks = [EarlyStopping(monitor='auc_roc', patience=500, verbose=1, mode='max')]

    print('Compiling lstm network...')
    model = Sequential()

    model.add(Masking(mask_value=-99, input_shape=(X_reshaped.shape[1], X_reshaped.shape[2])))
    model.add(LSTM(128, return_sequences=True, activation='tanh'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(16, activation='relu'))

    # Allows to compute one Dense layer per Timestep (instead of one dense Layer per sample),
    # e.g. model.add(TimeDistributed(Dense(1)) computes one Dense layer per timestep for each sample
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=_metrics)
    model.fit(X_reshaped, y_reshaped, epochs=_nepochs, batch_size=150,
              verbose=1, shuffle=False, callbacks=my_callbacks)

    print(model.summary())

    return model


"""
3. Calculate performance
"""


def calculate_performance(X_test_list, y_test_list, lstm_model):
    """
    Iterates over all featurematrices (one per logfile), pads it to max_len (since lstm got trained on that length,
    then opredicts labels, and discards the padded ones
    :param X_splitted:
    :param lstm_model:
    :return:
    """
    y_pred_list = []
    for X in X_test_list:
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
    y_true = list(itertools.chain.from_iterable(y_test_list))

    conf_mat = confusion_matrix(y_true, y_pred)
    _precision = metrics.precision_score(y_true, y_pred)
    _recall = metrics.recall_score(y_true, y_pred)
    _specificity = conf_mat[0, 0] / (conf_mat[0, 0] + conf_mat[0, 1])
    _roc_auc = metrics.roc_auc_score(y_true, y_pred)

    print(model_factory.create_string_from_scores('LSTM', _roc_auc, _recall, _specificity, _precision, conf_mat))


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


def split_into_train_and_test_data(X_splitted, y_splitted, leave_out=1):
    """

    :param X_splitted:
    :param y_splitted:
    :param leave_out:
    :return:
    """

    return X_splitted[:-leave_out], y_splitted[:-leave_out], X_splitted[-leave_out:], y_splitted[-leave_out:]


def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    _precision = true_positives / (predicted_positives + K.epsilon())
    return _precision


def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    _recall = true_positives / (possible_positives + K.epsilon())
    return _recall


def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())


def auc_roc(y_true, y_pred):
    """
    https: // stackoverflow.com / a / 46844409
 
    """"""
    # any tensorflow metric
    value, update_op = tf.metrics.auc(y_true, y_pred)

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value