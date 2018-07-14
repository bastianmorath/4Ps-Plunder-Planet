"""
This module is responsible to for the LSTM network

"""
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import itertools

import keras
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras import regularizers
from keras.models import Sequential
from keras.layers import LSTM, K, Dense, TimeDistributed, Masking, RNN, Dropout, Bidirectional, BatchNormalization, \
    Activation
from keras.optimizers import RMSprop
from keras.preprocessing import sequence
from sklearn.metrics import confusion_matrix, roc_auc_score
from numpy import array
from sklearn import metrics
from keras import optimizers

import tensorflow as tf

import model_factory
import setup_dataframes as sd
import plots_helpers

_maxlen = 0

# TODO: Save model in pickle file for later use
# TODO: https://towardsdatascience.com/hyperparameter-optimization-with-keras-b82e6364ca53


def get_trained_lstm_classifier(X, y, n_epochs):
    """
    Reshapes feature matrix X, applies LSTM and returns the performance of the neural network

    :param X: List of non-reshaped/original feature matrices (one per logfile)
    :param y: labels

    :return: neural network
    """

    X_list, y_list = get_splitted_up_feature_matrix_and_labels(X, y)
    globals()["_maxlen"] = max(len(fm) for fm in X_list)
    X_train_list, y_train_list, X_test_list, y_test_list, X_val, y_val = split_into_train_and_test_data(X_list, y_list, leave_out=3)

    X_lstm, y_lstm = get_reshaped_matrices(X_train_list, y_train_list)
    X_val, y_val = get_reshaped_matrices(X_val, y_val)

    model = generate_lstm_classifier((X_lstm.shape[1], X_lstm.shape[2]))

    trained_model = train_lstm(model, X_lstm, y_lstm, X_val, y_val, n_epochs)
    print('Performance training set: ')
    calculate_performance(X_lstm, y_lstm, trained_model)
    print('Performance test set: ')
    calculate_performance(X_test_list, y_test_list, trained_model)

    # calculate_performance(X_lstm, y_lstm, trained_model)

    return model


"""
1. Reshape feature matrix

We first need to get one feature matrix per logfile (Split up generated big feature matrix). Since datapoints are not 
sampled uniformly, the timedelta was added as a feature

Samples: Number of logfiles
Time Steps: Number of data points/obstacles
Features: 9 or 16 (depending on feature_reduction or not)
"""


def get_reshaped_matrices(new_X, new_y):
    """ Returns the reshaped feature matrix needed for applying LSTM. Also does zero-padding

    :param new_X: Non-reshaped/original feature matrix (list)

    :return: Reshaped feature matrix (3D) as arrays
    """

    _minlen = min(len(fm) for fm in new_X)

    print('Maxlen (=Max. #obstacles of logfiles) is ' + str(_maxlen) + ', minlen is ' + str(_minlen))
    # I pad with 0s (Even though there are features that also are 0, it never occurs that ALL features of one timestep
    # are all 0, so those won't be masked later :)
    new_X = sequence.pad_sequences(new_X, maxlen=_maxlen, padding='post', dtype='float64')
    new_y = sequence.pad_sequences(new_y, maxlen=_maxlen, padding='post', dtype='float64')

    X_reshaped = array(new_X).reshape(len(new_X), _maxlen, new_X[0].shape[1])
    y_reshaped = array(new_y).reshape(len(new_y), _maxlen, 1)

    return array(X_reshaped), array(y_reshaped)


"""
2. Generate LSTM
"""


def generate_lstm_classifier(shape):
    """
    :return: trained classifier
    """

    print('Compiling lstm network...')
    model = Sequential()
    model.add(Masking(input_shape=shape))  # Mask out padded rows
    # model.add(Bidirectional(LSTM(64, return_sequences=True)))

    model.add(LSTM(128, return_sequences=True))
    model.add(Activation('relu'))
    model.add(Dropout(0.15))

    model.add(Dense(96))
    model.add(Activation('relu'))
    model.add(Dropout(0.15))

    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.15))


    # Allows to compute one Dense layer per Timestep (instead of one dense Layer per sample),
    # e.g. model.add(TimeDistributed(Dense(1)) computes one Dense layer per timestep for each sample
    model.add(TimeDistributed(Dense(2, activation='softmax')))

    # loss = WeightedCategoricalCrossEntropy({0: 1, 1: 8})
    # weights = np.array([1, 6])  # Higher weight on class 1 should translate to higher recall
    # loss = weighted_categorical_crossentropy(weights)

    adam = optimizers.adam(lr=0.011, decay=0.00001)
    rms_prop = RMSprop(lr=0.03)
    model.compile(loss='categorical_crossentropy', optimizer=adam, sample_weight_mode='temporal')

    return model


def train_lstm(trained_model, X_reshaped, y_reshaped, X_val, y_val,  n_epochs):
    print('\nShape X: ' + str(X_reshaped.shape))
    print('Shape y: ' + str(array(y_reshaped).shape) + '\n')
    one_hot_labels = keras.utils.to_categorical(y_reshaped, num_classes=2)
    # my_callbacks = [EarlyStopping(monitor='loss', patience=500, min_delta=0.001, verbose=1, mode='min')]

    to_2d = y_reshaped.reshape(y_reshaped.shape[0], y_reshaped.shape[1])    # sample_weights need a 2D array with shape
                                                                            # `(samples, sequence_length)`,
                                                                            # to apply a different weight to every
                                                                            # timestep of every sample.

    _sample_weight = np.array([[1 if v == 0 else 4 for v in row] for row in to_2d])

    # early_stopping = EarlyStopping(monitor='val_loss', patience=20)

    history_callback = Histories((X_reshaped, one_hot_labels))
    history = trained_model.fit(X_reshaped, one_hot_labels, epochs=n_epochs, batch_size=32,
                                verbose=1, shuffle=False, validation_split=0.1,
                                sample_weight=_sample_weight,
                                callbacks=[history_callback]
                                )

    print(history_callback.aucs_test)
    print(history_callback.aucs_train)

    # trained_model.save('trained_model.h5')  # creates a HDF5 file 'my_model.h5'
    print(trained_model.summary())
    # Plot
    # summarize history for roc_auc
    '''
    name = 'roc_auc'
    plt.plot(history.history[name])
    # plt.plot(history.history['val_'+name])
    plt.title('model ' + name)
    plt.ylabel(name)
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plots_helpers.save_plot(plt, 'Performance/LSTM/', 'LSTM ' + name)
    '''
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plots_helpers.save_plot(plt, 'Performance/LSTM/', 'LSTM loss')

    return trained_model


"""
3. Calculate performance
"""


def calculate_performance(X_test_list, y_test_list, lstm_model):
    """
    Iterates over all featurematrices (one per logfile), pads it to max_len (since lstm got trained on that length),
    then predicts labels, and discards the padded ones
    :param X_test_list:
    :param y_test_list:
    :param lstm_model:
    """
    y_pred_list = []
    _roc_auc_scores = []
    _precision_scores = []
    _specificity_scores = []
    _recall_scores = []
    for idx, X in enumerate(X_test_list):
        # We need to predict on maxlen, and can then take the first len(X_original) values
        length_old = len(X)
        X_reshaped = X.reshape(1, X.shape[0], X.shape[1])
        X_padded = sequence.pad_sequences(X_reshaped, maxlen=_maxlen, padding='post', dtype='float64')
        predictions = lstm_model.predict_classes(X_padded, batch_size=16, verbose=0).ravel()

        y_pred = predictions[:length_old]
        y_true = y_test_list[idx]

        y_pred_list.append(y_pred)  # Remove predictions for padded values

        _roc_auc_scores.append(metrics.roc_auc_score(y_true, y_pred))
        _recall_scores.append(metrics.recall_score(y_true, y_pred))
        _precision_scores.append(metrics.precision_score(y_true, y_pred))
        conf_mat = confusion_matrix(y_true, y_pred)
        _specificity_scores.append(conf_mat[0, 0] / (conf_mat[0, 0] + conf_mat[0, 1]))

    y_pred = list(itertools.chain.from_iterable(y_pred_list))
    y_true = list(itertools.chain.from_iterable(y_test_list))

    conf_mat = confusion_matrix(y_true, y_pred)
    print(_roc_auc_scores)
    print(model_factory.create_string_from_scores('LSTM', np.mean(_roc_auc_scores), np.std(_roc_auc_scores),
          np.mean(_recall_scores), np.std(_recall_scores), np.mean(_specificity_scores), np.mean(_precision_scores),
          np.std(_precision_scores), conf_mat))


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
    Splits up the list of Feature matrices and labels into Training and test sets, where size of
    test_set = leave_out

    :param X_splitted:
    :param y_splitted:
    :param leave_out:

    :return: X_train, X_test, y_train, y_test, each as a list
    """
    if leave_out == 0:
        return X_splitted, y_splitted, [], []

    import random
    c = list(zip(X_splitted, y_splitted))
    random.shuffle(c)

    X_splitted, y_splitted = zip(*c)

    X_train = X_splitted[leave_out + 1:]
    X_test = X_splitted[:leave_out]
    X_val = [X_splitted[leave_out]]
    y_train = y_splitted[leave_out + 1:]
    y_test = y_splitted[:leave_out]
    y_val = [y_splitted[leave_out]]

    return X_train, y_train, X_test, y_test, X_val, y_val




# define roc_callback, inspired by https://github.com/keras-team/keras/issues/6050#issuecomment-329996505
def auc_roc(y_true, y_pred):
    # any tensorflow metric
    value, update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true)

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

'''
def f1_score(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    # How many relevant items are selected?
    recall = c1 / c3

    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


def precision(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    return precision


def recall(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    recall = c1 / c3

    return recall
'''

class Histories(keras.callbacks.Callback):

    def __init__(self, training_data):
        self.aucs_train = []
        self.aucs_test = []
        self.losses = []
        self.val_losses = []
        self.training_data = training_data

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        print(self.training_data[0].shape)
        print(self.training_data[1].shape)

        pred_y_train = self.model.predict_classes(self.training_data[0])
        y_pred_train_conc = list(itertools.chain.from_iterable(pred_y_train))
        y_true_train = list(itertools.chain.from_iterable(self.training_data[1]))
        y_true_train_conc = np.argmax(y_true_train, axis=1)

        roc_auc_train = roc_auc_score(y_true_train_conc, y_pred_train_conc)
        print('Roc_auc on training set: ' + str(round(roc_auc_train, 3)))

        val_x = self.validation_data[0]
        val_y_conc = list(itertools.chain.from_iterable(self.validation_data[1]))
        y_pred_conc = list(itertools.chain.from_iterable(self.model.predict_classes(val_x)))
        y_true_conc = np.argmax(val_y_conc, axis=1)
        roc_auc_test = roc_auc_score(y_true_conc, y_pred_conc)
        print('Roc_auc on validation set: ' + str(round(roc_auc_test, 3)))

        self.aucs_train.append(round(roc_auc_train, 3))
        self.aucs_test.append(round(roc_auc_test, 3))

        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return