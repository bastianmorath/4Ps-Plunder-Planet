"""
This module is responsible to for setting up, compiling, training and evaluating a LSTM Recurrent Neural Network

"""

import itertools
import random

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import optimizers
from keras.layers import LSTM, Dense, Dropout, Masking, TimeDistributed
from keras.models import Sequential
from keras.preprocessing import sequence
from matplotlib.ticker import FormatStrFormatter
from numpy import array
from sklearn import metrics
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, f1_score, roc_curve)

import model_factory
import plots_helpers
import setup_dataframes as sd

_maxlen = 0  # Length of the feature matrix of the logfile with the most obstacles
_seed = 1  # Seed used to split into testtrain/test/val data

threshold_tuning = True  # Whether optimal threshold of ROC should be used (calc. with Youdens j-score)


def get_finalscore(X, y, n_epochs, verbose=0):
    roc_aucs_mean: [float] = []
    f1s_mean: [float] = []
    for i in range(0, 15):
        roc_mean, f1_mean = get_performance_of_lstm_classifier(X, y, n_epochs, verbose, final_score=True)
        roc_aucs_mean.append(roc_mean)
        f1s_mean.append(f1_mean)

    print([round(float(r), 2) for r in roc_aucs_mean])

    print('Max roc: ' + str(round(np.max(roc_aucs_mean), 3)))
    print('Mean roc: ' + str(round(float(np.mean(roc_aucs_mean)), 3)) +
          ', (+-' + str(round(float(np.std(roc_aucs_mean)), 3)) + ')')
    print('Max f1: ' + str(round(np.max(f1s_mean), 3)))
    print('Mean f1: ' + str(round(float(np.mean(f1s_mean)), 3)) +
          ', (+-' + str(round(float(np.std(f1s_mean)), 3)) + ')')


def get_performance_of_lstm_classifier(X, y, n_epochs, verbose=1, final_score=False):
    """
    Reshapes feature matrix X, applies LSTM and returns the performance of the neural network

    :param X: List of non-reshaped/original feature matrices (one per logfile)
    :param y: labels
    :param n_epochs: Number of epochs the model should be trained
    :param verbose: verbose mode of keras_model.fit
    :param final_score: If final score should be printed, then don't use a validation set
    :return mean auroc, std auroc
    """

    X_list, y_list = _get_splitted_up_feature_matrix_and_labels(X, y)
    globals()["_maxlen"] = max(len(fm) for fm in X_list)

    if final_score:
        X_train_list, y_train_list, X_test_list, y_test_list, X_val, y_val = \
            _split_into_train_test_val_data(X_list, y_list, size_test_set=5, size_val_set=0)
        X_lstm, y_lstm = _get_reshaped_matrices(X_train_list, y_train_list)
        model = _generate_lstm_classifier((X_lstm.shape[1], X_lstm.shape[2]))

        trained_model = _train_lstm(model, X_lstm, y_lstm, n_epochs, verbose)

    else:
        X_train_list, y_train_list, X_test_list, y_test_list, X_val, y_val = \
            _split_into_train_test_val_data(X_list, y_list, size_test_set=3, size_val_set=2)
        X_lstm, y_lstm = _get_reshaped_matrices(X_train_list, y_train_list)
        model = _generate_lstm_classifier((X_lstm.shape[1], X_lstm.shape[2]))

        trained_model = _train_lstm(model, X_lstm, y_lstm, n_epochs, verbose, val_set=(X_val, y_val))
        print('Performance training set: ')
        _calculate_performance(X_lstm, y_lstm, trained_model)

    print('Performance test set: ')
    mean_auroc, std_auroc = _calculate_performance(X_test_list, y_test_list, trained_model)
    return mean_auroc, std_auroc


"""
1. Reshape feature matrix

We first need to get one feature matrix per logfile (Split up generated big feature matrix). Since datapoints are not 
sampled uniformly, the timedelta was added as a feature

Samples: Number of logfiles
Time Steps: Number of data points/obstacles
Features: 9 or 16 (depending on feature_reduction or not)
"""


def _get_reshaped_matrices(new_X, new_y):
    """ Returns the reshaped feature matrix needed for applying LSTM. Also does zero-padding

    :param new_X: Non-reshaped/original feature matrix (list)

    :return: Reshaped feature matrix (3D) as arrays
    """

    # _minlen = min(len(fm) for fm in new_X)

    # print('Maxlen (=Max. #obstacles of logfiles) is ' + str(_maxlen) + ', minlen is ' + str(_minlen))
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


def _generate_lstm_classifier(shape):
    """
    :return: trained classifier
    """

    dropout = 0.2
    print('Compiling lstm network...')
    model = Sequential()
    model.add(Masking(input_shape=shape))  # Mask out padded rows

    model.add(LSTM(96, return_sequences=True, activation='tanh'))
    model.add(Dropout(dropout))

    model.add(Dense(96, activation='relu'))
    model.add(Dropout(dropout))

    model.add(Dense(96, activation='relu'))
    model.add(Dropout(dropout))

    # Allows to compute one Dense layer per Timestep (instead of one dense Layer per sample),
    # e.g. model.add(TimeDistributed(Dense(1)) computes one Dense layer per timestep for each sample
    model.add(TimeDistributed(Dense(2, activation='softmax')))

    adam = optimizers.adam(lr=0.0003, decay=0.000004, amsgrad=True)
    model.compile(loss='categorical_crossentropy', optimizer=adam, sample_weight_mode='temporal')

    return model


def _train_lstm(trained_model, X_train, y_train, n_epochs, verbose=1, val_set=None):
    validation = val_set is not None
    one_hot_labels_train = keras.utils.to_categorical(y_train, num_classes=2)
    if validation:
        one_hot_labels_val = keras.utils.to_categorical(val_set[1], num_classes=2)

    # sample_weights need a 2D array with shape `(samples, sequence_length)`,
    # to apply a different weight to every timestep of every sample.
    to_2d = y_train.reshape(y_train.shape[0], y_train.shape[1])

    _sample_weight = np.array([[1 if v == 0 else 4 for v in row] for row in to_2d])

    history_callback = Histories((X_train, one_hot_labels_train), n_epochs, interval=10, drawing_enabled=True) \
        if validation else []

    trained_model.fit(X_train, one_hot_labels_train, epochs=n_epochs, batch_size=128,
                      verbose=verbose, shuffle=True, validation_data=None if not validation else
                      (val_set[0], one_hot_labels_val),
                      sample_weight=_sample_weight,
                      callbacks=history_callback
                      )

    # trained_model.save('trained_model.h5')  # creates a HDF5 file 'my_model.h5'

    # Plot
    if validation:
        print(trained_model.summary())

        _plot_losses_and_roc_aucs(history_callback.aucs_train, history_callback.aucs_val,
                                  history_callback.train_losses, history_callback.val_losses,
                                  history_callback.f1s_train, history_callback.f1s_val,
                                  n_epochs
                                  )

    return trained_model


"""
3. Calculate performance
"""


def _calculate_performance(X_test_list, y_test_list, lstm_model):
    """
    Iterates over all feature matrices (one per logfile), pads it to max_len (since lstm got trained on that length),
    then predicts labels, and discards the padded ones
    :param X_test_list:
    :param y_test_list:
    :param lstm_model:

    :return mean auroc, mean f1
    """
    y_pred_list = []
    _roc_auc_scores = []
    _precision_scores = []
    _f1_scores = []
    _specificity_scores = []
    _recall_scores = []
    for idx, X in enumerate(X_test_list):
        # We need to predict on maxlen, and can then take the first len(X_original) values
        length_old = len(X)
        X_reshaped = X.reshape(1, X.shape[0], X.shape[1])
        X_padded = sequence.pad_sequences(X_reshaped, maxlen=_maxlen, padding='post', dtype='float64')
        # predictions = lstm_model.predict_classes(X_padded, batch_size=16, verbose=0).ravel()
        probas = lstm_model.predict_proba(X_padded, batch_size=16, verbose=0)
        predictions_probas = [b for (a, b) in probas[0][:length_old]]
        y_true = y_test_list[idx]

        fpr, tpr, thresholds = roc_curve(y_true, predictions_probas)
        threshold = model_factory.cutoff_youdens_j(fpr, tpr, thresholds) if threshold_tuning else 0.5
        y_pred = [1 if p > threshold else 0 for p in predictions_probas]

        #  y_pred = predictions[:length_old]

        y_pred_list.append(y_pred)  # Remove predictions for padded values

        _roc_auc_scores.append(metrics.roc_auc_score(y_true, predictions_probas))
        _recall_scores.append(metrics.recall_score(y_true, y_pred))
        _precision_scores.append(metrics.precision_score(y_true, y_pred))
        _f1_scores.append(metrics.f1_score(y_true, y_pred))
        conf_mat = confusion_matrix(y_true, y_pred)
        _specificity_scores.append(conf_mat[0, 0] / (conf_mat[0, 0] + conf_mat[0, 1]))

    y_pred = list(itertools.chain.from_iterable(y_pred_list))
    y_true = list(itertools.chain.from_iterable(y_test_list))

    conf_mat = confusion_matrix(y_true, y_pred)

    print(model_factory.create_string_from_scores('LSTM', np.mean(_roc_auc_scores), np.std(_roc_auc_scores),
          np.mean(_recall_scores), np.std(_recall_scores), np.mean(_specificity_scores), np.std(_specificity_scores),
          np.mean(_precision_scores), np.std(_precision_scores), conf_mat))

    return np.mean(_roc_auc_scores), np.mean(_f1_scores)


"""
Helper methods
"""


def _plot_losses_and_roc_aucs(aucs_train, aucs_val, train_losses, val_losses, f1s_train, f1s_val, n_epochs):
    """
    Plots the losses, roc_auc/recall/precision/f1 scores that were obtained during the training phase
    (with the help of the callback).


    :param aucs_train:      list of roc_auc scores on training data
    :param aucs_val:        list of roc_auc scores on validation data
    :param train_losses:    list of losses on training data
    :param val_losses:      list of losses on validation data
    :param f1s_train:       list of f1s scores on training data
    :param f1s_val:         list of f1s scores on validation data
    :param n_epochs:        Number of epochs the model should be trained
    """

    # AUCS
    _, ax = plt.subplots()
    if n_epochs is not None:
        ax.set_xlim(0, n_epochs)
    plt.ylabel('roc_auc')
    plt.xlabel('Epochs')
    plt.title('Training and validation roc_auc after each epoch')
    index = np.arange(len(aucs_train))

    plt.plot(index, aucs_train, label='roc_auc train')
    plt.plot(index, aucs_val, label='roc_auc validation')
    plt.axhline(y=0.5, color=plots_helpers.red_color, linestyle='--', label='random guess')
    plt.legend()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    plots_helpers.save_plot(plt, 'LSTM/', 'roc_auc_plot_' + str(n_epochs) + '.pdf')

    # LOSSES
    _, ax = plt.subplots()
    if n_epochs is not None:
        ax.set_xlim(0, n_epochs)
    plt.ylabel('loss')
    plt.xlabel('Epochs')
    plt.title('Training and validation loss after each epoch')
    index = np.arange(len(train_losses))

    plt.plot(index, train_losses, label='training loss')
    plt.plot(index, val_losses, label='validation loss')
    plt.legend()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

    plots_helpers.save_plot(plt, 'LSTM/', 'losses_' + str(n_epochs) + '.pdf')

    # F1
    _, ax = plt.subplots()
    if n_epochs is not None:
        ax.set_xlim(0, n_epochs)
    plt.ylabel('f1')
    plt.xlabel('Epochs')
    plt.title('Training and validation f1 scores after each epoch')
    index = np.arange(len(train_losses))

    plt.plot(index, f1s_train, label='f1 training')
    plt.plot(index, f1s_val, label='f1 validation')
    plt.legend()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    plots_helpers.save_plot(plt, 'LSTM/', 'f1s_' + str(n_epochs) + '.pdf')


def _get_splitted_up_feature_matrix_and_labels(X, y):
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


def _split_into_train_test_val_data(X_splitted, y_splitted, size_test_set=3, size_val_set=3):
    """
    Splits up the list of Feature matrices and labels into Training, test and validation sets, where size of
    test_set = leave_out, val_set=2, training_set=rest.

    Also does preprocessing by computing mean/var on training_set only, then subtract on test_set

    :param X_splitted:
    :param y_splitted:
    :param size_test_set:
    :param size_val_set:
    :return: X_train, X_test, y_train, y_test, each as a list
    """

    # random.seed(_seed)
    c = list(zip(X_splitted, y_splitted))
    random.shuffle(c)

    X_splitted, y_splitted = zip(*c)

    X_train = X_splitted[size_test_set+size_val_set:]  # from leave_out up to end
    X_test = X_splitted[:size_test_set]  # 0 up to and including leave_out-1

    X_val = X_splitted[size_test_set:size_test_set+size_val_set]  # index leave_out itself
    y_train = y_splitted[size_test_set+size_val_set:]
    y_test = y_splitted[:size_test_set]
    y_val = y_splitted[size_test_set:size_test_set+size_val_set]

    return X_train, y_train, X_test, y_test, X_val, y_val


class Histories(keras.callbacks.Callback):

    def __init__(self, training_data, n_epochs, interval=10, drawing_enabled=True):
        super().__init__()
        self.aucs_train = []
        self.aucs_val = []
        self.f1s_train = []
        self.f1s_val = []

        self.train_losses = []
        self.val_losses = []

        self.training_data = training_data
        self.count = 0
        self.interval = interval
        self.n_epochs = n_epochs
        self.drawing_enabled = drawing_enabled

        if drawing_enabled:
            _plot_losses_and_roc_aucs([], [], [], [], [], [], n_epochs)

    def on_train_begin(self, logs=None):
        return

    def on_train_end(self, logs=None):
        return

    def on_epoch_begin(self, epoch, logs=None):
        return

    def on_epoch_end(self, epoch, logs=None):
        self.train_losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

        pred_y_train = self.model.predict_classes(self.training_data[0])
        y_pred_train_conc = list(itertools.chain.from_iterable(pred_y_train))

        y_true_train = list(itertools.chain.from_iterable(self.training_data[1]))
        y_true_train_conc = np.argmax(y_true_train, axis=1)

        roc_auc_train = roc_auc_score(y_true_train_conc, y_pred_train_conc)
        f1_train = f1_score(y_true_train_conc, y_pred_train_conc)

        val_x = self.validation_data[0]
        val_y_conc = list(itertools.chain.from_iterable(self.validation_data[1]))
        y_pred_conc = list(itertools.chain.from_iterable(self.model.predict_classes(val_x)))
        probas = list(itertools.chain.from_iterable(self.model.predict_proba(val_x)))
        y_pred_conc_probas = [b for (a, b) in probas]

        y_true_conc = np.argmax(val_y_conc, axis=1)
        roc_auc_val = roc_auc_score(y_true_conc, y_pred_conc_probas)
        f1_val = f1_score(y_true_conc, y_pred_conc)

        self.aucs_train.append(round(roc_auc_train, 3))
        self.aucs_val.append(round(roc_auc_val, 3))
        self.f1s_train.append(round(f1_train, 3))
        self.f1s_val.append(round(f1_val, 3))

        if self.drawing_enabled and self.count % self.interval == 0:  # Redraw plot every 10 iterations
            _plot_losses_and_roc_aucs(self.aucs_train, self.aucs_val,
                                      self.train_losses, self.val_losses,
                                      self.f1s_train, self.f1s_val,
                                      n_epochs=self.n_epochs
                                      )
        self.count += 1

        return

    def on_batch_begin(self, batch, logs=None):
        return

    def on_batch_end(self, batch, logs=None):
        return
