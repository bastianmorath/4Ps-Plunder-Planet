"""
This module is responsible to for the LSTM network

"""
import matplotlib.pyplot as plt
import numpy as np
import itertools

import keras
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import LSTM, K, Dense, TimeDistributed, Masking, Dropout
from keras.optimizers import SGD
from keras.preprocessing import sequence
from sklearn.metrics import confusion_matrix
from numpy import array
from sklearn import metrics
from keras import optimizers

import tensorflow as tf

import model_factory
import setup_dataframes as sd
import feature_plots

_maxlen = 0

# TODO: Split into train, test, validation set
# TODO: Save model in pickle file for later use
# TODO: Try softmax pred_classes
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
    X_train_list, y_train_list, X_test_list, y_test_list = split_into_train_and_synthesized_data(X_list, y_list, leave_out=5)

    X_lstm, y_lstm = get_reshaped_matrices(X_train_list, y_train_list)

    model = generate_lstm_classifier(X_lstm, y_lstm, n_epochs)

    calculate_performance(X_test_list, y_test_list, model)

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


def generate_lstm_classifier(X_reshaped, y_reshaped, n_epochs):
    """

    :param X_reshaped: Reshaped feature matrix
    :param y_reshaped: Reshaped labels
    :param n_epochs: NUmber of epochs to train

    :return: trained classifier
    """

    print('\nShape X: ' + str(X_reshaped.shape))
    print('Shape y: ' + str(array(y_reshaped).shape) + '\n')

    print('Compiling lstm network...')
    model = Sequential()
    model.add(Masking(input_shape=(X_reshaped.shape[1], X_reshaped.shape[2])))
    model.add(LSTM(32, return_sequences=True))
    model.add(Dense(32))
    # Allows to compute one Dense layer per Timestep (instead of one dense Layer per sample),
    # e.g. model.add(TimeDistributed(Dense(1)) computes one Dense layer per timestep for each sample
    model.add(TimeDistributed(Dense(2, activation='softmax')))

    # loss = WeightedCategoricalCrossEntropy({0: 3, 1: 1})  # TODO: Maybe 0.8?
    adam = optimizers.adam(lr=0.019, decay=0.00001)
    print('Learning rate and decay: ' + str(0.019) + ', ' + str(adam.decay))
    # Metrics are NOT used in training phase (only loss function is tried to be minimized)
    _metrics = [roc_auc, 'accuracy']
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=_metrics)

    # shuffle=True takes entire samples and shuffles those
    # TODO: Use validation_data or validation_split? Maybe validation_split doesn't take entire samples?

    one_hot_labels = keras.utils.to_categorical(y_reshaped, num_classes=2)
    my_callbacks = [EarlyStopping(monitor='roc_auc', patience=500, min_delta=0.001, verbose=1, mode='max')]

    history = model.fit(X_reshaped, one_hot_labels, epochs=n_epochs, batch_size=64,
                        verbose=1, shuffle=False, callbacks=my_callbacks, validation_split=0.2)
    # TODO: Use model.evaluate with test_Data
    # https: // github.com / keras - team / keras / issues / 1753
    print(model.summary())

    # Plot
    # summarize history for roc_auc

    plt.plot(history.history['roc_auc'])
    plt.plot(history.history['val_roc_auc'])
    plt.title('model roc_auc')
    plt.ylabel('roc_auc')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    feature_plots.save_plot(plt, 'Performance/LSTM/', 'LSTM roc_auc')

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    feature_plots.save_plot(plt, 'Performance/LSTM/', 'LSTM loss')

    return model


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
    for X in X_test_list:
        # We need to predict on maxlen, and can then take the first len(X_original) values
        length_old = len(X)
        X_reshaped = X.reshape(1, X.shape[0], X.shape[1])
        X_padded = sequence.pad_sequences(X_reshaped, maxlen=_maxlen, padding='post', dtype='float64', value=-2)
        predictions = lstm_model.predict_classes(X_padded, batch_size=10, verbose=0).ravel()
        y_pred_list.append(predictions[:length_old])  # Remove predictions for padded values

    y_pred = list(itertools.chain.from_iterable(y_pred_list))
    y_true = list(itertools.chain.from_iterable(y_test_list))
    #for a, b in zip(y_pred, y_true):
    #    print(a, b)
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
    Splits up the list of Feature matrices and labels into Training and test sets, where size of
    test_set = leave_out

    :param X_splitted:
    :param y_splitted:
    :param leave_out:

    :return: X_train, X_test, y_train, y_test, each as a list
    """

    import random
    c = list(zip(X_splitted, y_splitted))
    random.shuffle(c)
    X_splitted, y_splitted = zip(*c)

    X_train = X_splitted[:-leave_out]
    X_test = X_splitted[-leave_out:]
    y_train = y_splitted[:-leave_out]
    y_test = y_splitted[-leave_out:]

    return X_train, y_train, X_test, y_test


def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    # any tensorflow metric
    value, update_op = tf.metrics.precision(y_true, y_pred)

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'precision' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value


def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    # any tensorflow metric
    value, update_op = tf.metrics.recall(y_true, y_pred)

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'recall' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value

'''
def specificity(y_true, y_pred):
    
'''


def roc_auc(y_true, y_pred):
    """
    https: // stackoverflow.com / a / 46844409
 
    """
    # any tensorflow metric
    value, update_op = tf.metrics.auc(y_true, y_pred)

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'roc_auc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value


class WeightedBinaryCrossEntropy(object):

    def __init__(self, pos_ratio):
        """

        :param pos_ratio: Ratio of positive to negative labels (0.5 meaning 1:1)
        """
        neg_ratio = 1. - pos_ratio
        self.pos_ratio = tf.constant(pos_ratio, tf.float32)
        self.weights = tf.constant(neg_ratio / pos_ratio, tf.float32)
        self.__name__ = "weighted_binary_crossentropy({0})".format(pos_ratio)

    def __call__(self, y_true, y_pred):
        return self.weighted_binary_crossentropy(y_true, y_pred)

    def weighted_binary_crossentropy(self, y_true, y_pred):
        # Transform to logits
        epsilon = tf.convert_to_tensor(K.common._EPSILON, y_pred.dtype.base_dtype)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        y_pred = tf.log(y_pred / (1 - y_pred))

        cost = tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, self.weights)
        return K.mean(cost * self.pos_ratio, axis=-1)


class WeightedCategoricalCrossEntropy(object):

  def __init__(self, weights):
    nb_cl = len(weights)
    self.weights = np.ones((nb_cl, nb_cl))
    for class_idx, class_weight in weights.items():
      self.weights[0][class_idx] = class_weight
      self.weights[class_idx][0] = class_weight
    self.__name__ = 'w_categorical_crossentropy'

  def __call__(self, y_true, y_pred):
    return self.w_categorical_crossentropy(y_true, y_pred)

  def w_categorical_crossentropy(self, y_true, y_pred):
    nb_cl = len(self.weights)
    final_mask = K.zeros_like(y_pred[..., 0])
    y_pred_max = K.max(y_pred, axis=-1)
    y_pred_max = K.expand_dims(y_pred_max, axis=-1)
    y_pred_max_mat = K.equal(y_pred, y_pred_max)
    for c_p, c_t in itertools.product(range(nb_cl), range(nb_cl)):
        w = K.cast(self.weights[c_t, c_p], K.floatx())
        y_p = K.cast(y_pred_max_mat[..., c_p], K.floatx())
        y_t = K.cast(y_true[..., c_t], K.floatx())
        final_mask += w * y_p * y_t
    return K.categorical_crossentropy(y_pred, y_true) * final_mask