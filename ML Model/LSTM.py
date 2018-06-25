"""
This module is responsible to for the LSTM network

"""

import setup_dataframes as sd


def get_trained_lstm_classifier(X, y):
    """
    Reshapes feature matrix X, applies LSTM and returns the performance of the neural network

    :param X: List of non-reshaped/original feature matrices (one per logfile)
    :param y: labels

    :return: neural network
    """
    X_splitted = get_splitted_up_feature_matrix(X)
    X_reshaped = get_reshaped_feature_matrix(X_splitted)


"""
1. Reshape feature matrix

We first need to get one feature matrix per logfile (Split up generated big feature matrix). We also need to resample it
 (datapoints must be uniformly sampled!) We can also encode time as a new feature maybe...

Samples: Number of logfiles
Time Steps: Number of data points/obstacles
Features: 9 or 16 (depending on feature_reduction or not)
"""


def get_reshaped_feature_matrix(X):
    """ Returns the reshaped feature matrix needed for applying LSTM

    :param X: Non-reshaped/origianl feature matrix

    :return: Reshaped feature matrix (3D)
    """

    return X


"""
2. Apply LSTM
"""


def apply_lstm(X_reshaped, y):
    """

    :param X_reshaped: Reshaped feature matrix
    :param y: labels

    :return: trained classifier
    """


"""
3. Calculate performance
"""

"""
Helper methods
"""


def get_splitted_up_feature_matrix(X):
    """
    Feature amtrix X is the concatenation of all feature matrices per logfile. We need to split it up such that we
    have one feature matrix per logfile

    :param X: Non-reshaped, original feature matrix

    :return: List of feature matrices, non-reshaped
    """

    feature_matrices = []
    obstacles_so_far = 0
    for df in sd.obstacle_df_list:
        num_obstacles = len(df.index)
        feature_matrices.append(X.take(range(obstacles_so_far, obstacles_so_far + num_obstacles), axis=0))
        obstacles_so_far += num_obstacles

    return feature_matrices



