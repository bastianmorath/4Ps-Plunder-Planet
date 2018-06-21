"""This module contains different classifier classes. They are all subclasses of the Classifier-class and
contain hyperparameters to do grid search over and the classifier obejct itself

"""


import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from scipy.stats import randint as sp_randint
from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import (AdaBoostClassifier,
                              GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.tree import DecisionTreeClassifier

import features_factory as f_factory


EPSILON = 0.0001


def get_clf_with_name(clf_name, X, y):
    """Returns the classifier with the given name

    :param clf_name: name of the classifier

    :return: classifier
    """

    clfs = [
        CSVM(X, y),
        CLinearSVM(X, y),
        CNearestNeighbors(X, y),
        CQuadraticDiscriminantAnalysis(X, y),
        CGradientBoostingClassifier(X, y),
        CDecisionTreeClassifier(X, y),
        CRandomForest(X, y),
        CAdaBoost(X, y),
        CNaiveBayes(X, y),
    ]

    list_clf = [clf for clf in clfs if clf.name == clf_name]
    assert len(list_clf) != 0, 'Classifier does not exist. Available classifiers: ' + str([clf.name for clf in clfs])
    return list_clf[0]


class CClassifier(object):
    def __init__(self, X, y):

        self.X = X
        self.y = y


class CSVM(CClassifier):
    name = 'SVM'

    param1 = sp_randint(1, 100)  # C
    param1_name = 'C'
    param2 = sp_randint(EPSILON, 10)  # gamma
    param2_name = 'gamma'
    tuned_params = {'C': param1, 'gamma': param2}

    def __init__(self, X, y):
        CClassifier.__init__(self, X, y)
        self.clf = SVC(class_weight='balanced')


class CLinearSVM(CClassifier):
    name = 'Linear SVM'
    param1 = sp_randint(1, 100)  # C
    param1_name = 'C'
    param2 = ['l1', 'l2']  # penalty
    param2_name = 'penalty'
    tuned_params = {'C': param1, 'penalty': param2}

    def __init__(self, X, y):
        CClassifier.__init__(self, X, y)
        self.clf = LinearSVC(class_weight='balanced', dual=False)

'''
class CLinearSVM(CClassifier):
    name = 'Linear SVM'
    param1 = sp_randint(1, 100)  # C
    param1_name = 'C'
    param2 = sp_randint(EPSILON, 10)  # gamma
    param2_name = 'gamma'
    tuned_params = {'C': param1, 'gamma': param2}

    clf = SVC(class_weight='balanced', degree=1)

    def __init__(self, X, y):
        CClassifier.__init__(self, X, y)
'''


class CNearestNeighbors(CClassifier):
    name = 'Nearest Neighbor'

    param1 = sp_randint(1, 100)  # n_neighbors
    param1_name = 'n_neighbors'
    param2 = ['minkowski', 'euclidean', 'manhattan']  # metric
    param2_name = 'metric'
    tuned_params = {'n_neighbors': param1, 'metric': param2}

    def __init__(self, X, y):
        CClassifier.__init__(self, X, y)
        self.clf = KNeighborsClassifier()


class CQuadraticDiscriminantAnalysis(CClassifier):
    name = 'QDA'

    param1 = sp_randint(0, 1)  # reg_param
    param1_name = 'reg_param'
    param2 = np.random.uniform(EPSILON, 1)  # tol
    param2_name = 'tol'
    tuned_params = {'reg_param': param1, 'tol': param2}

    def __init__(self, X, y):
        CClassifier.__init__(self, X, y)
        self.clf = QuadraticDiscriminantAnalysis()


class CGradientBoostingClassifier(CClassifier):
    name = 'Gradient Boosting'

    param1 = ['deviance', 'exponential']  # loss
    param1_name = 'loss'
    param2 = np.random.uniform(EPSILON, 1)  # learning_rate
    param2_name = 'learning_rate'
    param3 = sp_randint(1, 10000)  # n_estimators
    param3_name = 'n_estimators'
    param4 = sp_randint(1, 5)  # max_depth
    param4_name = 'max_depth'
    param5 = np.random.uniform(EPSILON, 1)  # min_samples_split
    param5_name = 'min_samples_split'
    param6 = sp_randint(1, 5)  # min_samples_leaf
    param6_name = 'min_samples_leaf'
    param7 = sp_randint(len(f_factory.feature_names) - 5, len(f_factory.feature_names))  # max_features
    param7_name = 'max_features'
    param8 = np.random.uniform(EPSILON, 1)  # subsample
    param8_name = 'subsample'
    tuned_params = {'loss': param1, 'learning_rate': param2,
                    'n_estimators': param3, 'max_depth': param4,
                    'min_samples_split': param5, 'min_samples_leaf': param6,
                    'max_features': param7, 'subsample': param8
                    }

    clf = GradientBoostingClassifier()

    def __init__(self, X, y):
        CClassifier.__init__(self, X, y)


class CDecisionTreeClassifier(CClassifier):
    name = 'Decision Tree'

    param1 = ['gini', 'entropy']  # criterion
    param1_name = 'criterion'
    param2 = ['best', 'random']  # splitter
    param2_name = 'splitter'
    param3 = sp_randint(1, 5)  # max_depth
    param3_name = 'max_depth'
    param4 = np.random.uniform(EPSILON, 1)  # min_samples_split
    param4_name = 'min_samples_split'
    param5 = sp_randint(1, 5)  # min_samples_leaf
    param5_name = 'min_samples_leaf'
    param6 = len(f_factory.feature_names) - 5, len(f_factory.feature_names)  # max_features
    param6_name = 'max_features'
    tuned_params = {'criterion': param1, 'splitter': param2,
                    'max_depth': param3, 'min_samples_split': param4,
                    'min_samples_leaf': param5, 'max_features': param6,
                    }

    clf = DecisionTreeClassifier()

    def __init__(self, X, y):
        CClassifier.__init__(self, X, y)


class CRandomForest(CClassifier):
    name = 'Random Forest'

    param1 = [3, None]  # max_depth
    param1_name = 'max_depth'
    param2 = len(f_factory.feature_names) - 5, len(f_factory.feature_names)  # max_features
    param2_name = 'max_features'
    param3 = np.random.uniform(EPSILON, 1)  # min_samples_split
    param3_name = 'min_samples_split'
    param4 = sp_randint(1, 11)  # min_samples_leaf
    param4_name = 'min_samples_leaf'
    tuned_params = {'max_depth': param1, 'max_features': param2,
                    'min_samples_split': param3, 'min_samples_leaf': param4}

    clf = RandomForestClassifier()

    def __init__(self, X, y):
        CClassifier.__init__(self, X, y)


class CAdaBoost(CClassifier):

    name = 'Ada Boost'

    param1 = sp_randint(1, 200)  # n_estimators
    param1_name = 'n_estimators'
    param2 = np.random.uniform(EPSILON, 1)  # learning_rate
    param2_name = 'learning_rate'
    param3 = ['SAMME', 'SAMME.R']  # algorithm
    param3_name = 'algorithm'

    tuned_params = {'n_estimators': param1, 'learning_rate': param2,
                    'algorithm': param3,
                    }

    clf = AdaBoostClassifier()

    def __init__(self, X, y):
        CClassifier.__init__(self, X, y)


class CNaiveBayes(CClassifier):
    name = 'Naive Bayes'

    tuned_params = {}

    clf = GaussianNB()

    def __init__(self, X, y):
        CClassifier.__init__(self, X, y)

