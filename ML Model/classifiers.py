"""This module contains different classifier classes. They are all subclasses of the Classifier-class and
contain hyperparameters to do grid search over and the classifier obejct itself

"""


import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.utils import class_weight
from scipy.stats import randint as sp_randint
from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import (AdaBoostClassifier,
                              GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.tree import DecisionTreeClassifier

import features_factory as f_factory


EPSILON = 0.0001


def get_clf_with_name(clf_name):
    """Returns the classifier with the given name

    :param clf_name: name of the classifier

    :return: classifier
    """

    clfs = [
        CSVM,
        CLinearSVM,
        CNearestNeighbors,
        CQuadraticDiscriminantAnalysis,
        CGradientBoostingClassifier,
        CDecisionTreeClassifier,
        CRandomForest,
        CAdaBoost,
        CNaiveBayes
    ]

    return [clf for clf in clfs if clf.name == clf_name][0]


class CClassifier(object):
    def __init__(self, X, y):
        cw = class_weight.compute_class_weight('balanced', np.unique(y), y)
        self.class_weight_dict = dict(enumerate(cw))
        self.X = X
        self.y = y
        self.clf = None
        self.tuned_params = None


class CSVM(CClassifier):
    def __init__(self, X, y):
        CClassifier.__init__(self, X, y)

        self.name = 'SVM'

        self.param1 = sp_randint(1, 100)  # C
        self.param1_name = 'C'
        self.param2 = sp_randint(EPSILON, 10)  # gamma
        self.param2_name = 'gamma'
        self.tuned_params = {'C': self.param1, 'gamma': self.param2}

        self.clf = SVC(class_weight=self.class_weight_dict)


class CLinearSVM(CClassifier):
    def __init__(self, X, y):
        CClassifier.__init__(self, X, y)

        self.name = 'Linear SVM'

        self.param1 = sp_randint(1, 100)  # C
        self.param1_name = 'C'
        self.param2 = ['l1', 'l2']  # loss
        self.param2_name = 'loss'
        self.tuned_params = {'C': self.param1, 'gamma': self.param2}

        self.clf = SVC(class_weight=self.class_weight_dict)


class CNearestNeighbors(CClassifier):
    def __init__(self, X, y):
        CClassifier.__init__(self, X, y)

        self.name = 'Nearest Neighbor'

        self.param1 = sp_randint(1, 100)  # n_neighbors
        self.param1_name = 'n_neighbors'
        self.param2 = ['minkowski', 'euclidean', 'manhattan']  # metric
        self.param2_name = 'metric'

        self.tuned_params = {'n_neighbors': self.param1, 'metric': self.param2}

        self.clf = KNeighborsClassifier()


class CQuadraticDiscriminantAnalysis(CClassifier):
    def __init__(self, X, y):
        CClassifier.__init__(self, X, y)

        self.name = 'QDA'

        self.param1 = sp_randint(0, 1)  # reg_param
        self.param1_name = 'reg_param'
        self.param2 = np.random.uniform(EPSILON, 1)  # tol
        self.param2_name = 'tol'

        self.tuned_params = {'reg_param': self.param1, 'tol': self.param2}

        self.clf = QuadraticDiscriminantAnalysis()


class CGradientBoostingClassifier(CClassifier):
    def __init__(self, X, y):
        CClassifier.__init__(self, X, y)

        self.name = 'Gradient Boosting'

        self.param1 = ['deviance', 'exponential']  # loss
        self.param1_name = 'loss'
        self.param2 = np.random.uniform(EPSILON, 1)  # learning_rate
        self.param2_name = 'learning_rate'
        self.param3 = sp_randint(1, 10000)  # n_estimators
        self.param3_name = 'n_estimators'
        self.param4 = sp_randint(1, 5)  # max_depth
        self.param4_name = 'max_depth'
        self.param5 = np.random.uniform(EPSILON, 1)  # min_samples_split
        self.param5_name = 'min_samples_split'
        self.param6 = sp_randint(1, 5)  # min_samples_leaf
        self.param6_name = 'min_samples_leaf'
        self.param7 = sp_randint(len(f_factory.feature_names) - 5, len(f_factory.feature_names))  # max_features
        self.param7_name = 'max_features'
        self.param8 = np.random.uniform(EPSILON, 1)  # subsample
        self.param8_name = 'subsample'
        self.tuned_params = {'loss': self.param1, 'learning_rate': self.param2,
                             'n_estimators': self.param3, 'max_depth': self.param4,
                             'min_samples_split': self.param5, 'min_samples_leaf': self.param6,
                             'max_features': self.param7, 'subsample': self.param8}

        self.clf = GradientBoostingClassifier()


class CDecisionTreeClassifier(CClassifier):
    def __init__(self, X, y):
        CClassifier.__init__(self, X, y)

        self.name = 'Decision Tree'

        self.param1 = ['gini', 'entropy']  # criterion
        self.param1_name = 'criterion'
        self.param2 = ['best', 'random']  # splitter
        self.param2_name = 'splitter'
        self.param3 = sp_randint(1, 5)  # max_depth
        self.param3_name = 'max_depth'
        self.param4 = np.random.uniform(EPSILON, 1)  # min_samples_split
        self.param4_name = 'min_samples_split'
        self.param5 = sp_randint(1, 5)  # min_samples_leaf
        self.param5_name = 'min_samples_leaf'
        self.param6 = len(f_factory.feature_names) - 5, len(f_factory.feature_names)  # max_features
        self.param6_name = 'max_features'

        self.tuned_params = {'criterion': self.param1, 'splitter': self.param2,
                             'max_depth': self.param3, 'min_samples_split': self.param4,
                             'min_samples_leaf': self.param5, 'max_features': self.param6,
                             }

        self.clf = DecisionTreeClassifier()


class CRandomForest(CClassifier):
    def __init__(self, X, y):
        CClassifier.__init__(self, X, y)

        self.name = 'Random Forest'

        self.param1 = [3, None]  # max_depth
        self.param1_name = 'max_depth'
        self.param2 = len(f_factory.feature_names) - 5, len(f_factory.feature_names)  # max_features
        self.param2_name = 'max_features'
        self.param3 = np.random.uniform(EPSILON, 1)  # min_samples_split
        self.param3_name = 'min_samples_split'
        self.param4 = sp_randint(1, 11)  # min_samples_leaf
        self.param4_name = 'min_samples_leaf'

        self.tuned_params = {'max_depth': self.param1, 'max_features': self.param2,
                             'min_samples_split': self.param3, 'min_samples_leaf': self.param4}

        self.clf = RandomForestClassifier()


class CAdaBoost(CClassifier):
    def __init__(self, X, y):
        CClassifier.__init__(self, X, y)

        self.name = 'Ada Boost'

        self.param1 = sp_randint(1, 200)  # n_estimators
        self.param1_name = 'n_estimators'
        self.param2 = np.random.uniform(EPSILON, 1)  # learning_rate
        self.param2_name = 'learning_rate'
        self.param3 = ['SAMME', 'SAMME.R']  # algorithm
        self.param3_name = 'algorithm'

        self.tuned_params = {'n_estimators': self.param1, 'learning_rate': self.param2,
                             'algorithm': self.param3,
                             }

        self.clf = AdaBoostClassifier()


class CNaiveBayes(CClassifier):
    def __init__(self, X, y):
        CClassifier.__init__(self, X, y)

        self.name = 'Naive Bayes'

        self.tuned_params = {}

        self.clf = GaussianNB()
