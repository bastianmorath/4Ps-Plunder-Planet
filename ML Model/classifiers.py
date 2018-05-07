import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.utils import class_weight
from scipy.stats import uniform, randint as sp_randint

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import (AdaBoostClassifier,
                              GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.tree import DecisionTreeClassifier
import grid_search

EPSILON = 0.0001

class Classifier(object):
    def __init__(self, X, y):
        cw = class_weight.compute_class_weight('balanced', np.unique(y), y)
        self.class_weight_dict = dict(enumerate(cw))
        self.X = X
        self.y = y

    def optimal_clf(self, X, y, num_iter):
        return grid_search.get_optimal_clf(self.clf, X, y, self.tuned_params, num_iter)


class CSVM(Classifier):
    def __init__(self, X, y):
        Classifier.__init__(self, X, y)

        self.name = 'SVM'

        self.param1 = sp_randint(1, 100)  # C
        self.param1_name = 'C'
        self.param2 = sp_randint(EPSILON, 10)  # gamma
        self.param2_name = 'gamma'
        self.tuned_params = {'C': self.param1, 'gamma': self.param2}

        self.clf = SVC(class_weight=self.class_weight_dict)


class CNearestNeighbors(Classifier):
    def __init__(self, X, y):
        Classifier.__init__(self, X, y)

        self.name = 'Nearest Neighbor'

        self.param1 = sp_randint(1, 100)  # n_neighbors
        self.param1_name = 'n_neighbors'
        self.param2 = ['minkowski', 'euclidean', 'manhattan']  # metric
        self.param2_name = 'metric'

        self.tuned_params = {'n_neighbors': self.param1, 'metric': self.param2}

        self.clf = KNeighborsClassifier()


class CQuadraticDiscriminantAnalysis(Classifier):
    def __init__(self, X, y):
        Classifier.__init__(self, X, y)

        self.name = 'QDA'

        self.param1 = sp_randint(0, 1)  # reg_param
        self.param1_name = 'reg_param'
        self.param2 = sp_randint(0, 1)  # tol
        self.param2_name = 'tol'

        self.tuned_params = {'reg_param': self.param1, 'tol': self.param2}

        self.clf = QuadraticDiscriminantAnalysis()


class CGradientBoostingClassifier(Classifier):
    def __init__(self, X, y):
        Classifier.__init__(self, X, y)

        self.name = 'Gradient Boosting'

        self.param1 = ['deviance', 'exponential']  # loss
        self.param1_name = 'loss'
        self.param2 = uniform(EPSILON, 1)  # learning_rate
        self.param2_name = 'learning_rate'
        self.param3 = sp_randint(1, 10000)  # n_estimators
        self.param3_name = 'n_estimators'
        self.param4 = sp_randint(1, 5)  # max_depth
        self.param4_name = 'max_depth'
        self.param5 = uniform(EPSILON, 1)  # min_samples_split
        self.param5_name = 'min_samples_split'
        self.param6 = sp_randint(1, 5)  # min_samples_leaf
        self.param6_name = 'min_samples_leaf'
        self.param7 = sp_randint(10, 16)  # max_features
        self.param7_name = 'max_features'
        self.param8 = uniform(EPSILON, 1)  # subsample
        self.param8_name = 'subsample'
        self.tuned_params = {'loss': self.param1, 'learning_rate': self.param2,
                             'n_estimators': self.param3, 'max_depth': self.param4,
                             'min_samples_split': self.param5, 'min_samples_leaf': self.param6,
                             'max_features': self.param7, 'subsample': self.param8}

        self.clf = GradientBoostingClassifier()


class CDecisionTreeClassifier(Classifier):
    def __init__(self, X, y):
        Classifier.__init__(self, X, y)

        self.name = 'Decision Tree'

        self.param1 = ['gini', 'entropy']  # criterion
        self.param1_name = 'criterion'
        self.param2 = ['best', 'random']  # splitter
        self.param2_name = 'splitter'
        self.param3 = sp_randint(1, 5)  # max_depth
        self.param3_name = 'max_depth'
        self.param4 = uniform(EPSILON, 1)  # min_samples_split
        self.param4_name = 'min_samples_split'
        self.param5 = sp_randint(1, 5)  # min_samples_leaf
        self.param5_name = 'min_samples_leaf'
        self.param6 = sp_randint(10, 16)  # max_features
        self.param6_name = 'max_features'

        self.tuned_params = {'criterion': self.param1, 'splitter': self.param2,
                             'max_depth': self.param3, 'min_samples_split': self.param4,
                             'min_samples_leaf': self.param5, 'max_features': self.param6,
                            }

        self.clf = DecisionTreeClassifier()


class CRandomForest(Classifier):
    def __init__(self, X, y):
        Classifier.__init__(self, X, y)

        self.name = 'Random Forest'

        self.param1 = [3, None]  # max_depth
        self.param1_name = 'max_depth'
        self.param2 = sp_randint(1, 11)  # max_features
        self.param2_name = 'max_features'
        self.param3 = uniform(EPSILON, 1)  # min_samples_split
        self.param3_name = 'min_samples_split'
        self.param4 = sp_randint(1, 11)  # min_samples_leaf
        self.param4_name = 'min_samples_leaf'

        self.tuned_params = {'max_depth': self.param1, 'max_features': self.param2,
                             'min_samples_split': self.param3, 'min_samples_leaf': self.param4}

        self.clf = RandomForestClassifier()


class CAdaBoost(Classifier):
    def __init__(self, X, y):
        Classifier.__init__(self, X, y)

        self.name = 'Descision Tree'

        self.param1 = sp_randint(1, 200)  # n_estimators
        self.param1_name = 'n_estimators'
        self.param2 = uniform(EPSILON, 1)  # learning_rate
        self.param2_name = 'learning_rate'
        self.param3 = ['SAMME', 'SAMME.R']  # algorithm
        self.param3_name = 'algorithm'

        self.tuned_params = {'n_estimators': self.param1, 'learning_rate': self.param2,
                             'algorithm': self.param3,
                            }

        self.clf = AdaBoostClassifier()
