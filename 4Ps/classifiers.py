"""
This module contains different classifier classes. They are all subclasses of the Classifier-class and
contain hyperparameters to do grid search over and the classifier obeject itself

Note: estimator_name is needed for RandomizedSearchCV and pipeline!

"""

from scipy.stats import randint as sp_randint
from scipy.stats import uniform
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import features_factory as f_factory

_EPSILON = 0.0001

names = ['SVM', 'Linear SVM', 'Nearest Neighbor', 'QDA', 'Decision Tree', 'Random Forest', 'Ada Boost', 'Naive Bayes']

# RandomizedSearchCV parameter space is defined such that all classifiers should approximately take
# the same amount of time. With 'random_search_multiplier', one can increase or decrease the space linearily.
# multiplier = 1 -> ca. 50 seconds per classifier on i5 MacBook Pro 2017 and Euler cluster
_random_search_multiplier = 1


def get_cclassifier_with_name(clf_name, X, y):
    """Returns the CClassifier with the given name

    :param clf_name: name of the classifier
    :param X: Feature matrix
    :param y: labels

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
    """
    clf = Standard classifier without tuned parameters
    tuned_clf = With RandomizedSearchCV tuned classifier

    """

    def __init__(self, X, y):

        self.X = X
        self.y = y


class CSVM(CClassifier):
    name = 'SVM'
    estimator_name = 'svc'
    param1 = uniform(2**(-5), 2**5)  # C
    param1_name = 'C'
    param2 = uniform(2**(-15), 2**3)  # get_list_with_distr_and_opt_param(uniform(_EPSILON, 10), 'auto')  # gamma
    param2_name = 'gamma'
    param3 = sp_randint(1, 3)  # degree
    param3_name = 'degree'
    param4 = ['rbf', 'sigmoid']  # kernel
    param4_name = 'kernel'
    tuned_params = {'C': param1, 'gamma': param2, 'degree': param3, 'kernel': param4}
    num_iter = 3 * _random_search_multiplier

    def __init__(self, X, y):
        CClassifier.__init__(self, X, y)
        self.clf = SVC(class_weight='balanced', probability=True, cache_size=1500)
        self.tuned_clf = SVC(class_weight='balanced', probability=True, cache_size=1500, C=11.369, gamma=0.0028776, kernel='rbf', degree=2)


class CLinearSVM(CClassifier):
    name = 'Linear SVM'
    estimator_name = 'svc'
    param1 = uniform(2**(-5), 2**5)  # C
    param1_name = 'C'
    param2 = uniform(2**(-15), 2**3)  # get_list_with_distr_and_opt_param(uniform(_EPSILON, 10), 'auto')  # gamma
    param2_name = 'gamma'
    tuned_params = {'C': param1, 'gamma': param2}
    num_iter = 5 * _random_search_multiplier

    def __init__(self, X, y):
        CClassifier.__init__(self, X, y)
        self.clf = SVC(class_weight='balanced', kernel='linear', probability=True)


class CNearestNeighbors(CClassifier):
    name = 'Nearest Neighbor'
    estimator_name = 'kneighborsclassifier'

    param1 = sp_randint(1, 1000)  # n_neighbors
    param1_name = 'n_neighbors'
    param2 = ['minkowski', 'euclidean', 'manhattan']  # metric
    param2_name = 'metric'
    tuned_params = {'n_neighbors': param1, 'metric': param2}
    num_iter = 15 * _random_search_multiplier

    def __init__(self, X, y):
        CClassifier.__init__(self, X, y)
        self.clf = KNeighborsClassifier(weights='distance')
        self.tuned_clf = KNeighborsClassifier(weights='distance', n_neighbors=438)


class CQuadraticDiscriminantAnalysis(CClassifier):
    name = 'QDA'
    estimator_name = 'quadraticdiscriminantanalysis'

    param1 = uniform(0, 1)  # reg_param
    param1_name = 'reg_param'
    param2 = uniform(_EPSILON, 0.4)  # tol
    param2_name = 'tol'
    tuned_params = {'reg_param': param1, 'tol': param2}
    num_iter = 1000 * _random_search_multiplier

    def __init__(self, X, y):
        CClassifier.__init__(self, X, y)
        self.clf = QuadraticDiscriminantAnalysis()


class CGradientBoostingClassifier(CClassifier):
    name = 'Gradient Boosting'
    estimator_name = "gradientboostingclassifier"

    param1 = ['deviance', 'exponential']  # loss
    param1_name = 'loss'
    param2 = uniform(_EPSILON, 1)  # learning_rate
    param2_name = 'learning_rate'
    param3 = sp_randint(1, 10000)  # n_estimators
    param3_name = 'n_estimators'
    param4 = sp_randint(1, 100)  # max_depth
    param4_name = 'max_depth'
    param5 = sp_randint(2, 30)  # min_samples_split
    param5_name = 'min_samples_split'
    param6 = sp_randint(1, 10)  # min_samples_leaf
    param6_name = 'min_samples_leaf'

    param8 = uniform(_EPSILON, 1)  # subsample
    param8_name = 'subsample'

    num_iter = 10 * _random_search_multiplier

    def __init__(self, X, y):
        CClassifier.__init__(self, X, y)
        # Compute this after instantiations since feature_names are [] otherwise
        self.param7 = sp_randint(len(f_factory.feature_names) - 5, len(f_factory.feature_names))  # max_features
        self.param7_name = 'max_features'
        self.tuned_params = {'loss': CGradientBoostingClassifier.param1,
                             'learning_rate': CGradientBoostingClassifier.param2,
                             'n_estimators': CGradientBoostingClassifier.param3,
                             'max_depth': CGradientBoostingClassifier.param4,
                             'min_samples_split': CGradientBoostingClassifier.param5,
                             'min_samples_leaf': CGradientBoostingClassifier.param6,
                             'max_features': self.param7, 'subsample': CGradientBoostingClassifier.param8
                             }

        self.clf = GradientBoostingClassifier()


class CDecisionTreeClassifier(CClassifier):
    name = 'Decision Tree'
    estimator_name = "decisiontreeclassifier"

    param1 = ['gini', 'entropy']  # criterion
    param1_name = 'criterion'
    param2 = ['best', 'random']  # splitter
    param2_name = 'splitter'
    param4 = sp_randint(2, 50)  # min_samples_split
    param4_name = 'min_samples_split'
    param5 = sp_randint(1, 50)  # min_samples_leaf
    param5_name = 'min_samples_leaf'

    num_iter = 500 * _random_search_multiplier

    def __init__(self, X, y):
        CClassifier.__init__(self, X, y)

        self.tuned_params = {'criterion': CDecisionTreeClassifier.param1, 'splitter': CDecisionTreeClassifier.param2,
                             'min_samples_split': CDecisionTreeClassifier.param4,
                             'min_samples_leaf': CDecisionTreeClassifier.param5,
                             }

    clf = DecisionTreeClassifier(class_weight="balanced")


class CRandomForest(CClassifier):
    name = 'Random Forest'
    estimator_name = "randomforestclassifier"

    param3 = sp_randint(1, 50)  # min_samples_leaf
    param3_name = 'min_samples_leaf'
    param4 = sp_randint(1, 128)  # number of trees
    param4_name = 'n_estimators'

    num_iter = 10 * _random_search_multiplier

    def __init__(self, X, y):
        CClassifier.__init__(self, X, y)
        self.clf = RandomForestClassifier()
        self.tuned_clf = RandomForestClassifier(min_samples_leaf=48, n_estimators=13)
        self.tuned_params = {'min_samples_leaf': CRandomForest.param3,
                             'n_estimators': CRandomForest.param4}


class CAdaBoost(CClassifier):

    name = 'Ada Boost'
    estimator_name = "adaboostclassifier"

    param1 = sp_randint(1, 500)  # n_estimators
    param1_name = 'n_estimators'
    param2 = uniform(_EPSILON, 1)  # learning_rate
    param2_name = 'learning_rate'
    param3 = ['SAMME', 'SAMME.R']  # algorithm
    param3_name = 'algorithm'

    tuned_params = {'n_estimators': param1, 'learning_rate': param2,
                    'algorithm': param3,
                    }

    num_iter = 20 * _random_search_multiplier

    def __init__(self, X, y):
        CClassifier.__init__(self, X, y)
        self.clf = AdaBoostClassifier()


class CNaiveBayes(CClassifier):
    name = 'Naive Bayes'

    tuned_params = {}

    def __init__(self, X, y):
        CClassifier.__init__(self, X, y)
        self.clf = GaussianNB()
        self.tuned_clf = GaussianNB()
