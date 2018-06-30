"""This module contains different classifier classes. They are all subclasses of the Classifier-class and
contain hyperparameters to do grid search over and the classifier obejct itself

"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from scipy.stats import randint as sp_randint

from sklearn.naive_bayes import GaussianNB
from scipy.stats import uniform
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import (AdaBoostClassifier,
                              GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.tree import DecisionTreeClassifier

import features_factory as f_factory


EPSILON = 0.0001

# names = ['SVM', 'Linear SVM', 'Nearest Neighbor', 'QDA', 'Gradient Boosting', 'Decision Tree', 'Random Forest', 'Ada Boost', 'Naive Bayes']
names = ['SVM', 'Linear SVM', 'Nearest Neighbor', 'QDA', 'Decision Tree', 'Random Forest', 'Ada Boost', 'Naive Bayes']

length_param_list = 1000
# Note: Whenever there is 'auto' or None as an option, I add it 'length_param_list/3' times to the hyperparameter list,
# such that the likelhoode of drawing this is very high (because the default parameters often give good performance)


def get_cclassifier_with_name(clf_name, X, y):
    """Returns the CClassifier with the given name

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


def get_list_with_distr_and_opt_param(distr, param):
    """

    :param dist:
    :param param:
    :return:
    """
    return ([param] * int(length_param_list/2)) + list(distr.rvs(size=length_param_list))


class CClassifier(object):
    def __init__(self, X, y):

        self.X = X
        self.y = y


class CSVM(CClassifier):
    name = 'SVM'
    param1 = sp_randint(1, 100)  # C
    param1_name = 'C'
    param2 = get_list_with_distr_and_opt_param(uniform(EPSILON, 10), 'auto')  # gamma
    param2_name = 'gamma'
    tuned_params = {'C': param1, 'gamma': param2}

    def __init__(self, X, y):
        CClassifier.__init__(self, X, y)
        self.clf = SVC(class_weight='balanced')


class CLinearSVM(CClassifier):
    name = 'Linear SVM'
    param1 = sp_randint(1, 10000)  # C
    param1_name = 'C'
    param2 = ['l1', 'l2']  # penalty
    param2_name = 'penalty'
    tuned_params = {'C': param1, 'penalty': param2}

    def __init__(self, X, y):
        CClassifier.__init__(self, X, y)
        self.clf = LinearSVC(class_weight='balanced', dual=False)


class CNearestNeighbors(CClassifier):
    name = 'Nearest Neighbor'

    param1 = sp_randint(1, 1000)  # n_neighbors
    param1_name = 'n_neighbors'
    param2 = ['minkowski', 'euclidean', 'manhattan']  # metric
    param2_name = 'metric'
    tuned_params = {'n_neighbors': param1, 'metric': param2}

    def __init__(self, X, y):
        CClassifier.__init__(self, X, y)
        self.clf = KNeighborsClassifier()


class CQuadraticDiscriminantAnalysis(CClassifier):
    name = 'QDA'

    param1 = uniform(0, 1)  # reg_param
    param1_name = 'reg_param'
    param2 = uniform(EPSILON, 0.1)  # tol
    param2_name = 'tol'
    tuned_params = {'reg_param': param1, 'tol': param2}  # TODO redo

    def __init__(self, X, y):
        CClassifier.__init__(self, X, y)
        self.clf = QuadraticDiscriminantAnalysis()


class CGradientBoostingClassifier(CClassifier):
    name = 'Gradient Boosting'

    param1 = ['deviance', 'exponential']  # loss
    param1_name = 'loss'
    param2 = uniform(EPSILON, 1)  # learning_rate
    param2_name = 'learning_rate'
    param3 = sp_randint(1, 10000)  # n_estimators
    param3_name = 'n_estimators'
    param4 = sp_randint(1, 100)  # max_depth
    param4_name = 'max_depth'
    param5 = sp_randint(2, 20)  # min_samples_split
    param5_name = 'min_samples_split'
    param6 = sp_randint(1, 5)  # min_samples_leaf
    param6_name = 'min_samples_leaf'

    param8 = uniform(EPSILON, 1)  # subsample
    param8_name = 'subsample'

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

    param1 = ['gini', 'entropy']  # criterion
    param1_name = 'criterion'
    param2 = ['best', 'random']  # splitter
    param2_name = 'splitter'
    param3 = get_list_with_distr_and_opt_param(sp_randint(1, 50), None)   # max_depth
    param3_name = 'max_depth'
    param4 = sp_randint(2, 20)  # min_samples_split
    param4_name = 'min_samples_split'
    param5 = sp_randint(1, 20)  # min_samples_leaf
    param5_name = 'min_samples_leaf'

    def __init__(self, X, y):
        CClassifier.__init__(self, X, y)
        # Compute this after instantiations since feature_names are [] otherwise
        # self.param6 = get_list_with_distr_and_opt_param(sp_randint(len(f_factory.feature_names) - 5, len(f_factory.feature_names)),
        #                                                 None)   # max_features
        self.param6_name = 'max_features'
        self.tuned_params = {'criterion': CDecisionTreeClassifier.param1, 'splitter': CDecisionTreeClassifier.param2,
                             'max_depth': CDecisionTreeClassifier.param3, 'min_samples_split': CDecisionTreeClassifier.param4,
                             'min_samples_leaf': CDecisionTreeClassifier.param5,
                             }

    clf = DecisionTreeClassifier(class_weight='balanced')


class CRandomForest(CClassifier):
    name = 'Random Forest'

    param1 = get_list_with_distr_and_opt_param(sp_randint(1, 30), None)  # max_depth
    param1_name = 'max_depth'

    param3 = uniform(EPSILON, 1)  # min_samples_split
    param3_name = 'min_samples_split'
    param4 = sp_randint(1, 11)  # min_samples_leaf
    param4_name = 'min_samples_leaf'

    def __init__(self, X, y):
        CClassifier.__init__(self, X, y)
        self.clf = RandomForestClassifier()

        # self.param2 = get_list_with_distr_and_opt_param(sp_randint(len(f_factory.feature_names) - 5, len(f_factory.feature_names)),
        #                                                 'auto')  # max_features
        self.param2_name = 'max_features'
        self.tuned_params = {'max_depth': CRandomForest.param1,
                             'min_samples_split': CRandomForest.param3, 'min_samples_leaf': CRandomForest.param4}


class CAdaBoost(CClassifier):

    name = 'Ada Boost'

    param1 = sp_randint(1, 200)  # n_estimators
    param1_name = 'n_estimators'
    param2 = uniform(EPSILON, 1)  # learning_rate
    param2_name = 'learning_rate'
    param3 = ['SAMME', 'SAMME.R']  # algorithm
    param3_name = 'algorithm'

    tuned_params = {'n_estimators': param1, 'learning_rate': param2,
                    'algorithm': param3,
                    }

    def __init__(self, X, y):
        CClassifier.__init__(self, X, y)
        self.clf = AdaBoostClassifier()


class CNaiveBayes(CClassifier):
    name = 'Naive Bayes'

    tuned_params = {}

    def __init__(self, X, y):
        CClassifier.__init__(self, X, y)
        self.clf = GaussianNB()


