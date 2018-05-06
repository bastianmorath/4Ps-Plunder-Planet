import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.utils import class_weight
from scipy import stats
from scipy.stats import randint as sp_randint
import grid_search


class Classifier(object):
    def __init__(self, X, y):
        cw = class_weight.compute_class_weight('balanced', np.unique(y), y)
        self.class_weight_dict = dict(enumerate(cw))
        self.X = X
        self.y = y

    def optimal_clf(self, X, y):
        return grid_search.get_optimal_clf(self.clf, X, y, self.tuned_params)


class SVM(Classifier):
    def __init__(self, X, y):
        Classifier.__init__(self, X, y)

        self.name = 'SVM'

        self.param1 = sp_randint(1, 11)  # C
        self.param1_name = 'C'
        self.param2 = sp_randint(1, 11)  # gamma
        self.param2_name = 'gamma'
        self.tuned_params = [{'C': self.param1, 'gamma': self.param2}]

        self.clf = SVC(class_weight=self.class_weight_dict)


class NearestNeighbors(Classifier):
    def __init__(self, X, y):
        Classifier.__init__(self, X, y)

        self.name = 'Nearest Neighbor'

        self.param1 = stats.expon(scale=100)  # n_neighbors
        self.param1_name = 'n_neighbors'
        self.param2 = ['minkowski', 'euclidean', 'manhattan']  # metric
        self.param2_name = 'metric'

        self.tuned_params = [{'n_neighbors': self.param1, 'metric': self.param2}]

        self.clf = KNeighborsClassifier()


class QuadraticDiscriminantAnalysis(Classifier):
    def __init__(self, X, y):
        Classifier.__init__(self, X, y)

        self.name = 'SVM'

        self.param1 = [1, 10, 100, 1000]  # C
        self.param1_name = 'C'
        self.param2 = [0.001, 0.0001]  # gamma
        self.param2_name = 'gamma'

        self.tuned_params = [{'C': self.param1, 'gamma': self.param2}]


class GradientBoostingClassifier(Classifier):
    def __init__(self, X, y):
        Classifier.__init__(self, X, y)

        self.name = 'SVM'

        self.param1 = [1, 10, 100, 1000]  # C
        self.param1_name = 'C'
        self.param2 = [0.001, 0.0001]  # gamma
        self.param2_name = 'gamma'

        self.tuned_params = [{'C': self.param1, 'gamma': self.param2}]


class DecisionTreeClassifier(Classifier):
    def __init__(self, X, y):
        Classifier.__init__(self, X, y)

        self.name = 'SVM'

        self.param1 = [1, 10, 100, 1000]  # C
        self.param1_name = 'C'
        self.param2 = [0.001, 0.0001]  # gamma
        self.param2_name = 'gamma'

        self.tuned_params = [{'C': self.param1, 'gamma': self.param2}]