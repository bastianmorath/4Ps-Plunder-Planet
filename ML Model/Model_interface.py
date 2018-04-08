'''Defines the interface that all the models have to implement
'''

from abc import ABC, abstractmethod
from sklearn import metrics

import numpy as np
import globals as gl

class AbstractMLModelClass(ABC):

    @abstractmethod
    def predict(self, x_test):
        pass

    @staticmethod
    def print_score(y_true, y_predicted):
        print('Confusion matrix: \n' + str(metrics.confusion_matrix(y_true, y_predicted)))

        print('Null accuracy: ' + str(max(np.mean(y_true), 1 - np.mean(y_true)) * 100) + '%')
        percentage = metrics.accuracy_score(y_true, y_predicted)
        print('Correctly classified data: ' + str(percentage*100) + '%')

        print("Number of mislabeled points out of a total %d points : %d"
              % (len(gl.obstacle_df), (y_true != y_predicted).sum()))
