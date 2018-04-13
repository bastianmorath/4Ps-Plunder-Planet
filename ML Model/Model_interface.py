'''Defines the interface that all the models have to implement
'''

from abc import ABC, abstractmethod
from sklearn import metrics
import itertools

import numpy as np
import globals as gl


class AbstractMLModelClass(ABC):

    @abstractmethod
    def predict(self, x_test):
        pass

    @staticmethod
    def print_score(y_true, y_predicted):
        print('Confusion matrix: \n' + str(metrics.confusion_matrix(y_true, y_predicted)))

        null_accuracy = max(np.mean(y_true), 1 - np.mean(y_true)) * 100
        predicted_accuracy = metrics.accuracy_score(y_true, y_predicted) * 100

        print('Null accuracy: ' + str(null_accuracy) + '%')
        print('Correctly classified data: ' + str(predicted_accuracy) + '%')

        print("Number of mislabeled points out of a total %d points : %d"
              % (sum([len(df.index) for df in gl.obstacle_df_list]), (y_true != y_predicted).sum()))

        print(str(predicted_accuracy - null_accuracy) + '%: Difference in % correctly classified data '
                                                        'compared to Null Accuracy: ')
