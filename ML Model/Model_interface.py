'''Defines the interface that all the models have to implement
'''

from abc import ABC, abstractmethod
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split  # IMPORTANT: use sklearn.cross_val for of Euler

import numpy as np
import globals as gl

from sklearn.base import BaseEstimator
class AbstractMLModelClass(ABC, BaseEstimator):

    @abstractmethod
    def predict(self, x_test):
        pass

    @staticmethod
    def print_score(model, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3)

        y_pred = cross_val_predict(model, X, y, cv=10)
        conf_mat = confusion_matrix(y, y_pred)

        print('Confusion matrix: \n' + str(conf_mat))

        null_accuracy = max(np.mean(y_test), 1 - np.mean(y_test)) * 100
        predicted_accuracy = metrics.accuracy_score(y_test, y_pred) * 100

        print('Null accuracy: ' + str(null_accuracy) + '%')
        print('Correctly classified data: ' + str(predicted_accuracy) + '%')

        print("Number of mislabeled points out of a total %d points : %d"
              % (sum([len(df.index) for df in gl.obstacle_df_list]), (y_test != y_pred).sum()))

        print(str(predicted_accuracy - null_accuracy) + '%: Difference in % correctly classified data '
                                                        'compared to Null Accuracy: ')

        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        print('Score: ' + str(scores))

