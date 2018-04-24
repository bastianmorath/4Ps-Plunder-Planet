#     SVM as the binary classifier and 10-fold Cross-Validation is used


from sklearn import svm
import numpy as np

from Model_interface import AbstractMLModelClass

# TODO Maybe use optunity instead of GridCVSearch(look old git files)


class SVM(AbstractMLModelClass):

    def __init__(self, X, y):
        param_grid = {
            'gamma': np.linspace(1,10, 10),
            'C': np.linspace(1, 30, 10)
        }

        self.model = svm.SVC(probability=True)
        # self.model = GridSearchCV(self.model, param_grid, cv=10, scoring='accuracy')  # find best hyperparameters
        self.model.fit(X, y)

        # print(self.model.best_params_)

    def predict(self, x_test):
        return self.model.predict(x_test)


