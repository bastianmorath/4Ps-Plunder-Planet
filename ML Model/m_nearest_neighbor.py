#     K-nearest neighbor as the binary classifier and 10-fold Cross-Validation is used


from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

from Model_interface import AbstractMLModelClass

import globals as gl
# TODO Maybe use optunity instead of GridCVSearch(look old git files)


class NearestNeighbor(AbstractMLModelClass):

    def __init__(self, X_train, y_train):
        # subsetting just the odd ones
        neighbors = range(1, 200)
        cv_scores = []
        for k in neighbors:
            knn = KNeighborsClassifier(n_neighbors=k)
            scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
            cv_scores.append(scores.mean())

        # changing to misclassification error
        MSE = [1 - x for x in cv_scores]
        # determining best k

        optimal_k = neighbors[MSE.index(min(MSE))]
        print("The optimal number of neighbors is %d" % optimal_k)

        # plot misclassification error vs k
        plt.figure()
        plt.plot(neighbors, MSE)
        plt.xlabel('Number of Neighbors K')
        plt.ylabel('Misclassification Error')
        plt.savefig(gl.working_directory_path + '/Plots/nearest_neighbor_errors.pdf')
        self.model = KNeighborsClassifier(n_neighbors=optimal_k)
        self.model.fit(X_train, y_train)
        
        """
        self.model = KNeighborsClassifier()
        self.model = GridSearchCV(self.model, param_grid, cv=10, scoring='accuracy')  # find best hyperparameters
        self.model.fit(X_train, y_train)

        print(self.model.best_params_)
        """

    def predict(self, x_test):
        return self.model.predict(x_test)


