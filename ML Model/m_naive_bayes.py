#     Naives bayes as the binary classifier and 10-fold Cross-Validation is used


from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB


from Model_interface import AbstractMLModelClass

# TODO Maybe use optunity instead of GridCVSearch(look old git files)


class NaiveBayes(AbstractMLModelClass):

    def __init__(self, X_train, y_train):
        # self.plot_mse_vs_num_neigbors(X_train, y_train)

        self.model = GaussianNB()
        # find best hyperparameters
        # self.model = GridSearchCV(self.model, {'n_neighbors': range(1, 200)}, cv=5, scoring='accuracy')
        self.model.fit(X_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)

