#     SVM as the binary classifier and 10-fold Cross-Validation is used

from sklearn import svm

import optunity
import optunity.metrics

from Model_interface import AbstractMLModelClass

class SVM_Model(AbstractMLModelClass):

    def __init__(self, X_train, y_train):

        # TODO: Other metrics, e.g. precision
        @optunity.cross_validated(x=X_train, y=y_train, num_folds=10, num_iter=1)
        def svm_auc(x_train, y_train, x_test, y_test, log_c, log_gamma):
            model = svm.SVC(C=10 ** log_c, gamma=10 ** log_gamma).fit(x_train, y_train)
            decision_values = model.decision_function(x_test)
            return optunity.metrics.roc_auc(y_test, decision_values)

        # perform tuning
        optimal_rbf_pars, info, _ = optunity.maximize(svm_auc, num_evals=10, log_c=[-20, 0], log_gamma=[-15, 0])

        # train model on the full training set with tuned hyperparameters
        self.model = svm.SVC(C=10 ** optimal_rbf_pars['log_c'], gamma=10 ** optimal_rbf_pars['log_gamma'],
                                class_weight={0: 1, 1: 3}).fit(X_train, y_train)

        self.model.fit(X_train, y_train)

        print("Optimal parameters (10e): " + str(optimal_rbf_pars))
        print("AUROC of tuned SVM with RBF kernel: %1.3f" % info.optimum)


    def predict(self, x_test):
        return self.model.predict(x_test)


