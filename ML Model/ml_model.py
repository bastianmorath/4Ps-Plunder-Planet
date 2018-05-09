
from __future__ import division  # s.t. division uses float result
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (auc, classification_report, confusion_matrix,
                             roc_curve)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV, LeaveOneGroupOut,
                                     cross_val_predict, train_test_split, cross_val_score)

from sklearn.utils import class_weight
from sklearn.calibration import CalibratedClassifierCV

import features_factory as f_factory
import globals as gl
import plots


def get_performance(model, clf_name, X, y, verbose = False):
    """
        Get performance of the model by doing cross validation with 10 folds, using
        cross_val_predict

      :param model: the classifier that should be applied
      :param X: the feature matrix
      :param y: True labels

      :return: roc_auc, recall, specificity and precision

      """

    y_pred = cross_val_predict(model, X, y, cv=5)
    conf_mat = confusion_matrix(y, y_pred)

    precision = metrics.precision_score(y, y_pred)
    recall = metrics.recall_score(y, y_pred)
    specificity = conf_mat[0, 0] / (conf_mat[0, 0] + conf_mat[0, 1])
    auc = metrics.roc_auc_score(y, y_pred)
    
    predicted_accuracy = round(metrics.accuracy_score(y, y_pred) * 100, 2)
    null_accuracy = round(max(np.mean(y), 1 - np.mean(y)) * 100, 2)

    if verbose:
        print('CS scores for ' + clf_name)
        print('\n\t Recall = %0.3f = Probability of, given a crash, a crash is correctly predicted; '
              '\n\t Specificity = %0.3f = Probability of, given no crash, no crash is correctly predicted;'
              '\n\t Precision = %.3f = Probability that, given a crash is predicted, a crash really happened; \n'
              % (recall, specificity, precision))

        print('\t Confusion matrix: \n\t\t' + str(conf_mat).replace('\n', '\n\t\t'))
        print('\tCorrectly classified data: ' + str(predicted_accuracy) + '% (vs. null accuracy: '
              + str(null_accuracy) + '%)')
    return auc, recall, specificity, precision, conf_mat



def feature_selection(X, y):
    """Feature Selection. Prints and plots the importance of the features

    Source: http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html

    :param X:  Feature matrix
    :param y: labels

    """
    cw = class_weight.compute_class_weight('balanced', np.unique(y), y)
    class_weight_dict = dict(enumerate(cw))
    clf = ExtraTreesClassifier(n_estimators=250, class_weight=class_weight_dict)

    forest = clf.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    X_new = SelectFromModel(clf).fit_transform(X, y)

    # Print the feature ranking
    # print("Feature ranking:")
    x_ticks = []
    for f in range(X.shape[1]):
        x_ticks.append(f_factory.feature_names[indices[f]])
        # print("%d. feature %s (%.3f)" % (f + 1, f_factory.feature_names[indices[f]], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), x_ticks, rotation='vertical')
    plt.xlim([-1, X.shape[1]])
    plt.tight_layout()
    # plt.savefig(gl.working_directory_path + '/Plots/feature_importance.pdf')
    print('\n# features after feature-selection: ' + str(X_new.shape[1]))


def plot_roc_curve(classifier, X, y, classifier_name):
   
    # allows to add probability output to LinearSVC or any other classifier which implements decision_function method
    clf = CalibratedClassifierCV(classifier) 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    clf.fit(X_train, y_train)
    predicted_probas = clf.predict_proba(X_test)  # returns class probabilities for each class

    fpr, tpr, _ = roc_curve(y_test, predicted_probas[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(gl.working_directory_path + '/Plots/Roc Curves/roc_curve_'+classifier_name+'.pdf')
