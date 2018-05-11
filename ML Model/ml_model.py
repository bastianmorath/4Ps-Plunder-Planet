"""This module mainly serves as a factory for all machine learning modules

"""
from __future__ import division  # s.t. division uses float result
import matplotlib

# matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (auc, confusion_matrix, roc_curve)
from sklearn.model_selection import (cross_val_predict, train_test_split)

from sklearn.utils import class_weight
from sklearn.calibration import CalibratedClassifierCV

import features_factory as f_factory
import globals as gl
import plots
import classifiers


def get_performance(model, clf_name, X, y, verbose = False):
    """
        Computes performance of the model by doing cross validation with 10 folds, using
        cross_val_predict, and returns roc_auc, recall, specificity, precision and confusion matrix

      :param model: the classifier that should be applied
      :param clf_name: Name of the classifier (used to print scores)
      :param X: the feature matrix
      :param y: True labels

      :return: roc_auc, recall, specificity, precicion and confusion_matrix

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


def feature_selection(X, y, verbose=False):
    """Feature Selection with ExtraTreesClassifier. Prints and plots the importance of the features


    Source: http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html

    :param X:  Feature matrix
    :param y: labels

    :return new feature matrix with selected features

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
    if verbose:
        print("Feature ranking:")
        print('\n# features after feature-selection: ' + str(X_new.shape[1]))
    x_ticks = []
    for f in range(X.shape[1]):
        x_ticks.append(f_factory.feature_names[indices[f]])
        if verbose:
            print("%d. feature %s (%.3f)" % (f + 1, f_factory.feature_names[indices[f]], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), x_ticks, rotation='vertical')
    plt.xlim([-1, X.shape[1]])
    plt.tight_layout()

    plots.save_plot(plt, 'Features/', 'feature_importance.pdf')

    return X_new, y


def plot_roc_curve(classifier, X, y, classifier_name):
    """Plots roc_curve for a given classifier

    :param classifier:  Classifier
    :param X: Feature matrix
    :param y: labels
    :param classifier_name: Name of the classifier

    """

    # allows to add probability output to classifiers which implement decision_function()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    classifier.fit(X_train, y_train)
    clf = CalibratedClassifierCV(classifier)
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

    filename = 'roc_curve_'+classifier_name+'.pdf'
    plots.save_plot(plt, 'Performance/Roc Curves', filename)


def print_confidentiality_scores(X_train, X_test, y_train, y_test):
    """Prints all wrongly classifed datapoints and with which confidentiality the classifier classified them

    :param X_train: Training data (Feature matrix)
    :param X_test:  Test data (Feature matrix)
    :param y_train: labels of training data
    :param y_test:  labels of test data

    """

    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    probas = model.predict_proba(X_test)
    y_predicted = model.predict(X_test)
    for idx, [a, b] in enumerate(probas):
        if y_test[idx] != y_predicted[idx]:
            print('True/Predicted: (' + str(y_test[idx]) + ', ' + str(y_predicted[idx]) + '), Confidentiality: '
                  + str(max(a,b)*100) + '%')


def plot_performance_of_classifiers(X, y):
    # Plots performance of the given classifiers in a barchart for comparison without hyperparameter tuning

    clf_list = [
        classifiers.CLinearSVM(X, y),
        classifiers.CSVM(X, y),
        classifiers.CNearestNeighbors(X, y),
        classifiers.CNaiveBayes(X, y),
        classifiers.CQuadraticDiscriminantAnalysis(X, y),
    ]

    auc_scores = []
    auc_std = []
    for classifier in clf_list:
        print('Name: ' + classifier.name)
        # If NaiveBayes classifier is used, then use Boxcox since features must be gaussian distributed
        if classifier.name == 'Naive Bayes':
            old_bx = gl.use_boxcox
            gl.use_boxcox = True
            X, _ = f_factory.get_feature_matrix_and_label()
            gl.use_boxcox = old_bx

            classifier.clf.fit(X, y)
        auc_scores.append(get_performance(classifier.clf, classifier.name, X, y)[0])

        plot_roc_curve(classifier.clf, X, y, classifier.name)

    # Plots roc_auc for the different classifiers
    plots.plot_barchart(title='roc_auc w/out hyperparameter tuning',
                              x_axis_name='',
                              y_axis_name='roc_auc',
                              x_labels=[clf.name for clf in clf_list],
                              values=auc_scores,
                              filename='roc_auc_per_classifier.pdf',
                              lbl=None,
                              )


