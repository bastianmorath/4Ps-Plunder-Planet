
from __future__ import division  # s.t. division uses float result

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
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
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

    y_pred = cross_val_predict(model, X, y, cv=10)
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


def apply_cv_per_user_model(model, clf_name, X, y, per_logfile=False, verbose=False):
    """
        Take one entire user (i.e. two logfiles most of the time) as test-data, the rest as training-data. Result can be used
        as an indication which users are hard to predict

    :param model: the model that should be applied
    :param clf_name: String name of the classifier
    :param X: the feature matrix
    :param y: True labels
    :param per_logfile: Whether we should leave out one logfile or one user in cross validation
    :return: auc, recall, specificity and precision average
    """

    y = np.asarray(y)  # Used for .split() function

    # Each file should be a separate group, s.t. we can always leaveout one logfile (NOT one user)
    if per_logfile:
        groups_ids = (pd.concat(gl.obstacle_df_list)['userID'].map(str) + pd.concat(gl.obstacle_df_list)['logID'].map(str)).tolist()
    # Each user should be a separate group, s.t. we can always leaveout one user (NOT one logfile)
    else:
        groups_ids = pd.concat(gl.obstacle_df_list)['userID'].map(str).tolist()

    logo = LeaveOneGroupOut()
    scores_and_ids = []
    df_obstacles_concatenated = pd.concat(gl.obstacle_df_list, ignore_index=True)
    for train_index, test_index in logo.split(X, y, groups_ids):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        conf_mat = confusion_matrix(y_test, y_pred)

        recall = metrics.recall_score(y_test, y_pred)
        specificity = conf_mat[0, 0] / (conf_mat[0, 0] + conf_mat[0, 1])
        precision = metrics.precision_score(y_test, y_pred)
        auc = metrics.roc_auc_score(y_test, y_pred)
        # I calculate the indices that were left out, and map them back to one row of the data,
        # then taking its userid and logID
        left_out_group_indices = sorted(set(range(0, len(df_obstacles_concatenated))) - set(train_index))
        group = df_obstacles_concatenated.loc[[left_out_group_indices[0]]][['userID', 'logID']]

        if per_logfile:
            user_id, log_id = group['userID'].item(), group['logID'].item()
            scores_and_ids.append((auc, recall, specificity, precision, user_id, log_id))
        else:
            user_id = group['userID'].item()
            scores_and_ids.append((auc, recall, specificity, precision, user_id, -1))

    names = []
    # Print scores for each individual logfile/user left out in cross_validation
    for _, (auc, rec, spec, prec, user_id, log_id) in enumerate(scores_and_ids):
        for df_idx, df in enumerate(gl.df_list):
            # Get the logname out of userID and log_id (1 or 2)
            if per_logfile and df.iloc[0]['userID'] == user_id and df.iloc[0]['logID'] == log_id:
                name = gl.names_logfiles[df_idx][:2] + '_' + gl.names_logfiles[df_idx][-5]
                names.append(name)
                if verbose:
                    print(name + ':\t\t Auc= %.3f, Recall = %.3f, Specificity= %.3f,'
                                 'Precision = %.3f' % (auc, rec, spec, prec))
            # Get the username out of userID
            elif (not per_logfile) and df.iloc[0]['userID'] == user_id:
                name = gl.names_logfiles[df_idx][:2]
                names.append(name)
                if verbose:
                    print(name + ':\t\t Auc= %.3f, Recall = %.3f, Specificity= %.3f, '
                                 'Precision = %.3f' % (auc, rec, spec, prec))

    names = list(dict.fromkeys(names))  # Filter duplicates while preserving order

    index = np.arange(len(names))
    aucs = [a[0] for a in scores_and_ids]
    recalls = [a[1] for a in scores_and_ids]
    specificities = [a[2] for a in scores_and_ids]
    precisions = [a[3] for a in scores_and_ids]

    _, ax = plt.subplots()
    bar_width = 0.3
    opacity = 0.4
    r = plt.bar(index, aucs, bar_width,
                    alpha=opacity,
                    color=plots.blue_color,
                    label='Auc')

    if per_logfile:
        plt.xlabel('Logfile')
        plt.title('Auc scores by logfile with ' + clf_name)
    else:
        plt.xlabel('User')
        plt.title('Auc scores by user with ' + clf_name)

    plt.ylabel('Auc score')
    plt.xticks(index, names, rotation='vertical')
    plt.legend()

    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for i, rect in enumerate(rects):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 1.02 * height,
                    '%0.3f' % aucs[i],
                    ha='center', va='bottom', size=5)

    autolabel(r)

    plt.tight_layout()
    if per_logfile:
        plt.savefig(gl.working_directory_path + '/Plots/performance_per_logfile_' + clf_name+'.pdf')
    else:
        plt.savefig(gl.working_directory_path + '/Plots/performance_per_user_' + clf_name+'.pdf')

    # Print mean score
    auc_mean = np.mean([a[0] for a in scores_and_ids])
    recall_mean = np.mean([a[1] for a in scores_and_ids])
    specificity_mean = np.mean([a[2] for a in scores_and_ids])
    precision_mean = np.mean([a[3] for a in scores_and_ids])

    auc_std = np.std([a[0] for a in scores_and_ids])
    recall_std = np.std([a[1] for a in scores_and_ids])
    specificity_std = np.std([a[2] for a in scores_and_ids])
    precision_std = np.std([a[3] for a in scores_and_ids])

    auc_max = np.max([a[0] for a in scores_and_ids])
    recall_max = np.max([a[1] for a in scores_and_ids])
    specificity_max = np.max([a[2] for a in scores_and_ids])
    precision_max = np.max([a[3] for a in scores_and_ids])

    if verbose:
        print('Performance score when doing LeaveOneGroupOut with logfiles: ')

        print('\n Performance: '
              'Auc = %.3f (+-%.3f, max: %.3f), '
              'Recall = %.3f (+-%.3f, max: %.3f), '
              'Specificity= %.3f (+-%.3f, max: %.3f), '
              'Precision = %.3f (+-%.3f, max: %.3f)'
            %   (auc_mean, auc_std, auc_max,
                recall_mean, recall_std, recall_max,
                specificity_mean, specificity_std, specificity_max,
                precision_mean, precision_std, precision_max))

        y_pred = cross_val_predict(model, X, y, cv=logo.split(X, y, groups_ids))
        conf_mat = confusion_matrix(y, y_pred)
        print('\t Confusion matrix: \n\t\t' + str(conf_mat).replace('\n', '\n\t\t'))

        predicted_accuracy = round(metrics.accuracy_score(y, y_pred) * 100, 2)
        null_accuracy = round(max(np.mean(y), 1 - np.mean(y)) * 100, 2)

        print('\tCorrectly classified data: ' + str(predicted_accuracy) + '% (vs. null accuracy: ' + str
        (null_accuracy) + '%)')

    return auc_mean, recall_mean, specificity_mean, precision_mean


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
    predicted_probas = clf.predict_proba(X_test) # returns class probabilities for each class

    fpr, tpr, _ = roc_curve(y_test, predicted_probas[:,1])
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
    plt.savefig(gl.working_directory_path + '/Plots/roc_curve_'+classifier_name+'.pdf')
