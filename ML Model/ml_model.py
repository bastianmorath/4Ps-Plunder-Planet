
from __future__ import division  # s.t. division uses float result

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_predict, LeaveOneGroupOut, train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import class_weight

from sklearn.svm import SVC

import globals as gl
import features_factory as f_factory


def apply_cv_model(model, X, y):
    """
        Build feature matrix, then do cross-validation over the entire data

      :param model: the model that should be applied
      :param X: the feature matrix
      :param y: True labels


      """
    y_pred = cross_val_predict(model, X, y, cv=5)
    conf_mat = confusion_matrix(y, y_pred)

    precision = metrics.precision_score(y, y_pred)
    recall = metrics.recall_score(y, y_pred)
    specificity = conf_mat[0, 0] /(conf_mat[0, 0 ] +conf_mat[0, 1])
    print('\n\t Recall = %0.3f = Probability of, given a crash, a crash is correctly predicted; '
          '\n\t Specificity = %0.3f = Probability of, given no crash, no crash is correctly predicted;'
          '\n\t Precision = %.3f = Probability that, given a crash is predicted, a crash really happened; \n'
          % (recall, specificity, precision))

    print('\t Confusion matrix: \n\t\t' + str(conf_mat).replace('\n', '\n\t\t'))

    predicted_accuracy = round(metrics.accuracy_score(y, y_pred) * 100, 2)
    null_accuracy = round(max(np.mean(y), 1 - np.mean(y)) * 100, 2)

    print('\tCorrectly classified data: ' + str(predicted_accuracy) + '% (vs. null accuracy: ' + str
        (null_accuracy) + '%)')


def apply_cv_groups_model(model, X, y):
    """
        Take one entire logfile as test-data, the rest as training-data. Result can be used
        as an indication which files are hard to predict

    :param model: the model that should be applied
    :param X: the feature matrix
    :param y: True labels
    :return:
    """
    y = np.asarray(y)  # Used for .split() function

    # Each file should be a separate group, s.t. we can always leaveout one logfile (NOT one user)
    groups_ids = (pd.concat(gl.obstacle_df_list)['userID'].map(str) + pd.concat(gl.obstacle_df_list)['logID'].map(str)).tolist()
    logo = LeaveOneGroupOut()
    scores_and_logid = []
    df_obstacles_concatenated = pd.concat(gl.obstacle_df_list, ignore_index=True)
    for train_index, test_index in logo.split(X, y, groups_ids, ):
        # I calculate the indices that were left out, and map them back to one row of the data,
        # then taking its userid and logID
        left_out_group_indices = sorted(set(range(0, len(df_obstacles_concatenated))) - set(train_index))
        group = df_obstacles_concatenated.loc[[left_out_group_indices[0]]][['userID', 'logID']]
        user_id, log_id = group['userID'].item(), group['logID'].item()
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        conf_mat = confusion_matrix(y_test, y_pred)

        recall = metrics.recall_score(y_test, y_pred)
        specificity = conf_mat[0, 0] / (conf_mat[0, 0] + conf_mat[0, 1])
        precision = metrics.precision_score(y_test, y_pred)

        scores_and_logid.append((recall, specificity, precision, user_id, log_id))

    lognames = []
    # Print scores for each individual logfile left out in cross_validation
    for idx, (rec, spec, prec, user_id, log_id) in enumerate(scores_and_logid):
        # Get the logname out of userID and log_id (1 or 2)
        for df_idx, df in enumerate(gl.df_list):
            if df.iloc[0]['userID'] == user_id and df.iloc[0]['logID'] == log_id:
                logname = gl.names_logfiles[df_idx][:2] + '_' + gl.names_logfiles[df_idx][-5]
                lognames.append(logname)

        print(logname + ':\t\t Recall = %.3f, Specificity= %.3f,'
                                        'Precision = %.3f' % (rec, spec, prec))

    index = np.arange(len(gl.names_logfiles))
    recalls = [a for (a, _, _, _, _) in scores_and_logid]
    specificities = [b for (_, b, _, _, _) in scores_and_logid]
    precisions = [c for (_, _, c, _, _) in scores_and_logid]
    sum_per_logfile = [a+b+c for (a,b,c) in zip(recalls, specificities, precisions)]
    print('Sum of recall, specificity and precision per logfile: ' + str(sum_per_logfile))

    plt.subplots()
    bar_width = 0.3
    opacity = 0.4
    plt.bar(index, recalls, bar_width,
            alpha=opacity,
            color='r',
            label='Recall')

    plt.bar(index + bar_width, specificities, bar_width,
            alpha=opacity,
            color='b',
            label='Specificity')

    plt.bar(index + 2*bar_width, precisions, bar_width,
            alpha=opacity,
            color='g',
            label='Precision')
    plt.xlabel('Logfile')
    plt.ylabel('Performance')
    plt.title('Scores by logfile with SVM')
    plt.xticks(index + 3*bar_width / 2, lognames, rotation='vertical')
    plt.legend()

    plt.tight_layout()
    plt.savefig(gl.working_directory_path + '/Plots/performace_per_logfile.pdf')

    print('Performance score when doing LeaveOneGroupOut with logfiles: ')
    # Print mean score
    recall_mean = np.mean([a for (a, _, _, _, _) in scores_and_logid])
    specificity_mean = np.mean([b for (_, b,_, _, _) in scores_and_logid])
    precision_mean = np.mean([c for (_, _, c, _, _) in scores_and_logid])

    recall_std = np.std([a for (a, _, _, _, _) in scores_and_logid])
    specificity_std = np.std([b for (_, b, _, _, _) in scores_and_logid])
    precision_std = np.std([c for (_, _, c, _, _) in scores_and_logid])

    recall_max = np.max([a for (a, _, _, _, _) in scores_and_logid])
    specificity_max = np.max([b for (_, b, _, _, _) in scores_and_logid])
    precision_max = np.max([c for (_, _, c, _, _) in scores_and_logid])


    print('\n Performance:' + ': Recall = %.3f (+-%.3f, max: %.3f), Specificity= %.3f (+-%.3f, max: %.3f),'
                          ' Precision = %.3f (+-%.3f, max: %.3f)'
          % (recall_mean, recall_std, recall_max,
             specificity_mean, specificity_std, specificity_max,
             precision_mean, precision_std, precision_max))

    y_pred = cross_val_predict(model, X, y, cv=logo.split(X, y, groups_ids))
    conf_mat = confusion_matrix(y, y_pred)
    print('\t Confusion matrix: \n\t\t' + str(conf_mat).replace('\n', '\n\t\t'))

    predicted_accuracy = round(metrics.accuracy_score(y, y_pred) * 100, 2)
    null_accuracy = round(max(np.mean(y), 1 - np.mean(y)) * 100, 2)

    print('\tCorrectly classified data: ' + str(predicted_accuracy) + '% (vs. null accuracy: ' + str
    (null_accuracy) + '%)')


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

    apply_cv_model(clf, X_new, y)


def param_estimation_grid_cv(X, y, classifier, tuned_params):
    """This method optimizes hyperparameters of svm with cross-validation, which is done using GridSearchCV on a
        training set
        The performance of the selected hyper-parameters and trained model is then measured on a dedicated test
        set that was not used during the model selection step.

    :return: classifier with optimal tuned hyperparameters

    """


    # scores = ['precision', 'recall']
    scores = ['precision']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(classifier, tuned_params, cv=5,
                           scoring='%s_macro' % score)
        clf.fit(X, y)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()

        y_pred = cross_val_predict(clf, X, y, cv=5)
        conf_mat = confusion_matrix(y, y_pred)

        print('\t Confusion matrix: \n\t\t' + str(conf_mat).replace('\n', '\n\t\t'))
        print(classification_report(y, y_pred, target_names=['No Crash: ', 'Crash: ']))
        print()

        return clf


def test_classifiers(X, y):

        from sklearn.model_selection import train_test_split

        from sklearn.neural_network import MLPClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC
        from sklearn.gaussian_process import GaussianProcessClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis

        names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
                 "Gradient Boosting", "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
                 "Naive Bayes", "Linear DA", "QDA"]

        classifiers = [
            KNeighborsClassifier(3),
            SVC(kernel="linear"),
            SVC(),
            GaussianProcessClassifier(),
            GradientBoostingClassifier(),
            DecisionTreeClassifier(),
            RandomForestClassifier(),
            MLPClassifier(),
            AdaBoostClassifier(),
            GaussianNB(),
            LinearDiscriminantAnalysis(),
            QuadraticDiscriminantAnalysis(),
            ]

        # iterate over classifiers
        for name, clf in zip(names, classifiers):
            print('\n' + name)

            # clf.fit(X_train, y_train)
            apply_cv_model(clf, X, y)



