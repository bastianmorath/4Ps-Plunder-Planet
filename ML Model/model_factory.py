"""This module mainly serves as a factory with various methods used by the machine learning modules

"""
from __future__ import division  # s.t. division uses float result

# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import random

from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (auc, confusion_matrix, roc_curve)
from sklearn.model_selection import (cross_val_predict, train_test_split, cross_validate)

from sklearn.calibration import CalibratedClassifierCV

import features_factory as f_factory
import setup_dataframes as sd
import plots
import classifiers
import setup_dataframes
import hyperparameter_optimization


def get_tuned_params_dict(model, tuned_params_keys):
    """Given a list of tuned parameter keys and the model, create a dictionary with the tuned parameters and its values
    associated with it.
    Used for printing out scores

    :param model: Model, such that we can extract the parameters from
    :param tuned_params_keys: Parameters that we tuned in RandomSearchCV

    :return: Dictionary with tuned parameters and its values
    """

    values = [model.get_params()[x] for x in tuned_params_keys]
    return dict(zip(tuned_params_keys, values))


def get_performance(model, clf_name, X, y, tuned_params_keys=None, verbose=False, do_write_to_file=False):
    """Computes performance of the model by doing cross validation with 10 folds, using
        cross_val_predict, and returns roc_auc, recall, specificity, precision, confusion matrix and summary of those
        as a string (plus tuned hyperparameters optionally)

    :param model: the classifier that should be applied
    :param clf_name: Name of the classifier (used to print scores)
    :param X: Feature matrix
    :param y: labels
    :param tuned_params_keys: keys of parameters that got tuned (in classifiers.py) (optional)
    :param verbose: Whether a detailed score should be printed out (optional)
    :param do_write_to_file: Write summary of performance into a file (optional)

    :return: roc_auc, recall, specificity, precision, confusion_matrix and summary of those as a string
    """

    print('Calculating performance of %s...' % clf_name)

    sd.obstacle_df_list = setup_dataframes.get_obstacle_times_with_success()

    # Compute performance scores
    y_pred = cross_val_predict(model, X, y, cv=10)

    conf_mat = confusion_matrix(y, y_pred)

    precision = metrics.precision_score(y, y_pred)
    if clf_name == 'Decision Tree':
        plots.plot_graph_of_decision_classifier(model, X, y)

    recall = metrics.recall_score(y, y_pred)
    specificity = conf_mat[0, 0] / (conf_mat[0, 0] + conf_mat[0, 1])
    roc_auc = metrics.roc_auc_score(y, y_pred)

    if tuned_params_keys is None:
        s = create_string_from_scores(clf_name, roc_auc, recall, specificity, precision, conf_mat)
    else:
        tuned_params_dict = get_tuned_params_dict(model, tuned_params_keys)
        s = create_string_from_scores(clf_name, roc_auc, recall, specificity, precision, conf_mat, tuned_params_dict)

    if verbose:
        print(s)

    if do_write_to_file:
        # Write result to a file
        filename = 'performance_' + clf_name + '_windows_' + str(f_factory.hw) + '_' + str(f_factory.cw) + '_' + \
                    str(f_factory.gradient_w) + '.txt'
        write_to_file(s, 'Performance/', filename, 'w+')

    return roc_auc, recall, specificity, precision, conf_mat, s


def feature_selection(X, y, verbose=False):
    """Feature Selection with ExtraTreesClassifier. Prints and plots the importance of the features


    Source: http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html

    :param X:  Feature matrix
    :param y: labels
    :param verbose: Whether a detailed report should be printed out

    :return new feature matrix with selected features

    """

    clf = ExtraTreesClassifier(n_estimators=250, class_weight='balanced')

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


def create_string_from_scores(clf_name, roc_auc, recall, specificity, precision, conf_mat, tuned_params_dict=None):
    """
    Creates a formatted string from the performance scores, confusion matrix and optionally the tuned hyperparameters

    :param clf_name: name of the classifier
    :param roc_auc: roc_auc score
    :param recall: recall score
    :param specificity: specificity score
    :param precision: precision score
    :param conf_mat: confusion matrice
    :param tuned_params_dict: Dictionary containing the tuned parameters and its values

    :return: String

    """
    if tuned_params_dict is None:
        s = '\n******** Scores for %s (Windows:  %i, %i, %i) ******** \n\n' \
            '\troc_auc: %.3f, ' \
            'recall: %.3f, ' \
            'specificity: %.3f, ' \
            'precision: %.3f \n\n' \
            '\tConfusion matrix: \t %s \n\t\t\t\t %s\n\n\n' \
            % (clf_name, f_factory.hw, f_factory.cw, f_factory.gradient_w, roc_auc,
               recall, specificity, precision, conf_mat[0], conf_mat[1])
    else:
        s = '\n******** Scores for %s (Windows:  %i, %i, %i) ******** \n\n' \
            '\tHyperparameters: %s,\n' \
            '\troc_auc: %.3f, ' \
            'recall: %.3f, ' \
            'specificity: %.3f, ' \
            'precision: %.3f \n\n' \
            '\tConfusion matrix: \t %s \n\t\t\t\t %s\n\n\n' \
            % (clf_name, f_factory.hw, f_factory.cw, f_factory.gradient_w, tuned_params_dict, roc_auc,
               recall, specificity, precision, conf_mat[0], conf_mat[1])

    return s


def plot_roc_curve(classifier, X, y, filename, title='ROC'):
    """Plots roc_curve for a given classifier

    :param classifier:  Classifier
    :param X: Feature matrix
    :param y: labels
    :param filename: name of the file that the roc plot should be stored in
    :param title: title of the roc plot

    """

    # allows to add probability output to classifiers which implement decision_function()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    clf = CalibratedClassifierCV(classifier)
    clf.fit(X_train, y_train)

    predicted_probas = clf.predict_proba(X_test)  # returns class probabilities for each class

    fpr, tpr, _ = roc_curve(y_test, predicted_probas[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.title(title)
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    plots.save_plot(plt, 'Performance/Roc Curves/', filename)


def print_confidentiality_scores(X_train, X_test, y_train, y_test):
    """Prints all wrongly classifed datapoints of KNeighborsClassifier and with which confidentiality the classifier
    classified them

    :param X_train: Training data (Feature matrix)
    :param X_test:  Test data (Feature matrix)
    :param y_train: labels of training data
    :param y_test:  labels of test data

    """

    # TODO: Use this somewhere

    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    probas = model.predict_proba(X_test)
    y_predicted = model.predict(X_test)
    for idx, [a, b] in enumerate(probas):
        if y_test[idx] != y_predicted[idx]:
            print('True/Predicted: (' + str(y_test[idx]) + ', ' + str(y_predicted[idx]) + '), Confidentiality: '
                  + str(max(a,b)*100) + '%')


def analyse_performance(clf_list, clf_names, X, y, hyperparameters_are_tuned=False,
                        create_barchart=True, create_roc_curves=True, write_to_file=True):
    """Given a list of classifiers, computes performance (roc_auc, recall, specificity, precision, confusion matrix),
       writes it into a file and plots roc_auc scores of the classifiers in a barchart.


    :param clf_list: list of classifiers the performance should be computed from
    :param clf_names: names of the classifiers
    :param X: feature matrix
    :param y: labels
    :param hyperparameters_are_tuned: Whether or not hyperparameter are tuned (RandomSearchCV) -> To simplify printing scores
    :param create_barchart: Create a barchart consisting of the roc_auc scores
    :param create_roc_curves: Create roc_curves
    :param write_to_file: Write summary of performance into a file (optional)
    """

    scores = []
    names = []
    tuned_params = []
    conf_mats = []

    filename = 'clf_performances_with_hp_tuning' if hyperparameters_are_tuned else 'clf_performances_without_hp_tuning'

    for idx, clf in enumerate(clf_list):
        tuned_parameters = classifiers.get_cclassifier_with_name(clf_names[idx], X, y).tuned_params
        clf_name = clf_names[idx]
        names.append(clf_name)

        if clf_name == 'Naive Bayes':  # Naive Bayes doesn't have any hyperparameters to tune
            roc_auc, recall, specificity, precision, conf_mat, rep = get_performance(clf, clf_name, X, y)
        else:
            roc_auc, recall, specificity, precision, conf_mat, rep = \
                get_performance(clf, clf_name, X, y, tuned_parameters)

        scores.append([roc_auc, recall, specificity, precision])
        tuned_params.append(get_tuned_params_dict(clf, tuned_parameters))
        conf_mats.append(conf_mat)

        if create_roc_curves:
            fn = 'roc_scores_' + clf_name + '_with_hp_tuning.pdf' if hyperparameters_are_tuned \
                 else 'roc_scores_' + clf_name + '_without_hp_tuning.pdf'
            plot_roc_curve(clf, X, y, fn, 'ROC for ' + clf_name + ' without hyperparameter tuning')

    if create_barchart:
        title = 'Scores by classifier with hyperparameter tuning' if hyperparameters_are_tuned \
                else 'Scores by classifier without hyperparameter tuning'
        plot_barchart_scores(names, [s[0] for s in scores], title, filename + '.pdf')

    if write_to_file:
        write_scores_to_file(names, [s[0] for s in scores], [s[1] for s in scores], [s[2] for s in scores],
                             [s[3] for s in scores], tuned_params, conf_mats, filename + '.txt')

    return [s[0] for s in scores]


def calculate_performance_of_classifiers(X, y, tune_hyperparameters=False, reduced_clfs=False, num_iter=20):
    """

    :param X:
    :param y:
    :param tune_hyperparameters:
    :param reduced_clfs: Either do all classifiers or only SVM, Linear SVM, Nearest Neighbor, QDA and Naive Bayes
    :param num_iter: if tune_hyperparameters==True, then how many iterations should be done in RandomSearchCV

    """

    if reduced_clfs:
        clf_names = ['SVM', 'Linear SVM', 'Nearest Neighbor', 'QDA', 'Naive Bayes']
    else:
        clf_names = classifiers.names

    clf_list = [classifiers.get_cclassifier_with_name(name, X, y).clf for name in clf_names]

    if tune_hyperparameters:
        clf_list = [hyperparameter_optimization.get_tuned_clf_and_tuned_hyperparameters(X, y, name, num_iter,
                    verbose=False)[0] for name in clf_names]

    # Compute performance; Write to file and plot barchart

    auc_scores = analyse_performance(clf_list, clf_names, X, y, tune_hyperparameters)

    return auc_scores


def plot_barchart_scores(names, roc_auc_scores, title, filename):
    """Plots the roc_auc scores of each classifier into a barchart

    :param names: list of names of classifiers
    :param roc_auc_scores: roc_auc of each classifier
    :param title: title of the barchart
    :param filename: name of the file
    """

    plots.plot_barchart(title=title,
                        xlabel='Classifier',
                        ylabel='Performance',
                        x_tick_labels=names,
                        values=roc_auc_scores,
                        lbl='auc_score',
                        filename=filename
                        )


def test_clf_with_timedelta_only():
    """
    (Debugging purposes only). Calculates timedelta feature wihtout using anyother features. Since this also gives
    a good score, the timedelta_feature really is a good predictor!

    """

    print("\n################# Testing classifier using timedelta feature only #################\n")

    df_list = random.sample(setup_dataframes.df_list, len(setup_dataframes.df_list))
    # Compute y_true for each logfile
    y_list = []
    for df in df_list:
        y_true = []
        for _, row in df.iterrows():
            if (row['Logtype'] == 'EVENT_CRASH') | (row['Logtype'] == 'EVENT_OBSTACLE'):
                y_true.append(1 if row['Logtype'] == 'EVENT_CRASH' else 0)
        y_list.append(y_true)

    # compute feature matrix for each logfile
    X_matrices = []
    for df in df_list:
        X = []
        for _, row in df.iterrows():
            if (row['Logtype'] == 'EVENT_CRASH') | (row['Logtype'] == 'EVENT_OBSTACLE'):
                last_obstacles = df[(df['Time'] < row['Time']) & ((df['Logtype'] == 'EVENT_OBSTACLE') |
                                                                  (df['Logtype'] == 'EVENT_CRASH'))]
                if last_obstacles.empty:
                    X.append(2)
                else:
                    X.append(row['Time'] - last_obstacles.iloc[-1]['Time'])

        X_matrices.append(X)

    x_train = np.hstack(X_matrices).reshape(-1, 1)  # reshape bc. only one feature
    y_train = np.hstack(y_list).reshape(-1, 1)

    clf = classifiers.get_cclassifier_with_name('Decision Tree', x_train, y_train).clf
    score_dict = cross_validate(clf, x_train, y_train, scoring='roc_auc', cv=10)
    print('Mean roc_auc score with cross_validate: ' + str(np.mean(score_dict['test_score'])))

    ''' Timedeltas correctly computed
    timedeltas = f_factory.get_timedelta_last_obst_feature()['timedelta_to_last_obst']
    # print(sklearn.metrics.accuracy_score(timedeltas, x_train))
    for a, b in zip(timedeltas, x_train):
        print(a, b)

    from sklearn.model_selection import cross_val_score

    # Compute performance scores
    from sklearn.model_selection import cross_val_predict
    from sklearn import metrics
    y_pred = cross_val_predict(clf, x_train, y_train, cv=10)
    y = y_train
    from sklearn.metrics import confusion_matrix
    conf_mat = confusion_matrix(y, y_pred)

    precision = metrics.precision_score(y, y_pred)
    # if clf_name == 'Decision Tree':
    #         plots.plot_graph_of_decision_classifier(model, X, y)

    recall = metrics.recall_score(y, y_pred)
    specificity = conf_mat[0, 0] / (conf_mat[0, 0] + conf_mat[0, 1])
    roc_auc = metrics.roc_auc_score(y, y_pred)
    print(roc_auc, recall, specificity, precision)
    '''


def write_scores_to_file(names, roc_scores, recall_scores, specifity_scores, precision_scores, tuned_params, conf_mats, filename):
    """Writes a formatted text consisting of roc, recall, specificity, precision, tuned_params and confusion matrices
        into a file

    :param names: list of names of classifiers
    :param roc_scores: list of roc_auc scores
    :param recall_scores: list of recall scores
    :param specifity_scores: list of specificity scores
    :param precision_scores: list of precision scores
    :param tuned_params: list of dictionaries containing the tuned parameters and its values
    :param conf_mats: list of confusion matrices
    :param filename: name of the file to be stored

    """

    s = ''

    for i, name in enumerate(names):
        s += create_string_from_scores(name, roc_scores[i], recall_scores[i], specifity_scores[i], precision_scores[i],
                                       conf_mats[i], tuned_params[i])

    write_to_file(s, 'Performance/', filename, 'w+')


def write_to_file(string, folder, filename, mode, verbose=True):
    """Writes a string to a file while checking that the path already exists and creating it if not

        :param string:  Strin to be written to the file
        :param folder: Folder to be saved to
        :param filename: The name (.pdf) under which the plot should be saved\
        :param mode: w+, a+, etc..

    """
    path = sd.working_directory_path + '/Evaluation/' + folder + '/'

    if not os.path.exists(path):
        os.makedirs(path)

    savepath = path + filename

    if verbose:
        print('\nScores written to file ' + '/Evaluation/' + folder + '/' + filename)

    file = open(savepath, mode)
    file.write(string)
