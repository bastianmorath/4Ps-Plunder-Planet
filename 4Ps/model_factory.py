"""This module serves as a factory with various methods used by the machine learning modules

"""
from __future__ import division  # s.t. division uses float result

import os
import random
import itertools

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    auc, f1_score, roc_curve, recall_score, roc_auc_score, precision_score,
    confusion_matrix, precision_recall_curve
)
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.feature_selection import SelectFromModel
from custom_transformers import FindCorrelation

import classifiers
import plots_helpers
import plots_features
import features_factory as f_factory
import setup_dataframes as sd
import hyperparameter_optimization
import LSTM


# High level functions

threshold_tuning = True  # Whether optimal threshold of ROC should be used (calc. with Youdens j-score)


def calculate_performance_of_classifiers(X, y, tune_hyperparameters=False, reduced_clfs=True,
                                         create_barchart=True, create_curves=True, do_write_to_file=True,
                                         pre_set=False):
    """Computes performance (roc_auc, recall, specificity, precision, confusion matrix) of either all
    or only reduced classifiers, and optionally writes it into a file and plots roc_auc scores  in a barchart.

    :param X:                       Feature matrix
    :param y:                       labels
    :param tune_hyperparameters:    Whether or not hyperparameter should be tuned
    :param reduced_clfs:            All classifiers, or only SVM, Nearest Neighbor, Random Forest and Naive Bayes
    :param create_barchart:         Create a barchart consisting of the roc_auc scores
    :param create_curves:           Create roc_curves and precision_recall curve
    :param do_write_to_file:        Write summary of performance into a file (optional)
    :param pre_set:                 Some classifiers have pre_tuned parameters (on Euler). Take those instead of tuning

    :return list of roc_aucs, list of roc_auc_stds (one score for each classifier) and formatted string of scores
    """

    if reduced_clfs:
        clf_names = classifiers.reduced_names
    else:
        clf_names = classifiers.names

    clf_list = [classifiers.get_cclassifier_with_name(name, X, y).clf for name in clf_names]

    if tune_hyperparameters or pre_set:
        clf_list = [hyperparameter_optimization.get_tuned_clf_and_tuned_hyperparameters(X, y, name,
                    verbose=False, pre_set=pre_set)[0] for name in clf_names]

    scores_mean = []
    scores_std = []
    names = []
    tuned_params = []
    conf_mats = []

    windows = str(f_factory.hw) + '_' + str(f_factory.cw) + '_' + str(f_factory.gradient_w)

    filename = 'clf_performances_with_hp_tuning_' + windows if tune_hyperparameters \
        else 'clf_performances_without_hp_tuning_' + windows

    for idx, clf in enumerate(clf_list):
        tuned_parameters = classifiers.get_cclassifier_with_name(clf_names[idx], X, y).tuned_params
        clf_name = clf_names[idx]
        names.append(clf_name)

        if clf_name == 'Naive Bayes':  # Naive Bayes doesn't have any hyperparameters to tune
            X_n, y_n = f_factory.get_feature_matrix_and_label(True, True, True, True, False)
            roc_auc, roc_auc_std, recall, recall_std, specificity, specificity_std, precision, precision_std, \
                f1, f1_std, conf_mat, _ = get_performance(clf, clf_name, X_n, y_n, create_curves=create_curves)
        else:
            roc_auc, roc_auc_std, recall, recall_std, specificity, specificity_std, precision, precision_std, f1, \
                f1_std, conf_mat, _ = get_performance(clf, clf_name, X, y, tuned_parameters,
                                                      create_curves=create_curves)

        scores_mean.append([roc_auc, recall, specificity, precision, f1])
        scores_std.append([roc_auc_std, recall_std, specificity_std, precision_std, f1_std])
        tuned_params.append(get_tuned_params_dict(clf, tuned_parameters))
        conf_mats.append(conf_mat)

    if create_barchart:
        title = 'Scores by classifier with hyperparameter tuning' if tune_hyperparameters \
                else 'Scores by classifier without hyperparameter tuning'
        _plot_barchart_scores(names, [s[0] for s in scores_mean], [s[0] for s in scores_std], title, filename + '.pdf')

    s = ''

    roc_scores = [s[0] for s in scores_mean]
    roc_scores_std = [s[0] for s in scores_std]
    recall_scores = [s[1] for s in scores_mean]
    recall_scores_std = [s[1] for s in scores_std]
    specifity_scores = [s[2] for s in scores_mean]
    specifity_scores_std = [s[2] for s in scores_mean]
    precision_scores = [s[3] for s in scores_mean]
    precision_scores_std = [s[3] for s in scores_std]
    f1_scores = [s[4] for s in scores_mean]
    f1_scores_std = [s[4] for s in scores_std]

    for i, name in enumerate(names):
        s += create_string_from_scores(name, roc_scores[i], roc_scores_std[i], recall_scores[i], recall_scores_std[i],
                                       specifity_scores[i], specifity_scores_std[i], precision_scores[i],
                                       precision_scores_std[i], f1_scores[i], f1_scores_std[i],
                                       conf_mats[i], tuned_params[i])

    if do_write_to_file:
        write_to_file(s, 'Performance/', filename + '.txt', 'w+')

    return roc_scores, roc_scores_std, s


def get_performance(model, clf_name, X, y, tuned_params_keys=None, verbose=True, do_write_to_file=False,
                    create_curves=True):
    """Computes performance of the model by doing cross validation with 10 folds, using
        cross_val_predict, and returns roc_auc, recall, specificity, precision, f1, confusion matrix and summary of
        those as a string (plus tuned hyperparameters optionally)

    :param model:               The classifier that should be applied
    :param clf_name:            Name of the classifier (used to print scores)
    :param X:                   Feature matrix
    :param y:                   labels
    :param tuned_params_keys:   keys of parameters that got tuned (in classifiers.py) (optional)
    :param verbose: Whether     a detailed score should be printed out (optional)
    :param do_write_to_file:    Write summary of performance into a file (optional)
    :param create_curves:       Create roc_curves and precision_recall curve

    :return: roc_auc_mean, roc_auc_std, recall_mean, recall_std, specificity_mean, specificity_std,
             precision_mean, precision_std, f1_mean, f1_std, confusion_matrix and summary of those as a string

    Note: I could have also done this emthod with cross_val_predict etc, but this would have required multiple of those,
          which is inefficient.
    """

    if verbose:
        print('Calculating performance of %s...' % clf_name)

    y = np.asarray(y)

    precisions_ = []
    f1s_ = []
    recalls_ = []
    roc_aucs_ = []
    specificities_ = []
    y_pred_list = []
    y_true_list = []
    predicted_probas_list = []

    kf = KFold(n_splits=10)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        scaler = MinMaxScaler(feature_range=(0, 1))

        X_train = scaler.fit_transform(X_train)  # Fit and transform on trainig set, then transform test set too
        X_test = scaler.transform(X_test)

        corr = FindCorrelation(threshold=0.9)
        X_train = corr.fit(X_train).transform(X_train)
        X_test = corr.transform(X_test)

        model.fit(X_train, y_train)

        predicted_probas = model.predict_proba(X_test)
        predicted_probas_list.append(predicted_probas[:, 1])

        fpr, tpr, thresholds = roc_curve(y_test, predicted_probas[:, 1])
        threshold = cutoff_youdens_j(fpr, tpr, thresholds) if threshold_tuning else 0.5

        y_pred = [1 if b > threshold else 0 for (a, b) in predicted_probas]

        y_pred_list.append(y_pred)
        y_true_list.append(y_test)

        conf_mat = confusion_matrix(y_test, y_pred)
        specificities_.append(conf_mat[0, 0] / (conf_mat[0, 0] + conf_mat[0, 1]))

        precisions_.append(precision_score(y_test, y_pred))
        recalls_.append(recall_score(y_test, y_pred))
        f1s_.append(f1_score(y_test, y_pred))

        if len(np.unique(y_test)) == 2:
            roc_aucs_.append(roc_auc_score(y_test, predicted_probas[:, 1]))
        else:
            print('Only one class present in y_true! roc_auc is not defined.')
            recalls_.append(0.5)

    precision_mean = np.mean(precisions_)
    recall_mean = np.mean(recalls_)
    roc_auc_mean = np.mean(roc_aucs_)
    specificity_mean = np.mean(specificities_)
    f1_mean = np.mean(f1s_)

    precision_std = np.std(precisions_)
    recall_std = np.std(recalls_)
    roc_auc_std = np.std(roc_aucs_)
    specificity_std = np.std(specificities_)
    f1_std = np.std(f1s_)

    y_pred = list(itertools.chain.from_iterable(y_pred_list))
    y_true = list(itertools.chain.from_iterable(y_true_list))
    conf_mat = confusion_matrix(y_true, y_pred)

    if clf_name == 'Random Forest':
        try:
            print(X)
            plots_features.plot_graph_of_decision_classifier(model.estimators_[0], X, y)
        except:
            print('Decison Tree could not be plotted')

    if tuned_params_keys is None:
        s = create_string_from_scores(clf_name, roc_auc_mean, roc_auc_std, recall_mean, recall_std,
                                      specificity_mean, specificity_std, precision_mean, precision_std,
                                      f1_mean, f1_std, conf_mat)
    else:
        tuned_params_dict = get_tuned_params_dict(model, tuned_params_keys)
        s = create_string_from_scores(clf_name, roc_auc_mean, roc_auc_std, recall_mean, recall_std,
                                      specificity_mean, specificity_std, precision_mean, precision_std,
                                      f1_mean, f1_std, conf_mat, tuned_params_dict)

    if create_curves:
        fn = 'roc_scores_' + clf_name + '_with_hp_tuning.pdf' if tuned_params_keys is not None \
            else 'roc_scores_' + clf_name + '_without_hp_tuning.pdf'
        _plot_roc_curve(list(itertools.chain.from_iterable(predicted_probas_list)), y_true, fn, 'ROC for ' + clf_name +
                        ' without hyperparameter tuning', plot_thresholds=False)
        # plot_precision_recall_curve(model, X, y, 'precision_recall_curve_' + clf_name)

    if do_write_to_file:
        # Write result to a file
        filename = 'performance_' + clf_name + '_windows_' + str(f_factory.hw) + '_' + str(f_factory.cw) + '_' + \
                    str(f_factory.gradient_w) + '.txt'
        write_to_file(s, 'Performance/', filename, 'w+')

    return roc_auc_mean, roc_auc_std, recall_mean, recall_std, specificity_mean, specificity_std, precision_mean, \
        precision_std, f1_mean, f1_std, conf_mat, s


"""
Helper Functions

"""


def cutoff_youdens_j(fpr, tpr, thresholds):
    """
    Computes optimal threshold of roc curve via Youdens j-score

    :param fpr:         False Positive Rate
    :param tpr:         True Postitive Rate
    :param thresholds:  Thresholds from roc_curve

    :return optimal threshold

    """

    j_scores = tpr - fpr
    j_ordered = sorted(zip(j_scores, thresholds))
    return j_ordered[-1][1]


def write_to_file(string, folder, filename, mode, verbose=True):
    """Writes a string to a file while checking that the path already exists and creating it if not

        :param string:      String to be written to the file
        :param folder:      Folder to be saved to
        :param filename:    The name (.pdf) under which the plot should be saved\
        :param mode:        w+, a+, etc..
        :param verbose:     Print path of saved file

    """
    path = sd.working_directory_path + '/Evaluation/' + folder + '/'

    if not os.path.exists(path):
        os.makedirs(path)

    savepath = path + filename

    if verbose:
        print('\nScores written to file ' + '/Evaluation/' + folder + '/' + filename)

    file = open(savepath, mode)
    file.write(string)


def get_tuned_params_dict(model, tuned_params_keys):
    """Given a list of tuned parameter keys and the model, create a dictionary with the tuned parameters and its values
    associated with it.
    Used for printing out scores

    :param model: Model, such that we can extract the parameters from
    :param tuned_params_keys: Parameters that we tuned in RandomizedSearchCV

    :return: Dictionary with tuned parameters and its values
    """
    values = [model.get_params()[x] for x in tuned_params_keys]
    return dict(zip(tuned_params_keys, values))


def create_string_from_scores(clf_name, roc_auc_mean, roc_auc_std, recall_mean, recall_std, specificity_mean,
                              specificity_std, precision_mean, precision_std, f1_mean, f1_std, conf_mat,
                              tuned_params_dict=None):
    """
    Creates a formatted string from the performance scores, confusion matrix and optionally the tuned hyperparameters

    :param clf_name:            name of the classifier
    :param roc_auc_mean:        roc_auc mean
    :param roc_auc_std:         roc_auc standard deviation
    :param recall_mean:         recall mean
    :param recall_std:          recall standard deviation
    :param specificity_mean:    specificity mean
    :param specificity_std:     specificity  std
    :param precision_mean:      precision mean
    :param precision_std:       precision standard deviation
    :param f1_mean:
    :param f1_std:
    :param conf_mat:            confusion matrice
    :param tuned_params_dict:   Dictionary containing the tuned parameters and its values

    :return: Formatted string from scores

    """

    if tuned_params_dict is None:
        s = '\n\n******** Scores for %s (Windows:  %i, %i, %i) ******** \n\n' \
            '\troc_auc: %.3f (+-%.2f), ' \
            'recall: %.3f (+-%.2f), ' \
            'specificity: %.3f (+-%.2f), ' \
            'precision: %.3f (+-%.2f), ' \
            'f1: %.3f (+-%.2f) \n\n' \
            '\tConfusion matrix: \t %s \n\t\t\t\t %s\n\n\n' \
            % (clf_name, f_factory.hw, f_factory.cw, f_factory.gradient_w, roc_auc_mean, roc_auc_std, recall_mean,
               recall_std, specificity_mean, specificity_std, precision_mean, precision_std, f1_mean, f1_std,
               conf_mat[0], conf_mat[1])
    else:
        s = '\n******** Scores for %s (Windows:  %i, %i, %i) ******** \n\n' \
            '\tHyperparameters: %s,\n' \
            '\troc_auc: %.3f (+-%.2f), ' \
            'recall: %.3f (+-%.2f), ' \
            'specificity: %.3f (+-%.2f), ' \
            'precision: %.3f (+-%.2f), ' \
            'f1: %.3f (+-%.2f) \n\n' \
            '\tConfusion matrix: \t %s \n\t\t\t\t %s\n\n' \
            % (clf_name, f_factory.hw, f_factory.cw, f_factory.gradient_w, tuned_params_dict, roc_auc_mean, roc_auc_std,
               recall_mean, recall_std, specificity_mean, specificity_std, precision_mean, precision_std, f1_mean,
               f1_std, conf_mat[0], conf_mat[1])

    return s


# Plotting


def _plot_roc_curve(predicted_probas,  y, filename, title='ROC', plot_thresholds=False):
    """
    Plots roc_curve for a given classifier

    :param predicted_probas: Probabilities of positive label
    :param y: labels
    :param filename: name of the file that the roc plot should be stored in
    :param title: title of the roc plot
    :param plot_thresholds: Also plot thresholds

    """

    # allows to add probability output to classifiers which implement decision_function()
    # clf = CalibratedClassifierCV(classifier)

    fpr_, tpr_, thresholds_ = roc_curve(y, predicted_probas)
    roc_auc = auc(fpr_, tpr_)

    plt.figure()
    plt.title(title)
    plt.plot(fpr_, tpr_, plots_helpers.blue_color, label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], c='gray',  ls='--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    if plot_thresholds:
        # create the axis of thresholds (scores)
        ax2 = plt.gca().twinx()
        ax2.plot(fpr_, thresholds_, markeredgecolor='r', linestyle='dashed', color='r')
        ax2.set_ylabel('Threshold', color='r')

        ax2.set_ylim([thresholds_[-1], thresholds_[0]])
        ax2.set_xlim([fpr_[0], fpr_[-1]])

    plots_helpers.save_plot(plt, 'Report/', filename)


def plot_roc_curves(hyperparameter_tuning=False, pre_set=True, with_lstm=False):
    """
    Plots roc_curves for all classifier in one single plot

    :param hyperparameter_tuning: Do hyperparameter tuning
    :param pre_set: Some classifiers have pre_tuned parameters (on Euler). Take those instead of tuning
    :param with_lstm: Also include ROC of LSTM network (takes a little time...)

    Folder:     Report/
    Plot name:  roc_curves.pdf

    """

    X, y = f_factory.get_feature_matrix_and_label(
                verbose=False,
                use_cached_feature_matrix=True,
                save_as_pickle_file=True,
                reduced_features=False,
                use_boxcox=False
        )

    clf_names = ['SVM', 'Nearest Neighbor', 'Random Forest', 'Naive Bayes']

    if pre_set:
        clf_list = [classifiers.get_cclassifier_with_name(name, X, y).tuned_clf for name in clf_names]
    else:
        clf_list = [classifiers.get_cclassifier_with_name(name, X, y).clf for name in clf_names]

    tprs = []
    fprs = []
    roc_aucs = []

    for idx, classifier in enumerate(clf_list):
        if hyperparameter_tuning:
            classifier, _ = hyperparameter_optimization.get_tuned_clf_and_tuned_hyperparameters(
                X, y, clf_name=clf_names[idx], verbose=False, pre_set=True
            )

        # clf = CalibratedClassifierCV(classifier)
        clf = classifier
        kf = KFold(n_splits=10)
        predicted_probas_list = []
        y = np.array(y)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            scaler = MinMaxScaler(feature_range=(0, 1))

            X_train = scaler.fit_transform(X_train)  # Fit and transform on trainig set, then transform test set too
            X_test = scaler.transform(X_test)

            corr = FindCorrelation(threshold=0.9)
            X_train = corr.fit(X_train).transform(X_train)
            X_test = corr.transform(X_test)

            clf.fit(X_train, y_train)

            predicted_probas = clf.predict_proba(X_test)
            predicted_probas_list.append(predicted_probas[:, 1])

        fpr, tpr, _ = roc_curve(y, list(itertools.chain.from_iterable(predicted_probas_list)))
        roc_auc = auc(fpr, tpr)
        fprs.append(fpr)
        tprs.append(tpr)
        roc_aucs.append(roc_auc)

    # Also add LSTM scores:
    if with_lstm:
        clf_names.append("LSTM")
        fpr, tpr, roc_auc = LSTM.create_roc_curve(X, y, 130)
        fprs.append(fpr)
        tprs.append(tpr)
        roc_aucs.append(roc_auc)

    plt.figure()

    for idx, name in enumerate(clf_names):
        plt.plot(fprs[idx], tprs[idx], label=name + ' (AUC = %0.2f)' % roc_aucs[idx])

    plt.title('Roc curves')
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], c='gray', ls='--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    plots_helpers.save_plot(plt, 'Report/', 'roc_curves.pdf')


def _plot_barchart_scores(names, roc_auc_scores, roc_auc_scores_std, title, filename):
    """
    Plots the roc_auc scores of each classifier into a barchart

    :param names: list of names of classifiers
    :param roc_auc_scores: roc_auc of each classifier
    :param roc_auc_scores_std: standard deviations of roc_auc of each classifier
    :param title: title of the barchart
    :param filename: name of the file

    """

    plots_helpers.plot_barchart(title=title,
                                xlabel='Classifier',
                                ylabel='Performance',
                                x_tick_labels=names,
                                values=roc_auc_scores,
                                lbl='auc_score',
                                filename=filename,
                                std_err=roc_auc_scores_std,
                                plot_random_guess_line=True
                                )


def plot_precision_recall_curve(classifier, X, y, filename):
    """
    Plots and saves a precision recall curve

    :param classifier:  Classifier to generate precision-recall curve from
    :param X:           Feature matrix
    :param y:           labels
    :param filename:    Name of the file the plot should be stored to


    """

    # allows to add probability output to classifiers which implement decision_function()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    scaler = MinMaxScaler(feature_range=(0, 1))

    X_train = scaler.fit_transform(X_train)  # Fit and transform on trainig set, then transform test set too
    X_test = scaler.transform(X_test)

    corr = FindCorrelation(threshold=0.9)
    X_train = corr.fit(X_train).transform(X_train)
    X_test = corr.transform(X_test)
    classifier.fit(X_train, y_train)

    decision_fct = getattr(classifier, "decision_function", None)
    if callable(decision_fct):
        y_score = classifier.decision_function(X_test)
        precision, recall, _ = precision_recall_curve(y_test, y_score)

        plt.step(recall, precision, color='b', alpha=0.2,
                 where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2,
                         color='b')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('2-class Precision-Recall curve')

        plots_helpers.save_plot(plt, 'Performance/Precision Recall Curves/', filename)
    else:
        print('\tThis classifier doesn\'t implement decision_function(), '
              'thus no precision_recall curve can be generated')


# Note: Not used in the main program
def _feature_selection(X, y, verbose=False):
    """
    Feature Selection with ExtraTreesClassifier. Prints and plots the importance of the features


    Source: http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html

    :param X:       Feature matrix
    :param y:       labels
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

    plots_helpers.save_plot(plt, 'Features/', 'feature_importance_decision_tree.pdf')

    return X_new, y


# Note: Not used in the main program
def _test_clf_with_timedelta_only():
    """
    (Debugging purposes only). Calculates timedelta feature without using any other features. Since this also gives
    a good score, the timedelta_feature really is a good predictor!

    """

    print("\n################# Testing classifier using timedelta feature only #################\n")

    df_list = random.sample(sd.df_list, len(sd.df_list))
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

    ''' 
    # Timedeltas correctly computed
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
    # roc_auc = metrics.roc_auc_score(y, y_pred)  # Wrong, use probas!!
    print(roc_auc, recall, specificity, precision)
    '''


# Note: Not used in the main program
def _print_confidentiality_scores(X_train, X_test, y_train, y_test):
    """Prints all wrongly classifed datapoints of KNeighborsClassifier and with which confidentiality the classifier
    classified them

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
                  + str(max(a, b)*100) + '%')
