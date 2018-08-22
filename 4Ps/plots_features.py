"""
This module is responsible for generating plots that are involved with features

"""

import os

import graphviz
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb

from sklearn import tree
from sklearn.preprocessing import MinMaxScaler

import classifiers
import features_factory as f_factory
import plots_helpers as hp
import setup_dataframes as sd


def generate_plots_about_features(X, y):
    """
    Generate plots that are concerned with features

    :param X: Feature matrix
    :param y: labels

    """

    _plot_crashes_vs_timedelta(X)
    _plot_timedelta_vs_obstacle_scatter(X, y)  # TODO: Some plots are not necessary I think...
    _plot_feature_distributions(X)
    _plot_mean_value_of_feature_at_crash(X, y)
    for i in range(0, len(f_factory.feature_names)):
        _plot_feature(X, i)


def plot_graph_of_decision_classifier(model, X, y):
    """
    Stores the Decision Classifier Tree in a graph and plots a barchart of the feature_importances

    :param model: Fitted decision Classifier instance
    :param X: Feature matrix
    :param y: labels

    Folder:     Report/
    Plot name:  decision_tree_graph.pdf

    """
    print('Plotting Decision Tree...')

    params_decision_tree = {"max_depth": 2, "class_weight":'balanced'}
    model.set_params(**params_decision_tree)
    model.fit(X, y)

    sorted_importances = sorted(model.feature_importances_, reverse=True)
    sorted_feature_names = [x for _, x in
                            sorted(zip(model.feature_importances_, f_factory.feature_names), reverse=True)]

    hp.plot_barchart('Feature importances with Decision Tree Classifier', 'Feature', 'Importance',
                     sorted_feature_names, sorted_importances,
                     'Importance', 'feature_importances.pdf')

    tree.export_graphviz(
        model,
        out_file='decision_tree_graph',
        feature_names=f_factory.feature_names,
        class_names=['no crash', 'crash'],
        filled=True,
        rounded=True,
        proportion=True,
        special_characters=True,
    )
    graphviz.render('dot', 'pdf', 'decision_tree_graph')

    os.remove(sd.working_directory_path + '/decision_tree_graph')
    os.rename(sd.working_directory_path + '/decision_tree_graph.pdf',
              sd.working_directory_path + '/Plots/report/decision_tree_graph.pdf')


def _plot_feature_distributions(X):
    """
    Plots the distribution of the features in separate plots

    :param X: Feature matrix

    Folder:     Features/Feature_distributions/
    Plot name:  histogram_{feature name}.pdf

    """

    print("Plotting histogram of each feature...")

    f_names = f_factory.feature_names
    for idx, feature in enumerate(f_names):
        x = X[:, idx]
        plt.figure()
        if feature == 'timedelta_to_last_obst':
            mean: float = np.mean(x)
            std_dev: float = np.std(x)
            plt.hist(x, bins=np.arange(mean - 2 * std_dev, mean + 2 * std_dev, 0.005))
        else:
            plt.hist(x)
            # add a 'best fit' line
            # sb.distplot(x)

        plt.title(feature)
        plt.tight_layout()
        filename = 'histogram_' + feature + '.pdf'
        hp.save_plot(plt, 'Features/Feature_distributions/', filename)


def _plot_mean_value_of_feature_at_crash(X, y):
    """
    For each feature, print the average of it when there was a crash vs. there was no crash

    :param X: Feature matrix
    :param y: labels

    Folder:     Features/Crash Correlation/
    Plot name:  barplot_mean_{feature name}_at_crash.pdf

    """

    print("Plotting mean value of each feature when crash vs no crash happened...")

    rows_with_crash = [val for (idx, val) in enumerate(X) if y[idx] == 1]
    rows_without_crash = [val for (idx, val) in enumerate(X) if y[idx] == 0]

    # Iterate over all features and plot corresponding plot
    for i in range(0, len(X[0])):
        mean_when_crash = np.mean([l[i] for l in rows_with_crash])
        mean_when_no_crash = np.mean([l[i] for l in rows_without_crash])
        std_when_crash = np.std([l[i] for l in rows_with_crash])
        std_when_no_crash = np.std([l[i] for l in rows_without_crash])

        plt.subplots()

        plt.bar(1, mean_when_no_crash, width=0.5, yerr=std_when_crash, color=hp.blue_color)
        plt.bar(2, mean_when_crash, width=0.5, yerr=std_when_no_crash, color=hp.green_color)
        plt.ylim(0)
        plt.xticks([1, 2], ['No crash', 'Crash'])
        plt.ylabel(str(f_factory.feature_names[i]))

        plt.title('Average value of feature ' + str(f_factory.feature_names[i]) + ' when crash or not crash')

        filename = 'barplot_mean_' + str(f_factory.feature_names[i]) + '_at_crash.pdf'
        hp.save_plot(plt, 'Features/Crash Correlation/', filename)


def _plot_feature(X, i):
    """
    Plots the feature at position i of each logfile over time

    :param X: Feature matrix
    :param i: Feature index to plot (look at features_factoy for order)

    Folder:     Features/Feature Plots/
    Plot name:  lineplot_{feature name}_{logfile_name}.pdf

    """

    print('Plotting feature ' + f_factory.feature_names[i] + ' of each logfile over time...')

    # df_num_resampled = resample_dataframe(samples, resolution)
    feature_name = f_factory.feature_names[i]
    for idx, _ in enumerate(sd.df_list):
        obst_df = sd.obstacle_df_list[idx]
        times = obst_df['Time']
        start = sum([len(l) for l in sd.obstacle_df_list[:idx]])
        samples = list(X[start:start + len(times), i])
        _, ax1 = plt.subplots()

        # Plot crashes
        crash_times = [row['Time'] for _, row in obst_df.iterrows() if row['crash']]
        crash_values = [samples[index] for index, row in obst_df.iterrows() if row['crash']]

        plt.scatter(crash_times, crash_values, c='r', marker='.', label='crash')
        plt.legend()
        ax1.plot(times, samples, c=hp.blue_color)
        ax1.set_xlabel('Playing time (s)')
        ax1.set_ylabel(feature_name, color=hp.blue_color)
        plt.title('Feature ' + feature_name + ' for logfile ' + sd.names_logfiles[idx])
        ax1.tick_params('y', colors=hp.blue_color)
        # plt.ylim([max(np.mean(X[:, i]) - 3 * np.std(X[:, i]), min(X[:, i])), max(X[:, i])])
        # plt.ylim([0, 1])
        ax1.yaxis.grid(True, zorder=0, color='grey', linewidth=0.3)
        ax1.set_axisbelow(True)

        ax1.spines['top'].set_linewidth(0.3)
        ax1.spines['right'].set_linewidth(0.3)
        filename = 'lineplot_' + feature_name + '_' + sd.names_logfiles[idx] + '.pdf'
        hp.save_plot(plt, 'Features/Feature Plots/' + feature_name + '/', filename)


def _plot_crashes_vs_timedelta(X):
    """
    Plots the percentage of crashes happening depending on the timedelta-feature in a barchart

    :param X:  Feature matrix

    Folder:     Features/
    Plot name:  barplot_%crashes_vs_timedelta.pdf

    """

    print("Plotting percentage crashes vs timedelta...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)
    timedelta_values_at_crashes = []
    timedelta_values_at_non_crashes = []
    timedelta_feature_index = f_factory.feature_names.index('timedelta_to_last_obst')

    obst_conc = pd.concat(sd.obstacle_df_list)

    for idx, row in obst_conc.iterrows():
        if row['crash']:
            timedelta_values_at_crashes.append(X[idx, timedelta_feature_index])
        else:
            timedelta_values_at_non_crashes.append(X[idx, timedelta_feature_index])

    def get_percentage_crashes_for_bin(i):
        """
        Returns percentage of crashes when timedelta is in a certain bin, where bin i: [i/10 , i/10 + 0.1]

        :param i: Bin
        :return: tuple with (opercentage, #occurences)

        """

        conc = timedelta_values_at_crashes + timedelta_values_at_non_crashes
        try:
            return (len([x for x in timedelta_values_at_crashes if i/10 <= x <= i/10 + 0.1]) /
                    len([x for x in conc if i/10 <= x <= i/10 + 0.1]),
                    len([x for x in timedelta_values_at_crashes if i/10 <= x <= i/10 + 0.1]))

        except ZeroDivisionError:
            return 0, 0

    x_tick_labels = ['[0.0, 0.1]', '[0.1, 0.2]', '[0.2, 0.3]', '[0.3, 0.4]', '[0.4, 0.5]', '[0.5, 0.6]', '[0.6, 0.7]',
                     '[0.7, 0.8]', '[0.8, 0.9]', '[0.9, 1.0]']
    tuples = [get_percentage_crashes_for_bin(i) for i in range(0, 10)]
    value_list = [t[0] for t in tuples]
    occurences_list = [t[1] for t in tuples]

    bar_width = 0.2
    fig, ax = plt.subplots()

    plt.title('Percentage of crashes depending on timedelta')
    plt.ylabel('crashes (%)')
    plt.xlabel('timedelta to previous obstacle (s, normalized)')
    plt.xticks(np.arange(len(value_list)) + bar_width / 2, rotation='vertical')
    ax.set_xticklabels(x_tick_labels)
    # ax.set_ylim(0, ceil(max(value_list) * 10) / 10.0)
    plt.bar(np.arange(len(value_list)), value_list, color=hp.blue_color, width=bar_width, label='Crashes (%)')

    ax2 = ax.twinx()
    plt.bar(np.arange(len(value_list)) + bar_width, occurences_list, color=hp.red_color, width=bar_width,
            label='Occurences')
    ax2.set_ylabel('Occurences', color=hp.red_color)
    ax2.tick_params('y', colors=hp.red_color)

    # Add legend with two axis
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)

    ax.yaxis.grid(True, zorder=0, color='grey', linewidth=0.3)
    ax.set_axisbelow(True)
    [i.set_linewidth(0.3) for i in ax.spines.values()]

    hp.save_plot(plt, 'Features/', 'barplot_%crashes_vs_timedelta.pdf')


def _plot_timedelta_vs_obstacle_scatter(X, y):
    """
    Plots timedelta-feature and labels in a scatter plot and the histogram on top

    :param X: Feature matrix
    :param y: labels

    Folder:     Features/Timedelta vs crash/
    Plot name:  scatter_timedelta_crash_mean_over_all_users.pdf or scatter_timedelta_crash_{logfile_name}.pdf

    """

    # Split up feature matrix into one matrix for each logfile
    feature_matrices = []
    label_lists = []
    obstacles_so_far = 0
    for df in sd.obstacle_df_list:
        num_obstacles = len(df.index)
        feature_matrices.append(X.take(range(obstacles_so_far, obstacles_so_far + num_obstacles), axis=0))
        label_lists.append(y[obstacles_so_far:obstacles_so_far + num_obstacles])
        obstacles_so_far += num_obstacles

    X_old = X
    y_old = y

    for i in range(0, len(sd.df_list) + 1):
        plt.subplot()
        if i == len(sd.df_list):  # Do the plot with the entire feature matrix
            X = X_old
            y = y_old
            plt.title('Timedelta vs crash plot aggregated over all logfiles')

        else:
            X = feature_matrices[i]
            y = label_lists[i]
            plt.title('Timedelta vs crash plot for logfile ' + sd.names_logfiles[i])

        g = sb.jointplot(X[:, 0], X[:, 1], kind='reg')

        g.ax_joint.cla()
        plt.sca(g.ax_joint)

        colors = [hp.red_color if i == 1 else hp.green_color for i in y]
        plt.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.3, s=150)
        plt.xticks([0, 1], ['False', 'True'])
        plt.ylim([np.mean(X[:, 1]) - 3 * np.std(X[:, 1]), np.mean(X[:, 1]) + 3 * np.std(X[:, 1])])
        plt.ylabel('Time to last obstacle')
        plt.xlabel('Crash at last obstacle')
        green_patch = mpatches.Patch(color=hp.green_color, label='no crash')
        red_patch = mpatches.Patch(color=hp.red_color, label='crash')

        plt.legend(handles=[green_patch, red_patch])

        if i == len(sd.df_list):
            hp.save_plot(plt, 'Features/Timedelta vs crash/', 'scatter_timedelta_crash_mean_over_all_users.pdf')
        else:
            hp.save_plot(plt, 'Features/Timedelta vs crash/', 'scatter_timedelta_crash_'
                                                              + sd.names_logfiles[i] + '.pdf')


# NOTE: Not used anymore!!
def _plot_scores_with_different_feature_selections():
    """
    After trying different feature selcetions, I plot the scores for each classifier in a barchart.
    Note: The numbers were colelcted by analyzsing the performances!

    1. timedelta_to_last_obst only
    2. timedelta_to_last_obst + last_obstacle_crash
    3. all features
    4. old features (=all features without timedelta_to_last_obst)

    Folder:     Performance
    Plot name:  clf_performance_with_different_features.pdf

    """

    scores_timedelta_only = [0.69, 0.69, 0.84, 0.69, 0.86, 0.86, 0.8, 0.69]
    scores_timedelta_and_last_obst_crash = [0.745, 0.726, 0.99, 0.73, 0.99, 0.994, 0.96, 0.73]
    scores_all_features = [0.68, 0.68, 0.61, 0.64, 0.96, 0.95, 0.965, 0.65]
    scores_old_features = [0.62, 0.63, 0.57, 0.622, 0.53, 0.6, 0.64, 0.74]

    fix, ax = plt.subplots()
    bar_width = 0.2
    line_width = 0.3

    index = np.arange(len(scores_timedelta_only))
    ax.yaxis.grid(True, zorder=0, color='grey', linewidth=0.3)
    ax.set_axisbelow(True)
    [i.set_linewidth(line_width) for i in ax.spines.values()]

    plt.bar(index, scores_timedelta_and_last_obst_crash, bar_width,
            color=hp.red_color,
            label='timedelta_to_last_obst + last_obstacle_crash',
            )

    plt.bar(index + bar_width, scores_timedelta_only, bar_width,
            color=hp.blue_color,
            label='timedelta_to_last_obst',
            )

    plt.bar(index + 2 * bar_width, scores_all_features, bar_width,
            color=hp.green_color,
            label='all features',
            )

    plt.bar(index + 3 * bar_width, scores_old_features, bar_width,
            color=hp.yellow_color,
            label='all features, but without timedelta_to_last_obst',
            )

    plt.ylabel('roc_auc')
    plt.title('roc_auc when selecting different features')
    plt.xticks(index + bar_width / 4, classifiers.names, rotation='vertical')
    ax.set_ylim([0, 1.2])
    plt.legend(prop={'size': 6})

    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    plt.tight_layout()

    hp.save_plot(plt, 'Performance/', 'clf_performance_with_different_features.pdf')


# Note: Not used in the main program
def _plot_timedeltas_and_crash_per_logfile(do_normalize=True):
    """
    Plots for each logfile the mean and std of timedelta_to_last_obst at each obstacle  and if a crash or not happened

    :param do_normalize: Whether to normalize timedelta_feature over time

    Folder:     Features/Timedelta vs Crash Detailed
    Plot name:  crash_logfile_{logfile_name}.pdf

    """

    for idx, df in enumerate(sd.obstacle_df_list):
        timedelta_crash = []
        timedelta_no_crash = []
        computed_timedeltas = []
        for i in range(0, len(df.index)):
            current_obstacle_row = df.iloc[i]
            previous_obstacle_row = df.iloc[i - 1] if i > 0 else current_obstacle_row
            timedelta = current_obstacle_row['Time'] - previous_obstacle_row['Time']

            # Clamp outliers (e.g. because of tutorials etc.). If timedelta >3, it's most likely e.g 33 seconds, so I
            # clamp to c.a. the average
            if timedelta > 3 or timedelta < 1:
                timedelta = 2

            if do_normalize:
                # Normalization (since timedelta over time decreases slightly)
                if len(computed_timedeltas) >= 1:
                    normalized = timedelta / computed_timedeltas[-1]
                else:
                    normalized = 1

                if current_obstacle_row['crash']:
                    timedelta_crash.append(normalized)
                else:
                    timedelta_no_crash.append(normalized)
            else:
                if current_obstacle_row['crash']:
                    timedelta_crash.append(timedelta)
                else:
                    timedelta_no_crash.append(timedelta)

            computed_timedeltas.append(timedelta)

        # Rescale values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(np.array(timedelta_crash + timedelta_no_crash).reshape(-1, 1))

        # Evaluation
        mean_when_crash = np.mean(timedelta_crash)
        mean_when_no_crash = np.mean(timedelta_no_crash)
        std_when_crash = np.std(timedelta_crash)
        std_when_no_crash = np.std(timedelta_no_crash)

        _, _ = plt.subplots()
        plt.ylim(0, 1.2)
        plt.ylabel('Feature value')
        plt.bar(1, mean_when_no_crash, width=0.5, yerr=std_when_no_crash)
        plt.bar(2, mean_when_crash, width=0.5, yerr=std_when_crash, label='Crash')
        plt.xticks([1, 2], ['No crash', 'Crash'])
        plt.title('Average timedelta value for logfile ' + str(idx) + ' when crash or not crash')

        filename = 'crash_logfile_' + sd.names_logfiles[idx] + '.pdf'
        hp.save_plot(plt, 'Features/Timedelta vs Crash Detailed/', filename)
