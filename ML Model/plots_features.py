"""
This module is responsible for plotting various things

"""
import graphviz
from sklearn import tree
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sb
from math import ceil

import seaborn as sns
import setup_dataframes as sd
import features_factory as f_factory
import plots_helpers as hp
import classifiers

green_color = '#AEBD38'
blue_color = '#68829E'
red_color = '#A62A2A'

"""
Plots concerned with features

"""


def plot_graph_of_decision_classifier(model, X, y):
    """Stores the Decision Classifier Tree in a graph and plots a barchart of the feature_importances

    :param model: Fitted decision Classifier instance
    :param X: Feature matrix
    :param y: labels

    """

    # Set class_weight to balanced, such that the graph makes more sense to interpret.
    # I do not do this when actually predicting values  because the performance is better
    params_sdecicion_tree = {"class_weight": "balanced", "max_depth": 4}
    model.set_params(**params_sdecicion_tree)
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


def plot_correlation_matrix(X):
    """Function plots a heatmap of the correlation matrix for each pair of columns (=features) in the dataframe.

        Source: https://seaborn.pydata.org/examples/many_pairwise_correlations.html

    :param X: feature matrix
    """

    corr = X.corr()
    sns.set(style="white")
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    # Set up the matplotlib figure
    plt.subplots(figsize=(len(f_factory.feature_names), len(f_factory.feature_names)))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, annot=True,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, vmin=-1, vmax=1)

    # plt.tight_layout()
    if f_factory.use_reduced_features:
        hp.save_plot(plt, 'Features/', 'correlation_matrix_reduced_features.pdf')

    else:
        hp.save_plot(plt, 'Features/', 'correlation_matrix_all_features.pdf')


def plot_feature_distributions(X):
    """Plots the distribution of the features in separate plots

    :param X: Feature matrix
    """

    print("Plotting histogram of each feature...")

    f_names = f_factory.feature_names
    for idx, feature in enumerate(f_names):
        x = X[:, idx]
        plt.figure()
        if feature == 'timedelta_to_last_obst':
            plt.hist(x, bins=np.arange(np.mean(x) - 2 * np.std(x), np.mean(x) + 2* np.std(x), 0.005))
        else:
            plt.hist(x)

        plt.title(feature)
        plt.tight_layout()
        filename = feature + '.pdf'
        hp.save_plot(plt, 'Features/Feature_distributions/', filename)


def plot_mean_value_of_feature_at_crash(X, y):
    """For each feature, print the average of it when there was a crash vs. there was no crash

    :param X: Feature matrix
    :param y: labels

    """

    print("Plotting mean value of each feature when crash vs no crash happened...")

    # TODO: Maybe Make sure that data is not normalized/boxcrox when plotting

    rows_with_crash = [val for (idx, val) in enumerate(X) if y[idx] == 1]
    rows_without_crash = [val for (idx, val) in enumerate(X) if y[idx] == 0]
    # Iterate over all features and plot corresponding plot
    for i in range(0, len(X[0])):
        mean_when_crash = np.mean([l[i] for l in rows_with_crash])
        mean_when_no_crash = np.mean([l[i] for l in rows_without_crash])
        std_when_crash = np.std([l[i] for l in rows_with_crash])
        std_when_no_crash = np.std([l[i] for l in rows_without_crash])


        _, _ = plt.subplots()

        plt.bar(1,  mean_when_no_crash, width=0.5, yerr=std_when_crash)
        plt.bar(2,  mean_when_crash, width=0.5, yerr=std_when_no_crash)
        plt.ylim(0)
        plt.xticks([1, 2], ['No crash', 'Crash'])

        plt.title('Average value of feature ' + str(f_factory.feature_names[i]) + ' when crash or not crash')

        filename = str(f_factory.feature_names[i]) + '_crash.pdf'
        hp.save_plot(plt, 'Features/Crash Correlation/', filename)


def plot_feature(X, i):
    """Plots the feature at position i of each logfile over time

    :param X: Feature matrix
    :param i: Feature index to plot (look at features_factoy for order)

    """
        
    print('Plotting feature ' + f_factory.feature_names[i] + ' of each logfile over time...')

    # df_num_resampled = resample_dataframe(samples, resolution)
    feature_name = f_factory.feature_names[i]
    for idx, _ in enumerate(sd.df_list):
        obst_df = sd.obstacle_df_list[idx]
        times = obst_df['Time']
        start = sum([len(l) for l in sd.obstacle_df_list[:idx]])
        samples = list(X[start:start+len(times), i])
        _, ax1 = plt.subplots()

        # Plot crashes
        crash_times = [row['Time'] for _, row in obst_df.iterrows() if row['crash']]
        crash_values = [samples[index] for index, row in obst_df.iterrows() if row['crash']]

        plt.scatter(crash_times, crash_values, c='r', marker='.', label='crash')
        plt.legend()
        ax1.plot(times, samples, c=blue_color)
        ax1.set_xlabel('Playing time (s)')
        ax1.set_ylabel(feature_name, color=blue_color)
        plt.title('Feature ' + feature_name + ' for user ' + str(idx))
        ax1.tick_params('y', colors=blue_color)
        plt.ylim([max(np.mean(X[:, i]) - 2*np.std(X[:, i]), min(X[:, i])), max(X[:, i])])
        ax1.yaxis.grid(True, zorder=0, color='grey', linewidth=0.3)
        ax1.set_axisbelow(True)
        # [i.set_linewidth(0.3) for i in ax1.spines.values()]
        # removing top and right borders
        ax1.spines['top'].set_linewidth(0.3)
        ax1.spines['right'].set_linewidth(0.3)
        filename = 'user_' + str(idx) + '_' + feature_name + '.pdf'
        hp.save_plot(plt, 'Features/Feature_plots/' + feature_name + '/', filename)


def plot_crashes_vs_timedelta(X):
    print("Plotting percentage crashes vs timedelta...")

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
        Returns percentage of crashes when timedelta is in a certain bin, where bin i:
        0: timedelta < 0.4
        1: 0.4 <= timdelta < 0.5
        2: 0.5 <= timdelta < 0.6
        3: 0.7 <= timdelta < 0.8
        4: 0.8 <= timdelta < 0.9

        :param i: Bin
        :return: percentage
        """
        conc = timedelta_values_at_crashes + timedelta_values_at_non_crashes

        if i == 0:
            return len([x for x in timedelta_values_at_crashes if x < 0.4]) / len([x for x in conc if x < 0.4])
        if i == 1:
            return len([x for x in timedelta_values_at_crashes if 0.4 <= x < 0.5]) / len([x for x in conc if 0.4 <= x < 0.5])
        if i == 2:
            return len([x for x in timedelta_values_at_crashes if 0.5 <= x < 0.6]) / len([x for x in conc if 0.5 <= x < 0.6])
        if i == 3:
            return len([x for x in timedelta_values_at_crashes if 0.6 <= x < 0.7]) / len([x for x in conc if 0.6 <= x < 0.7])
        if i == 4:
            return len([x for x in timedelta_values_at_crashes if 0.7 <= x < 0.8]) / len([x for x in conc if 0.7 <= x < 0.8])

    x_tick_labels = ['<0.4', '[0.4, 0.5]', '[0.5, 0.6]', '[0.6, 0.7]', '[0.7, 0.8]']
    values = [get_percentage_crashes_for_bin(0), get_percentage_crashes_for_bin(1), get_percentage_crashes_for_bin(2),
              get_percentage_crashes_for_bin(3), get_percentage_crashes_for_bin(4)]

    fig, ax = plt.subplots()

    plt.title('Percentage of crashes depending on timedelta')
    plt.ylabel('crashes (%)')
    plt.xlabel('timedelta to previous obstacle (s)')
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(x_tick_labels)
    ax.set_ylim(0, ceil(max(values) * 100) / 100.0)
    plt.bar(np.arange(len(values)), values, color=blue_color)

    ax.yaxis.grid(True, zorder=0, color='grey', linewidth=0.3)
    ax.set_axisbelow(True)
    [i.set_linewidth(0.3) for i in ax.spines.values()]

    hp.save_plot(plt, 'Features/', 'percentage_crashes_vs_timedelta.pdf')


def plot_corr_knn_distr(X, y):
    """
    Creates 3 plots using seaborn
    1. Correlations between different features and class labels
    2. Features and labels in a scatter plot and the histogram on top
    3. NearestNeighborClassifier decision boundaries

    :param X: Feature matrix
    :param y: labels

    """

    print('Plotting correlations and knn boundaries')

    f1 = 'last_obstacle_crash'
    f2 = 'timedelta_to_last_obst'

    # 1. Plot correlations between different features and class labels
    dat2 = pd.DataFrame({'class': y})
    dat1 = pd.DataFrame({f1: X[:, 0], f2: X[:, 1]})

    matrix_df = dat1.join(dat2)
    sb.pairplot(matrix_df, hue='class')
    hp.save_plot(plt, 'Features/', 'correlation.pdf')

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

    # 2. Plot features and labels in a scatter plot and the histogram on top
    for i in range(0, len(sd.df_list)+1):
        if i == len(sd.df_list):  # Do the plot with the entire feature matrix
            X = X_old
            y = y_old
        else:
            X = feature_matrices[i]
            y = label_lists[i]

        plt.subplot()

        g = sns.jointplot(X[:, 0], X[:, 1], kind='reg')

        g.ax_joint.cla()
        plt.sca(g.ax_joint)

        colors = [red_color if i == 1 else green_color for i in y]
        plt.scatter(X[:, 0],  X[:, 1], c=colors, alpha=0.3, s=150)
        plt.xticks([0, 1], ['False', 'True'])
        plt.ylim([np.mean(X[:, 1]) - 3 * np.std(X[:, 1]), np.mean(X[:, 1]) + 3 * np.std(X[:, 1])])
        plt.ylabel('Time to last obstacle')
        plt.xlabel('Crash at last obstacle')
        green_patch = mpatches.Patch(color=green_color, label='no crash')
        red_patch = mpatches.Patch(color=red_color, label='crash')

        plt.legend(handles=[green_patch, red_patch])

        if i == len(sd.df_list):
            hp.save_plot(plt, 'Features/Correlations/', 'correlation_distr_all.pdf')
        else:
            hp.save_plot(plt, 'Features/Correlations/', 'correlation_distr' + str(i) + '.pdf')
    '''
    # Plot NearestNeighborClassifier decision boundaries
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    h = .02
    clf = neighbors.KNeighborsClassifier()
    clf.fit(X, y)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("2-Class classification (k = %i)"
              % clf.n_neighbors)
    hp.save_plot(plt, 'Features/', 'KNN_boundaries.pdf')
    '''


def plot_timedeltas_and_crash_per_logfile(do_normalize=True):
    """Plots for each logfile the mean and std of timedelta_to_last_obst at each obstacle  and if a crash or not happened

    :return:
    """
    for idx, df in enumerate(sd.obstacle_df_list):
        timedelta_crash = []
        timedelta_no_crash = []
        computed_timedeltas = []
        for i in range(0, len(df.index)):
            current_obstacle_row = df.iloc[i]
            previous_obstacle_row = df.iloc[i-1] if i > 0 else current_obstacle_row
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
        # timedelta_crash = scaler.transform(np.array(timedelta_crash).reshape(-1, 1))  # Rescale between 0 and 1
        # timedelta_no_crash = scaler.transform(np.array(timedelta_no_crash).reshape(-1, 1))  # Rescale between 0 and 1

        # Evaluation
        mean_when_crash = np.mean(timedelta_crash)
        mean_when_no_crash = np.mean(timedelta_no_crash)
        std_when_crash = np.std(timedelta_crash)
        std_when_no_crash = np.std(timedelta_no_crash)
        # print(str(round(mean_when_no_crash, 2)) + ' vs. ' + str(round(mean_when_crash, 2)) + '(std:' +
        #      str(round(std_when_no_crash, 2)) + ' vs. ' + str(round(std_when_crash, 2)),
        #      idx, sd.names_logfiles[idx])

        _, _ = plt.subplots()
        plt.ylim(0, 1.2)
        plt.ylabel('Feature value')
        plt.bar(1, mean_when_no_crash, width=0.5, yerr=std_when_no_crash)
        plt.bar(2, mean_when_crash, width=0.5, yerr=std_when_crash, label='Crash')
        plt.xticks([1, 2], ['No crash', 'Crash'])
        # print(idx, sd.names_logfiles[idx])
        plt.title('Average timedelta value for logfile ' + str(idx) + ' when crash or not crash')

        filename = str(idx) + '_crash.pdf'
        hp.save_plot(plt, 'Features/Crash Correlation_Detailed/', filename)


def plot_scores_with_different_feature_selections():
    """ After trying different feature selcetions, I plot the scores for each classifier in a barchart.
        Note: The numbers were colelcted by analyzsing the performances!

        1. timedelta_to_last_obst only
        2. timedelta_to_last_obst + last_obstacle_crash
        3. all features
        4. old features (=all features without timedelta_to_last_obst)

    """
    # TODO: Only use all or reduced features

    scores_timedelta_only = [0.69, 0.69, 0.84, 0.69, 0.86, 0.86, 0.8, 0.69]
    scores_timedelta_and_last_obst_crash = [ 0.745, 0.726, 0.99, 0.73, 0.99, 0.994, 0.96, 0.73]
    scores_all_features = [0.68, 0.68, 0.61, 0.64, 0.96, 0.95, 0.965, 0.65]
    scores_old_features = [0.62, 0.63, 0.57, 0.622, 0.53, 0.6, 0.64, 0.74]

    fix, ax = plt.subplots()
    bar_width = 0.2
    line_width = 0.3

    index = np.arange(len(scores_timedelta_only))
    ax.yaxis.grid(True, zorder=0, color='grey',  linewidth=0.3)
    ax.set_axisbelow(True)
    [i.set_linewidth(line_width) for i in ax.spines.values()]


    r1 = plt.bar(index, scores_timedelta_and_last_obst_crash, bar_width,
                 color=hp.red_color,
                 label='timedelta_to_last_obst + last_obstacle_crash',
                 )

    r2 = plt.bar(index + bar_width, scores_timedelta_only, bar_width,
                 color=hp.blue_color,
                 label='timedelta_to_last_obst',
                 )

    r3 = plt.bar(index + 2*bar_width, scores_all_features, bar_width,
                 color=hp.green_color,
                 label='all features',
                 )

    r4 = plt.bar(index + 3*bar_width, scores_old_features, bar_width,
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