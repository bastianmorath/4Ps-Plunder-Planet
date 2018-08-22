from collections import OrderedDict

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import plots_helpers as hp
import numpy as np
import seaborn as sb
import pandas as pd

import plots_logfiles as pl
import setup_dataframes as sd
import plots_helpers as ph
import features_factory as f_factory
import setup_dataframes
import window_optimization
import model_factory


def generate_plots_for_report():
    """
    Generate plots that are specifically for the report


    """
    model_factory.plot_roc_curves(hyperparameter_tuning=True, pre_set=True)

    _plot_heartrate_change()
    _plot_difficulties()
    _plot_mean_value_of_heartrate_at_crash()
    _plot_feature_correlation_matrix(reduced_features=False)
    _plot_heartrate_and_events()

    # window_optimization.test_all_windows()  # Potentially takes a long time. Uncomment if you want to use it


def _plot_difficulties():
    """
    Plots difficulties over time as a scatter time and exludes the ones where the difficulty is constant 2 or 3.

    Folder:     Logfiles/
    Plot name:  difficulties.pdf

    """

    resolution = 10  # resample every x seconds -> the bigger, the smoother
    fig, ax = plt.subplots()
    total = 0
    high = 0
    for idx, df in enumerate(sd.df_list):
        df = pl.transform_df_to_numbers(df)
        df_num_resampled = hp.resample_dataframe(df, resolution)
        ax.scatter(df_num_resampled['Time'], df_num_resampled['physDifficulty'], c=hp.green_color, alpha=0.3)
        high += len(df_num_resampled[df_num_resampled['physDifficulty'] == 3])
        total += len(df_num_resampled)

    print('Across all logfiles, the users are in ' + str(round(high/total, 2)) + '% on level HIGH')

    ax.set_ylabel('physDifficulty')
    ax.set_xlabel('Time (s)',)
    plt.yticks([1, 2, 3], ['Low', 'Medium', 'High'])
    plt.title('Difficulties')

    hp.save_plot(plt, 'Report/', 'difficulties.pdf')


def _plot_mean_value_of_heartrate_at_crash():
    """
    For each feature, print the average of it when there was a crash vs. there was no crash

    Folder:     Features/Crash Correlation/
    Plot name:  barplot_mean_{feature name}_at_crash.pdf

    """

    print("Plotting mean value of heartrate when crash vs no crash happened...")

    means_when_crash = []
    means_when_no_crash = []
    stds_when_crash = []
    stds_when_no_crash = []
    for df in sd.df_list:

        df_with_crash = df[df['Logtype'] == 'EVENT_CRASH']
        df_without_crash = df[df['Logtype'] == 'EVENT_OBSTACLE']

        means_when_crash.append(df_with_crash['Heartrate'].mean())
        means_when_no_crash.append(df_without_crash['Heartrate'].mean())
        stds_when_crash.append(df_with_crash['Heartrate'].std())
        stds_when_no_crash.append(df_without_crash['Heartrate'].std())

    fix, ax = plt.subplots()
    bar_width = 0.3
    line_width = 0.3

    index = np.arange(len(means_when_crash))
    ax.yaxis.grid(True, zorder=0, color='grey', linewidth=0.3)
    ax.set_axisbelow(True)
    [i.set_linewidth(line_width) for i in ax.spines.values()]

    plt.bar(index, means_when_crash, bar_width,
            color=ph.red_color,
            label='Heartrate when crash',
            yerr=stds_when_crash,
            error_kw={'elinewidth': line_width,
                      'capsize': 1.4,
                      'markeredgewidth': line_width},
            )

    plt.bar(index + bar_width, means_when_no_crash, bar_width,
            color=ph.blue_color,
            label='Heartrate when no crash',
            yerr=stds_when_no_crash,
            error_kw={'elinewidth': line_width,
                      'capsize': 1.4,
                      'markeredgewidth': line_width},
            )

    plt.ylabel('heartrate (normalized)')
    plt.xlabel('Logfile')
    plt.title('Average value of Heartrate when crash or not crash')
    plt.xticks(index + bar_width / 2, np.arange(1, 20), rotation='horizontal')
    plt.legend(prop={'size': 6})

    filename = 'barplot_mean_heartrate_at_crash.pdf'
    hp.save_plot(plt, 'Report/', filename)


def _plot_heartrate_change():
    """
    Plot number of times the heartrate changed more than {thresh} times

    Folder:     Logfiles/
    Plot name:  barplot_hr_change_thresh.pdf

    """
    thresh = 10

    bpm_changes_over_thresh = []  # Stores #points where change > thresh per logfile

    for idx, df in enumerate(sd.df_list):
        if not (df['Heartrate'] == -1).all():
            resampled = hp.resample_dataframe(df, 1)
            percentage_change = np.diff(resampled['Heartrate']) / resampled['Heartrate'][:-1] * 100.
            x = percentage_change[np.logical_not(np.isnan(percentage_change))]
            bpm_changes_over_thresh.append(len([i for i in x if i > thresh]))

    fig, ax = plt.subplots()

    # plt.title('Number of times the heartrate changed more than ' + str(thresh) + '%')
    plt.ylabel('Number of times')
    plt.xlabel('Logfile')
    index = np.arange(len(bpm_changes_over_thresh))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))  # Only show whole numbers as difficulties
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Only show whole numbers as difficulties
    plt.xticks(index, np.arange(1, 20), rotation='horizontal')

    plt.bar(index, bpm_changes_over_thresh, color=hp.blue_color, width=0.25)

    ax.yaxis.grid(True, zorder=0, color='grey',  linewidth=0.3)
    ax.set_axisbelow(True)
    [i.set_linewidth(0.3) for i in ax.spines.values()]

    hp.save_plot(plt, 'Report/', 'barplot_hr_change_thresh.pdf')


def _plot_feature_correlation_matrix(reduced_features=True):
    """
    Function plots a heatmap of the correlation matrix for each pair of columns (=features) in the dataframe.

    Source: https://seaborn.pydata.org/examples/many_pairwise_correlations.html

    :param reduced_features: Should we use all features or only the reduced ones?

    Folder:     Features/
    Plot name:  correlation_matrix_all_features.pdf or correlation_matrix_reduced_features.pdf

    """

    print("Plotting correlation matrix...")

    X, _ = f_factory.get_feature_matrix_and_label(False, True, True, False, reduced_features)
    X = pd.DataFrame(X)
    corr = X.corr()
    sb.set(style="white")
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask, k=0)] = True
    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(len(f_factory.feature_names), len(f_factory.feature_names)))
    # Generate a custom diverging colormap
    cmap = sb.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    ax.tick_params(labelsize=20)
    sb.heatmap(corr, mask=mask, cmap=cmap, center=0, annot=False, xticklabels=f_factory.feature_names,
               yticklabels=f_factory.feature_names, square=True,
               linewidths=0.0, cbar_kws={"shrink": .6}, vmin=-1, vmax=1)
    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize=20)
    if reduced_features:
        hp.save_plot(plt, 'Report/', 'correlation_matrix_reduced_features.pdf')
    else:
        hp.save_plot(plt, 'Report/', 'correlation_matrix_all_features.pdf')


def _plot_heartrate_and_events():
    """
    Plots the heartrate of logfile 4 (user Is), together with the crashes, Shieldtutorials and Brokenship events.
    Note: Same as plot_heartrate_and_events in plots_logfiles.py, but only for one specific logfile

    Folder:     Report/
    Plot name:  lineplot_hr_and_events.pdf

    """
    setup_dataframes.setup(
        fewer_data=False,  # Specify if we want fewer data (for debugging purposes...)
        normalize_heartrate=False,
        remove_tutorials=False  # We want tutorial to be exactly at 3 and 7.5 minutes!
    )
    print("Plotting heartrate and events...")

    idx = 4
    df = sd.df_list[idx]

    # Plot Heartrate
    _, ax1 = plt.subplots()
    ax1.plot(df['Time'], df['Heartrate'], hp.blue_color, linewidth=1.0, label='Heartrate')
    ax1.set_xlabel('Playing time (s)')
    ax1.set_ylabel('Heartrate', color=hp.blue_color)
    ax1.tick_params('y', colors=hp.blue_color)

    times_crashes = [row['Time'] for _, row in sd.obstacle_df_list[idx].iterrows() if row['crash']]
    heartrate_crashes = [df[df['Time'] == row['Time']].iloc[0]['Heartrate']
                         for _, row in sd.obstacle_df_list[idx].iterrows() if row['crash']]
    plt.scatter(times_crashes, heartrate_crashes, c='r', marker='.', label='Crash')

    # Plot Brokenships
    times_repairing = [row['Time'] for _, row in df.iterrows() if row['Gamemode'] == 'BROKENSHIP']
    hr_max = df['Heartrate'].max()
    hr_min = df['Heartrate'].min()
    for xc in times_repairing:
        plt.vlines(x=xc, ymin=hr_min, ymax=hr_max+0.2, color='y', linewidth=1, label='Ship broken')

    # Plot Shieldtutorial
    times_repairing = [row['Time'] for _, row in
                       df.iterrows() if row['Gamemode'] == 'SHIELDTUTORIAL']
    hr_max = df['Heartrate'].max()
    hr_min = df['Heartrate'].min()

    for xc in times_repairing:
        plt.vlines(x=xc, ymin=hr_min, ymax=hr_max + 0.2, color='g', linewidth=1, label='Shield tutorial')

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))  # Otherwise we'd have one label for each vline
    plt.legend(by_label.values(), by_label.keys())

    filename = 'lineplot_hr_and_events.pdf'
    hp.save_plot(plt, 'Report/', filename)
