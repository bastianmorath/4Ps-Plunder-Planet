import matplotlib.pyplot as plt
import plots_helpers as hp
import numpy as np

import plots_logfiles as pl
import setup_dataframes as sd
import plots_helpers as ph


def generate_plots_for_report():
    """
    Generate plots that are specifically for the report


    """

    _plot_difficulties()
    _plot_mean_value_of_heartrate_at_crash()


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
        high += len(df_num_resampled[df_num_resampled['physDifficulty']==3])
        total += len(df_num_resampled)

    print('Across all logfiles, the users are in ' + str(round(high/total, 2)) + '% on level HIGH')

    ax.set_ylabel('physDifficulty')
    ax.set_xlabel('Time (s)',)
    plt.yticks([1, 2, 3], ['Low', 'Medium', 'High'])
    plt.title('Difficulties')

    hp.save_plot(plt, 'Logfiles/', 'difficulties.pdf')


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
    plt.xticks(index + bar_width / 2, np.arange(1, 20), rotation='vertical')
    plt.legend(prop={'size': 6})

    filename = 'barplot_mean_heartrate_at_crash.pdf'
    hp.save_plot(plt, 'Features/', filename)
