"""
This module contains helper methods for setting up and generating plots

"""


import os

import numpy as np

import matplotlib.pyplot as plt
import setup_dataframes as sd

green_color = '#AEBD38'
blue_color = '#68829E'
red_color = '#A62A2A'
yellow_color = '#FABE3C'
"""
Helper methods

"""


def resample_dataframe(df, resolution):
    """Resamples a dataframe with a sampling frquency of 'resolution'
    -> Smoothes the plots

    :param df: Dataframe to be resampled. Must contain numbers only
    :param resolution: Resolution of the sampling to be done

    :return: Resampled dataframe

    """
    df = df.set_index('timedelta', drop=True)  # set timedelta as new index
    resampled = df.resample(str(resolution)+'S').mean()
    resampled.reset_index(inplace=True)

    # timedelta was resampled, so we need to do the same with the Time-column
    resampled['Time'] = resampled['timedelta'].apply(lambda time: time.total_seconds())

    return resampled


def save_plot(plot, folder, filename):
    """Saves plots and take cares that they are in either of three folders

    :param plot:  The plot to be saved
    :param folder: Folder to be saved to
    :param filename: The name (.pdf) under which the plot should be saved
    """

    path = sd.working_directory_path + '/Plots/' + folder + filename

    # In some cases, I provide sth like abc/test.pdf as filename. I need to split the
    # directory abc and add it to the folder
    directory = path.rsplit('/', 1)[0]  # Gives me everything up to last slash
    name = path.rsplit('/', 1)[1]
    if not os.path.exists(directory):
        os.makedirs(directory)

    savepath = directory + '/' + name

    plot.savefig(savepath, bbox_inches="tight")
    plot.close('all')


def plot_barchart(title, xlabel, ylabel, x_tick_labels, values, lbl, filename, std_err=None, verbose=True,
                  plot_random_guess_line=False):
    """Helper function to plot a barchart with the given arguments

    :param title: Title of the plot
    :param xlabel: name of the x_axis
    :param ylabel: name of the y-axis
    :param x_tick_labels: labels of the x_indices
    :param values: values to plot
    :param lbl: Name of the values label
    :param filename: filename to be stored
    :param std_err: if given, then plot std error of each bar
    :param plot_random_guess_line: If we plot roc_auc scores, we can plot a horizontal line at y=05 for random guess

    :return: The plot
    """

    fix, ax = plt.subplots()
    bar_width = 0.3
    line_width = 0.3
    index = np.arange(len(x_tick_labels))

    ax.yaxis.grid(True, zorder=0, color='grey',  linewidth=0.3)
    ax.set_axisbelow(True)  # Put vertical grid in background of plot
    # [i.set_linewidth(line_width) for i in ax.spines.values()]  # Set width of plot borders

    plt.bar(index, values, bar_width,
            color=blue_color,
            label=lbl,
            yerr=std_err,
            error_kw={'elinewidth': line_width,
                      'capsize': 1.4,
                      'markeredgewidth': line_width},
            zorder=2
            )

    if plot_random_guess_line:
        plt.axhline(y=0.5, color=red_color, linestyle='--', label='random guess', zorder=0)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    # ax.set_ylim([0, min(max(values) + 0.15, 1.2)])
    ax.set_ylim([0, max(values) + np.std(values)])
    # ax.set_ylim([0, 1.1])

    plt.xticks(index, x_tick_labels, rotation='vertical')
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.legend()

    plt.tight_layout()

    save_plot(plt, 'Performance/', filename)

    if verbose:
        print('Barchart plot saved in file Plots/Performance/' + filename)

    return plt
