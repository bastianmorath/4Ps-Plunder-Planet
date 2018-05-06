"""Plots the mean_hr and %crashes that were calulated for the last x seconds for each each second"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import seaborn as sns
import factory
import globals as gl
import features_factory as f_factory

green_color = '#AEBD38'
blue_color = '#68829E'
red_color = '#A62A2A'

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
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.savefig(gl.working_directory_path + '/Plots/Correlations/correlation_matrix.pdf')




def plot_feature_distributions(X):
    """Plots the distribution of the features

    :param X: Feature matrix
    """

    f_names = f_factory.feature_names
    for idx, feature in enumerate(f_names):
        x = X[:, idx]
        plt.figure()
        plt.hist(x)
        plt.title(feature)

        plt.tight_layout()
        plt.savefig(gl.working_directory_path + '/Plots/Feature_distributions/feature_distribution_' + feature + '.pdf')


def plot_hr_of_dataframes():
    """Plots heartrate of all dataframes (Used to compare normalized hr to original hr)
        Only works for real data at the moment, because of name_logfile not existing if test_data...

    :return:
    """
    resolution = 5
    for idx, df in enumerate(gl.df_list):
        if not (df['Heartrate'] == -1).all():
            df_num_resampled = factory.resample_dataframe(df, resolution)
            # Plot Heartrate
            _, ax1 = plt.subplots()
            ax1.plot(df_num_resampled['Time'], df_num_resampled['Heartrate'], blue_color)
            ax1.set_xlabel('Playing time [s]')
            ax1.set_ylabel('Heartrate', color=blue_color)
            ax1.tick_params('y', colors=blue_color)

            plt.savefig(gl.working_directory_path + '/Plots/Heartrates/hr_'+
                        gl.names_logfiles[idx] + '.pdf')
            plt.close('all')


def plot_heartrate_histogram():
    """ Plots a histogram of  heartrate data accumulated over all logfiles

    """

    _, _ = plt.subplots()
    df = pd.concat(gl.df_list, ignore_index=True)
    df = df[df['Heartrate'] != -1]['Heartrate']
    plt.hist(df)
    plt.title('Histogram of HR: $\mu=' + str(np.mean(df)) + '$, $\sigma=' + str(np.std(df)) + '$')
    plt.savefig(gl.working_directory_path + '/Plots/heartrate_distribution.pdf')





def print_mean_features_crash(X, y):
    """For each feature, print the average of it when there was a crash vs. there was no crash

    :param X: Feature matrix
    :param y: labels

    """
    # TODO: Maybe Make sure that data is not normalized/boxcrox when plotting

    rows_with_crash = [val for (idx, val) in enumerate(X) if y[idx] == 1]
    rows_without_crash = [val for (idx, val) in enumerate(X) if y[idx] == 0]
    # Iterate over all features and plot corresponding plot
    for i in range(0, len(X[0])):
        mean_with_obstacles = np.mean([l[i] for l in rows_with_crash])
        mean_without_obstacles = np.mean([l[i] for l in rows_without_crash])
        std_with_obstacles = np.std([l[i] for l in rows_with_crash])
        std_without_obstacles = np.std([l[i] for l in rows_without_crash])

        _, _ = plt.subplots()

        plt.bar(0,  mean_without_obstacles, width=0.5, yerr=std_without_obstacles, label='No crash')
        plt.bar(1,  mean_with_obstacles, width=0.5, yerr=std_with_obstacles, label='Crash')
        plt.legend()
        plt.title('Average value of feature ' + str(f_factory.feature_names[i]) + ' when crash or not crash')
        plt.savefig(gl.working_directory_path + '/Plots/Crash_Correlation/bar_feature_' + str(f_factory.feature_names[i]) + '_crash.pdf')
        plt.close('all')


def plot_barchart(title, x_axis_name, y_axis_name, x_labels, values, lbl):
    """Plots a barchart with the given arguments
    
    Arguments:
        title {String} -- Title of the plot
        x_axis_name {String} -- name of the x_axis
        y_axis_name {String} -- name of the y-axis
        x_labels {[String]} -- labels of the x_indices
        values {[type]} -- values to plot
        label {String} -- Name of the values label
    
    Returns:
        matplotlib.plt -- Generated plot
    """

    fix, ax = plt.subplots()
    bar_width = 0.3
    opacity = 0.4
    index = np.arange(len(x_labels))
    
    rects = plt.bar(index, values, bar_width,
            alpha=opacity,
            color=green_color,
            label=lbl)

    plt.xlabel(x_axis_name)
    plt.ylabel(y_axis_name)
    plt.title(title)
    plt.xticks(index, x_labels, rotation='vertical')
    plt.legend()

    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for i, rect in enumerate(rects):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 1.02 * height,
                    '%0.3f' % values[i],
                    ha='center', va='bottom', size=5)

    autolabel(rects)

    plt.tight_layout()

    return plt
