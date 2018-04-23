"""Plots the mean_hr and %crashes that were calulated for the last x seconds for each each second"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools


import factory
import globals as gl
import features_factory as f_factory

green_color = '#AEBD38'
blue_color = '#68829E'
red_color = '#A62A2A'

# TODO: Broken, since now the features are only stored in feature matrix (without time)


def plot_features(gamma, c, auroc, percentage):
    fig, ax1 = plt.subplots()
    fig.suptitle('%Crashes and mean_hr over last x seconds')

    # Plot mean_hr
    df = gl.df.sort_values('Time')
    ax1.plot(df['Time'], df['mean_hr'], blue_color)
    ax1.set_xlabel('Playing time [s]')
    ax1.set_ylabel('Heartrate', color=blue_color)
    ax1.tick_params('y', colors=blue_color)

    
    # Plot max_over_min_hr
    df = setup.df.sort_values('Time')
    ax1.plot(df['Time'], df['max_over_min'], blue_color)
    ax1.set_xlabel('Playing time [s]')
    ax1.set_ylabel('max_over_min_hr', color=blue_color)
    ax1.tick_params('y', colors=blue_color)
    
    # Plot %crashes
    ax2 = ax1.twinx()
    ax2.plot(df['Time'], df['%crashes'], red_color)
    ax2.set_ylabel('Crashes [%]', color=red_color)
    ax2.tick_params('y', colors=red_color)

    ax2.text(0.5, 0.35, 'Crash_window: ' + str(gl.cw),
         transform=ax2.transAxes, fontsize=10)
    ax2.text(0.5, 0.3, 'Max_over_min window: ' + str(gl.hw),
             transform=ax2.transAxes, fontsize=10)
    ax2.text(0.5, 0.25, 'Best gamma: 10e' + str(round(gamma, 3)),
             transform=ax2.transAxes, fontsize=10)
    ax2.text(0.5, 0.2, 'Best c: 10e' + str(round(c, 3)),
             transform=ax2.transAxes, fontsize=10)
    ax2.text(0.5, 0.15, 'Auroc: ' + str(round(auroc, 3)),
             transform=ax2.transAxes, fontsize=10)
    ax2.text(0.5, 0.1, 'Correctly predicted: ' + str(round(percentage, 2)),
             transform=ax2.transAxes, fontsize=10)

    plt.savefig(gl.working_directory_path + '/features_plot_'+ str(gl.cw) + '_'+ str(gl.hw) + '.pdf')


'''Plot features and corresponding labels to (hopefully) see patterns'''


def plot_feature_correlations(X, y):
    f_names = f_factory.feature_names

    color = ['red' if x else 'green' for x in y]

    for (f1, f2) in itertools.combinations(f_names, r=2):
        _, ax1 = plt.subplots()

        x1 = X[:, f_names.index(f1)]
        x2 = X[:, f_names.index(f2)]

        ax1.scatter(x1, x2, color=color)
        ax1.set_xlabel(f1)
        ax1.set_ylabel(f2)

        plt.savefig(gl.working_directory_path + '/Plots/Correlations/feature_correlation_' + f1 + '_' + f2 + '.pdf')
        plt.close('all')


'''Plots the distribution of the features'''


def plot_feature_distributions(X):
    f_names = f_factory.feature_names
    for idx, feature in enumerate(f_names):
        x = X[:, idx]
        plt.subplot(len(f_names), 1, idx+1)
        plt.hist(x)
        plt.title(feature)

    plt.tight_layout()
    plt.savefig(gl.working_directory_path + '/Plots/feature_distributions.pdf')


'''Plots heartrate of all dataframes (Used to compare normalized hr to original hr)
Only works for real data at the moment, because of name_logfile not existing if test_data...'''


def plot_hr_of_dataframes():
    resolution=5
    for idx, df in enumerate(gl.df_list):
        if not (df['Heartrate'] == -1).all():
            df_num_resampled = factory.resample_dataframe(df, resolution)
            # Plot Heartrate
            fig, ax1 = plt.subplots()
            ax1.plot(df_num_resampled['Time'], df_num_resampled['Heartrate'], blue_color)
            ax1.set_xlabel('Playing time [s]')
            ax1.set_ylabel('Heartrate', color=blue_color)
            ax1.tick_params('y', colors=blue_color)

            plt.savefig(gl.working_directory_path + '/Plots/Heartrates/HeartratesNormalized/hr_'+
                        gl.names_logfiles[idx] + '.pdf')


'''Plots the heartrate in a histogram'''


def plot_heartrate_histogram():
    _, _ = plt.subplots()
    df = pd.concat(gl.df_list, ignore_index=True)
    df = df[df['Heartrate'] != -1]['Heartrate']
    plt.hist(df)
    plt.title('Histogram of HR: $\mu=' + str(np.mean(df)) + '$, $\sigma=' + str(np.std(df)) + '$')
    plt.savefig(gl.working_directory_path + '/Plots/heartrate_distribution.pdf')


'''For each feature, print the average of it when there was a crash vs. there was no crash'''

# TODO: Maybe Make sure that data is not normalized/boxcrox when plotting


def print_mean_features_crash(X, y):
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

