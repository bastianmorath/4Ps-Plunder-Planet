"""Plots the mean_hr and %crashes that were calulated for the last x seconds for each each second"""

import matplotlib.pyplot as plt

import factory
import globals as gl


green_color = '#AEBD38'
blue_color = '#68829E'
red_color = '#A62A2A'


def plot_features(gamma, c, auroc, percentage):
    fig, ax1 = plt.subplots()
    fig.suptitle('%Crashes and mean_hr over last x seconds')

    # Plot mean_hr
    df = gl.df.sort_values('Time')
    ax1.plot(df['Time'], df['mean_hr'], blue_color)
    ax1.set_xlabel('Playing time [s]')
    ax1.set_ylabel('Heartrate', color=blue_color)
    ax1.tick_params('y', colors=blue_color)

    '''
    # Plot max_over_min_hr
    df = setup.df.sort_values('Time')
    ax1.plot(df['Time'], df['max_over_min'], blue_color)
    ax1.set_xlabel('Playing time [s]')
    ax1.set_ylabel('max_over_min_hr', color=blue_color)
    ax1.tick_params('y', colors=blue_color)
    '''
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


def plot_features_with_labels(X, y):
    _, ax1 = plt.subplots()
    # ax = Axes3D(fig)
    x1 = X[:, 0]  # mean_hr
    x2 = X[:, 1]  # %crashes
    x3 = X[:, 2]  # max_over_min_hr
    color = ['red' if x else 'green' for x in y]

    ax1.scatter(x2, x3, color=color)
    ax1.set_xlabel('crashes [%]')
    ax1.set_ylabel('max_over_min')
    # ax.set_zlabel('max_hr / min_hr')

    plt.savefig(gl.working_directory_path + '/Plots/features_label_crashes__max_over_min.pdf')

    _, ax2 = plt.subplots()
    ax2.scatter(x1, x3, color=color)
    ax2.set_xlabel('mean_hr [normalized]')
    ax2.set_ylabel('max_over_min')
    plt.savefig(gl.working_directory_path + '/Plots/features_label_mean_hr__max_over_min.pdf')

    _, ax3 = plt.subplots()

    ax3.scatter(x1, x2, color=color)
    ax3.set_xlabel('mean_hr [normalized]')
    ax3.set_ylabel('crashes [%]')
    plt.savefig(gl.working_directory_path + '/Plots/features_label_mean_hr__crashes.pdf')


'''Plots the ditribution of the features'''


def plot_feature_distributions(X, y):
    x1 = X[:, 0]  # mean_hr
    x2 = X[:, 1]  # %crashes
    x3 = X[:, 2]  # max_over_min_hr

    plt.subplot(3, 1, 1)
    plt.hist(x1)
    plt.title('mean_hr distribution')

    plt.subplot(3, 1, 2)
    plt.hist(x2)
    plt.title('crashes [%]')

    plt.subplot(3, 1, 3)
    plt.hist(x3)
    plt.title('min_over_max')

    plt.tight_layout()
    plt.savefig(gl.working_directory_path + '/Plots/feature_distributions.pdf')


'''Plots heartrate of all dataframes (Used to compare normaiized hr to original hr)'''


def plot_hr_of_dataframes():
    resolution=5
    for idx, df in enumerate(gl.df_list):
        if not (df['Heartrate'] == -1).all():
            X = []
            X.append(idx)
            df_num_resampled = factory.resample_dataframe(df, resolution)
            # Plot Heartrate
            fig, ax1 = plt.subplots()
            ax1.plot(df_num_resampled['Time'], df_num_resampled['Heartrate'], blue_color)
            ax1.set_xlabel('Playing time [s]')
            ax1.set_ylabel('Heartrate', color=blue_color)
            ax1.tick_params('y', colors=blue_color)

            plt.savefig(gl.working_directory_path + '/Plots/HeartratesNormalized/hr_'+
                        gl.names_logfiles[idx] + '.pdf')
