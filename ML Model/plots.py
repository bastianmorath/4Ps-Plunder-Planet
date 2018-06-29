"""
This module is responsible for plotting various things

"""
import matplotlib
from matplotlib.ticker import MaxNLocator

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

import seaborn as sns
import setup_dataframes as sd
import features_factory as f_factory


green_color = '#AEBD38'
blue_color = '#68829E'
red_color = '#A62A2A'

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


def plot_barchart(title, xlabel, ylabel, x_tick_labels, values, lbl, filename, std_err=None, verbose=True):
    """Helper function to plot a barchart with the given arguments

    :param title: Title of the plot
    :param xlabel: name of the x_axis
    :param ylabel: name of the y-axis
    :param x_tick_labels: labels of the x_indices
    :param values: values to plot
    :param lbl: Name of the values label
    :param filename: filename to be stored
    :param std_err: if given, then plot std error of each bar

    :return: The plot
    """

    fix, ax = plt.subplots()
    bar_width = 0.3
    opacity = 0.4
    index = np.arange(len(x_tick_labels))

    r = plt.bar(index, values, bar_width,
        alpha=opacity,
        color=blue_color,
        label=lbl,
        yerr=std_err)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    ax.set_ylim([0, min(max(values) + 0.15, 1.0)])

    plt.xticks(index, x_tick_labels, rotation='vertical')
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

    autolabel(r)

    plt.tight_layout()

    save_plot(plt, 'Performance/', filename)

    if verbose:
        print('Barchart plot saved in file Plots/Performance/' + filename)

    return plt


"""
Plots concerned with features

"""


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

    plt.tight_layout()
    if f_factory.use_reduced_features:
        save_plot(plt, 'Features/', 'correlation_matrix_reduced_features.pdf')

    else:
        save_plot(plt, 'Features/', 'correlation_matrix_all_features.pdf')


def plot_feature_distributions(X):
    """Plots the distribution of the features in separate plots

    :param X: Feature matrix
    """

    print("Plotting histogram of each feature...")

    f_names = f_factory.feature_names
    for idx, feature in enumerate(f_names):
        x = X[:, idx]
        plt.figure()
        if feature == 'timedelta_last_obst':
            print(x)
            plt.hist(x, bins=np.arange(0.03, 0.07 + 0.005, 0.005))
        else:
            plt.hist(x)

        plt.title(feature)
        plt.tight_layout()
        filename = feature + '.pdf'
        save_plot(plt, 'Features/Feature_distributions/', filename)


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
        save_plot(plt, 'Features/Crash Correlation/', filename)


def plot_feature(X, i):
    """Plots the feature at position i of each logfile over time

    :param X: Feature matrix
    :param i: Feature index to plot (look at features_factoy for order)

    """
        
    print('Plotting feature ' + f_factory.feature_names[i] + ' of each logfile over time...')

    # df_num_resampled = resample_dataframe(samples, resolution)
    # first dataframe only
    feature_name = f_factory.feature_names[i]
    for idx, df in enumerate(sd.df_list):
        times = sd.obstacle_df_list[idx]['Time']
        start = sum([len(l) for l in sd.obstacle_df_list[:idx]])
        samples = list(X[start:start+len(times), i])
        _, ax1 = plt.subplots()

        ax1.plot(times, samples, c=red_color)
        ax1.set_xlabel('Playing time [s]')
        ax1.set_ylabel(feature_name, color=blue_color)
        plt.title('Feature ' + feature_name + ' for user ' + str(idx))
        ax1.tick_params('y', colors=blue_color)

        filename = 'user_' + str(idx) + '_' + feature_name + '.pdf'
        save_plot(plt, 'Features/Feature_plots/' + feature_name + '/', filename)


"""
Plots concerned with logfiles

"""


def plot_hr_of_dataframes():
    """Plots heartrate of all dataframes (Used to compare normalized hr to original hr)
        Only works for real data at the moment, because of name_logfile not existing if test_data...

    :return:
    """

    print("Plotting heartrate of dataframes over time...")
    resolution = 5
    for idx, df in enumerate(sd.df_list):
        if not (df['Heartrate'] == -1).all():
            df_num_resampled = resample_dataframe(df, resolution)
            # Plot Heartrate
            _, ax1 = plt.subplots()
            ax1.plot(df_num_resampled['Time'], df_num_resampled['Heartrate'], blue_color)
            ax1.set_xlabel('Playing time [s]')
            ax1.set_ylabel('Heartrate', color=blue_color)
            ax1.tick_params('y', colors=blue_color)

            filename = 'hr_' + sd.names_logfiles[idx] + '.pdf'
            save_plot(plt, 'Logfiles/Heartrate/', filename)


def plot_heartrate_histogram():
    """ Plots a histogram of  heartrate data accumulated over all logfiles

    """
    print("Plotting histogram of heartrate of accumulated logfiles...")

    _, _ = plt.subplots()
    df = pd.concat(sd.df_list, ignore_index=True)
    df = df[df['Heartrate'] != -1]['Heartrate']
    plt.hist(df)
    plt.title('Histogram of HR: $\mu=%.3f$, $\sigma=%.3f$'
              % (np.mean(df), np.std(df)))

    save_plot(plt, 'Logfiles/', 'heartrate_distribution_all_logfiles.pdf')


def plot_average_hr_over_all_logfiles():
    """
    Plots average heartrate over all logfiles
    """
    fig, ax = plt.subplots()
    plt.ylabel('Heartrate [bpm]')
    plt.xlabel('Playing time [s]')
    plt.title('Average Heartrate across all users')

    conc_dataframes = pd.concat(sd.df_list, ignore_index=True)
    time_df = conc_dataframes.groupby(['userID', 'logID'])['Time'].max()

    min_time = time_df.min()

    conc_dataframes = conc_dataframes[
        conc_dataframes['Time'] < min_time]  # Cut all dataframes to the same minimal length

    df_copy = conc_dataframes.copy()  # to prevent SettingWithCopyWarning
    avg_hr_df = df_copy.groupby(['timedelta'])[['timedelta', 'Heartrate']].mean()  # Take mean over all logfiles
    avg_hr_df.reset_index(inplace=True)
    avg_hr_df_resampled = resample_dataframe(avg_hr_df, 10)

    plt.plot(avg_hr_df_resampled['Time'], avg_hr_df_resampled['Heartrate'])
    save_plot(plt, 'Logfiles/', 'average_heartrate.pdf')


def plot_mean_and_std_hr_boxplot():
    """
    Plots mean and std bpm per user in a box-chart

    """

    conc_dataframes = pd.concat(sd.df_list, ignore_index=True)

    df2 = conc_dataframes.pivot(columns=conc_dataframes.columns[1], index=conc_dataframes.index)
    df2.columns = df2.columns.droplevel()
    conc_dataframes[['Heartrate', 'userID']].boxplot(by='userID', grid=False, sym='r+')
    plt.ylabel('Heartrate [bpm]')
    plt.title('')
    save_plot(plt, 'Logfiles/', 'mean_heartrate_boxplot.pdf')


def plot_heartrate_change():
    ''' Plot Heartrate change
    '''
    bpm_changes_max = []  # Stores max. absolute change in HR per logfile
    bpm_changes_rel = []  # Stores max. percentage change in HR per logfile

    X = []
    for idx, df in enumerate(sd.df_list):
        if not (df['Heartrate'] == -1).all():
            X.append(idx)
            # new = df.set_index('timedelta', inplace=False)
            resampled = resample_dataframe(df, 1)
            percentage_change = np.diff(resampled['Heartrate']) / resampled['Heartrate'][:-1] * 100.
            x = percentage_change[np.logical_not(np.isnan(percentage_change))]
            bpm_changes_max.append(x.max())
            bpm_changes_rel.append(x)

    plt.ylabel('#Times HR changed')
    plt.xlabel('Change in Heartrate [%]')
    for idx, l in enumerate(bpm_changes_rel):
        name = str(sd.names_logfiles[idx])
        plt.figure()
        plt.title('Heartrate change for plot ' + name)
        plt.hist(l, color=blue_color)

        save_plot(plt, 'Logfiles/Abs Heartrate Changes/', 'heartrate_change_percentage_' + name + '.pdf')

    plt.figure()
    plt.title('Maximal heartrate change')
    plt.ylabel('Max heartrate change [%]')
    plt.xlabel('Logfile')
    plt.bar([x for x in X], bpm_changes_max, color=blue_color, width=0.25)
    save_plot(plt, 'Logfiles/', 'heartrate_change_abs.pdf')


def transform_df_to_numbers(df):
    """
    Subsitutes difficulties with numbers to work with them in a better way, from 1 to 3

    :param df:
    :return:
    """

    mapping = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'undef': -1}
    df = df.replace({'physDifficulty': mapping, 'psyStress': mapping, 'psyDifficulty': mapping})

    for col in ['physDifficulty', 'psyStress', 'psyDifficulty']:
        df[col] = df[col].astype('int64')
    return df


def plot_hr_or_points_corr_with_difficulty(to_compare):
    resolution = 10  # resample every x seconds -> the bigger, the smoother
    for idx, df in enumerate(sd.df_list):
        df = transform_df_to_numbers(df)
        if not (df['Heartrate'] == -1).all():
            X=[]
            X.append(idx)
            df_num_resampled = resample_dataframe(df, resolution)
            # Plot Heartrate
            fig, ax1 = plt.subplots()
            ax1.plot(df_num_resampled['Time'], df_num_resampled[to_compare], blue_color)
            ax1.set_xlabel('Playing time [s]')
            ax1.set_ylabel(to_compare, color=blue_color)
            ax1.tick_params('y', colors=blue_color)

            # Plot Difficulty
            ax2 = ax1.twinx()
            ax2.plot(df_num_resampled['Time'], df_num_resampled['physDifficulty'], green_color)
            ax2.set_ylabel('physDifficulty', color=green_color)
            ax2.tick_params('y', colors=green_color)
            ax2.yaxis.set_major_locator(MaxNLocator(integer=True))  # Only show whole numbers as difficulties
            plt.title('Difficulty and ' + to_compare + ' for user ' + sd.names_logfiles[idx])
            save_plot(plt, 'Logfiles/', to_compare + ' Difficulty Corr/' + to_compare + '_difficulty_' +
                      str(sd.names_logfiles[idx]) + '.pdf')



'''Returns a list which says how many times the obstacle has size {0,1,2,3,4}  for each difficulty level {1,2,3} in the form
    [0, 0, 0, 0, 0, 0, 143, 0, 581, 25, 0, 2659, 0, 299, 5589]
    # occurences where obstacle had size 0 in Diff=LOW, ... , 
    # occurences where obstacle had size 4 in Diff=LOW,
    # occurences where obstacle had size 0 in Diff=MEDIUM,...]
'''
def get_number_of_obstacles_per_difficulty():
    conc_dataframes = pd.concat(sd.df_list, ignore_index=True)
    conc_num = transform_df_to_numbers(conc_dataframes) # Transform Difficulties into integers
    # count num.obstacle parts per obstacle
    new = conc_num['obstacle'].apply(lambda x: 0 if x == 'none' else x.count(",") + 1)
    conc_num = conc_num.assign(numObstacles=new)

    # number of occurences per diff&numObstacles
    cnt = pd.DataFrame({'count': conc_num.groupby(['physDifficulty', 'numObstacles']).size()}).reset_index()
    numObst = [0]*15
    count = 0
    for a in range(0, len(cnt.index)):
        d = cnt['physDifficulty'][a]
        o = cnt['numObstacles'][a]
        if not o == 0: # Filter out when there is no obstacle at all
            numObst[(d-1)*5+o] = cnt['count'][count]
        count += 1
    return numObst


def plot_difficulty_vs_size_obstacle_scatter_plot():
    plt.figure()
    values = get_number_of_obstacles_per_difficulty()

    for i in [0, 1, 2]:
        li = values[5 * i:5 * i + 5]
        maximum = max(li) if (max(li) > 0) else 1
        values[5 * i:5 * i + 5] = [x / maximum * 2000 for x in li]
    fig, ax = plt.subplots()
    plt.title('Size of obstacle vs difficulty ')
    plt.ylabel('obstacle size')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))  # Only show whole numbers as difficulties

    plt.xlabel('Difficulty')
    x = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
    y = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]

    plt.scatter(x, y, s=values)

    save_plot(plt, 'Logfiles/', 'corr_difficulty_vs_num_obstacles.pdf')


def print_obstacle_information():
    # Idea: Get % of crashes per difficulty level
    # remove first max(hw, cw, gradient_w) seconds (to be consistent with --print_key_numbers_logfiles)
    max_window = max(f_factory.hw, f_factory.cw, f_factory.gradient_w)

    df_list = [df[df['Time'] > max_window] for df in sd.df_list]
    df = pd.concat(df_list, ignore_index=True)

    grouped2 = df.groupby(['physDifficulty', 'Logtype']).size()
    print(grouped2)

    print('\nOn physDifficulty=HIGH, there were ' + str(grouped2[2]+grouped2[1]) + \
          ' obstacles, out of which the user crashed ' + str(grouped2[1]) + \
          ', i.e. ' + str(round(grouped2[1] / grouped2[2], 2) * 100) + '%.')

    print('On physDifficulty=MEDIUM, there were ' + str(grouped2[10] + grouped2[9]) + \
          ' obstacles, out of which the user crashed ' + str(grouped2[9]) + \
          ', i.e. ' + str(round(grouped2[9] / grouped2[10], 2) * 100) + '%.')

    print('On physDifficulty=LOW, there were ' + str(grouped2[6]+grouped2[5]) + \
          ' obstacles, out of which the user crashed ' + str(grouped2[5]) + \
          ', i.e. ' + str(round(grouped2[5] / grouped2[6], 2) * 100) + '%.')


def plot_hr_vs_difficulty_scatter_plot():
    df = pd.concat(sd.df_list, ignore_index=True)
    df_num = transform_df_to_numbers(df)
    df_num.set_index('timedelta', inplace=True)
    resolution = 10
    # resample and take mean over difficulty. This means that a point can now have a difficulty "between"
    # Low/Medium/High, depending on how many seconds out of the resolution seconds it was on which level.
    avg_hr_df_resampled = df_num.resample(str(resolution)+'S').mean()

    plt.title('Difficulty vs. heartrate')
    plt.ylabel('heartrate')
    plt.xlabel('Difficulty')
    x = avg_hr_df_resampled['physDifficulty']
    y = avg_hr_df_resampled['Heartrate']
    plt.scatter(x, y, s=30)

    save_plot(plt, 'Logfiles/', 'corr_difficulty_vs_heartrate.pdf')


def plot_crashes_vs_size_of_obstacle():
    conc_dataframes = pd.concat(sd.df_list, ignore_index=True)
    conc_dataframes = transform_df_to_numbers(conc_dataframes)
    new = conc_dataframes['obstacle'].apply(
        lambda x: 0 if x == 'none' else x.count(",") + 1)  # count number of obstacle parts per obstacle
    conc_num = conc_dataframes.assign(numObstacles=new)
    # [a,b,c,d,e], e..g  #obstacles that had size 0,1,2,3,4 respectively
    num_obstacles_per_size = conc_num.groupby('numObstacles').size().tolist()

    # num_obstacles_per_size.insert(2, 0)  # No obstacles of size 2...
    num_crashes_per_size = [0, 0, 0, 0, 0]

    # For each crash, find corresponding row where we can find the size of the obstacle he crashed into.
    for index, row in conc_num.iterrows():
        if row['Logtype'] == 'EVENT_CRASH':
            sizeOfObstacle = row['numObstacles']
            num_crashes_per_size[sizeOfObstacle] += 1


    percentage_of_crashes = [0 if (x == 0 or y == 0) else x / y *100.0 for x, y in
                             zip(num_crashes_per_size, num_obstacles_per_size)]

    x = [0, 1, 2, 3, 4]
    plt.title('Crash percentage per size of obstacle')
    plt.ylabel('Crashes [%]')
    plt.xlabel('Size of obstacle')
    plt.bar(x, percentage_of_crashes)

    save_plot(plt, 'Logfiles/', 'crashes_percentage_per_size_of_obstacles.pdf')


def crashes_per_obstacle_arrangement():
    import collections

    df = pd.concat(sd.df_list, ignore_index=True)
    conc_dataframes = transform_df_to_numbers(df)

    # For each obstacle-arrangement, make a dictionary-entry with a list [#occurences, #crashes]
    obst_dict = {}
    # For each crash, find corresponding row where we can find the obstacle he crashed into.
    for index, row in conc_dataframes.iterrows():
        if row['Logtype'] == 'EVENT_CRASH':
            obstacle = row['obstacle']
            if obstacle in obst_dict:
                obst_dict[obstacle] = [obst_dict[obstacle][0] + 1, obst_dict[obstacle][1] + 1]
            else:
                obst_dict[obstacle] = [1, 1]
        if row['Logtype'] == 'EVENT_OBSTACLE': # TODO: Check if this is correct
            obstacle = row['obstacle']
            if obstacle in obst_dict:
                obst_dict[obstacle] = [obst_dict[obstacle][0] + 1, obst_dict[obstacle][1]]
            else:
                obst_dict[obstacle] = [1, 0]

    obst_dict = collections.OrderedDict(sorted(obst_dict.items(), key=lambda s: len(s[0])))
    index = obst_dict.keys()
    columns = ["#Occurences", "#Crashes", "Crashes in %"]
    data = np.zeros(shape=(len(index), 3))
    count = 0
    for key, value in obst_dict.items():
        data[count][0] = value[0]  # #Occurences
        data[count][1] = value[1]  # #Crashes
        data[count][2] = value[1] / value[0] * 100
        count += 1

    df = pd.DataFrame(data, index=index, columns=columns)
    plt.xticks(rotation=90)
    plt.title('Crashes vs. obstacle arrangement')
    plt.ylabel('Crashes at this arrangement [%]')
    plt.xlabel('Obstacle arrangement')
    plt.bar(df.index, df['Crashes in %'])

    save_plot(plt, 'Logfiles/', 'crashes_per_obstacle_arrangement.pdf')

