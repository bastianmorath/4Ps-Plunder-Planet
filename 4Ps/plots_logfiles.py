"""
This module is responsible for generating plots that are involved with statistics about logfiles

"""

from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import collections

from matplotlib.ticker import MaxNLocator

import features_factory as f_factory
import plots_helpers as hp
import setup_dataframes as sd

"""
Plots concerned with logfiles

"""

# TODO: Use helper barplot for all plots..


def generate_plots_about_logfiles():
    _plot_heartrate_change()
    _plot_heartrate_and_events()
    _crashes_per_obstacle_arrangement()
    _plot_crashes_vs_size_of_obstacle()
    _plot_hr_vs_difficulty_scatter_plot()
    _print_obstacle_information()
    _plot_difficulty_vs_size_obstacle_scatter_plot()
    _plot_hr_or_points_and_difficulty('Heartrate')
    _plot_hr_or_points_and_difficulty('Points')

    _plot_mean_and_std_hr_boxplot()
    _plot_hr_of_dataframes()
    _plot_average_hr_over_all_logfiles()
    _plot_heartrate_histogram()


def _plot_heartrate_and_events():
    """
    Plots the heartrate of each logfile, together with the crashes, Shieldtutorials and Brokenship events

    """

    print("Plotting heartrate and events...")
    # resolution = 3

    for idx, df in enumerate(sd.df_list):
        if not (df['Heartrate'] == -1).all():

            # df_num_resampled = hp.resample_dataframe(df, resolution)
            df_num_resampled = df
            # Plot Heartrate
            _, ax1 = plt.subplots()
            ax1.plot(df_num_resampled['Time'], df_num_resampled['Heartrate'], hp.blue_color, linewidth=1.0)
            ax1.set_xlabel('Playing time (s)')
            ax1.set_ylabel('Heartrate', color=hp.blue_color)
            ax1.tick_params('y', colors=hp.blue_color)

            # Plot crashes
            times_crashes = [row['Time'] for _, row in sd.obstacle_df_list[idx].iterrows() if row['crash']]
            heartrate_crashes = [df[df['Time'] == row['Time']].iloc[0]['Heartrate']
                                 for _, row in sd.obstacle_df_list[idx].iterrows() if row['crash']]
            plt.scatter(times_crashes, heartrate_crashes, c='r', marker='.', label='crash')

            # Plot Brokenships
            times_repairing = [row['Time'] for _, row in df.iterrows() if row['Gamemode'] == 'BROKENSHIP']
            hr_max = df['Heartrate'].max()
            hr_min = df['Heartrate'].min()
            for xc in times_repairing:
                plt.vlines(x=xc, ymin=hr_min, ymax=hr_max+0.2, color='y', linewidth=1, label='ship broken')

            # Plot Shieldtutorial
            times_repairing = [row['Time'] for _, row in df.iterrows() if row['Gamemode'] == 'SHIELDTUTORIAL']
            hr_max = df['Heartrate'].max()
            hr_min = df['Heartrate'].min()
            for xc in times_repairing:
                plt.vlines(x=xc, ymin=hr_min, ymax=hr_max + 0.2, color='g', linewidth=1, label='Shield tutorial')

            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))  # Otherwise we'd have one label for each vline
            plt.legend(by_label.values(), by_label.keys())

            filename = 'hr_and_events_' + sd.names_logfiles[idx] + '.pdf'
            hp.save_plot(plt, 'Logfiles/Heartrate_Events/', filename)


def _plot_hr_of_dataframes():
    """
    Generates one heartrate plot for each dataframes (Used to compare normalized hr to original hr)
        Only works for real data at the moment, because of name_logfile not existing if synthesized_data...

    """

    print("Plotting heartrate of dataframes over time...")
    resolution = 5
    for idx, df in enumerate(sd.df_list):
        if not (df['Heartrate'] == -1).all():
            df_num_resampled = hp.resample_dataframe(df, resolution)
            # Plot Heartrate
            _, ax1 = plt.subplots()
            ax1.plot(df_num_resampled['Time'], df_num_resampled['Heartrate'], hp.blue_color)
            ax1.set_xlabel('Playing time (s)')
            ax1.set_ylabel('Heartrate', color=hp.blue_color)
            ax1.tick_params('y', colors=hp.blue_color)

            filename = 'hr_' + sd.names_logfiles[idx] + '.pdf'
            hp.save_plot(plt, 'Logfiles/Heartrate/', filename)


def _plot_heartrate_histogram():
    """
    Plots a histogram of  heartrate data accumulated over all logfiles

    """

    print("Plotting histogram of heartrate of accumulated logfiles...")

    _, ax = plt.subplots()
    df = pd.concat(sd.df_list, ignore_index=True)
    df = df[df['Heartrate'] != -1]['Heartrate']
    plt.hist(df)
    plt.title('Histogram of HR: $\mu=%.3f$, $\sigma=%.3f$'
              % (float(np.mean(df)), float(np.std(df))))
    ax.yaxis.grid(True, zorder=0, color='grey', linewidth=0.3)
    ax.set_axisbelow(True)
    [i.set_linewidth(0.3) for i in ax.spines.values()]
    hp.save_plot(plt, 'Logfiles/', 'heartrate_distribution_all_logfiles.pdf')


def _plot_average_hr_over_all_logfiles():
    """
    Plots average heartrate over all logfiles

    """

    plt.subplots()
    plt.ylabel('Heartrate (bpm)')
    plt.xlabel('Playing time (s)')
    plt.title('Average Heartrate across all users')

    conc_dataframes = pd.concat(sd.df_list, ignore_index=True)
    time_df = conc_dataframes.groupby(['userID', 'logID'])['Time'].max()

    min_time = time_df.min()

    conc_dataframes = conc_dataframes[
        conc_dataframes['Time'] < min_time]  # Cut all dataframes to the same minimal length

    df_copy = conc_dataframes.copy()  # to prevent SettingWithCopyWarning
    avg_hr_df = df_copy.groupby(['timedelta'])[['timedelta', 'Heartrate']].mean()  # Take mean over all logfiles
    avg_hr_df.reset_index(inplace=True)
    avg_hr_df_resampled = hp.resample_dataframe(avg_hr_df, 10)

    plt.plot(avg_hr_df_resampled['Time'], avg_hr_df_resampled['Heartrate'])
    hp.save_plot(plt, 'Logfiles/', 'average_heartrate.pdf')


def _plot_mean_and_std_hr_boxplot():
    """
    Plots mean and std bpm per user in a box-chart

    """

    conc_dataframes = pd.concat(sd.df_list, ignore_index=True)

    df2 = conc_dataframes.pivot(columns=conc_dataframes.columns[1], index=conc_dataframes.index)
    df2.columns = df2.columns.droplevel()
    conc_dataframes[['Heartrate', 'userID']].boxplot(by='userID', grid=False, sym='r+')
    plt.ylabel('Heartrate (bpm)')
    plt.title('')
    hp.save_plot(plt, 'Logfiles/', 'mean_heartrate_boxplot.pdf')


def _plot_heartrate_change():
    """
    Plot Heartrate change

    """

    bpm_changes_max = []  # Stores max. absolute change in HR per logfile
    bpm_changes_rel = []  # Stores max. percentage change in HR per logfile

    X = []
    for idx, df in enumerate(sd.df_list):
        if not (df['Heartrate'] == -1).all():
            X.append(idx)
            # new = df.set_index('timedelta', inplace=False)
            resampled = hp.resample_dataframe(df, 1)
            percentage_change = np.diff(resampled['Heartrate']) / resampled['Heartrate'][:-1] * 100.
            x = percentage_change[np.logical_not(np.isnan(percentage_change))]
            bpm_changes_max.append(x.max())
            bpm_changes_rel.append(x)

    plt.ylabel('#Times HR changed')
    plt.xlabel('Change in Heartrate [%]')

    for idx, l in enumerate(bpm_changes_rel):  # Histogram per user
        name = str(sd.names_logfiles[idx])
        plt.figure()
        plt.title('Heartrate change for plot ' + name)
        plt.hist(l, color=hp.blue_color)
        hp.save_plot(plt, 'Logfiles/Abs Heartrate Changes/', 'heartrate_change_percentage_' + name + '.pdf')

    fig, ax = plt.subplots()

    plt.title('Maximal heartrate change')
    plt.ylabel('Max heartrate change [%]')
    plt.xlabel('Logfile')
    plt.bar([x for x in X], bpm_changes_max, color=hp.blue_color, width=0.25)

    ax.yaxis.grid(True, zorder=0, color='grey',  linewidth=0.3)
    ax.set_axisbelow(True)
    [i.set_linewidth(0.3) for i in ax.spines.values()]

    hp.save_plot(plt, 'Logfiles/', 'heartrate_change_abs.pdf')


def _transform_df_to_numbers(df):
    """
    Subsitutes difficulties with numbers to work with them in a better way, from 1 to 3

    :param df: Dataframe to transform to numbers to
    :return transformed datafarme

    """

    mapping = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'undef': -1}
    df = df.replace({'physDifficulty': mapping, 'psyStress': mapping, 'psyDifficulty': mapping})

    for col in ['physDifficulty', 'psyStress', 'psyDifficulty']:
        df[col] = df[col].astype('int64')
    return df


def _plot_hr_or_points_and_difficulty(to_compare):
    """
    Plots heartrate or points together with the difficulty in a line plot

    :param to_compare: 'Heartrate' or 'Points'

    """

    resolution = 10  # resample every x seconds -> the bigger, the smoother
    for idx, df in enumerate(sd.df_list):
        df = _transform_df_to_numbers(df)
        if not (df['Heartrate'] == -1).all():
            df_num_resampled = hp.resample_dataframe(df, resolution)
            # Plot Heartrate
            fig, ax1 = plt.subplots()
            ax1.plot(df_num_resampled['Time'], df_num_resampled[to_compare], hp.blue_color)
            ax1.set_xlabel('Playing time (s)')
            ax1.set_ylabel(to_compare, color=hp.blue_color)
            ax1.tick_params('y', colors=hp.blue_color)

            # Plot Difficulty
            ax2 = ax1.twinx()
            ax2.plot(df_num_resampled['Time'], df_num_resampled['physDifficulty'], hp.green_color)
            ax2.set_ylabel('physDifficulty', color=hp.green_color)
            ax2.tick_params('y', colors=hp.green_color)
            ax2.yaxis.set_major_locator(MaxNLocator(integer=True))  # Only show whole numbers as difficulties
            plt.title('Difficulty and ' + to_compare + ' for user ' + sd.names_logfiles[idx])
            hp.save_plot(plt, 'Logfiles/', to_compare + ' Difficulty Corr/' + to_compare + '_difficulty_' +
                         str(sd.names_logfiles[idx]) + '.pdf')


'''Returns a list which says how many times the obstacle has size {0,1,2,3,4} for each difficulty level in the form
    [0, 0, 0, 0, 0, 0, 143, 0, 581, 25, 0, 2659, 0, 299, 5589]
    # occurences where obstacle had size 0 in Diff=LOW, ... , 
    # occurences where obstacle had size 4 in Diff=LOW,
    # occurences where obstacle had size 0 in Diff=MEDIUM,...]
'''


def _get_number_of_obstacles_per_difficulty():
    conc_dataframes = pd.concat(sd.df_list, ignore_index=True)
    conc_num = _transform_df_to_numbers(conc_dataframes)  # Transform Difficulties into integers
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
        if not o == 0:  # Filter out when there is no obstacle at all
            numObst[(d-1)*5+o] = cnt['count'][count]
        count += 1
    return numObst


def _plot_difficulty_vs_size_obstacle_scatter_plot():
    """
    PLots the difficulty of the level and the size of the obstacle at a given difficulty in a scatter plot

    """

    plt.figure()
    values = _get_number_of_obstacles_per_difficulty()

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

    hp.save_plot(plt, 'Logfiles/', 'corr_difficulty_vs_num_obstacles.pdf')


def _print_obstacle_information():
    """
    Prints for each difficulty level the number of obstacles and how many the user crashed into

    """

    # TODO: As a plot
    max_window = max(f_factory.hw, f_factory.cw, f_factory.gradient_w)

    df_list = [df[df['Time'] > max_window] for df in sd.df_list]
    df = pd.concat(df_list, ignore_index=True)

    grouped2 = df.groupby(['physDifficulty', 'Logtype']).size()

    print('\nOn physDifficulty=HIGH, there were ' + str(grouped2[2]+grouped2[1]) +
          ' obstacles, out of which the user crashed ' + str(grouped2[1]) +
          ', i.e. ' + str(round(grouped2[1] / grouped2[2], 2) * 100) + '%.')

    print('On physDifficulty=MEDIUM, there were ' + str(grouped2[10] + grouped2[9]) +
          ' obstacles, out of which the user crashed ' + str(grouped2[9]) +
          ', i.e. ' + str(round(grouped2[9] / grouped2[10], 2) * 100) + '%.')

    print('On physDifficulty=LOW, there were ' + str(grouped2[6]+grouped2[5]) +
          ' obstacles, out of which the user crashed ' + str(grouped2[5]) +
          ', i.e. ' + str(round(grouped2[5] / grouped2[6], 2) * 100) + '%.')


def _plot_hr_vs_difficulty_scatter_plot():
    """
    PLots the heartrate vs the difficulty in a scatter plot

    """

    df = pd.concat(sd.df_list, ignore_index=True)
    df_num = _transform_df_to_numbers(df)
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

    hp.save_plot(plt, 'Logfiles/', 'corr_difficulty_vs_heartrate.pdf')


def _plot_crashes_vs_size_of_obstacle():
    """
    Plots the percentage of crashes depending on the size of the obstacle

    """

    conc_dataframes = pd.concat(sd.df_list, ignore_index=True)
    conc_dataframes = _transform_df_to_numbers(conc_dataframes)
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

    percentage_of_crashes = [0 if (x == 0 or y == 0) else x / y * 100.0 for x, y in
                             zip(num_crashes_per_size, num_obstacles_per_size)]

    x = [0, 1, 2, 3, 4]
    plt.title('Crash percentage per size of obstacle')
    plt.ylabel('Crashes [%]')
    plt.xlabel('Size of obstacle')
    plt.bar(x, percentage_of_crashes)

    hp.save_plot(plt, 'Logfiles/', 'crashes_percentage_per_size_of_obstacles.pdf')


def _crashes_per_obstacle_arrangement():
    """
    Plots the percentage of crashes vs the obstacle arrangement

    """

    df = pd.concat(sd.df_list, ignore_index=True)
    conc_dataframes = _transform_df_to_numbers(df)

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
        if row['Logtype'] == 'EVENT_OBSTACLE':
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

    fix, ax = plt.subplots()
    ax.yaxis.grid(True, zorder=0, color='grey',  linewidth=0.3)
    ax.set_axisbelow(True)
    [i.set_linewidth(0.3) for i in ax.spines.values()]
    plt.xticks(rotation=90)
    plt.title('Crashes vs. obstacle arrangement')
    plt.ylabel('Crashes at this arrangement [%]')
    plt.xlabel('Obstacle arrangement')
    plt.bar(df.index, df['Crashes in %'])

    hp.save_plot(plt, 'Logfiles/', 'crashes_per_obstacle_arrangement.pdf')
