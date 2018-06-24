""" This module generates synthesized data to test the pipeline of the classifier.

    Conf:
        -if make_plots=True, the generated heartrate is plotted
"""


import pandas as pd
import numpy as np
from scipy.stats import truncnorm

import setup_dataframes as sd
import refactoring_logfiles
import matplotlib.pyplot as plt

import setup_dataframes
import plots

make_plots = False

num_dataframes = 3  # How many dataframes should be created?
length_dataframe = 1000  # How many rows should one dataframe have?
mean_hr = 123.9  # Mean of normal distribution of heartrate
std_hr = 16.8  # std of normal distribution of heartrate

test_data_enabled = False


def init_with_testdata_events_const_hr_const():
    """Inits with very simple synthesized data to check model performance
        Alternates between heartrate 20 and 30 and crash/not crash

    """

    for i in range(0, num_dataframes):
        times = range(0, 400)
        logtypes = ['CONTINUOUS', 'EVENT_OBSTACLE', 'CONTINUOUS', 'EVENT_CRASH'] * 100
        heartrates = [20, 20, 30, 20] * 100
        points = [20, 20, 30, 20] * 100
        timedeltas = [pd.to_timedelta(t, unit='S') for t in times]

        dataframe = pd.DataFrame(data={'Time': times, 'Points': points, 'Logtype': logtypes, 'Heartrate': heartrates,
                                       'timedelta': timedeltas, 'userID': [i]*length_dataframe,
                                       'logID': [0]*length_dataframe})
        if make_plots:
            plot_hr(dataframe, i)

        sd.df_list.append(dataframe)

    sd.obstacle_df_list = setup_dataframes.get_obstacle_times_with_success()


def init_with_testdata_events_random_hr_const():
    """Inits with very simple synthesized data to check model performance
        Random events, but heartrate is either 1 or 10 depending on crash (with noise)

    """

    for i in range(0, num_dataframes):
        times = []
        logtypes = []
        heartrates = []
        timedeltas = []
        points = []

        distribution = get_truncated_normal(mean=0, sd=0.2, low=0.02, upp=0.2)
        noise = distribution.rvs(length_dataframe)

        types = ['CONTINUOUS', 'EVENT_OBSTACLE', 'EVENT_CRASH', 'EVENT_PICKUP']
        current_event = ''
        next_event = 'CONTINUOUS'
        for j in range(0, length_dataframe):

            if next_event == 'EVENT_CRASH':
                hr = np.random.normal(7, 1)
                heartrates.append(hr)
            else:
                hr = np.random.normal(1, 1)
                heartrates.append(hr)

            points.append(np.random.normal(7, 1))
            times.append(j + noise[j])
            logtypes.append(current_event)

            current_event = next_event
            next_event = np.random.choice(types, p=[0.6, 0.27, 0.1, 0.03])

            timedeltas.append(pd.to_timedelta(times[j], unit='S'))

        dataframe = pd.DataFrame(data={'Time': times, 'Points': points, 'Logtype': logtypes, 'Heartrate': heartrates,
                                       'timedelta': timedeltas, 'userID': [i]*length_dataframe,
                                       'logID': [0]*length_dataframe})
        if make_plots:
            plot_hr(dataframe, i)

        sd.df_list.append(dataframe)

    sd.obstacle_df_list = setup_dataframes.get_obstacle_times_with_success()


def init_with_testdata_events_random_hr_continuous():
    """Inits the dataframes not from the logfiles, but with synthesized data
        Times: from 0 to length_dataframe, one every second with noise
        logtypes: Randomly choosen; if EVENT_CRASH, then add EVENT_OBSTACLE in the next one!

    """
    for i in range(0, num_dataframes):
        times = []
        logtypes = []
        heartrates = []
        timedeltas = []
        points = []

        distribution = get_truncated_normal(mean=0, sd=0.2, low=0.02, upp=0.2)
        noise = distribution.rvs(length_dataframe)

        types = ['CONTINUOUS', 'EVENT_OBSTACLE', 'EVENT_CRASH', 'EVENT_PICKUP']
        current_event = ''
        next_event = 'CONTINUOUS'
        hr = mean_hr
        for j in range(0, length_dataframe):

            if next_event == 'EVENT_CRASH':
                hr = hr + 20
                heartrates.append(hr)
            else:
                hr = hr - 10
                heartrates.append(hr)

            points.append(np.random.normal(7, 1))
            times.append(j + noise[j])
            logtypes.append(current_event)

            current_event = next_event
            next_event = np.random.choice(types, p=[0.6, 0.27, 0.1, 0.03])

            timedeltas.append(pd.to_timedelta(times[j], unit='S'))

        dataframe = pd.DataFrame(data={'Time': times, 'Points': points, 'Logtype': logtypes, 'Heartrate': heartrates,
                                       'timedelta': timedeltas, 'userID': [i]*length_dataframe,
                                       'logID': [0]*length_dataframe})
        if make_plots:
            plot_hr(dataframe, i)

        sd.df_list.append(dataframe)

    sd.obstacle_df_list = setup_dataframes.get_obstacle_times_with_success()


def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    """Returns a value from a normal distribution, truncated to a boundary

    :return: Random value from normal distribution specified by arguments

    """

    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


def plot_hr(dataframe, i):
    """Plots the heartrate of the dataframe

    :param dataframe: Dataframe from which the heartrate should be plotted
    :param i: id to differentiate plots

    """
    fig, ax1 = plt.subplots()
    fig.suptitle('heartrate')

    # Plot mean_hr
    ax1.plot(dataframe['Time'], dataframe['Heartrate'])
    ax1.set_xlabel('Playing time [s]')
    ax1.set_ylabel('Heartrate')

    plots.save_plot(plt, 'Logfiles/test_data/', 'heartrate_testdata_' + str(i) + '.pdf')
