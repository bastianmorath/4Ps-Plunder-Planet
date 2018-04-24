import pandas as pd
import numpy as np
from scipy.stats import truncnorm

import setup
import globals as gl
import factory

num_dataframes = 3  # How many dataframes should be created?
length_dataframe = 500  # How many rows should one dataframe have?
mean_hr = 123.9  # Mean of normal distribution of heartrate
std_hr = 16.8  # std of normal distribution of heartrate

test_data_case = 1


def init_with_testdata():
    # map the inputs to the function blocks
    options = {0: init_with_testdata_events_const_hr_const, # Simplest: alternates between continuous, crash and not crash, setting hr = 30 or 30 if crash after
               1: init_with_testdata_events_random_hr_const, # Random events, depending on crash or not sets hr= 10 or 100 in event before crash
               2: init_with_testdata_events_random_hr_continuous
               }
    options[test_data_case]()


''' Inits with very simple synthesized data to check model performence
    Alternates between heartrate 20 and 30 and crash/not crash
'''


def init_with_testdata_events_const_hr_const():

    for i in range(0, num_dataframes):
        times = range(0, 400)
        logtypes = ['CONTINUOUS', 'EVENT_OBSTACLE', 'CONTINUOUS', 'EVENT_CRASH'] * 100
        heartrates = [20, 20, 30, 20] * 100
        points = [20, 20, 30, 20] * 100
        timedeltas = [pd.to_timedelta(t, unit='S') for t in times]

        dataframe = pd.DataFrame(data={'Time': times, 'Points': points, 'Logtype': logtypes, 'Heartrate': heartrates,
                                       'timedelta': timedeltas})
        # plot_hr(dataframe, i)

        gl.df_list.append(dataframe)

    setup.normalize_heartrate()
    gl.obstacle_df_list = factory.get_obstacle_times_with_success()


''' Inits with very simple synthesized data to check model performence
    Random events, but heartrate either 1 or 10 depending on crash (with noise)
'''


def init_with_testdata_events_random_hr_const():

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
                                       'timedelta': timedeltas})
        # plot_hr(dataframe, i)
        gl.df_list.append(dataframe)

    setup.normalize_heartrate()
    gl.obstacle_df_list = factory.get_obstacle_times_with_success()


''' Inits the dataframes not from the logfiles, but with synthesized data
    Times: from 0 to length_dataframe, one every second with noise
    logtypes: Randomly choosen; if EVENT_CRASH, then add EVENT_OBSTACLE in the next one!
'''


def init_with_testdata_events_random_hr_continuous():

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
                                       'timedelta': timedeltas})
        # plot_hr(dataframe, i)
        gl.df_list.append(dataframe)

    setup.normalize_heartrate()
    gl.obstacle_df_list = factory.get_obstacle_times_with_success()


'''Returns a value from a normal distribution, truncated to a boundary'''


def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


def plot_hr(dataframe, i):
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots()
    fig.suptitle('heartrate')

    # Plot mean_hr
    ax1.plot(dataframe['Time'], dataframe['Heartrate'])
    ax1.set_xlabel('Playing time [s]')
    ax1.set_ylabel('Heartrate')

    plt.savefig(gl.working_directory_path + '/Plots/Heartrate_Testdata/heartrate_testdata_' + str(i) + '.pdf')