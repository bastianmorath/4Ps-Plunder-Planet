import pandas as pd
import numpy as np
from scipy.stats import truncnorm

import setup
import globals as gl
import features_factory as f_factory

num_dataframes = 3  # How many dataframes should be created?
length_dataframe = 400  # How many rows should one dataframe have?
mean_hr = 123.9  # Mean of normal distribution of heartrate
std_hr = 16.8  # std of normal distribution of heartrate

''' Inits the dataframes not from the logfiles, but with synthesized data
    Times: from 0 to length_dataframe, one every second with noise
    logtypes: Randomly choosen; if EVENT_CRASH, then add EVENT_OBSTACLE in the next one!
'''


def init_with_testdata():
    crashes = []
    # Find distribution of Logtypes
    # c = Counter(gl.df_without_features['Logtype'])
    # print([(i, c[i] / len(gl.df_without_features['Logtype']) * 100.0) for i in c])
    for i in range(0, num_dataframes):
        times = [0]
        logtypes = ['CONTINUOUS']
        heartrates = [mean_hr]
        timedeltas = [pd.to_timedelta(0, unit='S')]

        distribution = get_truncated_normal(mean=0, sd=0.2, low=0.02, upp=0.2)
        noise = distribution.rvs(length_dataframe)
        last_event_was_a_crash = False
        crashes.append(False)
        hr = mean_hr
        for j in range(0, length_dataframe-1):
            types = ['CONTINUOUS', 'EVENT_OBSTACLE', 'EVENT_CRASH', 'EVENT_PICKUP']
            if last_event_was_a_crash:
                last_event_was_a_crash = False
                log = 'EVENT_OBSTACLE'
                logtypes.append(log)
                hr = hr + np.random.normal(4, 2)
                heartrates.append(hr)
                crashes.append(True)
                times.append(times[-1] + noise[j])  # Crash: Add EVENT_OBSTACLE right after EVENT_CRASH
            else:
                log = np.random.choice(types, p=[0.6, 0.27, 0.1, 0.03])
                logtypes.append(log)
                hr = hr + np.random.normal(-0.3, 0.7)
                heartrates.append(hr)
                if log == 'EVENT_CRASH':
                    crashes.append(True)
                    last_event_was_a_crash = True
                else:
                    crashes.append(False)
                    last_event_was_a_crash = False
                times.append(times[-1] + 1 + noise[j])

            timedeltas.append(pd.to_timedelta(times[j+1], unit='S'))

        dataframe = pd.DataFrame(data={'Time': times, 'Logtype': logtypes, 'Heartrate': heartrates,
                                       'timedelta': timedeltas})
        plot_hr(dataframe, i)
        gl.df_list.append(dataframe)

    setup.normalize_heartrate()
    gl.df_without_features = pd.concat(gl.df_list, ignore_index=True)

    gl.df = f_factory.get_df_with_feature_columns()

    gl.obstacle_df = pd.DataFrame({'Time': gl.df_without_features['Time'], 'crash': crashes})


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

    plt.savefig(gl.working_directory_path + '/Plots/heartrate_testdata' + str(i) + '.pdf')