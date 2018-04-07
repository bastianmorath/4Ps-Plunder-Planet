import pandas as pd
import numpy as np
from operator import add
from scipy.stats import truncnorm
import itertools
import globals as gl
import features_factory as f_factory

num_dataframes = 5 # How many dataframes should be created?
length_dataframe = 50  # How many rows should one dataframe have?
mean_hr = 123.9  # Mean of normal distribution of heartrate
std_hr = 16.8  # std of normal distribution of heartrate

''' Inits the dataframes not from the logfiles, but with synthesized data
'''


def init_with_testdata():
    crashes = []

    for i in range(0, num_dataframes):
        times = [0]
        logtypes = ['CONTINUOUS']
        heartrates = [mean_hr]
        timedeltas = [pd.to_timedelta(0, unit='S')]

        X = get_truncated_normal(mean=0, sd=0.2, low=0, upp=0.2)
        noise = X.rvs(length_dataframe)
        last_event_was_a_crash = False
        crashes.append(False)
        hr = mean_hr
        for j in range(0, length_dataframe-1):
            types = ['CONTINUOUS', 'EVENT_OBSTACLE', 'EVENT_CRASH']
            if last_event_was_a_crash:
                last_event_was_a_crash = False
                log = np.random.choice(types, 1, [0.7, 0.3, 0])
                logtypes.append(log)
                hr = hr + np.random.normal(2, 1)
                heartrates.append(hr)
                crashes.append(True)
            else:
                log = np.random.choice(types, 1, [0.7, 0.2, 0.1])
                logtypes.append(log)
                hr = hr + np.random.normal(-1, 1)
                heartrates.append(hr)
                if log == 'EVENT_CRASH':
                    crashes.append(True)
                    last_event_was_a_crash = True
                else:
                    crashes.append(False)
                    last_event_was_a_crash = False

            times.append(j + noise[j])
            timedeltas.append(pd.to_timedelta(times[j], unit='S'))

        dataframe = pd.DataFrame(data={'Time': times, 'Logtype': logtypes, 'Heartrate': heartrates,
                                      'timedelta': timedeltas})

        gl.df_list.append(dataframe)

    gl.df_without_features = pd.concat(gl.df_list, ignore_index=True)
    gl.df = f_factory.get_df_with_feature_columns()

    gl.obstacle_df = pd.DataFrame(
        {'Time': gl.df_without_features['Time'],
         'crash': crashes
        }
    )


'''Returns a value from a normal distribution, truncated to a boundary'''


def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)