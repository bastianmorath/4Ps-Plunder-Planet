import pandas as pd
import numpy as np
from operator import add
from scipy.stats import truncnorm
import itertools
import globals as gl
import features_factory as f_factory

num_dataframes = 10 # How many dataframes should be created?
length_dataframe = 100  # How many rows should one dataframe have?
mean_hr = 123.9  # Mean of normal distribution of heartrate
std_hr = 16.8  # std of normal distribution of heartrate

''' Inits the dataframes not from the logfiles, but with synthesized data
'''


def init_with_testdata():

    for i in range(0, num_dataframes):
        X = get_truncated_normal(mean=0, sd=0.2, low=0, upp=0.3)
        noise = X.rvs(length_dataframe)
        times = list(map(add,
                         range(0, length_dataframe),
                         noise))
        # EVENT_CRASH has to be followed by an EVENT_OBSTACLE
        # TODO: those two need to happen withing  milliseconds or so...
        types = [['EVENT_OBSTACLE', 'EVENT_OBSTACLE'], ['EVENT_CRASH', 'EVENT_OBSTACLE']]
        random_choices = np.random.choice([0, 1], int(length_dataframe/2), [0.9, 0.1])
        logtypes = list(itertools.chain.from_iterable([types[i] for i in random_choices]))

        heartrates = np.random.normal(mean_hr, std_hr, length_dataframe)
        timedeltas = pd.to_timedelta(times, unit='S')

        dataframe = pd.DataFrame(data={'Time': times, 'Logtype': logtypes,
                                        'Heartrate': heartrates, 'timedelta': timedeltas})
        print(dataframe)
        gl.df_list.append(dataframe)

    gl.df_without_features = pd.concat(gl.df_list, ignore_index=True)
    gl.df = f_factory.get_df_with_feature_columns()
    gl.obstacle_df =pd.DataFrame(
        {'Time': [1,2,3,4,5,6] * num_dataframes,
         'crash': [0, 1, 1, 0, 1, 1] * num_dataframes
        }
    )
    print(gl.obstacle_df)


'''Returns a value from a normal distribution, truncated to a boundary'''


def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)