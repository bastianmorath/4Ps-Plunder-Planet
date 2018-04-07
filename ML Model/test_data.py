from datetime import timedelta
import pandas as pd

import globals as gl
import features_factory as f_factory

num_dataframes = 10 # How many dataframes should be created?
mean_hr = 123.9  # Mean of normal distribution of heartrate
std_hr = 16.8  # std of normal distribution of heartrate

''' Inits the dataframes not from the logfiles, but with synthesized data
'''


def init_with_testdata():

    for i in range(0, num_dataframes):
        times = [1,2,3,4,5,6]
        logtypes = ['EVENT_OBSTACLE', 'EVENT_CRASH',  'EVENT_CRASH', 'EVENT_OBSTACLE', 'EVENT_CRASH', 'EVENT_OBSTACLE']
        heartrates = [378, 155, 77, 973, 973, 973]
        timedeltas = [timedelta(seconds =1), timedelta(seconds =2), timedelta(seconds=3), timedelta(seconds=4),
                      timedelta(seconds =5), timedelta(seconds =6)]
        dataframe = pd.DataFrame(data={'Time': times, 'Logtype': logtypes,
                                         'Heartrate': heartrates, 'timedelta': timedeltas})
        gl.df_list.append(dataframe)

    gl.df_without_features = pd.concat(gl.df_list, ignore_index=True)
    gl.df = f_factory.get_df_with_feature_columns()
    gl.obstacle_df =pd.DataFrame(
        {'Time': [1,2,3,4,5,6] * num_dataframes,
         'crash': [0, 1, 1, 0, 1, 1] * num_dataframes
        }
    )
    print(gl.obstacle_df)