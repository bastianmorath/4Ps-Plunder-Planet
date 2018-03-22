#factory_model.py



from __future__ import division # s.t. division uses float result
'''
    Adds a column at each timestamp that indicates the %Crashes 
    the user did in the last 'window_size' seconds
'''
def get_crashes_column(df, crash_window):
    def df_from_to(_from, _to):
        mask = (_from<df['Time']) & (df['Time']<=_to)
        return df[mask]
        
    def computeCrashes(row):
        last_x_seconds_df = df_from_to(max(0,row['Time']-crash_window), row['Time'])
        num_obstacles = len(last_x_seconds_df[last_x_seconds_df['Logtype']=='EVENT_OBSTACLE'].index)
        num_crashes = len(last_x_seconds_df[last_x_seconds_df['Logtype']=='EVENT_CRASH'].index)
        return (num_crashes/num_obstacles *100 if num_crashes<num_obstacles else 100) if num_obstacles!=0 else 0

    return  df[['Time', 'Logtype']].apply(computeCrashes,axis=1)


'''
    Adds a column at each timestamp that indicates the mean
    heartrate over the last 'heartrate_window' seconds
'''
def get_mean_heartrate_column(df, heartrate_window):

    def df_from_to(_from, _to):
        mask = (_from<df['Time']) & (df['Time']<=_to)
        return df[mask]
    def compute_mean_hr(row):
            if row['Time'] > heartrate_window:
                last_x_seconds_df = df_from_to(row['Time'] - heartrate_window, row['Time'])

                return last_x_seconds_df[last_x_seconds_df['Heartrate']!=-1]['Heartrate'].mean()
            else:
                return 0

    return df[['Time', 'Heartrate']].apply(compute_mean_hr,axis=1)



'''Resamples a dataframe with a sampling frquency of 'resolution'
    -> Smoothes the plots
'''
def resample_dataframe(df, resolution):
    df.set_index('timedelta', inplace=True) #set timedelta as new index
    return df.resample(str(resolution)+'S').mean() # Resample series'