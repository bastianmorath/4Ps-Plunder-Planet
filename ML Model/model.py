'''This is a first implementation of a ML model to predict the performance
    of the player.
    In particular, the model will predict whether or not the user is going to
    crash into the next obstacle.

    As features, we use:
        - %Crashes in last x seconds
        - mean HR
        - Max/Min HR ratio
        - Crystals (later...)
        - %Points change
    
    SVM as the binary classifier and 10-fold Cross-Validation is used
'''

import globals_model as gl
import factory_model as factory

crash_window = 30  # Over how many preceeding seconds should %crashes be calculated?
heartrate_window = 50  # Over how many preceeding seconds should the heartrate be averaged?


''' Get data and create feature matrix and labels
    Column 0: Id/Time
    Column 1: %Crashes in last x seconds
    Column 2: mean heartrate over last y seconds
'''
df = gl.init(crash_window, heartrate_window)
factory.plot(df)
df_obstacle = factory.get_obstacle_times_with_success()
print(df)
X = df[['%crashes', 'mean_hr']]

''' Train SVM model
'''

'''Test model with 10-fold Cross-Validation
'''