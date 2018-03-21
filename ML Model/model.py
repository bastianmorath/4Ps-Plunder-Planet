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

crash_window = 30 # Over how many preceeding seconds should %crashes be calculated?

''' Get data and create feature matrix and labels
    Column 0: Id/Time
    Column 1: %Crashes in last x seconds
    Column 2: mean heartrate over last y seconds
'''
gl.init()
df = gl.main_df
''' Train SVM model
'''

'''Test model with 10-fold Cross-Validation
'''