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
import matplotlib.pyplot as plt

crash_window = 30 # Over how many preceeding seconds should %crashes be calculated?
heartrate_window = 30 # Over how many preceeding seconds should the heartrate be averaged?

green_color = '#AEBD38'
blue_color = '#68829E'


''' Get data and create feature matrix and labels
    Column 0: Id/Time
    Column 1: %Crashes in last x seconds
    Column 2: mean heartrate over last y seconds
'''
df = gl.init(crash_window, heartrate_window)

fig, ax1 = plt.subplots()
fig.suptitle('%Crashes and mean_hr over last x seconds')

#Plot mean_hr

ax1.plot(df['Time'],df['%crashes'], blue_color)
ax1.set_xlabel('Playing time [s]')
ax1.set_ylabel('Heartrate', color=blue_color)
ax1.tick_params('y', colors=blue_color)

#Plot %crashes
ax2 = ax1.twinx()
ax2.plot(df['Time'],df['mean_hr'], green_color)
ax2.set_ylabel('Crashes [%]', color=green_color)
ax2.tick_params('y', colors=green_color)

plt.show()

''' Train SVM model
'''

'''Test model with 10-fold Cross-Validation
'''