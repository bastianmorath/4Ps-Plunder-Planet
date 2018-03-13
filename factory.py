import globals as gl
import pandas as pd

''' Factory Module
'''


def get_colors():
    colors = ['red'] * len(gl.conc_dataframes)
    for idx, user_id in enumerate(gl.conc_dataframes.groupby(['userID', 'round'])['userID'].max()):
        if user_id % 2 == 0:
            colors[idx] = '#68829E'
        else:
              colors[idx] = '#AEBD38'
    return colors


''' Since the MOVEMENTTUTORIAL at the beginning is different for each player/logfile,
    we need to align them, i.e. remove tutorial-log entries and then set timer to 0
'''
def remove_movement_tutorials():
    for idx, df in enumerate(gl.dataframes):
        if 'MOVEMENTTUTORIAL' in df['Gamemode'].values:

            tutorial_mask = df['Gamemode']=='MOVEMENTTUTORIAL'
            tutorial_entries = df[tutorial_mask]
            tutorial_endtime = tutorial_entries['Time'].max()

            df['Time'] = df['Time'].apply(lambda x: x - tutorial_endtime)
            gl.dataframes[idx] = df[~tutorial_mask].reset_index(drop=True)

''' Since the SHIELDTUTORIAL at the beginning is different for each player/logfile, 
    we need to remove them and adjust timer
'''
@staticmethod
def remove_shield_tutorial():
     for idx, df in enumerate(gl.dataframes):
        if 'SHIELDTUTORIAL' in df['Gamemode'].values: 
            tutorial_mask = df['Gamemode'] == 'SHIELDTUTORIAL'
            tutorial_entries = df[tutorial_mask]
            start_time = tutorial_entries['Time'].min()
            end_time = tutorial_entries['Time'].max()
            tutorial_endtime = end_time - start_time

            df['Time'] = df['Time'][df['Time']>start_time].apply(lambda x: x - tutorial_endtime)
            gl.dataframes[idx] = df[~tutorial_mask].reset_index(drop=True)

'''Subsitutes difficulties with numbers to work with them in a better way, from 1 to 3

'''
def transformToNumbers(df):
    mapping = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'undef': -1}
    df = df.replace({'physDifficulty': mapping, 'psyStress': mapping, 'psyDifficulty': mapping})  

    for col in ['physDifficulty', 'psyStress', 'psyDifficulty']:
        df[col] = df[col].astype('int64')      
    return df


def get_number_of_obstacles_per_difficulty():
    conc_num = transformToNumbers(gl.conc_dataframes) # Transform Difficulties into integers
    # count num.obstacle parts per obstacle
    new = conc_num['obstacle'].apply(lambda x: 0 if x=='none' else x.count(",") + 1 ) 
    conc_num = conc_num.assign(numObstacles=new)

    # number of occurences per diff&numObstacles
    cnt = pd.DataFrame({'count' : conc_num.groupby(['physDifficulty','numObstacles']).size()}).reset_index()    
    numObst = [0]*15
    count=0
    for a in range(0,len(cnt.index)):
        d= cnt['physDifficulty'][a]
        o= cnt['numObstacles'][a]
        if not o ==0: # Filter out when there is no obstacle at all 
            numObst[(d-1)*5+o] = cnt['count'][count]
        count +=1
    return numObst