import globals as gl


''' Factory Module
'''


class Helpers:
    @staticmethod
    def get_colors():
        colors = ['red'] * len(gl.conc_dataframes)
        for idx, user_id in enumerate(gl.conc_dataframes.groupby(['userID', 'round'])['userID'].max()):
            if user_id % 2 == 0:
                colors[idx] = '#68829E'
            else:
                colors[idx] = '#AEBD38'
        return colors


class Filter:
    ''' Since the MOVEMENTTUTORIAL at the beginning is different for each player/logfile,
        we need to align them, i.e. remove tutorial-log entries and then set timer to 0
    '''
    @staticmethod
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
