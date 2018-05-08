"""This file contains variables that are used across different modules"""

import os



# TODO: Whenever sth. changes here (e.g. window sizes), then automatically don't use cache no matter what settings we have


cw = 30  # Over how many preceeding seconds should %crashes be calculated?
hw = 30  # Over how many preceeding seconds should heartrate features such as min, max, mean be averaged?
gradient_w = 10  # Over how many preceeding seconds should hr features be calculated that have sth. do to with change (likely smaller than hw!)?


testing = True  # If Testing==True, only  a small sample of dataframes is used  to accelerate everything
use_cache = True  # If use_cache==True, use cached data (accelerates testing on same data)
test_data = False  # If test_data==True, the model uses synthesized data

# Whether we should use normalized heartrate (divide by baseline). If test_data, then don't normalize
# since dividing by minimum doesn't make sense
normalize_heartrate = not test_data and True
use_boxcox = False   # Use boxcox (transforms features into a normal distribution)

plots_enabled = False  # Whether plots should be created

working_directory_path = os.path.abspath(os.path.dirname(__file__))
project_path = os.path.abspath(os.path.join(working_directory_path, '../../..'))

abs_path_logfiles = project_path + '/Logs/text_logs_refactored_crashes'  # text_logs_refactored_crashes or text_logs_original
names_logfiles = []  # Name of the logfiles

df_list = []  # List with all dataframes; 1 dataframe per logfile

# list of dataframes, each dataframe has time of each obstacle and whether crash or not (1 df per logfile)
# First max(gl.cw, gl.hw) seconds removed
obstacle_df_list = []