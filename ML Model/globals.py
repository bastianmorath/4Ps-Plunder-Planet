"""This file contains variables that are used across different modules"""

import os

import m_svm
import m_nearest_neighbor
import m_naive_bayes


# TODO: Whenever sth. changes here (e.g. window sizes), then automatically don't use cache no matter what settings we have
# TODO: Maybe some unit tests?

cw = 10  # Over how many preceeding seconds should %crashes be calculated?
hw = 10  # Over how many preceeding seconds should the heartrate be averaged?


model = m_nearest_neighbor.NearestNeighbor  # Which model should be used?
# model = m_naive_bayes.NaiveBayes  # Which model should be used?
# model = m_svm.SVM  # Which model should be used?

testing = True  # If Testing==True, only  a small sample of dataframes is used  to accelerate everything
use_cache = True  # If use_cache==True, use cached data (accelerates testing on same data)
test_data = False  # If test_data==True, the model uses synthesized data
normalize_heartrate = True  # Whether we should use normalized heartrate (divide by baseline)
use_boxcox = True   # Use boxcox (transforms features into a normal distribution)

working_directory_path = os.path.abspath(os.path.dirname(__file__))
project_path = os.path.abspath(os.path.join(working_directory_path, '../../..'))

abs_path_logfiles = project_path + '/Logs/text_logs'
names_logfiles = []  # Name of the logfiles

df_list = []  # List with all dataframes; 1 dataframe per logfile

# list of dataframes, each dataframe has time of each obstacle and whether crash or not (1 df per logfile)
# First max(gl.cw, gl.hw) seconds removed
obstacle_df_list = []