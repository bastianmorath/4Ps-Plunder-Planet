
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
from __future__ import division  # s.t. division uses float result

from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from collections import Counter
import globals_model as gl
import factory_model as factory

crash_window = 30  # Over how many preceeding seconds should %crashes be calculated?
heartrate_window = 50  # Over how many preceeding seconds should the heartrate be averaged?


# TODO: Look that all methods have a return value instead of them modifying arguments
# TODO: Add print logs to log what the program is currently doing
''' Get data and create feature matrix and labels
    Column 0: Id/Time
    Column 1: %Crashes in last x seconds
    Column 2: mean heartrate over last y seconds
'''

df = gl.init(crash_window, heartrate_window)  # Entire dataframe with features-column

df_obstacle = factory.get_obstacle_times_with_success()  # Time of each obstacle and whether user crashed or not

(X, y) = factory.get_feature_matrix_and_label(df_obstacle, df)


''' Apply SVM Model with Cross-Validation
'''

kf = KFold(n_splits=5)
alpha = 1.0
clf = svm.SVC(C=alpha)

mean = []
for train, test in kf.split(X):
    x_training = X.ix[train]
    y_training = y.ix[train]
    x_test = X.ix[test]
    y_test = y.ix[test]

    clf.fit(x_training, y_training)  # Train on 9 bins
    # TODO: Model always predicts 0/False for all x....
    y_predicted = clf.predict(x_test)  # Predict on last bin
    print(Counter(y_predicted).keys())  # equals to list(set(words))
    print(Counter(y_predicted).values())  # counts the elements' frequency
    num_corr = len([a for (a, b) in zip(y_test, y_predicted) if a == b])
    percentage = num_corr / len(y_test) * 100
    mean.append(percentage)

print('With alpha=' + str(alpha) + ', the model got ' + str(sum(mean)/len(mean)) +
      '% right on average.')

'''Test model with 10-fold Cross-Validation
'''