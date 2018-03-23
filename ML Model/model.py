
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

import globals_model as gl
import factory_model as factory

crash_window = 30  # Over how many preceeding seconds should %crashes be calculated?
heartrate_window = 50  # Over how many preceeding seconds should the heartrate be averaged?


# TODO: Look that all methods have a return value instead of them modifying arguments

''' Get data and create feature matrix and labels
    Column 0: Id/Time
    Column 1: %Crashes in last x seconds
    Column 2: mean heartrate over last y seconds
'''

df = gl.init(crash_window, heartrate_window)
factory.plot(df)
df_obstacle = factory.get_obstacle_times_with_success()  # Time of each obstacle and whether user crashed or not
X = factory.get_feature_matrix_for_obstacles_and_df(df_obstacle, df)
y = df_obstacle['crash'].copy()

''' Apply SVM Model with Cross-Validation
'''
print(X)
print(y)
kf = KFold(n_splits=5)

# Find best gamma and alpha
C_range = [0.1, 0.2]  # np.logspace(-8, 1, num=5, base=2)
gamma_range = [0.1, 0.2]  # np.logspace(-4, 4, num=5, base=2)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv)
grid.fit(X, y)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

clf = svm.SVC(C=grid.best_params_['C'])
mean = []
for train, test in kf.split(X):
    x_training = X.ix[train]
    y_training = y.ix[train]
    x_test = X.ix[test]
    y_test = y.ix[test]

    clf.fit(x_training, y_training)  # Train on 9 bins

    y_predicted = clf.predict(x_test)  # Predict on last bin
    num_corr = len([a for (a, b) in zip(y_test, y_predicted) if a == b])
    percentage = num_corr / len(y_test) * 100
    mean.append(percentage)
print('With alpha=' + str(grid.best_params_['C']) + ', the model got ' + str(sum(mean)/len(mean)) +
      '% right on average.')

'''Test model with 10-fold Cross-Validation
'''