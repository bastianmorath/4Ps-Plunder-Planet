
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
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn import metrics
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
c = 0.1
svc = svm.SVC(kernel='rbf', C=c)

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.3, random_state=0)

# Fit training data
svc.fit(X_train, y_train)

# Print out cross validation mean score for the chosen model
scores = cross_val_score(svc, X_train, y_train, cv=10)
print('cross val mean score = ', scores.mean())
print('cross val std (+/-) = ', scores.std() * 2)

# Predict values on test data
y_test_predicted = svc.predict(X_test)

# Print result as %correctly predicted labels
print('Unique prediction values: ' + str(Counter(y_test_predicted).keys())) # equals to list(set(words))
print('Number of each unique values predicted: ' + str(Counter(y_test_predicted).values()))  # counts the elements' frequency

percentage = metrics.accuracy_score(y_test, y_test_predicted)
print('Percentage of correctly classified data: ' + str(percentage))


