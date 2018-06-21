# Performance Predictor: Machine learning based prediction of user performance in the game “Plunder Planet”

I built a model that in the game Plunder Planet predicts whether the user will crash into the next appearing obstacle or not.
For this I have datapoints of users that played the game. At approximately each second, we have given the heartrate, the number of points, the difficulty level and, if there is an obstacle, how it looks like.
From this data we generate features that are then used by the machine learning model.

## Requirements

python 3.6, pandas, sklearn, seaborn, scipy, numpy, matplotlib


## Usage

```
$ python main.py -h
usage: main.py [-h] [-t hw_window crash_window gc_window] [-g clf_name] [-l]
               [-k] [-f] [-p] [-s] [-d] [-n] [-v] [-r]

optional arguments:
  -h, --help            show this help message and exit
  
  -t hw_window crash_window gc_window, --test_windows hw_window crash_window gc_window
                        Trains and tests a SVM with the given window sizes.
                        Stores roc_auc score in a file in
                        /Evaluation/Performance/Windows. Note: Provide the
                        windows in seconds
                        
  -g clf_name, --grid_search clf_name
                        Optimizes the given classifier with RAndomSearchCV.
                        Setclf_name='all' if you want to test all classifiers
                        
  -l, --leave_one_out   Plot performance when leaving out a logfile vs leaving
                        out a whole user in crossvalidation
                        
  -k, --print_keynumbers_logfiles
                        Print important numbers and stats about the logfiles
                        
  -f, --generate_plots_about_features
                        Generates different plots from the feature matrix
                        (Look at main.py for details) and stores it in folder
                        /Evaluation/Features
                        
  -p, --generate_plots_about_logfiles
                        Generates different plots from the logfiles (Look at
                        main.py for details) and stores it in folder
                        /Evaluation/Logfiles
                        
  -s, --do_feature_selection
                        Do feature selection with cross_correlation matrix
                        
  -d, --use_test_data   
                        Plot performance when leaving out a logfile vs leaving
                        out a whole user in crossvalidation
                        
  -n, --do_not_normalize_heartrate
                        Do not normalize heartrate (e.g. if you want plots or
                        values with real heartrate)
                        
  -v, --verbose         Prints various information while computing
  
  -r, --reduced_data    Use only a small aprt of the data. Mostly for
                        debugging purposes
```

## More information
The thesis is divided into three parts:

1. Validating data:
  We made sure that the logfiles have the correct structure, no data is missing and there are no outliers.
  For this we generated various plots from the data. 

2. Basic Machine Learning models
  After generating the features, we use standard ML models such as SVM, NearestNeighbors, QDA, DecisionTreeClassifiers etc. 
  We achieved a performance of around roc_auc=0.612

3. LSTM Recurrent Neural Network
  (To be done)


## Publications
