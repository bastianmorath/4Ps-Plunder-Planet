# 4Ps: A Machine Learning based prediction of user performance in the game “Plunder Planet”

## Description

4Ps (The Plunder Planet Performance Predictor) builds a model that in the game Plunder Planet predicts whether the user will crash into the next appearing obstacle or not.
For this I have datapoints of users that played the game. At approximately each second, we have given the heartrate, the number of points, the difficulty level and -if there is an obstacle- how it looks like.
From this data we generate features that are then used by the machine learning model.

## Requirements

Keras==2.1.5, matplotlib==2.2.2, scipy==1.0.0, graphviz==0.8.3, numpy==1.14.2, seaborn==0.8.1, pandas==0.22.0, scikit_learn==0.19.2


Call 
```
$python -m pip install --user -r requirements.txt 
```
to automatically install all requirements

## Installation

It is recommended to install 4P inside a virtual environment.

Setup a virtual environment:
```
virtualenv --python=python3 <venv-name>
source <venv-name>/bin/activate
```

Install 4Ps:

```
git clone https://github.com/bastianmorath/4Ps
cd 4Ps
pip install -r requirements.txt
python setup.py install 
```


## Usage

```
$ python main.py -h
usage: main.py [-h] [-p clf_name] [-t clf_name]
               [-w hw_window crash_window gc_window] [-g] [-m n_epochs] [-k]
               [-f] [-l] [-a] [-s] [-d] [-v] [-r]


optional arguments:
  -h, --help            show this help message and exit
  
  -p clf_name, --performance_without_tuning clf_name
                        Outputs detailed scores of the given classifier
                        without doing hyperparameter tuning. Set
                        clf_name='all' if you want to test all classifiers
                        
  -t clf_name, --performance_with_tuning clf_name
                        Optimizes the given classifier with RandomizedSearchCV
                        and outputs detailed scores. Set clf_name='all' if you
                        want to test all classifiers
                        
  -w hw_window crash_window gc_window, --test_windows hw_window crash_window gc_window
                        Trains and tests a SVM with the given window sizes.
                        Stores roc_auc score in a file in
                        /Evaluation/Performance/Windows. Note: Provide the
                        windows in seconds
                        
  -g, --leave_one_group_out
                        Plot performance when leaving out a logfile vs leaving
                        out a whole user in crossvalidation
                        
  -m n_epochs, --get_trained_lstm n_epochs
                        Train an LSTM newtwork with n_epochs
                        
  -k, --print_keynumbers_logfiles
                        Print important numbers and stats about the logfiles
                        
  -f, --generate_plots_about_features
                        Generates different plots from the feature matrix
                        (Look at main.py for details) and stores it in folder
                        /Evaluation/Features
                        
  -l, --generate_plots_about_logfiles
                        Generates different plots from the logfiles (Look at
                        main.py for details) and stores it in folder
                        /Evaluation/Logfiles (Note: Probably use in
                        combination with -n, i.e. without normalizing
                        heartrate)
                        
  -a, --all_features    Do not do feature selection with cross_correlation
                        matrix, but use all features instead
                        
  -s, --use_synthesized_data
                        Use synthesized data. Might not work with everything.
                        
  -d, --do_not_normalize_heartrate
                        Do not normalize heartrate (e.g. if you want plots or
                        values with real heartrate)
                        
  -v, --verbose         Prints various information while computing
  
  -r, --reduced_data    Use only a small part of the data. Mostly for
                        debugging purposes


```


## More information
The thesis is divided into three parts:

1. Validating data:
  We made sure that the logfiles have the correct structure, no data is missing and there are no outliers.
  For this we generated various plots from the data. 

2. Basic Machine Learning models
  After generating the features, we use standard supervised ML models, namely SVM, NearestNeighbors, DecisionTreeClassifier, Naive Bayes and Ada Boost
  
  We achieved a performance of around auroc=0.65

3. LSTM Recurrent Neural Network
  We generate a Long-Short-Term-Memory Recurrent Neural Network. 
  
  We achieved a performance of around aurc=0.63


## Publications
