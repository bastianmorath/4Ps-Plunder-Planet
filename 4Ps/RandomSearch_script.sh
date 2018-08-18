#!/usr/bin/env bash

# This script tries out different window sizes and sends a job to the Euler-cluster which
# does RandomizedSearchCV for all classifiers

bsub -W 24:00 -R "rusage[mem=16048]" "python main.py -t 'SVM'"
bsub -W 24:00 -R "rusage[mem=16048]" "python main.py -t 'Nearest Neighbor'"
bsub -W 24:00 -R "rusage[mem=16048]" "python main.py -t 'Decision Tree'"
bsub -W 24:00 -R "rusage[mem=16048]" "python main.py -t 'Random Forest'"


