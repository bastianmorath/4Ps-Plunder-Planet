#!/usr/bin/env bash
# This script sends a euler-job for to tune each classifier's hyperparameter

declare -a arr=('SVM', 'Linear SVM', 'Nearest Neighbor', 'QDA', 'Gradient Boosting', 'Decision Tree', 'Random Forest', 'Ada Boost', 'Naive Bayes')

for clf_idx in "${arr[@]}"
do
    bsub  python "$PWD"/main.py -o '$clf_name'
done
