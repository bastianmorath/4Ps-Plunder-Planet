#!/usr/bin/env bash

# This script sends euler-jobs to compile, train and evaluate a LSTM network with different number of epochs

bsub  python "$PWD"/main.py -m 80
bsub  python "$PWD"/main.py -m 90
bsub  python "$PWD"/main.py -m 100
bsub  python "$PWD"/main.py -m 110
bsub  python "$PWD"/main.py -m 130
bsub  python "$PWD"/main.py -m 150
bsub  python "$PWD"/main.py -m 170
bsub  python "$PWD"/main.py -m 190
bsub  python "$PWD"/main.py -m 210
bsub  python "$PWD"/main.py -m 230
bsub  python "$PWD"/main.py -m 250
bsub  python "$PWD"/main.py -m 400

