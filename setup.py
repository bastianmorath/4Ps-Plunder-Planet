#!/usr/bin/env python3

from setuptools import find_packages, setup

with open('LICENSE') as f:
    LICENSE = f.read()

setup(
   name='4P: Plunder Planet Performance Predictor',
   version='0.1dev',
   author='Bastian Morath',
   author_email='bastian.ethz@gmail.com',
   description='A Machine learning based prediction of user performance in the game “Plunder Planet”',
   license=LICENSE,
   packages=find_packages(),
)