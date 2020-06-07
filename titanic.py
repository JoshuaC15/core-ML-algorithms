import os, sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc
import tensorflow as tf

dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')  # dataframe object
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')

y_train = dftrain.pop('survived')  # .pop() removes and returns the 'survived' column, so y_train is the survived column
y_eval = dfeval.pop('survived')

# print(dftrain.head())  # shows us the first 5 entries
# print(dftrain.loc[0])  # .loc[] to locate row 0
# print(dftrain["age"])  # use this syntax to print the age column
# print(dftrain.describe())  # gives us info like count, mean, std dev, min, max etc

