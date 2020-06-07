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
#
# dftrain.age.hist(bins=20)  # matplotlib to visualize the data
# plt.show()
# dftrain.sex.value_counts().plot(kind='barh')
# plt.show()
# dftrain['class'].value_counts().plot(kind='barh')
# plt.show()
# pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive')
# plt.show()

# categorical data is data without numerical value (Ex: yes, no, male, female etc)
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []  # to be fed into our linear model to make predictions
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique()  # gets a list of all unique values from given feature column
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))
    # encode the categorical column with the line above

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

print(feature_columns)


def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # take data passed in and turn into correct obj
        if shuffle:
            ds = ds.shuffle(1000)  # shuffle randomizes the order of entries passed in
        ds = ds.batch(batch_size).repeat(num_epochs)  # split our ds into blocks to be passed in (based on batch size and # of epochs
        return ds  # return a batch of the dataset

    return input_function  # return a function object for use


train_input_fn = make_input_fn(dftrain, y_train)  # this is equal to a function, (it is a function object itself)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)  # eval data, only need 1 epoch cuz we r evaluating not training

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)  # create estimator
linear_est.train(train_input_fn)  # train
result = linear_est.evaluate(eval_input_fn)  # store result in a variable to look at it

print(result)  # or print(result['accuracy']) to just see the accuracy since it is a dict object

# to see the predictions it made
pred_dicts = list(linear_est.predict(eval_input_fn))
print(dfeval.loc[0])  # show the person we are looking at (entry 0)
print(y_eval.loc[0])  # show whether or not the person actually survived (entry 0)
print(pred_dicts[0]['probabilities'])  # shows us what our model thought about probability of survival for entry 0

probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])
probs.plot(kind='hist', bins=20, title='predicted probabilities')
plt.show()
