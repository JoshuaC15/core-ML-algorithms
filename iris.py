import tensorflow as tf

import pandas as pd

# define constants based on information in dataset
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

# import both the training and eval datasets with keras.utils.get_file which will download it as iris_training.csv
train_path = tf.keras.utils.get_file(
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)

print(train.head())  # take a look at our data note that species is already encoded for us so we don't need to encode it
train_y = train.pop('Species')
test_y = test.pop('Species')
train.shape()  # note we have 120 entries with 4 features each, as we have popped off the species column

