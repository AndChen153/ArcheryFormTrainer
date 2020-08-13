# following this tutorial: https://www.youtube.com/watch?v=tPYj3fFJGjk

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import feature_column
#from IPython.display import clear_output
import tensorflow as tf

# removes the annoying info prints from tensor flow setting up
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2'} 0 = ingore info prints, 1 = ingore warnings?, 2 = ingore both?


# flower dataset
# setosa, versicolor, virginica


CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']
        #     0           1             2

# data in cm
train_path = tf.keras.utils.get_file(
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)

train_y = train.pop('Species')
test_y = test.pop('Species')

print(train.head())


# input function
def input_fn(features, labels, training=True, batch_size=256):
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle and repeat if you are in training mode.
    if training:
        dataset = dataset.shuffle(1000).repeat()
    
    return dataset.batch(batch_size)


# Feature columns describe how to use the input.
# no need to loop through for categorical columns like in 3.2 since there are just numbers 
my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
print(my_feature_columns)

# Build a DNN with 2 hidden layers with 30 and 10 hidden nodes each.
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # Two hidden layers of 30 and 10 nodes respectively.
    hidden_units=[30, 10],
    # The model must choose between 3 classes.
    n_classes=3)

# similar to epoch but telling data set to train until 5000 training samples have been seen
classifier.train(
    input_fn=lambda: input_fn(train, train_y, training=True),
    steps=5000)
# We include a lambda to avoid creating an inner function previously
# lambda is basically an anonymous function, use this so the classifier can keep calling the data input function
# this is similar to what we did in 3.2 but less convoluted and no need for a function inside of a function in def input_fn()

# testing the model, comparing the test dataset with the results of that same dataset that were popped earlier
eval_result = classifier.evaluate(input_fn=lambda: input_fn(test, test_y,  training=False))

print(eval_result)
print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))




# predicting on one flower value based on user input

def input_fn(features, batch_size=256):
    # Convert the inputs to a Dataset without labels.
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
predict = {}

print("Please type numeric values as prompted.")
for i in features:
  valid = True
  while valid: 
    val = input(i + ": ")
    if not val.isdigit(): valid = False

  predict[feature] = [float(val)]

predictions = classifier.predict(input_fn=lambda: input_fn(predict)) # loads all of the prediction values into a dictionary

for pred_dict in predictions: # cycling through every value in the dicitonary of prediction values
    print(pred_dict)
    class_id = pred_dict['class_ids'][0] # telling us what the prediction is in the pred_dict
    probability = pred_dict['probabilities'][class_id] # printing the probability of the prediciont from the previous line

    print('Prediction is "{}" ({:.1f}%)'.format(
        SPECIES[class_id], 100 * probability))