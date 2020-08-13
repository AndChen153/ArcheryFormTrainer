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


# titanic data with data about each passenger and if they survived or not
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # opens training data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # opens testing data
#print(dftrain.head())

y_train = dftrain.pop('survived') # removes the suvived column to seperate the data used to classify (survived) from the data used as input information (everything else)
y_eval = dfeval.pop('survived')

CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
  vocabulary = dftrain[feature_name].unique()  # gets a list of all unique values from given feature column
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

def make_input_fn(data_df, label_df, num_epochs=20, shuffle=True, batch_size=32):
  def input_function():  # inner function, this will be returned
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
    if shuffle:
      ds = ds.shuffle(1000)  # randomize order of data
    ds = ds.batch(batch_size).repeat(num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
    return ds  # return a batch of the dataset
  return input_function  # return a function object for use

train_input_fn = make_input_fn(dftrain, y_train)  # here we will call the input_function that was returned to us to get a dataset object we can feed to the model
                                                  # train_input_fun has the input_function stored inside it and if you call train_input_fn() it will run input_function, weird syntax but it just works this way
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

linear_est.train(train_input_fn)  # train
result = linear_est.evaluate(eval_input_fn)  # get model metrics/stats by testing on tetsing data

print(result['accuracy'])  # the result variable is simply a dict of stats about our model
                           # result found by comparing the results from the model with the actual survial/death stats provided in the data
print(result)

predictResult = list(linear_est.predict(eval_input_fn)) # send to list because the input function will run many times (once for each person's data)
for i in range(0,10):
  print(dfeval.loc[i]) # prints out entire row of data
  print(y_eval.loc[i]) # prints out if the person survived or not
  print(predictResult[i]['probabilities'][1]) # prints out chance of survival, ['probabilities'][0] prints chance of death