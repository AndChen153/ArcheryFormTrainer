# following this tutorial: https://www.youtube.com/watch?v=tPYj3fFJGjk

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.compat.v2.feature_column as fc
#from IPython.display import clear_output
from six.moves import urllib

import tensorflow as tf

# removes the annoying info prints from tensor flow setting up
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2'} 0 = ingore info prints, 1 = ingore warnings?, 2 = ingore both?


# titanic data with data about each passenger and if they survived or not
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # opens training data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # opens testing data
print(dftrain.head())

y_train = dftrain.pop('survived') # removes the suvived column to seperate the data used to classify (survived) from the data used as input information (everything else)
y_eval = dfeval.pop('survived')

print(dftrain["age"]) # print out specific column
print(dftrain.loc[0], y_train.loc[0]) # used to printout entire rows
print(dftrain.describe()) # statistical analysis of the dataset
print(dftrain.shape) # print out shape column
print(dftrain.sex)

hist1=plt.figure(1)
plt.hist(dftrain.age, bins = 20) # creates histogram of age column

hist2=plt.figure(2)
plt.hist(dftrain.sex, bins = 20)

plt.show()

