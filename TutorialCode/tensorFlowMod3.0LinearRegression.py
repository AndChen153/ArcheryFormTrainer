# following this tutorial: https://www.youtube.com/watch?v=tPYj3fFJGjk

# https://youtu.be/tPYj3fFJGjk?t=4115
# removes the annoying info prints from tensor flow setting up
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.compat.v2.feature_column as fc
#from IPython.display import clear_output
from six.moves import urllib

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2'} 0 = ingore info prints, 1 = ingore warnings?, 2 = ingore both?


x = [1, 2, 2.5, 3, 4]
y = [1, 4, 7, 9, 15]

# only one input variable since there are only two dimensions
plt.plot(x,y, "ro")
plt.axis([0,6,0,20]) # sets window values [x min, x max, y min, y max]
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x))) # show the line of best fit
plt.show()
