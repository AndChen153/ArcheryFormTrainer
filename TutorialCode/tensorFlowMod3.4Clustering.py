# following this tutorial: https://www.youtube.com/watch?v=tPYj3fFJGjk
# https://youtu.be/tPYj3fFJGjk?t=8229

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