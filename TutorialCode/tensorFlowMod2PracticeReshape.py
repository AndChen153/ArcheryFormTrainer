# removes the annoying info prints from tensor flow setting up
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2'} 0 = ingore info prints, 1 = ingore warnings?, 2 = ingore both?

import tensorflow as tf

print(tf.version)
print("\n")

t = tf.zeros([5,5,5,5]) # creates tensor with 5 outer lists, 5 lists in each of those lists, 5 more lists in each of those lists, and 5 elements in each list
print (t)

t = tf.reshape(t, [625]) # just one list of 0's
print (t)

t = tf.reshape(t, [125, -1]) # reshapes to lists of 125 elements, number of lists is pretedermined but the first number has to be a factor of the total number of elements
                             # if you do 100 instead of 125 it will error since 100 is not a factor of 625
print (t)

