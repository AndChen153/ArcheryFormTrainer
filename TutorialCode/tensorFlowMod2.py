# following this tutorial: https://www.youtube.com/watch?v=tPYj3fFJGjk

# removes the annoying info prints from tensor flow setting up
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2'} 0 = ingore info prints, 1 = ingore warnings?, 2 = ingore both?

import tensorflow as tf
import numpy as np

print(tf.version)
print("\n")

string = tf.Variable("this is a string", tf.string) 
number = tf.Variable(324, tf.int16)
floating = tf.Variable(3.567, tf.float64)

rank1_tensor = tf.Variable([1,1,2], tf.int32)
rank2_tensor = tf.Variable([[1,2],[1,3]], tf.int32)

tensor1 = tf.ones([1,2,3]) # one interior list, two lists inside, three elements in each list
tensor2 = tf.reshape(tensor1, [2,3,1]) # reshape existing data to 2 outer lists, 3 inner lists, 1 element per list
tensor3 = tf.reshape(tensor2, [3, -1]) # -1 tells tensor to calculate the shape depending on the amount of elements in the list
                                       # reshape to 3 inner lists and equal items (2) in each list

#print(tf.rank(rank1_tensor))
print(tensor2)
