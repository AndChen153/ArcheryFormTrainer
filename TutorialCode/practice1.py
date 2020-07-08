import matplotlib.pyplot as pyplot
import numpy as np
from sklearn import datasets, linear_model


# dataset
runnerDistance = [1,3,4,6,7]
runnerSpeed = [8,9,11,15,20]

# reshape(num of arrays, elements per array)  reforms array into different dimension array
# https://www.w3schools.com/python/numpy_array_reshape.asp
runnerSpeed2 = np.array(runnerSpeed).reshape((-1,1)) # reforms array into multiple arrays with 1 element per array

#assigns linear model to regr variable
regr = linear_model.LinearRegression()
regr.fit(runnerSpeed2, runnerDistance)