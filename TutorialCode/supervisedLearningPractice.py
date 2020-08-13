# https://www.youtube.com/watch?v=Q59X518JZHE

import matplotlib.pyplot as pyplot
import numpy as np
from sklearn import datasets, linear_model


# dataset, predicting runner distance
runnerSpeed = [8,9,11,15,20] # y axis
runnerDistance = [1,3,4,6,7] # x axis

# reshape(num of arrays, elements per array)  reforms array into different dimension array
# https://www.w3schools.com/python/numpy_array_reshape.asp
runnerDistance2 = np.array(runnerDistance).reshape((-1,1)) # reforms array into multiple arrays with 1 element per array

#assigns linear model to regr variable
regr = linear_model.LinearRegression()
regr.fit(runnerDistance2, runnerSpeed)

print("Coeff: ", regr.coef_)
print("Intercept: ", regr.intercept_)

distNew = 5
speedNew = regr.predict([[distNew]])
print(speedNew)

# creating graph of data
def graph(formula, x_range):
    x = np.array(x_range)
    y = eval(formula)
    pyplot.plot(x,y)

# plotting graph
graph('regr.coef_*x + regr.intercept_', range(0,10)) # formula is the formula for regression prediction, range is x axis
pyplot.scatter(runnerDistance, runnerSpeed, color = "black")
pyplot.xlabel('Runner Distance')
pyplot.ylabel('Runner Speed')
pyplot.show()