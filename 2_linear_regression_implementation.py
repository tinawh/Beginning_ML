# code linear regression

from statistics import mean 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

# xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
# ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)

def create_dataset(hm, variance, step=2, correlation=False):
	"""Return sample dataset
	@param hm: how much data
	@param variance: amount of variability
	@param step: how far on average to step up y value 
	@param correlation: what kind of correlation
	return: x-values, y-values
	"""
	val = 1 
	ys = []
	for i in range(hm):
		y = val + random.randrange(-variance, variance)
		ys.append(y)
		# add directionality
		if correlation and correlation == 'pos':
			val+=step
		elif correlation and correlation == 'neg':
			val-=step
	xs = [i for i in range(len(ys))]

	return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

def best_fit_slope_and_intercept(xs, ys):
	m = ( ((mean(xs) * mean(ys)) - mean(xs*ys)) /
		(mean(xs)**2 - mean(xs**2)) )
	b = mean(ys) - m*mean(xs)
	return m, b

def squared_error(ys_orig, ys_line):
	return sum((ys_line - ys_orig)**2)

def coefficient_of_determination(ys_orig, ys_line):
	y_mean_line = [mean(ys_orig) for y in ys_orig]
	squared_error_regr = squared_error(ys_orig, ys_line)
	squared_error_y_mean = squared_error(ys_orig, y_mean_line)
	return 1 - (squared_error_regr / squared_error_y_mean)

xs,ys = create_dataset(40, 40, 2, correlation='pos') # random dataset
xs, ys = create_dataset(40, 10, 2, correlation='pos') # decrease variance will increase r_squared

m,b = best_fit_slope_and_intercept(xs, ys)

regression_line = [(m*x) + b for x in xs]

predict_x = 8
predict_y = (m*predict_x) + b

r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared)

plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y, s=100, color = "r") # prediction, increase size to 100
plt.plot(xs, regression_line)
plt.show()
