# k-nearest neighbours example

# k-nearest neighbours handles outliers really badly
# always want an odd number as k to not get a tie 
# confidence (variance in voting part) and accuracy (right/wrong) can be very different 
# works on linear and non linear data 
# can have radius and threading so quite fast and accurate 

import numpy as np
from sklearn import preprocessing, neighbors, model_selection
import pandas as pd

accuracies = []
for i in range(25): # repeat multiple times to average accuracy
	df = pd.read_csv('breast-cancer-wisconsin.data.txt')
	df.replace('?', -99999, inplace=True) # replace missing data as outliers
	# remove useless data
	df.drop(['id'], 1, inplace=True) # if don't drop id column accuracy decreases significantly

	X = np.array(df.drop(['class'], 1))
	y = np.array(df['class'])

	X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

	# train classifier
	clf = neighbors.KNeighborsClassifier(n_jobs=-1) # threading 
	clf.fit(X_train, y_train)

	# score accuracy on test dataset
	accuracy = clf.score(X_test, y_test)
	# print(accuracy)

	# example_measures = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1], [4, 2, 1, 2, 2, 2, 3, 2, 1]]) # two examples
	# example_measures = example_measures.reshape(len(example_measures), -1) # reshape based on sample size

	# prediction = clf.predict(example_measures)
	# print(prediction)
	accuracies.append(accuracy)

print(sum(accuracies)/len(accuracies))