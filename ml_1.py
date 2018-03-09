# linear regression model using finance data

import pandas as pd 
import quandl, math
quandl.ApiConfig.api_key = "2_i9TyPLhXHXBuDs_zTM"
import numpy as np 
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open', 'Adj. High','Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

# want to predict the Adj. Close of the future as label
forecast_col = 'Adj. Close'
df.fillna(value=-99999, inplace=True) # replace NAs so they become outliers 

forecast_out = int(math.ceil(0.01*len(df))) # predict 1% in the future
df['label'] = df[forecast_col].shift(-forecast_out) # shift upwards 1% into future

df.dropna(inplace=True)

X = np.array(df.drop(['label'], 1)) # features
y = np.array(df['label']) # labels

# scaling may be faster and more accurate because want range of -1 to 1 but need to do \
# so with all other values in the future as well
X = preprocessing.scale(X)

# shuffles up and outputs training data and testing data 
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = LinearRegression() # linear regression classifier
clf.fit(X_train, y_train) # fit is train
accuracy = clf.score(X_test, y_test) # score is test 

print(accuracy)