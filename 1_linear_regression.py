# linear regression model using finance data

import pandas as pd 
import quandl, math, datetime
quandl.ApiConfig.api_key = "2_i9TyPLhXHXBuDs_zTM"
import numpy as np 
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt 
from matplotlib import style
import pickle

style.use('ggplot')

df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open', 'Adj. High','Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

# only Adj. Close is directly related to price so if drop Adj. Close becomes very bad
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

# want to predict the Adj. Close of the future as label
forecast_col = 'Adj. Close'
df.fillna(value=-99999, inplace=True) # replace NAs so they become outliers 

forecast_out = int(math.ceil(0.01*len(df))) # predict 1% in the future
print(forecast_out)
df['label'] = df[forecast_col].shift(-forecast_out) # shift upwards 1% into future

X = np.array(df.drop(['label'], 1)) # features
X = preprocessing.scale(X)
X_lately = X[-forecast_out:] # prediction data
X = X[:-forecast_out] # training and testing data

df.dropna(inplace=True)
y = np.array(df['label']) # labels

# scaling may be faster and more accurate because want range of -1 to 1 so do at same time

# shuffles up and outputs training data and testing data 
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

# easy to switch algorithms, check documentation for n_jobs to thread
clf = LinearRegression(n_jobs=-1) # linear regression
# clf = svm.SVR(kernel='poly') # SVR
clf.fit(X_train, y_train) # fit is for train

# pickling to save
with open('linearregression.pickle', 'wb') as f:
	pickle.dump(clf, f)

pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, y_test) # score is for test 
forecast_set = clf.predict(X_lately)
print(forecast_set)
# print(forecast_set, accuracy, forecast_out)
df['Forecast'] = np.nan # make new forecast column

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400 # number of seconds in a day
next_unix = last_unix + one_day

# reformatting for plotting
for i in forecast_set:
	next_date = datetime.datetime.fromtimestamp(next_unix)
	next_unix += one_day
	df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

# plot
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()