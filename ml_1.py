# linear regression model using finance data

import pandas as pd 
import quandl
import math

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
print(df.head())